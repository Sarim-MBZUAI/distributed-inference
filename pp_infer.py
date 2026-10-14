"""Pipeline-parallel inference for Qwen3.5-4B across two machines.

Rank 0: embeddings + layers [0, split)          -> sends hidden states
Rank 1: layers [split, 32) + final norm + lm_head -> sends back next token

Each rank builds a smaller Qwen3_5TextModel holding only its own layers,
re-indexed from 0, so the HF cache / mask code works unchanged per stage.
"""

import argparse
import glob
import json
import os
import time

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig, AutoTokenizer, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

PREFIX = "model.language_model."


def resolve_path(model):
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])


def build_stage(path, rank, split, device, dtype):
    cfg = AutoConfig.from_pretrained(path).text_config
    n = cfg.num_hidden_layers
    lo, hi = (0, split) if rank == 0 else (split, n)

    cfg.layer_types = cfg.layer_types[lo:hi]
    cfg.num_hidden_layers = hi - lo
    cfg._attn_implementation = "sdpa"

    with torch.device(device):
        model = Qwen3_5TextModel(cfg).to(dtype)

    # pick the weights this stage needs and shift layer indices to start at 0
    index = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    state = {}
    for f in sorted(set(index.values())):
        with safe_open(os.path.join(path, f), framework="pt", device=str(device)) as sf:
            for k in sf.keys():
                if not k.startswith(PREFIX):
                    continue
                name = k[len(PREFIX):]
                if name.startswith("layers."):
                    i = int(name.split(".")[1])
                    if not lo <= i < hi:
                        continue
                    name = name.replace(f"layers.{i}.", f"layers.{i - lo}.", 1)
                elif name.startswith("embed_tokens"):
                    pass  # rank 0 embeds, rank 1 uses it as tied lm_head
                elif name.startswith("norm") and rank == 0:
                    continue
                state[name] = sf.get_tensor(k).to(dtype)

    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [m for m in missing if not (rank == 0 and m.startswith("norm"))]
    assert not missing and not unexpected, (missing, unexpected)

    if rank == 0:
        model.norm = nn.Identity()  # final norm belongs to the last stage
        lm_head = None
    else:
        lm_head = model.embed_tokens.weight  # tie_word_embeddings=True
    model.eval()
    return model, cfg, lm_head


def sample(logits, temperature, top_p):
    if temperature <= 0:
        return logits.argmax(-1)
    probs = torch.softmax(logits.float() / temperature, -1)
    sorted_p, idx = probs.sort(-1, descending=True)
    keep = sorted_p.cumsum(-1) - sorted_p < top_p
    sorted_p = sorted_p * keep
    choice = torch.multinomial(sorted_p / sorted_p.sum(-1, keepdim=True), 1)
    return idx.gather(-1, choice).squeeze(-1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--prompt", default="Explain pipeline parallelism in two sentences.")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--split", type=int, default=16, help="first layer that lives on rank 1")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--backend", default="nccl", choices=["nccl", "gloo"])
    args = ap.parse_args()

    # works under srun (SLURM_*) or torchrun / manual env (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT)
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 2)))
    assert world == 2, "this script is written for exactly two stages"
    dist.init_process_group(args.backend, rank=rank, world_size=world)

    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    comm = device if args.backend == "nccl" else torch.device("cpu")
    dtype = torch.bfloat16

    path = resolve_path(args.model)
    tok = AutoTokenizer.from_pretrained(path)
    t0 = time.time()
    model, cfg, lm_head = build_stage(path, rank, args.split, device, dtype)
    print(f"[rank {rank} @ {os.uname().nodename}] {cfg.num_hidden_layers} layers loaded in "
          f"{time.time() - t0:.1f}s, {torch.cuda.memory_allocated(device) / 2**30:.2f} GiB", flush=True)

    gen_cfg = json.load(open(os.path.join(path, "config.json"))).get("text_config", {})
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), gen_cfg.get("eos_token_id")} - {None}
    cache = DynamicCache(config=cfg)
    hidden = cfg.hidden_size

    if rank == 0:
        msgs = [{"role": "user", "content": args.prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=args.thinking)
        ids = tok(text, return_tensors="pt").input_ids.to(device)
        dist.broadcast(torch.tensor([ids.shape[1]], device=comm), src=0)
    else:
        n = torch.zeros(1, dtype=torch.long, device=comm)
        dist.broadcast(n, src=0)

    out_ids, t_first, t_start = [], None, time.time()
    seq_len = ids.shape[1] if rank == 0 else int(n.item())
    for step in range(args.max_new_tokens):
        T = seq_len if step == 0 else 1
        if rank == 0:
            h = model(input_ids=ids, past_key_values=cache, use_cache=True).last_hidden_state
            dist.send(h.to(comm).contiguous(), dst=1)
            nxt = torch.zeros(1, dtype=torch.long, device=comm)
            dist.recv(nxt, src=1)
            ids = nxt.to(device).view(1, 1)
        else:
            h = torch.empty(1, T, hidden, dtype=dtype, device=comm)
            dist.recv(h, src=0)
            out = model(inputs_embeds=h.to(device), past_key_values=cache, use_cache=True).last_hidden_state
            logits = out[:, -1] @ lm_head.T
            nxt = sample(logits, args.temperature, args.top_p)
            dist.send(nxt.to(comm), dst=0)

        t = int(nxt.item())
        if step == 0:
            t_first = time.time() - t_start
        if t in eos:
            break
        out_ids.append(t)
        if rank == 0:
            print(tok.decode([t]), end="", flush=True)

    if rank == 0:
        dt = time.time() - t_start
        decode = (len(out_ids) - 1) / max(dt - t_first, 1e-9)
        print(f"\n\n[{len(out_ids)} tokens | first token {t_first * 1000:.0f} ms | "
              f"decode {decode:.1f} tok/s]", flush=True)
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/last_output.txt", "w") as f:
            f.write(tok.decode(out_ids))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
