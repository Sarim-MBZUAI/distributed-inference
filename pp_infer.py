"""Qwen3.5-4B pipeline-parallel over two machines.

rank 0: embed + layers [0, split)          -> sends hidden states
rank 1: layers [split, 32) + norm + lm_head -> sends back next token

Each rank builds a Qwen3_5TextModel with only its layers, re-indexed from 0,
so the HF cache and mask code work unchanged per stage.
"""

import argparse
import json
import os
import socket

import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig, AutoTokenizer, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

PREFIX = "model.language_model."


def build_stage(path, rank, split, device):
    cfg = AutoConfig.from_pretrained(path).text_config
    assert cfg.tie_word_embeddings and 0 < split < cfg.num_hidden_layers
    lo, hi = (0, split) if rank == 0 else (split, cfg.num_hidden_layers)
    cfg.layer_types, cfg.num_hidden_layers = cfg.layer_types[lo:hi], hi - lo
    cfg._attn_implementation = "sdpa"
    with torch.device(device):
        model = Qwen3_5TextModel(cfg).to(torch.bfloat16)

    files = set(json.load(open(f"{path}/model.safetensors.index.json"))["weight_map"].values())
    state = {}
    for f in files:
        with safe_open(f"{path}/{f}", framework="pt") as sf:
            for k in sf.keys():
                if not k.startswith(PREFIX):
                    continue
                name = k.removeprefix(PREFIX)
                if name.startswith("layers."):
                    i = int(name.split(".")[1])
                    if not lo <= i < hi:
                        continue
                    name = name.replace(f"layers.{i}.", f"layers.{i - lo}.", 1)
                state[name] = sf.get_tensor(k)
    model.load_state_dict(state)  # strict: any missing or extra weight raises

    if rank == 0:
        model.norm = nn.Identity()  # final norm belongs to the last stage
    return model.eval(), cfg


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, help="cuda | mps | cpu")
    ap.add_argument("--prompt", default="Explain pipeline parallelism in two sentences.")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--split", type=int, default=16, help="first layer on rank 1")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--thinking", action="store_true")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    dist.init_process_group("gloo")  # reads MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE
    assert dist.get_world_size() == 2

    path = snapshot_download(args.model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
    tok = AutoTokenizer.from_pretrained(path)
    model, cfg = build_stage(path, rank, args.split, args.device)
    print(f"[rank {rank} @ {socket.gethostname()}] {cfg.num_hidden_layers} layers on {args.device}", flush=True)

    cache = DynamicCache(config=cfg)
    n = torch.zeros(1, dtype=torch.long)
    if rank == 0:
        text = tok.apply_chat_template([{"role": "user", "content": args.prompt}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=args.thinking)
        ids = tok(text, return_tensors="pt").input_ids
        n[0] = ids.shape[1]
    dist.broadcast(n, src=0)

    nxt = torch.zeros(1, dtype=torch.long)
    for step in range(args.max_new_tokens):
        if rank == 0:
            h = model(input_ids=ids.to(args.device), past_key_values=cache).last_hidden_state
            dist.send(h.cpu(), dst=1)
            dist.recv(nxt, src=1)
            ids = nxt.view(1, 1)
        else:
            h = torch.empty(1, int(n) if step == 0 else 1, cfg.hidden_size, dtype=torch.bfloat16)
            dist.recv(h, src=0)
            out = model(inputs_embeds=h.to(args.device), past_key_values=cache).last_hidden_state
            logits = (out[0, -1] @ model.embed_tokens.weight.T).float()
            if args.temperature > 0:
                nxt = torch.multinomial(torch.softmax(logits / args.temperature, -1), 1).cpu()
            else:
                nxt = logits.argmax(-1, keepdim=True).cpu()
            dist.send(nxt, dst=0)
        if nxt.item() == tok.eos_token_id:
            break
        if rank == 0:
            print(tok.decode(nxt), end="", flush=True)
    print()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
