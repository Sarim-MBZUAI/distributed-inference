# distributed-inference

Pipeline-parallel inference of Qwen3.5-4B split across two machines.

- rank 0: embeddings + layers 0-15, sends hidden states to rank 1
- rank 1: layers 16-31 + norm + lm_head, samples the token and sends it back

```bash
sbatch run_2node.sbatch --prompt "your prompt" --max-new-tokens 200
```

Two separate machines (laptops, servers, anything that can reach each other over the network):

```bash
pip install torch "transformers>=5.8" safetensors huggingface_hub

# machine A (rank 0) - its IP must be reachable from B, port 29500 open
MASTER_ADDR=<A's ip> MASTER_PORT=29500 WORLD_SIZE=2 RANK=0 python pp_infer.py --prompt "hi"
# machine B (rank 1)
MASTER_ADDR=<A's ip> MASTER_PORT=29500 WORLD_SIZE=2 RANK=1 python pp_infer.py --prompt "hi"
```

Each machine picks cuda / mps / cpu automatically (`--device` to override). Default backend is
gloo, which works over plain TCP; use `--backend nccl` only when both sides have NVIDIA GPUs.
If gloo binds the wrong network interface, set `GLOO_SOCKET_IFNAME` (e.g. `en0`, `wlan0`).

`reference.py` runs the full model on one GPU for comparison.
