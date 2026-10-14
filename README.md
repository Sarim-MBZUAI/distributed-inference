# distributed-inference

Pipeline-parallel inference of Qwen3.5-4B split across two machines.

- rank 0: embeddings + layers 0-15, sends hidden states to rank 1
- rank 1: layers 16-31 + norm + lm_head, samples the token and sends it back

```bash
sbatch run_2node.sbatch --prompt "your prompt" --max-new-tokens 200
```

Outside SLURM, on each machine:

```bash
MASTER_ADDR=<machine A ip> MASTER_PORT=29531 WORLD_SIZE=2 RANK=0 python pp_infer.py   # machine A
MASTER_ADDR=<machine A ip> MASTER_PORT=29531 WORLD_SIZE=2 RANK=1 python pp_infer.py   # machine B
```

`reference.py` runs the full model on one GPU for comparison.
