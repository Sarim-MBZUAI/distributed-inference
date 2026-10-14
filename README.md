# distributed-inference

Qwen3.5-4B split across two machines: layers 0-15 on machine A, layers 16-31 on machine B.

```bash
pip install torch "transformers>=5.8" safetensors huggingface_hub

# machine A
MASTER_ADDR=<A's IP> MASTER_PORT=29500 WORLD_SIZE=2 RANK=0 python pp_infer.py --device cuda --prompt "hi"
# machine B
MASTER_ADDR=<A's IP> MASTER_PORT=29500 WORLD_SIZE=2 RANK=1 python pp_infer.py --device cuda --prompt "hi"
```

B must reach A's IP on the port. `--device`: cuda, mps or cpu.
