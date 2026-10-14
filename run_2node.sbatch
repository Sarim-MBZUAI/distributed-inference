#!/bin/bash
#SBATCH --job-name=pp-qwen
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=0:30:00
#SBATCH --output=logs/pp-%j.log

# one task per node: rank 0 on the first node, rank 1 on the second
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=${MASTER_PORT:-29531}
export WORLD_SIZE=2
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ens5f0np0}
PY=${PY:-$HOME/anaconda3/envs/wiki-vllm/bin/python}

cd "$SLURM_SUBMIT_DIR"
srun --kill-on-bad-exit=1 bash -c 'RANK=$SLURM_PROCID LOCAL_RANK=0 '"$PY"' pp_infer.py --backend nccl "$@"' _ "$@"
