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

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1) MASTER_PORT=29531 WORLD_SIZE=2
srun bash -c 'RANK=$SLURM_PROCID '"${PY:-$HOME/anaconda3/envs/wiki-vllm/bin/python}"' pp_infer.py --device cuda "$@"' _ "$@"
