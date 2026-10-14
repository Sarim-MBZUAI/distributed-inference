#!/bin/bash
#SBATCH --job-name=pp-ref
#SBATCH --partition=gpu
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=0:20:00
#SBATCH --output=logs/ref-%j.log
cd "$SLURM_SUBMIT_DIR"
${PY:-$HOME/anaconda3/envs/wiki-vllm/bin/python} reference.py "$@"
