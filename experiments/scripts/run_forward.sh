#!/bin/bash
#SBATCH --job-name=fwd_trial
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=fwd_trial_%j.log

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx
source .venv/bin/activate

echo "=== $(date) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

python trial_forward.py
