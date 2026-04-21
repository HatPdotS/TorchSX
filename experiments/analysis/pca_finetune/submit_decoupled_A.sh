#!/bin/bash
#SBATCH --job-name=decoup_A
#SBATCH --partition=gpu-day
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=analysis/pca_finetune/decoupled_A_%j.out
#SBATCH --error=analysis/pca_finetune/decoupled_A_%j.err

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx
source .venv/bin/activate
exec python -u analysis/pca_finetune/decoupled_refine.py --half A
