#!/bin/bash
#SBATCH --job-name=joint_B
#SBATCH --partition=gpu-day
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=analysis/pca_finetune/joint_B_%j.out
#SBATCH --error=analysis/pca_finetune/joint_B_%j.err

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx
source .venv/bin/activate
exec python -u analysis/pca_finetune/joint_refine.py --half B
