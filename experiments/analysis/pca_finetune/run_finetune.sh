#!/bin/bash
#SBATCH -p gpu-day
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=100G
#SBATCH -J pca_finetune
#SBATCH -o analysis/pca_finetune/finetune_%j.out
#SBATCH -e analysis/pca_finetune/finetune_%j.err
#SBATCH -t 4:00:00

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx
.venv/bin/python -u analysis/pca_finetune/finetune_pca.py
