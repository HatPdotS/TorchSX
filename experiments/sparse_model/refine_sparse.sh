#!/bin/bash
#SBATCH --job-name=torchsx-sparse
#SBATCH --partition=gpu-day
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=sparse_model/refine_sparse_%j.log

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx
.venv/bin/python -u sparse_model/refine_sparse.py
