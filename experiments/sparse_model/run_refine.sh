#!/bin/bash
#SBATCH -p gpu-day
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=100G
#SBATCH -J refine_sparse
#SBATCH -o sparse_model/refine_sparse_%j.out
#SBATCH -e sparse_model/refine_sparse_%j.err
#SBATCH -t 12:00:00

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx
.venv/bin/python -u sparse_model/refine_sparse.py
