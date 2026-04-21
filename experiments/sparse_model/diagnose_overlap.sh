#!/bin/bash
#SBATCH --job-name=torchsx-diag
#SBATCH --partition=gpu-day
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=sparse_model/diagnose_overlap_%j.log

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx
.venv/bin/python -u sparse_model/diagnose_overlap.py
