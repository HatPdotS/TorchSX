#!/bin/bash
#SBATCH --job-name=torchsx-refine
#SBATCH --partition=gpu-day
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=refine_e2e_%j.log

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx
.venv/bin/python -u refine_e2e.py
