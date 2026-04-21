#!/bin/bash
#SBATCH --job-name=sparse-comp
#SBATCH --partition=gpu-day
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=sparse_compress_%j.log

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx
.venv/bin/python -u sparse_compress.py
