#!/bin/bash
#SBATCH --job-name=bench-gpipe
#SBATCH --partition=gpu-day
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=bench_gpu_pipeline_%j.out

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx
.venv/bin/python -u bench_gpu_pipeline.py
