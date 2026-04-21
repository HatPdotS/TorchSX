#!/bin/bash
#SBATCH --job-name=bench-ram
#SBATCH --partition=gpu-day
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=bench_from_ram_%j.out

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx
.venv/bin/python -u bench_from_ram.py
