#!/bin/bash
#SBATCH --job-name=bench-frames
#SBATCH --partition=gpu-day
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=bench_frame_store_%j.out

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx
.venv/bin/python -u bench_frame_store.py
