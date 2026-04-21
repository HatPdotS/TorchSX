#!/bin/bash
#SBATCH --job-name=build-frames
#SBATCH --partition=gpu-day
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx
.venv/bin/python -u build_frame_store.py
