#!/bin/bash
#SBATCH -p gpu-day
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=100G
#SBATCH -J debug_match
#SBATCH -o sparse_model/debug_matching_%j.out
#SBATCH -e sparse_model/debug_matching_%j.err
#SBATCH -t 00:30:00

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx
.venv/bin/python -u sparse_model/debug_matching.py
