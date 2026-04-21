#!/bin/bash
#SBATCH --job-name=outliers
#SBATCH --partition=gpu-day
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=diagnose_outliers_%j.log

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx
.venv/bin/python -u diagnose_outliers.py
