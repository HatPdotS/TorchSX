#!/bin/bash
#SBATCH --job-name=excise-rois
#SBATCH --partition=gpu-day
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=excise_rois_%j.log

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx
.venv/bin/python -u excise_rois.py
