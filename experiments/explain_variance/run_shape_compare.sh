#!/bin/bash
#SBATCH --job-name=shape_cmp
#SBATCH --output=explain_variance/shape_compare.out
#SBATCH --error=explain_variance/shape_compare.err
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=day

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx
.venv/bin/python explain_variance/spectral_shape_comparison.py
