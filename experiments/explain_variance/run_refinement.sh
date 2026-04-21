#!/bin/bash
#SBATCH --job-name=refine_orient
#SBATCH --output=explain_variance/refine_orientation.out
#SBATCH --error=explain_variance/refine_orientation.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=day

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx
.venv/bin/python explain_variance/refine_orientation_partiality.py
