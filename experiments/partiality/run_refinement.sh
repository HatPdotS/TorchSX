#!/bin/bash
#SBATCH --job-name=rank_refine
#SBATCH --output=partiality/results/refinement.out
#SBATCH --error=partiality/results/refinement.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=day

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx
mkdir -p partiality/results
.venv/bin/python partiality/run_refinement.py

# Also run partiality estimation after refinement
.venv/bin/python partiality/estimate_partiality.py
