#!/bin/bash
#SBATCH --job-name=part_mle
#SBATCH --output=partiality/results/mle.out
#SBATCH --error=partiality/results/mle.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=day

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx
.venv/bin/python partiality/fit_partiality_mle.py
