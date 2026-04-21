#!/bin/bash
#SBATCH --job-name=fit_part
#SBATCH --output=partiality/results/fit_lorentz.out
#SBATCH --error=partiality/results/fit_lorentz.err
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=day

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx
.venv/bin/python partiality/fit_lorentzian_partiality.py
