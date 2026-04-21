#!/bin/bash
#SBATCH --job-name=scan_lam
#SBATCH --output=partiality/results/scan_restraint.out
#SBATCH --error=partiality/results/scan_restraint.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=day

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx
.venv/bin/python partiality/scan_restraint.py
