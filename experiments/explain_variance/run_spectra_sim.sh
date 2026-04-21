#!/bin/bash
#SBATCH --job-name=spectra_sim
#SBATCH --output=explain_variance/spectra_sim.out
#SBATCH --error=explain_variance/spectra_sim.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=day

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx
.venv/bin/python explain_variance/partiality_from_spectra.py
