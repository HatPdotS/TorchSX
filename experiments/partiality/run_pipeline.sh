#!/bin/bash
#SBATCH --job-name=part_pipe
#SBATCH --output=partiality/results/pipeline.out
#SBATCH --error=partiality/results/pipeline.err
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=day

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx
.venv/bin/python partiality/run_pipeline.py
