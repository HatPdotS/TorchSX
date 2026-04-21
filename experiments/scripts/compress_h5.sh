#!/bin/bash
#SBATCH --job-name=sparse-comp
#SBATCH --partition=gpu-day
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=compress_h5_%j.log

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx

.venv/bin/python -u -m torchsx.cli.sparse_compress \
    --stream merged_368.stream \
    --h5 h5files/run0121-s2_inj55_scan_0p5uJ/acq*.JF06T32V07.h5 \
    --outdir compressed/ \
    --z-threshold 2.0 \
    --min-neighbors 1 \
    --n-radial-bins 500 \
    --batch-size 100
