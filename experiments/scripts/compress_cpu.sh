#!/bin/bash
#SBATCH --job-name=sparse-cpu
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=compress_cpu_%j.log

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx

.venv/bin/python -u -m torchsx.cli.sparse_compress \
    --stream merged_368.stream \
    --h5 h5files/run0121-s2_inj55_scan_0p5uJ/acq0001.JF06T32V07.h5 \
    --outdir compressed_cpu/ \
    --z-threshold 2.0 \
    --min-neighbors 1 \
    --n-radial-bins 500 \
    --batch-size 100
