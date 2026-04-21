#!/bin/bash
#SBATCH --job-name=sparse-comp
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/compress_%A_%a.log
#SBATCH --array=1-126

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx
mkdir -p logs compressed

# Get the h5 file for this array task
H5=$(sed -n "${SLURM_ARRAY_TASK_ID}p" h5_filelist.txt)

echo "Task ${SLURM_ARRAY_TASK_ID}: ${H5}"

OMP_NUM_THREADS=8 .venv/bin/python -u -m torchsx.cli.sparse_compress \
    --stream merged_368.stream \
    --h5 "$H5" \
    --outdir compressed/ \
    --z-threshold 2.0 \
    --min-neighbors 1 \
    --n-radial-bins 500 \
    --batch-size 100
