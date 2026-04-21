#!/bin/bash
#SBATCH --job-name=sparse-z1.5
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=sparse_model/logs/compress_z1.5_%A_%a.log
#SBATCH --array=1-126

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx
mkdir -p sparse_model/logs /das/work/p17/p17490/Peter/torchsx/compressed_z1.5

# Get the h5 file for this array task
H5=$(sed -n "${SLURM_ARRAY_TASK_ID}p" h5_filelist.txt)

echo "Task ${SLURM_ARRAY_TASK_ID}: ${H5}"

OMP_NUM_THREADS=8 .venv/bin/python -u -m torchsx.cli.sparse_compress \
    --stream merged_368.stream \
    --h5 "$H5" \
    --outdir /das/work/p17/p17490/Peter/torchsx/compressed_z1.5/ \
    --z-threshold 1.5 \
    --min-neighbors 2 \
    --n-radial-bins 500 \
    --batch-size 100
