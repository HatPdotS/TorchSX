#!/bin/bash
#SBATCH --job-name=vis-sparse
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=sparse_model/logs/visualize_%j.log

cd /das/work/units/LBR-FEL/p17490/Peter/torchsx

SPARSE=${1:-/das/work/p17/p17490/Peter/torchsx/compressed_z1.5/run0121-s2_inj55_scan_0p5uJ_acq0001.JF06T32V07.sparse.safetensors}
FRAME=${2:-}

FRAME_ARG=""
[ -n "$FRAME" ] && FRAME_ARG="--frame $FRAME"

.venv/bin/python -u -m torchsx.cli.visualize_sparse \
    --sparse "$SPARSE" \
    --stream merged_368.stream \
    $FRAME_ARG \
    --output "sparse_model/$(basename "${SPARSE%.sparse.safetensors}")_frame.svg"
