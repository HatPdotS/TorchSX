#!/bin/bash
#SBATCH --job-name=copy_h5
#SBATCH --partition=hour
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=copy_h5_%j.log

set -e
cd /das/work/units/LBR-FEL/p17490/Peter/torchsx

echo "Copying 126 HDF5 files (~460 GB) from alvra..."
echo "Start: $(date)"

# Run 4 copies in parallel using xargs
cat copy_h5.sh | xargs -P 4 -I {} bash -c '{}'

echo "Done: $(date)"
echo "Total size:"
du -sh h5files/
