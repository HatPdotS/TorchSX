#!/bin/bash
#SBATCH -J oi
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -o /das/work/units/LBR-FEL/p17489/Peter/2025-01-31_SwissFEL_SFX/svd_time_signal_test/abismal/src/torchsx/experiments/optimize_integration/%x.out
export PYTHONPATH=/das/work/units/LBR-FEL/p17489/Peter/2025-01-31_SwissFEL_SFX/svd_time_signal_test/abismal/src/torchsx
cd /das/work/units/LBR-FEL/p17489/Peter/2025-01-31_SwissFEL_SFX/svd_time_signal_test/abismal/src/torchsx/experiments/optimize_integration
/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u "$@"
