#!/bin/bash
#SBATCH -J quadbench
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 4:00:00
#SBATCH -o /das/work/units/LBR-FEL/p17489/Peter/2025-01-31_SwissFEL_SFX/svd_time_signal_test/abismal/src/torchsx/experiments/optimize_integration/run.out
export PYTHONPATH=/das/work/units/LBR-FEL/p17489/Peter/2025-01-31_SwissFEL_SFX/svd_time_signal_test/abismal/src/torchsx
cd /das/work/units/LBR-FEL/p17489/Peter/2025-01-31_SwissFEL_SFX/svd_time_signal_test/abismal/src/torchsx/experiments/optimize_integration
PY=/das/work/p17/p17490/Peter/torchsx/.venv/bin/python
if [ ! -f /das/work/units/LBR-FEL/p17489/Peter/2025-01-31_SwissFEL_SFX/svd_time_signal_test/abismal/src/torchsx/experiments/optimize_integration/batch.pt ]; then
  $PY -u dump_batch.py --steps ${STEPS:-200} || exit 1
fi
$PY -u bench.py --self-check "$@"
