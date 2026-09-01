#!/bin/bash
cd /das/work/units/LBR-FEL/p17489/Peter/2025-01-31_SwissFEL_SFX/svd_time_signal_test/abismal/src/torchsx
/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -m pytest -q \
  tests/test_merging_layers.py tests/test_merging_chunked_data_nll.py \
  tests/test_merging_posterior.py tests/test_marginal_normalisation.py \
  tests/test_amortized_training.py tests/test_amortized_scale.py \
  tests/test_gaussian_kernel.py tests/test_consistency_loss.py 2>&1 | tail -20
