"""The hkl permutation monitor reads zero for a constant scale."""

from __future__ import annotations

import torch


def test_hkl_permutation_reads_zero_for_a_constant_scale():
    """A constant log_K has no hkl dependence — the guard must say 0, not NaN.

    NaN here would make the degeneracy monitor silently un-assertable in
    exactly the case it should pass (untrained or collapsed model).
    """
    from torchsx.nn.metrics import hkl_permutation_sensitivity

    class _Const:
        def predict(self, meta):
            class P:
                log_K = torch.zeros(meta.hkl_idx.numel())
            return P()

    from torchsx.merging.observations import ObsInfo

    meta = ObsInfo(
        crystal_id=torch.zeros(20, dtype=torch.long), s_sq=torch.rand(20),
        hkl_idx=torch.arange(20) % 5, n_crystals=1, obs_id=torch.arange(20),
    )
    assert hkl_permutation_sensitivity(_Const(), meta) == 0.0
