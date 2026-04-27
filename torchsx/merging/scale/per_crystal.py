"""Per-crystal overall scale + B-factor correction.

Equivalent to the legacy ``ScaleBCorrection`` in merging_clean.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from torchsx.merging.observations import ObsInfo
from torchsx.merging.parameters import Parameters
from torchsx.merging.scale.base import ScaleModel


class ScaleBPerCrystal(ScaleModel):
    """Per-crystal overall scale (log G) and B-factor.

    Log-space model::

        log K_j = (log_G − mean(log_G))[c_j]  −  2 · B[c_j] · s²_j

    The centred ``log_G`` enforces zero-mean across crystals (otherwise
    it trades with the per-HKL ``log_I0`` up to a global constant).
    Ridge penalty on centred log_G and on B anchors magnitudes — see
    :meth:`regularizer`.
    """

    def __init__(
        self,
        n_crystals: int,
        *,
        scale_prior: float = 10.0,
        bfactor_prior: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_crystals = int(n_crystals)
        self.scale_prior = float(scale_prior)
        self.bfactor_prior = float(bfactor_prior)
        self.log_G_raw = nn.Parameter(torch.zeros(n_crystals))
        self.B = nn.Parameter(torch.zeros(n_crystals))

    @classmethod
    def from_obs_info(
        cls,
        obs_info: ObsInfo,
        *,
        scale_prior: float = 10.0,
        bfactor_prior: float = 1.0,
    ) -> "ScaleBPerCrystal":
        return cls(
            n_crystals=int(obs_info.n_crystals),
            scale_prior=scale_prior,
            bfactor_prior=bfactor_prior,
        )

    def _log_G_centred(self) -> Tensor:
        return self.log_G_raw - self.log_G_raw.mean()

    def log_k(
        self,
        params: Parameters,
        meta: ObsInfo,
        obs_range: tuple[int, int] | None = None,
    ) -> Tensor:
        del params, obs_range  # indexed by crystal_id; slicing handled by caller
        cid = meta.crystal_id
        return self._log_G_centred()[cid] - 2.0 * self.B[cid] * meta.s_sq

    def regularizer(self, params: Parameters) -> Tensor:
        del params
        lg = self._log_G_centred()
        return (
            self.scale_prior * (lg * lg).sum()
            + self.bfactor_prior * (self.B * self.B).sum()
        )
