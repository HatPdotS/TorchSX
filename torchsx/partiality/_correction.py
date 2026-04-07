"""PartialityCorrection: bridge partiality model to the EM merge pipeline."""

from __future__ import annotations

import torch
from torch import Tensor

from torchsx.merge._corrections import Correction, ObsInfo
from ._excitation import ExcitationError
from ._profile import PartialityProfile


class PartialityCorrection(Correction):
    """Wraps ExcitationError + PartialityProfile as an EM Correction.

    On ``predict(obs_info)``: computes |r| from ExcitationError, then
    log_p from the profile, and returns -log_p (mean-subtracted) as
    a log-space correction.  Mean subtraction ensures the correction
    provides *shape only* — the overall scale is absorbed by ScaleB.

    On ``fit()``: no-op.  Parameters are fit separately by rank
    concordance (orientation) and NLL (profile).

    Parameters
    ----------
    excitation : ExcitationError — provides |r| per observation.
    profile : PartialityProfile — maps (|r|, 1/d) to log(p).
    hkl_obs : (N_obs, 3) — Miller indices (stream order).
    inv_d : (N_obs,) — 1/d per observation (stream order).
    sort_idx : (N_sorted,) — index mapping from stream order to sorted
        order used by the EM loop.  Set via ``set_sort_idx()`` before
        the first EM cycle.
    """

    def __init__(
        self,
        excitation: ExcitationError,
        profile: PartialityProfile,
        hkl_obs: Tensor,
        crystal_id: Tensor,
        inv_d: Tensor,
    ):
        self.excitation = excitation
        self.profile = profile
        self._hkl_obs = hkl_obs
        self._crystal_id = crystal_id
        self._inv_d = inv_d
        self._sort_idx: Tensor | None = None
        self._log_c_sorted: Tensor | None = None

    def set_sort_idx(self, sort_idx: Tensor) -> None:
        """Set the sort index mapping from stream to EM-sorted order."""
        self._sort_idx = sort_idx

    def _compute(self) -> Tensor:
        """Compute mean-subtracted -log_p in stream order."""
        with torch.no_grad():
            abs_r = self.excitation.abs_excitation_error(
                self._hkl_obs, self._crystal_id,
            )
            log_p = self.profile.log_partiality(abs_r, self._inv_d)
            # Mean-subtract: shape only, overall scale absorbed by ScaleB
            correction = -(log_p - log_p.mean())
        return correction

    def fit(self, residuals: Tensor, obs_info: ObsInfo, work_mask: Tensor) -> None:
        """No-op — partiality is fit separately."""
        pass

    def predict(self, obs_info: ObsInfo) -> Tensor:
        """Return per-observation correction in EM-sorted order."""
        correction = self._compute()  # stream order
        if self._sort_idx is not None:
            return correction[self._sort_idx]
        return correction

    def n_params(self) -> int:
        return sum(p.numel() for p in self.profile.parameters())
