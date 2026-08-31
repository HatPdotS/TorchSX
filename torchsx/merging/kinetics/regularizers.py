"""Kinetic-ensemble specific regularizers.

Complements :mod:`torchsx.merging.regularizers` (which targets the
standard :class:`Parameters` layout) with regularizers on the kinetic
:class:`KineticEnsembleParameters` layout (per-HKL log_F_dark, F_re,
F_im).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from torchsx.merging.kinetics.parameters import KineticEnsembleParameters


class ExcitedAmplitudeAnchor(nn.Module):
    """Anchor each excited-state F toward the dark state's F.

    Loss contribution::

        λ · Σ_{k, h}  [ (F_re[k, h] − F_dark(h))² + F_im[k, h]² ]

    Encodes the "small photocycle differential" prior: photoactivated
    states typically differ from dark by a few percent in |F|, so
    ``|F_k(h)|`` should be close to ``F_dark(h)`` and the phase small.

    Together with a low ``activation_level`` (fraction of crystals
    that actually photoactivate) this is what breaks the ambiguity
    that let LBFGS/Adam sink F_re and F_im into 0 while F_dark shrank
    to compensate.

    Parameters
    ----------
    lam :
        Strength.  Rough interpretation: at optimum the RMS deviation
        of |F_k| from F_dark contributes ~ 0.5 to the loss per HKL
        when ``λ`` is comparable to ``1 / F_dark_typical²``.
    """

    def __init__(self, lam: float) -> None:
        super().__init__()
        self.lam = float(lam)

    def forward(self, params: KineticEnsembleParameters) -> Tensor:
        if params.n_excited == 0:
            return params.log_F_dark.new_zeros(())
        F_dark = params.F_dark                          # (N_hkl,) real, ≥ 0
        F_re = params.F_re                              # (K-1, N_hkl)
        F_im = params.F_im                              # (K-1, N_hkl)
        d_re = F_re - F_dark.unsqueeze(0)               # (K-1, N_hkl)
        return self.lam * ((d_re * d_re).sum() + (F_im * F_im).sum())

    def extra_repr(self) -> str:  # noqa: D401
        return f"lam={self.lam}"
