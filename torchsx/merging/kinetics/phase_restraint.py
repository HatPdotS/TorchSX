"""Phase restraint: penalise large phase rotations away from the dark gauge.

In the dark-phase-zero gauge ``F_dark(h) ∈ ℝ₊``, the phase of an
excited state at HKL ``h`` is ``atan2(F_im_k[h], F_re_k[h])``.  The
small-angle Taylor expansion of the phase is

.. math::

    \\varphi_k(h) \\approx F_{\\text{im}, k}(h) / |F_k(h)|

so an L2 prior on ``F_im_k`` is — in the small-angle, well-measured
amplitude limit — a Gaussian prior on the phase with width controlled
by ``λ``.  Centric HKLs already have ``F_im ≡ 0`` enforced by the
parameter object's centric mask; the restraint only weighs acentric
HKLs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class PhaseRestraint(nn.Module):
    """``λ · Σ_{k ≥ 1, h ∉ centric} F_im_k[h]²``.

    Parameters
    ----------
    lam :
        Restraint strength.  Larger → tighter phase pull.  A good
        starting value keeps the typical |F_im| at least 5–10 × below
        the typical |F_re|, i.e. phase rotations within ~10–20°.
    """

    def __init__(self, lam: float) -> None:
        super().__init__()
        self.lam = float(lam)

    def forward(self, params) -> Tensor:
        # ``params`` is :class:`KineticEnsembleParameters`.
        if params.n_excited == 0:
            return params.log_F_dark.new_zeros(())
        acentric = ~params.centric_mask
        if not acentric.any():
            return params.log_F_dark.new_zeros(())
        F_im_acentric = params.F_im[:, acentric]
        return self.lam * (F_im_acentric ** 2).sum()

    def extra_repr(self) -> str:
        return f"lam={self.lam}"
