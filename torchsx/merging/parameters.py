"""Central container for the learnable per-HKL state of the merger.

Scale / scatter / likelihood submodules carry their own per-crystal or
per-panel parameters (they're still ``nn.Parameter``s and autograd sees
them through the module tree).  ``Parameters`` here holds exactly what
is *shared* across modules: the per-HKL latent log-intensity and the
GenGauss scatter exponents — the latter via a swappable
:class:`~torchsx.merging.scatter.shape_model.ScatterShape` submodule so
the parametrization (free per-HKL ↔ single global ↔ fixed at truth) can
be chosen at construction without rewiring the loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from torchsx.merging.observations import PreparedObs
from torchsx.merging.scatter_shape import FreePerHKLShape, ScatterShape


class Parameters(nn.Module):
    """Per-HKL learnable parameters shared across merging-loss modules.

    Fields
    ------
    log_I0          (n_hkl,) — log latent intensity; MLE-seeded from positive obs.
    scatter_shape   ScatterShape submodule producing log_α / log_β as
                    ``(n_hkl,)`` tensors.  ``log_alpha`` / ``log_beta`` are
                    read-only properties that delegate to it.
    pi_logit        (n_hkl,) — logit of the per-HKL background mixture weight
                    π_h ∈ (0, 1); only active when the likelihood subclass
                    reads it.
    q_a, q_b        scalars defining the per-obs background-mixture weight
                    ``q(|r|) = sigmoid(q_a + q_b · |r| · 1e3)`` (|r| scaled to
                    ×10⁻³ Å⁻¹ units so q_b is order-unity). Only read by the
                    ``GaussianPolynomialWithBgMixture`` likelihood; no-ops for
                    the plain GP / Poisson paths.
    """

    def __init__(
        self,
        log_I0_init: Tensor,
        scatter_shape: ScatterShape,
        pi_logit_init: Tensor | None = None,
        q_a_init: float = -2.197,
        q_b_init: float = 0.514,
    ) -> None:
        super().__init__()
        self.log_I0 = nn.Parameter(log_I0_init.clone())
        self.scatter_shape = scatter_shape
        if pi_logit_init is None:
            pi_logit_init = torch.zeros_like(log_I0_init)
        self.pi_logit = nn.Parameter(pi_logit_init.clone())
        self.q_a = nn.Parameter(torch.tensor(float(q_a_init), dtype=torch.float32))
        self.q_b = nn.Parameter(torch.tensor(float(q_b_init), dtype=torch.float32))

    @property
    def log_alpha(self) -> Tensor:
        return self.scatter_shape.log_alpha()

    @property
    def log_beta(self) -> Tensor:
        return self.scatter_shape.log_beta()

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    @property
    def beta(self) -> Tensor:
        return self.log_beta.exp()

    @property
    def pi(self) -> Tensor:
        return torch.sigmoid(self.pi_logit)
