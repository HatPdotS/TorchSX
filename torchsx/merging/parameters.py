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
    """

    def __init__(
        self,
        log_I0_init: Tensor,
        scatter_shape: ScatterShape,
        pi_logit_init: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.log_I0 = nn.Parameter(log_I0_init.clone())
        self.scatter_shape = scatter_shape
        if pi_logit_init is None:
            pi_logit_init = torch.zeros_like(log_I0_init)
        self.pi_logit = nn.Parameter(pi_logit_init.clone())

    @classmethod
    def from_prep(
        cls,
        prep: PreparedObs,
        scatter_shape: ScatterShape | None = None,
    ) -> "Parameters":
        """Seed parameters from ``PreparedObs`` init estimates.

        ``scatter_shape=None`` instantiates the legacy
        :class:`FreePerHKLShape` from ``prep.alpha_init`` /
        ``prep.beta_perhkl`` so callers that don't pass anything see
        identical behaviour to the previous per-HKL ``nn.Parameter``
        layout.
        """
        if scatter_shape is None:
            log_alpha_init = torch.log(prep.alpha_init.clamp(min=1e-3))
            log_beta_init = torch.log(prep.beta_perhkl.clamp(min=1e-3))
            scatter_shape = FreePerHKLShape(log_alpha_init, log_beta_init)
        return cls(log_I0_init=prep.mu_init, scatter_shape=scatter_shape)

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
