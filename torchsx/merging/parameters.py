"""Central container for the learnable per-HKL state of the merger.

Scale / scatter / likelihood submodules carry their own per-crystal or
per-panel parameters (they're still ``nn.Parameter``s and autograd sees
them through the module tree).  ``Parameters`` here holds exactly what
is *shared* across modules: the per-HKL latent log-intensity and its
GenGauss scatter exponents.

Design-doc §4.1 additionally lists `log_alpha_shell`/`log_beta_shell` +
per-HKL `log_gamma`; the user chose to keep per-HKL α, β directly
(refactor-only, no modeling change) so those fields are not introduced
here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from torchsx.merging.observations import PreparedObs


class Parameters(nn.Module):
    """Per-HKL learnable parameters shared across merging-loss modules.

    Fields
    ------
    log_I0      (n_hkl,) — log latent intensity; MLE-seeded from positive obs.
    log_alpha   (n_hkl,) — GenGauss log-scale in log-intensity space.
    log_beta    (n_hkl,) — GenGauss shape exponent (log-parameterised so β>0).
    pi_logit    (n_hkl,) — logit of the per-HKL background mixture weight
                            π_h ∈ (0, 1); only active when the likelihood
                            subclass reads it.
    """

    def __init__(
        self,
        log_I0_init: Tensor,
        log_alpha_init: Tensor,
        log_beta_init: Tensor,
        pi_logit_init: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.log_I0 = nn.Parameter(log_I0_init.clone())
        self.log_alpha = nn.Parameter(log_alpha_init.clone())
        self.log_beta = nn.Parameter(log_beta_init.clone())
        if pi_logit_init is None:
            pi_logit_init = torch.zeros_like(log_I0_init)
        self.pi_logit = nn.Parameter(pi_logit_init.clone())

    @classmethod
    def from_prep(cls, prep: PreparedObs) -> "Parameters":
        """Seed parameters from ``PreparedObs`` init estimates."""
        log_alpha_init = torch.log(prep.alpha_init.clamp(min=1e-3))
        log_beta_init = torch.log(prep.beta_perhkl.clamp(min=1e-3))
        return cls(
            log_I0_init=prep.mu_init,
            log_alpha_init=log_alpha_init,
            log_beta_init=log_beta_init,
        )

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    @property
    def beta(self) -> Tensor:
        return self.log_beta.exp()

    @property
    def pi(self) -> Tensor:
        return torch.sigmoid(self.pi_logit)
