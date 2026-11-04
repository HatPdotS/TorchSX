"""Abstract base for multiplicative-scatter density modules."""

from __future__ import annotations

from abc import abstractmethod

import torch.nn as nn
from torch import Tensor

from torchsx.merging.observations import ObsInfo


class ScatterModel(nn.Module):
    """Density of ``log S`` around ``log_mean = log(K · I_true)``.

    Concrete implementations must return the log-density evaluated at a
    per-observation / per-grid-point ``eps = log_S − log_mean`` tensor.

    A subclass supplies the **per-observation** shape parameters (e.g.
    predicted by a network) through :meth:`log_alpha_beta_obs`.  Because the
    eager path, the Triton kernels, the posterior moments and the residual
    diagnostics all route through this one hook, such a subclass needs no
    other change anywhere in the stack.  The hooks' ``params`` argument is
    always ``None``.
    """

    def log_alpha_beta_obs(
        self, params: None, meta: ObsInfo
    ) -> tuple[Tensor, Tensor]:
        """Per-observation ``(log α, log β)``."""
        raise NotImplementedError(f"{type(self).__name__} supplies no per-observation shape")

    def log_alpha_hi_obs(self, params: None, meta: ObsInfo) -> Tensor | None:
        """Per-observation upper-side ``log α`` for the two-width (asymmetric)
        GenGauss, or ``None`` for the symmetric family.  Default: symmetric."""
        return None

    @abstractmethod
    def log_prob(
        self,
        eps: Tensor,          # (n_obs, K) — log S − log_mean at each grid point
        params: None,
        meta: ObsInfo,
    ) -> Tensor:              # (n_obs, K)
        ...
