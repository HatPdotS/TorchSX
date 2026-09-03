"""Abstract base for multiplicative-scatter density modules."""

from __future__ import annotations

from abc import abstractmethod

import torch.nn as nn
from torch import Tensor

from torchsx.merging.observations import ObsInfo
from torchsx.merging.parameters import Parameters


class ScatterModel(nn.Module):
    """Density of ``log S`` around ``log_mean = log(K · I_true)``.

    Concrete implementations must return the log-density evaluated at a
    per-observation / per-grid-point ``eps = log_S − log_mean`` tensor.

    The scatter is stateless with respect to observations; any per-HKL
    parameters (e.g. GenGauss α, β per HKL) are read from ``params`` and
    indexed with ``meta.hkl_idx`` — via :meth:`log_alpha_beta_obs`, which is
    the single seam where that indexing happens.

    Overriding :meth:`log_alpha_beta_obs` is how a subclass supplies
    **per-observation** shape parameters (e.g. predicted by a network) instead
    of per-HKL ones.  Because the eager path, both Triton kernels, the
    posterior moments and the residual diagnostics all route through this one
    hook, such a subclass needs no other change anywhere in the stack.
    """

    def log_alpha_beta_obs(
        self, params: Parameters, meta: ObsInfo
    ) -> tuple[Tensor, Tensor]:
        """Per-observation ``(log α, log β)``.

        Default: gather the per-HKL parameters with ``meta.hkl_idx``.
        """
        h = meta.hkl_idx
        return params.log_alpha[h], params.log_beta[h]

    def log_alpha_hi_obs(self, params: Parameters, meta: ObsInfo) -> Tensor | None:
        """Per-observation upper-side ``log α`` for the two-width (asymmetric)
        GenGauss, or ``None`` for the symmetric family.  Default: symmetric."""
        return None

    @abstractmethod
    def log_prob(
        self,
        eps: Tensor,          # (n_obs, K) — log S − log_mean at each grid point
        params: Parameters,
        meta: ObsInfo,
    ) -> Tensor:              # (n_obs, K)
        ...
