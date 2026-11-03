"""The likelihood's response to reflection strength, held at the sum of spots.

The sum of spots ``I_lin = sum (n - bg) / sum K`` is unbiased whatever the
scatter model; the likelihood's own estimate is not, when the scatter model
treats dim and bright observations differently from how they behave.  Where the
two part ways with reflection strength the merged amplitudes acquire an
exponent, ``F ~ F_true^gamma`` with ``gamma`` set by the scatter model, not by
the data.

The statistic is the likelihood's estimating equation at the sum of spots.  For
a group of observations of one reflection, ``s`` is the slope of their summed
log-likelihood in ``v = log I`` at ``v = log I_lin`` and ``F`` its curvature
there, so ``s / F`` is one Newton step from ``I_lin`` towards the likelihood's
estimate, ``log(I_ML / I_lin)``.  :func:`response_slope` fits it against
covariates with nuisance columns (a constant and the smooth reciprocal basis
carry the level and its resolution trend, which this term leaves free).  ``s``
and ``F`` are sums over observations at a fixed ``I``, so a batch and a whole
merge add up the same terms.  Evaluated at a sum of spots from ``n``
observations the slope still carries a term of order ``1 / n``: the two
estimates part by different amounts of noise, most for the dimmest groups.
Groups of ``n`` and of ``n / 2`` observations of the same reflections remove
it, ``2 kappa_n - kappa_(n/2)``, so a batch measures what the whole merge would.
"""

from __future__ import annotations

import torch
from torch import Tensor

from torchsx.merging.tether.kmean import psd_solve

__all__ = ["STEPS", "response_slope", "slope_curvature"]

#: Node offsets in units of the node spacing: the curve is read at ``v - d, v, v + d``.
STEPS = (-1.0, 0.0, 1.0)


def slope_curvature(S: Tensor, v_nodes: Tensor) -> tuple[Tensor, Tensor]:
    """Slope (in the graph) and curvature ``F = -S''`` (outside it) of each row's curve
    ``S`` (``n x 3``) at the middle of its three nodes ``v_nodes``, in float64."""
    S, v = S.to(torch.float64), v_nodes.detach().to(torch.float64)
    d1, d2 = v[:, 1] - v[:, 0], v[:, 2] - v[:, 1]
    s = (S[:, 2] - S[:, 0]) / (d1 + d2)
    with torch.no_grad():
        F = -2.0 * ((S[:, 2] - S[:, 1]) / d2 - (S[:, 1] - S[:, 0]) / d1) / (d1 + d2)
    return s, F


def response_slope(s: Tensor, F: Tensor, A: Tensor, X: Tensor) -> tuple[Tensor, Tensor]:
    """The slopes ``kappa`` of ``s / F`` along the covariates ``X`` and their spread, in float64.

    ``s`` (in the graph) and ``F`` (positive, taken outside the graph) per
    group, ``A`` the nuisance columns (``n x p``, with a constant), ``X`` the
    covariates (``n x q``, or ``n``).  The weighted least-squares fit of
    ``s / F`` on ``[A, X]`` with weights ``F`` gives
    ``kappa = (X_hat^T F X_hat)^+ X_hat^T s`` with ``X_hat`` the part of ``X``
    that ``A`` does not explain.  Also returns ``C``, the ``F``-weighted
    covariance of ``X_hat`` (outside the graph): ``kappa^T C kappa`` is the mean
    square, in nats^2, of the log-intensity bend the slopes put into the merge.
    """
    F = F.detach().to(torch.float64)
    A = A.detach().to(torch.float64)
    X = X.detach().to(torch.float64).reshape(F.numel(), -1)
    AF = A * F[:, None]
    X_hat = X - A @ psd_solve(AF.T @ A, AF.T @ X)
    M = (X_hat * F[:, None]).T @ X_hat
    kappa = psd_solve(M, X_hat.T @ s.to(torch.float64))
    return kappa, M / F.sum().clamp(min=1e-300)
