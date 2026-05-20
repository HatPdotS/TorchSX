"""Per-observation background-significance gate.

Splits observations into "strong" (data inconsistent with pure background
at significance ``α_test``) and "weak" (consistent with background).
The gate is a function of the observed counts and the *pre-estimated*
background only — completely independent of (μ, α, β) — so the L-BFGS
objective remains a fixed function of the parameters.

Used by the bg-gate L_strong/L_weak split that drives (α, β) gradients
only from observations carrying scatter information, with an explicit
truncation-correction term per strong observation.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def compute_bg_gate(
    n_ph: Tensor,
    bg: Tensor,
    alpha_test: float = 1e-3,
) -> tuple[Tensor, Tensor]:
    """Per-observation significance gate against the pure-background null.

    For each observation `i`:

        p_i      = P(N ≥ n_ph_i | N ~ Poisson(bg_i))
        is_strong_i = (p_i < α_test)
        k_th_i   = smallest integer k such that P(N ≥ k | Pois(bg_i)) < α_test

    The truncation correction in the loss uses k_th_i directly (not the
    intensity threshold T_i = k_th_i − bg_i — that approximation breaks
    down when Poisson noise on the signal is comparable to T_i, which is
    the regime for low-bg crystallography data). See
    ``FixedLogGrid.marginalize_count_survival``.

    Both `is_strong` and `k_th` are fixed once `(n_ph, bg)` are known; no
    parameter dependence.

    Parameters
    ----------
    n_ph
        Per-observation total counts (signal + background), float tensor.
    bg
        Per-observation background-rate estimate (Poisson mean of the
        null), float tensor of the same shape.
    alpha_test
        One-sided significance level. Default ``1e-3`` matches the user's
        spec; tighter values (1e-4) keep fewer, more confident strong
        observations.

    Returns
    -------
    is_strong : BoolTensor of shape ``n_ph.shape``
    k_th      : Tensor of same shape and dtype as ``n_ph``
                (per-observation integer count threshold above which the
                observation rejects the background-only hypothesis at
                ``α_test``; stored as float for downstream gammaincc).
    """
    if n_ph.shape != bg.shape:
        raise ValueError(
            f"n_ph {tuple(n_ph.shape)} and bg {tuple(bg.shape)} must match",
        )
    if not (0.0 < alpha_test < 1.0):
        raise ValueError(f"alpha_test must be in (0, 1), got {alpha_test}")

    # SciPy is the simplest accurate route for the Poisson upper-tail
    # probabilities and the per-bg threshold lookup. Run on CPU once.
    from scipy.stats import poisson as _poisson

    n_np = n_ph.detach().to("cpu").to(torch.float64).numpy()
    bg_np = bg.detach().to("cpu").to(torch.float64).numpy()
    bg_safe = np.maximum(bg_np, 0.0)

    # p_i = P(N >= n_ph_i | Poisson(bg_i)) = sf(n_ph_i - 1, bg_i)
    # n_ph may be non-integer (gain-converted); round down for the lookup.
    n_int = np.floor(n_np).astype(np.int64)
    n_int = np.clip(n_int, 0, None)
    p_vals = _poisson.sf(np.maximum(n_int - 1, -1), bg_safe)
    is_strong_np = p_vals < alpha_test

    # k_th(b̂) = smallest integer k with sf(k-1, b̂) < α_test
    #         = ppf(1 − α_test, b̂) + 1 in most cases, but the Poisson PPF
    # has a small off-by-one on lattice; compute explicitly via isf:
    #   isf(α, μ) returns smallest k with sf(k, μ) ≤ α
    # so k_th = isf(α_test, b̂) + 1 (we want strict inequality on sf).
    # Use a uniqueness-aware path: compute on unique bg values for speed.
    bg_unique, inv = np.unique(bg_safe, return_inverse=True)
    k_th_unique = _poisson.isf(alpha_test, bg_unique).astype(np.float64) + 1.0
    # Guard against scipy returning inf for bg=0 + α_test close to 1
    k_th_unique = np.where(np.isfinite(k_th_unique), k_th_unique, 1.0)
    k_th_np = k_th_unique[inv]

    is_strong = torch.from_numpy(is_strong_np).to(n_ph.device).bool()
    k_th = torch.from_numpy(k_th_np).to(device=n_ph.device, dtype=n_ph.dtype)
    return is_strong, k_th
