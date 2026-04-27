"""Soft-assignment loss for sparse SFX data.

EM-style probabilistic assignment of sparse pixels to predicted reflections.
For each reflection's ROI, each pixel gets a distance-based assignment weight
and a model-based responsibility (signal fraction).  The loss is a weighted
Poisson NLL where pixels near the predicted center and with high predicted
signal contribute most.

The assignment width σ can be annealed: start wide (coarse alignment) and
narrow as the model improves (fine pixel-level fitting).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn, Tensor


@dataclass
class SoftAssignmentResult:
    """Result from SoftAssignmentLoss."""

    loss: Tensor          # scalar — mean weighted Poisson NLL per observation
    n_obs: int            # number of observations
    mean_weight: float    # mean assignment weight at sparse pixels — diagnostic
    mean_responsibility: float  # mean signal/(signal+bg) at sparse pixels

    val_loss: Tensor | None = None
    n_val: int = 0


class SoftAssignmentLoss(nn.Module):
    """Soft-assignment weighted Poisson NLL for sparse pixel data.

    Each pixel's Poisson NLL is weighted by two factors:

    1. **Distance prior** ``G(d, σ)``: Gaussian from the predicted spot center.
       Pixels far from the prediction contribute less. Width σ controls the
       search range — start wide for coarse alignment, narrow for refinement.

    2. **Responsibility** ``r = signal / (signal + bg)``: the model's own
       estimate of what fraction of counts at this pixel come from this
       reflection.  Self-consistent: as the model improves, responsibilities
       sharpen, focusing the loss on the most informative pixels.

    The combined weight ``w = G(d, σ) × (r + r_floor)`` ensures:
    - Pixels near the prediction with high predicted signal → strong weight
    - Pixels far away or background-dominated → low weight
    - r_floor prevents zero weight everywhere when signal is small

    Parameters
    ----------
    sigma : float
        Assignment width in pixels. Start at 3–5 for coarse alignment.
    r_floor : float
        Minimum responsibility floor. Prevents zero weights when predicted
        signal is very small, ensuring some gradient always flows.
    eps : float
        Rate clamp for numerical stability.
    val_frac : float
        Fraction of observations held out for validation.
    """

    def __init__(
        self,
        sigma: float = 3.0,
        r_floor: float = 0.05,
        eps: float = 1e-3,
        val_frac: float = 0.0,
    ):
        super().__init__()
        self.sigma = sigma
        self.r_floor = r_floor
        self.eps = eps
        self.val_frac = val_frac
        self._dist_weight: Tensor | None = None
        self._dist_shape: tuple[int, int] | None = None

    def _get_distance_weight(
        self, ph: int, pw: int, device: torch.device,
    ) -> Tensor:
        """Build and cache the distance-based Gaussian weight (ph, pw)."""
        if self._dist_weight is not None and self._dist_shape == (ph, pw):
            return self._dist_weight.to(device)

        ss = torch.arange(ph, device=device, dtype=torch.float32) - ph // 2
        fs = torch.arange(pw, device=device, dtype=torch.float32) - pw // 2
        ss_g, fs_g = torch.meshgrid(ss, fs, indexing="ij")
        d_sq = ss_g ** 2 + fs_g ** 2
        w = torch.exp(-d_sq / (2.0 * self.sigma ** 2))
        self._dist_weight = w
        self._dist_shape = (ph, pw)
        return w

    def set_sigma(self, sigma: float):
        """Update the assignment width (e.g. for annealing). Clears cache."""
        self.sigma = sigma
        self._dist_weight = None
        self._dist_shape = None

    def forward(
        self,
        n_signal: Tensor,       # (B, ph, pw) predicted signal
        background: Tensor,     # (B, ph, pw) per-pixel background
        is_sparse: Tensor,      # (B, ph, pw) bool
        sparse_counts: Tensor,  # (B, ph, pw) int32
        threshold: Tensor,      # (B, ph, pw) float — unused, API compat
        valid: Tensor,          # (B, ph, pw) bool
        gain: Tensor,           # (B, 1, 1) or (B, ph, pw)
    ) -> SoftAssignmentResult:
        B, ph, pw = n_signal.shape
        dev = n_signal.device

        # ── Assignment weights ────────────────────────────────────
        # Distance prior: G(d, σ) — precomputed (ph, pw)
        w_dist = self._get_distance_weight(ph, pw, dev)  # (ph, pw)

        # Predicted rate
        lam_signal = gain * n_signal                      # (B, ph, pw)
        lam_bg = gain * background                        # (B, ph, pw)
        lam = (lam_signal + lam_bg).clamp(min=self.eps)

        # Pure distance-based weight — no responsibility term.
        # Responsibility can be added later as a hyperparameter/annealing.
        w = w_dist  # (ph, pw) broadcasts to (B, ph, pw)

        # ── Only sparse pixels contribute to the loss ────────────
        # Censored pixels add a large negative constant that swamps
        # the signal.  Distance weighting on sparse pixels only.
        sparse_bool = is_sparse > 0
        valid_bool = valid > 0
        mask = sparse_bool & valid_bool
        w = w * mask.float()

        # ── Weighted Poisson NLL per pixel ────────────────────────
        # L_i = w_i × (λ_i - k_i × log(λ_i))
        k = sparse_counts.float()
        per_pixel_nll = w * (lam - k * torch.log(lam))

        # ── Per-observation NLL (sum over pixels in each ROI) ─────
        nll_per_obs = per_pixel_nll.sum(dim=(1, 2))  # (B,)

        # ── Diagnostics ──────────────────────────────────────────
        with torch.no_grad():
            if mask.any():
                mean_w = w[mask].mean().item()
                # Responsibility diagnostic (detached)
                r = (lam_signal / lam).detach()
                mean_r = r[mask].mean().item()
            else:
                mean_w = 0.0
                mean_r = 0.0

        # ── Train/val split (per-observation) ────────────────────
        if self.val_frac > 0 and B > 4:
            n_val = max(1, int(B * self.val_frac))
            perm = torch.randperm(B, device=dev)
            val_idx = perm[:n_val]
            train_idx = perm[n_val:]

            train_loss = nll_per_obs[train_idx].mean()
            with torch.no_grad():
                val_loss = nll_per_obs[val_idx].mean()

            return SoftAssignmentResult(
                loss=train_loss,
                n_obs=train_idx.numel(),
                mean_weight=mean_w,
                mean_responsibility=mean_r,
                val_loss=val_loss,
                n_val=val_idx.numel(),
            )

        return SoftAssignmentResult(
            loss=nll_per_obs.mean(),
            n_obs=B,
            mean_weight=mean_w,
            mean_responsibility=mean_r,
        )
