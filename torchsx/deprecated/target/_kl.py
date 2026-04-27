"""KLScalingTarget: unified log-space NLL + floor penalty scaling target."""

from __future__ import annotations

import math

import torch
from torch import nn

_SQRT2 = math.sqrt(2.0)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)

# Minimum log Φ to avoid log(0) — corresponds to Φ ≈ 1e-20
_MIN_LOG_PHI = -46.0


def _log_ndtr(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log Φ(x) = log(0.5 * (1 + erf(x/√2))).

    Uses the complementary form log(1 - Φ(-x)) = log erfc(-x/√2) - log(2)
    for large negative x to avoid log(0).
    """
    # For x >= -5, the standard form is fine
    # For x < -5, use log erfc form
    safe = x >= -5.0
    result = torch.empty_like(x)
    result[safe] = torch.log(
        (0.5 * (1.0 + torch.erf(x[safe] / _SQRT2))).clamp(min=1e-30)
    )
    if (~safe).any():
        # log erfc(-x/√2) - log 2 = log Φ(x) for large negative x
        result[~safe] = torch.log(
            torch.erfc(-x[~safe] / _SQRT2).clamp(min=1e-30)
        ) - math.log(2.0)
    return result.clamp(min=_MIN_LOG_PHI)


class KLScalingTarget(nn.Module):
    """Unified log-space NLL scaling target with floor penalty.

    Owns the structure factors (log_I_true) and per-shell noise parameters.
    The loss has three components:

    1. **Positive observations — log-space NLL**: δ²/(2σ_log²) + log(σ_log²)/2,
       where δ = log(I_obs) - log(scale) - log(I_true) and
       σ_log² = σ_LN² + σ_gauss²/I_true². Provides direct, un-suppressed
       gradients to each crystal's log scale factor.

    2. **Non-positive observations — truncated Gaussian NLL**: constrains
       I_true from above for weak reflections where additive noise pushes
       I_obs below zero. Corrects the positive-selection bias that would
       otherwise inflate I_true for reflections near the noise floor.

    3. **Floor penalty**: anchors σ_LN and σ_gauss to calibrated per-shell
       priors representing the irreducible noise floor, preventing the noise
       model from absorbing scale factor variance.

    Noise parameters (σ_LN, σ_gauss) are per-shell, not per-reflection,
    to prevent the noise model from absorbing per-reflection structure.

    Parameters
    ----------
    n_unique : int
        Number of unique reflections.
    n_shells : int
        Number of resolution shells for noise estimation and prior anchoring.
    lam : float
        Weight of the floor penalty term.
    """

    def __init__(self, n_unique: int, n_shells: int, lam: float = 1.0):
        super().__init__()
        self.log_I_true = nn.Parameter(torch.zeros(n_unique))
        # Noise params are per-shell (not per-reflection) to prevent
        # the noise model from absorbing per-reflection structure.
        self.log_sigma_LN = nn.Parameter(torch.full((n_shells,), math.log(0.1)))
        self.log_sigma_gauss = nn.Parameter(torch.full((n_shells,), math.log(10.0)))
        self.register_buffer("sigma_LN_prior", torch.full((n_shells,), 0.05))
        self.register_buffer("sigma_gauss_prior", torch.full((n_shells,), 10.0))
        self.register_buffer("shell_per_hkl", torch.zeros(n_unique, dtype=torch.long))
        self.lam = lam

    @property
    def I_true(self) -> torch.Tensor:
        return torch.exp(self.log_I_true)

    @property
    def sigma_LN(self) -> torch.Tensor:
        return torch.exp(self.log_sigma_LN)

    @property
    def sigma_gauss(self) -> torch.Tensor:
        return torch.exp(self.log_sigma_gauss)

    def _sigma_LN_per_hkl(self) -> torch.Tensor:
        """Broadcast per-shell sigma_LN to per-reflection."""
        return self.sigma_LN[self.shell_per_hkl]

    def _sigma_gauss_per_hkl(self) -> torch.Tensor:
        """Broadcast per-shell sigma_gauss to per-reflection."""
        return self.sigma_gauss[self.shell_per_hkl]

    def calibrate(
        self,
        I_obs: torch.Tensor,
        total_scale: torch.Tensor,
        hkl_id: torch.Tensor,
        shell: torch.Tensor,
        crystal_id: torch.Tensor | None = None,
    ) -> None:
        """Initialize parameters from data.

        Estimates the irreducible noise floor by optionally removing
        inter-crystal scale variation (when crystal_id is provided),
        then separates into Gaussian and lognormal components per shell.

        Parameters
        ----------
        I_obs : (N_obs,) float32
        total_scale : (N_obs,) float32
            Current total scale (use ones for initial calibration).
        hkl_id : (N_obs,) int64
        shell : (N_obs,) int64
        crystal_id : (N_obs,) int32, optional
            Crystal assignments. When provided, rough per-crystal scales
            are estimated and removed so the priors reflect the noise
            floor rather than total (noise + scale) variance.
        """
        n_unique = self.log_I_true.shape[0]
        n_shells = self.sigma_LN_prior.shape[0]
        device = self.log_I_true.device

        # Scale-reduce observations (use only positive for log-space stats)
        r = I_obs / total_scale.clamp(min=1e-30)
        pos = I_obs > 0
        r_pos = torch.where(pos, r, torch.zeros_like(r))
        w_pos = pos.float()

        # Per-reflection empirical means (from positive observations only)
        ones = torch.ones_like(r)
        count = torch.zeros(n_unique, device=device).scatter_add_(0, hkl_id, ones)
        count_pos = torch.zeros(n_unique, device=device).scatter_add_(0, hkl_id, w_pos)
        sum_r = torch.zeros(n_unique, device=device).scatter_add_(0, hkl_id, r_pos)
        n_clamp = count_pos.clamp(min=1)
        mu_hkl = sum_r / n_clamp

        # Rough per-crystal scale removal for noise floor estimation
        if crystal_id is not None and pos.any():
            n_crystals = crystal_id.max().item() + 1
            ratio = r_pos / mu_hkl[hkl_id].clamp(min=1e-10)
            # Use mean ratio per crystal (positive obs only)
            crys_long = crystal_id.long()
            c_count = torch.zeros(n_crystals, device=device).scatter_add_(0, crys_long, w_pos)
            c_sum = torch.zeros(n_crystals, device=device).scatter_add_(0, crys_long, ratio * w_pos)
            g_est = c_sum / c_count.clamp(min=1)
            # Correct positive observations by estimated crystal scale
            r_corrected = r_pos / g_est[crys_long].clamp(min=1e-10)
            r_corrected = torch.where(pos, r_corrected, torch.zeros_like(r))
        else:
            r_corrected = r_pos

        # Per-reflection stats from corrected positive observations (noise floor)
        sum_rc = torch.zeros(n_unique, device=device).scatter_add_(0, hkl_id, r_corrected)
        sum_rc2 = torch.zeros(n_unique, device=device).scatter_add_(0, hkl_id, r_corrected * r_corrected)

        valid = count_pos >= 2
        n = count_pos.clamp(min=2)
        mu_emp = sum_rc / n
        var_emp = ((sum_rc2 - sum_rc * sum_rc / n) / (n - 1).clamp(min=1)).clamp(min=1e-10)

        # Initialize I_true from uncorrected per-reflection mean
        I_init = (sum_r / n_clamp).clamp(min=1.0)
        with torch.no_grad():
            self.log_I_true.copy_(torch.log(I_init))

        # Assign shell per hkl — mode of per-obs shells via scatter
        shell_count = torch.zeros(n_unique, n_shells, device=device)
        shell_count.scatter_add_(
            0,
            hkl_id.unsqueeze(1).expand(-1, n_shells),
            torch.nn.functional.one_hot(shell.long(), n_shells).float(),
        )
        shell_per_hkl = shell_count.argmax(dim=1)
        with torch.no_grad():
            self.shell_per_hkl.copy_(shell_per_hkl)

        # Per-shell noise separation (using corrected variance = noise floor)
        sigma_gauss_shell = torch.zeros(n_shells, device=device)
        sigma_LN_shell = torch.zeros(n_shells, device=device)

        for s in range(n_shells):
            mask = (shell_per_hkl == s) & valid
            if mask.sum() < 2:
                sigma_gauss_shell[s] = 10.0
                sigma_LN_shell[s] = 0.1
                continue
            I_shell = I_init[mask]
            var_shell = var_emp[mask]

            # Gaussian floor ~ sqrt(mean intensity) (Poisson-like)
            sg2 = I_shell.mean().clamp(min=1.0)
            sigma_gauss_shell[s] = sg2.sqrt()

            # Lognormal component: var_reducible = var_emp - sigma_gauss²
            var_LN = ((var_shell - sg2) / (I_shell * I_shell).clamp(min=1.0)).clamp(min=1e-8)
            sigma_LN_shell[s] = var_LN.mean().sqrt()

        # Set priors and initialize per-shell params
        with torch.no_grad():
            self.sigma_LN_prior.copy_(sigma_LN_shell.clamp(min=1e-4))
            self.sigma_gauss_prior.copy_(sigma_gauss_shell.clamp(min=1e-2))
            self.log_sigma_LN.copy_(torch.log(self.sigma_LN_prior))
            self.log_sigma_gauss.copy_(torch.log(self.sigma_gauss_prior))

    def forward(
        self,
        I_obs: torch.Tensor,
        total_scale: torch.Tensor,
        hkl_id: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute log-space NLL + truncated Gaussian NLL + floor penalty.

        Positive observations use log-space NLL (direct gradient to log_G).
        Non-positive observations use truncated Gaussian NLL (constrains
        I_true from above for weak reflections, correcting selection bias).

        Parameters
        ----------
        I_obs : (N_obs,) float32
            May include negative values from additive noise.
        total_scale : (N_obs,) float32
            Per-observation total scale factor (G · exp(-2Bs²)).
        hkl_id : (N_obs,) int64
        weight : (N_obs,) float32, optional
            Per-observation EM weights (should sum to N_obs).

        Returns
        -------
        loss : scalar tensor
        """
        n_shells = self.log_sigma_LN.shape[0]
        n_obs = I_obs.shape[0]

        # Expand per-shell sigma to per-reflection to avoid 8M→20
        # scatter_add contention in backward (instead: 8M→17k→20).
        sigma_LN_hkl = self.sigma_LN[self.shell_per_hkl]      # (n_unique,)
        sigma_gauss_hkl = self.sigma_gauss[self.shell_per_hkl] # (n_unique,)
        I_true = self.I_true
        # Detach I_true in variance to avoid unstable gradient coupling
        I_true_det = I_true.detach()

        pos = I_obs > 0
        loss_obs = torch.zeros_like(I_obs)

        # --- Positive observations: log-space NLL ---
        if pos.any():
            log_r = (torch.log(I_obs[pos])
                     - torch.log(total_scale[pos].clamp(min=1e-30)))
            log_I_true_obs = self.log_I_true[hkl_id[pos]]
            h = hkl_id[pos]

            # Log-space variance (detach I_true to stabilize)
            var_log = (sigma_LN_hkl[h] ** 2
                       + sigma_gauss_hkl[h] ** 2 / I_true_det[h].clamp(min=1.0) ** 2
                       ).clamp(min=1e-10)

            delta = log_r - log_I_true_obs
            loss_obs[pos] = 0.5 * (delta ** 2 / var_log + torch.log(var_log))

        # --- Non-positive observations: truncated Gaussian NLL ---
        if (~pos).any():
            h_neg = hkl_id[~pos]
            I_pred = I_true[h_neg] * total_scale[~pos]
            sg = sigma_gauss_hkl[h_neg]

            # Gaussian NLL
            residual = I_obs[~pos] - I_pred
            nll_gauss = 0.5 * residual ** 2 / sg.clamp(min=1e-10) ** 2 + torch.log(sg.clamp(min=1e-10))

            # Truncation correction: -log Φ(I_pred / σ_gauss)
            # Penalizes large I_pred when observation is negative
            log_phi = _log_ndtr(I_pred / sg.clamp(min=1e-10))
            loss_obs[~pos] = nll_gauss - log_phi

        # --- Floor penalty — anchors per-shell sigma to calibrated priors ---
        sigma_LN_shell = self.sigma_LN
        sigma_gauss_shell = self.sigma_gauss
        l_floor = (
            (sigma_LN_shell - self.sigma_LN_prior) ** 2 / (self.sigma_LN_prior ** 2)
            + (sigma_gauss_shell - self.sigma_gauss_prior) ** 2 / (self.sigma_gauss_prior ** 2)
        )

        # Combine
        if weight is not None:
            loss = (
                (loss_obs * weight).sum() / weight.sum().clamp(min=1.0)
                + self.lam * l_floor.mean()
            )
        else:
            loss = (
                loss_obs.sum() / max(n_obs, 1)
                + self.lam * l_floor.mean()
            )

        return loss

    @torch.no_grad()
    def compute_crystal_weights(
        self,
        I_obs: torch.Tensor,
        total_scale: torch.Tensor,
        hkl_id: torch.Tensor,
        crystal_id: torch.Tensor,
        w_min: float = 0.1,
        cutoff: float = 2.0,
        steepness: float = 2.0,
    ) -> torch.Tensor:
        """Compute EM-style per-crystal confidence weights.

        Uses per-crystal correlation with the merged model as the
        quality criterion. This catches crystals with systematically
        wrong intensities (misindexed, wrong cell) that hurt CC½,
        not just crystals with high random noise.

        Parameters
        ----------
        I_obs : (N_obs,) float32
        total_scale : (N_obs,) float32
        hkl_id : (N_obs,) int64
        crystal_id : (N_obs,) int32
        w_min : float
            Minimum weight (no crystal fully excluded).
        cutoff : float
            z-score below which crystals are downweighted (in robust
            z-score units; crystals with cc well below median).
        steepness : float
            Sigmoid steepness — higher = sharper transition.

        Returns
        -------
        weight : (N_obs,) float32, normalized so sum = N_obs.
        """
        n_obs = I_obs.shape[0]
        crys_long = crystal_id.long()
        n_crystals = crys_long.max().item() + 1

        pos = I_obs > 0
        # Per-obs log-scaled observation and model
        x = (torch.log(I_obs[pos].clamp(min=1e-30))
             - torch.log(total_scale[pos].clamp(min=1e-30)))
        y = self.log_I_true[hkl_id[pos]]

        # Per-crystal means of x and y (for Pearson correlation)
        ones_pos = torch.ones(pos.sum(), device=I_obs.device)
        cid_pos = crys_long[pos]
        c_count = torch.zeros(n_crystals, device=I_obs.device).scatter_add_(
            0, cid_pos, ones_pos,
        )
        c_n = c_count.clamp(min=2.0)
        sum_x = torch.zeros(n_crystals, device=I_obs.device).scatter_add_(0, cid_pos, x)
        sum_y = torch.zeros(n_crystals, device=I_obs.device).scatter_add_(0, cid_pos, y)
        mean_x = sum_x / c_n
        mean_y = sum_y / c_n

        # Per-obs deviations from crystal mean
        dx = x - mean_x[cid_pos]
        dy = y - mean_y[cid_pos]

        # Per-crystal correlation components
        sum_dxdy = torch.zeros(n_crystals, device=I_obs.device).scatter_add_(0, cid_pos, dx * dy)
        sum_dx2 = torch.zeros(n_crystals, device=I_obs.device).scatter_add_(0, cid_pos, dx * dx)
        sum_dy2 = torch.zeros(n_crystals, device=I_obs.device).scatter_add_(0, cid_pos, dy * dy)

        denom = (sum_dx2 * sum_dy2).clamp(min=1e-20).sqrt()
        cc_crystal = sum_dxdy / denom
        # Crystals with too few obs get cc=0 (neutral)
        cc_crystal[c_count < 5] = 0.0

        # Robust z-score on cc (low cc = bad crystal)
        med = cc_crystal.median()
        mad = (cc_crystal - med).abs().median() * 1.4826
        z = (cc_crystal - med) / mad.clamp(min=1e-10)

        # Sigmoid: z >> 0 (good cc) → w≈1, z << -cutoff (bad cc) → w≈w_min
        w_crystal = w_min + (1.0 - w_min) * torch.sigmoid(steepness * (z + cutoff))

        # Broadcast to per-obs and normalize
        w_obs = w_crystal[crys_long]
        w_obs = w_obs * (n_obs / w_obs.sum().clamp(min=1.0))
        return w_obs

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Return merged data as a dictionary."""
        with torch.no_grad():
            I = self.I_true
            F = I.clamp(min=0).sqrt()
            return {
                "I": I,
                "F": F,
                "sigma_LN": self._sigma_LN_per_hkl(),
                "sigma_gauss": self._sigma_gauss_per_hkl(),
            }
