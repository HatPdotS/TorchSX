"""SelfConsistentTarget: heteroscedastic Gaussian NLL with predicted variance."""

from __future__ import annotations

import math

import torch
from torch import nn


class SelfConsistentTarget(nn.Module):
    """Heteroscedastic Gaussian NLL with predicted variance.

    sigma^2 = I_pred + 2*I_bg + I_pred^2 * sigma_LN(s)^2

    Only refined parameter: I_true (structure factor intensities, unconstrained).
    sigma_LN per shell is calibrated from data and fixed (buffer).
    I_bg per observation comes from integration (passed in forward).

    I_true is parametrized in log-space (log_I_true) to ensure positivity.
    Weak reflections are initialized with a low clamp (0.01) so they can
    reach near-zero values.

    Parameters
    ----------
    n_unique : int
        Number of unique reflections.
    n_shells : int
        Number of resolution shells for noise estimation.
    """

    def __init__(self, n_unique: int, n_shells: int):
        super().__init__()
        self.log_I_true = nn.Parameter(torch.zeros(n_unique))
        self.register_buffer("sigma_LN_prior", torch.full((n_shells,), 0.1))
        self.register_buffer("shell_per_hkl", torch.zeros(n_unique, dtype=torch.long))

    @property
    def I_true(self) -> torch.Tensor:
        return torch.exp(self.log_I_true)

    def calibrate(
        self,
        I_obs: torch.Tensor,
        total_scale: torch.Tensor,
        hkl_id: torch.Tensor,
        shell: torch.Tensor,
        crystal_id: torch.Tensor | None = None,
    ) -> None:
        """Initialize parameters from data.

        Same logic as KLScalingTarget.calibrate(): initializes log_I_true
        from per-reflection means, computes shell_per_hkl, and estimates
        sigma_LN_prior per shell. sigma_LN is stored as a buffer (never
        optimized).
        """
        n_unique = self.log_I_true.shape[0]
        n_shells = self.sigma_LN_prior.shape[0]
        device = self.log_I_true.device

        r = I_obs / total_scale.clamp(min=1e-30)
        pos = I_obs > 0
        r_pos = torch.where(pos, r, torch.zeros_like(r))
        w_pos = pos.float()

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
            crys_long = crystal_id.long()
            c_count = torch.zeros(n_crystals, device=device).scatter_add_(0, crys_long, w_pos)
            c_sum = torch.zeros(n_crystals, device=device).scatter_add_(0, crys_long, ratio * w_pos)
            g_est = c_sum / c_count.clamp(min=1)
            r_corrected = r_pos / g_est[crys_long].clamp(min=1e-10)
            r_corrected = torch.where(pos, r_corrected, torch.zeros_like(r))
        else:
            r_corrected = r_pos

        sum_rc = torch.zeros(n_unique, device=device).scatter_add_(0, hkl_id, r_corrected)
        sum_rc2 = torch.zeros(n_unique, device=device).scatter_add_(0, hkl_id, r_corrected * r_corrected)

        valid = count_pos >= 2
        n = count_pos.clamp(min=2)
        var_emp = ((sum_rc2 - sum_rc * sum_rc / n) / (n - 1).clamp(min=1)).clamp(min=1e-10)

        # Initialize I_true from per-reflection mean
        I_init = (sum_r / n_clamp).clamp(min=0.01)
        with torch.no_grad():
            self.log_I_true.copy_(torch.log(I_init))

        # Assign shell per hkl
        shell_count = torch.zeros(n_unique, n_shells, device=device)
        shell_count.scatter_add_(
            0,
            hkl_id.unsqueeze(1).expand(-1, n_shells),
            torch.nn.functional.one_hot(shell.long(), n_shells).float(),
        )
        shell_per_hkl = shell_count.argmax(dim=1)
        with torch.no_grad():
            self.shell_per_hkl.copy_(shell_per_hkl)

        # Per-shell sigma_LN estimation
        sigma_LN_shell = torch.zeros(n_shells, device=device)
        for s in range(n_shells):
            mask = (shell_per_hkl == s) & valid
            if mask.sum() < 2:
                sigma_LN_shell[s] = 0.1
                continue
            I_shell = I_init[mask]
            var_shell = var_emp[mask]

            # Gaussian floor ~ mean intensity (Poisson-like)
            sg2 = I_shell.mean().clamp(min=1.0)

            # Lognormal component: var_reducible = var_emp - sigma_gauss^2
            var_LN = ((var_shell - sg2) / (I_shell * I_shell).clamp(min=1.0)).clamp(min=1e-8)
            sigma_LN_shell[s] = var_LN.mean().sqrt()

        with torch.no_grad():
            self.sigma_LN_prior.copy_(sigma_LN_shell.clamp(min=1e-4))

    @torch.no_grad()
    def update_sigma(
        self,
        I_obs: torch.Tensor,
        total_scale: torch.Tensor,
        hkl_id: torch.Tensor,
    ) -> None:
        """Re-estimate sigma_LN per shell from current scale factors and I_true.

        Works in observation space: scale I_true up to predict I_obs, never
        divide I_obs by scale. This avoids noise amplification for
        low-partiality observations.
        """
        n_unique = self.log_I_true.shape[0]
        n_shells = self.sigma_LN_prior.shape[0]
        device = self.log_I_true.device
        shell_per_hkl = self.shell_per_hkl
        I_true = self.I_true

        # Predict in observation space — scale I_true, never divide I_obs
        I_pred = I_true[hkl_id] * total_scale
        pos = I_pred > 1.0
        w_pos = pos.float()

        # Relative residual: (I_obs - I_pred) / I_pred
        rel_resid = torch.where(
            pos,
            (I_obs - I_pred) / I_pred.clamp(min=1.0),
            torch.zeros_like(I_obs),
        )

        count_pos = torch.zeros(n_unique, device=device).scatter_add_(0, hkl_id, w_pos)
        sum_rr = torch.zeros(n_unique, device=device).scatter_add_(0, hkl_id, rel_resid)
        sum_rr2 = torch.zeros(n_unique, device=device).scatter_add_(0, hkl_id, rel_resid ** 2)

        valid = count_pos >= 2
        n = count_pos.clamp(min=2)
        var_rel = ((sum_rr2 - sum_rr ** 2 / n) / (n - 1).clamp(min=1)).clamp(min=1e-10)

        sigma_LN_shell = torch.zeros(n_shells, device=device)
        for s in range(n_shells):
            mask = (shell_per_hkl == s) & valid
            if mask.sum() < 2:
                sigma_LN_shell[s] = self.sigma_LN_prior[s]
                continue
            I_s = I_true[mask]
            var_s = var_rel[mask]

            # Poisson relative variance ~ 1/I_true
            poisson_rel = 1.0 / I_s.clamp(min=1.0)

            # Lognormal: var_rel - Poisson component
            var_LN = (var_s - poisson_rel).clamp(min=1e-8)
            sigma_LN_shell[s] = var_LN.median().sqrt()

        self.sigma_LN_prior.copy_(sigma_LN_shell.clamp(min=1e-4))

    def forward(
        self,
        I_obs: torch.Tensor,
        total_scale: torch.Tensor,
        hkl_id: torch.Tensor,
        I_bg: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute heteroscedastic Gaussian NLL.

        Parameters
        ----------
        I_obs : (N_obs,) float32
        total_scale : (N_obs,) float32
        hkl_id : (N_obs,) int64
        I_bg : (N_obs,) float32
            Background intensity per observation.
        weight : (N_obs,) float32, optional

        Returns
        -------
        loss : scalar tensor
        """
        I_pred = self.I_true[hkl_id] * total_scale
        sigma_LN_obs = self.sigma_LN_prior[self.shell_per_hkl[hkl_id]]

        # Predicted variance — detach I_pred to avoid unstable gradient coupling
        I_pred_det = I_pred.detach()
        sigma_sq = (
            I_pred_det.clamp(min=0)
            + 2.0 * I_bg.clamp(min=0)
            + I_pred_det ** 2 * sigma_LN_obs ** 2
        ).clamp(min=1.0)

        nll = (I_obs - I_pred) ** 2 / sigma_sq + torch.log(sigma_sq)

        if weight is not None:
            return (nll * weight).sum() / weight.sum().clamp(min=1.0)
        return nll.mean()

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
        quality criterion.

        Returns
        -------
        weight : (N_obs,) float32, normalized so sum = N_obs.
        """
        n_obs = I_obs.shape[0]
        crys_long = crystal_id.long()
        n_crystals = crys_long.max().item() + 1

        pos = I_obs > 0
        x = (torch.log(I_obs[pos].clamp(min=1e-30))
             - torch.log(total_scale[pos].clamp(min=1e-30)))
        y = self.log_I_true[hkl_id[pos]]

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

        dx = x - mean_x[cid_pos]
        dy = y - mean_y[cid_pos]

        sum_dxdy = torch.zeros(n_crystals, device=I_obs.device).scatter_add_(0, cid_pos, dx * dy)
        sum_dx2 = torch.zeros(n_crystals, device=I_obs.device).scatter_add_(0, cid_pos, dx * dx)
        sum_dy2 = torch.zeros(n_crystals, device=I_obs.device).scatter_add_(0, cid_pos, dy * dy)

        denom = (sum_dx2 * sum_dy2).clamp(min=1e-20).sqrt()
        cc_crystal = sum_dxdy / denom
        cc_crystal[c_count < 5] = 0.0

        med = cc_crystal.median()
        mad = (cc_crystal - med).abs().median() * 1.4826
        z = (cc_crystal - med) / mad.clamp(min=1e-10)

        w_crystal = w_min + (1.0 - w_min) * torch.sigmoid(steepness * (z + cutoff))

        w_obs = w_crystal[crys_long]
        w_obs = w_obs * (n_obs / w_obs.sum().clamp(min=1.0))
        return w_obs

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Return merged data as a dictionary."""
        with torch.no_grad():
            I = self.I_true
            F = I.clamp(min=0).sqrt()
            sigma_LN = self.sigma_LN_prior[self.shell_per_hkl]
            return {"I": I, "F": F, "sigma_LN": sigma_LN}
