"""UnifiedMergeLoss — single differentiable objective for the whole merge.

Parameters
----------
log_I0, log_alpha, log_beta   (n_valid,)
corrections                   nn.ModuleList of Correction submodules

Data term
---------
Poisson-marginal NLL for ALL observations (positive and negative):

    n_peak_j ~ Poisson(I0_h * scale_j * exp(ε) + bg_j)
    ε ~ GenGauss(0, α_h, β_h)

    log p(I_obs_j)  =  logsumexp_k [ log_gg(ε_jk; α_h, β_h)
                                     + log_poi_plus_jac_jk ]

    ε_jk = log(signal_jk / (I0_h · scale_j))
    scale_j = exp( Σ_c corrections[c](obs_info)_j ) · partiality_j

Regularisers
------------
- Dirichlet energy on log_α, log_β, (optionally) log_I0 over the ±1 Miller graph.
- Each correction's own regulariser() call (ridge on log_G, B, panel gain, ...).

Chunking
--------
`data_nll(start, end)` computes the NLL summed over observations in
[start, end).  Regularisers are parameter-only and computed by
`regularizers_only()` outside the chunk loop.
"""

from __future__ import annotations

import math
from typing import Iterator

import torch
import torch.nn as nn
from torch import Tensor

from torchsx.merging_clean.adjacency import Adjacency, dirichlet_energy
from torchsx.merging_clean.config import MergeConfig
from torchsx.merging_clean.corrections import Correction, ObsInfo
from torchsx.merging_clean.observations import PreparedObs
from torchsx.merging_clean.quadrature import QuadratureGrid


def _log_gg_norm(beta: Tensor) -> Tensor:
    """GenGauss log normalisation: log β − log 2 − lgamma(1/β)."""
    return torch.log(beta) - math.log(2.0) - torch.lgamma(1.0 / beta)


class UnifiedMergeLoss(nn.Module):
    def __init__(
        self,
        prep: PreparedObs,
        corrections: list[Correction],
        config: MergeConfig,
    ):
        super().__init__()
        self.config = config
        self.n_valid = prep.n_valid
        self.n_obs_total = prep.n_obs_total

        # ── Parameters ──
        log_I0_init = torch.log(torch.exp(prep.mu_init).clamp(min=1e-3))
        log_alpha_init = torch.log(prep.alpha_init.clamp(min=1e-3))
        log_beta_init = torch.log(prep.beta_perhkl.clamp(min=0.5))

        self.log_I0 = nn.Parameter(log_I0_init.clone())
        self.log_alpha = nn.Parameter(log_alpha_init.clone())
        self.log_beta = nn.Parameter(log_beta_init.clone())

        # Anchor targets for Gaussian priors (prevent collective drift that
        # Laplacian smoothing cannot catch).
        self.register_buffer("log_alpha_init", log_alpha_init.clone())
        self.register_buffer("log_beta_init", log_beta_init.clone())

        self.corrections = nn.ModuleList(corrections)

        # ── Non-parameter tensors (buffers) ──
        # Observation arrays (for chunked indexing)
        self.register_buffer("hkl_id", prep.hkl_id)                    # (N,)
        self.register_buffer("I_obs", prep.I_obs)                      # (N,)  for bg-anchor
        self.register_buffer("bg", prep.bg)                            # (N,)  for bg-anchor
        # Quadrature grid
        self.register_buffer("signal", prep.grid.signal)               # (N, K)
        self.register_buffer("log_poi_plus_jac", prep.grid.log_poi_plus_jac)  # (N, K)
        # Adjacency
        self.register_buffer("edge_src", prep.adjacency.edge_src)
        self.register_buffer("edge_dst", prep.adjacency.edge_dst)
        # Partiality (optional)
        if prep.partiality is not None:
            self.register_buffer("partiality", prep.partiality)
            self.has_partiality = True
        else:
            self.register_buffer("partiality", torch.empty(0))
            self.has_partiality = False

        # ObsInfo tensors as buffers so .to(device) / state_dict carry them along.
        oi = prep.obs_info
        self.register_buffer("_oi_crystal_id", oi.crystal_id)
        self.register_buffer("_oi_s_sq", oi.s_sq)
        self.register_buffer("_oi_hkl_idx", oi.hkl_idx)
        self._oi_n_crystals = oi.n_crystals
        self._oi_has_panel = oi.panel_id is not None
        if oi.panel_id is not None:
            self.register_buffer("_oi_panel_id", oi.panel_id)
        if oi.fs is not None:
            self.register_buffer("_oi_fs", oi.fs)
        if oi.ss is not None:
            self.register_buffer("_oi_ss", oi.ss)
        if oi.peak is not None:
            self.register_buffer("_oi_peak", oi.peak)
        if oi.background is not None:
            self.register_buffer("_oi_background", oi.background)

    def _slice_opt(self, name: str, s: int, e: int) -> Tensor | None:
        t = getattr(self, name, None)
        return None if t is None else t[s:e]

    def _obs_info_slice(self, s: int, e: int) -> ObsInfo:
        return ObsInfo(
            crystal_id=self._oi_crystal_id[s:e],
            s_sq=self._oi_s_sq[s:e],
            hkl_idx=self._oi_hkl_idx[s:e],
            n_crystals=self._oi_n_crystals,
            panel_id=self._slice_opt("_oi_panel_id", s, e),
            fs=self._slice_opt("_oi_fs", s, e),
            ss=self._slice_opt("_oi_ss", s, e),
            peak=self._slice_opt("_oi_peak", s, e),
            background=self._slice_opt("_oi_background", s, e),
        )

    # ── chunk iterator ──
    def obs_chunks(self, chunk_size: int) -> Iterator[tuple[int, int]]:
        n = self.n_obs_total
        if chunk_size is None or chunk_size <= 0 or chunk_size >= n:
            yield (0, n)
            return
        for s in range(0, n, chunk_size):
            yield (s, min(s + chunk_size, n))

    # ── total correction on a chunk ──
    def _total_correction(self, obs_info: ObsInfo, obs_range: tuple[int, int]) -> Tensor:
        s, e = obs_range
        if len(self.corrections) == 0:
            return torch.zeros(e - s, device=self.signal.device, dtype=self.signal.dtype)
        total = None
        for c in self.corrections:
            part = c(obs_info, obs_range=obs_range)
            total = part if total is None else total + part
        return total

    # ── data NLL on observations [s, e) ──
    def data_nll(self, s: int, e: int) -> Tensor:
        """Poisson-marginal NLL summed over obs[s:e]."""
        oi = self._obs_info_slice(s, e)
        # log(scale) per obs from corrections + partiality
        log_scale = self._total_correction(oi, (s, e))
        if self.has_partiality:
            log_scale = log_scale + torch.log(self.partiality[s:e].clamp(min=1e-12))

        h = self.hkl_id[s:e]
        log_I0 = self.log_I0[h]                 # (M,)
        log_alpha = self.log_alpha[h]           # (M,)
        log_beta = self.log_beta[h]             # (M,)
        beta = torch.exp(log_beta)
        alpha = torch.exp(log_alpha)

        # ε_jk = log(signal_jk) − log(I0) − log(scale)
        signal = self.signal[s:e]               # (M, K)
        log_poi_plus_jac = self.log_poi_plus_jac[s:e]  # (M, K)
        log_signal = torch.log(signal.clamp(min=1e-12))
        eps = log_signal - (log_I0[:, None] + log_scale[:, None])

        # Full GenGauss log-pdf (we optimize α, so the −log α term MUST be present;
        # fit_I0_poisson omits it because α is fixed there).
        z = eps.abs() / alpha[:, None].clamp(min=1e-8)
        log_gg = (
            -z.clamp(min=1e-12).pow(beta[:, None])
            + _log_gg_norm(beta)[:, None]
            - log_alpha[:, None]
        )

        log_integrand = log_gg + log_poi_plus_jac      # (M, K)

        # Background anchor: "this observation is pure bg" — appended as a
        # (K+1)th column.  log Poisson(n_peak | bg) with n_peak = I_obs + bg.
        # For observations that look indistinguishable from Poisson(bg), this
        # anchor dominates the logsumexp and signal-side gradients shrink
        # proportionally to P(H_sig | obs).
        n_peak_slice = (self.I_obs[s:e] + self.bg[s:e]).clamp(min=0)
        bg_slice = self.bg[s:e].clamp(min=1e-12)
        log_p_bg = (
            n_peak_slice * torch.log(bg_slice)
            - bg_slice
            - torch.lgamma(n_peak_slice + 1)
        )
        log_integrand_ext = torch.cat(
            [log_integrand, log_p_bg.unsqueeze(1)], dim=1
        )                                              # (M, K+1)

        log_p = torch.logsumexp(log_integrand_ext, dim=1)  # (M,)
        return -log_p.sum()

    def regularizers_only(self) -> Tensor:
        adj = Adjacency(self.edge_src, self.edge_dst,
                        torch.empty(0, device=self.edge_src.device))
        reg = torch.zeros((), device=self.log_I0.device, dtype=self.log_I0.dtype)
        cfg = self.config
        # Data-size scaling: every λ is multiplied by N_obs/reg_scale_ref_obs
        # so the regulariser:data ratio is invariant to dataset size.
        s = float(self.n_obs_total) / float(cfg.reg_scale_ref_obs)
        # Dirichlet smoothing over ±1 Miller graph
        if cfg.lam_alpha_smooth > 0:
            reg = reg + s * cfg.lam_alpha_smooth * dirichlet_energy(self.log_alpha, adj)
        if cfg.lam_beta_smooth > 0:
            reg = reg + s * cfg.lam_beta_smooth * dirichlet_energy(self.log_beta, adj)
        if cfg.lam_I0_smooth > 0:
            reg = reg + s * cfg.lam_I0_smooth * dirichlet_energy(self.log_I0, adj)
        # Gaussian anchors toward shell-fit init
        if cfg.lam_alpha_anchor > 0:
            da = self.log_alpha - self.log_alpha_init
            reg = reg + s * cfg.lam_alpha_anchor * (da * da).sum()
        if cfg.lam_beta_anchor > 0:
            db = self.log_beta - self.log_beta_init
            reg = reg + s * cfg.lam_beta_anchor * (db * db).sum()
        # Weak "zero-intensity floor" prior on log I₀:
        # pull HKLs with no evidence toward log I_target; data wins for
        # well-measured HKLs because F_data >> lam_I0_floor for those.
        if cfg.lam_I0_floor > 0:
            di = self.log_I0 - float(cfg.log_I0_floor_target)
            reg = reg + s * cfg.lam_I0_floor * (di * di).sum()
        # Half-Cauchy prior on GenGauss width σ = α · c(β).
        # Width-invariant under α↔β trades.
        if cfg.lam_sigma_cauchy > 0:
            beta = torch.exp(self.log_beta)
            log_c_beta = 0.5 * (torch.lgamma(3.0 / beta) - torch.lgamma(1.0 / beta))
            log_sigma = self.log_alpha + log_c_beta
            sigma2 = torch.exp(2.0 * log_sigma)
            tau2 = float(cfg.sigma_cauchy_scale) ** 2
            reg = reg + s * cfg.lam_sigma_cauchy * torch.log(tau2 + sigma2).sum()
        # Correction-specific regularizers (ridge on log_G, B, panel gain, ...)
        for c in self.corrections:
            reg = reg + c.regularizer()
        return reg

    def total_loss(self) -> Tensor:
        """Unchunked forward + all regularisers (for debugging / tests)."""
        data = self.data_nll(0, self.n_obs_total)
        return data + self.regularizers_only()
