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
import torch.nn.functional as F
from torch import Tensor

from torchsx.merging_clean.adjacency import Adjacency, dirichlet_energy
from torchsx.merging_clean.config import MergeConfig
from torchsx.merging_clean.corrections import Correction, ObsInfo
from torchsx.merging_clean.observations import PreparedObs
from torchsx.merging_clean.priors import FrenchWilsonPrior, WilsonMatchPrior
from torchsx.merging_clean.quadrature import QuadratureGrid, build_softplus_grid


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

        # Per-HKL background-hypothesis logit.  Only optimised when
        # bg_mode = "per_hkl".  Initialised at −2.2 so σ(π) = 0.1 (bright by
        # default); data can pull it up for genuinely dark HKLs.
        self.bg_mode = getattr(config, "bg_mode", "global")
        if self.bg_mode == "per_hkl":
            self.pi_logit = nn.Parameter(
                torch.full((prep.n_valid,), -2.2, dtype=log_I0_init.dtype)
            )
        else:
            self.register_buffer(
                "pi_logit", torch.zeros(prep.n_valid, dtype=log_I0_init.dtype)
            )

        # Anchor targets for Gaussian priors (prevent collective drift that
        # Laplacian smoothing cannot catch).
        self.register_buffer("log_alpha_init", log_alpha_init.clone())
        self.register_buffer("log_beta_init", log_beta_init.clone())

        self.corrections = nn.ModuleList(corrections)

        # ── Non-parameter tensors (buffers) ──
        # Observation arrays (for chunked indexing)
        self.register_buffer("hkl_id", prep.hkl_id)                    # (N,)
        # Quadrature inputs: store the minimal info to rebuild the grid
        # per-chunk rather than a full (N, K) tensor.  Avoids O(N·K)
        # persistent buffers that otherwise dominate GPU memory at scale.
        # `_n_ph` = I_obs + bg (used both as the bg-anchor's n_peak and to
        # build the grid); `_bg` is reused by the bg-anchor's log Poisson(n|bg).
        self.register_buffer("_n_ph", (prep.I_obs + prep.bg).clamp(min=0))  # (N,)
        self.register_buffer("_bg", prep.bg)                               # (N,)
        # Per-obs signal lower bound (rebuild same floor as observations.py)
        with torch.no_grad():
            if config.sig_lo_override is not None:
                sig_lo_per_obs = torch.full_like(prep.I_obs, float(config.sig_lo_override))
            else:
                bg_sum = torch.zeros(prep.n_valid).scatter_add_(
                    0, prep.hkl_id.cpu(), prep.bg.cpu().clamp(min=0))
                bg_per_hkl = (bg_sum / prep.obs_counts.float().cpu().clamp(min=1)).clamp(min=1.0)
                sigma_floor = (bg_per_hkl / prep.obs_counts.float().cpu().clamp(min=1)).sqrt()
                sig_lo_hkl = (sigma_floor * 0.5).clamp(min=0.01)
                sig_lo_per_obs = sig_lo_hkl[prep.hkl_id.cpu()]
        self.register_buffer("_sig_lo", sig_lo_per_obs)                    # (N,)
        # Cache scalar quadrature config
        self._n_quad = int(config.n_quad)
        self._n_sigma_quad = float(config.n_sigma_quad)
        # Adjacency
        self.register_buffer("edge_src", prep.adjacency.edge_src)
        self.register_buffer("edge_dst", prep.adjacency.edge_dst)
        # French–Wilson MAP log-prior on I₀ (shrinks each HKL toward I = Σ).
        self.fw_prior = FrenchWilsonPrior.from_prep(prep, config)
        # Wilson distribution-match prior (KL of empirical log(I/Σ) to Wilson
        # targets per shell × centricity group; preserves distribution shape).
        self.wm_prior = WilsonMatchPrior.from_prep(prep, config)
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
            return torch.zeros(e - s, device=self._n_ph.device, dtype=self._n_ph.dtype)
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

        # Build the softplus quadrature grid for THIS chunk only — avoids
        # persistent (N, K) buffers that otherwise dominate GPU memory.
        grid = build_softplus_grid(
            self._n_ph[s:e],
            self._bg[s:e],
            n_quad=self._n_quad,
            n_sigma=self._n_sigma_quad,
            signal_lo_override=self._sig_lo[s:e],
        )
        signal = grid.signal                    # (M, K)
        log_poi_plus_jac = grid.log_poi_plus_jac  # (M, K)
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

        if self.bg_mode == "off":
            log_p = torch.logsumexp(log_integrand, dim=1)
            return -log_p.sum()

        # Background hypothesis: log Poisson(n_peak | bg) with n_peak = I_obs
        # + bg (pre-computed and cached as `_n_ph`).
        n_peak_slice = self._n_ph[s:e]
        bg_slice = self._bg[s:e].clamp(min=1e-12)
        log_p_bg = (
            n_peak_slice * torch.log(bg_slice)
            - bg_slice
            - torch.lgamma(n_peak_slice + 1)
        )                                              # (M,)

        log_p_sig_marginal = torch.logsumexp(log_integrand, dim=1)  # (M,)

        if self.bg_mode == "global":
            # Fixed 50/50 mixture (implicit weight 1 on each via logsumexp).
            log_p = torch.logaddexp(log_p_sig_marginal, log_p_bg)
            return -log_p.sum()

        # bg_mode == "per_hkl" — learned per-HKL Bernoulli weight.
        pi_logit_slice = self.pi_logit[h]              # (M,)
        # Numerically stable log σ and log(1−σ) via softplus.
        log_pi = F.logsigmoid(pi_logit_slice)          # log π_h
        log_1mpi = F.logsigmoid(-pi_logit_slice)       # log(1 − π_h)
        log_p = torch.logaddexp(
            log_1mpi + log_p_sig_marginal,
            log_pi + log_p_bg,
        )
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
        # French–Wilson MAP log-prior on I₀ (per-HKL mode-pulling).
        if cfg.lam_frac_wilson > 0:
            reg = reg + s * cfg.lam_frac_wilson * self.fw_prior(self.log_I0).sum()
        # Wilson distribution-match prior (per-shell KL on the empirical
        # spread of log(I/Σ)).  Complementary to the MAP prior: preserves
        # distribution *shape* rather than concentrating toward the mode.
        if cfg.lam_wilson_match > 0:
            reg = reg + s * cfg.lam_wilson_match * self.wm_prior(self.log_I0)
        # Half-Cauchy prior on GenGauss width σ = α · c(β).
        # Width-invariant under α↔β trades.
        if cfg.lam_sigma_cauchy > 0:
            beta = torch.exp(self.log_beta)
            log_c_beta = 0.5 * (torch.lgamma(3.0 / beta) - torch.lgamma(1.0 / beta))
            log_sigma = self.log_alpha + log_c_beta
            sigma2 = torch.exp(2.0 * log_sigma)
            tau2 = float(cfg.sigma_cauchy_scale) ** 2
            reg = reg + s * cfg.lam_sigma_cauchy * torch.log(tau2 + sigma2).sum()
        # Beta(a, b) prior on learned π_h.  Only active in per-HKL bg mode.
        #   -log p = -(a-1)·log π - (b-1)·log(1-π)  [ignoring Beta-function
        #   constant; gradient wrt π_logit is well-defined via softplus].
        if self.bg_mode == "per_hkl" and cfg.lam_pi_beta > 0:
            log_pi = F.logsigmoid(self.pi_logit)
            log_1mpi = F.logsigmoid(-self.pi_logit)
            a_m1 = float(cfg.pi_beta_a) - 1.0
            b_m1 = float(cfg.pi_beta_b) - 1.0
            # Subtract because this is a *negative* log-likelihood (add to loss)
            reg = reg + cfg.lam_pi_beta * (-a_m1 * log_pi - b_m1 * log_1mpi).sum()
        # Correction-specific regularizers (ridge on log_G, B, panel gain, ...)
        for c in self.corrections:
            reg = reg + c.regularizer()
        return reg

    def total_loss(self) -> Tensor:
        """Unchunked forward + all regularisers (for debugging / tests)."""
        data = self.data_nll(0, self.n_obs_total)
        return data + self.regularizers_only()
