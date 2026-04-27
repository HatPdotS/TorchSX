"""Per-HKL generalized Gaussian distribution fitting with adjacency smoothing.

Model: log(I_obs) for a given HKL follows GenGauss(β, μ, α) in log-space.
β ≈ 1.2–1.5 (sub-Gaussian, between Laplace and Gaussian) is typical for SFX data.

Fitting procedure:
  Stage 1 — Per-shell β estimation via GenGauss+Gauss mixture (L-BFGS)
  Stage 2 — Per-HKL closed-form MLE for μ and α given β
  Stage 3 — Laplacian smoothing of α and β over ±1 Miller-index adjacency graph
  Stage 4 — EM refinement with modular corrections (scale, B-factor, ...)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ._corrections import (
    Correction,
    CorrectionPipeline,
    ObsInfo,
    ScaleBCorrection,
)
from ._noise import PoissonNoiseModel, linear_space_I0


# ── Dataclasses ──────────────────────────────────────────────


@dataclass
class HKLDistribution:
    """Per-HKL generalized Gaussian distribution parameters."""

    hkl: torch.Tensor  # (N, 3) int16
    I0: torch.Tensor  # (N,) mode intensity = exp(mu)
    alpha: torch.Tensor  # (N,) width parameter
    beta: torch.Tensor  # (N,) shape exponent
    mu: torch.Tensor  # (N,) log-space location
    d_spacing: torch.Tensor  # (N,) Å
    n_obs: torch.Tensor  # (N,) long
    sigma_mu: torch.Tensor  # (N,) std error of mu (from IRLS weights)
    sigma_I0: torch.Tensor  # (N,) std error of I0 = I0 * sigma_mu


@dataclass
class DistShellStats:
    """Per-resolution-shell comparison of distribution parameters."""

    d_min: float
    d_max: float
    cc_I0: float
    cc_alpha: float
    n_matched: int


@dataclass
class DistributionComparison:
    """Split-half comparison of per-HKL distribution parameters."""

    hkl: torch.Tensor  # (M, 3) matched HKLs
    d_spacing: torch.Tensor  # (M,)
    I0_1: torch.Tensor
    I0_2: torch.Tensor
    alpha_1: torch.Tensor
    alpha_2: torch.Tensor
    beta_1: torch.Tensor
    beta_2: torch.Tensor
    cc_I0: float
    cc_alpha: float
    shells: list[DistShellStats]


# ── Internal: L-BFGS shell fitting ──────────────────────────


def _fit_shell_lbfgs(data: torch.Tensor) -> tuple[float, ...] | None:
    """Fit GenGauss + Gauss mixture to log(I/⟨I⟩) via L-BFGS.

    Parameters
    ----------
    data : 1D float64 tensor of log(I/mean_I) values for one shell.

    Returns
    -------
    (beta, mu1, alpha, w1, mu2, sigma2) or None on failure.
    """
    y = data.double()
    mu_all = float(y.mean())
    sig_all = float(y.std())

    # Unconstrained parameters (reparameterized for positivity / bounds)
    beta_unc = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    mu1 = torch.tensor(mu_all - 0.3, dtype=torch.float64, requires_grad=True)
    log_alpha = torch.tensor(
        math.log(max(sig_all * 0.8, 0.05)), dtype=torch.float64, requires_grad=True
    )
    w_logit = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    mu2 = torch.tensor(mu_all + 1.0, dtype=torch.float64, requires_grad=True)
    log_sigma2 = torch.tensor(
        math.log(max(sig_all * 0.7, 0.05)), dtype=torch.float64, requires_grad=True
    )

    params = [beta_unc, mu1, log_alpha, w_logit, mu2, log_sigma2]
    optimizer = torch.optim.LBFGS(
        params,
        lr=1.0,
        max_iter=100,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
    )

    def closure():
        optimizer.zero_grad()
        beta = F.softplus(beta_unc) + 0.5
        alpha = torch.exp(log_alpha)
        sigma2 = torch.exp(log_sigma2)
        w1 = torch.sigmoid(w_logit)

        # GenGauss log-pdf
        z1 = torch.abs(y - mu1) / alpha.clamp(min=1e-8)
        log_pdf1 = (
            torch.log(beta)
            - torch.log(2.0 * alpha)
            - torch.lgamma(1.0 / beta)
            - z1.clamp(min=1e-8).pow(beta)
        )

        # Gaussian log-pdf
        log_pdf2 = (
            -0.5 * math.log(2 * math.pi)
            - log_sigma2
            - 0.5 * ((y - mu2) / sigma2.clamp(min=1e-8)) ** 2
        )

        # Mixture NLL via log-sum-exp
        log_mix = torch.logaddexp(
            torch.log(w1.clamp(min=1e-8)) + log_pdf1,
            torch.log((1 - w1).clamp(min=1e-8)) + log_pdf2,
        )
        loss = -log_mix.mean()
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except Exception:
        return None

    with torch.no_grad():
        beta_val = (F.softplus(beta_unc) + 0.5).item()
        mu1_val, mu2_val = mu1.item(), mu2.item()
        alpha_val = torch.exp(log_alpha).item()
        sigma2_val = torch.exp(log_sigma2).item()
        w1_val = torch.sigmoid(w_logit).item()

    # Ensure μ₁ < μ₂
    if mu1_val > mu2_val:
        beta_val, mu1_val, alpha_val, w1_val, mu2_val, sigma2_val = (
            beta_val,
            mu2_val,
            sigma2_val,
            1 - w1_val,
            mu1_val,
            alpha_val,
        )

    return (beta_val, mu1_val, alpha_val, w1_val, mu2_val, sigma2_val)


# ── Internal: adjacency graph + smoothing ────────────────────


def _build_adjacency(
    hkl: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ±1 Miller-index adjacency graph.

    Returns (edge_src, edge_dst, n_neighbors) tensors.
    """
    n = hkl.shape[0]
    lookup: dict[tuple[int, int, int], int] = {}
    for i in range(n):
        key = (hkl[i, 0].item(), hkl[i, 1].item(), hkl[i, 2].item())
        lookup[key] = i

    edge_src: list[int] = []
    edge_dst: list[int] = []
    deltas = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for i in range(n):
        h, k, l = hkl[i, 0].item(), hkl[i, 1].item(), hkl[i, 2].item()
        for dh, dk, dl in deltas:
            j = lookup.get((h + dh, k + dk, l + dl))
            if j is not None:
                edge_src.append(j)
                edge_dst.append(i)

    edge_src_t = torch.tensor(edge_src, dtype=torch.long)
    edge_dst_t = torch.tensor(edge_dst, dtype=torch.long)
    n_neighbors = torch.bincount(edge_dst_t, minlength=n).float().clamp(min=1)
    return edge_src_t, edge_dst_t, n_neighbors


def _laplacian_smooth(
    values: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    n_neighbors: torch.Tensor,
    n_iter: int,
    lam: float,
    *,
    node_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Laplacian smoothing over a Miller-index adjacency graph.

    Without *node_weights* (default): uniform smoothing
        x_new = (1-λ)x + λ·mean(neighbors)

    With *node_weights* w (e.g. observation counts): multiplicity-weighted
        x_new = (w·x + λ_eff·mean(neighbors)) / (w + λ_eff)
    where λ_eff = λ/(1-λ) maps λ∈(0,1) to (0,∞).  At w=1 this reduces
    to the uniform formula, so the two modes are consistent when
    node_weights = ones.  Nodes with w >> 1 resist smoothing; w << 1
    are pulled toward their neighbors.
    """
    x = values.clone()
    has_nb = n_neighbors > 0.5

    if node_weights is None:
        # ── Uniform smoothing (original) ──
        for _ in range(n_iter):
            nb_sum = torch.zeros_like(x)
            nb_sum.scatter_add_(0, edge_dst, x[edge_src])
            nb_mean = nb_sum / n_neighbors
            x[has_nb] = (1 - lam) * x[has_nb] + lam * nb_mean[has_nb]
    else:
        # ── Multiplicity-weighted smoothing ──
        # Normalize weights so median = 1 (λ retains its usual meaning)
        w = node_weights.float() / node_weights.float().median().clamp(min=1.0)
        lam_eff = lam / max(1 - lam, 1e-6)
        for _ in range(n_iter):
            nb_sum = torch.zeros_like(x)
            nb_sum.scatter_add_(0, edge_dst, x[edge_src])
            nb_mean = nb_sum / n_neighbors
            denom = w[has_nb] + lam_eff
            x[has_nb] = (w[has_nb] * x[has_nb] + lam_eff * nb_mean[has_nb]) / denom
    return x


# ── Public API ───────────────────────────────────────────────


def fit_gengauss(
    stream,
    *,
    min_obs: int = 20,
    n_shells: int = 20,
    n_smooth_iter: int = 15,
    smooth_lambda: float = 0.3,
    snr_thresh: float = 3.0,
    n_em_cycles: int = 0,
    corrections: Sequence[Correction] | None = None,
    free_fraction: float = 0.1,
    # Legacy convenience parameters (create ScaleBCorrection internally)
    scale_crystals: bool = False,
    scale_prior: float = 10.0,
    bfactor_prior: float = 1.0,
    weighted_smooth: bool = False,
    crystal_quality_weights: bool = False,
    crystal_quality_cutoff: float = 0.0,
    crystal_posterior: bool = False,
    poisson_noise: PoissonNoiseModel | None = None,
    mode: str = "log",
    joint_device: str = "cpu",
) -> HKLDistribution:
    """Fit per-HKL generalized Gaussian distribution in log-space with IRLS.

    .. note::

        This function works in log-space and uses only **positive**
        observations.  It does not model Poisson noise or handle negative
        reflections.  For a proper Poisson marginal likelihood approach that
        uses all observations (including negatives), use :func:`fit_em`
        instead.

    Parameters
    ----------
    stream : SFXStream
        Input stream with observations.
    min_obs : minimum positive observations per HKL to include.
    n_shells : number of resolution shells for Stage 1 β estimation.
    n_smooth_iter : Laplacian smoothing iterations.
    smooth_lambda : smoothing mixing weight (0=no smooth, 1=full replace).
    snr_thresh : SNR threshold for pooling observations in shell fit.
    n_em_cycles : number of EM refinement cycles (0 = single-pass, no EM).
    corrections : list of Correction instances to apply in the EM loop.
        If None and scale_crystals=True, a ScaleBCorrection is created
        automatically for backward compatibility.
    free_fraction : fraction of observations held out for overfitting
        monitoring (only used when corrections are active).
    scale_crystals : legacy shortcut — if True and corrections is None,
        creates a ScaleBCorrection with the given priors.
    scale_prior : L2 regularization weight on per-crystal log(G).
        Added to the (1,1) element of X'X in the ridge regression.
        Higher values pull log(G) toward 0 (scale=1).  Default 10.
    bfactor_prior : L2 regularization weight on per-crystal B-factor slope.
        Added to the (2,2) element of X'X.  Higher values pull B toward 0.
        Default 1.0.
    weighted_smooth : if True, weight Laplacian smoothing by per-HKL
        observation count.  Well-measured HKLs resist smoothing; sparse
        ones are pulled toward neighbors.  Default False.
    crystal_quality_weights : if True, continuously weight observations in
        the IRLS E-step by per-crystal aggregated p-value.  Noisy crystals
        contribute less to μ estimation.  Requires scale_crystals=True.
    crystal_quality_cutoff : fraction of crystals to reject (0–1).
        Crystals in the bottom quantile by aggregated p-value are excluded.
        E.g., 0.1 rejects the worst 10%.  Requires scale_crystals=True.
        Default 0.0 (no rejection).
    crystal_posterior : if True, use a two-component mixture model
        (good=GenGauss, bad=uniform) and weight observations by the
        posterior P(good|crystal).  This is the Bayesian version of
        crystal quality filtering.  Requires scale_crystals=True.

    Returns
    -------
    HKLDistribution with per-HKL parameters.
    """
    if mode == "joint" or poisson_noise is not None:
        raise ValueError(
            "mode='joint' and poisson_noise are no longer supported in "
            "fit_gengauss. Use fit_em() for Poisson marginal likelihood."
        )

    from torchref.base.reciprocal.hkl import get_d_spacing

    from torchsx.deprecated.merge._hkl import compute_hkl_index

    core = stream._core
    hkl_idx = compute_hkl_index(core.hkl)
    hkl_id = hkl_idx.hkl_id
    unique_hkl = hkl_idx.unique_hkl
    n_unique = hkl_idx.n_unique

    I_obs = core.I
    median_cell = core.cell.median(dim=0).values

    # ── Per-HKL statistics ──
    counts = torch.bincount(hkl_id, minlength=n_unique)
    order = hkl_id.argsort()
    groups_I = I_obs[order].split(counts.tolist())

    mean_I = torch.zeros(n_unique)
    var_I = torch.zeros(n_unique)
    mult = torch.zeros(n_unique, dtype=torch.long)
    for h, gI in enumerate(groups_I):
        if len(gI) == 0:
            continue
        mean_I[h] = gI.mean()
        mult[h] = len(gI)
        if len(gI) > 1:
            var_I[h] = gI.var(correction=1)

    # SNR: t-statistic for mean ≠ 0 (precision of the mean estimate)
    # SNR = |mean| * sqrt(n) / std
    snr = mean_I.abs() * mult.float().clamp(min=1).sqrt() / var_I.clamp(min=1e-6).sqrt()

    # Resolution
    d_spacing_all = get_d_spacing(unique_hkl.float(), median_cell)

    # ── Resolution shells (equal-count binning) ──
    rank = d_spacing_all.argsort().argsort()
    hkl_shell = (rank.float() / n_unique * n_shells).long().clamp(max=n_shells - 1)

    # Shell centers
    shell_d = torch.zeros(n_shells)
    for sid in range(n_shells):
        m = hkl_shell == sid
        if m.sum() > 0:
            shell_d[sid] = d_spacing_all[m].mean()

    # ── Stage 1: Per-shell GenGauss+Gauss mixture fit ──
    # Pool normalized positive observations per shell
    keep_pool = (mean_I > 0) & (snr > snr_thresh)
    shell_beta: dict[int, float] = {}

    for sid in range(n_shells):
        hm = keep_pool & (hkl_shell == sid)
        if hm.sum() < 10:
            continue
        hkl_indices = torch.where(hm)[0]
        pooled = []
        for h_idx in hkl_indices:
            h = h_idx.item()
            g = groups_I[h]
            pos = g[g > 0]
            if len(pos) < 5:
                continue
            mu_h = pos.mean()
            if mu_h > 0:
                pooled.append(pos / mu_h)
        if len(pooled) < 5:
            continue
        pooled_t = torch.cat(pooled)
        pooled_t = pooled_t[pooled_t > 0]
        if len(pooled_t) < 500:
            continue
        log_pooled = torch.log(pooled_t)
        result = _fit_shell_lbfgs(log_pooled)
        if result is not None:
            shell_beta[sid] = result[0]  # β from mixture fit

    # Smooth β across shells with Gaussian kernel (σ=1.5 shells)
    if shell_beta:
        sids_sorted = sorted(shell_beta.keys())
        betas = np.array([shell_beta[s] for s in sids_sorted])
        idx_arr = np.array(sids_sorted, dtype=float)
        smooth_sigma = 1.5
        smoothed = np.zeros_like(betas)
        for k in range(len(betas)):
            w = np.exp(-0.5 * ((idx_arr - idx_arr[k]) / smooth_sigma) ** 2)
            smoothed[k] = np.average(betas, weights=w)
        for k, sid in enumerate(sids_sorted):
            shell_beta[sid] = float(smoothed[k])

    default_beta = np.median(list(shell_beta.values())) if shell_beta else 1.5

    # ── Stage 2: Per-HKL MLE ──
    pos_mask = I_obs > 0
    log_I_all = torch.log(I_obs[pos_mask])
    hkl_id_pos = hkl_id[pos_mask]

    counts_pos = torch.bincount(hkl_id_pos, minlength=n_unique)
    valid_mask = counts_pos >= min_obs
    valid_indices = torch.where(valid_mask)[0]
    n_valid = valid_indices.shape[0]

    if n_valid == 0:
        return HKLDistribution(
            hkl=torch.zeros(0, 3, dtype=torch.int16),
            I0=torch.zeros(0),
            alpha=torch.zeros(0),
            beta=torch.zeros(0),
            mu=torch.zeros(0),
            d_spacing=torch.zeros(0),
            n_obs=torch.zeros(0, dtype=torch.long),
            sigma_mu=torch.zeros(0),
            sigma_I0=torch.zeros(0),
        )

    # Remap to contiguous indices
    remap = torch.full((n_unique,), -1, dtype=torch.long)
    remap[valid_indices] = torch.arange(n_valid)
    obs_new_hkl = remap[hkl_id_pos]
    obs_keep = obs_new_hkl >= 0
    log_I_fit = log_I_all[obs_keep]
    hkl_fit_idx = obs_new_hkl[obs_keep]

    # Sort by HKL for grouped access
    sort_order = hkl_fit_idx.argsort()
    log_I_sorted = log_I_fit[sort_order]
    obs_counts = torch.bincount(hkl_fit_idx, minlength=n_valid)
    offsets = torch.zeros(n_valid + 1, dtype=torch.long)
    offsets[1:] = obs_counts.cumsum(0)

    # ── Correction pipeline setup ──
    # Legacy: scale_crystals=True creates ScaleBCorrection automatically
    if corrections is None and scale_crystals:
        corrections = [ScaleBCorrection(scale_prior=scale_prior,
                                        bfactor_prior=bfactor_prior)]

    pipeline: CorrectionPipeline | None = None
    obs_info: ObsInfo | None = None
    if corrections and n_em_cycles > 0:
        s_sq_per_obs = 1.0 / (4.0 * d_spacing_all[hkl_id].clamp(min=0.1) ** 2)
        crystal_id_sorted = core.crystal_id[pos_mask][obs_keep][sort_order].long()
        s_sq_sorted = s_sq_per_obs[pos_mask][obs_keep][sort_order]
        n_cryst = int(crystal_id_sorted.max().item()) + 1

        # Optional per-obs metadata for panel/peak corrections
        _chain = lambda attr: getattr(core, attr)[pos_mask][obs_keep][sort_order]
        panel_id_sorted = _chain("panel_id").long() if hasattr(core, "panel_id") and core.panel_id is not None else None
        fs_sorted = _chain("fs").float() if hasattr(core, "fs") and core.fs is not None else None
        ss_sorted = _chain("ss").float() if hasattr(core, "ss") and core.ss is not None else None
        peak_sorted = _chain("peak").float() if hasattr(core, "peak") and core.peak is not None else None
        bg_sorted = _chain("background").float() if hasattr(core, "background") and core.background is not None else None

        obs_info = ObsInfo(
            crystal_id=crystal_id_sorted,
            s_sq=s_sq_sorted,
            hkl_idx=hkl_fit_idx[sort_order],
            n_crystals=n_cryst,
            panel_id=panel_id_sorted,
            fs=fs_sorted,
            ss=ss_sorted,
            peak=peak_sorted,
            background=bg_sorted,
        )
        pipeline = CorrectionPipeline(
            corrections=corrections,
            free_fraction=free_fraction,
        )
        pipeline.init_masks(len(log_I_sorted))
        print(f"  Corrections: {len(corrections)}, "
              f"free set: {pipeline.free_mask.sum()}/{len(log_I_sorted)} "
              f"({free_fraction:.0%})")
        # Correction starts at zero (no correction initially)
        scale_correction = torch.zeros_like(log_I_sorted)

    # Assign β per HKL from shell fit
    hkl_valid = unique_hkl[valid_indices]
    hkl_shell_valid = hkl_shell[valid_indices]
    beta_perhkl = torch.full((n_valid,), default_beta)
    for sid, bval in shell_beta.items():
        m = hkl_shell_valid == sid
        if m.sum() > 0:
            beta_perhkl[m] = bval

    # Closed-form MLE for μ and α
    mu_perhkl = torch.zeros(n_valid)
    alpha_perhkl = torch.zeros(n_valid)
    sigma_mu_perhkl = torch.zeros(n_valid)
    beta_np = beta_perhkl.numpy()

    for i in range(n_valid):
        s, e = offsets[i].item(), offsets[i + 1].item()
        vals = log_I_sorted[s:e]
        n_h = e - s
        beta_i = beta_np[i]

        # μ = median + one IRLS correction
        mu_i = vals.median().item()
        w_sum_unnorm = 1.0  # fallback
        if abs(beta_i - 1.0) > 0.05:
            residual = (vals - mu_i).abs().clamp(min=1e-8)
            w = residual.pow(beta_i - 2)
            w_sum_unnorm = float(w.sum().item())
            w = w / w.sum()
            mu_i = (w * vals).sum().item()

        # α^β = (β/n) Σ|y - μ|^β
        abs_r = (vals - mu_i).abs().clamp(min=1e-8)
        alpha_i = float(
            ((beta_i / n_h) * abs_r.pow(beta_i).sum().item()) ** (1.0 / beta_i)
        )

        mu_perhkl[i] = mu_i
        alpha_perhkl[i] = max(alpha_i, 0.01)

        # σ(μ̂) from IRLS: 1/√(Σw_unnorm)
        sigma_mu_perhkl[i] = 1.0 / max(w_sum_unnorm, 1e-8) ** 0.5

    # ── Stage 3: Laplacian smoothing ──
    edge_src, edge_dst, n_neighbors = _build_adjacency(hkl_valid)
    _smooth_wt = obs_counts.float() if weighted_smooth else None

    log_alpha_smooth = _laplacian_smooth(
        torch.log(alpha_perhkl), edge_src, edge_dst, n_neighbors,
        n_smooth_iter, smooth_lambda, node_weights=_smooth_wt,
    )
    beta_smooth = _laplacian_smooth(
        beta_perhkl, edge_src, edge_dst, n_neighbors,
        n_smooth_iter, smooth_lambda, node_weights=_smooth_wt,
    )

    alpha_final = torch.exp(log_alpha_smooth)

    # ── Stage 4: EM refinement cycles with warm-up ──
    # Early iterations: heavy smoothing (nearly shell averages) + few IRLS steps
    # Late iterations: relax smoothing, allow per-HKL structure, full IRLS
    #
    # Warm-up schedule (linear interpolation):
    #   smooth_iter:  n_smooth_start → n_smooth_iter
    #   smooth_lam:   lam_start → smooth_lambda
    #   IRLS iters:   1 → 3
    n_smooth_start = max(n_smooth_iter * 4, 60)   # heavy early smoothing
    lam_start = min(smooth_lambda + 0.3, 0.8)     # strong mixing early

    for em_iter in range(n_em_cycles):
        # ── Warm-up schedule ──
        progress = em_iter / max(n_em_cycles - 1, 1)  # 0 → 1
        n_smooth_now = int(n_smooth_start + progress * (n_smooth_iter - n_smooth_start))
        lam_now = lam_start + progress * (smooth_lambda - lam_start)
        n_irls = max(1, int(1 + 2 * progress))  # 1 → 3

        # Expand μ to per-observation
        mu_expanded = mu_perhkl.repeat_interleave(obs_counts)

        # ── S-step: fit corrections on work set ──
        total_correction = torch.zeros_like(log_I_sorted)
        if pipeline is not None:
            residual_all = log_I_sorted - mu_expanded
            total_correction = pipeline.fit_and_predict(residual_all, obs_info)

        # ── E-step + M-step: per-HKL μ and α (all observations) ──
        beta_em = beta_smooth.numpy()
        for i in range(n_valid):
            s_off, e_off = offsets[i].item(), offsets[i + 1].item()
            vals = log_I_sorted[s_off:e_off] - total_correction[s_off:e_off]
            n_h = e_off - s_off
            bi = beta_em[i]

            # IRLS location estimate (warm up iterations)
            mu_i = vals.median().item()
            w_sum_unnorm = 1.0
            for _irls in range(n_irls):
                residual = (vals - mu_i).abs().clamp(min=1e-8)
                w = residual.pow(bi - 2)
                w_sum_unnorm = float(w.sum().item())
                w = w / w.sum()
                mu_i = (w * vals).sum().item()

            # α MLE (all observations)
            abs_r = (vals - mu_i).abs().clamp(min=1e-8)
            alpha_i = float(
                ((bi / n_h) * abs_r.pow(bi).sum().item()) ** (1.0 / bi)
            )

            mu_perhkl[i] = mu_i
            alpha_perhkl[i] = max(alpha_i, 0.01)
            sigma_mu_perhkl[i] = 1.0 / max(w_sum_unnorm, 1e-8) ** 0.5

        # Re-smooth α with warm-up schedule
        # (β stays fixed from shell fit — stable across EM)
        log_alpha_smooth = _laplacian_smooth(
            torch.log(alpha_perhkl), edge_src, edge_dst, n_neighbors,
            n_smooth_now, lam_now, node_weights=_smooth_wt,
        )
        alpha_final = torch.exp(log_alpha_smooth)

        if em_iter == 0 or (em_iter + 1) % 3 == 0 or em_iter == n_em_cycles - 1:
            msg = (f"  EM {em_iter+1}/{n_em_cycles}: smooth={n_smooth_now} "
                   f"λ={lam_now:.2f} IRLS={n_irls} med(α)={alpha_final.median():.4f}")
            if pipeline is not None:
                fs = pipeline.monitor(
                    log_I_sorted, mu_expanded, total_correction,
                    obs_counts, beta_smooth,
                )
                msg += (f"  α_w={fs.alpha_work:.4f} α_f={fs.alpha_free:.4f}"
                        f" f/w={fs.ratio:.3f}")
            print(msg)

    d_perhkl = d_spacing_all[valid_indices]

    # ── Stage 5 (optional): Linear-space I₀ refinement with Poisson model ──
    # Uses ALL observations (positive AND negative) with combined variance:
    #   σ²_total = I₀² · σ²_LN + g² · σ²_poisson
    if poisson_noise is not None and core.background is not None:
        # Calibrate gain from negative observations
        poisson_noise.calibrate_gain(core.I, core.background)

        # Compute σ²_poisson for ALL observations
        sigma2_poi_all = poisson_noise.sigma2(core.I, core.background)

        # Map ALL observations (not just positive) to valid HKL indices
        obs_new_hkl_all = remap[hkl_id]
        obs_keep_all = obs_new_hkl_all >= 0
        I_all_fit = core.I[obs_keep_all]
        hkl_fit_idx_all = obs_new_hkl_all[obs_keep_all]
        sigma2_poi_fit = sigma2_poi_all[obs_keep_all]

        # Sort by HKL (same order as log-space pipeline)
        sort_all = hkl_fit_idx_all.argsort()
        I_all_sorted = I_all_fit[sort_all]
        hkl_idx_all_sorted = hkl_fit_idx_all[sort_all]
        sigma2_poi_sorted = sigma2_poi_fit[sort_all]
        obs_counts_all = torch.bincount(hkl_fit_idx_all, minlength=n_valid)

        # Fraction of positive observations per HKL (diagnostic)
        n_pos_per_hkl = torch.zeros(n_valid)
        pos_flag = (I_all_sorted > 0).float()
        n_pos_per_hkl.scatter_add_(0, hkl_idx_all_sorted, pos_flag)
        frac_pos = n_pos_per_hkl / obs_counts_all.float().clamp(min=1)

        I0_lin, sigma_I0_lin = linear_space_I0(
            I_all=I_all_sorted,
            hkl_id_all=hkl_idx_all_sorted,
            sigma2_poisson=sigma2_poi_sorted,
            mu_log=mu_perhkl,
            alpha=alpha_final,
            beta=beta_smooth,
            obs_counts_all=obs_counts_all,
            n_valid=n_valid,
            n_iter=10,
        )

        n_zero = (I0_lin <= 0).sum().item()
        n_weak = (frac_pos < 0.6).sum().item()
        print(f"  Linear-space I₀: {n_valid} HKLs, {n_zero} at zero, "
              f"{n_weak} weak (f_pos<0.6), "
              f"gain={poisson_noise.gain:.4f}")

        return HKLDistribution(
            hkl=hkl_valid.to(torch.int16),
            I0=I0_lin,
            alpha=alpha_final,
            beta=beta_smooth,
            mu=mu_perhkl,
            d_spacing=d_perhkl,
            n_obs=obs_counts_all,
            sigma_mu=sigma_mu_perhkl,
            sigma_I0=sigma_I0_lin,
        )

    I0 = torch.exp(mu_perhkl)
    return HKLDistribution(
        hkl=hkl_valid.to(torch.int16),
        I0=I0,
        alpha=alpha_final,
        beta=beta_smooth,
        mu=mu_perhkl,
        d_spacing=d_perhkl,
        n_obs=obs_counts,
        sigma_mu=sigma_mu_perhkl,
        sigma_I0=I0 * sigma_mu_perhkl,
    )


def compare_distributions(
    fit1: HKLDistribution,
    fit2: HKLDistribution,
    n_shells: int = 20,
) -> DistributionComparison:
    """Compare two independent per-HKL distribution fits.

    Matches HKLs by Miller index, computes overall and per-shell
    Pearson correlation for I0 and alpha.
    """
    # Build lookup for fit2
    lookup2: dict[tuple[int, int, int], int] = {}
    hkl2 = fit2.hkl.long()
    for i in range(hkl2.shape[0]):
        key = (hkl2[i, 0].item(), hkl2[i, 1].item(), hkl2[i, 2].item())
        lookup2[key] = i

    idx1_list: list[int] = []
    idx2_list: list[int] = []
    hkl1 = fit1.hkl.long()
    for i in range(hkl1.shape[0]):
        key = (hkl1[i, 0].item(), hkl1[i, 1].item(), hkl1[i, 2].item())
        j = lookup2.get(key)
        if j is not None:
            idx1_list.append(i)
            idx2_list.append(j)

    if not idx1_list:
        return DistributionComparison(
            hkl=torch.zeros(0, 3, dtype=torch.int16),
            d_spacing=torch.zeros(0),
            I0_1=torch.zeros(0),
            I0_2=torch.zeros(0),
            alpha_1=torch.zeros(0),
            alpha_2=torch.zeros(0),
            beta_1=torch.zeros(0),
            beta_2=torch.zeros(0),
            cc_I0=0.0,
            cc_alpha=0.0,
            shells=[],
        )

    i1 = torch.tensor(idx1_list, dtype=torch.long)
    i2 = torch.tensor(idx2_list, dtype=torch.long)

    matched_hkl = fit1.hkl[i1]
    d = fit1.d_spacing[i1]
    I0_1 = fit1.I0[i1]
    I0_2 = fit2.I0[i2]
    alpha_1 = fit1.alpha[i1]
    alpha_2 = fit2.alpha[i2]
    beta_1 = fit1.beta[i1]
    beta_2 = fit2.beta[i2]

    cc_I0 = _pearson_cc(I0_1, I0_2)
    cc_alpha = _pearson_cc(alpha_1, alpha_2)

    # Per-shell stats (equal-count binning)
    shells = _shell_dist_stats(d, I0_1, I0_2, alpha_1, alpha_2, n_shells)

    return DistributionComparison(
        hkl=matched_hkl,
        d_spacing=d,
        I0_1=I0_1,
        I0_2=I0_2,
        alpha_1=alpha_1,
        alpha_2=alpha_2,
        beta_1=beta_1,
        beta_2=beta_2,
        cc_I0=cc_I0,
        cc_alpha=cc_alpha,
        shells=shells,
    )


def split_and_compare(
    stream,
    *,
    seed: int = 42,
    n_shells: int = 20,
    **fit_kwargs,
) -> tuple[HKLDistribution, HKLDistribution, DistributionComparison]:
    """Split-half validation of per-HKL distribution fitting.

    Splits crystals randomly into two halves, fits each independently,
    then compares the results.
    """
    h1, h2 = stream.split_random(2, seed=seed)
    fit1 = fit_gengauss(h1, **fit_kwargs)
    fit2 = fit_gengauss(h2, **fit_kwargs)
    comp = compare_distributions(fit1, fit2, n_shells=n_shells)
    return fit1, fit2, comp


def format_dist_table(comp: DistributionComparison) -> str:
    """Format distribution comparison as a text table."""
    lines = []
    lines.append(
        f"{'d_max':>8s} {'d_min':>8s} {'N':>7s} {'CC(I0)':>8s} {'CC(α)':>8s}"
    )
    lines.append("-" * 43)
    for s in comp.shells:
        lines.append(
            f"{s.d_max:8.2f} {s.d_min:8.2f} {s.n_matched:7d} "
            f"{s.cc_I0:8.4f} {s.cc_alpha:8.4f}"
        )
    lines.append("-" * 43)
    n_total = sum(s.n_matched for s in comp.shells)
    lines.append(
        f"{'Overall':>17s} {n_total:7d} {comp.cc_I0:8.4f} {comp.cc_alpha:8.4f}"
    )
    return "\n".join(lines)


# ── Internal helpers ─────────────────────────────────────────


def _pearson_cc(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() < 2:
        return 0.0
    stacked = torch.stack([a.double(), b.double()])
    return torch.corrcoef(stacked)[0, 1].item()


def _shell_dist_stats(
    d: torch.Tensor,
    I0_1: torch.Tensor,
    I0_2: torch.Tensor,
    alpha_1: torch.Tensor,
    alpha_2: torch.Tensor,
    n_shells: int,
) -> list[DistShellStats]:
    """Equal-count resolution binning with per-shell CC."""
    if d.numel() == 0:
        return []
    order = d.argsort(descending=True)
    bins = torch.zeros(d.numel(), dtype=torch.long)
    chunk = d.numel() // n_shells
    rem = d.numel() % n_shells
    start = 0
    for i in range(n_shells):
        end = start + chunk + (1 if i < rem else 0)
        bins[order[start:end]] = i
        start = end

    shells = []
    for i in range(n_shells):
        mask = bins == i
        n = mask.sum().item()
        if n < 2:
            continue
        d_shell = d[mask]
        shells.append(
            DistShellStats(
                d_min=d_shell.min().item(),
                d_max=d_shell.max().item(),
                cc_I0=_pearson_cc(I0_1[mask], I0_2[mask]),
                cc_alpha=_pearson_cc(alpha_1[mask], alpha_2[mask]),
                n_matched=n,
            )
        )
    return shells
