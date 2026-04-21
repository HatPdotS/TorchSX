"""EM orchestrator: alternates D-step (distribution) and S-step (corrections).

The EM cycle cleanly separates:
  - **D-step (KL)**: fit per-HKL distribution parameters (I₀, α, β)
    using ALL observations (positive + negative) with the combined
    Poisson+LogNormal variance model.
  - **S-step (NLL)**: fit per-crystal/panel corrections (G, B, panel)
    via ridge-regularised OLS on log-space residuals of positive
    observations (work set only).

The corrections operate in log-space (where they are additive), and are
converted to linear-space multiplicative scale factors when passed to
the D-step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import Tensor

from ._corrections import (
    Correction,
    CorrectionPipeline,
    ObsInfo,
    ScaleBCorrection,
)
from ._distribution import (
    HKLDistribution,
    _build_adjacency,
    _fit_shell_lbfgs,
    _laplacian_smooth,
)
from ._dstep import d_step
from ._noise import PoissonNoiseModel


# ── Configuration ─────────────────────────────────────────────


@dataclass
class EMConfig:
    """Configuration for the EM refinement loop."""

    n_em_cycles: int = 15
    n_smooth_iter: int = 15
    smooth_lambda: float = 0.3
    min_obs: int = 20
    n_shells: int = 20
    snr_thresh: float = 3.0
    free_fraction: float = 0.1
    # Warm-up schedule
    n_smooth_start: int = 60
    lam_start: float = 0.6
    n_newton_start: int = 3
    n_newton_final: int = 10
    # Alpha regularization
    alpha_prior_lambda: float = 0.0  # >0 enables shell-level alpha; 0 = Laplacian (default)


# ── Observation preparation ──────────────────────────────────


@dataclass
class PreparedObs:
    """Pre-processed observation arrays for the EM loop."""

    # Valid HKL info
    hkl_valid: Tensor         # (n_valid, 3) int16
    d_spacing: Tensor         # (n_valid,)
    beta_perhkl: Tensor       # (n_valid,)
    n_valid: int

    # ALL observations (pos + neg), sorted by HKL
    I_all_sorted: Tensor      # (N_all,)
    hkl_id_all: Tensor        # (N_all,) long
    sigma2_poi_all: Tensor    # (N_all,)
    obs_counts_all: Tensor    # (n_valid,)

    # Positive observations only, sorted by HKL
    log_I_pos_sorted: Tensor  # (M,)
    I_pos_sorted: Tensor      # (M,) = exp(log_I_pos_sorted)
    hkl_id_pos: Tensor        # (M,) long
    obs_counts_pos: Tensor    # (n_valid,)
    offsets_pos: Tensor       # (n_valid+1,) long

    # Initial estimates from log-space MLE
    mu_init: Tensor           # (n_valid,)
    alpha_init: Tensor        # (n_valid,)

    # Adjacency graph for smoothing
    edge_src: Tensor
    edge_dst: Tensor
    n_neighbors: Tensor

    # Per-obs background (for Poisson model)
    bg_all: Tensor | None       # (N_all,) or None

    # External per-obs partiality (sorted order), or None
    partiality_all: Tensor | None
    partiality_pos: Tensor | None

    # ObsInfo for corrections (positive obs only)
    obs_info_pos: ObsInfo | None
    # ObsInfo for corrections mapped to ALL obs
    obs_info_all: ObsInfo | None


def prepare_observations(
    stream,
    config: EMConfig,
    poisson_noise: PoissonNoiseModel,
    corrections: Sequence[Correction] | None = None,
    partiality: Tensor | None = None,
) -> PreparedObs:
    """Extract and sort observation arrays, compute initial estimates.

    Builds BOTH positive-only sorted arrays (for S-step / α estimation)
    AND all-obs sorted arrays (for D-step / I₀ estimation).
    """
    import numpy as np
    from torchref.base.reciprocal.hkl import get_d_spacing
    from torchsx.merge._hkl import compute_hkl_index

    core = stream._core
    hkl_idx = compute_hkl_index(core.hkl)
    hkl_id = hkl_idx.hkl_id
    unique_hkl = hkl_idx.unique_hkl
    n_unique = hkl_idx.n_unique
    I_obs = core.I
    median_cell = core.cell.median(dim=0).values

    # ── Per-HKL statistics (for shell fit / SNR) ──
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

    snr = mean_I.abs() * mult.float().clamp(min=1).sqrt() / var_I.clamp(min=1e-6).sqrt()
    d_spacing_all = get_d_spacing(unique_hkl.float(), median_cell)

    # ── Shell β estimation (reuse existing infrastructure) ──
    rank = d_spacing_all.argsort().argsort()
    hkl_shell = (rank.float() / n_unique * config.n_shells).long().clamp(
        max=config.n_shells - 1
    )
    keep_pool = (mean_I > 0) & (snr > config.snr_thresh)
    shell_beta: dict[int, float] = {}

    for sid in range(config.n_shells):
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
            shell_beta[sid] = result[0]

    # Smooth β across shells
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

    # ── Identify valid HKLs (enough positive observations) ──
    pos_mask = I_obs > 0
    hkl_id_pos_raw = hkl_id[pos_mask]
    counts_pos_raw = torch.bincount(hkl_id_pos_raw, minlength=n_unique)
    valid_mask = counts_pos_raw >= config.min_obs
    valid_indices = torch.where(valid_mask)[0]
    n_valid = valid_indices.shape[0]

    if n_valid == 0:
        raise ValueError("No HKLs with sufficient observations")

    # Remap to contiguous [0, n_valid)
    remap = torch.full((n_unique,), -1, dtype=torch.long)
    remap[valid_indices] = torch.arange(n_valid)

    # ── Build POSITIVE-only sorted arrays ──
    obs_new_hkl_pos = remap[hkl_id_pos_raw]
    obs_keep_pos = obs_new_hkl_pos >= 0
    log_I_pos = torch.log(I_obs[pos_mask][obs_keep_pos])
    hkl_fit_idx_pos = obs_new_hkl_pos[obs_keep_pos]
    sort_pos = hkl_fit_idx_pos.argsort()
    log_I_pos_sorted = log_I_pos[sort_pos]
    hkl_id_pos_sorted = hkl_fit_idx_pos[sort_pos]
    obs_counts_pos = torch.bincount(hkl_id_pos_sorted, minlength=n_valid)
    offsets_pos = torch.zeros(n_valid + 1, dtype=torch.long)
    offsets_pos[1:] = obs_counts_pos.cumsum(0)

    # ── Build ALL-obs sorted arrays ──
    obs_new_hkl_all = remap[hkl_id]
    obs_keep_all = obs_new_hkl_all >= 0
    I_all_kept = I_obs[obs_keep_all]
    hkl_fit_idx_all = obs_new_hkl_all[obs_keep_all]
    sort_all = hkl_fit_idx_all.argsort()
    I_all_sorted = I_all_kept[sort_all]
    hkl_id_all_sorted = hkl_fit_idx_all[sort_all]
    obs_counts_all = torch.bincount(hkl_id_all_sorted, minlength=n_valid)

    # ── Poisson variance for ALL obs ──
    # Use gain=1.0 (detector calibrated: 1 ADU = 1 photon)
    # Skip calibrate_gain() which biases from truncated negatives
    poisson_noise.gain = 1.0
    print(f"  PoissonNoiseModel: gain=1.0 (detector calibrated)")

    bg_per_pixel = core.background[obs_keep_all][sort_all]
    sigma2_poi_all = poisson_noise.sigma2(I_all_sorted, bg_per_pixel)
    # Total background in peak region (for Poisson marginal likelihood)
    bg_all = bg_per_pixel * poisson_noise.n_peak

    # ── Initial log-space MLE for μ and α ──
    mu_init = torch.zeros(n_valid)
    alpha_init = torch.zeros(n_valid)
    for i in range(n_valid):
        s, e = offsets_pos[i].item(), offsets_pos[i + 1].item()
        vals = log_I_pos_sorted[s:e]
        n_h = e - s
        beta_i = default_beta
        sid_i = hkl_shell[valid_indices[i]].item()
        if sid_i in shell_beta:
            beta_i = shell_beta[sid_i]
        mu_i = vals.median()
        w = (vals - mu_i).abs().clamp(min=1e-8).pow(beta_i - 2)
        mu_i = (w * vals).sum() / w.sum().clamp(min=1e-12)
        mu_init[i] = mu_i
        r = (vals - mu_i).abs().clamp(min=1e-8)
        alpha_init[i] = (beta_i / n_h * r.pow(beta_i).sum()).pow(1.0 / beta_i).clamp(min=0.01)

    # ── β per HKL ──
    hkl_valid = unique_hkl[valid_indices]
    hkl_shell_valid = hkl_shell[valid_indices]
    beta_perhkl = torch.full((n_valid,), default_beta)
    for sid, bval in shell_beta.items():
        m = hkl_shell_valid == sid
        if m.sum() > 0:
            beta_perhkl[m] = bval

    # ── Adjacency graph ──
    edge_src, edge_dst, n_neighbors = _build_adjacency(hkl_valid)

    # ── Smooth initial α ──
    log_alpha_smooth = _laplacian_smooth(
        torch.log(alpha_init.clamp(min=1e-6)),
        edge_src, edge_dst, n_neighbors,
        config.n_smooth_iter, config.smooth_lambda,
    )
    alpha_init = torch.exp(log_alpha_smooth)

    # ── ObsInfo for corrections ──
    s_sq_per_obs = 1.0 / (4.0 * d_spacing_all[hkl_id].clamp(min=0.1) ** 2)

    obs_info_pos = None
    obs_info_all = None
    if corrections:
        # Positive-obs ObsInfo (for S-step fitting)
        cid_pos = core.crystal_id[pos_mask][obs_keep_pos][sort_pos].long()
        s_sq_pos = s_sq_per_obs[pos_mask][obs_keep_pos][sort_pos]
        n_cryst = int(core.crystal_id.max().item()) + 1

        _chain_pos = lambda attr: getattr(core, attr)[pos_mask][obs_keep_pos][sort_pos]
        obs_info_pos = ObsInfo(
            crystal_id=cid_pos,
            s_sq=s_sq_pos,
            hkl_idx=hkl_id_pos_sorted,
            n_crystals=n_cryst,
            panel_id=_chain_pos("panel_id").long() if core.panel_id is not None else None,
            fs=_chain_pos("fs").float() if core.fs is not None else None,
            ss=_chain_pos("ss").float() if core.ss is not None else None,
            peak=_chain_pos("peak").float() if core.peak is not None else None,
            background=_chain_pos("background").float() if core.background is not None else None,
        )

        # All-obs ObsInfo (for mapping corrections to all obs)
        cid_all = core.crystal_id[obs_keep_all][sort_all].long()
        s_sq_all = s_sq_per_obs[obs_keep_all][sort_all]

        _chain_all = lambda attr: getattr(core, attr)[obs_keep_all][sort_all]
        obs_info_all = ObsInfo(
            crystal_id=cid_all,
            s_sq=s_sq_all,
            hkl_idx=hkl_id_all_sorted,
            n_crystals=n_cryst,
            panel_id=_chain_all("panel_id").long() if core.panel_id is not None else None,
            fs=_chain_all("fs").float() if core.fs is not None else None,
            ss=_chain_all("ss").float() if core.ss is not None else None,
            peak=_chain_all("peak").float() if core.peak is not None else None,
            background=_chain_all("background").float() if core.background is not None else None,
        )

    # ── Reindex external partiality to sorted order ──
    partiality_all_sorted = None
    partiality_pos_sorted = None
    if partiality is not None:
        partiality_all_sorted = partiality[obs_keep_all][sort_all]
        partiality_pos_sorted = partiality[pos_mask][obs_keep_pos][sort_pos]

    return PreparedObs(
        hkl_valid=hkl_valid,
        d_spacing=d_spacing_all[valid_indices],
        beta_perhkl=beta_perhkl,
        n_valid=n_valid,
        I_all_sorted=I_all_sorted,
        hkl_id_all=hkl_id_all_sorted,
        sigma2_poi_all=sigma2_poi_all,
        obs_counts_all=obs_counts_all,
        log_I_pos_sorted=log_I_pos_sorted,
        I_pos_sorted=torch.exp(log_I_pos_sorted),
        hkl_id_pos=hkl_id_pos_sorted,
        obs_counts_pos=obs_counts_pos,
        offsets_pos=offsets_pos,
        mu_init=mu_init,
        alpha_init=alpha_init,
        edge_src=edge_src,
        edge_dst=edge_dst,
        n_neighbors=n_neighbors,
        bg_all=bg_all,
        partiality_all=partiality_all_sorted,
        partiality_pos=partiality_pos_sorted,
        obs_info_pos=obs_info_pos,
        obs_info_all=obs_info_all,
    )


# ── Main EM loop ─────────────────────────────────────────────


def _prep_to_device(prep: PreparedObs, device: str | torch.device) -> PreparedObs:
    """Move all PreparedObs tensors to the given device."""
    import dataclasses
    fields = {}
    for f in dataclasses.fields(prep):
        val = getattr(prep, f.name)
        if isinstance(val, torch.Tensor):
            fields[f.name] = val.to(device)
        elif isinstance(val, ObsInfo) and val is not None:
            fields[f.name] = ObsInfo(
                **{k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in val.__dict__.items()},
            )
        else:
            fields[f.name] = val
    return PreparedObs(**fields)


def fit_em(
    stream,
    *,
    config: EMConfig | None = None,
    corrections: Sequence[Correction] | None = None,
    poisson_noise: PoissonNoiseModel | None = None,
    device: str | torch.device = "cpu",
    partiality: Tensor | None = None,
) -> HKLDistribution:
    """Fit per-HKL distributions via EM with Poisson+LogNormal model.

    Parameters
    ----------
    stream : SFXStream
    config : EMConfig (defaults provided if None)
    corrections : list of Correction instances (e.g. ScaleBCorrection)
    poisson_noise : PoissonNoiseModel (created with defaults if None)
    device : "cpu" or "cuda" — all computation runs on this device
    partiality : (N_obs,) tensor in stream order, optional.
        Per-observation partiality scale factor (0 < p <= 1).
        Multiplied into the model's scale: I_obs ~ Poisson(I0 * p * scale + bg).
        This scales the distribution, not the observations, so Poisson
        variance remains correct.

    Returns
    -------
    HKLDistribution with I₀ estimated in linear space (on CPU).
    """
    if config is None:
        config = EMConfig()
    if poisson_noise is None:
        poisson_noise = PoissonNoiseModel()

    # ── Prepare observations (on CPU), then move to device ──
    prep = prepare_observations(stream, config, poisson_noise, corrections,
                                partiality=partiality)
    if str(device) != "cpu":
        prep = _prep_to_device(prep, device)

    # ── Correction pipeline ──
    pipeline: CorrectionPipeline | None = None
    if corrections and config.n_em_cycles > 0 and prep.obs_info_pos is not None:
        pipeline = CorrectionPipeline(
            corrections=corrections,
            free_fraction=config.free_fraction,
        )
        pipeline.init_masks(len(prep.log_I_pos_sorted), device=device)
        print(
            f"  Corrections: {len(corrections)}, "
            f"free set: {pipeline.free_mask.sum()}/{len(prep.log_I_pos_sorted)} "
            f"({config.free_fraction:.0%})"
        )

    # ── Initial I₀ from log-space MLE ──
    I0 = torch.exp(prep.mu_init).clamp(min=1e-6)
    alpha = prep.alpha_init.clone()
    beta = prep.beta_perhkl.clone()

    # ── EM loop ──
    for em_iter in range(config.n_em_cycles):
        # Warm-up schedule
        progress = em_iter / max(config.n_em_cycles - 1, 1)
        n_smooth_now = int(
            config.n_smooth_start + progress * (config.n_smooth_iter - config.n_smooth_start)
        )
        lam_now = config.lam_start + progress * (config.smooth_lambda - config.lam_start)
        n_newton = int(
            config.n_newton_start + progress * (config.n_newton_final - config.n_newton_start)
        )

        # ── S-step: fit corrections in log-space on positive obs ──
        correction_all = torch.zeros_like(prep.I_all_sorted)
        if pipeline is not None:
            # Residuals in log-space: log(I_pos) - log(I₀_h)
            log_I0_per_pos = torch.log(I0.clamp(min=1e-8)).repeat_interleave(
                prep.obs_counts_pos
            )
            residuals_pos = prep.log_I_pos_sorted - log_I0_per_pos
            # Fit and predict on positive obs
            correction_pos = pipeline.fit_and_predict(
                residuals_pos, prep.obs_info_pos
            )
            # Map corrections to ALL obs via predict (uses crystal_id/panel_id)
            correction_all = pipeline.predict(prep.obs_info_all)

        # ── D-step: fit I₀ (all obs), α (pos obs) ──
        # Apply corrections: I_corrected = I_obs / exp(correction)
        scale_all = torch.exp(correction_all)

        # Include external partiality in the model scale.
        # This scales the distribution: I_obs ~ Poisson(I0 * p * scale + bg),
        # keeping Poisson variance correct (based on expected, not amplified count).
        if prep.partiality_all is not None:
            scale_all = scale_all * prep.partiality_all

        I_all_corrected = prep.I_all_sorted / scale_all.clamp(min=1e-8)

        # Positive corrected observations
        scale_pos = torch.exp(
            pipeline.predict(prep.obs_info_pos) if pipeline else torch.zeros_like(prep.I_pos_sorted)
        )
        if prep.partiality_pos is not None:
            scale_pos = scale_pos * prep.partiality_pos
        I_pos_corrected = prep.I_pos_sorted / scale_pos.clamp(min=1e-8)

        # Adjust Poisson variance for correction scale
        sigma2_poi_corrected = prep.sigma2_poi_all / (scale_all ** 2).clamp(min=1e-8)

        result = d_step(
            I_all_corrected=I_all_corrected,
            hkl_id_all=prep.hkl_id_all,
            sigma2_poisson_all=sigma2_poi_corrected,
            obs_counts_all=prep.obs_counts_all,
            I_pos_corrected=I_pos_corrected,
            hkl_id_pos=prep.hkl_id_pos,
            obs_counts_pos=prep.obs_counts_pos,
            I0_prev=I0,
            alpha_prev=alpha,
            beta=beta,
            n_valid=prep.n_valid,
            n_newton_iter=n_newton,
            edge_src=prep.edge_src,
            edge_dst=prep.edge_dst,
            n_neighbors=prep.n_neighbors,
            n_smooth_iter=n_smooth_now,
            smooth_lambda=lam_now,
            # Raw obs + scales for Poisson marginal
            I_obs_all=prep.I_all_sorted,
            scale_all=scale_all,
            bg=prep.bg_all if prep.bg_all is not None else 0.0,
            # Resolution-dependent alpha prior
            d_spacing=prep.d_spacing,
            n_alpha_shells=config.n_shells,
            alpha_prior_lambda=config.alpha_prior_lambda,
        )

        I0, alpha = result.I0, result.alpha
        sigma_I0 = result.sigma_I0

        # ── Monitor ──
        if (em_iter + 1) % max(1, config.n_em_cycles // 5) == 0 or em_iter == config.n_em_cycles - 1:
            n_zero = (I0 <= 1e-6).sum().item()
            med_alpha = alpha.median().item()
            print(
                f"  EM {em_iter+1}/{config.n_em_cycles}: "
                f"Newton={n_newton} smooth={n_smooth_now} λ={lam_now:.2f} "
                f"med(α)={med_alpha:.4f} I₀=0: {n_zero}/{prep.n_valid}"
            )

    # ── Truncation bias correction ──
    # The quadrature grid floor biases I0 upward for weak reflections.
    # Correct using the inverse Mills ratio of the truncated normal:
    #   E[log(signal) | log(signal) > log(signal_lo)] = log(I0) + α·φ(z)/Φ_c(z)
    # so the unbiased log(I0) = log(I0_raw) - α·φ(z)/Φ_c(z).
    if prep.bg_all is not None:
        bg_hkl = torch.zeros(prep.n_valid, device=I0.device)
        bg_hkl.scatter_add_(0, prep.hkl_id_all, prep.bg_all.clamp(min=0))
        bg_hkl = (bg_hkl / prep.obs_counts_all.float().clamp(min=1)).clamp(min=1.0)
    else:
        bg_hkl = torch.ones(prep.n_valid, device=I0.device)
    sigma_floor = (bg_hkl / prep.obs_counts_all.float().clamp(min=1)).sqrt()
    sig_lo_hkl = (sigma_floor * 0.5).clamp(min=0.01)

    with torch.no_grad():
        log_I0_raw = torch.log(I0.clamp(min=1e-8))
        eps_min = torch.log(sig_lo_hkl.clamp(min=1e-8)) - log_I0_raw
        z_min = eps_min / alpha
        phi = torch.exp(-0.5 * z_min ** 2) * 0.3989422804014327  # 1/√(2π)
        Phi_c = (0.5 * torch.erfc(z_min * 0.7071067811865476)).clamp(min=1e-8)
        bias = alpha * phi / Phi_c
        I0 = torch.exp(log_I0_raw - bias)

    # ── Final σ(I₀) ──
    sigma_I0_final = result.sigma_I0 if config.n_em_cycles > 0 else I0 * 0.1
    mu_final = torch.log(I0.clamp(min=1e-8))
    sigma_mu_final = sigma_I0_final / I0.clamp(min=1e-8)

    return HKLDistribution(
        hkl=prep.hkl_valid.cpu(),
        I0=I0.cpu(),
        alpha=alpha.cpu(),
        beta=beta.cpu(),
        mu=mu_final.cpu(),
        d_spacing=prep.d_spacing.cpu(),
        n_obs=prep.obs_counts_all.cpu(),
        sigma_mu=sigma_mu_final.cpu(),
        sigma_I0=sigma_I0_final.cpu(),
    )





