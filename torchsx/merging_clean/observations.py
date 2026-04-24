"""Observation preparation: stream → sorted tensors + initial estimates.

Ported and trimmed from `torchsx.merge._em.prepare_observations`.  The
output is everything `UnifiedMergeLoss` needs as non-parameter buffers
(observation tensors, quadrature grid, adjacency, ObsInfo) plus the
initial parameter values (`mu_init`, `alpha_init`, `beta_perhkl`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from torchsx.merge._distribution import _fit_shell_lbfgs
from torchsx.merge._hkl import compute_hkl_index
from torchsx.merge._noise import PoissonNoiseModel
from torchsx.merging_clean.adjacency import Adjacency, build_adjacency
from torchsx.merging_clean.config import MergeConfig
from torchsx.merging_clean.corrections import ObsInfo
from torchsx.merging_clean.quadrature import QuadratureGrid  # noqa: F401  — re-exported


@dataclass
class PreparedObs:
    """Everything needed to build a UnifiedMergeLoss."""

    # Valid-HKL metadata
    hkl_valid: Tensor        # (n_valid, 3) int16
    d_spacing: Tensor        # (n_valid,)
    n_valid: int

    # ALL observations (pos + neg), sorted by HKL
    I_obs: Tensor            # (N_all,) raw intensities
    hkl_id: Tensor           # (N_all,) long — index into valid-HKL array
    bg: Tensor               # (N_all,)  — total bg in peak region
    obs_counts: Tensor       # (n_valid,) obs per HKL
    n_obs_total: int

    # Initial parameter values
    mu_init: Tensor          # (n_valid,) ≈ log I0
    alpha_init: Tensor       # (n_valid,)
    beta_perhkl: Tensor      # (n_valid,)

    # Adjacency graph over valid HKLs
    adjacency: Adjacency

    # Per-observation metadata for corrections
    obs_info: ObsInfo

    # External partiality (sorted), or None
    partiality: Tensor | None

    # French–Wilson prior inputs (per valid HKL)
    shell_id_fw: Tensor      # (n_valid,) long — rank-binned on d_spacing
    is_centric: Tensor       # (n_valid,) bool — from torchref is_centric_from_hkl

    def to(self, device) -> "PreparedObs":
        return PreparedObs(
            hkl_valid=self.hkl_valid.to(device),
            d_spacing=self.d_spacing.to(device),
            n_valid=self.n_valid,
            I_obs=self.I_obs.to(device),
            hkl_id=self.hkl_id.to(device),
            bg=self.bg.to(device),
            obs_counts=self.obs_counts.to(device),
            n_obs_total=self.n_obs_total,
            mu_init=self.mu_init.to(device),
            alpha_init=self.alpha_init.to(device),
            beta_perhkl=self.beta_perhkl.to(device),
            adjacency=self.adjacency.to(device),
            obs_info=self.obs_info.to(device),
            partiality=None if self.partiality is None else self.partiality.to(device),
            shell_id_fw=self.shell_id_fw.to(device),
            is_centric=self.is_centric.to(device),
        )


def prepare_observations(
    stream,
    config: MergeConfig,
    *,
    partiality: Tensor | None = None,
    poisson_noise: PoissonNoiseModel | None = None,
) -> PreparedObs:
    """Stream → PreparedObs (CPU).  Call `.to(device)` afterwards if needed."""
    from torchref.base.reciprocal.hkl import get_d_spacing

    if poisson_noise is None:
        poisson_noise = PoissonNoiseModel()

    core = stream._core
    core_I = core.I
    core_background = core.background

    # ── Detector gain (ADU / photon) ──
    # The Poisson likelihood treats I_obs + bg as photon counts, so we need to
    # divide ADU by gain before feeding them to the loss.
    gain_cfg = config.gain
    if isinstance(gain_cfg, (int, float)):
        gain = float(gain_cfg)
        gain_src = "explicit"
    elif gain_cfg == "stream":
        adu_per_eV = 1.0
        if core.geometry is not None:
            adu_per_eV = float(getattr(core.geometry, "adu_per_eV", 1.0) or 1.0)
        mean_eV = float(core.beam_energy.float().mean().item())
        gain = adu_per_eV * mean_eV
        if gain <= 0:
            gain = 1.0
        gain_src = f"stream (adu_per_eV={adu_per_eV:.6g} × {mean_eV:.1f} eV)"
    elif gain_cfg == "empirical":
        g = poisson_noise.calibrate_gain(core_I, core_background)
        gain = float(g) if g and g > 0 else 1.0
        gain_src = "empirical (negatives-Mandel)"
    else:
        raise ValueError(f"Unknown config.gain: {gain_cfg!r}")
    print(f"  gain = {gain:.4g} ADU/photon  [{gain_src}]", flush=True)
    poisson_noise.gain = gain

    # Convert ADU → photon-equivalent counts for the Poisson model.
    core_I = core_I / gain
    core_background = core_background / gain
    hkl_idx = compute_hkl_index(core.hkl)
    hkl_id_raw = hkl_idx.hkl_id
    unique_hkl = hkl_idx.unique_hkl
    n_unique = hkl_idx.n_unique
    I_obs_stream = core_I
    median_cell = core.cell.median(dim=0).values

    # ── Per-HKL statistics (for shell β fit / SNR) ──
    counts = torch.bincount(hkl_id_raw, minlength=n_unique)
    order = hkl_id_raw.argsort()
    groups_I = I_obs_stream[order].split(counts.tolist())

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

    # ── Shell β estimation (pooled GenGauss+Gauss mixture) ──
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
            g = groups_I[int(h_idx.item())]
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
        result = _fit_shell_lbfgs(torch.log(pooled_t))
        if result is not None:
            shell_beta[sid] = result[0]
    # Smooth across shells
    if shell_beta:
        sids = sorted(shell_beta.keys())
        betas = np.array([shell_beta[s] for s in sids])
        idx_arr = np.array(sids, dtype=float)
        smoothed = np.zeros_like(betas)
        for k in range(len(betas)):
            w = np.exp(-0.5 * ((idx_arr - idx_arr[k]) / 1.5) ** 2)
            smoothed[k] = np.average(betas, weights=w)
        for k, sid in enumerate(sids):
            shell_beta[sid] = float(smoothed[k])
    default_beta = float(np.median(list(shell_beta.values()))) if shell_beta else 1.5

    # ── Valid-HKL filter (min positive obs) ──
    pos_mask = I_obs_stream > 0
    hkl_id_pos_raw = hkl_id_raw[pos_mask]
    counts_pos = torch.bincount(hkl_id_pos_raw, minlength=n_unique)
    valid_mask = counts_pos >= config.min_obs
    valid_indices = torch.where(valid_mask)[0]
    n_valid = int(valid_indices.shape[0])
    if n_valid == 0:
        raise ValueError("No HKLs with sufficient positive observations")

    remap = torch.full((n_unique,), -1, dtype=torch.long)
    remap[valid_indices] = torch.arange(n_valid)

    # ── Build ALL-obs sorted arrays ──
    obs_new_hkl = remap[hkl_id_raw]
    keep_all = obs_new_hkl >= 0
    hkl_fit_idx = obs_new_hkl[keep_all]
    sort_all = hkl_fit_idx.argsort()

    I_obs = I_obs_stream[keep_all][sort_all]
    hkl_id = hkl_fit_idx[sort_all]
    obs_counts = torch.bincount(hkl_id, minlength=n_valid)
    n_obs_total = int(I_obs.shape[0])

    # Total background in peak region (bg_per_pixel * n_peak)
    bg_per_pixel = core_background[keep_all][sort_all]
    bg = bg_per_pixel * poisson_noise.n_peak

    # ── Initial log-space MLE for μ and α (positive obs per HKL) ──
    pos_in_sorted = I_obs > 0
    hkl_id_pos_sorted = hkl_id[pos_in_sorted]
    log_I_pos = torch.log(I_obs[pos_in_sorted].clamp(min=1e-8))
    obs_counts_pos = torch.bincount(hkl_id_pos_sorted, minlength=n_valid)
    offsets_pos = torch.zeros(n_valid + 1, dtype=torch.long)
    offsets_pos[1:] = obs_counts_pos.cumsum(0)
    # Re-sort positive-only by HKL (they are already sorted via sort_all; the
    # subset `pos_in_sorted` preserves order, so hkl_id_pos_sorted is sorted).

    mu_init = torch.zeros(n_valid)
    alpha_init = torch.zeros(n_valid)
    hkl_shell_valid = hkl_shell[valid_indices]
    for i in range(n_valid):
        s, e = int(offsets_pos[i]), int(offsets_pos[i + 1])
        vals = log_I_pos[s:e]
        n_h = e - s
        if n_h == 0:
            mu_init[i] = 0.0
            alpha_init[i] = 1.0
            continue
        beta_i = shell_beta.get(int(hkl_shell_valid[i].item()), default_beta)
        mu_i = vals.median()
        if abs(beta_i - 1.0) > 0.05:
            w = (vals - mu_i).abs().clamp(min=1e-8).pow(beta_i - 2)
            mu_i = (w * vals).sum() / w.sum().clamp(min=1e-12)
        mu_init[i] = mu_i
        r = (vals - mu_i).abs().clamp(min=1e-8)
        alpha_init[i] = (
            (beta_i / n_h * r.pow(beta_i).sum()).clamp(min=1e-6).pow(1.0 / beta_i)
        )
    alpha_init.clamp_(min=0.01)

    beta_perhkl = torch.full((n_valid,), default_beta)
    for sid, bval in shell_beta.items():
        m = hkl_shell_valid == sid
        if m.any():
            beta_perhkl[m] = bval

    hkl_valid = unique_hkl[valid_indices]
    d_valid = d_spacing_all[valid_indices]

    # ── French–Wilson prior metadata (per valid HKL) ──
    # Shell binning on d_spacing (rank-based, same recipe as hkl_shell above
    # but sized by config.fw_n_shells).
    rank_valid = d_valid.argsort().argsort()
    shell_id_fw = (
        rank_valid.float() / max(n_valid, 1) * config.fw_n_shells
    ).long().clamp(max=config.fw_n_shells - 1)
    # Centric classification via torchref symmetry ops.  Falls back to all-
    # acentric if space group is P1 or torchref cannot parse it (still
    # correct — P1 has no centric reflections).
    from torchref.base.french_wilson import is_centric_from_hkl
    try:
        sg = getattr(stream, "spacegroup", None) or "P 1"
        is_centric = is_centric_from_hkl(hkl_valid.long(), space_group=sg)
    except Exception as exc:  # noqa: BLE001 — surface the fallback for the user
        print(f"  [FW] centric detection failed ({exc!r}); treating all as acentric")
        is_centric = torch.zeros(n_valid, dtype=torch.bool)

    # ── Adjacency graph (±1 Miller) ──
    adjacency = build_adjacency(hkl_valid)

    # ── Per-obs metadata for corrections ──
    s_sq_per_obs = 1.0 / (4.0 * d_spacing_all[hkl_id_raw].clamp(min=0.1) ** 2)

    def _chain(attr: str):
        val = getattr(core, attr, None)
        if val is None:
            return None
        return val[keep_all][sort_all]

    cid = core.crystal_id[keep_all][sort_all].long()
    n_crystals = int(core.crystal_id.max().item()) + 1
    obs_info = ObsInfo(
        crystal_id=cid,
        s_sq=s_sq_per_obs[keep_all][sort_all],
        hkl_idx=hkl_id,
        n_crystals=n_crystals,
        panel_id=_chain("panel_id").long() if core.panel_id is not None else None,
        fs=_chain("fs").float() if core.fs is not None else None,
        ss=_chain("ss").float() if core.ss is not None else None,
        peak=_chain("peak").float() if core.peak is not None else None,
        background=_chain("background").float() if core.background is not None else None,
    )

    # ── Quadrature grid (Poisson detection floor per HKL, or override) ──
    n_ph = (I_obs + bg).clamp(min=0)
    with torch.no_grad():
        if config.sig_lo_override is not None:
            sig_lo_per_obs = torch.full(
                (n_obs_total,), float(config.sig_lo_override)
            )
        else:
            bg_sum = torch.zeros(n_valid).scatter_add_(0, hkl_id, bg.clamp(min=0))
            bg_per_hkl = (bg_sum / obs_counts.float().clamp(min=1)).clamp(min=1.0)
            sigma_floor = (bg_per_hkl / obs_counts.float().clamp(min=1)).sqrt()
            sig_lo_hkl = (sigma_floor * 0.5).clamp(min=0.01)
            sig_lo_per_obs = sig_lo_hkl[hkl_id]

    # Quadrature grid is built per-chunk inside UnifiedMergeLoss.data_nll so
    # we don't store a global (N_obs × n_quad) tensor that would otherwise
    # dominate GPU memory at large N.  The loss uses sig_lo_per_obs, n_ph,
    # and bg (all O(N_obs)) to rebuild grids on demand.
    del sig_lo_per_obs  # not stored; loss recomputes sig_lo identically

    # ── External partiality (sorted to match) ──
    partiality_sorted = None
    if partiality is not None:
        partiality_sorted = partiality[keep_all][sort_all]

    return PreparedObs(
        hkl_valid=hkl_valid,
        d_spacing=d_valid,
        n_valid=n_valid,
        I_obs=I_obs,
        hkl_id=hkl_id,
        bg=bg,
        obs_counts=obs_counts,
        n_obs_total=n_obs_total,
        mu_init=mu_init,
        alpha_init=alpha_init,
        beta_perhkl=beta_perhkl,
        adjacency=adjacency,
        obs_info=obs_info,
        partiality=partiality_sorted,
        shell_id_fw=shell_id_fw,
        is_centric=is_centric,
    )
