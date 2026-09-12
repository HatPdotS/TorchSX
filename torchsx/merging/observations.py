"""Observation preparation: stream → sorted tensors + initial estimates.

Ported from ``torchsx.merging_clean.observations`` with imports redirected
to the new ``torchsx.merging.hkl`` primitives.  ``ObsInfo`` lives
here too (it is per-observation *metadata*, not a correction-specific
construct — keeping it next to ``PreparedObs`` avoids a circular import
path once ``ScaleModel`` subclasses need it).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from torchsx.measurement import ResolvedMeasurement
from torchsx.merging.config import MergeConfig
from torchsx.merging.hkl import (
    Adjacency,
    PoissonNoiseModel,
    build_adjacency,
    compute_hkl_index,
    fit_shell_gengauss_gauss,
)


# ── Per-observation metadata ───────────────────────────────────────────


@dataclass
class ObsInfo:
    """Per-observation metadata, sorted identically to ``PreparedObs`` tensors.

    ScaleModel subclasses index into this to compute their per-observation
    log-correction without having to stash extra tensors.  Optional fields
    stay ``None`` when the stream doesn't carry the corresponding channel.

    Kinetic-ensemble fields (``delay_bin_idx``, ``unique_timepoints``) are
    populated by :func:`prepare_observations` when the stream carries
    image-level ``timepoint`` metadata (see
    :meth:`SFXStream.assign_timepoints`); otherwise they stay ``None``.

    ``q_dir`` is the **lab-frame** unit scattering vector.  Direction is what
    makes an orientation error expressible: if the refined orientation is wrong
    by a small rotation ``delta``, the induced change in excitation error is
    ``(delta x q) . s_hat`` — a function of *where* the reflection sits, not
    just how far out.  With only ``s_sq`` (a magnitude) and a scalar
    ``ewald_offset``, "reflections on this side of the pattern run high" is not
    representable at all, so the peers cannot convey an orientation error and
    the target reflection could not apply one.  Lab frame specifically: the
    crystal-frame direction is just ``hkl`` and would be degenerate with I0,
    whereas the lab-frame direction varies with each crystal's orientation.

    ``obs_id`` is each observation's row index in the *full* (unsliced) list.
    :meth:`slice` gives contiguous chunks, for which the caller's
    ``obs_range=(s, e)`` is enough to index a per-observation side-channel; but
    crystal-batched training gathers **non-contiguous** subsets, where
    ``obs_range`` is meaningless.  Carrying ``obs_id`` lets any per-observation
    cache — predicted scatter, ``FixedScale``, an EMA — be indexed correctly
    under an arbitrary gather.
    """

    crystal_id: Tensor
    s_sq: Tensor
    hkl_idx: Tensor
    n_crystals: int
    panel_id: Tensor | None = None
    fs: Tensor | None = None
    ss: Tensor | None = None
    peak: Tensor | None = None
    background: Tensor | None = None
    ewald_offset: Tensor | None = None
    sigma: Tensor | None = None          # CrystFEL σ(I), same units as I_obs
    q_dir: Tensor | None = None          # (n, 3) unit scattering vector, LAB frame
    beam_energy: Tensor | None = None    # (n,) per-image photon energy (eV)
    crystal_res: Tensor | None = None    # (n,) per-crystal diffraction limit
    #: Per-CRYSTAL metadata, ``(n_crystals, len(CRYSTAL_FEAT_COLUMNS))``, indexed
    #: by ``crystal_id`` -- the full table is kept under slice/gather, like
    #: ``delay_bin_idx``.  Consumed by the crystal-level input head of the scale
    #: encoder; carries no intensity.
    crystal_feat: Tensor | None = None
    obs_id: Tensor | None = None         # (n,) int64 — row index in the FULL obs list
    # Kinetic-ensemble plumbing — both populated together or both ``None``.
    delay_bin_idx: Tensor | None = None     # (n_crystals,) int64, indexes unique_timepoints
    unique_timepoints: Tensor | None = None  # (n_timepoints,) float64, sorted

    def slice(self, s: int, e: int) -> "ObsInfo":
        """Restrict to ``obs[s:e]`` (for chunked closures)."""
        return ObsInfo(
            crystal_id=self.crystal_id[s:e],
            s_sq=self.s_sq[s:e],
            hkl_idx=self.hkl_idx[s:e],
            n_crystals=self.n_crystals,
            panel_id=None if self.panel_id is None else self.panel_id[s:e],
            fs=None if self.fs is None else self.fs[s:e],
            ss=None if self.ss is None else self.ss[s:e],
            peak=None if self.peak is None else self.peak[s:e],
            background=None if self.background is None else self.background[s:e],
            ewald_offset=None if self.ewald_offset is None else self.ewald_offset[s:e],
            sigma=None if self.sigma is None else self.sigma[s:e],
            crystal_res=None if self.crystal_res is None else self.crystal_res[s:e],
            beam_energy=None if self.beam_energy is None else self.beam_energy[s:e],
            q_dir=None if self.q_dir is None else self.q_dir[s:e],
            crystal_feat=self.crystal_feat,              # per-crystal — full table kept
            obs_id=None if self.obs_id is None else self.obs_id[s:e],
            delay_bin_idx=self.delay_bin_idx,            # per-crystal — full table kept
            unique_timepoints=self.unique_timepoints,
        )

    def gather(self, idx: Tensor) -> "ObsInfo":
        """Restrict to an **arbitrary** (possibly non-contiguous) row selection.

        The crystal-batched counterpart of :meth:`slice`.  ``obs_id`` is carried
        through so downstream per-observation lookups still resolve against the
        full observation list.
        """
        def _g(t):
            return None if t is None else t[idx]
        if self.obs_id is None:
            raise ValueError(
                "ObsInfo.gather requires obs_id; build via prepare_observations "
                "or set obs_id=torch.arange(n_obs) explicitly."
            )
        return ObsInfo(
            crystal_id=self.crystal_id[idx],
            s_sq=self.s_sq[idx],
            hkl_idx=self.hkl_idx[idx],
            n_crystals=self.n_crystals,
            panel_id=_g(self.panel_id),
            fs=_g(self.fs),
            ss=_g(self.ss),
            peak=_g(self.peak),
            background=_g(self.background),
            ewald_offset=_g(self.ewald_offset),
            sigma=_g(self.sigma),
            q_dir=_g(self.q_dir),
            beam_energy=_g(self.beam_energy),
            crystal_res=_g(self.crystal_res),
            crystal_feat=self.crystal_feat,
            obs_id=self.obs_id[idx],
            delay_bin_idx=self.delay_bin_idx,
            unique_timepoints=self.unique_timepoints,
        )

    def to(self, device) -> "ObsInfo":
        def _m(t):
            return None if t is None else t.to(device)
        return ObsInfo(
            crystal_id=self.crystal_id.to(device),
            s_sq=self.s_sq.to(device),
            hkl_idx=self.hkl_idx.to(device),
            n_crystals=self.n_crystals,
            panel_id=_m(self.panel_id),
            fs=_m(self.fs),
            ss=_m(self.ss),
            peak=_m(self.peak),
            background=_m(self.background),
            ewald_offset=_m(self.ewald_offset),
            sigma=_m(self.sigma),
            q_dir=_m(self.q_dir),
            beam_energy=_m(self.beam_energy),
            crystal_res=_m(self.crystal_res),
            crystal_feat=_m(self.crystal_feat),
            obs_id=_m(self.obs_id),
            delay_bin_idx=_m(self.delay_bin_idx),
            unique_timepoints=_m(self.unique_timepoints),
        )


#: Columns of ``ObsInfo.crystal_feat``, in order.  One row per crystal; every
#: column is metadata (no intensity), so the scale encoder may read it for a
#: crystal without any leave-one-out.  A column the stream does not carry is
#: emitted as zeros, which the feature standardisation then treats as constant.
CRYSTAL_FEAT_COLUMNS: tuple[str, ...] = (
    "cell_a", "cell_b", "cell_c", "cell_alpha", "cell_beta", "cell_gamma",
    "beam_energy",      # per image, broadcast to the crystal
    "crystal_res",      # diffraction limit
    "log_n_refl",       # log1p(indexed reflections)
    "det_shift_x", "det_shift_y",
    "n_peaks",
)


def _crystal_feature_table(core, n_crystals: int) -> Tensor:
    """``(n_crystals, len(CRYSTAL_FEAT_COLUMNS))`` float32 from the stream core."""
    def _col(attr: str, level: str):
        val = getattr(core, attr, None)
        if val is None:
            return None
        if level == "image":
            if not hasattr(core, "broadcast"):
                return None
            val = core.broadcast(attr, "image", "crystal")
        val = torch.as_tensor(val)
        if val.shape[0] != n_crystals:
            return None
        return val.float()

    cols: list[Tensor] = []
    cell = _col("cell", "crystal")
    if cell is not None and cell.ndim == 2 and cell.shape[1] == 6:
        cols.extend(cell[:, i] for i in range(6))
    else:
        cols.extend(torch.zeros(n_crystals) for _ in range(6))
    for attr, level, tf in (
        ("beam_energy", "image", None),
        ("resolution", "crystal", None),
        ("n_refl", "crystal", lambda t: torch.log1p(t.clamp(min=0))),
        ("det_shift_x", "image", None),
        ("det_shift_y", "image", None),
        ("n_peaks", "image", None),
    ):
        v = _col(attr, level)
        if v is None:
            v = torch.zeros(n_crystals)
        elif tf is not None:
            v = tf(v)
        cols.append(v.reshape(n_crystals))
    table = torch.stack(cols, dim=1)
    assert table.shape[1] == len(CRYSTAL_FEAT_COLUMNS)
    return torch.nan_to_num(table, nan=0.0, posinf=0.0, neginf=0.0)


def _build_delay_bins(stream, n_crystals: int) -> tuple[Tensor | None, Tensor | None]:
    """Build per-crystal ``delay_bin_idx`` + the unique-timepoint table.

    Returns ``(None, None)`` if the stream has no image-level
    ``timepoint`` metadata (i.e. :meth:`SFXStream.assign_timepoints`
    was never called).  Otherwise returns

      * ``delay_bin_idx`` — ``(n_crystals,) int64`` index into
        ``unique_timepoints`` for each crystal.
      * ``unique_timepoints`` — ``(n_t,) float64`` sorted unique
        non-NaN timepoints observed across labelled images.

    Crystals whose image has a NaN timepoint get bin index ``-1``
    (sentinel — callers must mask before indexing).
    """
    image_tp = []
    meta = stream._core.metadata if hasattr(stream, "_core") else None
    if meta and "image" in meta:
        image_tp = meta["image"].get("timepoint", []) or []
    if not image_tp:
        return None, None

    image_tp_t = torch.tensor(image_tp, dtype=torch.float64)
    if image_tp_t.numel() != stream._core.n_images:
        # Mismatched length: refuse to silently mis-align.
        raise ValueError(
            f"metadata['image']['timepoint'] has {image_tp_t.numel()} entries "
            f"but stream has {stream._core.n_images} images"
        )

    per_crystal_tp = image_tp_t[stream._core.image_id.long()]    # (n_crystals,)
    finite = ~torch.isnan(per_crystal_tp)
    if not finite.any():
        return None, None

    uniq_vals = torch.unique(per_crystal_tp[finite])             # already sorted
    uniq_vals = torch.sort(uniq_vals).values                     # explicit, safe

    # searchsorted gives the bin index for finite timepoints; NaNs get -1.
    bin_idx = torch.full((n_crystals,), -1, dtype=torch.int64)
    if finite.any():
        bin_idx[finite] = torch.searchsorted(uniq_vals, per_crystal_tp[finite])
    return bin_idx, uniq_vals


def _ewald_offset(core, return_q: bool = False):
    """Per-observation signed Ewald sphere offset ``‖q+s₀‖ − 1/λ`` (Å⁻¹).

    Physically-motivated proxy for predicted partiality: large ``|r|`` →
    predicted off-Bragg → signal ≈ 0.  Used by the σ²(bg) calibration to
    select a clean noise-dominated sample and fed to the scale model as a
    feature.  Evaluated over the symmetry images of a canonicalised index,
    see below.

    Wavelength is taken as the stream-median (per-image variation at
    LCLS/EuXFEL is < 0.01 %, well below the |r| binning resolution).
    """
    wavelength = 12398.4 / float(core.beam_energy.float().median().item())
    k0 = 1.0 / wavelength
    cid = core.crystal_id.long()
    hkl = core.hkl.float()
    # ``symmetrize()`` replaces each observed Miller index by its ASU
    # representative: a symmetry mate of the reflection that was measured.  The
    # excitation error of a mate is not the one that was recorded -- the other
    # mates sit of order a reciprocal lattice spacing away from the sphere --
    # so evaluate every point-group image (and its Friedel partner) and keep
    # the one nearest the sphere.  Only one image can be near diffracting
    # condition, which makes the choice unambiguous.  For a stream that was
    # not symmetrised the observed index is the right one and only the
    # identity is tried.
    if bool(getattr(core, "friedel_merged", True)):
        from torchsx.merging.priors.reciprocal_basis import _hkl_rotations
        ops = _hkl_rotations(getattr(core, "spacegroup", None)).to(hkl.dtype)
        ops = torch.cat([ops, -ops], dim=0)
    else:
        ops = torch.eye(3, dtype=hkl.dtype)[None]
    best_offset = None
    best_q = None
    chunk = 8_000_000
    parts_offset, parts_q = [], []
    for start in range(0, hkl.shape[0], chunk):
        h_c = hkl[start:start + chunk]
        c_c = cid[start:start + chunk]
        A, Bv, Cv = core.astar[c_c], core.bstar[c_c], core.cstar[c_c]
        best_offset = best_q = None
        for R in ops:
            h_op = h_c @ R                                         # (M, 3), row convention h' = h @ R
            q = h_op[:, 0:1] * A + h_op[:, 1:2] * Bv + h_op[:, 2:3] * Cv
            q_plus = q.clone()
            q_plus[:, 2] += k0
            off = q_plus.norm(dim=1) - k0
            if best_offset is None:
                best_offset, best_q = off, q
            else:
                better = off.abs() < best_offset.abs()
                best_offset = torch.where(better, off, best_offset)
                best_q = torch.where(better[:, None], q, best_q)
        parts_offset.append(best_offset)
        parts_q.append(best_q)
    offset = torch.cat(parts_offset)
    q_lab = torch.cat(parts_q)
    if return_q:
        return offset, q_lab / q_lab.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return offset


# ── Prepared-observation bundle ────────────────────────────────────────


@dataclass
class PreparedObs:
    """Everything a merging loss needs as non-parameter state."""

    # Valid-HKL metadata
    hkl_valid: Tensor
    d_spacing: Tensor
    n_valid: int
    median_cell: Tensor

    # Sorted observations (all obs, positive + negative)
    I_obs: Tensor
    hkl_id: Tensor
    bg: Tensor
    obs_counts: Tensor
    n_obs_total: int

    # Initial per-HKL parameter values
    mu_init: Tensor
    alpha_init: Tensor
    beta_perhkl: Tensor

    # Miller-graph adjacency
    adjacency: Adjacency

    # Per-observation metadata for scale corrections
    obs_info: ObsInfo

    # External partiality (sorted), or None
    partiality: Tensor | None

    # French–Wilson prior inputs (per valid HKL)
    shell_id_fw: Tensor
    is_centric: Tensor

    # Per-observation likelihood weight (sorted), or None.
    # Multiplies ``log p`` per observation in the data NLL; the default
    # ``None`` recovers the unweighted ``-Σ log p`` sum (bit-for-bit).
    # Was used by the time-binned merge to apply a Gaussian-over-crystal-rank
    # weighting; that module is gone, but the hook is kept for any weighting
    # scheme (the ``timepoint`` metadata it keyed off still exists).
    obs_weights: Tensor | None = None

    def to(self, device) -> "PreparedObs":
        return PreparedObs(
            hkl_valid=self.hkl_valid.to(device),
            d_spacing=self.d_spacing.to(device),
            n_valid=self.n_valid,
            median_cell=self.median_cell.to(device),
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
            obs_weights=None if self.obs_weights is None else self.obs_weights.to(device),
        )


# ── prepare_observations ────────────────────────────────────────────────


def prepare_observations(
    stream,
    config: MergeConfig,
    *,
    partiality: Tensor | None = None,
    poisson_noise: PoissonNoiseModel | None = None,
    crystal_weights: Tensor | None = None,
    resolved_measurement: ResolvedMeasurement | None = None,
) -> PreparedObs:
    """Stream → PreparedObs (CPU).  Call ``.to(device)`` afterwards.

    ``crystal_weights`` is an optional ``(n_crystals,)`` float tensor that
    will be broadcast to per-observation weights via
    ``crystal_weights[obs_info.crystal_id]`` and stored on
    :attr:`PreparedObs.obs_weights`.  Multiplies ``log p`` per
    observation in the data NLL.  Default ``None`` → no weighting,
    bit-for-bit identical to the unweighted pipeline.
    """
    from torchref.base.reciprocal.hkl import get_d_spacing
    from torchsx.measurement import (
        gain_from_measurement, legacy_stream_gain, parse_gain,
        resolve_measurement, validate_measurement,
    )

    spec = stream.measurement
    if resolved_measurement is not None:
        validate_measurement(spec, explicit_rings=dict(resolved_measurement.provenance)["radii"] != "source")
    if resolved_measurement is None and spec is not None:
        radii = (None if poisson_noise is None else
                 (poisson_noise.r_peak, poisson_noise.r_gap, poisson_noise.r_bg))
        resolved_measurement = resolve_measurement(spec, radii=radii, gain=config.gain)
    if poisson_noise is None:
        radii = resolved_measurement.radii if resolved_measurement else (7.0, 9.0, 11.0)
        poisson_noise = PoissonNoiseModel(r_peak=radii[0], r_gap=radii[1], r_bg=radii[2])

    core = stream._core
    core_I = core.I
    core_background = core.background

    # Phase timing, off unless TORCHSX_PREP_TIMING is set: this function is the
    # 50-minute front end of every merge and the cost is not where one guesses.
    import os as _os
    import time as _time
    _t_prev = _time.perf_counter()
    _timing = bool(_os.environ.get("TORCHSX_PREP_TIMING"))

    def _lap(what: str):
        nonlocal _t_prev
        now = _time.perf_counter()
        if _timing:
            print(f"  [prep] {what:<28s} {now - _t_prev:8.1f} s", flush=True)
        _t_prev = now

    # Per-observation signed Ewald sphere offset (Å⁻¹). Used by the σ²(bg)
    # gain calibration to gate to off-Bragg (predicted-zero-signal) obs,
    # and propagated to ObsInfo for downstream use.
    ewald_offset_core, q_dir_core = _ewald_offset(core, return_q=True)
    _lap("ewald offset")

    # ── Detector gain (ADU / photon) ──
    gain_cfg = parse_gain(resolved_measurement.gain if resolved_measurement else config.gain)
    if isinstance(gain_cfg, (int, float)):
        gain = float(gain_cfg)
        gain_src = dict(resolved_measurement.provenance)["gain"] if resolved_measurement else "explicit"
    elif gain_cfg == "stream":
        mean_eV = float(core.beam_energy.float().mean().item())
        gain = gain_from_measurement(spec, mean_eV) if spec is not None else legacy_stream_gain(core)
        gain_src = "source measurement declaration" if spec is not None else "legacy geometry (unverified)"
    elif gain_cfg == "empirical":
        g = poisson_noise.calibrate_gain(core_I, core_background)
        gain = float(g) if g and g > 0 else 1.0
        gain_src = "empirical (negatives-Mandel; biased high)"
    elif gain_cfg == "auto":
        g, mu_bias, read_var_adu_sq = poisson_noise.calibrate_sigma2_of_bg(
            core_I, core_background, ewald_offset_core,
        )
        gain = float(g) if g and g > 0 else 1.0
        gain_src = (
            f"auto (σ²(bg) off-Bragg, μ_z={mu_bias:+.3f}σ, "
            f"read_var_ADU²={read_var_adu_sq:.4g})"
        )
    else:
        raise ValueError(f"Unknown config.gain: {gain_cfg!r}")
    print(f"  gain = {gain:.4g} ADU/photon  [{gain_src}]", flush=True)
    poisson_noise.gain = gain

    # ADU → photon counts for the Poisson likelihood.
    core_I = core_I / gain
    core_background = core_background / gain
    hkl_idx = compute_hkl_index(core.hkl)
    _lap("hkl index")
    hkl_id_raw = hkl_idx.hkl_id
    unique_hkl = hkl_idx.unique_hkl
    n_unique = hkl_idx.n_unique
    I_obs_stream = core_I
    median_cell = core.cell.median(dim=0).values

    # ── Per-HKL statistics (for shell β fit / SNR) ──
    counts = torch.bincount(hkl_id_raw, minlength=n_unique)
    order = hkl_id_raw.argsort()
    groups_I = I_obs_stream[order].split(counts.tolist())
    _lap("argsort + group split")

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
    _lap("per-HKL mean/var loop")
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
        if config.beta_pool_cap:
            # Take an evenly spaced subset rather than the first N: the first N
            # would be the shell's lowest Miller indices.  A stride keeps the
            # indices ASCENDING, so the reads into groups_I stay sequential --
            # a shuffle costs more in lost locality than the cap saves when the
            # data is small enough that the cap never bites.
            n_obs_shell = int(counts[hkl_indices].sum())
            stride = max(1, -(-n_obs_shell // config.beta_pool_cap))
            hkl_indices = hkl_indices[::stride]
        pooled = []
        n_pooled = 0
        for h_idx in hkl_indices.tolist():
            g = groups_I[h_idx]
            pos = g[g > 0]
            if len(pos) < 5:
                continue
            mu_h = pos.mean()
            if mu_h > 0:
                pooled.append(pos / mu_h)
                n_pooled += pos.numel()
                if config.beta_pool_cap and n_pooled >= config.beta_pool_cap:
                    break
        if len(pooled) < 5:
            continue
        pooled_t = torch.cat(pooled)
        pooled_t = pooled_t[pooled_t > 0]
        if len(pooled_t) < 500:
            continue
        result = fit_shell_gengauss_gauss(torch.log(pooled_t))
        if result is not None:
            shell_beta[sid] = result[0]

    # Smooth β across shells (simple Gaussian kernel over shell index).
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
    _lap("shell beta fit")

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
    _lap("valid filter + argsort")

    I_obs = I_obs_stream[keep_all][sort_all]
    hkl_id = hkl_fit_idx[sort_all]
    obs_counts = torch.bincount(hkl_id, minlength=n_valid)
    n_obs_total = int(I_obs.shape[0])

    bg_per_pixel = core_background[keep_all][sort_all]
    bg = bg_per_pixel * poisson_noise.n_peak
    _lap("gather I/bg")

    # ── Initial log-space MLE for μ and α (positive obs per HKL) ──
    pos_in_sorted = I_obs > 0
    hkl_id_pos_sorted = hkl_id[pos_in_sorted]
    log_I_pos = torch.log(I_obs[pos_in_sorted].clamp(min=1e-8))
    obs_counts_pos = torch.bincount(hkl_id_pos_sorted, minlength=n_valid)
    offsets_pos = torch.zeros(n_valid + 1, dtype=torch.long)
    offsets_pos[1:] = obs_counts_pos.cumsum(0)

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
    _lap("mu/alpha loop")

    beta_perhkl = torch.full((n_valid,), default_beta)
    for sid, bval in shell_beta.items():
        m = hkl_shell_valid == sid
        if m.any():
            beta_perhkl[m] = bval

    hkl_valid = unique_hkl[valid_indices]
    d_valid = d_spacing_all[valid_indices]

    # ── French–Wilson prior metadata (per valid HKL) ──
    rank_valid = d_valid.argsort().argsort()
    shell_id_fw = (
        rank_valid.float() / max(n_valid, 1) * config.fw_n_shells
    ).long().clamp(max=config.fw_n_shells - 1)
    import warnings
    from torchref.symmetry import SpaceGroup
    sg = getattr(stream, "spacegroup", None)
    if not sg or sg == "P 1":
        warnings.warn(
            f"prepare_observations: stream.spacegroup={sg!r}; centric "
            f"detection will be no-op (all reflections marked acentric). "
            f"Set the spacegroup on the stream before merging.",
            stacklevel=2,
        )
        is_centric = torch.zeros(n_valid, dtype=torch.bool)
    else:
        try:
            is_centric = SpaceGroup(sg).is_centric(hkl_valid.long())
        except Exception as exc:  # noqa: BLE001 — surface the fallback
            print(f"  [FW] centric detection failed ({exc!r}); treating all as acentric")
            is_centric = torch.zeros(n_valid, dtype=torch.bool)

    # ── Adjacency graph (±1 Miller) ──
    adjacency = build_adjacency(hkl_valid)
    _lap("adjacency")

    # ── Per-obs metadata for corrections ──
    s_sq_per_obs = 1.0 / (4.0 * d_spacing_all[hkl_id_raw].clamp(min=0.1) ** 2)

    def _chain(attr: str):
        val = getattr(core, attr, None)
        if val is None:
            return None
        return val[keep_all][sort_all]

    cid = core.crystal_id[keep_all][sort_all].long()
    n_crystals = int(core.crystal_id.max().item()) + 1

    # ── Kinetic-ensemble metadata (only when the stream is labelled) ──
    delay_bin_idx, unique_t_tensor = _build_delay_bins(stream, n_crystals)

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
        ewald_offset=ewald_offset_core[keep_all][sort_all].float(),
        # CrystFEL σ(I), carried in the same photon units as I_obs (both are
        # divided by the detector gain above).  Consumed by the agreement /
        # marginal objective and by a supplied-σ likelihood.
        sigma=(
            (core.sigma / gain)[keep_all][sort_all].float()
            if getattr(core, "sigma", None) is not None else None
        ),
        q_dir=q_dir_core[keep_all][sort_all].float(),
        beam_energy=(
            core.broadcast("beam_energy", "image", "obs")[keep_all][sort_all].float()
            if hasattr(core, "broadcast") else None
        ),
        crystal_res=(
            core.resolution[cid].float() if getattr(core, "resolution", None) is not None
            else None
        ),
        crystal_feat=_crystal_feature_table(core, n_crystals),
        obs_id=torch.arange(n_obs_total, dtype=torch.long),
        delay_bin_idx=delay_bin_idx,
        unique_timepoints=unique_t_tensor,
    )

    _lap("obs_info gathers")

    # ── External partiality (sorted to match) ──
    partiality_sorted = None
    if partiality is not None:
        partiality_sorted = partiality[keep_all][sort_all]

    # ── Per-observation weights (broadcast from per-crystal, then sorted) ──
    obs_weights_sorted = None
    if crystal_weights is not None:
        if crystal_weights.ndim != 1 or crystal_weights.shape[0] != n_crystals:
            raise ValueError(
                f"crystal_weights must be 1-D with length n_crystals={n_crystals}, "
                f"got shape {tuple(crystal_weights.shape)}"
            )
        # Broadcast per-crystal weight to all (pre-filter) observations via
        # the stream's crystal_id, then apply the same keep/sort permutation
        # used above to align with ``I_obs`` / ``hkl_id`` / ``obs_info``.
        per_obs_w = crystal_weights[core.crystal_id.long()].to(torch.float32)
        obs_weights_sorted = per_obs_w[keep_all][sort_all]

    return PreparedObs(
        hkl_valid=hkl_valid,
        d_spacing=d_valid,
        n_valid=n_valid,
        median_cell=median_cell,
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
        obs_weights=obs_weights_sorted,
    )
