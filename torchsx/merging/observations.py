"""Observation preparation: stream → sorted tensors + initial estimates.

Ported from ``torchsx.merging_clean.observations`` with imports redirected
to the new ``torchsx.merging.hkl`` primitives.  ``ObsInfo`` lives
here too (it is per-observation *metadata*, not a correction-specific
construct — keeping it next to ``PreparedObs`` avoids a circular import
path once ``ScaleModel`` subclasses need it).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from torchsx.measurement import ResolvedMeasurement
from torchsx.merging.config import MergeConfig
from torchsx.merging.hkl import PoissonNoiseModel, compute_hkl_index

# Prepared caches must have passed this policy, even when settings are trusted.
# 2: the per-HKL initial parameters (mu/alpha/beta) and the Miller adjacency are
# no longer prepared -- the predicted scatter never read them.
OBSERVATION_FILTER_VERSION = 2


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
    #: The observation's ``PreparedObs.I_obs`` and ``PreparedObs.bg``, carried
    #: when the observations come from an on-disk store
    #: (:class:`~torchsx.merging.store.PreparedStore`), which has no
    #: whole-set tensors to address by ``obs_id``.
    I_obs: Tensor | None = None
    bg: Tensor | None = None

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
            I_obs=None if self.I_obs is None else self.I_obs[s:e],
            bg=None if self.bg is None else self.bg[s:e],
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
            I_obs=_g(self.I_obs),
            bg=_g(self.bg),
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
            I_obs=_m(self.I_obs),
            bg=_m(self.bg),
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
    k0 = ewald_k0(core.beam_energy)
    cid = core.crystal_id.long()
    # ``symmetrize()`` replaces the observed index by its ASU representative,
    # whose excitation error is not the one that was recorded -- the other mates
    # sit of order a reciprocal lattice spacing off the sphere.  The parser keeps
    # the measured index, so read it; a legacy cache that did not has to recover
    # it by trying every symmetry image and keeping the one nearest the sphere,
    # which only one of them can be near.
    observed = getattr(core, "observed_hkl", None)
    if observed is not None:
        hkl = observed.float()
        ops = torch.eye(3, dtype=hkl.dtype)[None]
    else:
        from torchsx.merging.priors.reciprocal_basis import _hkl_rotations
        hkl = core.hkl.float()
        ops = _hkl_rotations(getattr(core, "spacegroup", None)).to(hkl.dtype)
        ops = torch.cat([ops, -ops], dim=0)
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
            off, q = ewald_rows(h_op, A, Bv, Cv, k0)
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
        return offset, unit_rows(q_lab)
    return offset


def ewald_k0(beam_energy: Tensor) -> float:
    """``1/λ`` (Å⁻¹) at the median photon energy (eV) of the images."""
    wavelength = 12398.4 / float(beam_energy.float().median().item())
    return 1.0 / wavelength


def ewald_rows(hkl: Tensor, astar: Tensor, bstar: Tensor, cstar: Tensor, k0: float):
    """Signed Ewald offset and lab-frame ``q`` of float indices ``hkl`` (M, 3)
    against per-row reciprocal axes (M, 3) each."""
    q = hkl[:, 0:1] * astar + hkl[:, 1:2] * bstar + hkl[:, 2:3] * cstar
    q_plus = q.clone()
    q_plus[:, 2] += k0
    return q_plus.norm(dim=1) - k0, q


def unit_rows(q: Tensor) -> Tensor:
    return q / q.norm(dim=1, keepdim=True).clamp(min=1e-12)


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
            obs_info=self.obs_info.to(device),
            partiality=None if self.partiality is None else self.partiality.to(device),
            shell_id_fw=self.shell_id_fw.to(device),
            is_centric=self.is_centric.to(device),
            obs_weights=None if self.obs_weights is None else self.obs_weights.to(device),
        )


def resolve_gain(core, spec, resolved_measurement, config: MergeConfig,
                 poisson_noise: PoissonNoiseModel, calibration_sample) -> float:
    """The detector gain (ADU / photon) of a run, recorded on ``poisson_noise``.

    ``core`` supplies the images' ``beam_energy`` (and, for legacy inputs, the
    ``geometry``).  ``calibration_sample()`` returns ``(I, background,
    ewald_offset)`` in ADU over the observations, or over a subset holding at
    least every observation a calibration selects
    (:func:`~torchsx.merging.hkl.noise.calibration_candidates`); it is called only
    when the gain is calibrated.
    """
    from torchsx.measurement import gain_from_measurement, legacy_stream_gain, parse_gain

    gain_cfg = parse_gain(resolved_measurement.gain if resolved_measurement else config.gain)
    if isinstance(gain_cfg, (int, float)):
        gain = float(gain_cfg)
        gain_src = dict(resolved_measurement.provenance)["gain"] if resolved_measurement else "explicit"
    elif gain_cfg == "trust_input":
        mean_eV = float(core.beam_energy.float().mean().item())
        gain = gain_from_measurement(spec, mean_eV) if spec is not None else legacy_stream_gain(core)
        gain_src = "source measurement declaration" if spec is not None else "legacy geometry (unverified)"
    elif gain_cfg in ("auto", "calibrate"):
        import math
        tolerance = float(config.gain_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("gain_tolerance must be finite and nonnegative")
        declared, declaration_error = None, None
        try:
            declared = gain_from_measurement(spec, float(core.beam_energy.float().mean().item()))
        except ValueError as exc:
            declaration_error = str(exc)
        if declared is not None:
            poisson_noise.gain = declared
        cal_I, cal_background, cal_ewald = calibration_sample()
        g, mu_bias, read_var_adu_sq = poisson_noise.calibrate_sigma2_of_bg(
            cal_I, cal_background, cal_ewald,
        )
        if not (poisson_noise.gain_calibration_info.get("valid", True)
                and g is not None and math.isfinite(g) and g > 0):
            # No off-Bragg sample (CrystFEL streams predict only near the Ewald
            # sphere): the truncated normal on the negative sums.  Biased (see its
            # docstring), so it decides a wrong declaration only past the tolerance.
            off_bragg = poisson_noise.gain_calibration_info
            g, mu_bias = poisson_noise.calibrate_gain_truncnorm(cal_I, cal_background)
            read_var_adu_sq = 0.0
            poisson_noise.gain_calibration_info = {
                **poisson_noise.gain_calibration_info,
                "fallback_reason": off_bragg.get("reason", "invalid off-Bragg calibration"),
                "off_bragg": off_bragg}
        calibrated = float(g) if g is not None and math.isfinite(g) and g > 0 else None
        info = poisson_noise.gain_calibration_info
        valid_calibration = calibrated is not None and info.get("valid", True)
        difference = abs(calibrated - declared) / declared if declared is not None and calibrated is not None else None
        if gain_cfg == "calibrate" and not valid_calibration:
            raise ValueError(f"Gain calibration unavailable; supply an explicit --gain or use trust_input. {info}")
        if valid_calibration and (gain_cfg == "calibrate" or declared is None or difference > tolerance):
            gain = calibrated
            reason = ("calibration_requested" if gain_cfg == "calibrate" else
                      "declaration_unavailable" if declared is None else "disagreement_exceeds_tolerance")
            selected = "calibration"
        elif declared is not None:
            gain = declared
            reason = "within_tolerance" if valid_calibration else "calibration_unavailable"
            selected = "input"
        else:
            raise ValueError("No usable stream gain or gain calibration; supply an explicit --gain. "
                             f"Stream: {declaration_error}. Calibration: {info}")
        poisson_noise.gain_selection = dict(mode=gain_cfg, declared_gain=declared,
            calibrated_gain=calibrated, calibration_valid=bool(valid_calibration),
            relative_difference=difference, tolerance=tolerance, selected=selected,
            reason=reason, selected_gain=gain, declaration_error=declaration_error,
            calibration={**info, "mu_bias": float(mu_bias) if math.isfinite(mu_bias) else None,
                         "read_var_adu_sq": float(read_var_adu_sq) if math.isfinite(read_var_adu_sq) else None})
        gain_src = f"{gain_cfg}: {selected}, {reason}; declared={declared}, calibrated={calibrated}"
    else:
        raise ValueError(f"Unknown config.gain: {gain_cfg!r}")
    print(f"  gain = {gain:.4g} ADU/photon  [{gain_src}]", flush=True)
    poisson_noise.gain = gain
    if gain_cfg not in ("auto", "calibrate"):
        poisson_noise.gain_selection = dict(mode=gain_cfg, selected_gain=gain, reason=gain_src)
    return gain


def fw_shell_ids(d_valid: Tensor, n_shells: int) -> Tensor:
    """Equal-population resolution shell of each reflection (low d last)."""
    n_valid = d_valid.numel()
    rank_valid = d_valid.argsort().argsort()
    return (rank_valid.float() / max(n_valid, 1) * n_shells).long().clamp(max=n_shells - 1)


def centric_flags(spacegroup: str | None, hkl_valid: Tensor) -> Tensor:
    """Centric flag per reflection; all acentric (with a warning) for P 1 or no group."""
    import warnings
    from torchref.symmetry import SpaceGroup

    n_valid = hkl_valid.shape[0]
    if not spacegroup or spacegroup == "P 1":
        warnings.warn(
            f"prepare_observations: stream.spacegroup={spacegroup!r}; centric "
            f"detection will be no-op (all reflections marked acentric). "
            f"Set the spacegroup on the stream before merging.",
            stacklevel=3,
        )
        return torch.zeros(n_valid, dtype=torch.bool)
    try:
        return SpaceGroup(spacegroup).is_centric(hkl_valid.long())
    except Exception as exc:  # noqa: BLE001 — surface the fallback
        print(f"  [FW] centric detection failed ({exc!r}); treating all as acentric")
        return torch.zeros(n_valid, dtype=torch.bool)


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
    from torchsx.measurement import resolve_measurement, validate_measurement

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
    # Reject unusable measurements before calibration, pooling or feature
    # construction: one NaN sigma can contaminate every prediction in a crystal.
    # Negative intensities remain valid measurements. Missing sigma retains the
    # existing noise-model fallback; supplied sigma must be finite and positive.
    valid = torch.ones_like(core.I, dtype=torch.bool)
    rejected = {}
    for name, values in (("I", core.I), ("background", core.background),
                         ("sigma", getattr(core, "sigma", None))):
        if values is None:
            continue
        bad = ~torch.isfinite(values)
        if name == "sigma":
            bad |= values <= 0
        rejected[name] = int(bad.sum().item())
        valid &= ~bad
    n_rejected = int((~valid).sum().item())
    if n_rejected:
        reasons = ", ".join(f"{name}={n:,}" for name, n in rejected.items() if n)
        print(f"  observation filter: removed {n_rejected:,}/{valid.numel():,} "
              f"invalid measurements ({reasons}; reason counts may overlap)", flush=True)
    if not valid.any():
        raise ValueError("No valid observations remain after measurement filtering")
    if n_rejected:
        # Subset all observation columns together, retaining parent IDs/tables
        # so crystal weights, split labels and timepoints keep their alignment.
        core = core.select_cascade({"obs": valid})
        if partiality is not None:
            partiality = partiality[valid]
    core_I = core.I
    core_background = core.background

    # Phase timing, off unless TORCHSX_PREP_TIMING is set: on a large data set
    # this function is the slow front end of every merge and the cost is not
    # where one guesses.
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
    gain = resolve_gain(core, spec, resolved_measurement, config, poisson_noise,
                        lambda: (core_I, core_background, ewald_offset_core))

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

    d_spacing_all = get_d_spacing(unique_hkl.float(), median_cell)

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

    hkl_valid = unique_hkl[valid_indices]
    d_valid = d_spacing_all[valid_indices]

    # ── French–Wilson prior metadata (per valid HKL) ──
    shell_id_fw = fw_shell_ids(d_valid, config.fw_n_shells)
    is_centric = centric_flags(getattr(stream, "spacegroup", None), hkl_valid)

    _lap("centric flags")

    # ── Per-obs metadata for corrections ──
    s_sq_per_obs = 1.0 / (4.0 * d_spacing_all[hkl_id_raw].clamp(min=0.1) ** 2)

    def _chain(attr: str):
        val = getattr(core, attr, None)
        if val is None:
            return None
        return val[keep_all][sort_all]

    cid = core.crystal_id[keep_all][sort_all].long()
    n_crystals = core.n_crystals

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
        obs_info=obs_info,
        partiality=partiality_sorted,
        shell_id_fw=shell_id_fw,
        is_centric=is_centric,
        obs_weights=obs_weights_sorted,
    )
