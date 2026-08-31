"""Classic time-binned merge with Gaussian-in-crystal-rank weighting.

A model-free time-resolved merge: split the dataset into a sequence of
overlapping bins along the temporal axis and run the standard merging
pipeline on each.  One MTZ per bin, ready for phenix / map calculation.

Bins are defined by a **Gaussian sliding window in crystal rank**
(rank = position in the time-sorted crystal index), *not* in absolute
time.  Each bin contains the same effective number of crystals
(≈ √(2π)·σ_rank) regardless of how densely the experiment sampled
different times, so data quality stays roughly constant across bins.
Where temporal density is high the bin spans a narrow time range; where
it's sparse the bin widens automatically.

Pipeline per bin
----------------

::

    w[c] = exp(-½ (rank[c] - center_rank_b)² / σ_rank²)            # Gaussian
    w[c] = 0  if |rank[c] - center_rank_b| > truncate_σ · σ_rank   # support

    sub = stream.filter_crystals(w > 0)
    loss = sub.make_loss(crystal_weights=w[w > 0], …)
    loss.refine(...)
    data = loss.posterior('flat').integrate()
    data.write_mtz(path_b)

The ``crystal_weights`` argument multiplies ``log p`` per observation in
the data NLL (see :class:`torchsx.merging.targets.data.DataNLLTarget`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import math

import torch
from torch import Tensor


# ── Bin specification ──────────────────────────────────────────────────


@dataclass
class GaussianRankBins:
    """A sequence of overlapping Gaussian-in-rank bins.

    Attributes
    ----------
    centers_rank :
        ``(n_bins,)`` int64 — crystal rank (in time-sorted order) at the
        Gaussian peak of each bin.
    sigma_rank :
        Gaussian σ in rank-space (shared across bins).
    truncate_sigma :
        Zero weights beyond this many σ from the center.
    centers_time :
        ``(n_bins,)`` float64 — weighted-mean timepoint of each bin (the
        natural human-readable label).
    effective_n :
        ``(n_bins,)`` float — ``Σ_c w_b[c]``: the effective number of
        crystals contributing to each bin.  Bins near the rank edges
        have a smaller effective n because their Gaussian is truncated.
    crystal_rank :
        ``(n_crystals,)`` int64 — temporal rank of each crystal (or -1
        for crystals with NaN timepoint, which receive zero weight in
        every bin).  Kept on the spec so per-bin weights can be
        recomputed without reshuffling.
    """

    centers_rank: Tensor
    sigma_rank: float
    truncate_sigma: float
    centers_time: Tensor
    effective_n: Tensor
    crystal_rank: Tensor

    @property
    def n_bins(self) -> int:
        return int(self.centers_rank.numel())

    def weights(self, bin_idx: int) -> Tensor:
        """``(n_crystals,)`` Gaussian weights for ``bin_idx``."""
        c = float(self.centers_rank[bin_idx].item())
        delta = self.crystal_rank.to(torch.float64) - c
        sig = float(self.sigma_rank)
        w = torch.exp(-0.5 * (delta / sig) ** 2)
        # Truncate to compact support — speeds up filter_crystals dramatically
        # at large n_bins (only ~2·truncate_σ·σ_rank crystals survive per bin).
        outside = delta.abs() > self.truncate_sigma * sig
        w = w.masked_fill(outside, 0.0)
        # Crystals with no timepoint (rank == -1) drop out.
        w = w.masked_fill(self.crystal_rank < 0, 0.0)
        return w.to(torch.float32)


def _crystal_rank(stream) -> tuple[Tensor, Tensor]:
    """Return ``(rank, crystal_tp)`` for the given stream.

    ``rank[c] = position of crystal c in the temporal sort order``;
    crystals with NaN timepoint get ``-1``.

    ``crystal_tp`` is ``(n_crystals,)`` float64 with NaNs for unlabelled
    crystals.  Both tensors are on CPU.
    """
    crystal_tp_list = stream.crystal_timepoints()
    if crystal_tp_list is None:
        raise ValueError(
            "Stream has no timepoint metadata. "
            "Call stream.assign_timepoints(times_file) first."
        )
    if isinstance(crystal_tp_list, Tensor):
        crystal_tp = crystal_tp_list.to(torch.float64).cpu()
    else:
        crystal_tp = torch.tensor(crystal_tp_list, dtype=torch.float64)
    if crystal_tp.numel() == 0:
        raise ValueError("Stream has no crystals")

    finite = ~torch.isnan(crystal_tp)
    if not finite.any():
        raise ValueError(
            "All crystal timepoints are NaN — assign_timepoints matched no images"
        )

    # Build rank only over finite-timepoint crystals.
    finite_idx = torch.where(finite)[0]                            # (N,)
    sort_order = crystal_tp[finite_idx].argsort()                  # sort finites by t
    sorted_finite_idx = finite_idx[sort_order]                     # (N,) crystal ids in time order
    rank = torch.full((crystal_tp.numel(),), -1, dtype=torch.int64)
    rank[sorted_finite_idx] = torch.arange(sorted_finite_idx.numel(), dtype=torch.int64)
    return rank, crystal_tp


def _build_bins(
    centers_rank: Tensor,
    sigma_rank: float,
    truncate_sigma: float,
    crystal_rank: Tensor,
    crystal_tp: Tensor,
) -> GaussianRankBins:
    """Common postprocessing: derive ``centers_time`` and ``effective_n``."""
    centers_rank = centers_rank.to(torch.int64)
    n_bins = int(centers_rank.numel())
    n_crystals = int(crystal_rank.numel())

    rank_f = crystal_rank.to(torch.float64)
    valid = crystal_rank >= 0
    centers_time = torch.zeros(n_bins, dtype=torch.float64)
    effective_n = torch.zeros(n_bins, dtype=torch.float64)
    sig = float(sigma_rank)
    trunc = float(truncate_sigma)
    for b in range(n_bins):
        c = float(centers_rank[b].item())
        delta = rank_f - c
        w = torch.exp(-0.5 * (delta / sig) ** 2)
        w = w.masked_fill(delta.abs() > trunc * sig, 0.0)
        w = w.masked_fill(~valid, 0.0)
        eff = w.sum()
        effective_n[b] = eff
        if eff > 0:
            centers_time[b] = (w * crystal_tp.nan_to_num(0.0)).sum() / eff
        else:
            centers_time[b] = float("nan")

    return GaussianRankBins(
        centers_rank=centers_rank,
        sigma_rank=sig,
        truncate_sigma=trunc,
        centers_time=centers_time,
        effective_n=effective_n,
        crystal_rank=crystal_rank,
    )


def gaussian_rank_bins(
    stream,
    n_bins: int,
    *,
    sigma_rank: float | None = None,
    truncate_sigma: float = 3.0,
    inset_rank: float | None = None,
) -> GaussianRankBins:
    """Evenly-spaced Gaussian-in-rank bins.

    Parameters
    ----------
    stream :
        :class:`SFXStream` that has been timepoint-labelled via
        :meth:`SFXStream.assign_timepoints`.
    n_bins :
        Number of bins.
    sigma_rank :
        Gaussian σ in rank-space.  Default ``n_valid / (2·n_bins)`` —
        each bin's 1-σ radius spans half the gap to the next bin, so
        adjacent bins meet at their respective 1-σ contours (≈ 60 %
        overlap by integrated weight).
    truncate_sigma :
        Zero weights past this many σ.  Default 3 → ~99.7 % of the
        Gaussian mass is retained.
    inset_rank :
        If given, place the first/last bin centers at ``inset_rank`` and
        ``n_valid − 1 − inset_rank`` respectively (instead of at the rank
        boundaries 0 and ``n_valid − 1``).  Useful for sweeping σ_rank:
        set ``inset_rank ≥ 2·σ_rank`` so the truncated Gaussian doesn't
        clip the data edge, giving every bin equal effective_n.  Bin
        centers are stable across a sweep if ``inset_rank`` is fixed —
        which makes per-bin SVD time courses commensurate across configs.
        Default ``None`` recovers the original 0…N−1 placement.

    Returns
    -------
    GaussianRankBins
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be ≥ 1, got {n_bins}")

    rank, crystal_tp = _crystal_rank(stream)
    n_valid = int((rank >= 0).sum().item())
    if n_valid == 0:
        raise ValueError("Stream has no crystals with finite timepoints")

    if sigma_rank is None:
        sigma_rank = max(1.0, n_valid / max(2 * n_bins, 1))

    if n_bins == 1:
        centers_rank = torch.tensor([(n_valid - 1) // 2], dtype=torch.int64)
    else:
        if inset_rank is None:
            lo, hi = 0.0, float(n_valid - 1)
        else:
            if inset_rank < 0:
                raise ValueError(f"inset_rank must be ≥ 0, got {inset_rank}")
            lo = float(inset_rank)
            hi = float(n_valid - 1) - float(inset_rank)
            if hi < lo:
                raise ValueError(
                    f"inset_rank={inset_rank} too large for n_valid={n_valid}: "
                    f"would invert the center range"
                )
        edges = torch.linspace(lo, hi, n_bins, dtype=torch.float64)
        centers_rank = edges.round().to(torch.int64).clamp(0, n_valid - 1)

    return _build_bins(centers_rank, sigma_rank, truncate_sigma, rank, crystal_tp)


def gaussian_rank_bins_at_times(
    stream,
    centers_time: Sequence[float] | Tensor,
    *,
    sigma_rank: float,
    truncate_sigma: float = 3.0,
) -> GaussianRankBins:
    """Bins centered at user-specified timepoints.

    For each requested time ``t``, the center crystal-rank is the rank
    of the crystal whose timepoint is closest to ``t``.

    Parameters
    ----------
    centers_time :
        Sequence of target timepoints (same units as the labelled stream).
    sigma_rank :
        Gaussian σ in rank-space.  Required (no obvious default once
        centers are arbitrary).
    """
    rank, crystal_tp = _crystal_rank(stream)
    valid = rank >= 0
    if not valid.any():
        raise ValueError("Stream has no crystals with finite timepoints")

    if isinstance(centers_time, Tensor):
        ct = centers_time.to(torch.float64).cpu()
    else:
        ct = torch.tensor(list(centers_time), dtype=torch.float64)

    # Per-time-center: pick the valid crystal whose timepoint is closest.
    crystal_tp_valid = crystal_tp.clone()
    crystal_tp_valid[~valid] = float("inf")  # mask out NaN so they never win
    centers_rank = torch.zeros(ct.numel(), dtype=torch.int64)
    for i, t in enumerate(ct.tolist()):
        nearest_crystal = (crystal_tp_valid - t).abs().argmin().item()
        centers_rank[i] = int(rank[nearest_crystal].item())

    return _build_bins(centers_rank, sigma_rank, truncate_sigma, rank, crystal_tp)


# ── Per-bin result ──────────────────────────────────────────────────────


@dataclass
class BinResult:
    """Output of one bin of :func:`time_binned_merge`.

    Attributes
    ----------
    bin_idx :
        0-based bin index.
    center_rank :
        Crystal rank at the Gaussian peak.
    center_time :
        Weighted-mean timepoint of the bin.
    effective_n_crystals :
        ``Σ_c w_b[c]`` — effective crystal count contributing to this bin.
    n_contributing_crystals :
        Number of crystals with strictly positive weight (post-truncation).
    mtz_path :
        Location of the written MTZ.
    diagnostics :
        Per-bin loss / refine / posterior diagnostics (gain, final NLL,
        number of HKLs, n_observations in the sub-stream, etc.).
    """

    bin_idx: int
    center_rank: int
    center_time: float
    effective_n_crystals: float
    n_contributing_crystals: int
    mtz_path: Path
    diagnostics: dict[str, Any] = field(default_factory=dict)


# ── Shared α/β pre-fit (extracted helper) ──────────────────────────────


PrefitLookup = dict[tuple[int, int, int], tuple[float, float, float]]


def compute_prefit_lookup(
    stream,
    *,
    make_loss_kwargs: dict[str, Any],
    refine_kwargs: dict[str, Any] | None = None,
    verbose: bool = True,
) -> PrefitLookup:
    """Run the joint α, β pre-fit on the full stream and return a lookup.

    The lookup maps each HKL tuple ``(h, k, l)`` to
    ``(log_α, log_β, log_I0)`` from the joint fit (α, β under the
    Laplacian smoothness prior, log_I0 as the warm-start anchor for
    per-bin LBFGS).

    Reusable across e.g. a σ_rank sweep where the α, β pre-fit depends
    only on the input stream — not on the bin geometry.

    Parameters
    ----------
    stream :
        Timepoint-labelled :class:`SFXStream`.
    make_loss_kwargs :
        Forwarded to :meth:`SFXStream.make_loss`.  Should not contain
        ``crystal_weights``.
    refine_kwargs :
        Forwarded to :meth:`LogSpaceLoss.refine`.  Optional.
    verbose :
        Print progress + summary on completion.
    """
    refine_kwargs = dict(refine_kwargs or {})
    if verbose:
        print(
            f"[prefit] joint α, β fit on {stream.n_crystals} crystals, "
            f"{stream.n_observations} obs ...",
            flush=True,
        )
    loss = stream.make_loss(**make_loss_kwargs)
    loss.refine(**refine_kwargs)
    with torch.no_grad():
        hkl = loss._prep.hkl_valid.detach().cpu().long()
        log_alpha = loss._params.log_alpha.detach().cpu()
        log_beta = loss._params.log_beta.detach().cpu()
        log_I0 = loss._params.log_I0.detach().cpu()
    lookup: PrefitLookup = {
        (int(hkl[i, 0]), int(hkl[i, 1]), int(hkl[i, 2])):
            (float(log_alpha[i]), float(log_beta[i]), float(log_I0[i]))
        for i in range(hkl.shape[0])
    }
    if verbose:
        la_med = float(log_alpha.median())
        lb_med = float(log_beta.median())
        lI_med = float(log_I0.median())
        print(
            f"[prefit] done: {len(lookup)} HKLs, "
            f"median log α={la_med:.3f} (α≈{math.exp(la_med):.3f}), "
            f"median log β={lb_med:.3f} (β≈{math.exp(lb_med):.3f}), "
            f"median log I0={lI_med:.3f} (I0≈{math.exp(lI_med):.3f}), "
            f"final NLL={loss.final_nll:.4e}",
            flush=True,
        )
    del loss
    return lookup


def save_prefit_lookup(lookup: PrefitLookup, path: str | Path) -> None:
    """Persist a pre-fit lookup as a small safetensors blob."""
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not lookup:
        raise ValueError("Refusing to save an empty pre-fit lookup")
    keys = list(lookup.keys())
    hkl = torch.tensor(keys, dtype=torch.int32)
    n = len(keys)
    log_alpha = torch.empty(n, dtype=torch.float32)
    log_beta = torch.empty(n, dtype=torch.float32)
    log_I0 = torch.empty(n, dtype=torch.float32)
    for i, k in enumerate(keys):
        la, lb, lI = lookup[k]
        log_alpha[i] = la
        log_beta[i] = lb
        log_I0[i] = lI
    save_file(
        {
            "hkl": hkl,
            "log_alpha": log_alpha,
            "log_beta": log_beta,
            "log_I0": log_I0,
        },
        str(path),
    )


def load_prefit_lookup(path: str | Path) -> PrefitLookup:
    """Reverse of :func:`save_prefit_lookup`."""
    from safetensors.torch import load_file

    blob = load_file(str(path))
    hkl = blob["hkl"].to(torch.int64)
    log_alpha = blob["log_alpha"].to(torch.float64)
    log_beta = blob["log_beta"].to(torch.float64)
    log_I0 = blob["log_I0"].to(torch.float64)
    n = hkl.shape[0]
    return {
        (int(hkl[i, 0]), int(hkl[i, 1]), int(hkl[i, 2])):
            (float(log_alpha[i]), float(log_beta[i]), float(log_I0[i]))
        for i in range(n)
    }


# ── Driver ──────────────────────────────────────────────────────────────


def time_binned_merge(
    stream,
    bins: GaussianRankBins,
    output_dir: str | Path,
    *,
    output_template: str = "bin_{idx:02d}_t{t_center:+.1f}.mtz",
    make_loss_kwargs: dict[str, Any] | None = None,
    refine_kwargs: dict[str, Any] | None = None,
    posterior_prior: str = "flat",
    shared_alpha_beta: bool = False,
    prefit_refine_kwargs: dict[str, Any] | None = None,
    prefit_lookup: PrefitLookup | None = None,
    prefit_cache_path: str | Path | None = None,
    merge_dark: bool = False,
    dark_mtz_name: str = "dark.mtz",
    skip_existing: bool = False,
    verbose: bool = True,
) -> list[BinResult]:
    """Run the full merge pipeline once per bin, writing one MTZ each.

    Parameters
    ----------
    stream :
        :class:`SFXStream`, timepoint-labelled.
    bins :
        :class:`GaussianRankBins` from :func:`gaussian_rank_bins` or
        :func:`gaussian_rank_bins_at_times`.
    output_dir :
        Directory for the per-bin MTZs.  Created if absent.
    output_template :
        ``str.format``-style template.  Substitutions: ``{idx}`` (int),
        ``{t_center}`` (float).
    make_loss_kwargs :
        Forwarded to :meth:`SFXStream.make_loss` per bin.  ``crystal_weights``
        is overridden internally.
    refine_kwargs :
        Forwarded to :meth:`LogSpaceLoss.refine`.
    posterior_prior :
        Passed to :meth:`LogSpaceLoss.posterior` (default ``"flat"``).
    shared_alpha_beta :
        If True, run a one-shot pre-fit on the full input stream to get
        per-HKL (α, β), then **freeze** those values across all bins via
        :class:`FixedShape`.  The scatter shape becomes a property of the
        protein/dataset (not per-bin), so cross-bin intensity comparisons
        aren't contaminated by per-bin α/β drift.  Only log_I0 / scale /
        partiality refine per bin.
    prefit_refine_kwargs :
        ``LogSpaceLoss.refine`` kwargs for the full-stream pre-fit when
        ``shared_alpha_beta=True``.  Defaults to ``refine_kwargs`` (same
        budget as a regular per-bin refine).
    prefit_lookup :
        Pre-computed ``{(h,k,l): (log_α, log_β, log_I0)}`` from
        :func:`compute_prefit_lookup`.  When provided (with
        ``shared_alpha_beta=True``), the in-line pre-fit is skipped and
        this lookup is used directly.  Lets a caller (e.g. a σ_rank
        sweep) reuse one pre-fit across many bin configurations.
    prefit_cache_path :
        Optional safetensors path for the pre-fit lookup.  If it exists
        on disk, the lookup is loaded from it (skipping the pre-fit);
        otherwise the pre-fit is computed and the result saved here for
        future runs.  Ignored unless ``shared_alpha_beta=True`` and
        ``prefit_lookup`` is None.
    merge_dark :
        If True, after the time-bin loop also merge the crystals in
        ``stream`` whose timepoint is NaN (no label / dark) as a single
        extra MTZ.  Requires ``shared_alpha_beta=True`` so the dark merge
        uses the same per-HKL α, β as every light bin — yields an
        unbiased F(light_b) − F(dark) baseline.  The dark merge's input
        is ``stream.filter_crystals(crystal_tp.isnan())``.
    dark_mtz_name :
        Filename for the dark MTZ (placed directly in ``output_dir``).
    skip_existing :
        If True, bins whose target MTZ already exists are skipped (the
        result list still includes a ``BinResult`` for them).

    Returns
    -------
    list[BinResult]
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    make_loss_kwargs = dict(make_loss_kwargs or {})
    # The driver controls ``crystal_weights``; reject user attempts to set it.
    if "crystal_weights" in make_loss_kwargs:
        raise ValueError(
            "make_loss_kwargs['crystal_weights'] is set by the driver per bin; "
            "drop it from the caller-supplied dict."
        )
    refine_kwargs = dict(refine_kwargs or {})

    # ── Resolve gain on the FULL stream so all bins share one number ──
    # The detector gain is a physical constant of the experiment, not a
    # per-bin quantity.  Calibrating per-bin (``gain="auto"`` on each
    # filtered sub-stream) gives ~0.5 % bin-to-bin drift that contaminates
    # cross-bin intensity comparisons (difference maps, time evolution).
    # Calibrating once on the full input and pinning the resulting float
    # for every bin eliminates this — and is cheaper too (one calibration
    # instead of N_bins).
    gain_spec = make_loss_kwargs.get("gain", "auto")
    if isinstance(gain_spec, str):
        from torchsx.merging.hkl import PoissonNoiseModel
        from torchsx.merging.observations import _ewald_offset
        pn = PoissonNoiseModel(
            r_peak=make_loss_kwargs.get("r_peak", 7.0),
            r_gap=make_loss_kwargs.get("r_gap", 9.0),
            r_bg=make_loss_kwargs.get("r_bg", 11.0),
        )
        core = stream._core
        # Calibration runs on raw ADU values (gain is what divides them).
        I_adu = core.I.float()
        bg_adu = core.background.float()
        if gain_spec == "stream":
            adu_per_eV = float(
                getattr(core.geometry, "adu_per_eV", 1.0) or 1.0
            ) if core.geometry is not None else 1.0
            mean_eV = float(core.beam_energy.float().mean().item())
            gain_resolved = max(adu_per_eV * mean_eV, 1.0)
            src = f"stream (adu_per_eV={adu_per_eV:.6g} × {mean_eV:.1f} eV)"
        elif gain_spec == "empirical":
            g = pn.calibrate_gain(I_adu, bg_adu)
            gain_resolved = float(g) if g and g > 0 else 1.0
            src = "empirical (negatives-Mandel)"
        elif gain_spec == "auto":
            ewald = _ewald_offset(core)
            g, mu_bias, _ = pn.calibrate_sigma2_of_bg(I_adu, bg_adu, ewald)
            gain_resolved = float(g) if g and g > 0 else 1.0
            src = f"auto (σ²(bg) off-Bragg, μ_z={mu_bias:+.3f}σ)"
        else:
            raise ValueError(
                f"Unknown gain spec {gain_spec!r}; expected float, "
                "'auto', 'stream', or 'empirical'"
            )
        make_loss_kwargs["gain"] = gain_resolved
        if verbose:
            print(
                f"[time-binned merge] gain resolved on full stream: "
                f"{gain_resolved:.4g} ADU/photon  [{src}]",
                flush=True,
            )

    # ── Resolve the shared α, β pre-fit lookup ────────────────────────
    # Three paths, in order of precedence:
    #   1) caller passed an explicit ``prefit_lookup`` — use it as-is.
    #   2) caller passed ``prefit_cache_path`` and the file exists — load.
    #   3) otherwise compute via :func:`compute_prefit_lookup` on the
    #      input stream, and save to ``prefit_cache_path`` if given.
    # Either way, the lookup feeds :class:`FixedShape` (frozen α, β)
    # and warm-starts log_I0 per bin (LBFGS converges from joint optimum
    # without per-bin overshoot).
    prefit_hkl_lookup: PrefitLookup | None = None
    if shared_alpha_beta:
        if prefit_lookup is not None:
            prefit_hkl_lookup = prefit_lookup
            if verbose:
                print(
                    f"\n[time-binned merge] using caller-supplied pre-fit lookup "
                    f"({len(prefit_hkl_lookup)} HKLs)",
                    flush=True,
                )
        elif prefit_cache_path is not None and Path(prefit_cache_path).exists():
            prefit_hkl_lookup = load_prefit_lookup(prefit_cache_path)
            if verbose:
                print(
                    f"\n[time-binned merge] loaded pre-fit lookup from cache: "
                    f"{prefit_cache_path}  ({len(prefit_hkl_lookup)} HKLs)",
                    flush=True,
                )
        else:
            prefit_hkl_lookup = compute_prefit_lookup(
                stream,
                make_loss_kwargs=make_loss_kwargs,
                refine_kwargs=prefit_refine_kwargs or refine_kwargs,
                verbose=verbose,
            )
            if prefit_cache_path is not None:
                save_prefit_lookup(prefit_hkl_lookup, prefit_cache_path)
                if verbose:
                    print(
                        f"[time-binned merge] saved pre-fit lookup to "
                        f"{prefit_cache_path}",
                        flush=True,
                    )
    elif prefit_lookup is not None or prefit_cache_path is not None:
        if verbose:
            print(
                "[time-binned merge] warning: prefit_lookup / prefit_cache_path "
                "ignored because shared_alpha_beta=False",
                flush=True,
            )

    results: list[BinResult] = []
    n_bins = bins.n_bins
    if verbose:
        print(
            f"[time-binned merge] {n_bins} bins, sigma_rank={bins.sigma_rank:.1f}, "
            f"truncate_sigma={bins.truncate_sigma:.1f}",
            flush=True,
        )

    for b in range(n_bins):
        t_center = float(bins.centers_time[b].item())
        eff_n = float(bins.effective_n[b].item())
        mtz_name = output_template.format(idx=b, t_center=t_center)
        mtz_path = output_dir / mtz_name

        # ── Per-crystal Gaussian weights ───────────────────────────────
        w_full = bins.weights(b)                                    # (n_crystals,)
        crystal_mask = w_full > 0
        n_contrib = int(crystal_mask.sum().item())
        if verbose:
            print(
                f"\n[bin {b:02d}/{n_bins}] t≈{t_center:.2f}  "
                f"effective_n={eff_n:.1f}  contributing crystals={n_contrib}",
                flush=True,
            )

        if n_contrib == 0:
            if verbose:
                print(f"  ⚠ bin {b} has zero contributing crystals — skipping", flush=True)
            results.append(BinResult(
                bin_idx=b,
                center_rank=int(bins.centers_rank[b].item()),
                center_time=t_center,
                effective_n_crystals=eff_n,
                n_contributing_crystals=0,
                mtz_path=mtz_path,
                diagnostics={"skipped": "no_contributing_crystals"},
            ))
            continue

        if skip_existing and mtz_path.exists():
            if verbose:
                print(f"  ↳ {mtz_path} exists — skipping (skip_existing=True)", flush=True)
            results.append(BinResult(
                bin_idx=b,
                center_rank=int(bins.centers_rank[b].item()),
                center_time=t_center,
                effective_n_crystals=eff_n,
                n_contributing_crystals=n_contrib,
                mtz_path=mtz_path,
                diagnostics={"skipped": "existing_mtz"},
            ))
            continue

        # ── Sub-stream + sub-weights ───────────────────────────────────
        sub = stream.filter_crystals(crystal_mask)
        sub_w = w_full[crystal_mask]                                # (n_sub,)

        if verbose:
            print(
                f"  sub-stream: n_crystals={sub.n_crystals}  n_obs={sub.n_observations}",
                flush=True,
            )

        # ── Loss → refine → posterior → MTZ ────────────────────────────
        loss = sub.make_loss(crystal_weights=sub_w, **make_loss_kwargs)
        if prefit_hkl_lookup is not None:
            # Build a FixedShape with the pre-fit (log α, log β) values
            # looked up by HKL for this bin's HKL subset.  HKLs absent
            # from the pre-fit (rare — only if min_obs filtered them on
            # the full stream but the bin somehow has more) keep their
            # default per-HKL init.
            with torch.no_grad():
                bin_hkl = loss._prep.hkl_valid.detach().cpu().long()
                n_valid_bin = bin_hkl.shape[0]
                la_default = loss._params.log_alpha.detach().cpu().clone()
                lb_default = loss._params.log_beta.detach().cpu().clone()
                lI_default = loss._params.log_I0.detach().cpu().clone()
                la_bin = la_default.clone()
                lb_bin = lb_default.clone()
                lI_bin = lI_default.clone()
                n_missing = 0
                for i in range(n_valid_bin):
                    key = (int(bin_hkl[i, 0]), int(bin_hkl[i, 1]), int(bin_hkl[i, 2]))
                    pre = prefit_hkl_lookup.get(key)
                    if pre is None:
                        n_missing += 1
                        continue
                    la_bin[i] = pre[0]
                    lb_bin[i] = pre[1]
                    lI_bin[i] = pre[2]
            from torchsx.merging.scatter_shape import FixedShape
            fixed = FixedShape(la_bin, lb_bin)
            loss.swap_scatter_shape(fixed)
            # Warm-start log_I0 from the pre-fit too.  Starting LBFGS near
            # the joint-fit optimum avoids the per-bin overshoot that
            # produced all-NaN posterior moments on large bins.
            with torch.no_grad():
                loss._params.log_I0.data.copy_(lI_bin.to(loss._params.log_I0.device))
            if verbose:
                miss_msg = f"  ({n_missing} HKLs not in pre-fit — kept init)" if n_missing else ""
                print(
                    f"  shared α, β, log_I0: frozen α/β + warm-started log_I0 on "
                    f"{n_valid_bin - n_missing}/{n_valid_bin} HKLs{miss_msg}",
                    flush=True,
                )
        loss.refine(**refine_kwargs)
        # LBFGS guard: if any log_I0 went non-finite, restore those entries
        # from the warm-start anchor.  Without this, 100 % non-finite log_I0
        # produces an all-NaN posterior whose sanity_F mask masks everything
        # and torchref errors out, killing the whole driver loop.  The
        # restored entries are the joint-pre-fit estimate (finite by
        # construction) — a conservative fallback for the affected HKLs.
        n_log_I0_restored = 0
        if prefit_hkl_lookup is not None:
            with torch.no_grad():
                lI_dev = loss._params.log_I0.data
                bad = ~torch.isfinite(lI_dev)
                if bad.any():
                    n_log_I0_restored = int(bad.sum().item())
                    # lI_bin lives on CPU (built from prefit lookup); move it
                    # to the param's device before indexing — indexing across
                    # devices throws "indices should be on the same device".
                    lI_dev[bad] = lI_bin.to(lI_dev.device)[bad]
                    if verbose:
                        print(
                            f"  ⚠ {n_log_I0_restored}/{n_valid_bin} log_I0 "
                            f"went non-finite during refine — restored from "
                            f"warm-start anchor (joint pre-fit estimate)",
                            flush=True,
                        )
        posterior = loss.posterior(prior=posterior_prior)
        data = posterior.integrate()
        data.write_mtz(str(mtz_path))
        if verbose:
            print(f"  ↳ wrote {mtz_path}  (final NLL = {loss.final_nll})", flush=True)

        diag = {
            "n_crystals": sub.n_crystals,
            "n_observations": sub.n_observations,
            "gain": loss.stats.get("gain"),
            "n_valid": loss.stats.get("n_valid"),
            "final_data_nll": loss.final_nll,
            "shared_alpha_beta": prefit_hkl_lookup is not None,
            "n_log_I0_restored": n_log_I0_restored,
        }
        results.append(BinResult(
            bin_idx=b,
            center_rank=int(bins.centers_rank[b].item()),
            center_time=t_center,
            effective_n_crystals=eff_n,
            n_contributing_crystals=n_contrib,
            mtz_path=mtz_path,
            diagnostics=diag,
        ))

    # ── Optional dark merge (NaN-timepoint crystals) ──────────────────
    # Uses the same gain (pinned) and α, β (from pre-fit on light+dark
    # combined) as every light bin.  The dark merge is a single, unweighted
    # full-stream merge on the NaN-timepoint sub-stream, so cross-bin
    # difference F(light_b) − F(dark) is calibrated against the same
    # scatter shape.  Requires shared_alpha_beta=True — without the
    # shared pre-fit, the dark MTZ would have its own α, β and the
    # difference becomes ill-defined.
    if merge_dark:
        if prefit_hkl_lookup is None:
            raise ValueError(
                "merge_dark=True requires shared_alpha_beta=True so the dark "
                "merge uses the same α, β as the time bins."
            )
        crystal_tp_list = stream.crystal_timepoints()
        if isinstance(crystal_tp_list, torch.Tensor):
            crystal_tp = crystal_tp_list.to(torch.float64).cpu()
        else:
            crystal_tp = torch.tensor(crystal_tp_list, dtype=torch.float64)
        dark_mask = torch.isnan(crystal_tp)
        n_dark = int(dark_mask.sum().item())
        if verbose:
            print(
                f"\n[dark merge] {n_dark} dark crystals "
                f"(NaN timepoint) — shared α, β from full pre-fit",
                flush=True,
            )
        if n_dark == 0:
            if verbose:
                print(
                    "  ⚠ merge_dark=True but no NaN-timepoint crystals found — skipping",
                    flush=True,
                )
        else:
            dark_path = output_dir / dark_mtz_name
            skip_dark = skip_existing and dark_path.exists()
            if skip_dark:
                if verbose:
                    print(f"  ↳ {dark_path} exists — skipping", flush=True)
            else:
                dark_stream = stream.filter_crystals(dark_mask)
                if verbose:
                    print(
                        f"  sub-stream: n_crystals={dark_stream.n_crystals}  "
                        f"n_obs={dark_stream.n_observations}",
                        flush=True,
                    )
                dark_loss = dark_stream.make_loss(**make_loss_kwargs)
                with torch.no_grad():
                    dark_hkl = dark_loss._prep.hkl_valid.detach().cpu().long()
                    n_valid_dark = dark_hkl.shape[0]
                    la_default = dark_loss._params.log_alpha.detach().cpu().clone()
                    lb_default = dark_loss._params.log_beta.detach().cpu().clone()
                    lI_default = dark_loss._params.log_I0.detach().cpu().clone()
                    la_dark = la_default.clone()
                    lb_dark = lb_default.clone()
                    lI_dark = lI_default.clone()
                    n_missing = 0
                    for i in range(n_valid_dark):
                        key = (int(dark_hkl[i, 0]), int(dark_hkl[i, 1]), int(dark_hkl[i, 2]))
                        pre = prefit_hkl_lookup.get(key)
                        if pre is None:
                            n_missing += 1
                            continue
                        la_dark[i] = pre[0]
                        lb_dark[i] = pre[1]
                        lI_dark[i] = pre[2]
                from torchsx.merging.scatter_shape import FixedShape
                dark_loss.swap_scatter_shape(FixedShape(la_dark, lb_dark))
                with torch.no_grad():
                    dark_loss._params.log_I0.data.copy_(
                        lI_dark.to(dark_loss._params.log_I0.device)
                    )
                if verbose:
                    miss_msg = (
                        f"  ({n_missing} HKLs not in pre-fit — kept init)"
                        if n_missing else ""
                    )
                    print(
                        f"  shared α, β, log_I0: frozen α/β + warm-started log_I0 on "
                        f"{n_valid_dark - n_missing}/{n_valid_dark} HKLs{miss_msg}",
                        flush=True,
                    )
                dark_loss.refine(**refine_kwargs)
                dark_posterior = dark_loss.posterior(prior=posterior_prior)
                dark_data = dark_posterior.integrate()
                dark_data.write_mtz(str(dark_path))
                if verbose:
                    print(
                        f"  ↳ wrote {dark_path}  (final NLL = {dark_loss.final_nll})",
                        flush=True,
                    )
                results.append(BinResult(
                    bin_idx=-1,
                    center_rank=-1,
                    center_time=float("nan"),
                    effective_n_crystals=float(n_dark),
                    n_contributing_crystals=n_dark,
                    mtz_path=dark_path,
                    diagnostics={
                        "n_crystals": dark_stream.n_crystals,
                        "n_observations": dark_stream.n_observations,
                        "gain": dark_loss.stats.get("gain"),
                        "n_valid": dark_loss.stats.get("n_valid"),
                        "final_data_nll": dark_loss.final_nll,
                        "shared_alpha_beta": True,
                        "kind": "dark",
                    },
                ))

    return results


# Convenience re-exports.
__all__ = [
    "BinResult",
    "GaussianRankBins",
    "PrefitLookup",
    "compute_prefit_lookup",
    "gaussian_rank_bins",
    "gaussian_rank_bins_at_times",
    "load_prefit_lookup",
    "save_prefit_lookup",
    "time_binned_merge",
]
