"""Native PyTorch spot finder (radial-background, peakfinder8-style).

A vectorized, GPU-capable re-implementation of the statistical model used by the
Cheetah ``peakfinder8`` (see ``submodules/pf8/ext/peakfinder8/peakfinders.cpp``):

1. **Radial background + σ** — bin pixels by radius from the beam center and, with
   a few iterations of positive-outlier rejection, estimate a per-radius mean and
   standard deviation. The per-radius threshold is ``max(mean + min_snr·σ,
   adc_thresh)`` (peaks sit above it).
2. **Candidate mask** — a pixel is a peak candidate when it is good and exceeds the
   threshold at its radius (equivalently ``(I − mean)/σ > min_snr`` with the
   ``adc_thresh`` floor).
3. **Connected components** — 8-connectivity, computed *per panel* (panels are not
   physically adjacent and must not merge) by iterative label propagation
   (3×3 max-pool of labels masked to candidates until stable).
4. **Per-peak reduction** — background-subtracted intensity-weighted centroid,
   pixel count, total intensity; filtered by ``min_pix_count`` / ``max_pix_count``.

The output contract is identical to :func:`torchsx.datareduction.spotfinding.find_spots`
— a ``(M, 4)`` ``[panel, assembled_ss, fs, intensity]`` table — so the two finders
are interchangeable. peakfinder8 stays the reference oracle.

The kernels reduce over flat ``(panel, bin)`` / ``(panel, label)`` indices, so an
optional leading frame dimension can be added later without a rewrite; this version
processes a single frame's module stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# column schema (incl. PROB, the 5th probabilistic-finder column) — see ._schema
from ._schema import FS, INTENSITY, PANEL, PROB, SS  # noqa: F401


@dataclass
class TorchPeakFinderParams:
    """Parameters for :func:`find_spots_torch`.

    Mirrors :class:`~torchsx.datareduction.spotfinding.PeakFinder8Params` (data is
    gain-corrected to photons, so thresholds are in photon units), plus two knobs
    specific to the radial model: ``n_bg_iters`` (outlier-rejection passes) and
    ``bin_width`` (radial bin size in pixels).
    """

    adc_thresh: float = 150.0
    min_snr: float = 8.0
    min_pix_count: int = 2
    max_pix_count: int = 40
    max_num_peaks: int = 5000
    n_bg_iters: int = 5
    bin_width: float = 1.0
    # ── probabilistic finder (find_peaks_prob) ───────────────────────────────
    # No detection threshold: rank local maxima of the significance map and keep
    # the top `top_n`. `prob_*` map significance z=(I-bg)/σ to a probability
    # ∈[0,1] via a logistic (p=0.5 at z=prob_midpoint). adc_thresh/min_snr above
    # still only govern background-estimate outlier rejection, not detection.
    top_n: int = 200
    prob_midpoint: float = 5.0
    prob_scale: float = 1.5
    peak_min_distance: int = 2   # local-max neighbourhood radius (peaks ≥ this apart)
    window: int = 3              # integration half-window (px)
    # bayes centroid: "smax" = sub-pixel parabolic peak of the matched-filter score
    # (the score is the denoised matched signal, and for the symmetric kernel its peak
    # IS the Bragg position — tighter than the box COM, and reuses the score map);
    # "com" = the legacy intensity-weighted box centroid.
    centroid_mode: str = "smax"
    # NB on parity with peakfinder8: pf8 grows each blob below the seed threshold
    # and so keeps single-pixel peaks even at min_pix_count=2; this finder applies
    # min_pix_count to the above-threshold blob directly, so the default rejects
    # lone hot pixels. On real lyso hits the two agree to within a few percent when
    # this is set to ``1`` (and both index to the same HEWL cell); the default of
    # ``2`` is the more conservative, zinger-resistant choice.


def _radial_stats(
    val: torch.Tensor,
    good: torch.Tensor,
    radius: torch.Tensor,
    *,
    n_iters: int,
    min_snr: float,
    adc_thresh: float,
    bin_width: float,
):
    """Per-radius background mean / σ / threshold with outlier rejection.

    All inputs are flat ``(N,)`` tensors over the whole frame. Returns
    ``(mean_px, sigma_px, thresh_px)`` — the background mean, standard deviation,
    and detection threshold gathered back to every pixel — using float64
    accumulation for a stable variance.
    """
    bins = torch.round(radius / bin_width).long().clamp(min=0)
    nbin = int(bins.max().item()) + 1 if bins.numel() else 1

    vald = val.to(torch.float64)
    goodd = good.to(torch.float64)
    adc = float(adc_thresh)

    # First pass includes every good pixel (threshold = +inf).
    threshold = torch.full((nbin,), float("inf"), dtype=torch.float64, device=val.device)
    mean = torch.zeros(nbin, dtype=torch.float64, device=val.device)
    for _ in range(max(1, n_iters)):
        include = goodd * (vald < threshold[bins]).to(torch.float64)
        cnt = torch.zeros(nbin, dtype=torch.float64, device=val.device)
        cnt.scatter_add_(0, bins, include)
        s1 = torch.zeros(nbin, dtype=torch.float64, device=val.device)
        s1.scatter_add_(0, bins, vald * include)
        s2 = torch.zeros(nbin, dtype=torch.float64, device=val.device)
        s2.scatter_add_(0, bins, vald * vald * include)

        cnt_safe = cnt.clamp(min=1.0)
        mean = s1 / cnt_safe
        var = (s2 / cnt_safe - mean * mean).clamp(min=0.0)
        sigma = torch.sqrt(var)
        threshold = torch.clamp(mean + min_snr * sigma, min=adc)
        # Empty bins → mean 0, threshold = adc floor (no spurious peaks there).

    return (mean[bins].to(val.dtype), sigma[bins].to(val.dtype),
            threshold[bins].to(val.dtype))


def radial_background(
    val: torch.Tensor,
    good: torch.Tensor,
    radius: torch.Tensor,
    *,
    n_iters: int = 5,
    min_snr: float = 8.0,
    bin_width: float = 1.0,
) -> torch.Tensor:
    """Per-frame radial background ``b(r)`` gathered to every pixel.

    Robust azimuthal mean per radial shell with iterative positive-outlier
    rejection (peaks/gain-flips excluded). Self-contained per frame — a shell at
    radius r holds ~2*pi*r pixels, so a single frame already determines b(r) well.
    This is just the *background level*; the *noise* is the separate
    :class:`~torchsx.datareduction.noise_model.NoiseModel` ``sigma(b)``.

    All inputs are flat ``(N,)`` tensors. Float64 accumulation for stability.
    """
    bins = torch.round(radius / bin_width).long().clamp(min=0)
    nbin = int(bins.max().item()) + 1 if bins.numel() else 1
    vald = val.to(torch.float64)
    goodd = good.to(torch.float64)

    threshold = torch.full((nbin,), float("inf"), dtype=torch.float64, device=val.device)
    mean = torch.zeros(nbin, dtype=torch.float64, device=val.device)
    for _ in range(max(1, n_iters)):
        include = goodd * (vald < threshold[bins]).to(torch.float64)
        cnt = torch.zeros(nbin, dtype=torch.float64, device=val.device).scatter_add_(0, bins, include)
        s1 = torch.zeros(nbin, dtype=torch.float64, device=val.device).scatter_add_(0, bins, vald * include)
        s2 = torch.zeros(nbin, dtype=torch.float64, device=val.device).scatter_add_(0, bins, vald * vald * include)
        cnt_safe = cnt.clamp(min=1.0)
        mean = s1 / cnt_safe
        sigma = torch.sqrt((s2 / cnt_safe - mean * mean).clamp(min=0.0))
        threshold = mean + min_snr * sigma   # reject positive outliers next pass
    return mean[bins].to(val.dtype)


def _connected_components(candidate: torch.Tensor, n_ss: int, n_fs: int) -> torch.Tensor:
    """8-connected component labels per panel, via iterative label propagation.

    ``candidate`` is ``(P, ss, fs)`` bool. Returns a ``(P, ss, fs)`` int64 tensor of
    panel-local component labels (≥1 on candidates, 0 on background). Propagation is
    a 3×3 max-pool masked to the candidate set, run per panel plane (so components
    never cross panels), iterated until the labels stop changing.
    """
    P = candidate.shape[0]
    cand_f = candidate.to(torch.float32).view(P, 1, n_ss, n_fs)
    # Seed each candidate with its panel-local flat index + 1 (background = 0).
    local_idx = torch.arange(n_ss * n_fs, device=candidate.device, dtype=torch.float32)
    seed = (local_idx + 1.0).view(1, 1, n_ss, n_fs) * cand_f
    lbl = seed
    for _ in range(n_ss * n_fs):  # generous cap; breaks as soon as stable
        pooled = F.max_pool2d(lbl, kernel_size=3, stride=1, padding=1)
        nxt = pooled * cand_f
        if torch.equal(nxt, lbl):
            break
        lbl = nxt
    return lbl.view(P, n_ss, n_fs).round().long()


def find_spots_torch(
    panels: torch.Tensor,
    mask_panels: torch.Tensor,
    pix_r_panels: torch.Tensor,
    params: TorchPeakFinderParams | None = None,
) -> torch.Tensor:
    """Find spots on a module stack with the native radial-background model.

    Parameters
    ----------
    panels : (P, ss, fs) tensor — per-module detector data (photons).
    mask_panels : (P, ss, fs) tensor — True/1 = good pixel.
    pix_r_panels : (P, ss, fs) tensor — radial distance from beam center (px).
    params : :class:`TorchPeakFinderParams`, optional.

    Returns
    -------
    spots : (M, 4) float32 tensor with columns ``[panel, ss, fs, intensity]`` in
        **assembled** detector coordinates (``ss`` includes the ``panel * ss`` row
        offset), matching :func:`spotfinding.find_spots`. Empty → ``(0, 4)``.
    """
    params = params or TorchPeakFinderParams()
    device = panels.device
    P, n_ss, n_fs = panels.shape

    val = panels.to(torch.float32)
    good = mask_panels.to(torch.bool)

    # ── 1. radial background + per-radius threshold ──────────────────────────
    mean_px, _sigma_px, thresh_px = _radial_stats(
        val.reshape(-1), good.reshape(-1), pix_r_panels.to(torch.float32).reshape(-1),
        n_iters=params.n_bg_iters, min_snr=params.min_snr,
        adc_thresh=params.adc_thresh, bin_width=params.bin_width,
    )
    mean_px = mean_px.view(P, n_ss, n_fs)
    thresh_px = thresh_px.view(P, n_ss, n_fs)

    # ── 2. candidate (above-threshold, good) pixels ──────────────────────────
    candidate = good & (val > thresh_px)
    if not bool(candidate.any()):
        return torch.empty((0, 4), dtype=torch.float32, device=device)

    # ── 3. per-panel 8-connected components ──────────────────────────────────
    labels = _connected_components(candidate, n_ss, n_fs)

    # ── 4. reduce each component to a peak ───────────────────────────────────
    panel_grid = torch.arange(P, device=device).view(P, 1, 1).expand(P, n_ss, n_fs)
    ss_grid = torch.arange(n_ss, device=device).view(1, n_ss, 1).expand(P, n_ss, n_fs)
    fs_grid = torch.arange(n_fs, device=device).view(1, 1, n_fs).expand(P, n_ss, n_fs)

    sel = candidate.reshape(-1)
    ibs = (val - mean_px).reshape(-1)[sel].clamp(min=0.0)  # background-subtracted I
    panel_s = panel_grid.reshape(-1)[sel]
    ss_s = ss_grid.reshape(-1)[sel].to(torch.float32)
    fs_s = fs_grid.reshape(-1)[sel].to(torch.float32)

    # Global component key = (panel, panel-local label); unique within a panel.
    big = n_ss * n_fs + 1
    gkey = panel_s.to(torch.int64) * big + labels.reshape(-1)[sel]
    _, comp_id = torch.unique(gkey, return_inverse=True)
    K = int(comp_id.max().item()) + 1

    def _bin(values):
        out = torch.zeros(K, dtype=values.dtype, device=device)
        return out.scatter_add_(0, comp_id, values)

    npix = _bin(torch.ones_like(ibs))
    totI = _bin(ibs)
    sum_ss = _bin(ibs * ss_s)
    sum_fs = _bin(ibs * fs_s)
    totI_safe = totI.clamp(min=1e-12)
    com_ss = sum_ss / totI_safe
    com_fs = sum_fs / totI_safe

    comp_panel = torch.zeros(K, dtype=torch.int64, device=device)
    comp_panel[comp_id] = panel_s  # constant within a component

    keep = (npix >= params.min_pix_count) & (npix <= params.max_pix_count)
    if not bool(keep.any()):
        return torch.empty((0, 4), dtype=torch.float32, device=device)

    assembled_ss = com_ss + comp_panel.to(torch.float32) * n_ss
    out = torch.empty((K, 4), dtype=torch.float32, device=device)
    out[:, PANEL] = comp_panel.to(torch.float32)
    out[:, SS] = assembled_ss
    out[:, FS] = com_fs
    out[:, INTENSITY] = totI
    out = out[keep]

    # ── 5. cap to the brightest max_num_peaks ────────────────────────────────
    if out.shape[0] > params.max_num_peaks:
        order = torch.argsort(out[:, INTENSITY], descending=True)
        out = out[order[: params.max_num_peaks]]

    return out


def find_peaks_prob(
    panels: torch.Tensor,
    mask_panels: torch.Tensor,
    pix_r_panels: torch.Tensor,
    params: TorchPeakFinderParams | None = None,
    *,
    noise_model=None,
    null=None,
) -> torch.Tensor:
    """Thresholdless probabilistic spot finder — always returns the top-N peaks.

    Every local maximum of the significance map ``z = (I − b(r)) / σ`` is a
    candidate; the ``top_n`` most significant are kept, each tagged with a
    probability of being a real Bragg peak. The hit/no-hit decision is deferred to
    the (weighted) indexer.

    The background ``b(r)`` is the per-frame :func:`radial_background`. The noise σ
    and the probability are calibrated when the optional objects are supplied:

    * ``noise_model`` (:class:`~torchsx.datareduction.noise_model.NoiseModel`) — σ =
      ``noise_model.sigma(b(r))`` (detector variance law). If ``None``, falls back to
      the empirical per-radius σ from :func:`_radial_stats`.
    * ``null`` (:class:`~torchsx.datareduction.noise_model.EmpiricalNull`) — the
      per-peak probability is the multiplicity-corrected ``null.prob(z)`` (tail FDR).
      If ``None``, falls back to the logistic ``sigmoid((z − prob_midpoint)/prob_scale)``.

    Returns
    -------
    spots : (M, 5) float32 tensor ``[panel, ss, fs, intensity, prob]`` in assembled
        coordinates, ranked by significance (``M ≤ top_n``). Empty → ``(0, 5)``.
    """
    params = params or TorchPeakFinderParams()
    device = panels.device
    P, n_ss, n_fs = panels.shape
    eps = 1e-6

    val = panels.to(torch.float32)
    good = mask_panels.to(torch.bool)
    radius = pix_r_panels.to(torch.float32)

    # ── 1. per-frame radial background b(r) + noise σ ────────────────────────
    if noise_model is not None:
        mean_px = radial_background(
            val.reshape(-1), good.reshape(-1), radius.reshape(-1),
            n_iters=params.n_bg_iters, min_snr=params.min_snr, bin_width=params.bin_width,
        ).view(P, n_ss, n_fs)
        sigma_px = noise_model.sigma(mean_px)             # σ at the background level
    else:
        mean_px, sigma_px, _thr = _radial_stats(
            val.reshape(-1), good.reshape(-1), radius.reshape(-1),
            n_iters=params.n_bg_iters, min_snr=params.min_snr,
            adc_thresh=params.adc_thresh, bin_width=params.bin_width,
        )
        mean_px = mean_px.view(P, n_ss, n_fs)
        sigma_px = sigma_px.view(P, n_ss, n_fs)

    # ── 2. significance map (−inf at masked pixels so they're never maxima) ───
    z = (val - mean_px) / sigma_px.clamp(min=eps)
    z = torch.where(good, z, torch.full_like(z, float("-inf")))

    # ── 3. per-panel local maxima ────────────────────────────────────────────
    d = params.peak_min_distance
    pooled = F.max_pool2d(z.unsqueeze(1), kernel_size=2 * d + 1, stride=1, padding=d)
    is_max = (z == pooled.squeeze(1)) & good

    flat_max = is_max.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    if flat_max.numel() == 0:
        return torch.empty((0, 5), dtype=torch.float32, device=device)

    # ── 4. rank by significance, keep top_n ──────────────────────────────────
    zf = z.reshape(-1)
    zvals = zf[flat_max]
    top_n = min(params.top_n, flat_max.numel())
    top = torch.topk(zvals, top_n)
    sel = flat_max[top.indices]                       # (M,) flat pixel indices
    zsel = top.values

    pk_panel = sel // (n_ss * n_fs)
    rem = sel % (n_ss * n_fs)
    pk_ss = rem // n_fs
    pk_fs = rem % n_fs

    # ── 5. window centroid + background-subtracted integrated intensity ───────
    w = params.window
    offs = torch.arange(-w, w + 1, device=device)
    doff_ss, doff_fs = torch.meshgrid(offs, offs, indexing="ij")
    doff_ss = doff_ss.reshape(-1)
    doff_fs = doff_fs.reshape(-1)                      # (W,)
    win_ss = pk_ss[:, None] + doff_ss[None, :]         # (M, W)
    win_fs = pk_fs[:, None] + doff_fs[None, :]
    valid = (win_ss >= 0) & (win_ss < n_ss) & (win_fs >= 0) & (win_fs < n_fs)
    win_ss_c = win_ss.clamp(0, n_ss - 1)
    win_fs_c = win_fs.clamp(0, n_fs - 1)
    flat_win = pk_panel[:, None] * (n_ss * n_fs) + win_ss_c * n_fs + win_fs_c

    vwin = val.reshape(-1)[flat_win]
    mwin = mean_px.reshape(-1)[flat_win]
    gwin = good.reshape(-1)[flat_win] & valid
    ibs = (vwin - mwin).clamp(min=0.0) * gwin.to(torch.float32)   # (M, W) weights
    wsum = ibs.sum(dim=1).clamp(min=eps)
    com_ss = (ibs * win_ss.to(torch.float32)).sum(dim=1) / wsum
    com_fs = (ibs * win_fs.to(torch.float32)).sum(dim=1) / wsum
    intensity = ibs.sum(dim=1)

    if null is not None:
        prob = null.prob(zsel)                            # multiplicity-corrected
    else:
        prob = torch.sigmoid((zsel - params.prob_midpoint) / params.prob_scale)

    out = torch.empty((top_n, 5), dtype=torch.float32, device=device)
    out[:, PANEL] = pk_panel.to(torch.float32)
    out[:, SS] = com_ss + pk_panel.to(torch.float32) * n_ss
    out[:, FS] = com_fs
    out[:, INTENSITY] = intensity
    out[:, PROB] = prob
    return out


@torch.no_grad()
def bayes_score_map(
    panels: torch.Tensor,
    mask_panels: torch.Tensor,
    pix_r_panels: torch.Tensor,
    angle_panels: torch.Tensor,
    params: TorchPeakFinderParams | None = None,
    *,
    noise_model=None,
    profile=None,
    operator=None,
):
    """Whitened residual ``w``, background ``b``, and matched-filter score ``S``.

    The single statistic shared by :func:`find_peaks_bayes` and
    :func:`~torchsx.datareduction.noise_model.calibrate_null` so the FDR null is
    computed on exactly the score the finder ranks on. ``b`` is the per-frame
    :func:`radial_background`; ``σ`` is ``noise_model.sigma(b)`` (or the empirical
    per-radius σ when ``noise_model`` is ``None``); ``S`` is the oriented
    matched-filter score of :func:`~torchsx.datareduction.spot_profile.matched_filter_score`.

    Returns ``(S, b, good)``, each ``(P, ss, fs)`` (``good`` is bool).
    """
    from .spot_profile import (
        SpotProfile, matched_filter_score, matched_filter_score_fused,
    )

    params = params or TorchPeakFinderParams()
    profile = profile or SpotProfile.FALLBACK
    P, n_ss, n_fs = panels.shape
    eps = 1e-6

    val = panels.to(torch.float32)
    good = mask_panels.to(torch.bool)
    radius = pix_r_panels.to(torch.float32)
    angle = angle_panels.to(torch.float32)

    if noise_model is not None:
        mean_px = radial_background(
            val.reshape(-1), good.reshape(-1), radius.reshape(-1),
            n_iters=params.n_bg_iters, min_snr=params.min_snr, bin_width=params.bin_width,
        ).view(P, n_ss, n_fs)
        sigma_px = noise_model.sigma(mean_px)
    else:
        mean_px, sigma_px, _thr = _radial_stats(
            val.reshape(-1), good.reshape(-1), radius.reshape(-1),
            n_iters=params.n_bg_iters, min_snr=params.min_snr,
            adc_thresh=params.adc_thresh, bin_width=params.bin_width,
        )
        mean_px = mean_px.view(P, n_ss, n_fs)
        sigma_px = sigma_px.view(P, n_ss, n_fs)

    w = (val - mean_px) / sigma_px.clamp(min=eps)
    w = torch.where(good, w, torch.zeros_like(w))         # masked → 0 evidence
    if operator is not None and w.is_cuda:
        # exact-fp32 fused Triton path; ~13x the conv path, no materialized bank
        S = matched_filter_score_fused(w, operator)
    else:
        S = matched_filter_score(w, good, angle, profile, radius=radius)
    S = torch.where(good, S, torch.full_like(S, float("-inf")))   # masked never maxima
    return S, mean_px, good


@torch.no_grad()
def find_peaks_bayes(
    panels: torch.Tensor,
    mask_panels: torch.Tensor,
    pix_r_panels: torch.Tensor,
    angle_panels: torch.Tensor,
    params: TorchPeakFinderParams | None = None,
    *,
    noise_model=None,
    null=None,
    profile=None,
    operator=None,
) -> torch.Tensor:
    """Fully-Bayesian matched-filter spot finder — always returns the top-N peaks.

    Like :func:`find_peaks_prob`, but a pixel's significance is the **oriented
    matched-filter score** ``S`` rather than its own ``z = (I − b)/σ``: the
    convolution with the anisotropic spot kernel ``g`` folds in the neighbours'
    evidence weighted by the expected spot shape, so a coherent multi-pixel blob
    outscores an isolated hot pixel of the same height (a property the per-pixel
    z-finder lacks). See :mod:`~torchsx.datareduction.spot_profile`.

    Parameters mirror :func:`find_peaks_prob` plus ``angle_panels`` (per-pixel
    radial angle, ``geom.pixel_angle_rad``) and ``profile`` (the anisotropic
    :class:`~torchsx.datareduction.spot_profile.SpotProfile`; defaults to the
    isotropic fallback). ``noise_model`` / ``null`` are used exactly as in
    :func:`find_peaks_prob` — the per-peak probability is the multiplicity-corrected
    ``null.prob(S)`` (tail FDR) when ``null`` is given, else a logistic of ``S``.

    .. note::
       **GPU-only hit-finder.** The score is ``n_bands × n_orient`` convolutions
       over the full module stack per frame — milliseconds on a GPU, but several
       seconds per frame on CPU (and ``calibrate_null`` repeats it once per
       calibration frame). Run with ``device="cuda"``; the CPU path is for tests
       and small checks only.

    Returns
    -------
    spots : (M, 5) float32 ``[panel, ss, fs, intensity, prob]`` in assembled
        coordinates, ranked by ``S`` (``M ≤ top_n``). Empty → ``(0, 5)``.
    """
    params = params or TorchPeakFinderParams()
    device = panels.device
    _P, n_ss, n_fs = panels.shape
    eps = 1e-6

    val = panels.to(torch.float32)

    # ── 1. background + matched-filter score map ─────────────────────────────
    z, mean_px, good = bayes_score_map(
        panels, mask_panels, pix_r_panels, angle_panels, params,
        noise_model=noise_model, profile=profile, operator=operator,
    )

    # ── 2. per-panel local maxima of the score ───────────────────────────────
    d = params.peak_min_distance
    pooled = F.max_pool2d(z.unsqueeze(1), kernel_size=2 * d + 1, stride=1, padding=d)
    is_max = (z == pooled.squeeze(1)) & good

    flat_max = is_max.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    if flat_max.numel() == 0:
        return torch.empty((0, 5), dtype=torch.float32, device=device)

    # ── 3. rank by score, keep top_n ─────────────────────────────────────────
    zf = z.reshape(-1)
    zvals = zf[flat_max]
    top_n = min(params.top_n, flat_max.numel())
    top = torch.topk(zvals, top_n)
    sel = flat_max[top.indices]
    zsel = top.values

    pk_panel = sel // (n_ss * n_fs)
    rem = sel % (n_ss * n_fs)
    pk_ss = rem // n_fs
    pk_fs = rem % n_fs

    # ── 4. centroid + background-subtracted integrated intensity ──────────────
    w = params.window
    offs = torch.arange(-w, w + 1, device=device)
    doff_ss, doff_fs = torch.meshgrid(offs, offs, indexing="ij")
    doff_ss = doff_ss.reshape(-1)
    doff_fs = doff_fs.reshape(-1)
    win_ss = pk_ss[:, None] + doff_ss[None, :]
    win_fs = pk_fs[:, None] + doff_fs[None, :]
    valid = (win_ss >= 0) & (win_ss < n_ss) & (win_fs >= 0) & (win_fs < n_fs)
    win_ss_c = win_ss.clamp(0, n_ss - 1)
    win_fs_c = win_fs.clamp(0, n_fs - 1)
    flat_win = pk_panel[:, None] * (n_ss * n_fs) + win_ss_c * n_fs + win_fs_c

    vwin = val.reshape(-1)[flat_win]
    mwin = mean_px.reshape(-1)[flat_win]
    gwin = good.reshape(-1)[flat_win] & valid
    ibs = (vwin - mwin).clamp(min=0.0) * gwin.to(torch.float32)
    intensity = ibs.sum(dim=1)

    if params.centroid_mode == "smax":
        # Sub-pixel peak of the matched-filter score: separable 3-point parabolic
        # fit on the score map (reuses zf). For the even-symmetric kernel the score
        # peaks at the Bragg position, so this is tighter than the box COM (which is
        # noisy + box-truncated for radially-elongated spots) and needs no extra gather.
        def _samp(ss, fs):
            idx = pk_panel * (n_ss * n_fs) + ss.clamp(0, n_ss - 1) * n_fs + fs.clamp(0, n_fs - 1)
            return zf[idx]

        s_c = zsel
        s_up, s_dn = _samp(pk_ss + 1, pk_fs), _samp(pk_ss - 1, pk_fs)
        s_r, s_l = _samp(pk_ss, pk_fs + 1), _samp(pk_ss, pk_fs - 1)

        def _subpix(s_lo, s_hi, at_border):
            den = s_lo - 2.0 * s_c + s_hi
            d = 0.5 * (s_lo - s_hi) / den
            ok = torch.isfinite(s_lo) & torch.isfinite(s_hi) & (den < -1e-6) & (~at_border)
            return torch.where(ok, d.clamp(-1.0, 1.0), torch.zeros_like(d))

        com_ss = pk_ss.to(torch.float32) + _subpix(s_dn, s_up, (pk_ss == 0) | (pk_ss == n_ss - 1))
        com_fs = pk_fs.to(torch.float32) + _subpix(s_l, s_r, (pk_fs == 0) | (pk_fs == n_fs - 1))
    else:  # "com" — legacy intensity-weighted box centroid
        wsum = ibs.sum(dim=1).clamp(min=eps)
        com_ss = (ibs * win_ss.to(torch.float32)).sum(dim=1) / wsum
        com_fs = (ibs * win_fs.to(torch.float32)).sum(dim=1) / wsum

    if null is not None:
        prob = null.prob(zsel)
    else:
        prob = torch.sigmoid((zsel - params.prob_midpoint) / params.prob_scale)

    out = torch.empty((top_n, 5), dtype=torch.float32, device=device)
    out[:, PANEL] = pk_panel.to(torch.float32)
    out[:, SS] = com_ss + pk_panel.to(torch.float32) * n_ss
    out[:, FS] = com_fs
    out[:, INTENSITY] = intensity
    out[:, PROB] = prob
    return out
