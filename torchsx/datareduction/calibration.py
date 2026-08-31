"""Per-pixel empirical calibration for the SFX hit-finder.

The Gaussian noise model ``σ²(μ)`` is badly miscalibrated on JUNGFRAU: the residual
``e = I − b(r)`` is a skewed Poisson comb with ~20% neighbour charge-sharing, not a
Gaussian. Whitening with ``(I − b)/σ`` therefore produces a skewed statistic whose
positive tail rides above the trough-based FDR null, and the finder over-calls every
frame. Measuring the residual distribution **empirically** fixes this: the per-pixel PIT
is flat (KS ≈ 0.0006 vs the Gaussian's 0.077 — ~130× better) and the false-positive rate
matches nominal where the Gaussian is 2–6× off (validated in
``experiments/hitfind_validation/empirical_likelihood.py``).

This module turns that validated result into per-run calibration, gain-agnostic (the
on-disk frames are already gain-corrected to photons; the 2-bit gain-stage codes are
gone, so we model each pixel's *marginal* distribution rather than stratifying by stage):

1. :class:`PerPixelModel` — the **constant channel**: a per-pixel robust temporal
   location ``c_p`` (median of ``I − b(r)``) and scale ``σ_p`` (IQR/1.349). Replaces the
   static fixed-pattern ``f(p)`` (mean → median, σ-units → photons) and, by being
   *per-pixel*, subsumes fixed occlusions / edges / beamstop penumbra with no spatial
   window to straddle — the mask-free handling of the design doc §7.

2. :class:`EmpiricalNoise` — the **noise distribution**: the empirical CDF of the score
   residual conditioned on the local background ``μ = b(r)``, consumed as a normal-score
   transform ``z' = Φ⁻¹(F(e | μ))`` so the whitened residual ``w`` is ``N(0,1)`` per pixel
   under background. The matched filter / FDR null then operate on a genuinely white ``w``
   (validated in ``converter_mf.py``: skew +0.43 → −0.2, blank→hit separation doubles).

The whitening shared by calibration and per-frame finding is :func:`whitened_residual`;
calibration and eval go through the same :func:`_score_residual` so the two can never
drift apart (the documented failure mode is the null decoupling from the finder).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..geometry import Geometry
from .noise_model import NoiseModel, _load_panels
from .peakfinder import radial_background

_SQRT2 = math.sqrt(2.0)

# Score-residual histogram grids. ``f(p)``/raw mode keeps the validated counts grid from
# converter_mf.py; per-pixel mode standardizes by (c_p, σ_p) first, so its residual is in
# σ units and uses a symmetric grid.
_RAW_GRID = (-10.0, 60.0, 1400)        # (lo, hi, n_bins) for e = I − b − f·σ  (photons)
_STD_GRID = (-12.0, 12.0, 1200)        # for the standardized t = (I − b − c_p)/σ_p


def _score_residual(v, b, sigma, *, fixed_pattern=None, c=None, scale=None):
    """The residual the empirical CDF is built on / looked up at — one definition for
    calibration and eval. All tensors flat ``(N,)`` (or all ``(P,ss,fs)``), same device.

    * ``c``/``scale`` given → standardized ``t = (I − b − c_p) / σ_p`` (σ units);
    * else ``fixed_pattern`` given → ``e = I − b − f(p)·σ`` (photons, the converter_mf path);
    * else → ``e = I − b`` (photons).
    """
    if c is not None:
        return (v - b - c) / scale
    if fixed_pattern is not None:
        return v - b - fixed_pattern * sigma
    return v - b


def _whiten_from_bg(v, b, sig, gf, *, empirical, per_pixel=None, fixed_pattern=None):
    """Normal-score ``w = Φ⁻¹(F(e|μ=b))`` from a precomputed background ``b`` and ``σ``.

    The shared primitive behind both :func:`whitened_residual` (standalone / benchmark) and
    the in-finder path in :func:`~torchsx.datareduction.peakfinder.bayes_score_map`, so the
    finder and the FDR null whiten identically. All tensors flat ``(N,)`` on one device;
    masked pixels (``~gf``) → 0.
    """
    c = per_pixel.c.reshape(-1) if per_pixel is not None else None
    s = per_pixel.sigma.reshape(-1) if per_pixel is not None else None
    fp = fixed_pattern.reshape(-1) if fixed_pattern is not None else None
    e = _score_residual(v, b, sig, fixed_pattern=fp, c=c, scale=s)
    # signal-model detector (Bragg log-LR) when a logratio table is built, else the
    # null-only normal-score. The log-LR re-ranks weak peaks above the white background;
    # the normal-score alone (calibrated null) loses them — see calib_compare/index_eval.
    score = (empirical.bragg_loglr(e, b) if empirical.logratio is not None
             else empirical.normal_score(e, b))
    return torch.where(gf, score, torch.zeros_like(v))


def _bragg_logratio(Pn, egrid, bragg_mean, *, lr_clip=8.0):
    """Bragg-vs-background log-LR table ``ℓ(e|μ) = log[P_peak/P_0]`` (design doc §2).

    ``P_peak`` is the illuminated-pixel distribution — the background convolved with a
    one-sided exponential (Wilson/acentric) Bragg-intensity prior of mean ``bragg_mean``
    (in e-grid units): ``P_peak(e) = Σ_{A≥0} P_0(e−A)·Exp(A; bragg_mean)``. Returns an
    ``(NMU, NE)`` table clipped to ``±lr_clip``. Built once at calibration; the explicit
    shift-add over ``K ~ 6·bragg_mean/de`` taps is a one-time NMU×NE cost.
    """
    import torch as _t
    NMU, NE = Pn.shape
    de = float(egrid[1] - egrid[0])
    K = int(min(NE, max(2, round(6.0 * float(bragg_mean) / de))))
    k = _t.arange(K, device=Pn.device, dtype=Pn.dtype)
    f = _t.exp(-k * de / float(bragg_mean))
    f = f / f.sum()
    Pp = _t.zeros_like(Pn)
    for j in range(K):                          # P_peak[e] += P_0[e−j]·f[j]  (A = j·de ≥ 0)
        Pp[:, j:] += Pn[:, :NE - j] * f[j]
    Pp = Pp / Pp.sum(1, keepdim=True).clamp(min=1e-30)
    eps = 1e-12
    return (_t.log(Pp + eps) - _t.log(Pn + eps)).clamp(-lr_clip, lr_clip)


# ──────────────────────────── constant channel ──────────────────────────────
@dataclass
class PerPixelModel:
    """Per-pixel static baseline ``c_p`` (photons) + robust scale ``σ_p`` (photons).

    Both are ``(P, 514, fs)``. ``c_p`` is the per-pixel temporal **median** of the
    radial-background-subtracted residual ``I − b(r)`` (robust to the ~1e-5 per-pixel
    Bragg/zinger contamination); ``σ_p`` is its **IQR/1.349**, floored by the detector
    variance law so rare-sample pixels stay sane. Subtracting ``c_p`` and dividing ``σ_p``
    whitens the static per-pixel structure that ``f(p)`` handled, but per-pixel and robust.
    """

    c: torch.Tensor
    sigma: torch.Tensor
    n_samples: torch.Tensor          # per-pixel good-frame count (for shrinkage / QA)

    def to(self, device) -> "PerPixelModel":
        return PerPixelModel(c=self.c.to(device), sigma=self.sigma.to(device),
                             n_samples=self.n_samples.to(device))


def calibrate_per_pixel(
    reader,
    geom: Geometry,
    noise_model: NoiseModel,
    frame_indices,
    *,
    n_iters: int = 5,
    bin_width: float = 1.0,
    win: float = 30.0,
    n_bins: int = 64,
    t_range: float = 8.0,
    std_floor: float = 0.3,
    device: str = "cpu",
    mask=None,
) -> PerPixelModel:
    """Fit the per-pixel constant channel ``(c_p, σ_p)`` by streaming over ``frame_indices``.

    Two streaming passes (cannot hold all frames: 1000×16.9MP ≈ 68 GB):

    1. winsorized per-pixel mean/variance of ``d = I − b(r)`` (``±win`` photons kills the
       sparse Bragg/zinger and shadow outliers) → a provisional centre ``m_p`` / scale
       ``s_p`` that lets pass 2 use a single shared standardized grid despite the huge
       pixel-to-pixel dynamic range (beamstop vs hot pixel);
    2. per-pixel histogram of the standardized ``t = (d − m_p)/s_p`` on a shared
       ``±t_range`` grid (``n_bins`` bins) → per-pixel median and IQR, mapped back to
       photons: ``c_p = m_p + t_med·s_p``, ``σ_p = s_p·IQR_t/1.349``.

    Memory is bounded: the histogram is ``P·514·fs × n_bins`` float32 (~4.3 GB at
    ``n_bins=64`` on the 16.9 MP detector), freed on return. ``σ_p`` is floored at
    ``max(std_floor, sqrt(noise_model.variance(c_p)))`` so dead / rare-sample pixels fall
    back to the detector variance law.
    """
    dev = torch.device(device)
    good = (reader.mask_panels() if mask is None else mask).to(torch.bool).to(dev)
    P, n_ss, n_fs = good.shape
    npix = P * n_ss * n_fs
    r_flat = geom.pixel_radius_px.reshape(-1).to(torch.float32).to(dev)
    gf = good.reshape(-1)
    gfl = gf.to(torch.float64)
    frame_indices = [int(i) for i in frame_indices]

    # ── pass 1: winsorized per-pixel mean/var of d = I − b(r) ────────────────
    s1 = torch.zeros(npix, dtype=torch.float64, device=dev)
    s2 = torch.zeros_like(s1)
    cnt = torch.zeros_like(s1)
    for i in frame_indices:
        v = _load_panels(reader, i, dev).reshape(-1)
        b = radial_background(v, gf, r_flat, n_iters=n_iters, bin_width=bin_width)
        d = (v - b).clamp(-win, win).to(torch.float64)
        s1 += d * gfl
        s2 += d * d * gfl
        cnt += gfl
    cs = cnt.clamp(min=1.0)
    mean_p = (s1 / cs)
    std_p = (s2 / cs - mean_p * mean_p).clamp(min=1e-6).sqrt().clamp(min=std_floor)
    mean_f = mean_p.to(torch.float32)
    std_f = std_p.to(torch.float32)

    # ── pass 2: per-pixel histogram of standardized t = (d − m_p)/s_p ────────
    tgrid = torch.linspace(-t_range, t_range, n_bins + 1, device=dev)
    hist = torch.zeros(npix, n_bins, dtype=torch.float32, device=dev)
    pix = torch.arange(npix, device=dev)
    for i in frame_indices:
        v = _load_panels(reader, i, dev).reshape(-1)
        b = radial_background(v, gf, r_flat, n_iters=n_iters, bin_width=bin_width)
        t = (v - b - mean_f) / std_f
        tb = torch.bucketize(t, tgrid[1:-1]).clamp(0, n_bins - 1)
        # add 1 only for good pixels (masked → src 0); flat index valid for all pixels
        hist.view(-1).scatter_add_(0, pix * n_bins + tb, gf.to(torch.float32))

    cum = torch.cumsum(hist, dim=1)
    tot = cum[:, -1].clamp(min=1.0)
    cumn = cum / tot[:, None]
    tc = (0.5 * (tgrid[1:] + tgrid[:-1]))                     # bin centres (n_bins,)

    def _quantile_t(p: float) -> torch.Tensor:
        # first bin whose normalized cumulative ≥ p (≈ the p-quantile in t units)
        idx = (cumn >= p).to(torch.uint8).argmax(dim=1)
        return tc[idx]

    t_med = _quantile_t(0.5)
    iqr_t = (_quantile_t(0.75) - _quantile_t(0.25)).clamp(min=1e-3)
    c = (mean_f + t_med * std_f).view(P, n_ss, n_fs)
    sigma = (std_f * iqr_t / 1.349).view(P, n_ss, n_fs)
    # detector-law floor: a pixel with too few samples shouldn't get a tiny σ_p
    sigma = torch.maximum(sigma, noise_model.sigma(c.clamp(min=0.0)))
    n_samples = cnt.to(torch.float32).view(P, n_ss, n_fs)
    # masked / dead pixels: neutral centre, law-based scale (never NaN downstream)
    c = torch.where(good, c, torch.zeros_like(c))
    return PerPixelModel(c=c.cpu(), sigma=sigma.cpu(), n_samples=n_samples.cpu())


# ──────────────────────────── empirical noise ───────────────────────────────
@dataclass
class EmpiricalNoise:
    """Empirical CDF of the score residual, conditioned on ``μ = b(r)``.

    ``Pn[m, e]`` is ``P(residual in e-bin | μ in m-bin)`` and ``cdf_lo`` its exclusive
    cumulative ``F(·<)``; ``medges`` / ``egrid`` are the μ- and residual-bin edges. The
    residual is whatever :func:`_score_residual` produced at calibration time (photons for
    the ``f(p)``/raw path, σ units for the per-pixel path), so :func:`whitened_residual`
    must recompute it the same way. ``normal_score`` maps a residual to ``N(0,1)`` via the
    mid-PIT normal score ``Φ⁻¹(F)``.
    """

    Pn: torch.Tensor                 # (NMU, NE) float64
    cdf_lo: torch.Tensor             # (NMU, NE) float64, exclusive CDF
    medges: torch.Tensor             # (NMU+1,)
    egrid: torch.Tensor              # (NE+1,)
    noise_model: NoiseModel
    logratio: torch.Tensor | None = None   # (NMU, NE) Bragg-vs-bg log-LR ℓ(e|μ), if built

    def to(self, device) -> "EmpiricalNoise":
        return EmpiricalNoise(Pn=self.Pn.to(device), cdf_lo=self.cdf_lo.to(device),
                              medges=self.medges.to(device), egrid=self.egrid.to(device),
                              noise_model=self.noise_model,
                              logratio=None if self.logratio is None else self.logratio.to(device))

    def _mbin(self, b: torch.Tensor) -> torch.Tensor:
        return torch.bucketize(b, self.medges[1:-1]).clamp(0, self.Pn.shape[0] - 1)

    def _ebin(self, e: torch.Tensor) -> torch.Tensor:
        return torch.bucketize(e, self.egrid[1:-1]).clamp(0, self.Pn.shape[1] - 1)

    def cdf(self, e: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Mid-PIT CDF value ``u = F(e | μ=b)`` ∈ (0,1) — Uniform(0,1) under background
        (the PIT calibration statistic). ``e``/``b`` flat."""
        mb = self._mbin(b)
        eb = self._ebin(e)
        return (self.cdf_lo[mb, eb] + 0.5 * self.Pn[mb, eb]).clamp(1e-6, 1 - 1e-6)

    def survival(self, e: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Right-tail ``q = P(E ≥ e | μ=b)`` (for false-positive-rate calibration)."""
        mb = self._mbin(b)
        eb = self._ebin(e)
        return (1.0 - self.cdf_lo[mb, eb]).clamp(min=1e-9)

    def normal_score(self, e: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """``Φ⁻¹(F(e|μ=b))`` via mid-PIT — ``N(0,1)`` under background. ``e``/``b`` flat."""
        return (_SQRT2 * torch.erfinv(2 * self.cdf(e, b) - 1)).to(torch.float32)

    def bragg_loglr(self, e: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Per-pixel Bragg-vs-background log-LR ``ℓ = log[P_peak(e|μ)/P_0(e|μ)]`` (design
        doc §2). ≈0 on background, >0 on positive Bragg-like excursions (growing with
        intensity), <0 on subtractive shadows. Convolved by the spot kernel downstream,
        this is the GLRT-style detector that ranks weak peaks above the white null —
        the signal term the null-only normal-score lacks. Requires ``logratio`` built."""
        return self.logratio[self._mbin(b), self._ebin(e)].to(torch.float32)


def calibrate_empirical_noise(
    reader,
    geom: Geometry,
    noise_model: NoiseModel,
    frame_indices,
    *,
    fixed_pattern: torch.Tensor | None = None,
    per_pixel: PerPixelModel | None = None,
    bragg_mean: float | None = None,
    n_mu: int = 48,
    n_iters: int = 5,
    bin_width: float = 1.0,
    device: str = "cpu",
    mask=None,
) -> EmpiricalNoise:
    """Build the empirical ``P(residual | μ)`` table by streaming over ``frame_indices``.

    The residual is :func:`_score_residual` (per-pixel standardized when ``per_pixel`` is
    given, else the ``f(p)``-corrected photon residual). μ-bin edges are ``n_mu`` quantiles
    of ``b(r)`` sampled from the first few frames; the residual grid is fixed
    (:data:`_STD_GRID` / :data:`_RAW_GRID`). Construction lifts ``converter_mf.py`` /
    ``empirical_likelihood.py`` verbatim (validated PIT KS ≈ 0.0006).
    """
    dev = torch.device(device)
    good = (reader.mask_panels() if mask is None else mask).to(torch.bool).to(dev)
    P, n_ss, n_fs = good.shape
    r_flat = geom.pixel_radius_px.reshape(-1).to(torch.float32).to(dev)
    gf = good.reshape(-1)
    ng = int(gf.sum())
    frame_indices = [int(i) for i in frame_indices]

    pp = per_pixel.to(dev) if per_pixel is not None else None
    fp = fixed_pattern.to(dev).reshape(-1) if fixed_pattern is not None else None
    cflat = pp.c.reshape(-1) if pp is not None else None
    sflat = pp.sigma.reshape(-1) if pp is not None else None

    ELO, EHI, NE = _STD_GRID if pp is not None else _RAW_GRID
    egrid = torch.linspace(ELO, EHI, NE + 1, device=dev)

    def _resid(v, b, sig):
        return _score_residual(v, b, sig, fixed_pattern=fp, c=cflat, scale=sflat)

    # μ edges from b(r) on the first few frames
    bsamp = []
    for i in frame_indices[:8]:
        v = _load_panels(reader, i, dev).reshape(-1)
        b = radial_background(v, gf, r_flat, n_iters=n_iters, bin_width=bin_width)
        s = b[gf]
        bsamp.append(s[torch.linspace(0, s.numel() - 1, min(s.numel(), 1_000_000),
                                      device=dev).long()])
    medges = torch.unique(torch.quantile(torch.cat(bsamp),
                                          torch.linspace(0, 1, n_mu + 1, device=dev)))
    NMU = int(medges.numel() - 1)

    def mbin(b): return torch.bucketize(b, medges[1:-1]).clamp(0, NMU - 1)
    def ebin(e): return torch.bucketize(e, egrid[1:-1]).clamp(0, NE - 1)

    Pem = torch.zeros(NMU, NE, dtype=torch.float64, device=dev)
    ones = torch.ones(ng, dtype=torch.float64, device=dev)
    for i in frame_indices:
        v = _load_panels(reader, i, dev).reshape(-1)
        b = radial_background(v, gf, r_flat, n_iters=n_iters, bin_width=bin_width)
        sig = noise_model.sigma(b).clamp(min=1e-6)
        e = _resid(v, b, sig)
        flat = mbin(b[gf]) * NE + ebin(e[gf])
        Pem.view(-1).scatter_add_(0, flat, ones)
    Pn = Pem / Pem.sum(1, keepdim=True).clamp(min=1)
    cdf_lo = torch.cumsum(Pn, 1) - Pn
    logratio = _bragg_logratio(Pn, egrid, bragg_mean) if bragg_mean is not None else None
    return EmpiricalNoise(Pn=Pn.cpu(), cdf_lo=cdf_lo.cpu(),
                          medges=medges.cpu(), egrid=egrid.cpu(), noise_model=noise_model,
                          logratio=None if logratio is None else logratio.cpu())


# ─────────────────────────── calibration bundle ─────────────────────────────
@dataclass
class Calibration:
    """Everything the per-frame finder needs to whiten in empirical mode — threaded as
    one object through ``find_spots_on_frame`` so the legacy path stays a plain ``None``.

    ``per_pixel`` (constant channel) XOR ``fixed_pattern`` (legacy ``f(p)``) selects the
    centering; ``basis`` is the Phase-2 low-rank background (``None`` → radial ``b(r)``).
    """

    noise_model: NoiseModel
    empirical: EmpiricalNoise
    per_pixel: PerPixelModel | None = None
    fixed_pattern: torch.Tensor | None = None
    basis: object | None = None

    def to(self, device) -> "Calibration":
        return Calibration(
            noise_model=self.noise_model,
            empirical=self.empirical.to(device),
            per_pixel=self.per_pixel.to(device) if self.per_pixel is not None else None,
            fixed_pattern=(self.fixed_pattern.to(device)
                           if self.fixed_pattern is not None else None),
            basis=self.basis.to(device) if self.basis is not None else None,
        )


# ──────────────────────────── eval transform ────────────────────────────────
@torch.no_grad()
def whitened_residual(
    panels: torch.Tensor,
    good: torch.Tensor,
    radius: torch.Tensor,
    *,
    noise_model: NoiseModel,
    empirical: EmpiricalNoise,
    fixed_pattern: torch.Tensor | None = None,
    per_pixel: PerPixelModel | None = None,
    basis=None,
    n_iters: int = 5,
    bin_width: float = 1.0,
) -> torch.Tensor:
    """Per-frame whitened residual ``w = Φ⁻¹(F(e|μ))`` — ``N(0,1)`` under background.

    The empirical-calibration analogue of the legacy ``(I − b)/σ − f(p)`` whitening in
    :func:`~torchsx.datareduction.peakfinder.bayes_score_map`. The shot-varying background
    ``b`` is :func:`~torchsx.datareduction.peakfinder.radial_background` (or, when ``basis``
    is given, the low-rank reconstruction — Phase 2). ``e`` is the matching
    :func:`_score_residual`; masked pixels → 0. Returns ``(P, ss, fs)``.

    ``empirical`` / ``per_pixel`` / ``fixed_pattern`` are expected **already on
    ``panels.device``** (migrate once per run, like the matched-filter operator) — this is
    a per-frame hot path, so it does no device moves.
    """
    P, n_ss, n_fs = panels.shape
    v = panels.to(torch.float32).reshape(-1)
    gf = good.to(torch.bool).reshape(-1)
    r_flat = radius.to(torch.float32).reshape(-1)
    if basis is not None:                                  # Phase 2: low-rank shot-varying bg
        resid = v - per_pixel.c.reshape(-1) if per_pixel is not None else v
        b = basis.reconstruct(basis.project(resid, gf)).reshape(-1)
    else:
        b = radial_background(v, gf, r_flat, n_iters=n_iters, bin_width=bin_width)
    sig = noise_model.sigma(b).clamp(min=1e-6)
    w = _whiten_from_bg(v, b, sig, gf, empirical=empirical,
                        per_pixel=per_pixel, fixed_pattern=fixed_pattern)
    return w.view(P, n_ss, n_fs)
