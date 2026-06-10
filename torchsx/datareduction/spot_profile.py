"""Anisotropic Bragg-spot profile and the matched-filter detection score.

A Bragg spot on a still JUNGFRAU frame is not a single hot pixel but a small
*blob* whose shape is well modelled by an **anisotropic 2-D Gaussian** with
distinct widths along the **radial** direction (away from the beam center) and
the **azimuthal / tangential** direction. The two widths capture, respectively,
spectral-bandwidth / detector-distance streaking (radial) and
mosaicity / divergence (tangential), and both generally **grow with resolution**
— so the profile is stored as a *curve* over radius: width control points
``σ_radial(r)`` / ``σ_azimuthal(r)`` linearly interpolated between them.

The matched filter that detects such spots is the core of the *fully Bayesian*
finder (see :func:`~torchsx.datareduction.peakfinder.find_peaks_bayes`): treating
each pixel's whitened residual ``w = (I − b) / σ`` as standardized evidence
against the background hypothesis (``w ~ N(0,1)`` under background), the
log-likelihood that a spot of profile ``g`` sits centred at pixel ``c`` is, to
leading order, **linear** in ``w`` and given by the convolution ``(g ⋆ w)(c)``.
Normalizing by the (mask-corrected) kernel norm yields a statistic with unit
variance under the null:

    S(c) = (g ⋆ w)(c) / sqrt( (g² ⋆ good)(c) )

This is the matched-filter / GLRT score and, equivalently, the one-pass
mean-field approximation of a Markov-random-field where neighbouring pixels share
their spot/background label — the convolution *is* the neighbour coupling.

Because the profile is oriented along the (position-dependent) radial direction
*and* varies with radius, the score is evaluated with an **oriented, banded filter
bank**: for each radius band and each orientation ``θ_j ∈ [0, π)`` an anisotropic
kernel is convolved with ``w``, and every pixel's score is gathered from the
band(s) bracketing its radius and the orientation(s) bracketing its local radial
angle (bilinear interpolation). The width curves are normally
:func:`~torchsx.datareduction.noise_model.calibrate_spot_profile`-bootstrapped
from the data, with :meth:`SpotProfile.constant` (isotropic) as the fallback.

This module is pure-torch and imports nothing from ``peakfinder`` / ``noise_model``
to keep the dependency direction one-way (those modules import this one).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _oriented_bank(sigma_radial: float, sigma_azimuthal: float, R: int,
                   n_orient: int, device, dtype=torch.float32):
    """L2-normalized oriented anisotropic-Gaussian kernels + their squares.

    Returns ``(g, g2)``, each ``(n_orient, 1, 2R+1, 2R+1)``. Unit L2 norm makes
    the unmasked matched-filter response unit-variance under an ``N(0,1)`` null.
    """
    offs = torch.arange(-R, R + 1, device=device, dtype=dtype)
    # dss/dfs are slow/fast-scan *pixel* offsets; the steering angle (geom.angle_map)
    # is the radial direction already resolved into this pixel frame, so u/v below are
    # the radial/tangential axes on the pixel grid regardless of panel orientation.
    dss, dfs = torch.meshgrid(offs, offs, indexing="ij")
    sr2 = max(float(sigma_radial), 1e-3) ** 2
    sa2 = max(float(sigma_azimuthal), 1e-3) ** 2
    thetas = torch.arange(n_orient, device=device, dtype=dtype) * (math.pi / n_orient)
    kernels = []
    for th in thetas:
        ct, st = torch.cos(th), torch.sin(th)
        u = dfs * ct + dss * st          # along radial
        v = -dfs * st + dss * ct         # along azimuthal/tangential
        g = torch.exp(-(u * u) / (2.0 * sr2) - (v * v) / (2.0 * sa2))
        g = g / g.norm().clamp(min=1e-12)
        kernels.append(g)
    g = torch.stack(kernels, 0).unsqueeze(1)
    return g, g * g


@dataclass(frozen=True, eq=False)
class SpotProfile:
    """Anisotropic Gaussian spot profile, as width curves over radius.

    ``radii`` are ascending control-point radii (px); ``sigma_radial`` /
    ``sigma_azimuthal`` are the matching Gaussian widths (px) along the radial and
    tangential directions at each control point, linearly interpolated (flat
    extrapolation) in between. A single control point ⇒ constant (radius-
    independent) widths. ``kernel_radius`` is the half-size of the (square) kernel
    support; ``n_orient`` is the number of orientations in the steerable bank (a
    Gaussian is 180°-symmetric, so they span ``[0, π)``).
    """

    radii: torch.Tensor
    sigma_radial: torch.Tensor
    sigma_azimuthal: torch.Tensor
    kernel_radius: int = 3
    n_orient: int = 12

    @classmethod
    def constant(cls, sigma_radial: float = 1.0, sigma_azimuthal: float = 1.0,
                 *, kernel_radius: int = 3, n_orient: int = 12) -> "SpotProfile":
        """Radius-independent profile (single control point) — the fallback shape."""
        return cls(
            radii=torch.zeros(1),
            sigma_radial=torch.tensor([float(sigma_radial)]),
            sigma_azimuthal=torch.tensor([float(sigma_azimuthal)]),
            kernel_radius=kernel_radius, n_orient=n_orient,
        )

    @property
    def n_bands(self) -> int:
        return int(self.radii.numel())


# Isotropic default used when the bootstrap can't find enough spots.
SpotProfile.FALLBACK = SpotProfile.constant(1.0, 1.0)


def _band_weights(radius: torch.Tensor, radii: torch.Tensor, eps: float = 1e-6):
    """Per-pixel ``(lo_index, frac)`` bracketing the radius control points.

    Band ``lo`` gets weight ``1 − frac`` and band ``lo+1`` gets ``frac``; flat
    extrapolation outside ``[radii[0], radii[-1]]``.
    """
    B = radii.numel()
    flat = radius.reshape(-1).contiguous()
    hi = torch.searchsorted(radii, flat).clamp(1, B - 1).reshape(radius.shape)
    lo = hi - 1
    denom = (radii[hi] - radii[lo]).clamp(min=eps)
    frac = ((radius - radii[lo]) / denom).clamp(0.0, 1.0)
    return lo, frac


def matched_filter_score(
    w: torch.Tensor,
    good: torch.Tensor,
    angle: torch.Tensor,
    profile: SpotProfile,
    radius: torch.Tensor | None = None,
) -> torch.Tensor:
    """Oriented, radius-banded matched-filter score ``S`` (unit-variance null).

    Parameters
    ----------
    w : (P, ss, fs) float — whitened residual ``(I − b) / σ`` with masked pixels
        already set to 0.
    good : (P, ss, fs) — True/1 = good pixel (mask-corrects the kernel norm).
    angle : (P, ss, fs) float — per-pixel radial angle ``atan2(y, x)`` (radians).
    profile : :class:`SpotProfile`.
    radius : (P, ss, fs) float, optional — per-pixel radius (px); selects the width
        band. Ignored (and unnecessary) for a constant single-band profile.

    Returns
    -------
    S : (P, ss, fs) float — matched-filter score, ``≈ N(0,1)`` under background.
    """
    P, n_ss, n_fs = w.shape
    R = int(profile.kernel_radius)
    eps = 1e-6

    wf = w.to(torch.float32).unsqueeze(1)
    goodf = good.to(torch.float32).unsqueeze(1)

    # orientation bracket + interpolation fraction (radial-angle steering)
    n = profile.n_orient
    pos = angle.remainder(math.pi) / (math.pi / n)
    olo = torch.floor(pos).long() % n
    oup = (olo + 1) % n
    ofrac = pos - torch.floor(pos)

    B = profile.n_bands
    radii = profile.radii.to(w.device)
    sr, sa = profile.sigma_radial, profile.sigma_azimuthal
    if B == 1 or radius is None:
        rlo = rfrac = None
    else:
        rlo, rfrac = _band_weights(radius.to(torch.float32), radii, eps)

    S = torch.zeros(P, n_ss, n_fs, device=w.device, dtype=torch.float32)
    for b in range(B):
        if rlo is None:
            rweight = None
        else:
            rweight = ((rlo == b).to(torch.float32) * (1.0 - rfrac)
                       + ((rlo + 1) == b).to(torch.float32) * rfrac)
            if float(rweight.max()) == 0.0:
                continue
        g, g2 = _oriented_bank(float(sr[b]), float(sa[b]), R, n, w.device)
        # all orientations in one batched conv (P, n_orient, ss, fs) — far cheaper
        # on CPU than n separate small convs.
        num = F.conv2d(wf, g, padding=R)
        den = F.conv2d(goodf, g2, padding=R).clamp(min=eps).sqrt()
        s = num / den
        for j in range(n):
            aweight = (olo == j).to(torch.float32) * (1.0 - ofrac) + (oup == j).to(torch.float32) * ofrac
            contrib = s[:, j] * aweight
            S = S + (contrib if rweight is None else contrib * rweight)
    return S


# ───────────────────── fused GPU matched filter (Triton) ─────────────────────
# The conv path above computes B·n_orient full convolutions and a B·n_orient
# accumulation loop, ~145 ms/frame on an A100 — and streams ~800 MB intermediates
# per band. But the score's only frame-dependent input is ``w``. Even the
# per-(band,orient) denominator den_bo(p) = sqrt(Σ_k g_bo[k]²·good[p+k]) is
# frame-independent (``good`` is the constant mask), so we precompute, per pixel,
# the ≤4 active combo coefficients ``α_c = (band_w·orient_w)/den_c`` ONCE and the
# per-frame kernel is just
#
#     num_c(p) = Σ_k g_{combo_c}[k]·w[p+k]        S(p) = Σ_c α_c(p)·num_c(p)
#
# — one w-load + 4 tiny cache-resident gtab gathers per tap, no mask touch, no
# sqrt. Reads only the w-patches (L2 reuse over the overlapping windows); no dense
# per-pixel kernel, no unfold. ~7.5 ms/frame, autotuned, exact-fp32 (the conv path
# is TF32-lossy on Ampere; this is the accurate one). Falls back to the conv path
# off-CUDA / without Triton.
_mf_triton = None


def _load_mf_triton():
    global _mf_triton
    if _mf_triton is not None:
        return _mf_triton
    import triton
    import triton.language as tl

    configs = [
        triton.Config({"BLOCK_J": bj}, num_warps=nw, num_stages=ns)
        for bj in (128, 256)
        for nw in (1, 2, 4)
        for ns in (2, 3, 4)
    ]

    @triton.autotune(configs=configs, key=["ss", "fs", "R", "n_orient", "B"])
    @triton.jit
    def kernel(
        w_ptr, S_ptr,                                 # (P,ss,fs) float32
        rlo_ptr, olo_ptr, oup_ptr,                    # (P,ss,fs) int32 brackets
        a0_ptr, a1_ptr, a2_ptr, a3_ptr,               # (P,ss,fs) float32 combo coeffs
        gtab_ptr, ss, fs, n_orient, B,
        R: tl.constexpr, BLOCK_J: tl.constexpr,
    ):
        pid0 = tl.program_id(0)            # pp*ss + i
        pp = pid0 // ss
        i = pid0 % ss
        js = tl.program_id(1) * BLOCK_J + tl.arange(0, BLOCK_J)
        jmask = js < fs
        Kw = (2 * R + 1) * (2 * R + 1)
        panel = pp * ss * fs
        centre = panel + i * fs + js

        rlo = tl.load(rlo_ptr + centre, mask=jmask, other=0)
        olo = tl.load(olo_ptr + centre, mask=jmask, other=0)
        oup = tl.load(oup_ptr + centre, mask=jmask, other=0)
        a0 = tl.load(a0_ptr + centre, mask=jmask, other=0.0)
        a1 = tl.load(a1_ptr + centre, mask=jmask, other=0.0)
        a2 = tl.load(a2_ptr + centre, mask=jmask, other=0.0)
        a3 = tl.load(a3_ptr + centre, mask=jmask, other=0.0)
        bhi = tl.minimum(rlo + 1, B - 1)

        b0 = (rlo * n_orient + olo) * Kw
        b1 = (rlo * n_orient + oup) * Kw
        b2 = (bhi * n_orient + olo) * Kw
        b3 = (bhi * n_orient + oup) * Kw

        num0 = tl.zeros((BLOCK_J,), tl.float32)
        num1 = tl.zeros((BLOCK_J,), tl.float32)
        num2 = tl.zeros((BLOCK_J,), tl.float32)
        num3 = tl.zeros((BLOCK_J,), tl.float32)

        W = 2 * R + 1
        for k in tl.range(0, Kw):       # one real (non-unrolled) loop over taps
            di = k // W - R
            dj = k % W - R
            ii = i + di
            jj = js + dj
            m = jmask & (ii >= 0) & (ii < ss) & (jj >= 0) & (jj < fs)
            wv = tl.load(w_ptr + panel + ii * fs + jj, mask=m, other=0.0)
            num0 += tl.load(gtab_ptr + b0 + k) * wv
            num1 += tl.load(gtab_ptr + b1 + k) * wv
            num2 += tl.load(gtab_ptr + b2 + k) * wv
            num3 += tl.load(gtab_ptr + b3 + k) * wv

        S = a0 * num0 + a1 * num1 + a2 * num2 + a3 * num3
        tl.store(S_ptr + centre, S, mask=jmask)

    _mf_triton = (triton, kernel)
    return _mf_triton


@dataclass(frozen=True, eq=False)
class MatchedFilterOperator:
    """Precomputed, frame-independent operator for the fused matched filter.

    Built once per run from ``(angle, radius, good, profile)`` and reused for every
    frame (and by ``calibrate_null``). Holds the per-pixel band/orient brackets
    (``rlo/olo/oup``), the ≤4 per-pixel combo coefficients ``a0..a3`` (band·orient
    weight ÷ the frame-independent denominator), and the tiny oriented-Gaussian
    table ``gtab`` (``B·n_orient·Kw``). The mask is folded into ``a*`` so the
    per-frame kernel never touches it.
    """

    rlo: torch.Tensor
    olo: torch.Tensor
    oup: torch.Tensor
    a0: torch.Tensor
    a1: torch.Tensor
    a2: torch.Tensor
    a3: torch.Tensor
    gtab: torch.Tensor          # (B*n_orient*Kw,) flat
    kernel_radius: int
    n_orient: int
    n_bands: int


def build_matched_filter_operator(
    angle: torch.Tensor,
    radius: torch.Tensor,
    good: torch.Tensor,
    profile: SpotProfile,
) -> MatchedFilterOperator:
    """Precompute the fused matched-filter operator (see :class:`MatchedFilterOperator`).

    Folds the frame-independent denominator into per-pixel combo coefficients:
    ``α_c = (band_w·orient_w) / sqrt(clamp(conv(good, g²_c), eps))`` — same
    denominator as :func:`matched_filter_score`, computed once."""
    P, n_ss, n_fs = angle.shape
    R = int(profile.kernel_radius)
    n = int(profile.n_orient)
    B = int(profile.n_bands)
    Kw = (2 * R + 1) ** 2
    dev = angle.device
    eps = 1e-6

    pos = angle.remainder(math.pi) / (math.pi / n)
    olo = torch.floor(pos).long() % n
    oup = (olo + 1) % n
    ofrac = (pos - torch.floor(pos)).to(torch.float32)
    if B == 1:
        rlo = torch.zeros(P, n_ss, n_fs, dtype=torch.long, device=dev)
        rfrac = torch.zeros(P, n_ss, n_fs, dtype=torch.float32, device=dev)
    else:
        lo, frac = _band_weights(radius.to(torch.float32), profile.radii.to(dev), eps)
        rlo = lo.to(torch.long)
        rfrac = frac.to(torch.float32)
    bhi = (rlo + 1).clamp(max=B - 1)

    # gtab + the per-(band,orient) denominator bank (one-time; ~3.25 GB transient
    # for B=4·n=12 on the full module stack), then per-pixel gather for the 4 combos.
    gtab = torch.empty(B, n, Kw, device=dev, dtype=torch.float32)
    goodf = good.to(torch.float32).reshape(P, 1, n_ss, n_fs)
    den_all = torch.empty(P, B * n, n_ss, n_fs, device=dev, dtype=torch.float32)
    sr, sa = profile.sigma_radial, profile.sigma_azimuthal
    for b in range(B):
        g, g2 = _oriented_bank(float(sr[b]), float(sa[b]), R, n, dev)   # (n,1,2R+1,2R+1)
        gtab[b] = g.reshape(n, Kw)
        den_all[:, b * n:(b + 1) * n] = F.conv2d(goodf, g2, padding=R)
    den_all = den_all.clamp(min=eps).sqrt()

    def den_at(bidx, oidx):
        idx = (bidx * n + oidx).unsqueeze(1)        # (P,1,ss,fs)
        return torch.gather(den_all, 1, idx).squeeze(1)

    a0 = ((1.0 - rfrac) * (1.0 - ofrac)) / den_at(rlo, olo)
    a1 = ((1.0 - rfrac) * ofrac) / den_at(rlo, oup)
    a2 = (rfrac * (1.0 - ofrac)) / den_at(bhi, olo)
    a3 = (rfrac * ofrac) / den_at(bhi, oup)

    return MatchedFilterOperator(
        rlo=rlo.to(torch.int32).contiguous(),
        olo=olo.to(torch.int32).contiguous(), oup=oup.to(torch.int32).contiguous(),
        a0=a0.contiguous(), a1=a1.contiguous(), a2=a2.contiguous(), a3=a3.contiguous(),
        gtab=gtab.reshape(-1).contiguous(),
        kernel_radius=R, n_orient=n, n_bands=B,
    )


def matched_filter_score_fused(
    w: torch.Tensor,
    operator: MatchedFilterOperator,
) -> torch.Tensor:
    """Fused-Triton matched-filter score ``S`` — exact-fp32 equivalent of
    :func:`matched_filter_score`, ~20x faster on CUDA (~7.5 ms vs ~145 ms). Requires
    a CUDA ``w`` and Triton; the caller falls back to the conv path otherwise."""
    triton, kernel = _load_mf_triton()
    P, n_ss, n_fs = w.shape
    S = torch.empty_like(w, dtype=torch.float32)
    grid = lambda meta: (P * n_ss, triton.cdiv(n_fs, meta["BLOCK_J"]))
    kernel[grid](
        w.contiguous(), S,
        operator.rlo, operator.olo, operator.oup,
        operator.a0, operator.a1, operator.a2, operator.a3,
        operator.gtab, n_ss, n_fs, operator.n_orient, operator.n_bands,
        R=operator.kernel_radius,
    )
    return S
