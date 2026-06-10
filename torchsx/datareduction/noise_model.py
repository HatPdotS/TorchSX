"""Per-run detector noise model and empirical-null peak calibration.

Two calibrations, both estimated once per run from a sample of that run's frames
(no saved artifact):

1. :class:`NoiseModel` — the variance-vs-mean law ``sigma^2(mu) = a0 + a1*mu +
   a2*mu^2 + a3*mu^3`` (read floor / Poisson / super-Poisson gain-flip). A detector
   property: a function of *local intensity only*, independent of frame or region.
   Calibrated by pooling **per-frame-background residuals** binned by intensity, with
   a robust within-bin variance — NOT a per-pixel temporal variance, which would be
   contaminated by the shot-to-shot background jitter.

2. :class:`EmpiricalNull` — the null distribution of *local-maximum significance*
   under background, plus the mean number of maxima per frame. Used to turn a peak's
   significance into a **multiplicity-corrected** probability (tail FDR), so the
   look-elsewhere effect (millions of pixels → chance maxima) is controlled rather
   than ignored.

The variance form mirrors ``GaussianPolynomialVariance`` in
``torchsx/merging/likelihood/gaussian_polynomial.py`` (extended here to cubic).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..geometry import Geometry
from .peakfinder import TorchPeakFinderParams, bayes_score_map, radial_background
from .spot_profile import SpotProfile


# ─────────────────────────────── noise model ────────────────────────────────
@dataclass
class NoiseModel:
    """Detector variance law ``sigma^2(mu) = a0 + a1 mu + a2 mu^2 + a3 mu^3``.

    ``a0`` ~ read-noise floor, ``a1`` ~ gain (Poisson moment-match), ``a2``/``a3``
    capture super-Poisson excess from JUNGFRAU gain-stage misclassification. Units
    are whatever the frame data is in (here gain-corrected keV-ish), so ``a1`` comes
    out near the per-photon conversion factor.
    """

    a0: float = 0.0
    a1: float = 1.0
    a2: float = 0.0
    a3: float = 0.0
    var_floor: float = 1e-6

    def variance(self, mu: torch.Tensor) -> torch.Tensor:
        m = mu.clamp(min=0.0)
        v = self.a0 + self.a1 * m + self.a2 * m * m + self.a3 * m * m * m
        return v.clamp(min=self.var_floor)

    def sigma(self, mu: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.variance(mu))


def _load_panels(reader, i: int, device):
    """Frame ``i`` as float32 on ``device``. Uses the GPU bslz4 decode
    (``panels_gpu``, bit-exact vs h5py) on CUDA when available; CPU h5py otherwise."""
    dev = torch.device(device)
    if dev.type == "cuda" and hasattr(reader, "panels_gpu"):
        return reader.panels_gpu(int(i), dev).to(torch.float32)
    return reader.panels(int(i)).to(torch.float32).to(dev)


def _frame_residuals(reader, geom: Geometry, i: int, *, n_iters, bin_width, device="cpu",
                     mask=None):
    """(mu, e, good) flat tensors for frame i: mu=b(r), e=I-b(r). Computed on
    ``device`` (radial background runs there), matching the per-frame finder path.

    ``mask`` (a ``(P,514,fs)`` good-mask) overrides ``reader.mask_panels()`` when given.
    """
    dev = torch.device(device)
    panels = _load_panels(reader, i, dev)
    good = (reader.mask_panels() if mask is None else mask).to(torch.bool).to(dev)
    P, n_ss, n_fs = panels.shape
    r = geom.pixel_radius_px.reshape(P, n_ss, n_fs).to(torch.float32).to(dev)
    val = panels.reshape(-1)
    g = good.reshape(-1)
    b = radial_background(val, g, r.reshape(-1), n_iters=n_iters, bin_width=bin_width)
    return b, (val - b), g


def calibrate_noise(
    reader,
    geom: Geometry,
    frame_indices,
    *,
    n_bins: int = 60,
    degree: int = 3,
    n_iters: int = 5,
    bin_width: float = 1.0,
    # Kept under torch.quantile's ~16.7M-element limit (used for the bin edges);
    # the strided draw below — not the raw size — is what makes the fit reproducible.
    max_pool_px: int = 12_000_000,
    device: str = "cpu",
    mask=None,
) -> tuple[NoiseModel, dict]:
    """Fit ``sigma^2(mu)`` by pooling per-frame-background residuals over frames.

    Residual ``e = I - b(r)`` paired with local mean ``mu = b(r)``; pooled over good
    pixels and the given frames (subsampled to ``max_pool_px`` total to bound
    memory), binned by ``mu`` into ``n_bins`` quantile bins, with a robust within-bin
    variance ``(1.4826*MAD)^2`` (peaks/gain-flips rejected). Weighted polynomial LSQ
    of degree ``degree`` → ``NoiseModel``. Returns ``(model, debug)`` with the
    per-bin (mu, sigma2, count) arrays.
    """
    frame_indices = list(frame_indices)
    per_frame_cap = max(1, max_pool_px // max(1, len(frame_indices)))
    mus, es = [], []
    for i in frame_indices:
        mu, e, g = _frame_residuals(reader, geom, int(i), n_iters=n_iters,
                                    bin_width=bin_width, device=device, mask=mask)
        sel = (g & torch.isfinite(e)).nonzero(as_tuple=False).squeeze(1)
        if sel.numel() > per_frame_cap:                  # deterministic strided subsample
            # Evenly-spaced (not random) so the σ²(μ) fit is reproducible run-to-run.
            # Unseeded random draws here made the fitted coefficients — hence every
            # spot's significance z=(I−b)/σ — vary between otherwise-identical runs.
            sub = torch.linspace(0, sel.numel() - 1, per_frame_cap, device=sel.device).long()
            sel = sel[sub]
        mus.append(mu[sel].to(device))
        es.append(e[sel].to(device))
    mu = torch.cat(mus)
    e = torch.cat(es)

    # quantile bins over mu so each bin has comparable counts
    qs = torch.linspace(0, 1, n_bins + 1, device=mu.device)
    edges = torch.quantile(mu, qs)
    edges = torch.unique(edges)                       # collapse ties
    idx = torch.bucketize(mu, edges[1:-1])            # bin index per pixel
    nb = int(edges.numel() - 1)

    mu_c = torch.zeros(nb, device=mu.device)
    s2 = torch.zeros(nb, device=mu.device)
    cnt = torch.zeros(nb, device=mu.device)
    for b in range(nb):
        m = idx == b
        c = int(m.sum())
        if c < 32:
            continue
        eb = e[m]
        mu_c[b] = mu[m].median()
        mad = (eb - eb.median()).abs().median()       # residuals ~ zero-mean
        s2[b] = (1.4826 * mad) ** 2                    # robust sigma^2
        cnt[b] = c
    keep = cnt > 0
    mu_c, s2, w = mu_c[keep], s2[keep], cnt[keep]

    # weighted polynomial fit sigma^2 = sum a_k mu^k
    powers = torch.stack([mu_c ** k for k in range(degree + 1)], dim=1)   # (nb, deg+1)
    W = torch.diag(w / w.sum())
    coef = torch.linalg.lstsq(W @ powers, W @ s2.unsqueeze(1)).solution.squeeze(1)
    coef = coef.tolist() + [0.0, 0.0, 0.0, 0.0]
    model = NoiseModel(a0=coef[0], a1=coef[1], a2=coef[2], a3=coef[3])
    debug = {"mu": mu_c.cpu().tolist(), "sigma2": s2.cpu().tolist(),
             "count": w.cpu().tolist(), "coef": [model.a0, model.a1, model.a2, model.a3]}
    return model, debug


# ──────────────────────────── empirical null ────────────────────────────────
@dataclass
class EmpiricalNull:
    """Null distribution of background local-maximum significance.

    ``z_sorted`` is a (sub)sample of null maxima significances (ascending);
    ``n_lmax_per_frame`` is the mean number of local maxima per frame. Together they
    give, for a peak at significance z, the expected number of *false* maxima at
    least that significant per frame: ``n_lmax_per_frame * (1 - F0(z))``.
    """

    z_sorted: torch.Tensor
    n_lmax_per_frame: float

    def to(self, device) -> "EmpiricalNull":
        """Return a copy with ``z_sorted`` on ``device`` (e.g. migrate to GPU once)."""
        return EmpiricalNull(z_sorted=self.z_sorted.to(device),
                             n_lmax_per_frame=self.n_lmax_per_frame)

    def sf(self, z: torch.Tensor) -> torch.Tensor:
        """Null survival function 1 - F0(z) via the stored sample.

        ``z_sorted`` is kept on the working device (CUDA when calibrated on GPU);
        ``torch.searchsorted`` runs there with no host round-trip. The ``.to(...)``
        guards are no-ops when ``z`` and ``z_sorted`` already share a device.
        """
        n = self.z_sorted.numel()
        rank = torch.searchsorted(self.z_sorted, z.to(self.z_sorted.device), right=True)
        return ((n - rank).clamp(min=0).to(torch.float32) / max(n, 1)).to(z.device)

    def prob(self, zsel: torch.Tensor) -> torch.Tensor:
        """Multiplicity-corrected P(real) for a frame's selected maxima (tail FDR).

        For maxima sorted by descending z, ``E_false(z_k)=n_lmax*(1-F0(z_k))`` and
        the observed count >= z_k is the rank ``k``; ``FDR_k = E_false/k`` made
        monotone (q-value); ``p_real = 1 - q``. On a blank (observed ~ expected
        false) p->0; on a hit (real peaks pile up far above the null tail) p->1.
        """
        order = torch.argsort(zsel, descending=True)
        z = zsel[order]
        k = torch.arange(1, z.numel() + 1, device=z.device, dtype=torch.float32)
        e_false = self.n_lmax_per_frame * self.sf(z)
        fdr = (e_false / k).clamp(0.0, 1.0)
        # enforce monotone q-value down the sorted list
        q = torch.flip(torch.cummin(torch.flip(fdr, [0]), 0).values, [0])
        p = (1.0 - q).clamp(0.0, 1.0)
        out = torch.empty_like(p)
        out[order] = p
        return out


def calibrate_null(
    reader,
    geom: Geometry,
    noise_model: NoiseModel,
    frame_indices,
    *,
    peak_min_distance: int = 2,
    n_iters: int = 5,
    bin_width: float = 1.0,
    max_keep: int = 150_000_000,
    device: str = "cpu",
    profile: SpotProfile | None = None,
    clean_null: bool = True,
    mask=None,
) -> EmpiricalNull:
    """Pool background local-maximum significance over frames → :class:`EmpiricalNull`.

    Most frames (~96%) and most maxima even on hits are background, so the pooled
    local-max significance is null-dominated; the handful of real-peak maxima are a
    negligible high-z tail. A reservoir caps memory at ``max_keep`` z-values while the
    true maximum count feeds ``n_lmax_per_frame``.

    By default the significance is the per-pixel ``z = (I − b)/σ`` (for the z-map
    finder :func:`~torchsx.datareduction.peakfinder.find_peaks_prob`). When a
    ``profile`` is supplied the significance is instead the oriented matched-filter
    score ``S`` (for :func:`~torchsx.datareduction.peakfinder.find_peaks_bayes`), so
    the FDR null matches exactly the statistic that finder ranks on.
    """
    bayes = profile is not None
    operator = None
    if bayes:
        params = TorchPeakFinderParams(
            n_bg_iters=n_iters, bin_width=bin_width, peak_min_distance=peak_min_distance,
        )
        angle_full = geom.pixel_angle_rad.to(torch.float32).to(device)
        # Build the fused matched-filter operator ONCE and reuse it for every
        # calibration frame — the null is computed on exactly the score the finder
        # ranks on, so it must use the same fast path (geometry+mask+profile fixed).
        if torch.device(device).type == "cuda":
            from .spot_profile import build_matched_filter_operator
            P0 = reader.n_panels
            n_ss0, n_fs0 = 514, geom.detector_shape[1]
            good0 = (reader.mask_panels() if mask is None else mask).to(torch.bool).to(device)
            r0 = geom.pixel_radius_px.reshape(P0, n_ss0, n_fs0).to(torch.float32).to(device)
            ang0 = angle_full.reshape(P0, n_ss0, n_fs0)
            operator = build_matched_filter_operator(ang0, r0, good0, profile)
    pool, total_lmax, n_frames = [], 0, 0
    per_frame_cap = max(1, max_keep // max(1, len(frame_indices)))
    for i in frame_indices:
        panels = _load_panels(reader, int(i), device)
        good = (reader.mask_panels() if mask is None else mask).to(torch.bool).to(device)
        P, n_ss, n_fs = panels.shape
        r = geom.pixel_radius_px.reshape(P, n_ss, n_fs).to(torch.float32).to(device)
        val = panels
        if bayes:
            angle = angle_full.reshape(P, n_ss, n_fs)
            z, _b, _g = bayes_score_map(val, good, r, angle, params,
                                        noise_model=noise_model, profile=profile,
                                        operator=operator)
        else:
            b = radial_background(val.reshape(-1), good.reshape(-1), r.reshape(-1),
                                  n_iters=n_iters, bin_width=bin_width).view(P, n_ss, n_fs)
            sig = noise_model.sigma(b)
            z = torch.where(good, (val - b) / sig.clamp(min=1e-6),
                            torch.full_like(val, float("-inf")))
        d = peak_min_distance
        # Pool the null from background TROUGHS — local maxima of −z. Spots are
        # positive excursions, so −z contains none of them: this is a
        # contamination-free random-fluctuation null even on hit-rich runs (its
        # depth distribution mirrors the background's positive maxima by the
        # near-symmetry of the background; measured skew ~0.04). The old positive
        # path (clean_null=False) pools real Bragg peaks into the high-z tail and
        # biases p low whenever the calibration frames contain crystals.
        score = torch.where(good, -z, torch.full_like(z, float("-inf"))) if clean_null else z
        pooled = F.max_pool2d(score.unsqueeze(1), kernel_size=2 * d + 1, stride=1, padding=d)
        is_max = (score == pooled.squeeze(1)) & good
        zmax = score[is_max]
        total_lmax += int(zmax.numel())
        n_frames += 1
        if zmax.numel() > per_frame_cap:               # deterministic strided subsample
            # Evenly-spaced (not random) so the empirical null — especially its high-z
            # tail that sets prob(z) — is reproducible. The default cap now keeps
            # ~all maxima (~772k/frame here); only an unusually large pool subsamples,
            # and then deterministically.
            sub = torch.linspace(0, zmax.numel() - 1, per_frame_cap, device=zmax.device).long()
            zmax = zmax[sub]
        pool.append(zmax)                              # keep on the working device
    z_sorted = torch.sort(torch.cat(pool))[0]
    return EmpiricalNull(z_sorted=z_sorted,
                         n_lmax_per_frame=total_lmax / max(1, n_frames))


def calibrate_spot_profile(
    reader,
    geom: Geometry,
    noise_model: NoiseModel,
    frame_indices,
    *,
    init: SpotProfile | None = None,
    min_snr: float = 8.0,
    min_spots: int = 200,
    n_radius_bins: int = 4,
    min_per_bin: int = 30,
    window: int = 4,
    n_iters: int = 5,
    bin_width: float = 1.0,
    device: str = "cpu",
    mask=None,
) -> SpotProfile:
    """Bootstrap the anisotropic spot-width **curves** σ_radial(r), σ_azimuthal(r).

    Detects high-SNR peaks on the calibration frames with the isotropic ``init``
    profile, then measures, for each bright peak, the background-subtracted,
    intensity-weighted second moments of the local pixel patch in the
    **(radial, tangential)** basis (the moment along an axis equals that axis'
    Gaussian variance). Peaks are pooled, **binned by radius** into ``n_radius_bins``
    quantile bins, and the robust (median) width per bin gives a control point of
    the radius curve — capturing the growth of (especially radial) spot size with
    resolution. The matched-filter kernel radius is sized to the widest band.

    **Fallbacks**: if fewer than ``min_spots`` bright peaks are found, returns
    ``init`` unchanged; bins with fewer than ``min_per_bin`` peaks are dropped, and
    if only one survives the result is a constant (radius-independent) profile.
    """
    init = init or SpotProfile.FALLBACK
    angle_full = geom.pixel_angle_rad.to(torch.float32).to(device)
    var_r, var_a, rad_pk = [], [], []
    n_spots = 0
    for i in frame_indices:
        panels = _load_panels(reader, int(i), device)
        good = (reader.mask_panels() if mask is None else mask).to(torch.bool).to(device)
        P, n_ss, n_fs = panels.shape
        r = geom.pixel_radius_px.reshape(P, n_ss, n_fs).to(torch.float32).to(device)
        b = radial_background(panels.reshape(-1), good.reshape(-1), r.reshape(-1),
                              n_iters=n_iters, bin_width=bin_width).view(P, n_ss, n_fs)
        sig = noise_model.sigma(b)
        z = torch.where(good, (panels - b) / sig.clamp(min=1e-6),
                        torch.full_like(panels, float("-inf")))
        # bright, isolated local maxima (3×3) above the SNR cut
        pooled = F.max_pool2d(z.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        is_max = (z == pooled) & good & (z > min_snr)
        idx = is_max.reshape(-1).nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        pk_panel = idx // (n_ss * n_fs)
        rem = idx % (n_ss * n_fs)
        pk_ss = rem // n_fs
        pk_fs = rem % n_fs
        theta = angle_full.reshape(-1)[idx]                   # radial angle at each peak
        r_at = r.reshape(-1)[idx]                             # radius at each peak

        w = window
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
        ibs = ((panels.reshape(-1)[flat_win] - b.reshape(-1)[flat_win]).clamp(min=0.0)
               * (good.reshape(-1)[flat_win] & valid).to(torch.float32))   # (M, W)
        wsum = ibs.sum(dim=1).clamp(min=1e-6)
        # intensity-weighted centroid offsets within the window
        cen_ss = (ibs * doff_ss[None, :].to(torch.float32)).sum(1) / wsum
        cen_fs = (ibs * doff_fs[None, :].to(torch.float32)).sum(1) / wsum
        dss = doff_ss[None, :].to(torch.float32) - cen_ss[:, None]
        dfs = doff_fs[None, :].to(torch.float32) - cen_fs[:, None]
        # rotate offsets into the local (radial u, tangential v) frame
        ct, st = torch.cos(theta)[:, None], torch.sin(theta)[:, None]
        u = dfs * ct + dss * st
        v = -dfs * st + dss * ct
        vr = (ibs * u * u).sum(1) / wsum
        va = (ibs * v * v).sum(1) / wsum
        var_r.append(vr)
        var_a.append(va)
        rad_pk.append(r_at)
        n_spots += int(idx.numel())

    if n_spots < min_spots:
        return init
    vr = torch.cat(var_r).clamp(min=0.0)
    va = torch.cat(var_a).clamp(min=0.0)
    rr = torch.cat(rad_pk)

    # quantile radius bins → one (radius, σ_radial, σ_azimuthal) control point each
    nb = max(1, int(n_radius_bins))
    qs = torch.linspace(0.0, 1.0, nb + 1, device=rr.device)
    edges = torch.quantile(rr, qs)
    radii, sr_pts, sa_pts = [], [], []
    for k in range(nb):
        lo, hi = edges[k], edges[k + 1]
        sel = (rr >= lo) & (rr <= hi) if k == nb - 1 else (rr >= lo) & (rr < hi)
        if int(sel.sum()) < min_per_bin:
            continue
        radii.append(float(rr[sel].median()))
        sr_pts.append(max(float(torch.sqrt(vr[sel].median())), 0.5))
        sa_pts.append(max(float(torch.sqrt(va[sel].median())), 0.5))
    if not radii:                                   # too sparse per bin → global median
        radii = [float(rr.median())]
        sr_pts = [max(float(torch.sqrt(vr.median())), 0.5)]
        sa_pts = [max(float(torch.sqrt(va.median())), 0.5)]

    widest = max(max(sr_pts), max(sa_pts))
    kernel_radius = int(min(6, max(init.kernel_radius, math.ceil(2.5 * widest))))
    return SpotProfile(
        radii=torch.tensor(radii),
        sigma_radial=torch.tensor(sr_pts),
        sigma_azimuthal=torch.tensor(sa_pts),
        kernel_radius=kernel_radius, n_orient=init.n_orient,
    )
