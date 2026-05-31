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

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..geometry import Geometry
from .peakfinder import radial_background


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


def _frame_residuals(reader, geom: Geometry, i: int, *, n_iters, bin_width):
    """(mu, e, good) flat tensors for frame i: mu=b(r), e=I-b(r)."""
    panels = reader.panels(i).to(torch.float32)
    good = reader.mask_panels().to(torch.bool)
    P, n_ss, n_fs = panels.shape
    r = geom.pixel_radius_px.reshape(P, n_ss, n_fs).to(torch.float32)
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
    max_pool_px: int = 6_000_000,
    device: str = "cpu",
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
        mu, e, g = _frame_residuals(reader, geom, int(i), n_iters=n_iters, bin_width=bin_width)
        sel = (g & torch.isfinite(e)).nonzero(as_tuple=False).squeeze(1)
        if sel.numel() > per_frame_cap:                  # subsample pixels
            sub = torch.randint(sel.numel(), (per_frame_cap,))
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

    def sf(self, z: torch.Tensor) -> torch.Tensor:
        """Null survival function 1 - F0(z) via the stored sample."""
        n = self.z_sorted.numel()
        rank = torch.searchsorted(self.z_sorted, z.to(self.z_sorted.device), right=True)
        return (n - rank).clamp(min=0).to(torch.float32) / max(n, 1)

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
    max_keep: int = 2_000_000,
    device: str = "cpu",
) -> EmpiricalNull:
    """Pool background local-maximum significance over frames → :class:`EmpiricalNull`.

    Most frames (~96%) and most maxima even on hits are background, so the pooled
    local-max significance is null-dominated; the handful of real-peak maxima are a
    negligible high-z tail. A reservoir caps memory at ``max_keep`` z-values while the
    true maximum count feeds ``n_lmax_per_frame``.
    """
    pool, total_lmax, n_frames = [], 0, 0
    per_frame_cap = max(1, max_keep // max(1, len(frame_indices)))
    for i in frame_indices:
        panels = reader.panels(int(i)).to(torch.float32).to(device)
        good = reader.mask_panels().to(torch.bool).to(device)
        P, n_ss, n_fs = panels.shape
        r = geom.pixel_radius_px.reshape(P, n_ss, n_fs).to(torch.float32).to(device)
        val = panels
        b = radial_background(val.reshape(-1), good.reshape(-1), r.reshape(-1),
                              n_iters=n_iters, bin_width=bin_width).view(P, n_ss, n_fs)
        sig = noise_model.sigma(b)
        z = torch.where(good, (val - b) / sig.clamp(min=1e-6),
                        torch.full_like(val, float("-inf")))
        d = peak_min_distance
        pooled = F.max_pool2d(z.unsqueeze(1), kernel_size=2 * d + 1, stride=1, padding=d)
        is_max = (z == pooled.squeeze(1)) & good
        zmax = z[is_max]
        total_lmax += int(zmax.numel())
        n_frames += 1
        if zmax.numel() > per_frame_cap:               # reservoir subsample
            sub = torch.randint(zmax.numel(), (per_frame_cap,), device=zmax.device)
            zmax = zmax[sub]
        pool.append(zmax.cpu())
    z_sorted = torch.sort(torch.cat(pool))[0]
    return EmpiricalNull(z_sorted=z_sorted,
                         n_lmax_per_frame=total_lmax / max(1, n_frames))
