"""Per-observation Poisson noise model for SFX integration.

Provides σ²_poisson per observation from CrystFEL ring-integration geometry
(peak counts, background, integration radii) and a gain calibration using
off-Bragg observations, or the negative intensities when a source has none.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor


def compute_sigma2_LN(alpha: Tensor, beta: Tensor) -> Tensor:
    """Log-space variance of GenGauss noise (first-order approximation):
    σ²_LN = α² · Γ(3/β) / Γ(1/β)."""
    lgamma_1b = torch.lgamma(1.0 / beta)
    lgamma_3b = torch.lgamma(3.0 / beta)
    return alpha ** 2 * torch.exp(lgamma_3b - lgamma_1b)


def compute_sigma2_LN_exact(alpha: Tensor, beta: Tensor) -> Tensor:
    """Exact linear-space variance of exp(ε) for ε ~ GenGauss(0, α, β):
    Var = exp(σ²_log) · (exp(σ²_log) − 1)."""
    sigma2_log = compute_sigma2_LN(alpha, beta)
    return torch.exp(sigma2_log) * torch.expm1(sigma2_log)


def _crystfel_pixel_counts(r_peak: float, r_gap: float, r_bg: float) -> tuple[int, int]:
    """Exact CrystFEL ring-mask pixel counts.

    Mirrors ``setup_ring_masks`` in CrystFEL's ``libcrystfel/src/integration.c``:
    pixels are classified by integer ``rsq = (p−halfw)² + (q−halfw)²`` with
    *peak* ⇔ ``rsq ≤ r_peak²``, *bg* ⇔ ``r_gap² ≤ rsq ≤ r_bg²`` (inner edge
    inclusive), *gap* (excluded) ⇔ ``r_peak² < rsq < r_gap²``.

    Returns the integer pixel counts ``(n_peak, n_bg)``.  These differ from
    the continuous-disk formula ``π·r²`` by ~3–5 % at typical SFX radii;
    using the exact counts removes that bias from gain calibration and
    the bg-variance term.
    """
    r_max = int(math.ceil(r_bg))
    out_sq = int(r_bg * r_bg)
    mid_sq = int(r_gap * r_gap)
    lim_sq = int(r_peak * r_peak)
    n_peak = n_bg = 0
    for fs in range(-r_max, r_max + 1):
        fs2 = fs * fs
        for ss in range(-r_max, r_max + 1):
            rsq = fs2 + ss * ss
            if rsq > out_sq:
                continue
            if rsq >= mid_sq:
                n_bg += 1
            elif rsq <= lim_sq:
                n_peak += 1
    return n_peak, n_bg


@dataclass
class PoissonNoiseModel:
    """Per-observation Poisson noise from CrystFEL ring integration.

    Parameters
    ----------
    r_peak, r_gap, r_bg :
        Integration ring radii in pixels (default 7/9/11 as CrystFEL).
        These **must match the indexamajig run** that produced the upstream
        stream — Rhodopsin (Alvra/JF06T08V01) uses 2/4/6, tubulin uses
        2/3/6, BR (CSPAD) uses 3/4/7.  Mismatch silently corrupts both
        the gain auto-fit AND every per-observation bg-in-peak value.
    gain :
        ADU-to-photon conversion factor; calibrate via ``calibrate_sigma2_of_bg``
        or, without off-Bragg observations, ``calibrate_gain_truncnorm``.
    exact_pixel_counts :
        If ``True`` (default), ``n_peak`` and ``n_bg`` are the exact integer
        pixel counts produced by CrystFEL's mask kernel
        ([integration.c:404-445](https://gitlab.desy.de/thomas.white/crystfel/-/blob/master/libcrystfel/src/integration.c#L404)).
        If ``False``, the legacy continuous-disk approximation
        ``π·r²`` / ``π·(r_bg²−r_gap²)`` is used — only useful for
        reproducing pre-2026-05 results.
    """

    r_peak: float = 7.0
    r_gap: float = 9.0
    r_bg: float = 11.0
    gain: float = 1.0
    exact_pixel_counts: bool = True
    sigma2_bg_slope: float = 0.0
    # σ²_read in ADU² per peak-ring observation (summed over n_peak pixels,
    # including the bg-ring leakage 1+n_peak/n_bg term).  Set by
    # ``calibrate_sigma2_of_bg`` from the per-bg-bin σ²_z(bg) curve.
    read_var_adu_sq: float = 0.0
    gain_calibration_info: dict = field(default_factory=dict, init=False, repr=False)
    gain_selection: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.exact_pixel_counts:
            n_p, n_b = _crystfel_pixel_counts(self.r_peak, self.r_gap, self.r_bg)
            self.n_peak = float(n_p)
            self.n_bg = float(n_b)
        else:
            self.n_peak = math.pi * self.r_peak ** 2
            self.n_bg = math.pi * (self.r_bg ** 2 - self.r_gap ** 2)

    def sigma2_raw(self, I_obs: Tensor, background: Tensor) -> Tensor:
        bg_contrib = background * self.n_peak * (1.0 + self.n_peak / self.n_bg)
        return (I_obs + bg_contrib).clamp(min=1.0)

    def sigma2(self, I_obs: Tensor, background: Tensor) -> Tensor:
        return self.gain ** 2 * self.sigma2_raw(I_obs, background)

    def calibrate_gain(self, I_obs: Tensor, background: Tensor) -> float:
        """Mandel gain calibration from background-limited (negative) peaks.

        Derivation (n_pk photons in peak ring, λ_bg = per-pixel bg rate):
            Var[I_obs_photon] = n_peak · λ_bg · (1 + n_peak/n_bg)
        In ADU (g = ADU/photon):
            Var[I_obs_ADU] = g · n_peak · bg_ADU_per_px · (1 + n_peak/n_bg)
        Conditioned on I_obs<0 (zero mean),
            ⟨I²_neg⟩ = Var[I_obs_ADU]  ⇒  g = ⟨I²_neg⟩ / denom.

        Note: assumes the negatives are zero-centred. Real data has a
        positive μ-bias (signal contamination + centroiding), so this
        estimator runs high. Prefer ``calibrate_sigma2_of_bg``.
        """
        neg = I_obs < 0
        n_neg = int(neg.sum().item())
        if n_neg < 100:
            print(f"  PoissonNoiseModel: only {n_neg} negative obs, "
                  f"keeping gain={self.gain:.3f}")
            return self.gain

        bg_neg = background[neg].clamp(min=0)
        sigma2_pred_ADU = bg_neg * self.n_peak * (1.0 + self.n_peak / self.n_bg)
        var_ADU = (I_obs[neg] ** 2).mean()
        denom = sigma2_pred_ADU.mean().clamp(min=1e-12)
        g = (var_ADU / denom).item()
        self.gain = g
        print(f"  PoissonNoiseModel: gain={g:.4f} ADU/photon "
              f"[n_peak={self.n_peak:.2f}, n_bg={self.n_bg:.2f}, "
              f"{n_neg} negatives]")
        return g

    def _fit_truncnorm_z(
        self,
        z: Tensor,
        *,
        lo: float = -math.inf,
        max_iter: int = 200,
    ) -> tuple[float, float]:
        """Inner LBFGS MLE: fit Normal(μ, σ) to a sample of z-values that
        have already been truncated to ``lo < z < 0``; both truncation points
        are in the likelihood.  Returns (sigma, mu)."""
        device = z.device
        sigma0 = (z.std(unbiased=False) / math.sqrt(1.0 - 2.0 / math.pi)).clamp(min=1e-3)
        mu = torch.zeros((), device=device, dtype=z.dtype, requires_grad=True)
        log_sigma = torch.tensor(
            float(sigma0.log()), device=device, dtype=z.dtype, requires_grad=True,
        )
        opt = torch.optim.LBFGS(
            [mu, log_sigma], max_iter=max_iter, history_size=10,
            tolerance_grad=1e-8, tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def _log_phi_cdf(x: Tensor) -> Tensor:
            return torch.log(0.5 * torch.erfc(-x / math.sqrt(2.0)).clamp(min=1e-300))

        def closure() -> Tensor:
            opt.zero_grad(set_to_none=False)
            sigma = log_sigma.exp()
            zc = (z - mu) / sigma
            ll_point = -0.5 * zc.pow(2) - 0.5 * math.log(2.0 * math.pi) - log_sigma
            log_trunc = _log_phi_cdf(-mu / sigma)
            if math.isfinite(lo):
                # log(Phi(-mu/s) - Phi((lo-mu)/s)) = log Phi(-mu/s) + log1p(-Phi((lo-mu)/s) / Phi(-mu/s))
                log_trunc = log_trunc + torch.log1p(
                    -torch.exp(_log_phi_cdf((lo - mu) / sigma) - log_trunc).clamp(max=1 - 1e-12))
            nll = -(ll_point.sum() - z.numel() * log_trunc)
            nll.backward()
            return nll

        opt.step(closure)
        return float(log_sigma.detach().exp().item()), float(mu.detach().item())

    def calibrate_gain_truncnorm(
        self,
        I_obs: Tensor,
        background: Tensor,
        *,
        bg_min: float = 0.1,
        tail_cut: float = 3.0,
        max_pool: int = 1_000_000,
        max_iter: int = 200,
        seed: int = 0,
    ) -> tuple[float, float]:
        """Gain from the negative integrated intensities: the stream fallback.

        Each negative ``I_j`` is whitened by its predicted σ at g=1,
        ``z_j = I_j / sqrt(bg_j · n_peak · (1 + n_peak/n_bg))``. The pool of
        ``z`` values in (-tail_cut·s0, 0), s0 their robust scale, is fit by MLE
        to a Normal(μ, σ) truncated to that interval; the gain is ``g = σ²``.
        Signal contamination is absorbed into μ instead of inflating σ as in
        the naive Mandel estimator.  It needs no off-Bragg predictions, which
        CrystFEL streams do not carry.

        The estimate is biased: the integrated negatives are not a clean
        zero-signal sample.  So ``auto``
        uses it only to detect a wrong declaration (tolerance 50 %), and a gain
        it replaces is refined in training by one global B
        (``--detector-noise auto``).

        Returns
        -------
        (gain, mu_bias in z units)
        """
        device = I_obs.device
        dtype = torch.float64

        neg = (I_obs < 0) & (background > bg_min)
        n_neg = int(neg.sum().item())
        self.gain_calibration_info = {"method": "negative_truncnorm", "valid": False,
                                      "n_observations": n_neg}
        if n_neg < 100:
            print(f"  PoissonNoiseModel.truncnorm: only {n_neg} valid negatives, "
                  "gain calibration unavailable")
            return float("nan"), 0.0

        I_neg = I_obs[neg].to(dtype=dtype)
        bg_neg = background[neg].to(dtype=dtype)
        sigma_pred_g1 = (bg_neg * self.n_peak * (1.0 + self.n_peak / self.n_bg)).sqrt()
        z = I_neg / sigma_pred_g1.clamp(min=1e-12)
        z = z[z < 0.0]
        # Over-subtracted outliers (a strong neighbour in the ring) give the negatives
        # a heavy tail that inflates sigma.  So the fit keeps z > -tail_cut * s0
        # (s0 = median |z| / 0.6745, the negatives' robust scale) and models that cut
        # in the likelihood: unbiased for a Normal at any gain.
        lo = -math.inf
        if z.numel():
            lo = -tail_cut * float(z.abs().median()) / 0.6745
            z = z[z > lo]
        if z.numel() < 100:
            self.gain_calibration_info["n_observations"] = int(z.numel())
            return float("nan"), 0.0

        if z.numel() > max_pool:
            gen = torch.Generator(device=device).manual_seed(seed)
            idx = torch.randperm(z.numel(), generator=gen, device=device)[:max_pool]
            z = z[idx]

        sigma_hat, mu_hat = self._fit_truncnorm_z(z, lo=lo, max_iter=max_iter)
        gain = sigma_hat ** 2
        self.gain_calibration_info.update(valid=math.isfinite(gain) and gain > 0,
                                          n_observations=int(z.numel()), mu_bias=mu_hat)
        self.gain = gain
        print(
            f"  PoissonNoiseModel.truncnorm: gain={gain:.4f} ADU/photon  "
            f"σ_z={sigma_hat:.3f}  μ_z={mu_hat:+.3f}σ  "
            f"[n_peak={self.n_peak:.2f}, n_bg={self.n_bg:.2f}, "
            f"{int(z.numel())}/{n_neg} negatives]",
            flush=True,
        )
        return gain, mu_hat

    def calibrate_sigma2_of_bg(
        self,
        I_obs: Tensor,
        background: Tensor,
        ewald_offset: Tensor,
        *,
        bg_min: float = 1.0,
        z_max: float = 25.0,
        ewald_min: float = 0.25,
        n_bins: int = 12,
        min_per_bin: int = 2000,
        mu_z_max: float = 0.5,
    ) -> tuple[float, float, float]:
        """Off-Bragg σ²(bg) calibration via Ewald-offset gating, read-noise model.

        Per-observation Var in ADU² for an off-Bragg (no-signal) sample is

            Var[I_obs_ADU] = g · σ²_pred_g1²  +  σ²_read_ADU

        where ``σ²_pred_g1² = bg · n_peak · (1 + n_peak/n_bg)`` is the
        Poisson predicted variance at unit gain.  Dividing by σ²_pred_g1²,

            σ²_z(x) = g  +  c / x,    x = σ²_pred_g1²,   c = σ²_read_ADU

        Fitting per-bg-bin (x_median, σ²_z) with weighted linear
        regression in ``1/x`` recovers ``(g, c)``.  The intercept ``g``
        is the effective detector gain (ADU/photon); ``c`` is the
        constant-in-ADU² additive read-noise floor (per peak-ring obs,
        already including the 1+n_peak/n_bg bg-leak factor).

        Parameters
        ----------
        ewald_offset :
            Per-observation signed Ewald sphere offset ``‖q+s₀‖ − 1/λ``
            in Å⁻¹.  Computed from per-crystal reciprocal basis and the
            stream's beam energy.
        ewald_min :
            |Ewald offset| threshold for the off-Bragg sample.  Default
            0.25 Å⁻¹, empirically validated on Rhodopsin/optimization4
            (μ_z drops below 0.1σ at this scale).  Datasets with broader
            mosaicity may need lower thresholds.
        mu_z_max :
            Per-bin bias QC: bins where ``|μ_z| > mu_z_max`` are dropped
            from the regression (residual signal contamination).

        Returns
        -------
        (gain, mu_bias_in_sigma_units, read_var_adu_sq)
            ``mu_bias`` is the median μ_z across kept bins (diagnostic).
            Sets ``self.gain = gain`` and
            ``self.read_var_adu_sq = read_var_adu_sq``; also clears
            ``self.sigma2_bg_slope = 0`` (legacy field).
            Insufficient observations or background bins return
            ``(nan, 0.0, 0.0)`` with ``gain_calibration_info['valid'] = False``
            and preserve ``self.gain``. There is no negative-only fallback.
        """
        device = I_obs.device
        dtype = torch.float64

        mask = (
            (background > bg_min)
            & (ewald_offset.abs() > ewald_min)
            & torch.isfinite(ewald_offset)
        )
        n_sel = int(mask.sum().item())
        self.gain_calibration_info = {"method": "off_bragg_variance", "valid": False,
                                      "n_observations": n_sel}

        def unavailable(reason: str) -> tuple[float, float, float]:
            self.gain_calibration_info["reason"] = reason
            self.sigma2_bg_slope = 0.0
            self.read_var_adu_sq = 0.0
            print(f"  PoissonNoiseModel.sigma2_of_bg: gain calibration unavailable: "
                  f"{reason}.", flush=True)
            return float("nan"), 0.0, 0.0

        if n_sel < 5000:
            return unavailable("insufficient off-Bragg observations")

        I_off = I_obs[mask].to(dtype=dtype)
        bg_off = background[mask].to(dtype=dtype)
        sigma_pred_g1 = (bg_off * self.n_peak * (1.0 + self.n_peak / self.n_bg)).sqrt()
        z = I_off / sigma_pred_g1.clamp(min=1e-12)
        keep = z.abs() < z_max
        z = z[keep]
        bg_off = bg_off[keep]
        self.gain_calibration_info["n_usable_observations"] = int(z.numel())
        if z.numel() < 5000:
            return unavailable("insufficient usable off-Bragg observations")

        # Quantile-bin in log(bg).  ``torch.quantile`` caps at 2²⁴ elements
        # (int32 internal sort indexing), so do the quantile by sort+gather
        # which uses 64-bit indices.
        log_bg = bg_off.log10()
        sorted_log_bg, _ = log_bg.sort()
        n_sorted = sorted_log_bg.numel()
        qs = torch.linspace(0.0, 1.0, n_bins + 1, dtype=dtype, device=device)
        edge_idx = (qs * (n_sorted - 1)).round().long().clamp(max=n_sorted - 1)
        edges = sorted_log_bg[edge_idx]
        edges[-1] = edges[-1] + 1e-9
        bin_idx = (
            torch.bucketize(log_bg, edges, right=True) - 1
        ).clamp(min=0, max=n_bins - 1)

        denom = float(self.n_peak * (1.0 + self.n_peak / self.n_bg))
        bg_med_per_bin: list[float] = []
        x_med_per_bin: list[float] = []   # sigma_pred_g1² median per bin
        sigma2_per_bin: list[float] = []
        mu_per_bin: list[float] = []
        n_per_bin: list[int] = []
        for k in range(n_bins):
            bin_mask = bin_idx == k
            n_k = int(bin_mask.sum().item())
            if n_k < min_per_bin:
                continue
            z_k = z[bin_mask]
            bg_k = bg_off[bin_mask]
            # Off-Bragg obs are symmetric around 0 → ordinary mean / var.
            mu_k = float(z_k.mean().item())
            sigma2_k = float(z_k.var(unbiased=True).item())
            bg_med = float(bg_k.median().item())
            bg_med_per_bin.append(bg_med)
            x_med_per_bin.append(bg_med * denom)
            sigma2_per_bin.append(sigma2_k)
            mu_per_bin.append(mu_k)
            n_per_bin.append(n_k)

        self.gain_calibration_info["n_bins"] = len(sigma2_per_bin)
        if len(sigma2_per_bin) < 3:
            return unavailable("insufficient background bins")

        # Bin QC: keep bins where |μ_z| < mu_z_max (rules out residual
        # signal-contaminated bins that slipped through Ewald-gating).
        clean = [i for i, m in enumerate(mu_per_bin) if abs(m) < mu_z_max]
        clean_enough = len(clean) >= 3
        if len(clean) < 3:
            clean = list(range(len(sigma2_per_bin)))

        # Fit σ²_z = g + c/x via weighted linear lstsq on inverse-x.
        x_t = torch.tensor([x_med_per_bin[i] for i in clean], dtype=dtype, device=device)
        s2_t = torch.tensor([sigma2_per_bin[i] for i in clean], dtype=dtype, device=device)
        w_t = torch.tensor([n_per_bin[i] for i in clean], dtype=dtype, device=device)
        inv_x = 1.0 / x_t
        A = torch.stack([torch.ones_like(inv_x), inv_x], dim=1)        # (k, 2)
        Aw = A * w_t.unsqueeze(1)
        yw = s2_t * w_t
        coef = torch.linalg.lstsq(Aw, yw.unsqueeze(1)).solution.squeeze(1)
        g_fit, c_fit = float(coef[0].item()), float(coef[1].item())
        # Both intercept and read-noise term must be ≥ 0 physically.
        if not (g_fit > 0):
            g_fit = float("nan")
        c_fit = max(c_fit, 0.0)

        mu_median = float(
            torch.tensor([mu_per_bin[i] for i in clean]).median().item()
        )
        self.gain = g_fit
        self.read_var_adu_sq = c_fit
        self.sigma2_bg_slope = 0.0  # legacy field, unused after refit
        self.gain_calibration_info.update(valid=clean_enough and math.isfinite(g_fit) and g_fit > 0,
                                          n_bins=len(sigma2_per_bin), n_clean_bins=sum(abs(m) < mu_z_max for m in mu_per_bin))

        for i in range(len(sigma2_per_bin)):
            mark = "*" if i in clean else " "
            print(
                f"    bin{mark} bg_med={bg_med_per_bin[i]:8.2f}  "
                f"σ²_z={sigma2_per_bin[i]:7.3f}  μ_z={mu_per_bin[i]:+.3f}σ  "
                f"n={n_per_bin[i]:>7d}",
                flush=True,
            )
        read_var_ph = c_fit / (g_fit * g_fit) if g_fit and g_fit > 0 else float("nan")
        print(
            f"  PoissonNoiseModel.sigma2_of_bg: gain(intercept)={g_fit:.4f} "
            f"ADU/photon  read_var_ADU²={c_fit:.4g}  "
            f"(read_var_photons={read_var_ph:.4g})  "
            f"kept_bins={len(clean)}/{len(sigma2_per_bin)}  "
            f"μ_z_median={mu_median:+.3f}σ  "
            f"[n_peak={self.n_peak:.2f}, n_bg={self.n_bg:.2f}, "
            f"{int(z.numel())}/{n_sel} off-Bragg obs, "
            f"ewald_min={ewald_min:.3f} Å⁻¹]",
            flush=True,
        )
        return g_fit, mu_median, c_fit
