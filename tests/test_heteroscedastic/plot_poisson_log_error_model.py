#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Investigate the error distribution of fit_I0_poisson_log in linear space.

Question: can we model the error as Gaussian in linear space, and how does
sigma(I0) depend on the fitted I0?
"""

from __future__ import annotations

import os, sys, time
import torch
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tests.test_heteroscedastic.generate import sample_gengauss
from torchsx.deprecated.merge._dstep import fit_I0_poisson_log


def generate_one_trial(I0_true, alpha_true, beta_val, obs_per_hkl, background, seed):
    n_hkl = len(I0_true)
    N = n_hkl * obs_per_hkl
    gen = torch.Generator().manual_seed(seed)

    hkl_id = torch.arange(n_hkl).repeat_interleave(obs_per_hkl)
    I0_po = I0_true.repeat_interleave(obs_per_hkl)
    alpha_po = alpha_true.repeat_interleave(obs_per_hkl)

    epsilon = torch.zeros(N)
    for h in range(n_hkl):
        lo = h * obs_per_hkl
        hi = lo + obs_per_hkl
        epsilon[lo:hi] = sample_gengauss(
            float(alpha_true[h]), beta_val, obs_per_hkl, generator=gen,
        )

    I_signal = (I0_po * torch.exp(epsilon)).clamp(min=0)
    total_photons = torch.poisson(I_signal + background, generator=gen)
    I_obs = total_photons - background

    obs_counts = torch.full((n_hkl,), obs_per_hkl, dtype=torch.long)
    return dict(I_obs=I_obs, hkl_id=hkl_id, obs_counts=obs_counts, n_hkl=n_hkl)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    n_hkl = 500
    obs_per_hkl = 50
    beta_val = 1.3
    background = 10.0
    n_trials = 100
    alpha_val = 0.4

    gen_gt = torch.Generator().manual_seed(0)
    log_I0 = torch.empty(n_hkl).uniform_(np.log(1.0), np.log(10000.0), generator=gen_gt)
    I0_true = torch.exp(log_I0)
    alpha_true = torch.full((n_hkl,), alpha_val)
    beta_true = torch.full((n_hkl,), beta_val)

    print(f"n_hkl={n_hkl}, obs_per_hkl={obs_per_hkl}, alpha={alpha_val}, "
          f"beta={beta_val}, bg={background}, trials={n_trials}")

    # Collect estimates
    I0_est = torch.zeros(n_trials, n_hkl)
    t0 = time.time()

    for trial in range(n_trials):
        data = generate_one_trial(I0_true, alpha_true, beta_val, obs_per_hkl,
                                  background, seed=1000 + trial)
        I_obs = data["I_obs"].to(device)
        hkl_id = data["hkl_id"].to(device)
        obs_counts = data["obs_counts"].to(device)
        scale = torch.ones(len(I_obs), device=device)

        # Median init
        offsets = torch.zeros(n_hkl + 1, dtype=torch.long, device=device)
        offsets[1:] = obs_counts.cumsum(0)
        med = torch.zeros(n_hkl, device=device)
        for h in range(n_hkl):
            med[h] = I_obs[offsets[h]:offsets[h+1]].median()
        I0_init = med.clamp(min=0.1)

        I0_p, _ = fit_I0_poisson_log(
            I_obs, hkl_id, scale, alpha_true.to(device), beta_true.to(device),
            I0_init, obs_counts, n_hkl, bg=background, n_quad=51, max_iter=50,
        )
        I0_est[trial] = I0_p.cpu()

        if (trial + 1) % 20 == 0:
            print(f"  trial {trial+1}/{n_trials}  ({time.time()-t0:.1f}s)")

    print(f"Done in {time.time()-t0:.1f}s")

    true_np = I0_true.numpy()
    est_np = I0_est.numpy()  # (n_trials, n_hkl)

    # ── Per-HKL statistics ──────────────────────────────────────
    mean_est = np.mean(est_np, axis=0)
    std_est = np.std(est_np, axis=0)
    err = est_np - true_np[None, :]  # linear error (n_trials, n_hkl)
    mean_err = np.mean(err, axis=0)
    std_err = np.std(err, axis=0)

    outdir = os.path.dirname(__file__)

    # ── Figure 1: Histograms at selected I0 values ─────────────
    sorted_idx = np.argsort(true_np)
    # Pick ~8 HKLs spanning the range
    pick_pct = [2, 5, 15, 30, 50, 70, 85, 95]
    picks = [sorted_idx[int(p * n_hkl / 100)] for p in pick_pct]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.ravel()

    for i, h in enumerate(picks):
        ax = axes[i]
        vals = est_np[:, h]
        true_val = true_np[h]

        ax.hist(vals, bins=25, alpha=0.7, density=True, color="steelblue",
                edgecolor="white", linewidth=0.5)

        # Gaussian fit
        mu, sig = np.mean(vals), np.std(vals)
        x = np.linspace(mu - 4*sig, mu + 4*sig, 200)
        ax.plot(x, norm.pdf(x, mu, sig), "r-", lw=2, label=f"N({mu:.1f}, {sig:.1f})")

        ax.axvline(true_val, color="black", ls="--", lw=2, label=f"true={true_val:.1f}")
        ax.axvline(mu, color="red", ls=":", lw=1.5)

        # Shapiro-Wilk test for normality
        _, p_sw = shapiro(vals)

        ax.set_title(f"I0={true_val:.1f}\n"
                     f"mean={mu:.1f}, std={sig:.1f}\n"
                     f"Shapiro p={p_sw:.3f}",
                     fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlabel("Fitted I0")
        if i % 4 == 0:
            ax.set_ylabel("Density")

    fig.suptitle(f"poisson_log: distribution of fitted I0 across {n_trials} trials "
                 f"(α={alpha_val}, β={beta_val}, {obs_per_hkl} obs/HKL)",
                 fontsize=12)
    fig.tight_layout()
    path1 = os.path.join(outdir, "mc_error_histograms_linear.png")
    fig.savefig(path1, dpi=150)
    print(f"Saved {path1}")

    # ── Figure 2: sigma(I0) vs I0 — the parametrization ────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: std of estimates vs true I0
    ax = axes2[0]
    ax.scatter(true_np, std_est, s=3, alpha=0.5, label="empirical std")
    # Fit a power law: sigma = a * I0^b
    log_I = np.log(true_np[true_np > 3])
    log_s = np.log(std_est[true_np > 3])
    b, log_a = np.polyfit(log_I, log_s, 1)
    a = np.exp(log_a)
    I_fit = np.logspace(0, 4, 100)
    ax.plot(I_fit, a * I_fit**b, "r-", lw=2, label=f"σ = {a:.3f} · I0^{b:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("True I0"); ax.set_ylabel("std(fitted I0)")
    ax.set_title("Error magnitude vs intensity")
    ax.legend()

    # Panel 2: relative error (CV = std/mean) vs I0
    ax = axes2[1]
    cv = std_est / mean_est
    ax.scatter(true_np, cv, s=3, alpha=0.5)
    # If sigma ~ I0, then CV = const
    ax.axhline(np.median(cv[true_np > 30]), color="red", ls="--", lw=1.5,
               label=f"median CV (I0>30) = {np.median(cv[true_np > 30]):.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("True I0"); ax.set_ylabel("CV = std/mean")
    ax.set_title("Relative error vs intensity")
    ax.legend()

    # Panel 3: std/I0 vs I0 — does sigma/I0 = const?
    ax = axes2[2]
    rel_std = std_est / true_np
    ax.scatter(true_np, rel_std, s=3, alpha=0.5)
    # Fit: sigma/I0 = c0 + c1/I0 (additive + multiplicative components)
    from scipy.optimize import curve_fit
    def sigma_model(I0, c_mult, c_add):
        return np.sqrt(c_mult**2 + (c_add / I0)**2)
    popt, _ = curve_fit(sigma_model, true_np, rel_std, p0=[0.05, 1.0])
    I_fit = np.logspace(0, 4, 200)
    ax.plot(I_fit, sigma_model(I_fit, *popt), "r-", lw=2,
            label=f"σ/I0 = √({popt[0]:.4f}² + ({popt[1]:.1f}/I0)²)")
    ax.set_xscale("log")
    ax.set_xlabel("True I0"); ax.set_ylabel("std(I0) / true I0")
    ax.set_title("Relative error model: multiplicative + additive")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.5)

    fig2.tight_layout()
    path2 = os.path.join(outdir, "mc_error_sigma_model.png")
    fig2.savefig(path2, dpi=150)
    print(f"Saved {path2}")

    # ── Print the error model parametrization ───────────────────
    print(f"\n{'='*60}")
    print("ERROR MODEL PARAMETRIZATION")
    print(f"{'='*60}")
    print(f"\nPower-law fit (I0 > 3):  sigma(I0) = {a:.4f} * I0^{b:.3f}")
    print(f"  (if b ≈ 1.0, error is purely multiplicative)")
    print(f"\nTwo-component model:  sigma(I0)/I0 = sqrt(c_mult² + (c_add/I0)²)")
    print(f"  c_mult = {popt[0]:.4f}  (fractional multiplicative noise floor)")
    print(f"  c_add  = {popt[1]:.2f}   (additive noise component, in photons)")
    print(f"\nFor strong reflections: sigma ≈ {popt[0]:.4f} * I0  (CV = {popt[0]:.4f})")
    print(f"For weak reflections:   sigma ≈ {popt[1]:.1f}            (constant)")

    # Shapiro-Wilk summary: fraction of HKLs where Gaussian is plausible
    p_vals = np.array([shapiro(est_np[:, h])[1] for h in range(n_hkl)])
    frac_normal_05 = np.mean(p_vals > 0.05)
    frac_normal_01 = np.mean(p_vals > 0.01)
    print(f"\nShapiro-Wilk normality test (n={n_trials} samples per HKL):")
    print(f"  {frac_normal_05*100:.0f}% of HKLs pass at p>0.05")
    print(f"  {frac_normal_01*100:.0f}% of HKLs pass at p>0.01")

    # Break down by intensity
    print(f"\n  {'I0 range':>16s}  {'% normal (p>.05)':>16s}  {'mean Shapiro p':>16s}")
    bins = np.array([0, 3, 10, 30, 100, 300, 1000, 10000, 1e6])
    for i in range(len(bins) - 1):
        mask = (true_np >= bins[i]) & (true_np < bins[i+1])
        if mask.sum() == 0:
            continue
        frac = np.mean(p_vals[mask] > 0.05) * 100
        mp = np.mean(p_vals[mask])
        print(f"  [{bins[i]:8.0f}, {bins[i+1]:8.0f})  {frac:14.0f}%  {mp:16.3f}")


if __name__ == "__main__":
    main()
