#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Test the sigma(F) prediction formula using fitted (not true) parameters.

The question: can we predict sigma(F) from quantities we actually have?
  - fitted I0 (from poisson_log)
  - fitted alpha (from fit_alpha, which is biased at low I0)
  - known beta
  - known n_obs

Formula:  sigma(log I0) = sqrt(c^2 * alpha^2 / n + 1 / (n * I0))
          sigma(F) = F * sigma(log I0) / 2

We test with both true alpha and fitted alpha to see which works better.
We also fit c from the data to find the best constant.
"""

from __future__ import annotations

import os, sys, time
import torch
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tests.test_heteroscedastic.generate import sample_gengauss
from torchsx.merge._dstep import fit_I0_poisson_log, fit_alpha


def generate_one_trial(I0_true, alpha_true, beta_val, obs_per_hkl, background, seed):
    n_hkl = len(I0_true)
    N = n_hkl * obs_per_hkl
    gen = torch.Generator().manual_seed(seed)
    hkl_id = torch.arange(n_hkl).repeat_interleave(obs_per_hkl)
    I0_po = I0_true.repeat_interleave(obs_per_hkl)

    epsilon = torch.zeros(N)
    for h in range(n_hkl):
        lo, hi = h * obs_per_hkl, (h + 1) * obs_per_hkl
        epsilon[lo:hi] = sample_gengauss(
            float(alpha_true[h]), beta_val, obs_per_hkl, generator=gen)

    I_signal = (I0_po * torch.exp(epsilon)).clamp(min=0)
    total_photons = torch.poisson(I_signal + background, generator=gen)
    I_obs = total_photons - background
    obs_counts = torch.full((n_hkl,), obs_per_hkl, dtype=torch.long)
    return dict(I_obs=I_obs, hkl_id=hkl_id, obs_counts=obs_counts, n_hkl=n_hkl)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    n_hkl = 300
    obs_per_hkl = 50
    beta_val = 1.3
    background = 10.0
    alpha_val = 1.2
    n_trials = 500

    gen_gt = torch.Generator().manual_seed(0)
    log_I0 = torch.empty(n_hkl).uniform_(np.log(1.0), np.log(10000.0), generator=gen_gt)
    I0_true = torch.exp(log_I0)
    alpha_true = torch.full((n_hkl,), alpha_val)
    beta_true = torch.full((n_hkl,), beta_val)

    print(f"n_hkl={n_hkl}, obs_per_hkl={obs_per_hkl}, alpha={alpha_val}, "
          f"beta={beta_val}, bg={background}, trials={n_trials}")

    I0_est = torch.zeros(n_trials, n_hkl)
    alpha_est = torch.zeros(n_trials, n_hkl)
    t0 = time.time()

    for trial in range(n_trials):
        data = generate_one_trial(I0_true, alpha_true, beta_val, obs_per_hkl,
                                  background, seed=1000 + trial)
        I_obs = data["I_obs"].to(device)
        hkl_id = data["hkl_id"].to(device)
        obs_counts = data["obs_counts"].to(device)
        scale = torch.ones(len(I_obs), device=device)

        offsets = torch.zeros(n_hkl + 1, dtype=torch.long, device=device)
        offsets[1:] = obs_counts.cumsum(0)
        med = torch.zeros(n_hkl, device=device)
        for h in range(n_hkl):
            med[h] = I_obs[offsets[h]:offsets[h+1]].median()
        I0_init = med.clamp(min=0.1)

        I0_p, _ = fit_I0_poisson_log(
            I_obs, hkl_id, scale, alpha_true.to(device), beta_true.to(device),
            I0_init, obs_counts, n_hkl, bg=background, n_quad=51, max_iter=50)
        I0_est[trial] = I0_p.cpu()

        pos = I_obs > 0
        I_pos = I_obs[pos]
        hkl_pos = hkl_id[pos]
        obs_counts_pos = torch.zeros(n_hkl, dtype=torch.long, device=device)
        obs_counts_pos.scatter_add_(0, hkl_pos, torch.ones_like(hkl_pos))
        alpha_fit = fit_alpha(I_pos, hkl_pos, I0_p, beta_true.to(device),
                              obs_counts_pos, n_hkl)
        alpha_est[trial] = alpha_fit.cpu()

        if (trial + 1) % 50 == 0:
            print(f"  trial {trial+1}/{n_trials}  ({time.time()-t0:.1f}s)")

    print(f"Done in {time.time()-t0:.1f}s")

    true_np = I0_true.numpy()
    est_np = I0_est.numpy()
    alpha_np = alpha_est.numpy()
    n = obs_per_hkl

    # Empirical sigma(log I0) per HKL
    log_err = np.log(np.clip(est_np, 1e-6, None)) - np.log(true_np)[None, :]
    sigma_log_emp = np.std(log_err, axis=0)

    # Empirical sigma(F)
    F_true = np.sqrt(true_np)
    F_est = np.sqrt(np.clip(est_np, 0.01, None))
    sigma_F_emp = np.std(F_est, axis=0)

    # Mean fitted alpha per HKL (what we'd have from a single run)
    alpha_fitted_mean = np.mean(alpha_np, axis=0)

    # ── Fit the constant c ──────────────────────────────────────
    # sigma_log^2 = c^2 * alpha^2 / n + 1 / (n * I0)

    # Fit c using TRUE alpha (gold standard)
    def mse_true(c):
        pred = np.sqrt(c**2 * alpha_val**2 / n + 1.0 / (n * np.clip(true_np, 1, None)))
        return np.mean((pred - sigma_log_emp)**2)

    res_true = minimize_scalar(mse_true, bounds=(0.5, 2.0), method="bounded")
    c_true = res_true.x

    # Fit c using FITTED alpha (what we'd do in practice)
    def mse_fitted(c):
        pred = np.sqrt(c**2 * alpha_fitted_mean**2 / n +
                       1.0 / (n * np.clip(true_np, 1, None)))
        return np.mean((pred - sigma_log_emp)**2)

    res_fitted = minimize_scalar(mse_fitted, bounds=(0.5, 2.0), method="bounded")
    c_fitted = res_fitted.x

    print(f"\n{'='*70}")
    print("SIGMA PREDICTION MODELS")
    print(f"{'='*70}")
    print(f"\nOptimal c (true alpha):   {c_true:.4f}")
    print(f"Optimal c (fitted alpha): {c_fitted:.4f}")

    # Predictions
    sigma_log_true_alpha = np.sqrt(c_true**2 * alpha_val**2 / n +
                                    1.0 / (n * np.clip(true_np, 1, None)))
    sigma_log_fitted_alpha = np.sqrt(c_fitted**2 * alpha_fitted_mean**2 / n +
                                      1.0 / (n * np.clip(true_np, 1, None)))

    # Per-trial prediction using single-trial fitted values
    # This is the realistic scenario: you have ONE reconstruction
    sigma_log_single = []
    sigma_F_single_pred = []
    sigma_F_single_emp = []
    for trial in range(n_trials):
        I0_t = est_np[trial]
        alpha_t = alpha_np[trial]
        sl = np.sqrt(c_fitted**2 * alpha_t**2 / n +
                     1.0 / (n * np.clip(I0_t, 1, None)))
        F_t = np.sqrt(np.clip(I0_t, 0.01, None))
        sf_pred = F_t * sl / 2
        sigma_log_single.append(sl)
        sigma_F_single_pred.append(sf_pred)

    sigma_log_single = np.array(sigma_log_single)  # (n_trials, n_hkl)
    sigma_F_single_pred = np.array(sigma_F_single_pred)

    # Compare: for each HKL, mean predicted sigma_F vs empirical sigma_F
    mean_sf_pred = np.mean(sigma_F_single_pred, axis=0)

    bins = np.array([0, 3, 10, 30, 100, 300, 1000, 10000, 1e6])
    print(f"\n  {'I0 range':>16s}  {'σ_log emp':>10s}  {'pred(true α)':>12s}  "
          f"{'pred(fit α)':>12s}  {'σ(F) emp':>10s}  {'σ(F) pred':>10s}  {'ratio':>8s}")
    for i in range(len(bins) - 1):
        mask = (true_np >= bins[i]) & (true_np < bins[i+1])
        if mask.sum() == 0:
            continue
        sle = np.mean(sigma_log_emp[mask])
        slt = np.mean(sigma_log_true_alpha[mask])
        slf = np.mean(sigma_log_fitted_alpha[mask])
        sfe = np.mean(sigma_F_emp[mask])
        sfp = np.mean(mean_sf_pred[mask])
        r = sfp / sfe if sfe > 0 else 0
        print(f"  [{bins[i]:8.0f}, {bins[i+1]:8.0f})  {sle:10.4f}  {slt:12.4f}  "
              f"{slf:12.4f}  {sfe:10.3f}  {sfp:10.3f}  {r:8.3f}")

    # ── Check if fitted alpha absorbs Poisson perfectly ─────────
    # If alpha_fitted^2 ≈ alpha_true^2 + 1/I0, then the Poisson term
    # is already in alpha and we can use: sigma_log = c * alpha_fitted / sqrt(n)
    print(f"\n{'='*70}")
    print("DOES FITTED ALPHA ABSORB THE POISSON TERM?")
    print(f"{'='*70}")
    alpha_excess_sq = alpha_fitted_mean**2 - alpha_val**2
    poisson_on_log = 1.0 / np.clip(true_np, 1, None)
    print(f"\n  {'I0 range':>16s}  {'α_fit²−α²':>12s}  {'1/I0':>10s}  {'ratio':>8s}")
    for i in range(len(bins) - 1):
        mask = (true_np >= bins[i]) & (true_np < bins[i+1])
        if mask.sum() == 0:
            continue
        ae = np.mean(alpha_excess_sq[mask])
        pi = np.mean(poisson_on_log[mask])
        r = ae / pi if pi > 1e-6 else 0
        print(f"  [{bins[i]:8.0f}, {bins[i+1]:8.0f})  {ae:12.4f}  {pi:10.4f}  {r:8.2f}")

    # ── Simplest model: just use fitted alpha ───────────────────
    # sigma_log = c_simple * alpha_fitted / sqrt(n)
    c_simple_vals = sigma_log_emp * np.sqrt(n) / alpha_fitted_mean
    c_simple = np.median(c_simple_vals[true_np > 3])  # exclude weakest
    sigma_log_simple = c_simple * alpha_fitted_mean / np.sqrt(n)

    print(f"\n{'='*70}")
    print(f"SIMPLEST MODEL: σ(log I0) = {c_simple:.3f} * α_fitted / √n")
    print(f"{'='*70}")
    print(f"\n  {'I0 range':>16s}  {'σ_log emp':>10s}  {'simple pred':>12s}  {'ratio':>8s}")
    for i in range(len(bins) - 1):
        mask = (true_np >= bins[i]) & (true_np < bins[i+1])
        if mask.sum() == 0:
            continue
        sle = np.mean(sigma_log_emp[mask])
        slp = np.mean(sigma_log_simple[mask])
        r = slp / sle
        print(f"  [{bins[i]:8.0f}, {bins[i+1]:8.0f})  {sle:10.4f}  {slp:12.4f}  {r:8.3f}")

    # ── Plots ───────────────────────────────────────────────────
    outdir = os.path.dirname(__file__)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 0,0: sigma_log predictions vs empirical
    ax = axes[0, 0]
    ax.scatter(true_np, sigma_log_emp, s=3, alpha=0.4, label="empirical", zorder=3)
    I_fit = np.logspace(0, 4, 300)
    pred_ta = np.sqrt(c_true**2 * alpha_val**2 / n + 1.0 / (n * I_fit))
    ax.plot(I_fit, pred_ta, "r-", lw=2, label=f"c={c_true:.2f}, true α")
    # fitted alpha version
    alpha_fit_interp = np.interp(I_fit, true_np[np.argsort(true_np)],
                                  alpha_fitted_mean[np.argsort(true_np)])
    pred_fa = np.sqrt(c_fitted**2 * alpha_fit_interp**2 / n + 1.0 / (n * I_fit))
    ax.plot(I_fit, pred_fa, "g--", lw=2, label=f"c={c_fitted:.2f}, fitted α")
    # simple
    pred_simple = c_simple * alpha_fit_interp / np.sqrt(n)
    ax.plot(I_fit, pred_simple, "m:", lw=2, label=f"simple: {c_simple:.2f}·α_fit/√n")
    ax.set_xscale("log")
    ax.set_xlabel("True I0"); ax.set_ylabel("σ(log I0)")
    ax.set_title("Log-space error prediction")
    ax.legend(fontsize=7)

    # 0,1: sigma(F) predicted vs empirical (scatter)
    ax = axes[0, 1]
    ax.scatter(mean_sf_pred, sigma_F_emp, s=3, alpha=0.4)
    lim = [0, max(sigma_F_emp.max(), mean_sf_pred.max()) * 1.05]
    ax.plot(lim, lim, "r--", lw=1)
    r = np.corrcoef(mean_sf_pred, sigma_F_emp)[0, 1]
    ax.set_xlabel("Predicted σ(F)"); ax.set_ylabel("Empirical σ(F)")
    ax.set_title(f"σ(F): predicted vs empirical (r={r:.4f})")

    # 0,2: sigma(F)/F predicted vs empirical
    ax = axes[0, 2]
    rel_pred = mean_sf_pred / F_true
    rel_emp = sigma_F_emp / F_true
    ax.scatter(true_np, rel_emp, s=3, alpha=0.4, label="empirical")
    ax.scatter(true_np, rel_pred, s=3, alpha=0.4, label="predicted", color="red")
    ax.set_xscale("log")
    ax.set_xlabel("True I0"); ax.set_ylabel("σ(F)/F")
    ax.set_title("Relative amplitude error")
    ax.legend()

    # 1,0: alpha_fitted^2 - alpha_true^2 vs 1/I0
    ax = axes[1, 0]
    ax.scatter(1.0 / np.clip(true_np, 1, None), alpha_excess_sq, s=3, alpha=0.5)
    x = np.linspace(0, 1, 100)
    slope_ab, _ = np.polyfit(1.0 / np.clip(true_np[true_np > 1], 1, None),
                              alpha_excess_sq[true_np > 1], 1)
    ax.plot(x, slope_ab * x, "r-", lw=2, label=f"slope={slope_ab:.2f}")
    ax.set_xlabel("1/I0"); ax.set_ylabel("α_fitted² − α_true²")
    ax.set_title("Alpha absorbs Poisson variance")
    ax.legend()

    # 1,1: single-trial sigma(F) prediction quality
    # Pick one trial and scatter predicted vs "error"
    trial = 0
    I0_t = est_np[trial]
    alpha_t = alpha_np[trial]
    F_t = np.sqrt(np.clip(I0_t, 0.01, None))
    sl_t = np.sqrt(c_fitted**2 * alpha_t**2 / n +
                    1.0 / (n * np.clip(I0_t, 1, None)))
    sf_pred_t = F_t * sl_t / 2
    # "True" error for this trial
    F_err_t = np.abs(F_t - F_true)
    ax = axes[1, 1]
    ax.scatter(sf_pred_t, F_err_t, s=3, alpha=0.4)
    lim = [0, max(F_err_t.max(), sf_pred_t.max()) * 0.8]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("Predicted σ(F) (single trial)")
    ax.set_ylabel("|F_est − F_true| (single trial)")
    ax.set_title("Single-trial: predicted σ vs actual error")

    # 1,2: histogram of |error|/sigma for single trial
    ax = axes[1, 2]
    z = F_err_t / np.clip(sf_pred_t, 1e-6, None)
    ax.hist(z[z < 5], bins=50, density=True, alpha=0.7, color="steelblue")
    x = np.linspace(0, 5, 200)
    # Half-normal (expected for |Z| where Z ~ N(0,1))
    ax.plot(x, 2 * np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), "r-", lw=2,
            label="half-normal")
    ax.set_xlabel("|F_est − F_true| / σ(F)")
    ax.set_ylabel("Density")
    ax.set_title(f"|error|/σ distribution (med={np.median(z):.2f}, expect 0.67)")
    ax.legend()

    fig.suptitle(f"σ(F) prediction from fitted parameters "
                 f"(α={alpha_val}, β={beta_val}, n={obs_per_hkl})", fontsize=12)
    fig.tight_layout()
    path = os.path.join(outdir, "mc_sigma_prediction.png")
    fig.savefig(path, dpi=150)
    print(f"\nSaved {path}")
    plt.close("all")


if __name__ == "__main__":
    main()
