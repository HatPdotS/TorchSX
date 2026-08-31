#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Investigate:
1. Can we predict sigma(F) from the fitted I0 and GenGauss (alpha, beta)?
2. Does fitted alpha drift systematically across trials?

For amplitude error propagation:
  F = sqrt(I0)  =>  sigma(F) = sigma(I0) / (2*F) = sigma(I0) / (2*sqrt(I0))

But sigma(I0) itself depends on the noise model. For a log-normal estimator
with width sigma_log in log-space:
  sigma(I0) ≈ I0 * sigma_log   (for small sigma_log)
  sigma(F) ≈ F * sigma_log / 2  (half the relative error)

The question is whether sigma_log is predictable from alpha and beta alone
(without needing Monte Carlo).
"""

from __future__ import annotations

import os, sys, time
import torch
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tests.test_heteroscedastic.generate import sample_gengauss
from torchsx.deprecated.merge._dstep import fit_I0_poisson_log, fit_alpha


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
    alpha_val = 0.4
    n_trials = 500

    gen_gt = torch.Generator().manual_seed(0)
    log_I0 = torch.empty(n_hkl).uniform_(np.log(1.0), np.log(10000.0), generator=gen_gt)
    I0_true = torch.exp(log_I0)
    alpha_true = torch.full((n_hkl,), alpha_val)
    beta_true = torch.full((n_hkl,), beta_val)

    print(f"n_hkl={n_hkl}, obs_per_hkl={obs_per_hkl}, alpha={alpha_val}, "
          f"beta={beta_val}, bg={background}, trials={n_trials}")

    # Collect: I0 estimates and fitted alpha per trial
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

        # Median init
        offsets = torch.zeros(n_hkl + 1, dtype=torch.long, device=device)
        offsets[1:] = obs_counts.cumsum(0)
        med = torch.zeros(n_hkl, device=device)
        for h in range(n_hkl):
            med[h] = I_obs[offsets[h]:offsets[h+1]].median()
        I0_init = med.clamp(min=0.1)

        # Fit I0
        I0_p, _ = fit_I0_poisson_log(
            I_obs, hkl_id, scale, alpha_true.to(device), beta_true.to(device),
            I0_init, obs_counts, n_hkl, bg=background, n_quad=51, max_iter=50)
        I0_est[trial] = I0_p.cpu()

        # Fit alpha from residuals (using I_obs as "corrected" since scale=1)
        # fit_alpha needs positive obs only
        pos = I_obs > 0
        I_pos = I_obs[pos]
        hkl_pos = hkl_id[pos]
        obs_counts_pos = torch.zeros(n_hkl, dtype=torch.long, device=device)
        obs_counts_pos.scatter_add_(0, hkl_pos, torch.ones_like(hkl_pos))

        alpha_fit = fit_alpha(
            I_pos, hkl_pos, I0_p, beta_true.to(device),
            obs_counts_pos, n_hkl)
        alpha_est[trial] = alpha_fit.cpu()

        if (trial + 1) % 50 == 0:
            print(f"  trial {trial+1}/{n_trials}  ({time.time()-t0:.1f}s)")

    print(f"Done in {time.time()-t0:.1f}s")

    true_np = I0_true.numpy()
    est_np = I0_est.numpy()      # (n_trials, n_hkl)
    alpha_np = alpha_est.numpy()  # (n_trials, n_hkl)

    # ════════════════════════════════════════════════════════════
    # PART 1: Alpha stability
    # ════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("ALPHA STABILITY")
    print(f"{'='*70}")

    mean_alpha = np.mean(alpha_np, axis=0)  # per-HKL mean over trials
    std_alpha = np.std(alpha_np, axis=0)
    global_mean_alpha = np.mean(alpha_np)
    global_std_alpha = np.std(np.mean(alpha_np, axis=1))  # std of per-trial means

    print(f"\nTrue alpha = {alpha_val}")
    print(f"Mean fitted alpha (all HKLs, all trials) = {global_mean_alpha:.4f}")
    print(f"Std of per-trial mean alpha = {global_std_alpha:.6f}")
    print(f"Per-HKL mean alpha: median = {np.median(mean_alpha):.4f}, "
          f"std = {np.std(mean_alpha):.4f}")

    # Check for systematic drift with I0
    bins = np.array([0, 3, 10, 30, 100, 300, 1000, 10000, 1e6])
    print(f"\n  {'I0 range':>16s}  {'mean α':>10s}  {'std α':>10s}  {'bias':>10s}")
    for i in range(len(bins) - 1):
        mask = (true_np >= bins[i]) & (true_np < bins[i+1])
        if mask.sum() == 0:
            continue
        ma = np.mean(alpha_np[:, mask])
        sa = np.std(alpha_np[:, mask])
        bias = ma - alpha_val
        print(f"  [{bins[i]:8.0f}, {bins[i+1]:8.0f})  {ma:10.4f}  {sa:10.4f}  {bias:+10.4f}")

    # Check for drift over trials (e.g. do later trials give different alpha?)
    trial_mean_alpha = np.mean(alpha_np, axis=1)  # per-trial mean
    # Linear regression: alpha vs trial index
    trial_idx = np.arange(n_trials)
    slope, intercept = np.polyfit(trial_idx, trial_mean_alpha, 1)
    print(f"\nDrift over trials: slope = {slope:.2e} per trial")
    print(f"  (drift over {n_trials} trials: {slope*n_trials:.4f})")

    # ════════════════════════════════════════════════════════════
    # PART 2: Amplitude error prediction
    # ════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("AMPLITUDE ERROR PREDICTION")
    print(f"{'='*70}")

    F_true = np.sqrt(true_np)
    F_est = np.sqrt(np.clip(est_np, 0.01, None))  # (n_trials, n_hkl)

    # Empirical sigma(F) per HKL
    sigma_F_emp = np.std(F_est, axis=0)  # (n_hkl,)
    mean_F = np.mean(F_est, axis=0)

    # Empirical sigma(I0) per HKL
    sigma_I_emp = np.std(est_np, axis=0)

    # Predicted sigma(F) from error propagation:
    #   sigma(F) = sigma(I) / (2*sqrt(I))
    # where sigma(I) = I0 * sigma_log
    # For GenGauss: sigma_log^2 = alpha^2 * Gamma(3/beta) / Gamma(1/beta) / n_obs
    # But the MLE concentrates the noise, so sigma_log of the *estimator* is
    # sigma_log_est ≈ alpha * sqrt(Gamma(3/beta)/Gamma(1/beta)) / sqrt(n_obs)
    # This is the standard error of the MLE (Cramér-Rao-like)

    beta_f = float(beta_val)
    from math import lgamma, exp
    # Variance of |epsilon|^beta under GenGauss — Fisher information approach
    # For GenGauss MLE of location in log-space:
    # Var(log(I0_hat)) ≈ alpha^2 * beta * Gamma(1/beta) / (n * (beta-1) * Gamma((beta-1)/beta))
    # Simpler: just compute empirical sigma_log and see if it's predictable
    log_ratio = np.log(np.clip(est_np, 1e-6, None)) - np.log(true_np)[None, :]
    sigma_log_emp = np.std(log_ratio, axis=0)  # per-HKL

    # The predicted sigma_log from Fisher information of GenGauss MLE:
    # For the log-space location parameter mu = log(I0):
    # I_F(mu) = n * beta / alpha^beta * E[|eps|^{beta-2}]
    # For GenGauss: E[|eps|^{beta-2}] = alpha^{beta-2} * Gamma((beta-1)/beta) / Gamma(1/beta)
    # => I_F = n * beta * Gamma((beta-1)/beta) / (alpha^2 * Gamma(1/beta))
    # => sigma_log = alpha * sqrt(Gamma(1/beta) / (n * beta * Gamma((beta-1)/beta)))

    # But this is for pure GenGauss. With Poisson noise on top, the Fisher info
    # is harder to compute analytically. Let's just check the empirical relationship.

    # Prediction 1: sigma_log = c * alpha / sqrt(n_obs)
    # Fit c from the data (for HKLs with I0 > 10 where Poisson doesn't dominate)
    strong = true_np > 10
    c_fit = np.median(sigma_log_emp[strong] * np.sqrt(obs_per_hkl) / alpha_val)
    sigma_log_pred_simple = c_fit * alpha_val / np.sqrt(obs_per_hkl)

    # Prediction 2: use the actual Fisher info formula for GenGauss
    g1b = exp(lgamma(1.0 / beta_f))
    g_bm1_b = exp(lgamma((beta_f - 1.0) / beta_f))
    sigma_log_fisher = alpha_val * np.sqrt(g1b / (obs_per_hkl * beta_f * g_bm1_b))

    # Prediction 3: empirical — account for Poisson contribution
    # sigma_log^2 = sigma_log_gg^2 + sigma_log_poi^2
    # sigma_log_poi ≈ 1/(sqrt(n_obs) * sqrt(I0))  (Poisson noise on log scale)
    sigma_log_poi = 1.0 / (np.sqrt(obs_per_hkl) * np.sqrt(np.clip(true_np, 1, None)))
    sigma_log_pred_combined = np.sqrt(sigma_log_fisher**2 + sigma_log_poi**2)

    print(f"\nEmpirical sigma(log I0):")
    print(f"  median (all)    = {np.median(sigma_log_emp):.4f}")
    print(f"  median (I0>10)  = {np.median(sigma_log_emp[strong]):.4f}")
    print(f"  median (I0>100) = {np.median(sigma_log_emp[true_np>100]):.4f}")

    print(f"\nFisher-info prediction: sigma_log = {sigma_log_fisher:.4f}")
    print(f"  (alpha * sqrt(Gamma(1/β) / (n*β*Gamma((β-1)/β))))")
    print(f"Simple fit: sigma_log = {c_fit:.4f} * alpha / sqrt(n_obs) = {sigma_log_pred_simple:.4f}")

    # sigma(F) prediction
    sigma_F_pred_fisher = F_true * sigma_log_fisher / 2
    sigma_F_pred_combined = F_true * sigma_log_pred_combined / 2

    print(f"\nAmplitude error: sigma(F) = F * sigma_log / 2")
    print(f"  sigma(F)/F (strong) ≈ {sigma_log_fisher/2:.4f} (Fisher)")

    # Comparison table
    print(f"\n  {'I0 range':>16s}  {'σ(F) emp':>10s}  {'σ(F) Fisher':>10s}  "
          f"{'σ(F) comb':>10s}  {'σ_log emp':>10s}  {'σ_log Fish':>10s}")
    for i in range(len(bins) - 1):
        mask = (true_np >= bins[i]) & (true_np < bins[i+1])
        if mask.sum() == 0:
            continue
        sf_e = np.mean(sigma_F_emp[mask])
        sf_f = np.mean(sigma_F_pred_fisher[mask])
        sf_c = np.mean(sigma_F_pred_combined[mask])
        sl_e = np.mean(sigma_log_emp[mask])
        sl_f = sigma_log_fisher
        print(f"  [{bins[i]:8.0f}, {bins[i+1]:8.0f})  {sf_e:10.3f}  {sf_f:10.3f}  "
              f"{sf_c:10.3f}  {sl_e:10.4f}  {sl_f:10.4f}")

    # ════════════════════════════════════════════════════════════
    # PLOTS
    # ════════════════════════════════════════════════════════════
    outdir = os.path.dirname(__file__)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # ── Row 0: Alpha ────────────────────────────────────────────

    # Panel 0,0: fitted alpha vs true I0
    ax = axes[0, 0]
    ax.scatter(true_np, mean_alpha, s=3, alpha=0.5, label="mean fitted α")
    ax.fill_between(
        np.sort(true_np),
        np.interp(np.sort(true_np), true_np[np.argsort(true_np)],
                  (mean_alpha - std_alpha)[np.argsort(true_np)]),
        np.interp(np.sort(true_np), true_np[np.argsort(true_np)],
                  (mean_alpha + std_alpha)[np.argsort(true_np)]),
        alpha=0.15, color="blue",
    )
    ax.axhline(alpha_val, color="red", ls="--", lw=2, label=f"true α={alpha_val}")
    ax.set_xscale("log")
    ax.set_xlabel("True I0")
    ax.set_ylabel("Fitted α")
    ax.set_title("Alpha bias vs intensity")
    ax.legend()

    # Panel 0,1: alpha over trials (drift check)
    ax = axes[0, 1]
    ax.plot(trial_idx, trial_mean_alpha, ".", ms=2, alpha=0.5)
    ax.axhline(alpha_val, color="red", ls="--", lw=1.5)
    ax.plot(trial_idx, intercept + slope * trial_idx, "g-", lw=2,
            label=f"slope={slope:.2e}/trial")
    ax.set_xlabel("Trial index")
    ax.set_ylabel("Mean fitted α (across HKLs)")
    ax.set_title("Alpha drift over trials")
    ax.legend()

    # Panel 0,2: histogram of per-trial mean alpha
    ax = axes[0, 2]
    ax.hist(trial_mean_alpha, bins=40, alpha=0.7, density=True, color="steelblue")
    ax.axvline(alpha_val, color="red", ls="--", lw=2, label=f"true={alpha_val}")
    ax.axvline(np.mean(trial_mean_alpha), color="blue", ls="-", lw=1.5,
               label=f"mean={np.mean(trial_mean_alpha):.4f}")
    ax.set_xlabel("Mean α per trial")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of trial-averaged α")
    ax.legend()

    # ── Row 1: Amplitude error ──────────────────────────────────

    # Panel 1,0: empirical sigma(F) vs predicted
    ax = axes[1, 0]
    ax.scatter(sigma_F_pred_combined, sigma_F_emp, s=3, alpha=0.5)
    lim = [0, max(sigma_F_emp.max(), sigma_F_pred_combined.max()) * 1.1]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("Predicted σ(F) (Fisher + Poisson)")
    ax.set_ylabel("Empirical σ(F) from MC")
    ax.set_title("σ(F) prediction quality")
    r = np.corrcoef(sigma_F_pred_combined, sigma_F_emp)[0, 1]
    ax.text(0.05, 0.9, f"r={r:.4f}", transform=ax.transAxes)

    # Panel 1,1: sigma_log empirical vs I0
    ax = axes[1, 1]
    ax.scatter(true_np, sigma_log_emp, s=3, alpha=0.5, label="empirical")
    I_fit = np.logspace(0, 4, 200)
    sl_pred = np.sqrt(sigma_log_fisher**2 +
                      1.0 / (obs_per_hkl * np.clip(I_fit, 1, None)))
    ax.plot(I_fit, sl_pred, "r-", lw=2, label="Fisher + Poisson prediction")
    ax.axhline(sigma_log_fisher, color="green", ls=":", lw=1.5,
               label=f"Fisher floor = {sigma_log_fisher:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("True I0")
    ax.set_ylabel("σ(log I0)")
    ax.set_title("Log-space error vs intensity")
    ax.legend(fontsize=8)

    # Panel 1,2: sigma(F)/F vs I0
    ax = axes[1, 2]
    rel_sigma_F = sigma_F_emp / F_true
    ax.scatter(true_np, rel_sigma_F, s=3, alpha=0.5, label="empirical σ(F)/F")
    pred_rel = np.sqrt(sigma_log_fisher**2 +
                       1.0 / (obs_per_hkl * np.clip(I_fit, 1, None))) / 2
    ax.plot(I_fit, pred_rel, "r-", lw=2, label="prediction / 2")
    ax.axhline(sigma_log_fisher / 2, color="green", ls=":", lw=1.5,
               label=f"floor = {sigma_log_fisher/2:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("True I0")
    ax.set_ylabel("σ(F) / F")
    ax.set_title("Relative amplitude error")
    ax.legend(fontsize=8)

    fig.suptitle(f"Alpha stability & amplitude error: {n_trials} trials, "
                 f"α={alpha_val}, β={beta_val}, {obs_per_hkl} obs/HKL",
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(outdir, "mc_amplitude_and_alpha.png")
    fig.savefig(path, dpi=150)
    print(f"\nSaved {path}")
    plt.close("all")


if __name__ == "__main__":
    main()
