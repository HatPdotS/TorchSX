#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Test whether the poisson_log I0 estimates follow Gaussian or log-normal.

With only CV~5%, a log-normal and Gaussian are nearly indistinguishable.
Use more trials (1000) and directly compare log-likelihoods + QQ residuals.
"""

from __future__ import annotations

import os, sys, time
import torch
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, shapiro, skew, kurtosis

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tests.test_heteroscedastic.generate import sample_gengauss
from torchsx.deprecated.merge._dstep import fit_I0_poisson_log


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

    n_hkl = 200
    obs_per_hkl = 50
    beta_val = 1.3
    background = 10.0
    alpha_val = 0.4
    n_trials = 1000  # more trials to resolve the distribution shape

    gen_gt = torch.Generator().manual_seed(0)
    log_I0 = torch.empty(n_hkl).uniform_(np.log(1.0), np.log(10000.0), generator=gen_gt)
    I0_true = torch.exp(log_I0)
    alpha_true = torch.full((n_hkl,), alpha_val)
    beta_true = torch.full((n_hkl,), beta_val)

    print(f"n_hkl={n_hkl}, obs_per_hkl={obs_per_hkl}, alpha={alpha_val}, "
          f"beta={beta_val}, bg={background}, trials={n_trials}")

    I0_est = torch.zeros(n_trials, n_hkl)
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

        if (trial + 1) % 100 == 0:
            print(f"  trial {trial+1}/{n_trials}  ({time.time()-t0:.1f}s)")

    print(f"Done in {time.time()-t0:.1f}s")

    true_np = I0_true.numpy()
    est_np = I0_est.numpy()

    # ── For each HKL: compare Gaussian vs log-normal ────────────
    # Log-likelihood difference: ΔLL = LL(lognormal) - LL(gaussian)
    # Positive means log-normal fits better
    dll = np.zeros(n_hkl)
    skewness = np.zeros(n_hkl)
    excess_kurt = np.zeros(n_hkl)
    sw_linear = np.zeros(n_hkl)
    sw_log = np.zeros(n_hkl)

    for h in range(n_hkl):
        vals = est_np[:, h]
        vals_pos = vals[vals > 0]
        if len(vals_pos) < 50:
            continue

        # Gaussian fit
        mu_g, sig_g = np.mean(vals_pos), np.std(vals_pos)
        ll_gauss = np.sum(norm.logpdf(vals_pos, mu_g, sig_g))

        # Log-normal fit (fit Gaussian to log(vals))
        log_vals = np.log(vals_pos)
        mu_ln, sig_ln = np.mean(log_vals), np.std(log_vals)
        ll_lognorm = np.sum(norm.logpdf(log_vals, mu_ln, sig_ln) - log_vals)

        dll[h] = ll_lognorm - ll_gauss
        skewness[h] = skew(vals_pos)
        excess_kurt[h] = kurtosis(vals_pos)  # excess kurtosis

        # Shapiro-Wilk on linear and log
        _, sw_linear[h] = shapiro(vals_pos[:500])  # cap at 500 for shapiro
        _, sw_log[h] = shapiro(log_vals[:500])

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"GAUSSIAN vs LOG-NORMAL (n_trials={n_trials})")
    print(f"{'='*70}")
    print(f"\nΔLL = LL(lognormal) - LL(gaussian) per HKL:")
    print(f"  median ΔLL = {np.median(dll):.1f}")
    print(f"  mean   ΔLL = {np.mean(dll):.1f}")
    print(f"  fraction lognormal wins: {np.mean(dll > 0)*100:.0f}%")

    print(f"\nSkewness of linear-space estimates:")
    print(f"  mean skewness = {np.mean(skewness):.4f}")
    print(f"  (Gaussian: 0, log-normal with CV=0.05: ~3*CV ≈ 0.17)")

    print(f"\nShapiro-Wilk p-values (fraction passing p>0.05):")
    print(f"  Linear space: {np.mean(sw_linear > 0.05)*100:.0f}%")
    print(f"  Log space:    {np.mean(sw_log > 0.05)*100:.0f}%")

    bins = np.array([0, 3, 10, 30, 100, 300, 1000, 10000, 1e6])
    print(f"\n  {'I0 range':>16s}  {'ΔLL':>8s}  {'skew':>8s}  "
          f"{'SW lin%':>8s}  {'SW log%':>8s}  {'LN wins%':>8s}")
    for i in range(len(bins) - 1):
        mask = (true_np >= bins[i]) & (true_np < bins[i+1])
        if mask.sum() == 0:
            continue
        d = dll[mask]
        s = skewness[mask]
        sl = np.mean(sw_linear[mask] > 0.05) * 100
        slog = np.mean(sw_log[mask] > 0.05) * 100
        lnw = np.mean(d > 0) * 100
        print(f"  [{bins[i]:8.0f}, {bins[i+1]:8.0f})  {np.mean(d):8.1f}  "
              f"{np.mean(s):8.4f}  {sl:7.0f}%  {slog:7.0f}%  {lnw:7.0f}%")

    # ── Plots ───────────────────────────────────────────────────
    outdir = os.path.dirname(__file__)

    # Pick HKLs at different intensities
    sorted_idx = np.argsort(true_np)
    picks = [sorted_idx[int(p * n_hkl / 100)] for p in [3, 15, 50, 85, 97]]

    fig, axes = plt.subplots(2, len(picks), figsize=(20, 8))

    for col, h in enumerate(picks):
        vals = est_np[:, h]
        vals_pos = vals[vals > 0]
        true_val = true_np[h]

        # Top row: histogram with Gaussian and log-normal overlay
        ax = axes[0, col]
        ax.hist(vals_pos, bins=40, density=True, alpha=0.6, color="steelblue",
                edgecolor="white", linewidth=0.3)

        mu_g, sig_g = np.mean(vals_pos), np.std(vals_pos)
        x = np.linspace(vals_pos.min(), vals_pos.max(), 300)
        ax.plot(x, norm.pdf(x, mu_g, sig_g), "r-", lw=2, label="Gaussian")

        log_vals = np.log(vals_pos)
        mu_ln, sig_ln = np.mean(log_vals), np.std(log_vals)
        ax.plot(x, lognorm.pdf(x, s=sig_ln, scale=np.exp(mu_ln)),
                "g--", lw=2, label="Log-normal")

        ax.axvline(true_val, color="black", ls="--", lw=1.5)
        ax.set_title(f"I0={true_val:.1f}\nΔLL={dll[h]:.1f}, skew={skewness[h]:.3f}",
                     fontsize=9)
        ax.legend(fontsize=7)
        if col == 0:
            ax.set_ylabel("Density")

        # Bottom row: QQ plot against Gaussian
        ax = axes[1, col]
        sorted_vals = np.sort(vals_pos)
        n = len(sorted_vals)
        theoretical = norm.ppf((np.arange(1, n+1) - 0.5) / n)
        standardized = (sorted_vals - mu_g) / sig_g
        ax.scatter(theoretical, standardized, s=1, alpha=0.3)
        ax.plot([-4, 4], [-4, 4], "r-", lw=1)
        ax.set_xlabel("Theoretical (Gaussian)")
        if col == 0:
            ax.set_ylabel("Observed (standardized)")
        ax.set_title("QQ plot", fontsize=9)
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)

    fig.suptitle(f"Gaussian vs Log-normal: {n_trials} trials, "
                 f"α={alpha_val}, β={beta_val}", fontsize=12)
    fig.tight_layout()
    path = os.path.join(outdir, "mc_gauss_vs_lognormal.png")
    fig.savefig(path, dpi=150)
    print(f"\nSaved {path}")
    plt.close("all")


if __name__ == "__main__":
    main()
