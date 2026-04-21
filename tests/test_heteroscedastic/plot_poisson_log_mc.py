#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Monte Carlo study of the poisson_log I0 estimator.

Draw 100 independent datasets from the same (I0, alpha, beta) ground truth,
fit each independently with fit_I0_poisson_log, and analyze the statistics
of the resulting I0 estimates.
"""

from __future__ import annotations

import os, sys, time
import torch
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tests.test_heteroscedastic.generate import sample_gengauss, _make_positive_arrays
from torchsx.merge._dstep import fit_I0_poisson_log, fit_I0_log, _log_gg_norm


# ── Data generation (one trial) ─────────────────────────────────


def generate_one_trial(
    I0_true: torch.Tensor,
    alpha_true: torch.Tensor,
    beta_val: float,
    obs_per_hkl: int,
    background: float,
    seed: int,
) -> dict:
    """Generate one dataset from fixed ground truth with a given seed."""
    n_hkl = len(I0_true)
    N = n_hkl * obs_per_hkl
    gen = torch.Generator().manual_seed(seed)

    hkl_id = torch.arange(n_hkl).repeat_interleave(obs_per_hkl)
    I0_po = I0_true.repeat_interleave(obs_per_hkl)
    alpha_po = alpha_true.repeat_interleave(obs_per_hkl)

    # Per-obs GenGauss noise (vectorised per HKL because alpha varies)
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
    return dict(
        I_obs=I_obs,
        hkl_id=hkl_id,
        obs_counts=obs_counts,
        n_hkl=n_hkl,
    )


# ── Main ────────────────────────────────────────────────────────


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Fixed ground truth
    n_hkl = 500
    obs_per_hkl = 50
    beta_val = 1.3
    background = 10.0
    n_trials = 100

    # Log-uniform I0 spanning weak to strong
    gen_gt = torch.Generator().manual_seed(0)
    log_I0 = torch.empty(n_hkl).uniform_(
        np.log(1.0), np.log(10000.0), generator=gen_gt,
    )
    I0_true = torch.exp(log_I0)

    # Constant alpha for simplicity (can test variable later)
    alpha_val = 0.4
    alpha_true = torch.full((n_hkl,), alpha_val)
    beta_true = torch.full((n_hkl,), beta_val)

    print(f"n_hkl={n_hkl}, obs_per_hkl={obs_per_hkl}, alpha={alpha_val}, "
          f"beta={beta_val}, bg={background}, trials={n_trials}")
    print(f"I0 range: [{I0_true.min():.1f}, {I0_true.max():.1f}], "
          f"median={I0_true.median():.1f}")

    # Collect I0 estimates across trials: (n_trials, n_hkl)
    I0_poisson = torch.zeros(n_trials, n_hkl)
    I0_logsp = torch.zeros(n_trials, n_hkl)
    I0_median = torch.zeros(n_trials, n_hkl)

    scale_ones = None  # will init on first trial
    t0 = time.time()

    for trial in range(n_trials):
        data = generate_one_trial(
            I0_true, alpha_true, beta_val, obs_per_hkl, background,
            seed=1000 + trial,
        )

        I_obs = data["I_obs"].to(device)
        hkl_id = data["hkl_id"].to(device)
        obs_counts = data["obs_counts"].to(device)
        n = data["n_hkl"]

        if scale_ones is None:
            scale_ones = torch.ones(len(I_obs), device=device)
        elif len(scale_ones) != len(I_obs):
            scale_ones = torch.ones(len(I_obs), device=device)

        # Median init
        I_sorted = I_obs.clone()
        offsets = torch.zeros(n + 1, dtype=torch.long, device=device)
        offsets[1:] = obs_counts.cumsum(0)
        med = torch.zeros(n, device=device)
        for h in range(n):
            vals = I_sorted[offsets[h]:offsets[h+1]]
            med[h] = vals.median()
        I0_init = med.clamp(min=0.1)

        I0_median[trial] = med.cpu().clamp(min=0.1)

        # fit_I0_poisson_log
        alpha_d = alpha_true.to(device)
        beta_d = beta_true.to(device)
        I0_p, _ = fit_I0_poisson_log(
            I_obs, hkl_id, scale_ones, alpha_d, beta_d,
            I0_init, obs_counts, n, bg=background, n_quad=51, max_iter=50,
        )
        I0_poisson[trial] = I0_p.cpu()

        # fit_I0_log
        I0_l, _ = fit_I0_log(
            I_obs, hkl_id, scale_ones, alpha_d, beta_d,
            I0_init, obs_counts, n, max_iter=50,
        )
        I0_logsp[trial] = I0_l.cpu()

        if (trial + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  trial {trial+1}/{n_trials}  ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

    true_np = I0_true.numpy()
    log_true = np.log(true_np)

    # ── Summary statistics ──────────────────────────────────────
    methods = [("poisson_log", I0_poisson),
               ("log", I0_logsp),
               ("median", I0_median)]

    # Per-trial accuracy metrics (what you'd see from a SINGLE reconstruction)
    print("\n" + "=" * 70)
    print("PER-TRIAL ACCURACY (single reconstruction, not averaged)")
    print("=" * 70)
    bins = np.array([0, 3, 10, 30, 100, 300, 1000, 3000, 10000, 1e6])

    for name, I0_all in methods:
        arr = I0_all.numpy()  # (n_trials, n_hkl)
        log_est = np.log(np.clip(arr, 1e-6, None))

        # Per-trial metrics
        log_err = log_est - log_true[None, :]  # (n_trials, n_hkl)
        rmse_log = np.sqrt(np.mean(log_err ** 2, axis=1))  # (n_trials,)
        mae_log = np.mean(np.abs(log_err), axis=1)
        corr = np.array([np.corrcoef(log_true, log_est[t])[0, 1]
                         for t in range(n_trials)])

        print(f"\n=== {name} ===")
        print(f"  RMSE(log):  {np.mean(rmse_log):.4f} +/- {np.std(rmse_log):.4f}  "
              f"(median {np.median(rmse_log):.4f})")
        print(f"  MAE(log):   {np.mean(mae_log):.4f} +/- {np.std(mae_log):.4f}")
        print(f"  r(log,log): {np.mean(corr):.6f} +/- {np.std(corr):.6f}")
        print(f"  R² (log):   {np.mean(corr**2):.6f}")

        # Per-trial RMSE broken down by intensity bin
        print(f"  {'I0 range':>16s}  {'RMSE(log)':>10s}  {'MAE(log)':>10s}  {'bias(log)':>10s}")
        for i in range(len(bins) - 1):
            mask = (true_np >= bins[i]) & (true_np < bins[i+1])
            if mask.sum() == 0:
                continue
            le = log_err[:, mask]  # (n_trials, n_in_bin)
            rmse = np.sqrt(np.mean(le ** 2))
            mae = np.mean(np.abs(le))
            bias = np.mean(le)
            print(f"  [{bins[i]:8.0f}, {bins[i+1]:8.0f})  {rmse:10.4f}  {mae:10.4f}  {bias:+10.4f}")

    # ── Cross-method comparison table ───────────────────────────
    print("\n" + "=" * 70)
    print("CROSS-METHOD COMPARISON (mean over 100 trials)")
    print("=" * 70)
    header = f"  {'I0 range':>16s}"
    for name, _ in methods:
        header += f"  {name:>14s}"
    print(header + "    (RMSE in log-space)")
    for i in range(len(bins) - 1):
        mask = (true_np >= bins[i]) & (true_np < bins[i+1])
        if mask.sum() == 0:
            continue
        row = f"  [{bins[i]:8.0f}, {bins[i+1]:8.0f})"
        for name, I0_all in methods:
            arr = I0_all.numpy()
            le = np.log(np.clip(arr[:, mask], 1e-6, None)) - log_true[None, mask]
            rmse = np.sqrt(np.mean(le ** 2))
            row += f"  {rmse:14.4f}"
        print(row)
    # Overall
    row = f"  {'OVERALL':>19s}"
    for name, I0_all in methods:
        arr = I0_all.numpy()
        le = np.log(np.clip(arr, 1e-6, None)) - log_true[None, :]
        rmse = np.sqrt(np.mean(le ** 2))
        row += f"  {rmse:14.4f}"
    print(row)

    # ── BIAS + PRECISION breakdown (original table) ─────────────
    print("\n" + "=" * 70)
    print("BIAS AND PRECISION (mean_ratio and CV over 100 trials)")
    print("=" * 70)
    for name, I0_all in methods:
        arr = I0_all.numpy()
        ratio = arr / true_np[None, :]
        mean_ratio = np.mean(ratio, axis=0)
        std_ratio = np.std(ratio, axis=0)
        cv = std_ratio / mean_ratio

        print(f"\n=== {name} ===")
        print(f"  Overall median(mean_ratio) = {np.median(mean_ratio):.4f}")
        print(f"  Overall mean(mean_ratio)   = {np.mean(mean_ratio):.4f}")
        print(f"  median(CV)                 = {np.median(cv):.4f}")
        print(f"  mean(std_ratio)            = {np.mean(std_ratio):.4f}")

        print(f"  {'I0 range':>16s}  {'mean ratio':>10s}  {'std ratio':>10s}  {'CV':>8s}")
        for i in range(len(bins) - 1):
            mask = (true_np >= bins[i]) & (true_np < bins[i+1])
            if mask.sum() == 0:
                continue
            mr = np.mean(mean_ratio[mask])
            sr = np.mean(std_ratio[mask])
            c = np.mean(cv[mask])
            print(f"  [{bins[i]:8.0f}, {bins[i+1]:8.0f})  {mr:10.4f}  {sr:10.4f}  {c:8.4f}")

    # ── Plots ───────────────────────────────────────────────────
    outdir = os.path.dirname(__file__)

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))

    for col, (name, I0_all) in enumerate(methods):
        arr = I0_all.numpy()
        log_est = np.log(np.clip(arr, 1e-6, None))
        ratio = arr / true_np[None, :]
        mean_ratio = np.mean(ratio, axis=0)
        std_ratio = np.std(ratio, axis=0)

        # Row 0: bias (mean ratio) vs true I0
        ax = axes[0, col]
        ax.scatter(true_np, mean_ratio, s=2, alpha=0.5)
        ax.axhline(1.0, color="red", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("True I0")
        ax.set_ylabel("Mean(est/true) over 100 trials")
        ax.set_title(f"{name}: bias")
        ax.set_ylim(0.5, 2.0)

        # Row 1: CV vs true I0
        ax = axes[1, col]
        cv = std_ratio / mean_ratio
        ax.scatter(true_np, cv, s=2, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("True I0")
        ax.set_ylabel("CV (std/mean) over 100 trials")
        ax.set_title(f"{name}: precision")
        ax.set_ylim(0, 1.5)

        # Row 2: per-HKL RMSE(log) vs true I0  (ACCURACY)
        ax = axes[2, col]
        log_err = log_est - log_true[None, :]  # (n_trials, n_hkl)
        per_hkl_rmse = np.sqrt(np.mean(log_err ** 2, axis=0))  # (n_hkl,)
        ax.scatter(true_np, per_hkl_rmse, s=2, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("True I0")
        ax.set_ylabel("RMSE(log I0) over 100 trials")
        ax.set_title(f"{name}: accuracy")
        ax.set_ylim(0, 1.5)

    fig.suptitle(f"MC study: {n_trials} trials, {n_hkl} HKLs, "
                 f"{obs_per_hkl} obs/HKL, α={alpha_val}, β={beta_val}, "
                 f"bg={background}", fontsize=12)
    fig.tight_layout()
    path = os.path.join(outdir, "mc_poisson_log_bias_cv.png")
    fig.savefig(path, dpi=150)
    print(f"\nSaved {path}")

    # ── Accuracy comparison: single-trial scatter plots ─────────
    # Show one trial (trial 0) as scatter: est vs true
    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
    trial_idx = 0
    for col, (name, I0_all) in enumerate(methods):
        ax = axes3[col]
        est = I0_all[trial_idx].numpy()
        ax.scatter(true_np, est, s=3, alpha=0.5)
        lim = [0.5, true_np.max() * 2]
        ax.plot(lim, lim, "r--", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("True I0"); ax.set_ylabel("Estimated I0")
        r = np.corrcoef(log_true, np.log(np.clip(est, 1e-6, None)))[0, 1]
        rmse = np.sqrt(np.mean((np.log(np.clip(est, 1e-6, None)) - log_true)**2))
        ax.set_title(f"{name}: r={r:.4f}, RMSE(log)={rmse:.3f}")
    fig3.suptitle(f"Single trial (trial {trial_idx}): est vs true", fontsize=12)
    fig3.tight_layout()
    path3 = os.path.join(outdir, "mc_poisson_log_single_trial.png")
    fig3.savefig(path3, dpi=150)
    print(f"Saved {path3}")

    # ── Distribution of estimates for a few selected HKLs ───────
    # Pick HKLs at different intensity levels
    sorted_idx = np.argsort(true_np)
    pick = [sorted_idx[i] for i in [10, 100, 250, 400, 490]]

    fig2, axes2 = plt.subplots(len(pick), 3, figsize=(16, 3 * len(pick)))
    for row, h in enumerate(pick):
        for col, (name, I0_all) in enumerate([("poisson_log", I0_poisson),
                                               ("log", I0_logsp),
                                               ("median", I0_median)]):
            ax = axes2[row, col]
            vals = I0_all[:, h].numpy()
            ax.hist(vals, bins=30, alpha=0.7, density=True)
            ax.axvline(true_np[h], color="red", ls="--", lw=2,
                       label=f"true={true_np[h]:.1f}")
            ax.axvline(np.mean(vals), color="blue", ls="-", lw=1.5,
                       label=f"mean={np.mean(vals):.1f}")
            ax.axvline(np.median(vals), color="green", ls=":", lw=1.5,
                       label=f"med={np.median(vals):.1f}")
            ax.legend(fontsize=7)
            if row == 0:
                ax.set_title(name)
            if col == 0:
                ax.set_ylabel(f"I0={true_np[h]:.1f}")

    fig2.suptitle("Distribution of I0 estimates across 100 trials "
                  "(selected HKLs)", fontsize=12)
    fig2.tight_layout()
    path2 = os.path.join(outdir, "mc_poisson_log_histograms.png")
    fig2.savefig(path2, dpi=150)
    print(f"Saved {path2}")
    plt.close("all")


if __name__ == "__main__":
    main()
