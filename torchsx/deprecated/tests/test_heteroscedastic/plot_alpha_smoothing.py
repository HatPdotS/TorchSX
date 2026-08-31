#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Compare alpha regularization methods.

1. Raw fit_alpha (no smoothing)
2. Uniform Laplacian smoothing
3. I0-weighted Laplacian smoothing
4. Resolution prior + penalized NLL (NEW)
"""

from __future__ import annotations

import os, sys, time
import torch
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tests.test_heteroscedastic.generate import sample_gengauss
from torchsx.deprecated.merge._dstep import (
    fit_I0_poisson_log, fit_alpha, _fit_alpha_resolution_prior,
)
from torchsx.deprecated.merge._distribution import _laplacian_smooth


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


def build_simple_adjacency(n_hkl, device="cpu"):
    src, dst = [], []
    for delta in [-2, -1, 1, 2]:
        for h in range(n_hkl):
            nb = (h + delta) % n_hkl
            src.append(h)
            dst.append(nb)
    edge_src = torch.tensor(src, dtype=torch.long, device=device)
    edge_dst = torch.tensor(dst, dtype=torch.long, device=device)
    n_neighbors = torch.zeros(n_hkl, device=device)
    n_neighbors.scatter_add_(0, edge_dst, torch.ones(len(edge_dst), device=device))
    n_neighbors = n_neighbors.clamp(min=1)
    return edge_src, edge_dst, n_neighbors


def run_comparison(alpha_val, device):
    n_hkl = 300
    obs_per_hkl = 50
    beta_val = 1.3
    background = 10.0
    n_trials = 200
    smooth_lambda = 0.3
    n_smooth_iter = 5

    gen_gt = torch.Generator().manual_seed(0)
    log_I0 = torch.empty(n_hkl).uniform_(np.log(1.0), np.log(10000.0), generator=gen_gt)
    I0_true = torch.exp(log_I0)
    alpha_true = torch.full((n_hkl,), alpha_val)
    beta_true = torch.full((n_hkl,), beta_val)

    # Fake d_spacing: monotonically decreasing with HKL index (sorted by I0)
    # This simulates the real case where low I0 correlates with high resolution
    sorted_idx = I0_true.argsort(descending=True)
    d_spacing = torch.zeros(n_hkl)
    d_spacing[sorted_idx] = torch.linspace(20.0, 1.5, n_hkl)  # strong=low-res, weak=high-res
    d_spacing = d_spacing.to(device)

    edge_src, edge_dst, n_neighbors = build_simple_adjacency(n_hkl, device)

    print(f"\nn_hkl={n_hkl}, obs_per_hkl={obs_per_hkl}, alpha={alpha_val}, "
          f"beta={beta_val}, bg={background}, trials={n_trials}")

    alpha_raw_all = torch.zeros(n_trials, n_hkl)
    alpha_uniform_all = torch.zeros(n_trials, n_hkl)
    alpha_weighted_all = torch.zeros(n_trials, n_hkl)
    alpha_prior_all = torch.zeros(n_trials, n_hkl)
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

        pos = I_obs > 0
        I_pos = I_obs[pos]
        hkl_pos = hkl_id[pos]
        obs_counts_pos = torch.zeros(n_hkl, dtype=torch.long, device=device)
        obs_counts_pos.scatter_add_(0, hkl_pos, torch.ones_like(hkl_pos))

        # 1. Raw
        alpha_raw = fit_alpha(I_pos, hkl_pos, I0_p, beta_true.to(device),
                              obs_counts_pos, n_hkl)

        # 2. Uniform Laplacian
        log_alpha = torch.log(alpha_raw.clamp(min=1e-6))
        log_alpha_uniform = _laplacian_smooth(
            log_alpha, edge_src, edge_dst, n_neighbors,
            n_smooth_iter, smooth_lambda)

        # 3. I0-weighted Laplacian
        log_alpha_weighted = _laplacian_smooth(
            log_alpha, edge_src, edge_dst, n_neighbors,
            n_smooth_iter, smooth_lambda, node_weights=I0_p)

        # 4. Resolution prior + penalized NLL
        a_prior = _fit_alpha_resolution_prior(
            alpha_raw, I0_p, d_spacing, beta_true.to(device),
            bg=background, n_shells=20)
        I0_ref = I0_p[I0_p > 1.0].median().clamp(min=1.0)
        prior_lam = 1.0 * I0_ref / I0_p.clamp(min=0.1)
        alpha_reg = fit_alpha(
            I_pos, hkl_pos, I0_p, beta_true.to(device),
            obs_counts_pos, n_hkl,
            alpha_prior=a_prior, prior_lambda=prior_lam)

        alpha_raw_all[trial] = alpha_raw.cpu()
        alpha_uniform_all[trial] = torch.exp(log_alpha_uniform).cpu()
        alpha_weighted_all[trial] = torch.exp(log_alpha_weighted).cpu()
        alpha_prior_all[trial] = alpha_reg.cpu()

        if (trial + 1) % 50 == 0:
            print(f"  trial {trial+1}/{n_trials}  ({time.time()-t0:.1f}s)")

    print(f"Done in {time.time()-t0:.1f}s")

    true_np = I0_true.numpy()
    bins = np.array([0, 3, 10, 30, 100, 300, 1000, 10000, 1e6])

    methods = [("raw", alpha_raw_all), ("uniform", alpha_uniform_all),
               ("I0-weighted", alpha_weighted_all), ("prior+penalty", alpha_prior_all)]

    print(f"\n{'='*70}")
    print(f"ALPHA BIAS BY INTENSITY (true α={alpha_val})")
    print(f"{'='*70}")
    header = f"  {'I0 range':>16s}"
    for name, _ in methods:
        header += f"  {name:>14s}"
    print(header)
    for i in range(len(bins) - 1):
        mask = (true_np >= bins[i]) & (true_np < bins[i+1])
        if mask.sum() == 0:
            continue
        row = f"  [{bins[i]:8.0f}, {bins[i+1]:8.0f})"
        for _, arr in methods:
            row += f"  {np.mean(arr.numpy()[:, mask]):14.4f}"
        print(row)

    for name, arr in methods:
        a = arr.numpy()
        rmse = np.sqrt(np.mean((a - alpha_val)**2))
        print(f"\n  {name}: mean={np.mean(a):.4f}, RMSE={rmse:.4f}")

    # Plot
    outdir = os.path.dirname(__file__)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    for col, (name, arr) in enumerate(methods):
        ax = axes[col]
        mean_alpha = np.mean(arr.numpy(), axis=0)
        std_alpha = np.std(arr.numpy(), axis=0)
        ax.scatter(true_np, mean_alpha, s=4, alpha=0.5, zorder=3)
        si = np.argsort(true_np)
        ax.fill_between(
            true_np[si],
            np.interp(true_np[si], true_np[si], (mean_alpha - std_alpha)[si]),
            np.interp(true_np[si], true_np[si], (mean_alpha + std_alpha)[si]),
            alpha=0.15, color="blue")
        ax.axhline(alpha_val, color="red", ls="--", lw=2, label=f"true α={alpha_val}")
        ax.set_xscale("log")
        ax.set_xlabel("True I0")
        ax.set_ylabel("Fitted α")
        ax.set_title(name)
        ax.set_ylim(alpha_val * 0.5, alpha_val * 2.5)
        ax.legend()

    fig.suptitle(f"Alpha regularization: α={alpha_val}, {n_trials} trials", fontsize=12)
    fig.tight_layout()
    path = os.path.join(outdir, f"mc_alpha_smoothing_a{alpha_val:.1f}.png")
    fig.savefig(path, dpi=150)
    print(f"\nSaved {path}")
    plt.close("all")

    return methods


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    for alpha_val in [0.4, 1.2]:
        print(f"\n{'#'*70}")
        print(f"# ALPHA = {alpha_val}")
        print(f"{'#'*70}")
        run_comparison(alpha_val, device)


if __name__ == "__main__":
    main()
