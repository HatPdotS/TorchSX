#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Investigate analytical Poisson correction for fit_alpha.

The problem: fit_alpha computes α^β = (β/n) Σ |log(I_obs/I0)|^β
But log(I_obs/I0) = ε + log(1 + δ_poi/(I0·exp(ε)))
where δ_poi is the Poisson noise. So the measured |r|^β has a Poisson
contribution that inflates alpha at low I0.

Approach: estimate E[|ε_poi|^β | I0, bg] where ε_poi = log(n_photon/λ)
with n ~ Poisson(λ) and λ = I0 + bg. Then subtract this from α_raw^β.

For β close to 2 (Gaussian case), E[|ε_poi|^2] ≈ 1/λ (well-known).
For general β, we compute it numerically via MC and fit a simple formula.
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


def expected_poisson_abs_r_beta(I0_vals, bg, beta, n_mc=100000):
    """Monte Carlo estimate of E[|log(n/λ)|^β] where n ~ Poisson(λ), λ = I0 + bg.

    Returns array of shape (len(I0_vals),).
    """
    results = np.zeros(len(I0_vals))
    for i, I0 in enumerate(I0_vals):
        lam = I0 + bg
        # Sample from Poisson
        n = np.random.poisson(lam, size=n_mc).astype(float)
        n = np.clip(n, 1, None)  # avoid log(0)
        eps_poi = np.log(n / lam)
        results[i] = np.mean(np.abs(eps_poi)**beta)
    return results


def fit_alpha_corrected(I_obs, hkl_id, I0, beta_tensor, obs_counts_pos, n_hkl,
                        bg, device):
    """Fit alpha with Poisson correction subtracted."""
    # Standard fit_alpha gives α_raw
    alpha_raw = fit_alpha(I_obs, hkl_id, I0, beta_tensor, obs_counts_pos, n_hkl)

    beta = float(beta_tensor[0])

    # Compute expected Poisson contribution per HKL:
    # E[|ε_poi|^β] where ε_poi = log(Poisson(λ)/λ), λ = I0 + bg
    # For large λ: ε_poi ≈ N(0, 1/λ), so E[|ε_poi|^β] ≈ E[|N(0,1)|^β] / λ^(β/2)
    # E[|N(0,1)|^β] = 2^(β/2) * Gamma((β+1)/2) / √π
    import math
    E_abs_z_beta = 2**(beta/2) * math.gamma((beta + 1) / 2) / math.sqrt(math.pi)

    lam = (I0 + bg).clamp(min=1.0)  # λ = I0 + bg
    poisson_correction = E_abs_z_beta / lam.pow(beta / 2)

    # α_raw^β = α_true^β + (β/n) · n · poisson_correction
    #         = α_true^β + β · poisson_correction
    # Wait — let's be more careful.
    # fit_alpha computes: α^β = (β/n) Σ |r_j|^β
    # The residuals are r_j = log(I_obs_j / I0_h) = ε_j + ε_poi_j
    # If ε and ε_poi are independent:
    #   E[|r|^β] ≈ E[|ε|^β] + E[|ε_poi|^β]  (NOT exact for β≠2)
    # For β=2 this is exact (variances add). For β≠2 it's an approximation.
    # So: α_raw^β ≈ (β/n) · n · (E[|ε|^β] + E[|ε_poi|^β])
    #             = α_true^β + β · E[|ε_poi|^β]
    # Therefore: α_corrected^β = α_raw^β - β · E[|ε_poi|^β]

    alpha_raw_pow_b = alpha_raw.pow(beta)
    correction = beta * poisson_correction
    alpha_corr_pow_b = (alpha_raw_pow_b - correction).clamp(min=1e-8)
    alpha_corrected = alpha_corr_pow_b.pow(1.0 / beta)

    return alpha_corrected, alpha_raw


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
    n_trials = 200

    gen_gt = torch.Generator().manual_seed(0)
    log_I0 = torch.empty(n_hkl).uniform_(np.log(1.0), np.log(10000.0), generator=gen_gt)
    I0_true = torch.exp(log_I0)

    # ── First verify the Gaussian approximation for E[|ε_poi|^β] ──
    print("Verifying Poisson correction formula...")
    import math
    E_abs_z_beta = 2**(beta_val/2) * math.gamma((beta_val + 1) / 2) / math.sqrt(math.pi)
    I0_test = np.logspace(0, 4, 50)
    mc_result = expected_poisson_abs_r_beta(I0_test, background, beta_val)
    approx = E_abs_z_beta / (I0_test + background)**(beta_val / 2)

    print(f"  E[|N(0,1)|^β] = {E_abs_z_beta:.4f}  (β={beta_val})")
    print(f"  {'I0':>8s}  {'MC':>10s}  {'approx':>10s}  {'ratio':>8s}")
    for j in [0, 5, 10, 20, 30, 40, 49]:
        print(f"  {I0_test[j]:8.1f}  {mc_result[j]:10.6f}  {approx[j]:10.6f}  "
              f"{approx[j]/mc_result[j]:8.3f}")

    # ── Run MC trials with both alpha estimators ─────────────────
    for alpha_val in [0.4, 1.2]:
        alpha_true = torch.full((n_hkl,), alpha_val)
        beta_true = torch.full((n_hkl,), beta_val)

        alpha_raw_all = torch.zeros(n_trials, n_hkl)
        alpha_corr_all = torch.zeros(n_trials, n_hkl)
        I0_est_all = torch.zeros(n_trials, n_hkl)
        t0 = time.time()

        print(f"\n{'='*60}")
        print(f"ALPHA = {alpha_val}")
        print(f"{'='*60}")

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

            alpha_corr, alpha_raw = fit_alpha_corrected(
                I_pos, hkl_pos, I0_p, beta_true.to(device),
                obs_counts_pos, n_hkl, background, device)

            alpha_raw_all[trial] = alpha_raw.cpu()
            alpha_corr_all[trial] = alpha_corr.cpu()
            I0_est_all[trial] = I0_p.cpu()

            if (trial + 1) % 50 == 0:
                print(f"  trial {trial+1}/{n_trials}  ({time.time()-t0:.1f}s)")

        print(f"Done in {time.time()-t0:.1f}s")

        true_np = I0_true.numpy()
        bins = np.array([0, 3, 10, 30, 100, 300, 1000, 10000, 1e6])

        print(f"\n  {'I0 range':>16s}  {'raw α':>10s}  {'corrected α':>12s}  {'true α':>8s}")
        for i in range(len(bins) - 1):
            mask = (true_np >= bins[i]) & (true_np < bins[i+1])
            if mask.sum() == 0:
                continue
            raw = np.mean(alpha_raw_all.numpy()[:, mask])
            corr = np.mean(alpha_corr_all.numpy()[:, mask])
            print(f"  [{bins[i]:8.0f}, {bins[i+1]:8.0f})  {raw:10.4f}  "
                  f"{corr:12.4f}  {alpha_val:8.4f}")

        raw_rmse = np.sqrt(np.mean((alpha_raw_all.numpy() - alpha_val)**2))
        corr_rmse = np.sqrt(np.mean((alpha_corr_all.numpy() - alpha_val)**2))
        print(f"\n  RMSE: raw={raw_rmse:.4f}, corrected={corr_rmse:.4f}")

    # ── Plot ────────────────────────────────────────────────────
    outdir = os.path.dirname(__file__)

    # Verification plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(I0_test, mc_result, "bo", ms=4, label="MC (100k samples)")
    ax.loglog(I0_test, approx, "r-", lw=2, label=f"Gaussian approx: {E_abs_z_beta:.3f}/(I0+bg)^(β/2)")
    ax.set_xlabel("I0"); ax.set_ylabel(f"E[|log(Poi(λ)/λ)|^{beta_val}]")
    ax.set_title("Poisson correction: MC vs Gaussian approximation")
    ax.legend()
    fig.tight_layout()
    path0 = os.path.join(outdir, "mc_poisson_correction_verify.png")
    fig.savefig(path0, dpi=150)
    print(f"\nSaved {path0}")
    plt.close("all")


if __name__ == "__main__":
    main()
