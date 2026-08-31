#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Scatter plots with realistic per-HKL parameters from real data.

Compares four estimators:
  1. Median — robust baseline
  2. Log IRLS — fit_gengauss style (positive obs only)
  3. Poisson MLE (fixed grid) — standard quadrature
  4. Poisson MLE (obs-centered) — per-obs λ-space quadrature
"""

from __future__ import annotations

import math
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ═══════════════════════════════════════════════════════════
# Data generation with per-HKL real parameters
# ═══════════════════════════════════════════════════════════

def sample_gengauss(alpha, beta, size, gen=None, generator=None):
    g = gen or generator
    gamma = torch._standard_gamma(torch.full((size,), 1.0 / beta), generator=g)
    signs = 2.0 * torch.bernoulli(torch.full((size,), 0.5), generator=g) - 1.0
    return signs * alpha * gamma.pow(1.0 / beta)


def generate_realistic(*, real_params, n_hkl, beta, bg=10., seed=42, device="cpu"):
    gen = torch.Generator().manual_seed(seed)
    I0_real = real_params["I0"]
    alpha_real = real_params["alpha"]
    n_obs_real = real_params["n_obs"].long()

    idx = torch.randint(len(I0_real), (n_hkl,), generator=gen)
    I0_true = I0_real[idx].clone()
    alpha_true = alpha_real[idx].clone()
    obs_per_hkl = n_obs_real[idx].clone().clamp(min=3, max=300)
    N = int(obs_per_hkl.sum())

    hkl_id = torch.arange(n_hkl).repeat_interleave(obs_per_hkl)
    I0_po = I0_true.repeat_interleave(obs_per_hkl)

    eps = torch.zeros(N)
    offset = 0
    for h in range(n_hkl):
        n = int(obs_per_hkl[h])
        eps[offset:offset + n] = sample_gengauss(
            float(alpha_true[h]), beta, n, generator=gen)
        offset += n

    I_signal = (I0_po * torch.exp(eps)).clamp(min=0)
    I_obs = torch.poisson(I_signal + bg, generator=gen) - bg

    return dict(
        I0_true=I0_true.to(device), alpha_true=alpha_true.to(device),
        obs_per_hkl=obs_per_hkl.to(device), I_obs=I_obs.to(device),
        hkl_id=hkl_id.to(device), bg=bg, N=N, n_hkl=n_hkl,
    )


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

def _quad_range(beta, coverage=0.999):
    a = torch.tensor(1.0 / beta)
    lo, hi = 1.0, 20.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if float(torch.special.gammainc(a, torch.tensor(mid**beta))) < coverage:
            lo = mid
        else:
            hi = mid
    return min(hi, 15.0)


def _log_gg_norm(beta):
    return math.log(beta) - math.log(2) - float(torch.lgamma(torch.tensor(1.0 / beta)))


def _sorted_obs(data):
    """Sort observations by hkl_id, return sorted arrays + offsets."""
    dev = data["I_obs"].device
    order = data["hkl_id"].argsort()
    I_sorted = data["I_obs"][order]
    offsets = torch.zeros(data["n_hkl"] + 1, dtype=torch.long, device=dev)
    offsets[1:] = data["obs_per_hkl"].cumsum(0)
    return I_sorted, offsets


# ═══════════════════════════════════════════════════════════
# Estimators
# ═══════════════════════════════════════════════════════════

def estimate_median(data):
    I_sorted, offsets = _sorted_obs(data)
    n_hkl = data["n_hkl"]
    dev = data["I_obs"].device
    I0 = torch.zeros(n_hkl, device=dev)
    for h in range(n_hkl):
        I0[h] = I_sorted[offsets[h]:offsets[h+1]].median()
    return I0.clamp(min=0.1)


def estimate_irls(data, beta, n_irls=3):
    I_sorted, offsets = _sorted_obs(data)
    n_hkl = data["n_hkl"]
    dev = data["I_obs"].device
    I0 = torch.zeros(n_hkl, device=dev)
    for h in range(n_hkl):
        vals = I_sorted[offsets[h]:offsets[h+1]]
        pos = vals[vals > 0]
        if len(pos) < 3:
            I0[h] = vals.median().clamp(min=0.1)
            continue
        log_pos = torch.log(pos)
        mu = log_pos.median()
        for _ in range(n_irls):
            r = (log_pos - mu).abs().clamp(min=1e-8)
            w = r.pow(beta - 2)
            w = w / w.sum()
            mu = (w * log_pos).sum()
        I0[h] = torch.exp(mu)
    return I0


def estimate_poisson_fixed(data, beta, n_quad=31, max_iter=50):
    """Poisson MLE with fixed ε-grid quadrature."""
    dev = data["I_obs"].device
    n_hkl = data["n_hkl"]
    bg = data["bg"]
    hkl_id = data["hkl_id"]
    obs_per_hkl = data["obs_per_hkl"]

    R = _quad_range(beta)
    K = max(n_quad, round(n_quad * R / 3.0))
    eps_tilde = torch.linspace(-R, R, K, device=dev)
    d_eps = float(eps_tilde[1] - eps_tilde[0])
    log_gg_w = -(eps_tilde.abs() ** beta) + _log_gg_norm(beta) + math.log(d_eps)

    alpha_po = data["alpha_true"].repeat_interleave(obs_per_hkl)
    n_ph = (data["I_obs"] + bg).clamp(min=0)
    log_nf = torch.lgamma(n_ph + 1)

    log_I0 = torch.log(estimate_median(data).clamp(min=0.1)).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([log_I0], lr=1.0, max_iter=max_iter,
                             line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        I0_po = torch.exp(log_I0).repeat_interleave(obs_per_hkl)
        s = torch.exp(alpha_po[None, :] * eps_tilde[:, None])
        lam = (I0_po[None, :] * s + bg).clamp(min=1e-6)
        log_poi = n_ph[None, :] * torch.log(lam) - lam - log_nf[None, :]
        log_joint = log_poi + log_gg_w[:, None]
        log_p = torch.logsumexp(log_joint, dim=0)
        nll = torch.zeros(n_hkl, device=dev)
        nll.scatter_add_(0, hkl_id, -log_p)
        loss = nll.sum()
        loss.backward()
        return loss

    opt.step(closure)
    return torch.exp(log_I0.detach())


def estimate_poisson_obs_centered(data, beta, n_per_obs=31, max_iter=50):
    """Poisson MLE with per-observation λ-space quadrature.

    For each obs j with n_j photons, 31 quadrature points sample the
    Poisson(n_j; λ) peak in λ-space, converted to ε via Jacobian.
    """
    dev = data["I_obs"].device
    n_hkl = data["n_hkl"]
    bg = data["bg"]
    hkl_id = data["hkl_id"]
    obs_per_hkl = data["obs_per_hkl"]
    I_obs = data["I_obs"]
    N = data["N"]
    log_norm = _log_gg_norm(beta)

    alpha_po = data["alpha_true"].repeat_interleave(obs_per_hkl)  # (N,)
    n_ph = (I_obs + bg).clamp(min=0)       # (N,)
    log_nf = torch.lgamma(n_ph + 1)        # (N,)

    # Per-obs λ grid sampling the Poisson peak
    sigma = n_ph.clamp(min=1).sqrt()        # (N,)
    lam_center = torch.maximum(n_ph, torch.full_like(n_ph, bg + 0.1))
    lam_lo = (lam_center - 4 * sigma).clamp(min=bg + 0.1)
    lam_hi = lam_center + 4 * sigma

    t = torch.linspace(0, 1, n_per_obs, device=dev)  # (K,)
    lam = lam_lo[:, None] + (lam_hi - lam_lo)[:, None] * t[None, :]  # (N, K)
    d_lam = (lam_hi - lam_lo) / (n_per_obs - 1)  # (N,)
    signal = lam - bg  # (N, K) — always > 0 since lam > bg + 0.1

    # Precompute I0-independent parts
    log_poi = n_ph[:, None] * torch.log(lam) - lam - log_nf[:, None]  # (N, K)
    log_jac = -torch.log(signal)             # (N, K)
    log_d_lam = torch.log(d_lam)             # (N,)
    log_poi_jac = log_poi + log_jac + log_d_lam[:, None]  # (N, K)

    log_I0 = torch.log(estimate_median(data).clamp(min=0.1)).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([log_I0], lr=1.0, max_iter=max_iter,
                             line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        I0_po = torch.exp(log_I0).repeat_interleave(obs_per_hkl)  # (N,)
        # ε at each quadrature point — depends on per-obs alpha
        # ε = log(signal / I0), but GenGauss uses ε/α
        eps = torch.log(signal / I0_po[:, None])  # (N, K)
        log_gg = -(eps.abs() / alpha_po[:, None]).pow(beta) + log_norm  # (N, K)

        log_integrand = log_gg + log_poi_jac  # (N, K)
        log_p = torch.logsumexp(log_integrand, dim=1)  # (N,)

        nll = torch.zeros(n_hkl, device=dev)
        nll.scatter_add_(0, hkl_id, -log_p)
        loss = nll.sum()
        loss.backward()
        return loss

    opt.step(closure)
    return torch.exp(log_I0.detach())


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    params_path = os.path.join(os.path.dirname(__file__), "real_hkl_params.pt")
    real_params = torch.load(params_path, weights_only=True)
    print(f"Real data: {len(real_params['I0'])} HKLs, "
          f"I0 med={real_params['I0'].median():.1f}, "
          f"alpha med={real_params['alpha'].median():.3f}, "
          f"n_obs med={real_params['n_obs'].float().median():.0f}")

    beta = 1.3
    n_hkl = 5000

    data = generate_realistic(real_params=real_params, n_hkl=n_hkl,
                              beta=beta, device=device)
    true = data["I0_true"].cpu().numpy()
    lt = np.log(true)

    def _r(est):
        return np.corrcoef(lt, np.log(np.clip(est, 1e-6, None)))[0, 1]

    # Compute all estimators
    print("  Median ...", flush=True)
    med = estimate_median(data).cpu().numpy()
    print("  IRLS ...", flush=True)
    irls = estimate_irls(data, beta).cpu().numpy()
    print("  Poisson MLE (fixed grid) ...", flush=True)
    mle_fix = estimate_poisson_fixed(data, beta).cpu().numpy()
    print("  Poisson MLE (obs-centered) ...", flush=True)
    mle_oc = estimate_poisson_obs_centered(data, beta).cpu().numpy()

    r_med, r_irls, r_fix, r_oc = _r(med), _r(irls), _r(mle_fix), _r(mle_oc)
    print(f"\n  Results (real alpha, real I0, real n_obs):")
    print(f"    Median:             r={r_med:.4f}")
    print(f"    IRLS:               r={r_irls:.4f}")
    print(f"    Poisson fixed-grid: r={r_fix:.4f}")
    print(f"    Poisson obs-center: r={r_oc:.4f}")

    # ── Plot: 2×3 ──
    alphas_fixed = [0.8, 1.2, 1.5]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    lo, hi = 0.3, 2e3

    def _setup(ax, title):
        ax.plot([lo, hi], [lo, hi], "k-", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("I₀ true"); ax.set_ylabel("I₀ estimated")
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

    # (0,0) All four methods
    ax = axes[0, 0]
    ax.scatter(true, med, s=2, alpha=0.12, c="tomato", edgecolors="none", label="Median")
    ax.scatter(true, irls, s=2, alpha=0.12, c="seagreen", edgecolors="none", label="IRLS")
    ax.scatter(true, mle_fix, s=2, alpha=0.12, c="steelblue", edgecolors="none", label="Fixed grid")
    ax.scatter(true, mle_oc, s=2, alpha=0.12, c="darkorchid", edgecolors="none", label="Obs-centered")
    _setup(ax, f"Real α (med={real_params['alpha'].median():.2f})")
    ax.text(0.05, 0.95,
            f"Obs-ctr: r={r_oc:.4f}\nIRLS:    r={r_irls:.4f}\n"
            f"Median:  r={r_med:.4f}\nFixed:   r={r_fix:.4f}",
            transform=ax.transAxes, fontsize=7, va="top", family="monospace",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.legend(loc="lower right", fontsize=6, markerscale=3)

    # (0,1) IRLS vs obs-centered head-to-head
    ax = axes[0, 1]
    ax.scatter(true, irls, s=2, alpha=0.15, c="seagreen", edgecolors="none", label="IRLS")
    ax.scatter(true, mle_oc, s=2, alpha=0.15, c="darkorchid", edgecolors="none", label="Obs-centered")
    _setup(ax, "IRLS vs Obs-centered (real α)")
    ax.text(0.05, 0.95,
            f"Obs-ctr: r={r_oc:.4f}\nIRLS:    r={r_irls:.4f}",
            transform=ax.transAxes, fontsize=8, va="top", family="monospace",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.legend(loc="lower right", fontsize=7, markerscale=3)

    # (0,2) Bar chart
    ax = axes[0, 2]
    methods = ["Median", "IRLS", "Fixed grid", "Obs-centered"]
    rs = [r_med, r_irls, r_fix, r_oc]
    colors = ["tomato", "seagreen", "steelblue", "darkorchid"]
    bars = ax.barh(range(len(methods)), rs, color=colors, alpha=0.7)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel("r(log I₀_true, log I₀_est)")
    ax.set_title("Correlation comparison (real α)")
    ax.set_xlim(min(rs) - 0.02, 1.0)
    for bar, r in zip(bars, rs):
        ax.text(r + 0.002, bar.get_y() + bar.get_height()/2, f"{r:.4f}",
                va="center", fontsize=8)

    # Row 2: fixed alpha
    for j, alpha_fix in enumerate(alphas_fixed):
        print(f"\n  alpha={alpha_fix} (fixed) ...", end=" ", flush=True)
        pf = dict(real_params)
        pf["alpha"] = torch.full_like(real_params["alpha"], alpha_fix)
        df = generate_realistic(real_params=pf, n_hkl=n_hkl, beta=beta, device=device)
        tf = df["I0_true"].cpu().numpy()
        ltf = np.log(tf)

        med_f = estimate_median(df).cpu().numpy()
        irls_f = estimate_irls(df, beta).cpu().numpy()
        oc_f = estimate_poisson_obs_centered(df, beta).cpu().numpy()

        def _rf(est):
            return np.corrcoef(ltf, np.log(np.clip(est, 1e-6, None)))[0, 1]

        r_med_f, r_irls_f, r_oc_f = _rf(med_f), _rf(irls_f), _rf(oc_f)
        print(f"OC r={r_oc_f:.4f} | IRLS r={r_irls_f:.4f} | Med r={r_med_f:.4f}")

        ax = axes[1, j]
        ax.scatter(tf, med_f, s=2, alpha=0.12, c="tomato", edgecolors="none", label="Median")
        ax.scatter(tf, irls_f, s=2, alpha=0.12, c="seagreen", edgecolors="none", label="IRLS")
        ax.scatter(tf, oc_f, s=2, alpha=0.12, c="darkorchid", edgecolors="none", label="Obs-centered")
        _setup(ax, f"α={alpha_fix} fixed, real I₀ + n_obs")
        ax.text(0.05, 0.95,
                f"Obs-ctr: r={r_oc_f:.4f}\nIRLS:    r={r_irls_f:.4f}\nMedian:  r={r_med_f:.4f}",
                transform=ax.transAxes, fontsize=8, va="top", family="monospace",
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        if j == 0:
            ax.legend(loc="lower right", fontsize=7, markerscale=3)

    fig.suptitle(
        f"Obs-centered Poisson quadrature vs IRLS vs Median — "
        f"real distributions, β={beta}, {n_hkl} HKLs",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "scatter_bias.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
