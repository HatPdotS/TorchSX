#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Grid-scan MLE: is there a systematic relationship between I0 and bias?

For each HKL, compute ratio = I0_MLE / I0_true, then check whether
log(ratio) correlates with log(I0_true).  Fine I0 bins to see the shape.

Run:
    python tests/test_heteroscedastic/grid_scan_bias_vs_I0.py --device cuda
"""

from __future__ import annotations

import argparse
import math
import time

import torch
from torch import Tensor


# ═══════════════════════════════════════════════════════════
# Data generation
# ═══════════════════════════════════════════════════════════

def sample_gengauss(alpha, beta, size, gen=None):
    g = torch._standard_gamma(torch.full((size,), 1.0 / beta), generator=gen)
    signs = 2.0 * torch.bernoulli(torch.full((size,), 0.5), generator=gen) - 1.0
    return signs * alpha * g.pow(1.0 / beta)


def generate(*, n_hkl, obs_per_hkl, alpha, beta, I0_range=(10., 100_000.),
             bg=10., seed=42, device="cpu"):
    gen = torch.Generator().manual_seed(seed)
    N = n_hkl * obs_per_hkl
    log_I0 = torch.empty(n_hkl).uniform_(
        math.log(I0_range[0]), math.log(I0_range[1]), generator=gen)
    I0_true = torch.exp(log_I0)

    eps = sample_gengauss(alpha, beta, N, gen)
    I0_po = I0_true.repeat_interleave(obs_per_hkl)
    I_signal = (I0_po * torch.exp(eps)).clamp(min=0)
    I_obs = torch.poisson(I_signal + bg, generator=gen) - bg
    sigma2_poi = I_obs.clamp(min=1.0) + bg

    I_obs_2d = I_obs.view(n_hkl, obs_per_hkl)
    sigma2_2d = sigma2_poi.view(n_hkl, obs_per_hkl)
    I0_median = I_obs_2d.median(dim=1).values.clamp(min=1.0)

    return dict(
        I0_true=I0_true.to(device), I0_median=I0_median.to(device),
        I_obs=I_obs_2d.to(device), sigma2_poi=sigma2_2d.to(device), bg=bg,
    )


# ═══════════════════════════════════════════════════════════
# Quadrature
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


def _quad_setup(alpha, beta, n_quad, device):
    R = _quad_range(beta)
    K = max(n_quad, round(n_quad * R / 3.0))
    eps = torch.linspace(-R, R, K, device=device)
    d = float(eps[1] - eps[0])
    log_w = -(eps.abs() ** beta) + math.log(beta) - math.log(2) \
            - float(torch.lgamma(torch.tensor(1.0 / beta))) + math.log(d)
    s = torch.exp(alpha * eps)
    return s, log_w


# ═══════════════════════════════════════════════════════════
# Grid-scan
# ═══════════════════════════════════════════════════════════

def grid_scan_gauss(data, alpha, beta, n_grid=1000, n_quad=31):
    dev = data["I_obs"].device
    H, J = data["I_obs"].shape
    s, log_w = _quad_setup(alpha, beta, n_quad, dev)
    inv2s2 = 0.5 / data["sigma2_poi"].clamp(min=1e-6)
    lnorm = -0.5 * torch.log(2 * math.pi * data["sigma2_poi"].clamp(min=1e-6))
    ratio = torch.logspace(-2, 2, n_grid, device=dev)
    nll = torch.empty(H, n_grid, device=dev)
    ref = data["I0_median"]
    for g in range(n_grid):
        I0t = ref * ratio[g]
        pred = s[:, None] * I0t[None, :]
        resid = data["I_obs"][None, :, :] - pred[:, :, None]
        lg = -resid**2 * inv2s2[None, :, :] + lnorm[None, :, :]
        lj = lg + log_w[:, None, None]
        nll[:, g] = -torch.logsumexp(lj, dim=0).sum(dim=1)
    return ref * ratio[nll.argmin(dim=1)]


def grid_scan_poisson(data, alpha, beta, n_grid=1000, n_quad=31):
    dev = data["I_obs"].device
    H, J = data["I_obs"].shape
    bg = data["bg"]
    s, log_w = _quad_setup(alpha, beta, n_quad, dev)
    n_ph = (data["I_obs"] + bg).clamp(min=0)
    log_nf = torch.lgamma(n_ph + 1)
    ratio = torch.logspace(-2, 2, n_grid, device=dev)
    nll = torch.empty(H, n_grid, device=dev)
    ref = data["I0_median"]
    for g in range(n_grid):
        I0t = ref * ratio[g]
        lam = (s[:, None] * I0t[None, :])[:, :, None] + bg
        lam = lam.expand(-1, -1, J).clamp(min=1e-6)
        lp = n_ph[None, :, :] * torch.log(lam) - lam - log_nf[None, :, :]
        lj = lp + log_w[:, None, None]
        nll[:, g] = -torch.logsumexp(lj, dim=0).sum(dim=1)
    return ref * ratio[nll.argmin(dim=1)]


# ═══════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════

def analyze_bias_vs_I0(I0_mle, I0_true, label, n_bins=15):
    """Fine-binned bias vs I0 + correlation analysis."""
    ratio = (I0_mle / I0_true).cpu()
    log_ratio = torch.log(ratio)
    log_I0 = torch.log(I0_true.cpu())

    # Pearson correlation between log(I0) and log(ratio)
    x = log_I0 - log_I0.mean()
    y = log_ratio - log_ratio.mean()
    r_pearson = float((x * y).sum() / (x.norm() * y.norm() + 1e-12))

    # Spearman rank correlation
    rx = log_I0.argsort().argsort().float()
    ry = log_ratio.argsort().argsort().float()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    r_spearman = float((rx * ry).sum() / (rx.norm() * ry.norm() + 1e-12))

    # Linear regression: log(ratio) = a * log(I0) + b
    # slope a tells us: ratio ∝ I0^a, so a=0 means no I0-dependence
    A = torch.stack([log_I0, torch.ones_like(log_I0)], dim=1)
    sol = torch.linalg.lstsq(A, log_ratio).solution
    slope = float(sol[0])
    intercept = float(sol[1])

    print(f"\n  {label}:")
    print(f"    Pearson  r(log I0, log ratio) = {r_pearson:+.4f}")
    print(f"    Spearman ρ(log I0, log ratio) = {r_spearman:+.4f}")
    print(f"    Linear fit: log(ratio) = {slope:+.6f} * log(I0) + {intercept:+.4f}")
    print(f"    → ratio ∝ I0^{slope:.4f}  (0 = no I0-dependence)")

    # Fine bins
    log_edges = torch.linspace(float(log_I0.min()), float(log_I0.max()), n_bins + 1)
    print(f"\n    {'I0_center':>10} | {'n':>5} | {'med(ratio)':>10} | {'mean(ratio)':>11} | {'std(ratio)':>10}")
    print(f"    {'-'*10}-+-{'-'*5}-+-{'-'*10}-+-{'-'*11}-+-{'-'*10}")
    for i in range(n_bins):
        mask = (log_I0 >= log_edges[i]) & (log_I0 < log_edges[i + 1])
        n = int(mask.sum())
        if n < 5:
            continue
        r = ratio[mask]
        center = math.exp(float((log_edges[i] + log_edges[i + 1]) / 2))
        print(f"    {center:>10.0f} | {n:>5} | {float(r.median()):>10.4f} | "
              f"{float(r.mean()):>11.4f} | {float(r.std()):>10.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-hkl", type=int, default=2000)
    parser.add_argument("--n-grid", type=int, default=1000)
    args = parser.parse_args()
    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "cpu"
    print(f"Device: {dev}, n_hkl={args.n_hkl}, grid={args.n_grid}")

    beta = 1.3

    # Representative settings
    settings = [
        (0.5, 100),
        (1.0, 100),
        (1.2,  50),
        (1.2, 100),
        (1.2, 200),
        (1.5, 100),
        (2.0, 100),
    ]

    for alpha, obs in settings:
        print(f"\n{'='*72}")
        print(f"  alpha={alpha}, beta={beta}, obs/hkl={obs}")
        print(f"{'='*72}")

        data = generate(n_hkl=args.n_hkl, obs_per_hkl=obs,
                        alpha=alpha, beta=beta, device=dev)
        true = data["I0_true"]

        # Median baseline
        analyze_bias_vs_I0(data["I0_median"], true, "Median estimator")

        # Gaussian grid scan
        I0_g = grid_scan_gauss(data, alpha, beta, n_grid=args.n_grid)
        analyze_bias_vs_I0(I0_g, true, "Gaussian NLL MLE")

        # Poisson grid scan
        I0_p = grid_scan_poisson(data, alpha, beta, n_grid=args.n_grid)
        analyze_bias_vs_I0(I0_p, true, "Poisson NLL MLE")

    print("\n\nDone.")


if __name__ == "__main__":
    main()
