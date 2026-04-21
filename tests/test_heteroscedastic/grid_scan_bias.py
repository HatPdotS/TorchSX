#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Grid-scan MLE to characterize I0 bias vs (I0, alpha, n_obs).

For each synthetic HKL with known (I0, alpha, beta=1.3):
  1. Generate n_obs observations from GenGauss × Poisson model
  2. Evaluate marginal NLL on a dense I0 grid (true alpha, beta fixed)
  3. Find I0_MLE = argmin NLL(I0)
  4. Compute ratio = I0_MLE / I0_true

This eliminates optimizer effects — any bias is intrinsic to the likelihood.

Two noise models compared:
  - Gaussian: N(I_obs; I0·exp(ε), σ²_poi) — as used in the EM D-step
  - Poisson:  Poisson(n; I0·exp(ε) + bg) — exact photon counting

Run:
    python tests/test_heteroscedastic/grid_scan_bias.py [--device cuda]
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


def sample_gengauss(alpha: float, beta: float, size: int, gen=None) -> Tensor:
    g = torch._standard_gamma(torch.full((size,), 1.0 / beta), generator=gen)
    signs = 2.0 * torch.bernoulli(torch.full((size,), 0.5), generator=gen) - 1.0
    return signs * alpha * g.pow(1.0 / beta)


def generate(
    *, n_hkl: int, obs_per_hkl: int, alpha: float, beta: float,
    I0_range: tuple[float, float] = (10.0, 100_000.0),
    bg: float = 10.0, seed: int = 42, device: str = "cpu",
) -> dict:
    gen = torch.Generator().manual_seed(seed)
    N = n_hkl * obs_per_hkl

    log_I0 = torch.empty(n_hkl).uniform_(
        math.log(I0_range[0]), math.log(I0_range[1]), generator=gen,
    )
    I0_true = torch.exp(log_I0)

    eps = sample_gengauss(alpha, beta, N, gen)
    I0_po = I0_true.repeat_interleave(obs_per_hkl)
    I_signal = (I0_po * torch.exp(eps)).clamp(min=0)
    I_obs = torch.poisson(I_signal + bg, generator=gen) - bg
    sigma2_poi = I_obs.clamp(min=1.0) + bg

    # Reshape to (H, J) for vectorised grid scan
    I_obs_2d = I_obs.view(n_hkl, obs_per_hkl)
    sigma2_2d = sigma2_poi.view(n_hkl, obs_per_hkl)

    # Per-HKL median as I0 reference for the grid
    I0_median = I_obs_2d.median(dim=1).values.clamp(min=1.0)

    return dict(
        I0_true=I0_true.to(device),
        I0_median=I0_median.to(device),
        I_obs=I_obs_2d.to(device),
        sigma2_poi=sigma2_2d.to(device),
        bg=bg,
    )


# ═══════════════════════════════════════════════════════════
# Quadrature setup
# ═══════════════════════════════════════════════════════════


def _quad_range(beta: float, coverage: float = 0.999) -> float:
    """Adaptive quadrature range for GenGauss(0,1,β)."""
    a = torch.tensor(1.0 / beta)
    lo, hi = 1.0, 20.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if float(torch.special.gammainc(a, torch.tensor(mid ** beta))) < coverage:
            lo = mid
        else:
            hi = mid
    return min(hi, 15.0)


def _quad_setup(alpha: float, beta: float, n_quad: int, device: str):
    """Quadrature grid, scale factors, and GenGauss log-weights."""
    R = _quad_range(beta)
    K = max(n_quad, round(n_quad * R / 3.0))
    eps = torch.linspace(-R, R, K, device=device)
    d_eps = float(eps[1] - eps[0])

    log_w = (
        -(eps.abs() ** beta)
        + math.log(beta) - math.log(2)
        - float(torch.lgamma(torch.tensor(1.0 / beta)))
        + math.log(d_eps)
    )  # (K,) log p_GG(ε̃) · Δε
    s = torch.exp(alpha * eps)  # (K,) scale factors

    return s, log_w, K


# ═══════════════════════════════════════════════════════════
# Grid-scan MLE
# ═══════════════════════════════════════════════════════════


def grid_scan_gauss(
    data: dict, alpha: float, beta: float,
    n_grid: int = 1000, n_quad: int = 31,
) -> Tensor:
    """Grid-scan MLE with Gaussian noise approximation (as in D-step)."""
    dev = data["I_obs"].device
    H, J = data["I_obs"].shape
    s, log_w, K = _quad_setup(alpha, beta, n_quad, dev)

    inv2s2 = 0.5 / data["sigma2_poi"].clamp(min=1e-6)   # (H, J)
    lnorm = -0.5 * torch.log(2 * math.pi * data["sigma2_poi"].clamp(min=1e-6))

    ratio = torch.logspace(-2, 2, n_grid, device=dev)    # 0.01× to 100× median
    nll = torch.empty(H, n_grid, device=dev)
    ref = data["I0_median"]                               # (H,)

    for g in range(n_grid):
        I0t = ref * ratio[g]                              # (H,)
        pred = s[:, None] * I0t[None, :]                  # (K, H)
        resid = data["I_obs"][None, :, :] - pred[:, :, None]  # (K, H, J)
        lg = -resid ** 2 * inv2s2[None, :, :] + lnorm[None, :, :]
        lj = lg + log_w[:, None, None]
        nll[:, g] = -torch.logsumexp(lj, dim=0).sum(dim=1)

    return ref * ratio[nll.argmin(dim=1)]


def grid_scan_poisson(
    data: dict, alpha: float, beta: float,
    n_grid: int = 1000, n_quad: int = 31,
) -> Tensor:
    """Grid-scan MLE with exact Poisson likelihood."""
    dev = data["I_obs"].device
    H, J = data["I_obs"].shape
    bg = data["bg"]
    s, log_w, K = _quad_setup(alpha, beta, n_quad, dev)

    n_ph = (data["I_obs"] + bg).clamp(min=0)              # (H, J) photon counts
    log_nf = torch.lgamma(n_ph + 1)                       # (H, J)

    ratio = torch.logspace(-2, 2, n_grid, device=dev)
    nll = torch.empty(H, n_grid, device=dev)
    ref = data["I0_median"]

    for g in range(n_grid):
        I0t = ref * ratio[g]
        lam = (s[:, None] * I0t[None, :])[:, :, None] + bg   # (K, H, 1)
        lam = lam.expand(-1, -1, J).clamp(min=1e-6)          # (K, H, J)
        lp = n_ph[None, :, :] * torch.log(lam) - lam - log_nf[None, :, :]
        lj = lp + log_w[:, None, None]
        nll[:, g] = -torch.logsumexp(lj, dim=0).sum(dim=1)

    return ref * ratio[nll.argmin(dim=1)]


# ═══════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════

# I0 bins (log10 scale)
I0_BIN_EDGES = [10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]


def bin_stats(I0_mle: Tensor, I0_true: Tensor) -> list[dict]:
    """Compute bias statistics in I0 bins."""
    ratio = (I0_mle / I0_true).cpu()
    I0_cpu = I0_true.cpu()
    rows = []
    for i in range(len(I0_BIN_EDGES) - 1):
        lo, hi = I0_BIN_EDGES[i], I0_BIN_EDGES[i + 1]
        mask = (I0_cpu >= lo) & (I0_cpu < hi)
        n = int(mask.sum())
        if n < 3:
            continue
        r = ratio[mask]
        med = float(r.median())
        q25 = float(r.quantile(0.25))
        q75 = float(r.quantile(0.75))
        rows.append(dict(
            lo=lo, hi=hi, n=n,
            median=med, q25=q25, q75=q75,
            iqr=q75 - q25,
        ))
    # Overall
    med = float(ratio.median())
    q25 = float(ratio.quantile(0.25))
    q75 = float(ratio.quantile(0.75))
    rows.append(dict(
        lo=I0_BIN_EDGES[0], hi=I0_BIN_EDGES[-1], n=len(ratio),
        median=med, q25=q25, q75=q75, iqr=q75 - q25,
    ))
    return rows


def print_table(rows: list[dict], label: str = ""):
    """Print bin statistics table."""
    print(f"  {'I0 range':>16} | {'n':>5} | {'med(ratio)':>10} | {'Q25':>7} {'Q75':>7} | {'IQR':>7}")
    print(f"  {'-'*16}-+-{'-'*5}-+-{'-'*10}-+-{'-'*7}-{'-'*7}-+-{'-'*7}")
    for r in rows[:-1]:
        print(f"  {r['lo']:>7.0f}-{r['hi']:>7.0f} | {r['n']:>5} | "
              f"{r['median']:>10.4f} | {r['q25']:>7.4f} {r['q75']:>7.4f} | {r['iqr']:>7.4f}")
    # Overall
    r = rows[-1]
    print(f"  {'Overall':>16} | {r['n']:>5} | "
          f"{r['median']:>10.4f} | {r['q25']:>7.4f} {r['q75']:>7.4f} | {r['iqr']:>7.4f}")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-hkl", type=int, default=1000)
    parser.add_argument("--n-grid", type=int, default=1000)
    parser.add_argument("--n-quad", type=int, default=31)
    args = parser.parse_args()
    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "cpu"
    print(f"Device: {dev}, n_hkl={args.n_hkl}, grid={args.n_grid}, quad={args.n_quad}")

    beta = 1.3
    alphas = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    obs_list = [10, 50, 100, 200]

    # ── Summary collectors ──
    summary_gauss = {}   # (alpha, obs) → median ratio
    summary_poisson = {}
    summary_median = {}

    for alpha in alphas:
        for obs in obs_list:
            print(f"\n{'='*72}")
            print(f"  alpha={alpha}, beta={beta}, obs/hkl={obs}, n_hkl={args.n_hkl}")
            print(f"{'='*72}")

            data = generate(
                n_hkl=args.n_hkl, obs_per_hkl=obs,
                alpha=alpha, beta=beta, device=dev,
            )
            true = data["I0_true"]

            # ── Median baseline ──
            ratio_med = (data["I0_median"] / true).cpu()
            med_bias = float(ratio_med.median())
            summary_median[(alpha, obs)] = med_bias

            # ── Gaussian grid scan ──
            t0 = time.perf_counter()
            I0_g = grid_scan_gauss(data, alpha, beta,
                                   n_grid=args.n_grid, n_quad=args.n_quad)
            if dev != "cpu":
                torch.cuda.synchronize()
            tg = time.perf_counter() - t0

            rows_g = bin_stats(I0_g, true)
            summary_gauss[(alpha, obs)] = rows_g[-1]["median"]

            # ── Poisson grid scan ──
            t0 = time.perf_counter()
            I0_p = grid_scan_poisson(data, alpha, beta,
                                     n_grid=args.n_grid, n_quad=args.n_quad)
            if dev != "cpu":
                torch.cuda.synchronize()
            tp = time.perf_counter() - t0

            rows_p = bin_stats(I0_p, true)
            summary_poisson[(alpha, obs)] = rows_p[-1]["median"]

            # ── Print per-bin results ──
            print(f"\n  Median estimator: overall med(ratio)={med_bias:.4f}")
            print(f"\n  Gaussian NLL grid scan ({tg:.1f}s):")
            print_table(rows_g)
            print(f"\n  Poisson NLL grid scan ({tp:.1f}s):")
            print_table(rows_p)

    # ══════════════════════════════════════════════════════
    # Summary tables
    # ══════════════════════════════════════════════════════
    print(f"\n\n{'='*72}")
    print("SUMMARY: median(I0_MLE / I0_true)  [1.0 = unbiased]")
    print(f"{'='*72}")

    # Header
    hdr = f"{'alpha':>6} |"
    for obs in obs_list:
        hdr += f" {'n='+str(obs):>18} |"
    sep = "-" * len(hdr)

    print(f"\n  Median estimator:")
    print(f"  {hdr}")
    print(f"  {sep}")
    for alpha in alphas:
        row = f"  {alpha:>6.1f} |"
        for obs in obs_list:
            v = summary_median[(alpha, obs)]
            row += f" {v:>18.4f} |"
        print(row)

    print(f"\n  Gaussian NLL grid-scan MLE:")
    print(f"  {hdr}")
    print(f"  {sep}")
    for alpha in alphas:
        row = f"  {alpha:>6.1f} |"
        for obs in obs_list:
            v = summary_gauss[(alpha, obs)]
            row += f" {v:>18.4f} |"
        print(row)

    print(f"\n  Poisson NLL grid-scan MLE:")
    print(f"  {hdr}")
    print(f"  {sep}")
    for alpha in alphas:
        row = f"  {alpha:>6.1f} |"
        for obs in obs_list:
            v = summary_poisson[(alpha, obs)]
            row += f" {v:>18.4f} |"
        print(row)

    print(f"\n\nDone.")


if __name__ == "__main__":
    main()
