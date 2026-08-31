#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Effect of quadrature resolution on the NLL landscape.

Same data, same (I0=10, alpha=1.2, beta=1.3), but vary n_quad from 10 to 10000.
"""

from __future__ import annotations

import math
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sample_gengauss(alpha, beta, size, gen=None):
    g = torch._standard_gamma(torch.full((size,), 1.0 / beta), generator=gen)
    signs = 2.0 * torch.bernoulli(torch.full((size,), 0.5), generator=gen) - 1.0
    return signs * alpha * g.pow(1.0 / beta)


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


def compute_nll_curve(I_obs, alpha, beta, bg, I0_grid, n_quad, device="cpu"):
    """Compute Poisson × GenGauss marginal NLL with specified n_quad. GPU-friendly."""
    n_grid = len(I0_grid)
    n_obs = len(I_obs)

    R = _quad_range(beta)
    eps = torch.linspace(-R, R, n_quad, device=device)
    d_eps = float(eps[1] - eps[0])
    log_w = -(eps.abs() ** beta) + math.log(beta) - math.log(2) \
            - float(torch.lgamma(torch.tensor(1.0 / beta))) + math.log(d_eps)
    s = torch.exp(alpha * eps)  # (K,)

    I_obs_d = I_obs.to(device)
    I0_grid_d = I0_grid.to(device)
    n_ph = (I_obs_d + bg).clamp(min=0)       # (J,)
    log_nf = torch.lgamma(n_ph + 1)          # (J,)

    # Vectorise over grid: (G, K, J) — batch in chunks if needed
    chunk = max(1, min(n_grid, 50_000_000 // (n_quad * n_obs)))
    nll = torch.zeros(n_grid, device=device)

    for g0 in range(0, n_grid, chunk):
        g1 = min(g0 + chunk, n_grid)
        I0_chunk = I0_grid_d[g0:g1]                          # (C,)
        # lam[c, k] = I0[c] * s[k] + bg
        lam = (I0_chunk[:, None] * s[None, :] + bg).clamp(min=1e-6)  # (C, K)
        # log_poi[c, k, j]
        log_poi = n_ph[None, None, :] * torch.log(lam[:, :, None]) \
                  - lam[:, :, None] - log_nf[None, None, :]  # (C, K, J)
        log_joint = log_poi + log_w[None, :, None]            # (C, K, J)
        log_p = torch.logsumexp(log_joint, dim=1)             # (C, J)
        nll[g0:g1] = -log_p.sum(dim=1)                        # (C,)

    return nll.cpu()


def main():
    alpha = 1.2
    beta = 1.3
    bg = 10.0
    n_obs = 100
    quads = [10, 100, 1000, 10000]

    cases = [
        (10.0, 42,  "I₀=10, seed A"),
        (10.0, 777, "I₀=10, seed B"),
        (3.0,  42,  "I₀=3, seed A"),
        (3.0,  99,  "I₀=3, seed B"),
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    I0_grid = torch.logspace(-0.5, 3.5, 500)
    I0_np = I0_grid.numpy()

    fig, axes = plt.subplots(len(cases), len(quads), figsize=(20, 16))

    for row, (I0_true, seed, label) in enumerate(cases):
        gen = torch.Generator().manual_seed(seed)
        eps = sample_gengauss(alpha, beta, n_obs, gen=gen)
        signal = (I0_true * torch.exp(eps)).clamp(min=0)
        I_obs = torch.poisson(signal + bg, generator=gen) - bg
        med = float(I_obs.median())
        n_neg = int((I_obs <= 0).sum())

        print(f"{label}: range=[{I_obs.min():.0f}, {I_obs.max():.0f}], "
              f"neg={n_neg}, median={med:.1f}")

        for col, nq in enumerate(quads):
            print(f"  n_quad={nq} ...", end=" ", flush=True)
            nll = compute_nll_curve(I_obs, alpha, beta, bg, I0_grid, nq, device=device)
            mle = float(I0_grid[nll.argmin()])
            nll_np = nll.numpy()
            nll_rel = nll_np - nll_np.min()

            ax = axes[row, col]
            ax.plot(I0_np, nll_rel, "k-", lw=1.5)
            ax.axvline(I0_true, color="red", ls="--", lw=2, label=f"True={I0_true}")
            ax.axvline(med, color="seagreen", ls="-", lw=1.5, label=f"Med={med:.1f}")
            ax.axvline(mle, color="darkorchid", ls="-", lw=1.5, label=f"MLE={mle:.1f}")
            ax.set_xscale("log")
            ax.set_xlabel("I₀")
            ax.set_ylabel("NLL − min")
            ax.set_ylim(0, min(float(nll_rel.max()), 150))
            ax.set_title(f"{label}\nn_quad={nq}, MLE={mle:.1f} (ratio={mle/I0_true:.2f})",
                        fontsize=9)
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.2, which="both")
            print(f"MLE={mle:.1f}", flush=True)

    fig.suptitle(
        f"Quadrature resolution effect on NLL  "
        f"(α={alpha}, β={beta}, bg={bg}, n_obs={n_obs})",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    out = "/das/work/units/LBR-FEL/p17490/Peter/torchsx/tests/test_heteroscedastic/nll_quadrature.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
