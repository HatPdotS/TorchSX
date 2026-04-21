#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Extended sweep: N_quad up to 1001."""

from __future__ import annotations
import math, os, sys, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from plot_scatter_bias import (
    generate_realistic, estimate_median, estimate_irls, _log_gg_norm,
)
from plot_midpoint_comparison import estimate_gg_lbfgs
from plot_convergence_sweep import estimate_poisson_midpoint


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    params_path = os.path.join(os.path.dirname(__file__), "real_hkl_params.pt")
    real_params = torch.load(params_path, weights_only=True)

    beta = 1.3
    n_hkl = 5000

    data = generate_realistic(real_params=real_params, n_hkl=n_hkl,
                              beta=beta, device=device)
    true = data["I0_true"].cpu().numpy()
    lt = np.log(true)

    def _r(est):
        return np.corrcoef(lt, np.log(np.clip(est, 1e-6, None)))[0, 1]

    # Baselines
    r_med = _r(estimate_median(data).cpu().numpy())
    r_irls = _r(estimate_irls(data, beta).cpu().numpy())
    r_gg = _r(estimate_gg_lbfgs(data, beta, max_iter=200).cpu().numpy())
    print(f"  Median:        r={r_med:.4f}")
    print(f"  IRLS:          r={r_irls:.4f}")
    print(f"  GG LBFGS(200): r={r_gg:.4f}")

    N_vals = [31, 51, 101, 201, 501]
    max_iter = 50
    results = []

    for N in N_vals:
        print(f"  Poisson midpoint N={N} ...", end=" ", flush=True)
        est, n_ev = estimate_poisson_midpoint(data, beta, N=N, max_iter=max_iter)
        r = _r(est.cpu().numpy())
        results.append(r)
        print(f"r={r:.4f} ({n_ev} evals)")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(N_vals, results, "o-", color="darkorchid", lw=2, markersize=8,
            label="Poisson midpoint")
    ax.axhline(r_gg, color="steelblue", ls="--", lw=2, label=f"GG LBFGS ({r_gg:.4f})")
    ax.axhline(r_irls, color="seagreen", ls=":", lw=2, label=f"IRLS ({r_irls:.4f})")
    ax.axhline(r_med, color="tomato", ls=":", lw=1, label=f"Median ({r_med:.4f})")

    for N, r in zip(N_vals, results):
        ax.annotate(f"{r:.4f}", (N, r), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9)

    ax.set_xlabel("Quadrature points N")
    ax.set_ylabel("r(log I₀_true, log I₀_est)")
    ax.set_xscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Poisson midpoint vs N_quad  (β={beta}, {n_hkl} HKLs, max_iter={max_iter})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "convergence_sweep2.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
