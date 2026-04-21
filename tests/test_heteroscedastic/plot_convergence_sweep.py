#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Sweep quadrature points and LBFGS iterations for Poisson midpoint.

Tests:
  N_quad  ∈ {11, 21, 31, 51, 101}
  max_iter ∈ {20, 50, 100, 200}
"""

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


def estimate_poisson_midpoint(data, beta, N=31, n_sigma=5, max_iter=50):
    """Poisson MLE with midpoint-rule integration in λ-space."""
    dev = data["I_obs"].device
    n_hkl = data["n_hkl"]
    bg = data["bg"]
    hkl_id = data["hkl_id"]
    obs_per_hkl = data["obs_per_hkl"]
    alpha_true = data["alpha_true"].to(dev)
    log_gg_norm = _log_gg_norm(beta)

    alpha_po = alpha_true.repeat_interleave(obs_per_hkl)
    n_ph = (data["I_obs"] + bg).clamp(min=0)
    log_nf = torch.lgamma(n_ph + 1)

    sigma = n_ph.clamp(min=1).sqrt()
    k = torch.linspace(-n_sigma, n_sigma, N, device=dev)
    lam = n_ph[:, None] + sigma[:, None] * k[None, :]
    lam = lam.clamp(min=bg + 0.1)

    widths = torch.zeros_like(lam)
    widths[:, 1:-1] = (lam[:, 2:] - lam[:, :-2]) / 2
    widths[:, 0] = (lam[:, 1] - lam[:, 0]) / 2
    widths[:, -1] = (lam[:, -1] - lam[:, -2]) / 2

    signal = lam - bg

    log_poi = n_ph[:, None] * torch.log(lam) - lam - log_nf[:, None]
    log_jac = -torch.log(signal)
    log_w = torch.log(widths.clamp(min=1e-30))
    log_poi_jac_w = log_poi + log_jac + log_w

    log_I0 = torch.log(estimate_median(data).clamp(min=0.1)).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([log_I0], lr=1.0, max_iter=max_iter,
                             line_search_fn="strong_wolfe")

    n_evals = [0]
    def closure():
        opt.zero_grad()
        I0_po = torch.exp(log_I0).repeat_interleave(obs_per_hkl)
        eps = torch.log(signal / I0_po[:, None])
        log_gg = -(eps.abs() / alpha_po[:, None]).pow(beta) + log_gg_norm
        log_integrand = log_gg + log_poi_jac_w
        log_p = torch.logsumexp(log_integrand, dim=1)
        nll = torch.zeros(n_hkl, device=dev)
        nll.scatter_add_(0, hkl_id, -log_p)
        loss = nll.sum()
        loss.backward()
        n_evals[0] += 1
        return loss

    opt.step(closure)
    return torch.exp(log_I0.detach()), n_evals[0]


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
    print("  Baselines ...", flush=True)
    r_med = _r(estimate_median(data).cpu().numpy())
    r_irls = _r(estimate_irls(data, beta).cpu().numpy())
    r_gg = _r(estimate_gg_lbfgs(data, beta, max_iter=200).cpu().numpy())
    print(f"    Median:         r={r_med:.4f}")
    print(f"    IRLS:           r={r_irls:.4f}")
    print(f"    GG LBFGS(200):  r={r_gg:.4f}")

    # Sweep
    N_vals = [11, 21, 31, 51, 101]
    iter_vals = [20, 50, 100, 200]

    results = np.zeros((len(N_vals), len(iter_vals)))

    print(f"\n{'N':>5} | " + " | ".join(f"iter={it:>3}" for it in iter_vals) + " | evals")
    print("-" * (8 + 12 * len(iter_vals) + 10))

    for i, N in enumerate(N_vals):
        row = []
        evals_row = []
        for j, mi in enumerate(iter_vals):
            print(f"  N={N}, iter={mi} ...", end=" ", flush=True)
            est, n_ev = estimate_poisson_midpoint(data, beta, N=N, max_iter=mi)
            r = _r(est.cpu().numpy())
            results[i, j] = r
            row.append(r)
            evals_row.append(n_ev)
            print(f"r={r:.4f} ({n_ev} evals)")

        print(f"{N:>5} | " + " | ".join(f"   {r:.4f}  " for r in row)
              + f" | {evals_row}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (0) Heatmap
    ax = axes[0]
    im = ax.imshow(results, aspect="auto", cmap="viridis",
                   vmin=min(results.min(), r_irls - 0.01),
                   vmax=max(results.max(), r_gg + 0.01))
    ax.set_xticks(range(len(iter_vals)))
    ax.set_xticklabels(iter_vals)
    ax.set_yticks(range(len(N_vals)))
    ax.set_yticklabels(N_vals)
    ax.set_xlabel("LBFGS max_iter")
    ax.set_ylabel("Quadrature points N")
    for i in range(len(N_vals)):
        for j in range(len(iter_vals)):
            ax.text(j, i, f"{results[i,j]:.4f}", ha="center", va="center",
                    fontsize=9, color="white" if results[i,j] < results.mean() else "black")
    ax.set_title("r(log I₀_true, log I₀_est)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Reference lines
    ax2 = axes[1]
    for i, N in enumerate(N_vals):
        ax2.plot(iter_vals, results[i, :], "o-", label=f"N={N}", markersize=5)
    ax2.axhline(r_gg, color="steelblue", ls="--", lw=2, label=f"GG LBFGS ({r_gg:.4f})")
    ax2.axhline(r_irls, color="seagreen", ls=":", lw=2, label=f"IRLS ({r_irls:.4f})")
    ax2.axhline(r_med, color="tomato", ls=":", lw=1, label=f"Median ({r_med:.4f})")
    ax2.set_xlabel("LBFGS max_iter")
    ax2.set_ylabel("r(log I₀_true, log I₀_est)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Convergence curves")

    fig.suptitle(f"Poisson midpoint: N_quad × max_iter sweep  (β={beta}, {n_hkl} HKLs)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "convergence_sweep.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
