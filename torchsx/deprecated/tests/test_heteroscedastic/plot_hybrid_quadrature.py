#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Hybrid quadrature: 31 linear points + 21 log-spaced points near bg.

The linear points cover the Poisson peak well for medium/strong obs.
The extra log-spaced points fill the near-bg region for weak obs.
All points are merged, sorted, and get midpoint-rule widths.
"""

from __future__ import annotations
import math, os, sys, torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from plot_scatter_bias import (
    generate_realistic, estimate_median, estimate_irls, _log_gg_norm,
)
from plot_midpoint_comparison import estimate_gg_lbfgs
from plot_convergence_sweep import estimate_poisson_midpoint
from verify_quadrature import _log_gg_density, _log_poi, mc_log_marginal


def estimate_poisson_hybrid(data, beta, N_lin=31, N_log=21, n_sigma=5, max_iter=50):
    """Poisson MLE with hybrid linear + log-spaced quadrature grid."""
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

    # Linear grid: ±5σ around Poisson peak
    k_lin = torch.linspace(-n_sigma, n_sigma, N_lin, device=dev)
    lam_lin = n_ph[:, None] + sigma[:, None] * k_lin[None, :]
    lam_lin = lam_lin.clamp(min=bg + 0.1)

    # Log-spaced grid: signal from 0.01 to ~3 (near-bg region)
    log_signal = torch.linspace(math.log(0.01), math.log(3.0), N_log, device=dev)
    lam_log = torch.exp(log_signal)[None, :] + bg  # (1, N_log) broadcast

    # Merge and sort
    lam_all = torch.cat([lam_lin, lam_log.expand(len(n_ph), -1)], dim=1)  # (N_obs, N_lin+N_log)
    lam_all, _ = lam_all.sort(dim=1)

    # Deduplicate: remove points that are too close (within 0.01)
    # Just keep all — midpoint rule handles duplicates gracefully
    K = lam_all.shape[1]

    # Midpoint widths
    widths = torch.zeros_like(lam_all)
    widths[:, 1:-1] = (lam_all[:, 2:] - lam_all[:, :-2]) / 2
    widths[:, 0] = (lam_all[:, 1] - lam_all[:, 0]) / 2
    widths[:, -1] = (lam_all[:, -1] - lam_all[:, -2]) / 2

    signal = (lam_all - bg).clamp(min=1e-8)

    log_poi = n_ph[:, None] * torch.log(lam_all) - lam_all - log_nf[:, None]
    log_jac = -torch.log(signal)
    log_w = torch.log(widths.clamp(min=1e-30))
    log_poi_jac_w = log_poi + log_jac + log_w

    log_I0 = torch.log(estimate_median(data).clamp(min=0.1)).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([log_I0], lr=1.0, max_iter=max_iter,
                             line_search_fn="strong_wolfe")

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
        return loss

    opt.step(closure)
    return torch.exp(log_I0.detach())


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
    print(f"  Median:           r={r_med:.4f}")
    print(f"  IRLS:             r={r_irls:.4f}")
    print(f"  GG LBFGS:         r={r_gg:.4f}")

    # Clamp baselines
    for N in [31, 52]:
        est, _ = estimate_poisson_midpoint(data, beta, N=N, max_iter=50)
        print(f"  Clamp N={N:>3}:       r={_r(est.cpu().numpy()):.4f}")

    # Hybrid: 31 linear + varying log points
    for N_log in [11, 21, 31]:
        est = estimate_poisson_hybrid(data, beta, N_lin=31, N_log=N_log, max_iter=50)
        r = _r(est.cpu().numpy())
        print(f"  Hybrid 31+{N_log:<3}:     r={r:.4f}  (total={31+N_log})")

    # Also try clamp at equivalent total points
    for N in [42, 52, 62]:
        est, _ = estimate_poisson_midpoint(data, beta, N=N, max_iter=50)
        print(f"  Clamp N={N:>3}:       r={_r(est.cpu().numpy()):.4f}")

    # Clamp N=101 and N=201 for reference
    for N in [101, 201]:
        est, _ = estimate_poisson_midpoint(data, beta, N=N, max_iter=50)
        print(f"  Clamp N={N:>3}:       r={_r(est.cpu().numpy()):.4f}")


if __name__ == "__main__":
    main()
