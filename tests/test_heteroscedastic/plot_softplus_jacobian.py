#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Softplus quadrature with analytical Jacobian — no midpoint rule.

Change of variables: t → signal = softplus(t), ε = log(signal/I0)

The composite Jacobian dε/dt = sigmoid(t)/softplus(t) is bounded:
  - t → -∞: sigmoid(t)/softplus(t) → 1  (both ≈ exp(t))
  - t → +∞: sigmoid(t)/softplus(t) → 1/t

This cancels the 1/signal singularity analytically.

Integration: ∫ GG(ε) · Poi(n;λ) dε = Σ_k GG(ε_k) · Poi(n;λ_k) · (dε/dt)_k · Δt
"""

from __future__ import annotations
import math, os, sys, torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from plot_scatter_bias import (
    generate_realistic, sample_gengauss, estimate_median, estimate_irls, _log_gg_norm,
)
from plot_midpoint_comparison import estimate_gg_lbfgs
from plot_convergence_sweep import estimate_poisson_midpoint
from verify_quadrature import _log_gg_density, _log_poi, mc_log_marginal


def _softplus_inv(y):
    if isinstance(y, torch.Tensor):
        return torch.where(y > 20, y, torch.log(torch.expm1(y.clamp(min=1e-8))))
    return y if y > 20 else math.log(math.expm1(max(y, 1e-8)))


def estimate_poisson_spjac(data, beta, N=51, n_sigma=5, max_iter=50,
                            signal_min=0.01):
    """Poisson MLE with softplus mapping + analytical Jacobian."""
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

    # Per-obs bounds in t-space
    signal_max = (n_ph - bg + n_sigma * sigma).clamp(min=1.0)
    signal_lo = (signal_max * 0.01).clamp(min=signal_min)
    t_hi = _softplus_inv(signal_max)
    t_lo = _softplus_inv(signal_lo)

    # Uniform grid in t-space (both bounds per-obs)
    frac = torch.linspace(0, 1, N, device=dev)           # (K,)
    dt = (t_hi - t_lo) / (N - 1)                         # (N_obs,)
    t = t_lo[:, None] + (t_hi - t_lo)[:, None] * frac[None, :]  # (N_obs, K)

    signal = F.softplus(t)                     # (N_obs, K)
    lam = signal + bg                          # (N_obs, K)

    # Analytical Jacobian: dε/dt = sigmoid(t) / softplus(t)
    # In log-space: log(sigmoid(t)) - log(softplus(t))
    # = log(sigmoid(t)) - log(signal)
    # Use log_sigmoid for numerical stability
    log_deps_dt = F.logsigmoid(t) - torch.log(signal)  # (N_obs, K)

    # Precompute I0-independent terms
    log_poi = n_ph[:, None] * torch.log(lam) - lam - log_nf[:, None]
    log_dt = torch.log(dt)                     # (N_obs,)
    log_poi_jac = log_poi + log_deps_dt + log_dt[:, None]  # (N_obs, K)

    log_I0 = torch.log(estimate_median(data).clamp(min=0.1)).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([log_I0], lr=1.0, max_iter=max_iter,
                             line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        I0_po = torch.exp(log_I0).repeat_interleave(obs_per_hkl)
        eps = torch.log(signal / I0_po[:, None])
        log_gg = -(eps.abs() / alpha_po[:, None]).pow(beta) + log_gg_norm
        log_integrand = log_gg + log_poi_jac
        log_p = torch.logsumexp(log_integrand, dim=1)
        nll = torch.zeros(n_hkl, device=dev)
        nll.scatter_add_(0, hkl_id, -log_p)
        loss = nll.sum()
        loss.backward()
        return loss

    opt.step(closure)
    return torch.exp(log_I0.detach())


def generate_weak(*, n_hkl, beta, alpha=1.2, I0_lo=0.1, I0_hi=3.0,
                  n_obs_per=100, bg=10., seed=42, device="cpu"):
    gen = torch.Generator().manual_seed(seed)
    log_I0 = torch.linspace(math.log(I0_lo), math.log(I0_hi), n_hkl)
    I0_true = torch.exp(log_I0)
    alpha_true = torch.full((n_hkl,), alpha)
    obs_per_hkl = torch.full((n_hkl,), n_obs_per, dtype=torch.long)
    N = n_hkl * n_obs_per
    hkl_id = torch.arange(n_hkl).repeat_interleave(n_obs_per)
    I0_po = I0_true.repeat_interleave(n_obs_per)
    eps = sample_gengauss(alpha, beta, N, generator=gen)
    I_signal = (I0_po * torch.exp(eps)).clamp(min=0)
    I_obs = torch.poisson(I_signal + bg, generator=gen) - bg
    return dict(
        I0_true=I0_true.to(device), alpha_true=alpha_true.to(device),
        obs_per_hkl=obs_per_hkl.to(device), I_obs=I_obs.to(device),
        hkl_id=hkl_id.to(device), bg=bg, N=N, n_hkl=n_hkl,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    params_path = os.path.join(os.path.dirname(__file__), "real_hkl_params.pt")
    real_params = torch.load(params_path, weights_only=True)
    beta = 1.3

    def _r(est, lt):
        return np.corrcoef(lt, np.log(np.clip(est, 1e-6, None)))[0, 1]

    # ── Part 1: Integral accuracy ──
    print("=== Integral accuracy (signal_min=0.01) ===")
    I0 = 3.0; alpha = 1.2; bg = 10.0

    print(f"{'I_obs':>6} {'n_ph':>5} | {'MC':>8} | {'spj31':>8} {'spj51':>8} {'mid31':>8} {'mid101':>8}")
    print("-" * 68)

    for I_obs in [-8, -5, 0, 1, 3, 10, 50, 100]:
        n_ph_v = max(I_obs + bg, 0.0)
        mc = mc_log_marginal(n_ph_v, I0, alpha, beta, bg, n_mc=2_000_000)

        # Softplus+Jacobian
        for N_q, label in [(31, "spj31"), (51, "spj51")]:
            n_ph_t = torch.tensor([n_ph_v])
            sigma_v = max(math.sqrt(max(n_ph_v, 1)), 1.0)
            sig_max = max(n_ph_v - bg + 5 * sigma_v, 1.0)
            t_hi_v = _softplus_inv(sig_max)
            t_lo_v = _softplus_inv(0.01)
            dt_v = (t_hi_v - t_lo_v) / (N_q - 1)
            t_v = torch.linspace(t_lo_v, t_hi_v, N_q)
            sig_v = F.softplus(t_v)
            lam_v = sig_v + bg
            log_deps = F.logsigmoid(t_v) - torch.log(sig_v)
            log_poi = _log_poi(n_ph_v, lam_v)
            eps_v = torch.log(sig_v / I0)
            log_gg = _log_gg_density(eps_v, alpha, beta)
            log_int = log_gg + log_poi + log_deps + math.log(dt_v)
            val = float(torch.logsumexp(log_int, dim=0))
            if label == "spj31":
                spj31 = val
            else:
                spj51 = val

        # Midpoint baselines
        from plot_integration_boxes import midpoint_log_marginal
        mid31 = midpoint_log_marginal(n_ph_v, I0, alpha, beta, bg, N=31)[0]
        mid101 = midpoint_log_marginal(n_ph_v, I0, alpha, beta, bg, N=101)[0]

        print(f"{I_obs:>6} {n_ph_v:>5.0f} | {mc:>8.3f} | {spj31:>8.3f} {spj51:>8.3f} {mid31:>8.3f} {mid101:>8.3f}")

    # ── Part 2: Estimation across regimes ──
    for regime, data_fn, label in [
        ("Realistic (I0 med=9.2)", lambda: generate_realistic(
            real_params=real_params, n_hkl=5000, beta=beta, device=device), "realistic"),
        ("Weak (I0 ∈ [0.1, 3])", lambda: generate_weak(
            n_hkl=2000, beta=beta, device=device), "weak"),
        ("Strong (I0 ∈ [10, 1000])", lambda: generate_weak(
            n_hkl=2000, beta=beta, I0_lo=10., I0_hi=1000., device=device), "strong"),
    ]:
        print(f"\n=== {regime} ===")
        data = data_fn()
        true = data["I0_true"].cpu().numpy()
        lt = np.log(true)

        r_gg = _r(estimate_gg_lbfgs(data, beta, max_iter=200).cpu().numpy(), lt)
        print(f"  GG LBFGS:        r={r_gg:.4f}")

        for N in [31, 51, 101, 201]:
            est_mid, _ = estimate_poisson_midpoint(data, beta, N=N, max_iter=50)
            r_mid = _r(est_mid.cpu().numpy(), lt)
            est_spj = estimate_poisson_spjac(data, beta, N=N, max_iter=50)
            r_spj = _r(est_spj.cpu().numpy(), lt)
            print(f"  N={N:>3}  midpoint: r={r_mid:.4f}   spjac: r={r_spj:.4f}   Δ={r_spj-r_mid:+.4f}")


if __name__ == "__main__":
    main()
