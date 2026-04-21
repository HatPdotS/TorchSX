#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Clean implementation: sample ±5σ of Poisson, compute dε analytically per point.

For each obs with n photons:
  λ_k = n + k·σ  for k in linspace(-5, 5, 31), σ = √n, clamp λ > bg+δ
  ε_k = log((λ_k - bg) / I0)
  dε_k = Δλ / (λ_k - bg)  — the analytical ε-width per point
  integrand_k = GG(ε_k) · Poi(n; λ_k) · dε_k
"""

from __future__ import annotations
import math, os, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_scatter_bias import (
    generate_realistic, estimate_median, estimate_irls, _log_gg_norm,
)


def estimate_gg_lbfgs(data, beta, max_iter=50):
    """GenGauss NLL in log-space, positive obs only."""
    dev = data["I_obs"].device
    n_hkl = data["n_hkl"]
    hkl_id = data["hkl_id"]
    obs_per_hkl = data["obs_per_hkl"]
    alpha_true = data["alpha_true"].to(dev)

    pos = data["I_obs"] > 0
    log_I_pos = torch.log(data["I_obs"][pos])
    hkl_pos = hkl_id[pos]
    alpha_po_pos = alpha_true.repeat_interleave(obs_per_hkl)[pos]

    log_I0 = torch.log(estimate_median(data).clamp(min=0.1)).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([log_I0], lr=1.0, max_iter=max_iter,
                             line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        mu = log_I0[hkl_pos]
        r = (log_I_pos - mu).abs().clamp(min=1e-8)
        loss = (r / alpha_po_pos).pow(beta).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return torch.exp(log_I0.detach())


def estimate_poisson_v2(data, beta, n_sigma=5, n_per_obs=31, max_iter=50):
    """Poisson MLE with analytically computed per-point dε widths.

    Positive observations only. For each obs j with n_j photons:
      σ_j = √n_j
      λ_k = n_j + k·σ_j  for k ∈ linspace(-n_sigma, n_sigma, n_per_obs)
      clamp λ_k > bg + 0.1
      ε_k = log((λ_k - bg) / I0)
      Δλ = 2·n_sigma·σ_j / (n_per_obs - 1)
      dε_k = Δλ / (λ_k - bg)   ← per-point analytical width in ε-space
    """
    dev = data["I_obs"].device
    n_hkl = data["n_hkl"]
    bg = data["bg"]
    hkl_id = data["hkl_id"]
    obs_per_hkl = data["obs_per_hkl"]
    alpha_true = data["alpha_true"].to(dev)
    log_gg_norm_base = (math.log(beta) - math.log(2)
                        - float(torch.lgamma(torch.tensor(1.0 / beta))))

    # Positive observations only
    pos = data["I_obs"] > 0
    I_pos = data["I_obs"][pos]
    hkl_pos = hkl_id[pos]
    alpha_po = alpha_true.repeat_interleave(obs_per_hkl)[pos]
    N_pos = int(pos.sum())

    n_ph = I_pos + bg  # photon counts (all > bg since I_pos > 0)
    log_nf = torch.lgamma(n_ph + 1)
    sigma_poi = n_ph.sqrt()  # Poisson width per obs

    # Build per-obs λ grid: ±5σ around Poisson peak
    k = torch.linspace(-n_sigma, n_sigma, n_per_obs, device=dev)  # (K,)
    lam = n_ph[:, None] + sigma_poi[:, None] * k[None, :]  # (N, K)
    lam = lam.clamp(min=bg + 0.1)

    # Per-point signal and ε-width
    signal = lam - bg  # (N, K) — always > 0.1
    d_lam = (2 * n_sigma * sigma_poi / (n_per_obs - 1))  # (N,) uniform Δλ per obs
    d_eps = d_lam[:, None] / signal  # (N, K) — analytical ε-width per point

    # Precompute Poisson (I0-independent)
    log_poi = n_ph[:, None] * torch.log(lam) - lam - log_nf[:, None]  # (N, K)

    log_I0 = torch.log(estimate_median(data).clamp(min=0.1)).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([log_I0], lr=1.0, max_iter=max_iter,
                             line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        I0_po = torch.exp(log_I0)[hkl_pos]  # (N,)

        # ε at each quadrature point
        eps = torch.log(signal / I0_po[:, None])  # (N, K)

        # GenGauss density — full normalization including -log(alpha)
        log_gg = (-(eps.abs() / alpha_po[:, None]).pow(beta)
                  + log_gg_norm_base - torch.log(alpha_po[:, None]))

        # Integrand: GG(ε) × Poi(n; λ) × dε
        log_integrand = log_gg + log_poi + torch.log(d_eps)  # (N, K)

        # Per-obs marginal log-likelihood
        log_p = torch.logsumexp(log_integrand, dim=1)  # (N,)

        # Sum per HKL
        nll = torch.zeros(n_hkl, device=dev)
        nll.scatter_add_(0, hkl_pos, -log_p)
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

    print("  Median ...", flush=True)
    med = estimate_median(data).cpu().numpy()
    print("  IRLS ...", flush=True)
    irls = estimate_irls(data, beta).cpu().numpy()
    print("  GenGauss LBFGS (pos only) ...", flush=True)
    gg = estimate_gg_lbfgs(data, beta).cpu().numpy()
    print("  Poisson v2 ±5σ (pos only) ...", flush=True)
    poi = estimate_poisson_v2(data, beta).cpu().numpy()

    r_med, r_irls, r_gg, r_poi = _r(med), _r(irls), _r(gg), _r(poi)

    print(f"\n  Median:         r={r_med:.4f}")
    print(f"  IRLS:           r={r_irls:.4f}")
    print(f"  GenGauss LBFGS: r={r_gg:.4f}")
    print(f"  Poisson v2:     r={r_poi:.4f}")

    # ── Also verify: MC the integral for a few HKLs ──
    # First verify the quadrature is correct by checking a single obs
    print("\n  Verifying quadrature for single observations (I0=3):")
    from verify_quadrature import mc_log_marginal, obs_centered_log_marginal
    I0_test = 3.0
    for I_obs in [1, 3, 10, 50, 100]:
        n = I_obs + 10.0
        mc = mc_log_marginal(n, I0_test, 1.2, beta, 10.0, n_mc=1_000_000)
        # v2 quadrature
        sigma = math.sqrt(n)
        k_pts = torch.linspace(-5, 5, 31)
        lam_pts = torch.tensor(n) + sigma * k_pts
        lam_pts = lam_pts.clamp(min=10.1)
        sig_pts = lam_pts - 10.0
        d_lam_v = 10 * sigma / 30
        d_eps_v = d_lam_v / sig_pts
        eps_pts = torch.log(sig_pts / I0_test)
        log_gg = -(eps_pts.abs() / 1.2).pow(beta) + _log_gg_norm(beta) - math.log(1.2)
        log_poi = n * torch.log(lam_pts) - lam_pts - float(torch.lgamma(torch.tensor(n + 1)))
        log_int = log_gg + log_poi + torch.log(d_eps_v)
        v2 = float(torch.logsumexp(log_int, dim=0))
        print(f"    I_obs={I_obs:>4}: MC={mc:.3f}  v2={v2:.3f}  diff={v2-mc:+.3f}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    lo, hi = 0.3, 2e3

    ax = axes[0]
    ax.scatter(true, gg, s=2, alpha=0.15, c="steelblue", edgecolors="none", label="GenGauss")
    ax.scatter(true, poi, s=2, alpha=0.15, c="darkorchid", edgecolors="none", label="Poisson v2")
    ax.plot([lo, hi], [lo, hi], "k-", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("I₀ true"); ax.set_ylabel("I₀ estimated")
    ax.set_title(f"GenGauss (r={r_gg:.4f}) vs Poisson v2 (r={r_poi:.4f})")
    ax.legend(loc="lower right", fontsize=8, markerscale=3)
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[1]
    methods = ["Median", "IRLS", "GenGauss LBFGS", "Poisson v2 ±5σ"]
    rs = [r_med, r_irls, r_gg, r_poi]
    colors = ["tomato", "seagreen", "steelblue", "darkorchid"]
    bars = ax.barh(range(4), rs, color=colors, alpha=0.7)
    ax.set_yticks(range(4))
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel("r(log I₀_true, log I₀_est)")
    ax.set_xlim(min(rs) - 0.02, max(rs) + 0.01)
    for bar, r in zip(bars, rs):
        ax.text(r + 0.002, bar.get_y() + bar.get_height()/2, f"{r:.4f}",
                va="center", fontsize=9)

    fig.suptitle(f"Poisson v2 (±5σ, analytical dε) vs GenGauss — pos only, β={beta}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "posonly_v2.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
