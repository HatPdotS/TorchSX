#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Compare arithmetic mean vs all other estimators."""

from __future__ import annotations
import math, os, sys, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from plot_scatter_bias import (
    generate_realistic, sample_gengauss, estimate_median, estimate_irls,
    _log_gg_norm, _sorted_obs,
)
from plot_midpoint_comparison import estimate_gg_lbfgs
from plot_softplus_jacobian import estimate_poisson_spjac, generate_weak


def estimate_mean(data):
    """Simple arithmetic mean per HKL."""
    I_sorted, offsets = _sorted_obs(data)
    n_hkl = data["n_hkl"]
    dev = data["I_obs"].device
    I0 = torch.zeros(n_hkl, device=dev)
    for h in range(n_hkl):
        I0[h] = I_sorted[offsets[h]:offsets[h+1]].mean()
    return I0.clamp(min=0.1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    params_path = os.path.join(os.path.dirname(__file__), "real_hkl_params.pt")
    real_params = torch.load(params_path, weights_only=True)
    beta = 1.3

    def _r(est, lt):
        return np.corrcoef(lt, np.log(np.clip(est, 1e-6, None)))[0, 1]

    for regime, data_fn in [
        ("Realistic (I0 med=9.2)", lambda: generate_realistic(
            real_params=real_params, n_hkl=5000, beta=beta, device=device)),
        ("Weak (I0 ∈ [0.1, 3])", lambda: generate_weak(
            n_hkl=2000, beta=beta, device=device)),
        ("Strong (I0 ∈ [10, 1000])", lambda: generate_weak(
            n_hkl=2000, beta=beta, I0_lo=10., I0_hi=1000., device=device)),
    ]:
        print(f"\n=== {regime} ===")
        data = data_fn()
        true = data["I0_true"].cpu().numpy()
        lt = np.log(true)

        mean = estimate_mean(data).cpu().numpy()
        med = estimate_median(data).cpu().numpy()
        irls = estimate_irls(data, beta).cpu().numpy()
        gg = estimate_gg_lbfgs(data, beta, max_iter=200).cpu().numpy()
        spj = estimate_poisson_spjac(data, beta, N=51, max_iter=50).cpu().numpy()

        # Bias check: mean ratio
        ratio_mean = np.median(mean / true)
        ratio_med = np.median(med / true)

        print(f"  Mean:       r={_r(mean, lt):.4f}   median(est/true)={ratio_mean:.3f}")
        print(f"  Median:     r={_r(med, lt):.4f}   median(est/true)={ratio_med:.3f}")
        print(f"  IRLS:       r={_r(irls, lt):.4f}")
        print(f"  GG LBFGS:   r={_r(gg, lt):.4f}")
        print(f"  SPJac N=51: r={_r(spj, lt):.4f}")


if __name__ == "__main__":
    main()
