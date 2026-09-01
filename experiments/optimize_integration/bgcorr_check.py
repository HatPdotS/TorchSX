"""Triton BG_VAR path vs the eager reference, and the effect on ground truth."""
from __future__ import annotations
import math, torch
from torchsx.merging.likelihood import Poisson
from torchsx.merging.quadrature._fixed_log_grid_triton import marginalize_signal_triton

def main():
    dev = "cuda"; M, K = 4096, 2001
    g = torch.Generator(device=dev).manual_seed(0)
    log_S = torch.linspace(-34.5, 34.5, K, device=dev)
    log_w = torch.full((K,), math.log(69.0 / (K - 1)), device=dev)
    log_w[0] -= math.log(2.0); log_w[-1] -= math.log(2.0)
    lm = torch.randn(M, device=dev, generator=g)
    n = (torch.rand(M, device=dev, generator=g) * 300).round()
    bg = torch.rand(M, device=dev, generator=g) * 200 + 5
    la = torch.randn(M, device=dev, generator=g) * 0.2 + 0.25
    lb = torch.zeros(M, device=dev) + 0.07
    bvf = 149.0 / 128.0

    for tag, f in (("bg_var OFF", 0.0), ("bg_var ON", bvf)):
        lik = Poisson(bg_var_factor=f)
        S = log_S.exp()[None, :].expand(M, K)
        from torchsx.merging.scatter.gengauss import gengauss_log_prob_from_obs
        eager = torch.logsumexp(
            gengauss_log_prob_from_obs(log_S[None, :] - lm[:, None], la, lb)
            + lik.log_prob_given_signal(n, S, bg) + log_w[None, :], dim=1)
        trit = marginalize_signal_triton(
            lm, n, bg, la, lb, log_S, log_w,
            sig_b2=(f * bg) if f > 0 else None)
        d = (trit - eager).abs()
        print(f"{tag:<12} triton vs eager: max {float(d.max()):.3e}  "
              f"median {float(d.median()):.3e}")
    off = marginalize_signal_triton(lm, n, bg, la, lb, log_S, log_w)
    on = marginalize_signal_triton(lm, n, bg, la, lb, log_S, log_w, sig_b2=bvf * bg)
    print(f"\ncorrection size: median |log p on - off| = "
          f"{float((on - off).abs().median()):.4f} nats")
main()
