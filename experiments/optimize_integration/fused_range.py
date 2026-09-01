"""Is the fused kernel's closed-form shift losing dynamic range?

It replaces the online-logsumexp maximum with an analytic UPPER bound on the
integrand.  Anything more than ~87 nats below that bound underflows float32 and
gets floored, so ``log_p`` is reported far too HIGH exactly where the true value
is far too low.  The per-column kernel cannot do this: its running max always
finds the actual maximum.

If that is the mechanism, the error should be zero for observations whose
integrand peaks near the bound and enormous for the rest -- and it should grow
with the count, since a bright observation's Poisson peak is far from its
GenGauss centre.
"""
from __future__ import annotations

import torch

from torchsx.merging.quadrature._fixed_log_grid_triton import (
    marginalize_signal_triton,
)
from torchsx.merging.quadrature._fused_posterior_triton import (
    fused_posterior_logp,
)
from fused_check import _inputs, _percol


def _f64(base, n_ph, bg, la, lb, hkl, v, log_S, log_w):
    """float64 eager reference.  The per-column kernel cannot arbitrate here:
    at n = 2e6 it evaluates the same near-cancelling float32 expression, so a
    disagreement between the two says only that they differ, not which is
    wrong."""
    import math
    f64 = torch.float64
    x, S = log_S.to(f64), log_S.to(f64).exp()
    beta, alpha = lb.to(f64).exp(), la.to(f64).exp()
    lam = (S[None, :] + bg[:, None].to(f64)).clamp(min=1e-300)
    log_f = (n_ph[:, None].to(f64) * lam.log() - lam
             - torch.lgamma(n_ph[:, None].to(f64) + 1))
    lgn = (beta.log() - math.log(2.0) - torch.lgamma(1.0 / beta)
           - la.to(f64))
    cols = []
    for c in range(v.shape[1]):
        mu = v[hkl, c].to(f64) + base.to(f64)
        z = (x[None, :] - mu[:, None]).abs() / alpha[:, None]
        lg = -z.clamp(min=1e-300).pow(beta[:, None]) + lgn[:, None]
        cols.append(torch.logsumexp(lg + log_f + log_w[None, :].to(f64), dim=1))
    return torch.stack(cols, dim=1)


def main() -> None:
    dev, K, C, M = "cuda", 2001, 21, 2048
    for nmax, w in ((200.0, 1.8), (2e4, 1.8), (2e6, 1.8), (2e6, 0.6)):
        d = _inputs(M, K, C, 1024, dev)
        d["n_ph"] = (torch.rand(M, device=dev) * nmax).round()
        d["la"] = torch.full((M,), float(torch.tensor(w).log()), device=dev)
        lnf = torch.lgamma(d["n_ph"] + 1)
        with torch.no_grad():
            ref = _f64(d["base"], d["n_ph"], d["bg"], d["la"], d["lb"],
                       d["hkl"], d["v"], d["log_S"], d["log_w"])
            impl = {
                "per-column": _percol(d["base"], d["n_ph"], d["bg"], d["la"],
                                      d["lb"], d["hkl"], d["v"], d["log_S"],
                                      d["log_w"], lnf),
                "fused": fused_posterior_logp(
                    d["base"], d["n_ph"], d["bg"], d["la"], d["lb"], d["hkl"],
                    d["v"], d["log_S"], d["log_w"], log_nfact=lnf),
            }
        depth = ref.max(dim=1, keepdim=True).values - ref   # nats below peak
        print(f"n_max {nmax:8.0e}  alpha {w:4.1f}   (vs float64 reference)")
        for nm, got in impl.items():
            err = got.double() - ref
            print(f"    {nm:<11} wrong by >1 nat: "
                  f"{100*float((err.abs()>1).float().mean()):5.1f}%   "
                  f"max over-est {float(err.max()):8.1f}   ", end="")
            m = depth < 20
            print(f"median |err| at depth<20: "
                  f"{float(err[m].abs().median()):9.2e}")


if __name__ == "__main__":
    main()
