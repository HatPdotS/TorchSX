"""Can the negative-binomial form be evaluated in float32 on this grid?

`bg_forms.py` ranked the moment-matched negative binomial (mean lam, variance
lam + sigma_b^2) as the closest cheap approximation to the exact marginal --
0.10 nats median against the deviance form's 0.13, and 2.4 nats worst case
against 5.2.  The obstacle is conditioning, not cost.

Written as `log NB - log Poisson`, with r = lam^2/sigma_b^2:

    Delta = (n+r) log((n+r)/(lam+r)) - (n - lam) - 0.5 log(1 + n/r)

both terms are O(1e9) on the far grid and cancel to O(1).  Substituting
p = n sigma_b^2/lam^2, q = sigma_b^2/lam and g(x) = log(1+x) - x removes the
cancellation exactly:

    Delta = (lam^2/sigma_b^2) [g(p) - g(q)] + n [log(1+p) - log(1+q)] - 0.5 log(1+p)

because (lam^2/sigma_b^2)(p - q) IS (n - lam).  What is left is g, which still
cancels for small x, so it needs a series there.  This measures both forms
against a float64 reference over the whole grid the kernel actually walks.
"""
from __future__ import annotations

import math

import torch

DT64, DT32 = torch.float64, torch.float32


def nb_direct(n, lam, s2, dt):
    n, lam, s2 = n.to(dt), lam.to(dt), s2.to(dt)
    r = lam * lam / s2
    return ((n + r) * torch.log((n + r) / (lam + r)) - (n - lam)
            - 0.5 * torch.log1p(n / r))


def g_series(x):
    """log(1+x) - x by Horner, for |x| <= 0.5."""
    c = [-0.5, 1 / 3, -0.25, 0.2, -1 / 6, 1 / 7, -0.125, 1 / 9]
    acc = torch.full_like(x, c[-1])
    for a in reversed(c[:-1]):
        acc = acc * x + a
    return x * x * acc


def g(x, cut=0.5):
    return torch.where(x.abs() < cut, g_series(x), torch.log1p(x) - x)


def nb_stable(n, lam, s2, dt):
    n, lam, s2 = n.to(dt), lam.to(dt), s2.to(dt)
    inv_lam = 1.0 / lam
    q = s2 * inv_lam
    p = n * s2 * inv_lam * inv_lam
    return ((lam * lam / s2) * (g(p) - g(q))
            + n * (torch.log1p(p) - torch.log1p(q))
            - 0.5 * torch.log1p(p))


def main():
    # the real grid: log S in [-34.5, 34.5], lam = S + bg
    K = 1355
    log_S = torch.linspace(-34.5, 34.5, K, dtype=DT64)
    S = log_S.exp()
    cases = [(136.0, 116.0), (8.0, 12.0), (2700.0, 300.0), (0.0, 116.0),
             (270000.0, 116.0)]
    print(f"{'n':>9} {'bg':>7} | {'direct f32':>26} | {'stable f32':>26}")
    print(f"{'':>9} {'':>7} | {'max err':>12} {'max Delta':>13} | "
          f"{'max err':>12} {'max Delta':>13}")
    for n_, bg_ in cases:
        n = torch.full((K,), n_, dtype=DT64)
        bg = torch.full((K,), bg_, dtype=DT64)
        lam = (S + bg).clamp(min=1e-6)
        s2 = 1.1641 * bg
        ref = nb_stable(n, lam, s2, DT64)
        # sanity: the stable rewrite must equal the direct formula in float64
        chk = float((ref - nb_direct(n, lam, s2, DT64)).abs().max())
        d32 = nb_direct(n, lam, s2, DT32).to(DT64)
        s32 = nb_stable(n, lam, s2, DT32).to(DT64)
        print(f"{n_:>9.0f} {bg_:>7.0f} | {float((d32-ref).abs().max()):>12.3e} "
              f"{float(d32.max()):>13.3e} | {float((s32-ref).abs().max()):>12.3e} "
              f"{float(s32.max()):>13.3e}   [f64 rewrite check {chk:.2e}]")

    print("\nrestricted to grid points within 50 nats of the Poisson peak "
          "(the only ones that carry mass)")
    print(f"{'n':>9} {'bg':>7} | {'direct f32':>12} {'stable f32':>12} "
          f"{'max |Delta|':>12}")
    for n_, bg_ in cases:
        n = torch.full((K,), n_, dtype=DT64)
        bg = torch.full((K,), bg_, dtype=DT64)
        lam = (S + bg).clamp(min=1e-6)
        s2 = 1.1641 * bg
        nn = n.clamp(min=1e-12)
        half_D = nn * torch.log(nn / lam) - (nn - lam)
        live = half_D < 50.0
        if not bool(live.any()):
            print(f"{n_:>9.0f} {bg_:>7.0f} |  (no grid point within 50 nats "
                  f"of the peak: lam >= bg = {bg_:.0f} already costs "
                  f"{float(half_D.min()):.0f})")
            continue
        ref = nb_stable(n, lam, s2, DT64)
        d32 = nb_direct(n, lam, s2, DT32).to(DT64)
        s32 = nb_stable(n, lam, s2, DT32).to(DT64)
        print(f"{n_:>9.0f} {bg_:>7.0f} | "
              f"{float((d32-ref)[live].abs().max()):>12.3e} "
              f"{float((s32-ref)[live].abs().max()):>12.3e} "
              f"{float(ref[live].abs().max()):>12.3f}")


if __name__ == "__main__":
    main()
