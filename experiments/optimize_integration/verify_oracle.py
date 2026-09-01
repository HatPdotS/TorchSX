"""Certify the oracle before trusting it to judge anything else.

Four checks, in increasing order of what they can catch:

1. **FFT vs direct.**  The convolution route must reproduce a literal
   ``logsumexp`` over the same grid.  Catches an indexing or wrap-order slip.
2. **Closed-form derivatives vs autograd.**  ``G_{dk/dtheta}/G_k`` must equal
   what autograd gets by differentiating the direct sum.  Catches an algebra
   error in the three derivative kernels.
3. **Convergence.**  Halve the inner spacing, double the outer nodes, widen
   both truncations.  Whatever moves is the oracle's own error floor, and every
   candidate comparison below that number is noise.  This is what makes the
   oracle an oracle rather than just another implementation.
4. **Recovery.**  The problem is drawn from the model, so ``E[I0|data]`` must
   converge on ``I0_true`` as multiplicity grows.  1-3 only prove we evaluate
   our formula correctly; this is the one that would notice the formula itself
   being wrong -- a sign error in the mean/median shift, say, which every
   numerical check would happily confirm on both sides.
"""
from __future__ import annotations

import argparse
import math
import time

import torch

import oracle as ORACLE
from oracle import _conv_bank, _cubic, make_problem, oracle


def _direct(base, n_ph, bg, log_alpha, log_beta, v_obs, x, h):
    """Literal ``log sum_i f k dx``, no FFT.  ``v_obs`` is (M, C)."""
    lam = (x[None, :].exp() + bg[:, None]).clamp(min=1e-300)
    log_f = n_ph[:, None] * lam.log() - lam - torch.lgamma(n_ph[:, None] + 1)
    beta, alpha = log_beta.exp(), log_alpha.exp()
    log_norm = beta.log() - math.log(2.0) - torch.lgamma(1.0 / beta) - log_alpha
    cols = []
    for c in range(v_obs.shape[1]):
        mu = v_obs[:, c] + base
        z = (x[None, :] - mu[:, None]).abs() / alpha[:, None]
        lg = -z.clamp(min=1e-300).pow(beta[:, None]) + log_norm[:, None]
        cols.append(torch.logsumexp(lg + log_f, dim=1) + math.log(h))
    return torch.stack(cols, dim=1)


def _rel(a, b):
    d = (a.double() - b.double()).abs()
    s = b.double().abs().clamp(min=1e-300)
    return float(d.max()), float((d / s).median())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="tiny sizes; shakes out wiring on CPU in seconds")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    f64 = torch.float64

    print("=" * 78)
    print("1/2  FFT convolution and closed-form derivatives vs a direct sum")
    print("=" * 78)
    p = make_problem(n_per_mult=2, mults=(3,), sigma_ladder=(2.0, 50.0),
                     seed=7, device=dev)
    N, C = 1 << 15, 65
    x_hi = float((p.n_ph + 5 * p.n_ph.sqrt() + p.bg).clamp(min=1).log().max()) + 3.0
    reach = float((p.log_alpha.exp() * 30.0 ** (1.0 / p.log_beta.exp())).max())
    # sweep mu across the WHOLE outer range, not just near the peak: the FFT's
    # error is absolute, so its usable dynamic range has to be measured where
    # log p is far down, which is exactly where a peak-local check sees nothing.
    v_obs = (p.wilson_sigma.log()[p.hkl_id][:, None]
             + torch.linspace(-25, 6, C, device=dev, dtype=f64)[None, :])
    x_lo = float((v_obs[:, 0] + p.base).min()) - reach
    x = torch.linspace(x_lo, x_hi, N, device=dev, dtype=f64)
    h = float(x[1] - x[0])

    lp, r1, ra, rb = _conv_bank(p.n_ph, p.bg, p.log_alpha, p.log_beta, x, h)
    idx = ((v_obs + p.base[:, None] - x_lo) / h).clamp(0.0, N - 1.0)
    got = [_cubic(t, idx) for t in (lp, r1, ra, rb)]

    b = p.base.clone().requires_grad_(True)
    la = p.log_alpha.clone().requires_grad_(True)
    lb = p.log_beta.clone().requires_grad_(True)
    ref = _direct(b, p.n_ph, p.bg, la, lb, v_obs, x, h)
    rd = ref.detach()

    depth = rd.max(dim=1, keepdim=True).values - rd          # nats below peak
    err = (got[0] - rd).abs()
    print(f"  log p error vs depth below its own peak "
          f"(FFT floor is {ORACLE.FFT_NOISE:.0e} -> "
          f"{-math.log(ORACLE.FFT_NOISE):.0f} nats):")
    print(f"    {'depth':>12} {'n':>6} {'median abs':>12} {'max abs':>12}")
    edges = [0, 5, 10, 20, 30, 50, 1e9]
    for a_, b_ in zip(edges[:-1], edges[1:]):
        m = (depth >= a_) & (depth < b_)
        if not bool(m.any()):
            continue
        lab = f"{a_:.0f}-{b_:.0f}" if b_ < 1e8 else f">{a_:.0f}"
        print(f"    {lab:>12} {int(m.sum()):>6} {float(err[m].median()):>12.2e} "
              f"{float(err[m].max()):>12.2e}")

    near = depth < 10.0
    for i, (name, par) in enumerate(
        (("d/d base", b), ("d/d log alpha", la), ("d/d log beta", lb))
    ):
        torch.manual_seed(100 + i)
        g = torch.randn(ref.shape, device=dev, dtype=f64) * near
        (gr,) = torch.autograd.grad((ref * g).sum(), par, retain_graph=True)
        mine = (got[i + 1] * g).sum(1)
        mx, md = _rel(mine, gr)
        print(f"  {name:<13} (weighted to depth<10)  max abs {mx:.3e}   "
              f"median rel {md:.3e}")

    print("\n  end-to-end: FFT oracle vs direct oracle, small problem")
    ps = make_problem(n_per_mult=2, mults=(2, 16), sigma_ladder=(2.0, 50.0),
                      seed=3, device=dev)
    kw = dict(n_inner=1 << 14, n_outer=192, chunk=8)
    of = oracle(ps, **kw, method="fft")
    od = oracle(ps, **kw, method="direct")
    for nm, a_, r_ in (("log Z", of.log_Z, od.log_Z),
                       ("I0_mean", of.I0_mean, od.I0_mean),
                       ("I0_sd", of.I0_sd, od.I0_sd),
                       ("g_base", of.grad_base, od.grad_base),
                       ("g_log_alpha", of.grad_log_alpha, od.grad_log_alpha),
                       ("g_log_beta", of.grad_log_beta, od.grad_log_beta)):
        mx, md = _rel(a_, r_)
        print(f"    {nm:<12} max abs {mx:.3e}   median rel {md:.3e}")

    print("\n" + "=" * 78)
    print("3    convergence -- the error floor below which comparisons are noise")
    print("=" * 78)
    p = (make_problem(n_per_mult=2, mults=(2, 16), seed=0, device=dev)
         if args.quick else make_problem(seed=0, device=dev))
    big = dict(n_inner=1 << 13, n_outer=512, chunk=8) if args.quick else {}
    print(f"  problem: {p.n_hkl} reflections, M = {p.M:,} observations, "
          f"mult {tuple(int(m) for m in p.mult.unique())}", flush=True)
    t0 = time.time()
    ref = oracle(p, **big)
    print(f"  base run [{ref.meta['n_inner']} x {ref.meta['n_outer']}]: "
          f"{time.time()-t0:.0f}s   edge_mass max {float(ref.edge_mass.max()):.2e}",
          flush=True)

    variants = {
        "inner x2": dict(n_inner=(1 << 14) if args.quick else (1 << 19)),
        "outer x2": dict(n_outer=1024 if args.quick else 16384),
        "tail 30>60": dict(z_tail=60.0),
        "v_lo -25>-32": dict(v_lo=-32.0),
    }
    print(f"\n  {'variant':<14} {'log Z':>11} {'I0_mean':>11} {'I0_sd':>11} "
          f"{'g_base':>11} {'g_logalph':>11} {'g_logbeta':>11}")
    for name, kw in variants.items():
        t0 = time.time()
        alt = oracle(p, **{**big, **kw})
        row = []
        for a, r in ((alt.log_Z, ref.log_Z), (alt.I0_mean, ref.I0_mean),
                     (alt.I0_sd, ref.I0_sd), (alt.grad_base, ref.grad_base),
                     (alt.grad_log_alpha, ref.grad_log_alpha),
                     (alt.grad_log_beta, ref.grad_log_beta)):
            d = (a - r).abs()
            row.append(float((d / r.abs().clamp(min=1e-12)).median()))
        print(f"  {name:<14} " + " ".join(f"{x:11.2e}" for x in row)
              + f"   [{time.time()-t0:.0f}s]", flush=True)

    print("\n" + "=" * 78)
    print("4    recovery of I0_true -- is it the RIGHT formula, not just "
          "correctly evaluated")
    print("=" * 78)
    mult_h = torch.zeros(p.n_hkl, dtype=torch.long, device=dev)
    mult_h.index_add_(0, p.hkl_id, torch.ones_like(p.hkl_id))
    print(f"  {'mult':>6} {'n':>5} {'CC(I0_mean,I0_true)':>21} "
          f"{'med I0_mean/I0_true':>21} {'med |bias|/I0_sd':>18}")
    for m in sorted(int(x) for x in mult_h.unique()):
        sel = mult_h == m
        a, t = ref.I0_mean[sel], p.I0_true[sel]
        cc = float(torch.corrcoef(torch.stack([a, t]))[0, 1])
        ratio = float((a / t.clamp(min=1e-12)).median())
        z = float(((a - t).abs() / ref.I0_sd[sel].clamp(min=1e-12)).median())
        print(f"  {m:>6} {int(sel.sum()):>5} {cc:>21.5f} {ratio:>21.4f} {z:>18.3f}")


if __name__ == "__main__":
    main()
