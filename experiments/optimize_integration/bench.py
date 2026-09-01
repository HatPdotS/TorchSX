"""Accuracy + speed harness for the posterior accumulator.

Two independent measurements, deliberately on different data sizes:

* **Accuracy** on a subsample with ``hkl_id = arange(M)``, so ``S`` is the
  per-observation ``log p`` table and no group sum can hide an error.  Scored
  against a float64 dense-grid reference, for values *and* for gradients.
* **Speed** on the full batch with the real ``hkl_id``, forward and
  forward+backward, plus peak allocation.

Values and gradients are reported separately because they trade off: a grid
that follows ``log_mean`` covers the scatter tail (good values) but breaks the
assumption that the nodes are parameter-free, which is what makes the analytic
backward exact (bad gradients).  A candidate that wins on one and loses on the
other is the normal case, not a bug.

``S`` is only ever consumed through ``logsumexp_c``, so an error constant in c
is invisible downstream.  ``prof`` columns score ``S - mean_c(S)``; ``raw`` is
shown alongside only to make that gap visible.

    python bench.py --batch batch.pt --only current,eager
    python bench.py --self-check          # is the reference itself converged?
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

import ops

HERE = Path(__file__).parent


# ── reference gradients ────────────────────────────────────────────────


def _ref_grads(b, n, bg, la, lb, v, g, *, n_points, lo, hi, chunk=256):
    """d/d(base, log_alpha, log_beta) of ``sum(g * log_p)`` in float64.

    Chunked over observations and accumulated: every parameter here is
    per-observation, so chunks are independent and this is exact, not an
    approximation of the full-batch backward.
    """
    f64 = torch.float64
    dev = b.device
    log_S = torch.linspace(lo, hi, n_points, device=dev, dtype=f64)
    S = log_S.exp()
    log_w = ops._trap_log_w(n_points, float(log_S[1] - log_S[0]), dev, f64)
    gb = torch.zeros_like(b, dtype=f64)
    ga = torch.zeros_like(b, dtype=f64)
    gl = torch.zeros_like(b, dtype=f64)

    for s in range(0, b.shape[0], chunk):
        e = min(s + chunk, b.shape[0])
        bc = b[s:e].to(f64).requires_grad_(True)
        ac = la[s:e].to(f64).requires_grad_(True)
        lc = lb[s:e].to(f64).requires_grad_(True)
        beta, alpha = lc.exp(), ac.exp()
        log_norm = ops._log_gg_norm(beta) - ac
        lam = (S[None, :] + bg[s:e, None].to(f64)).clamp(min=1e-300)
        log_lik = (
            n[s:e, None].to(f64) * lam.log() - lam
            - torch.lgamma(n[s:e].to(f64) + 1)[:, None]
        )
        tot = 0.0
        for c in range(v.shape[1]):
            mu = v[s:e, c].to(f64) + bc
            z = (log_S[None, :] - mu[:, None]).abs() / alpha[:, None]
            log_gg = -z.clamp(min=1e-300).pow(beta[:, None]) + log_norm[:, None]
            lp = torch.logsumexp(log_gg + log_lik + log_w[None, :], dim=1)
            tot = tot + (lp * g[s:e, c].to(f64)).sum()
        d = torch.autograd.grad(tot, (bc, ac, lc))
        gb[s:e], ga[s:e], gl[s:e] = d
    return gb, ga, gl


# ── metrics ────────────────────────────────────────────────────────────


def _err(a, r):
    """(max, median) absolute deviation, and RMS-relative deviation."""
    d = (a.double() - r.double()).abs()
    rms = r.double().pow(2).mean().sqrt().clamp(min=1e-300)
    return float(d.max()), float(d.median()), float(d.pow(2).mean().sqrt() / rms)


def _profile(S):
    return S - S.mean(dim=1, keepdim=True)


def _mismatch(a, r):
    """``1 - cos`` between the candidate's gradient and the reference's.

    The median relative error treats a per-observation error that is RANDOM --
    and so averages out of the parameter gradient over 1.5 M observations --
    the same as one that is systematic and does not.  What the optimiser sees
    is the direction of the summed vector.  Reported alongside the median so
    the two can be compared; where they disagree, this is the one that matters.
    """
    a, r = a.double().flatten(), r.double().flatten()
    return float(1.0 - (a @ r) / (a.norm() * r.norm()).clamp(min=1e-300))


# ── main ───────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=Path, default=HERE / "batch.pt")
    ap.add_argument("--acc-obs", type=int, default=4096)
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--no-grad", action="store_true")
    ap.add_argument("--self-check", action="store_true",
                    help="halve the reference density and re-run: the gap is "
                         "the reference's own error floor, and every number "
                         "below it is noise")
    ap.add_argument("--out", type=Path, default=HERE / "results.json")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.load(args.batch, map_location=dev)
    M_full = d["base"].shape[0]
    C = d["v_nodes"].shape[1]
    print(f"batch: M={M_full:,}  n_hkl={d['n_hkl']:,}  C={C}  "
          f"(after {d['steps']} warm-up steps)\n", flush=True)

    # accuracy subsample: hkl_id = arange, so S IS the per-obs log p table
    gen = torch.Generator(device="cpu").manual_seed(0)
    sub = torch.randperm(M_full, generator=gen)[: args.acc_obs].to(dev)
    b, n, bg = d["base"][sub], d["n_ph"][sub], d["bg"][sub]
    la, lb = d["log_alpha"][sub], d["log_beta"][sub]
    v = d["v_nodes"][d["hkl_id"][sub]]
    ident = torch.arange(sub.numel(), device=dev)
    Ma = int(sub.numel())
    g = torch.randn(Ma, C, generator=torch.Generator().manual_seed(1)).to(dev)

    t0 = time.time()
    R = ops.reference(b, n, bg, la, lb, ident, v, Ma)
    print(f"reference: {time.time()-t0:.1f}s", flush=True)
    if args.self_check:
        R2 = ops.reference(b, n, bg, la, lb, ident, v, Ma, n_points=16001)
        mx, md, rel = _err(_profile(R2), _profile(R))
        print(f"  self-check (16001 vs 32001 nodes): profile max {mx:.2e}  "
              f"med {md:.2e}  rel {rel:.2e}\n"
              f"  -> treat anything below this as noise", flush=True)
    Rg = None
    if not args.no_grad:
        t0 = time.time()
        Rg = _ref_grads(b, n, bg, la, lb, v, g, n_points=32001, lo=-60.0, hi=60.0)
        print(f"reference grads: {time.time()-t0:.1f}s", flush=True)

    names = [x for x in (args.only.split(",") if args.only else ops.CANDIDATES)
             if x in ops.CANDIDATES]
    if dev == "cpu":
        names = [x for x in names if x != "current"]   # Triton needs CUDA
        args.no_time = True
    rows = []
    hdr = (f"\n{'candidate':<16} {'prof med':>9} {'prof rel':>9} "
           f"{'raw med':>9} | {'g_base rel/cos':>15} {'g_lalpha':>15} "
           f"{'g_lbeta':>15} | {'fwd ms':>8} {'f+b ms':>8}")
    print(hdr)
    print("-" * len(hdr.strip()))

    for name in names:
        fn = ops.CANDIDATES[name]
        S = fn(b, n, bg, la, lb, ident, v, Ma)
        pmax, pmed, prel = _err(_profile(S), _profile(R))
        _, rmed, _ = _err(S, R)
        rec = {"name": name, "prof_max": pmax, "prof_med": pmed,
               "prof_rel": prel, "raw_med": rmed}

        gcell = ["    n/a  "] * 3
        if Rg is not None:
            bb, aa, ll = (x.clone().requires_grad_(True) for x in (b, la, lb))
            Sg = fn(bb, n, bg, aa, ll, ident, v, Ma)
            (Sg * g).sum().backward()
            for i, (got, ref) in enumerate(
                zip((bb.grad, aa.grad, ll.grad), Rg)
            ):
                if got is None:
                    continue
                _, _, rel = _err(got, ref)
                cos = _mismatch(got, ref)
                nm = ['base', 'log_alpha', 'log_beta'][i]
                rec[f"grad_rel_{nm}"] = rel
                rec[f"grad_cos_{nm}"] = cos
                gcell[i] = f"{rel:.1e}/{cos:.0e}"

        fwd = fb = peak = float("nan")
        if not args.no_time:
            B = (d["base"], d["n_ph"], d["bg"], d["log_alpha"], d["log_beta"],
                 d["hkl_id"], d["v_nodes"], d["n_hkl"])
            torch.cuda.reset_peak_memory_stats()
            for _ in range(2):
                fn(*B)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(args.reps):
                fn(*B)
            torch.cuda.synchronize()
            fwd = (time.time() - t0) / args.reps * 1e3

            bb = B[0].clone().requires_grad_(True)
            aa = B[3].clone().requires_grad_(True)
            ll = B[4].clone().requires_grad_(True)
            gf = torch.ones(d["n_hkl"], C, device=dev)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(args.reps):
                out = fn(bb, B[1], B[2], aa, ll, B[5], B[6], B[7])
                (out * gf).sum().backward()
                bb.grad = aa.grad = ll.grad = None
            torch.cuda.synchronize()
            fb = (time.time() - t0) / args.reps * 1e3
            peak = torch.cuda.max_memory_allocated() / 1e9
        rec.update(fwd_ms=fwd, fb_ms=fb, peak_gb=peak)
        rows.append(rec)
        print(f"{name:<16} {pmed:9.2e} {prel:9.2e} {rmed:9.2e} | "
              f"{gcell[0]:>15} {gcell[1]:>15} {gcell[2]:>15} | "
              f"{fwd:8.1f} {fb:8.1f}", flush=True)

    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
