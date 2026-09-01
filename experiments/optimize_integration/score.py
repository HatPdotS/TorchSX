"""Score the production posterior integration against the certified oracle.

This is the optimisation loop: one synthetic problem, one oracle run (7 s), then
every candidate configuration measured against it for values *and* gradients.

What is deliberately handed to the candidate for free, so that the numbers
isolate the **integration** rather than the surrounding state:

* ``wilson_sigma`` is set to the problem's true Sigma ladder, not estimated;
* ``ema_I0`` is seeded with a plain unity merge, i.e. the grid centre starts
  where production's does after its first step;
* ``log_e_s`` is zeroed and ``log_K`` carries ``base`` directly, so
  ``d/d log_K`` is the oracle's ``d/d base`` with no chain rule in between.

Everything else -- ``locate``'s scan, the coarse->fine interpolation, the node
count, the inner grid, float32 -- is under test.

The error that matters for ``I0_mean`` is not relative: it is the error as a
fraction of ``I0_sd``, because that is the statistical width the merge already
carries.  Both are reported; the ``/sd`` column is the one to read.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from oracle import make_problem, oracle


def _make_scatter(log_alpha, log_beta):
    from torchsx.merging.scatter.gengauss import GenGaussLogScatter
    # Stands in for PredictedGenGaussScatter: per-observation (alpha, beta)
    # with no Parameters/ObsInfo behind them.  Must stay a GenGaussLogScatter
    # instance or FixedLogGrid._can_use_triton drops the fast path.
    s = GenGaussLogScatter()
    s.log_alpha_beta_obs = lambda params, meta: (log_alpha, log_beta)  # noqa: E731
    return s


def production(prob, *, likelihood="poisson", k_coarse=21, m_fine=129,
               n_sigma=6.0, scan_points=9,
               scan_halfwidth=8.0, prior="wilson", node_placement="laplace",
               inner_points=2001, inner_half=34.5, dtype=torch.float32,
               eager=False, grad=True, use_fused=True):
    """One forward of ``PosteriorMergeTarget`` on the synthetic problem."""
    from torchsx.merging.likelihood import Poisson
    from torchsx.merging.quadrature.fixed_log_grid import FixedLogGrid
    from torchsx.nn.heads import ScalePrediction
    from torchsx.nn.objective import MarginalNLL
    from torchsx.nn.posterior_target import PosteriorMergeTarget

    dev = prob.base.device
    cast = lambda t: t.to(dtype)                                   # noqa: E731
    n_ph, bg, hkl = cast(prob.n_ph), cast(prob.bg), prob.hkl_id

    base = cast(prob.base).clone().requires_grad_(grad)
    la = cast(prob.log_alpha).clone().requires_grad_(grad)
    lb = cast(prob.log_beta).clone().requires_grad_(grad)
    pred = ScalePrediction(
        log_K=base, log_w=cast(prob.log_w), beta=lb.exp(),
        log_alpha=la, log_e_s=torch.zeros_like(base),
    )

    # unity seed for the grid centre: mean signal per reflection
    num = torch.zeros(prob.n_hkl, device=dev, dtype=dtype)
    cnt = torch.zeros(prob.n_hkl, device=dev, dtype=dtype)
    num.index_add_(0, hkl, n_ph - bg)
    cnt.index_add_(0, hkl, torch.ones_like(n_ph))
    I0_init = (num / cnt.clamp(min=1)).clamp(min=1e-3)

    tgt = PosteriorMergeTarget(
        I0_init, k_coarse=k_coarse, m_fine=m_fine, n_sigma=n_sigma,
        prior=prior, node_placement=node_placement,
        scan_points=scan_points, scan_halfwidth=scan_halfwidth,
        use_fused=use_fused,
    ).to(dev)
    tgt.wilson_sigma.copy_(cast(prob.wilson_sigma))     # true Sigma, not estimated

    if likelihood == "poisson":
        lik = Poisson()
    else:
        # The Gaussian path already carries the background's ESTIMATION
        # variance, bg_var_factor * bg with bg_var_factor = n_peak/n_bg, which
        # the Poisson path has no way to express -- it takes bg as an exact
        # constant.  At median n_ph = 136 the Poisson/Gaussian distinction is
        # immaterial (q01 is 8.6, so <1 % of observations are in the regime
        # where it matters), and the missing variance term is worth a 143 %
        # bias on weak reflections.
        from torchsx.merging.likelihood import GaussianPolynomialVariance
        # sigma^2 = mu_obs + (n_peak/n_bg)*bg -- the Poisson variance PLUS the
        # background estimate's variance, which is the correct total for
        # n - bg_est and is exactly what the Poisson path cannot express.
        lik = GaussianPolynomialVariance(
            var_const=0.0, var_linear=1.0, var_quadratic=0.0,
            bg_var_factor=149.0 / 128.0,
        )
    obj = MarginalNLL(
        FixedLogGrid(S_min=math.exp(-inner_half), S_max=math.exp(inner_half),
                     n_points=inner_points).to(dev).to(dtype),
        lik,
    )
    scatter = _make_scatter(la, lb)
    sigma = torch.maximum(n_ph, bg).clamp(min=1e-6).sqrt()

    import contextlib
    from torchsx.merging.quadrature import fixed_log_grid as _flg
    eager_ctx = _flg.eager_only() if (eager and hasattr(_flg, "eager_only")) \
        else contextlib.nullcontext()
    with (torch.enable_grad() if grad else torch.no_grad()), eager_ctx:
        out = tgt(obj, scatter, None, pred, n_ph, bg, sigma, hkl, None)
        g = (None, None, None)
        if grad:
            g = torch.autograd.grad(out.log_Z[out.seen].sum(), (base, la, lb))
    return out, g, tgt


def _relmed(a, r):
    return float(((a.double() - r.double()).abs()
                  / r.double().abs().clamp(min=1e-30)).median())


def _mismatch(a, r):
    """``1 - cos`` between a candidate gradient and the oracle's.

    The median relative error is the wrong summary for a gradient: it treats a
    per-observation error that is random (and averages out of the parameter
    gradient over 1.1 M observations) the same as one that is systematic (and
    does not).  What the optimiser actually sees is the direction of the summed
    vector, so this is the number that decides whether a node count is enough.
    """
    a, r = a.double().flatten(), r.double().flatten()
    return float(1.0 - (a @ r) / (a.norm() * r.norm()).clamp(min=1e-300))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--mode", choices=("survey", "pareto", "nodes", "accuracy"),
                    default="survey")
    ap.add_argument("--real-like", action="store_true",
                    help="generative parameters fitted to the real dark batch "
                         "(oracle.REAL_LIKE).  This is the operating point that "
                         "matters; --realistic and --bright were guesses that "
                         "put the median count 5x too low.")
    ap.add_argument("--bright", action="store_true",
                    help="realistic, plus a bright tail.  The inner grid has to "
                         "resolve the Poisson factor, whose width in log S is "
                         "1/sqrt(n) -- so the node count needed is set by the "
                         "BRIGHTEST observations, and a Sigma ladder that tops "
                         "out at 200 never tests that.")
    ap.add_argument("--realistic", action="store_true",
                    help="match the dark stream's operating point instead of "
                         "the deliberately hard default: median multiplicity "
                         "751 (not 2..751) and w around the fitted 1.74 (not "
                         "0.4..3.0).  The hard problem is the right one for "
                         "ranking configurations; this one says whether the "
                         "ranking still holds where production actually sits.")
    ap.add_argument("--out", type=Path, default=Path("score.json"))
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from oracle import REAL_LIKE
    if args.real_like:
        prob = make_problem(seed=args.seed, device=dev, n_per_mult=16,
                            **REAL_LIKE)
    else:
      prob = make_problem(
        seed=args.seed, device=dev,
        **(dict(n_per_mult=32, mults=(128, 751), w_range=(1.2, 2.4),
                beta_range=(1.3, 1.5),
                **(dict(sigma_ladder=(5.0, 50.0, 500.0, 5000.0))
                   if args.bright else {}))
           if (args.realistic or args.bright) else {}))
    t0 = time.time()
    ref = oracle(prob)
    q = prob.n_ph.quantile(torch.tensor([0.5, 0.9, 0.99, 1.0], device=dev,
                                        dtype=prob.n_ph.dtype))
    print(f"oracle: {prob.n_hkl} refl, M={prob.M:,}, {time.time()-t0:.0f}s   "
          f"(error floor ~3e-9 values / 1e-9 grads)")
    # the inner node count is set by the sharpest Poisson factor, width
    # 1/sqrt(n) in log S, so record what counts were actually exercised
    print(f"  n_ph  med {q[0]:.0f}  q90 {q[1]:.0f}  q99 {q[2]:.0f}  "
          f"max {q[3]:.0f}   -> narrowest Poisson width "
          f"{1/max(float(q[3]),1)**0.5:.4f} in log S\n", flush=True)

    BASE = dict(k_coarse=21, m_fine=129, n_sigma=6.0, scan_points=9,
                inner_points=2001)
    configs = [("production", {})]
    if args.mode == "pareto":
        # m_fine made no difference at all in the survey (65..513 identical), so
        # it is pinned at its cheapest value here and the budget goes to the two
        # knobs that did move: how well the grid is PLACED (scan_points) and how
        # many outer nodes sit on it (k_coarse).
        configs = [("production", {}),
                   ("f64 eager", dict(dtype=torch.float64, eager=True,
                                      inner_points=1001))]
        for sp in (9, 17, 33):
            for kc in (15, 21, 31):
                for ns in (6.0, 8.0):
                    for ip in (501, 1001):
                        configs.append((
                            f"s{sp} k{kc} ns{ns:.0f} i{ip}",
                            dict(scan_points=sp, k_coarse=kc, n_sigma=ns,
                                 inner_points=ip, m_fine=65),
                        ))
        for hw in (4.0, 12.0):
            configs.append((f"halfwidth={hw:.0f}",
                            dict(scan_points=17, k_coarse=21, n_sigma=8.0,
                                 inner_points=1001, m_fine=65,
                                 scan_halfwidth=hw)))
    configs += [(f"k_coarse={k}", dict(k_coarse=k)) for k in (9, 15, 31, 51)]
    configs += [(f"m_fine={m}", dict(m_fine=m)) for m in (65, 257, 513)]
    configs += [(f"n_sigma={s}", dict(n_sigma=s)) for s in (4.0, 8.0, 12.0)]
    configs += [(f"scan_points={s}", dict(scan_points=s)) for s in (17, 33)]
    configs += [(f"inner={n}", dict(inner_points=n)) for n in (501, 1001, 4001)]
    configs += [("float64", dict(dtype=torch.float64))]
    configs += [("k31_m257_s8_i4001",
                 dict(k_coarse=31, m_fine=257, n_sigma=8.0, inner_points=4001))]

    if args.mode == "nodes":
        # one knob at a time, everything else at production, on the realistic
        # problem: where does each curve stop paying for itself?
        configs = [("production", {})]
        configs += [(f"inner {n}", dict(inner_points=n))
                    for n in (401, 601, 801, 1001, 1501, 2001, 3001, 4001, 6001)]
        configs += [(f"k_coarse {k}", dict(k_coarse=k))
                    for k in (7, 9, 13, 17, 21, 25, 31, 41, 51)]
        configs += [(f"scan {s_}", dict(scan_points=s_))
                    for s_ in (5, 7, 9, 13, 17, 25)]
        configs += [(f"m_fine {m}", dict(m_fine=m)) for m in (33, 65, 129)]

    if args.mode == "accuracy":
        # Accuracy first: push every knob toward its accurate end at once and
        # find the floor.  The one-at-a-time sweep says k_coarse dominates and
        # has no knee, the inner grid saturates by 1501, m_fine must be 129+,
        # and a +-20 extent beats +-34.5 -- so hold all of those at their good
        # values and drive k_coarse until something stops improving.  What stops
        # it is the thing worth fixing next.
        ACC = dict(inner_points=2001, inner_half=20.0, m_fine=257,
                   scan_points=13, n_sigma=6.0)
        configs = [("production", {}),
                   ("acc base k21", {**ACC, "k_coarse": 21})]
        configs += [(f"acc k{k}", {**ACC, "k_coarse": k})
                    for k in (31, 41, 51, 71, 101)]
        # is the +-20 extent still right once everything else is accurate?
        configs += [(f"acc k51 half={h:.0f}", {**ACC, "k_coarse": 51,
                                               "inner_half": h})
                    for h in (15.0, 25.0, 34.5)]
        # and does the inner grid re-enter once the outer one is resolved?
        configs += [(f"acc k51 inner={n}", {**ACC, "k_coarse": 51,
                                            "inner_points": n})
                    for n in (1001, 4001, 8001)]
        configs += [("acc k101 f64ref", {**ACC, "k_coarse": 101,
                                         "scan_points": 25, "m_fine": 513})]
        # ... and +-34.5 was merely the widest thing tested, not a measured
        # optimum.  With alpha up to 1.8 and beta ~ 1.07 the GenGauss still
        # holds exp(-30^1.07) = 3e-17 of mass out at z = 30, i.e. |eps| = 54,
        # so the production grid may be truncating too.
        WIDE = {**ACC, "inner_half": 34.5}
        configs += [(f"wide k{k} half={h:.0f}", {**WIDE, "k_coarse": k,
                                                 "inner_half": h})
                    for k in (51, 101) for h in (34.5, 45.0, 60.0)]
        # at a wide extent the inner grid has to stretch further, so re-check
        # whether it needs more nodes to keep up
        configs += [(f"wide k101 h45 inner={n}",
                     {**WIDE, "k_coarse": 101, "inner_half": 45.0,
                      "inner_points": n}) for n in (2001, 4001, 8001)]
        # ISO-COST reallocation.  Work is k_coarse x inner_points, and the two
        # integrals want opposite things: the merge value is set by the OUTER
        # node count (flat in inner nodes from 1001 to 8001), the gradient
        # direction by the INNER one (21x better over the same range).  So at a
        # fixed budget there is a best split, and production's 21 x 2001 was
        # never chosen -- it was inherited.
        BUDGET = 21 * 2001
        for k in (21, 25, 27, 31, 35, 41, 45, 51):
            n = int(BUDGET / k) | 1
            if n < 900:          # 800 is a cliff, not a soft limit -- stay clear
                continue
            configs.append((f"iso k{k} inner={n}",
                            {**ACC, "inner_half": 34.5, "k_coarse": k,
                             "inner_points": n}))
        # and the same split at 1.5x and 0.7x budget, to see whether the
        # OPTIMAL RATIO moves with the budget or is a property of the problem
        for mult, tag in ((0.7, "0.7x"), (1.5, "1.5x")):
            for k in (21, 31, 41, 51):
                n = int(BUDGET * mult / k) | 1
                if n < 900:
                    continue
                configs.append((f"{tag} k{k} inner={n}",
                                {**ACC, "inner_half": 34.5, "k_coarse": k,
                                 "inner_points": n}))

    hdr = (f"{'config':<20} {'log Z':>10} {'I0_mean':>10} {'/sd':>9} "
           f"{'I0_sd':>10} | {'1-cos b':>10} {'1-cos a':>10} {'1-cos B':>10} | "
           f"{'ms':>7} {'edge':>8}")
    print(hdr)
    print("-" * len(hdr))
    production(prob, **BASE)        # pay the Triton JIT before timing anything
    rows = []
    for name, over in configs:
        cfg = {**BASE, **over}
        try:
            out, g, tgt = production(prob, **cfg)
        except Exception as exc:                              # noqa: BLE001
            print(f"{name:<20} FAILED: {type(exc).__name__}: {exc}")
            continue
        e_lz = _relmed(out.log_Z, ref.log_Z)
        e_im = _relmed(out.I0_mean, ref.I0_mean)
        e_sd = _relmed(out.I0_sd, ref.I0_sd)
        # the number that matters: error as a fraction of the posterior width
        e_ov = float(((out.I0_mean.double() - ref.I0_mean)
                      / ref.I0_sd.clamp(min=1e-30)).abs().median())
        gref = (ref.grad_base, ref.grad_log_alpha, ref.grad_log_beta)
        eg = [_relmed(x, r) for x, r in zip(g, gref)]
        cg = [_mismatch(x, r) for x, r in zip(g, gref)]

        if dev == "cuda":
            for _ in range(2):
                production(prob, **cfg)
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(args.reps):
                production(prob, **cfg)
            torch.cuda.synchronize()
            ms = (time.time() - t) / args.reps * 1e3
        else:
            ms = float("nan")
        em = float(out.edge_mass.max())
        rows.append(dict(name=name, cfg={k: str(v) for k, v in cfg.items()},
                         log_Z=e_lz, I0_mean=e_im, I0_mean_over_sd=e_ov,
                         I0_sd=e_sd, g_base=eg[0], g_log_alpha=eg[1],
                         g_log_beta=eg[2], ms=ms, edge_max=em,
                         cos_base=cg[0], cos_log_alpha=cg[1],
                         cos_log_beta=cg[2]))
        print(f"{name:<20} {e_lz:10.2e} {e_im:10.2e} {e_ov:9.2e} {e_sd:10.2e} | "
              f"{cg[0]:10.2e} {cg[1]:10.2e} {cg[2]:10.2e} | {ms:7.1f} {em:8.1e}",
              flush=True)

    # where does the error live?
    print("\nproduction error by multiplicity (|dI0|/I0_sd, median):")
    out, _, _ = production(prob, **BASE)
    mult_h = torch.zeros(prob.n_hkl, dtype=torch.long, device=dev)
    mult_h.index_add_(0, prob.hkl_id, torch.ones_like(prob.hkl_id))
    d = ((out.I0_mean.double() - ref.I0_mean) / ref.I0_sd.clamp(min=1e-30)).abs()
    print(f"  {'mult':>6} {'n':>4} {'|dI0|/sd':>10} {'log Z rel':>11} "
          f"{'Sigma':>8}")
    for m in sorted(int(x) for x in mult_h.unique()):
        sel = mult_h == m
        lz = ((out.log_Z[sel].double() - ref.log_Z[sel]).abs()
              / ref.log_Z[sel].abs().clamp(min=1e-30)).median()
        print(f"  {m:>6} {int(sel.sum()):>4} {float(d[sel].median()):>10.2e} "
              f"{float(lz):>11.2e} {float(prob.wilson_sigma[sel].median()):>8.1f}")

    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
