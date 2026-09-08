"""``torchsx-merge-global`` — merge with no weighting, one global scatter.

The reference every trained run has to beat.  There is no scale model at all:
``K = 1`` for every observation, and the multiplicative scatter is described by
a single global width, shape and skew fitted to the same marginal evidence the
network is trained on.  Per-reflection ``I0`` comes from the same Wilson
posterior the trained merge uses, so the *only* difference from
``torchsx-merge-nn`` is what the scale model can express.

Why fit rather than fix.  A frozen ``w = 1.65`` silently absorbs whatever else
is unmodelled — part of it was standing in for the background estimate's own
error, so adding that error explicitly while ``w`` stayed put double-counted the
noise (median F down 40 %, F/sigF halved).  With ``w`` refitted under the
likelihood actually in use, that confound is gone.  This replaces the manual
``w`` × background-variance grid that had to be run by hand.

The fit is cheap: nothing reads the encoder embedding, so the features and the
Deep Sets stack are skipped entirely and a step is quadrature only.

Examples
--------
::

    torchsx-merge-global dark.stream --spacegroup "P 1" --int-radius 7,9,11 \\
        --gain 1.0 --dmin 0.9 -o unity.mtz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from torchsx.cli import _pipeline as P


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="torchsx-merge-global",
        description="Unweighted merge (K = 1) with a globally fitted scatter.",
    )
    P.add_data_args(p)
    P.add_model_args(p)
    P.add_free_args(p)
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument("--estimator", default="posterior",
                   choices=("posterior", "flat", "ml", "pooled", "mean"))
    p.add_argument("--fit-epochs", type=int, default=20,
                   help="passes over the data fitting the global scalars; the "
                        "evidence is additive over observations, so this "
                        "converges in very few (0 = no fit, use the inits)")
    p.add_argument("--fit-lr", type=float, default=5e-2,
                   help="Adam step for the two or three scalars")
    p.add_argument("--fit-tol", type=float, default=1e-4,
                   help="stop when the epoch's mean NLL/obs improves by less")
    p.add_argument("--max-obs", type=int, default=1_500_000)
    p.add_argument("--merge-split", default=None, metavar="NAME_OR_INDEX")
    p.add_argument("--all-splits", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    from torchsx.nn.train_amortized import AmortizedTrainer, TrainConfig

    t0 = time.time()
    bundle = P.load_prep(args)
    prep, params, scale, obj, S = P.build_stack(bundle, args, global_scale=True)
    assert not scale.heads.needs_embedding, "global scale must not read the encoder"

    cfg = TrainConfig(
        max_obs_per_batch=args.max_obs, epochs=args.fit_epochs, lr=args.fit_lr,
        # no EMA prior on the in-batch target and no warmup: with two or three
        # parameters there is no basin to escape and nothing to anneal.
        lam0=0.0, lam_halflife_steps=0, warmup_steps=0, grad_clip=20.0,
        log_every=10**9, merge_target="posterior", loss="evidence",
        posterior_k_coarse=S["k_coarse"], posterior_m_fine=S["m_fine"],
        posterior_n_sigma=S["n_sigma"], posterior_max_halfwidth=S["max_halfwidth"],
        stream_batches=args.stream_batches,
        locate_windows=tuple(float(x) for x in args.locate_windows.split(","))
        if args.locate_windows else (8.0, 1.0, 0.125),
        posterior_prior=S["prior"],
        wilson_basis=S["wilson_basis"], wilson_degree=S["wilson_degree"], heldout_frac=0.0,
        device=S["device"], outdir=str(args.out.parent),
    )
    tr = AmortizedTrainer(prep, scale, obj, params, config=cfg)
    if S.get("basis") is not None:
        tr.set_reciprocal_basis(S["basis"], smooth_prior=S["wilson_basis"] == "smooth")
    tr.data_recipe = dict(bundle.recipe)
    tr.split_names = list(bundle.split_names)

    def report(tag: str) -> None:
        h = scale.heads
        with torch.no_grad():
            sk = (float(h.global_log_skew) if getattr(h, "global_skew", False) else 0.0)
            miss = (f"  miss {float(torch.sigmoid(obj.miss_logit)):.4f}"
                    if obj.miss_logit is not None else "")
            print(f"  {tag}: w {float(h.global_log_w.exp()):.4f}  "
                  f"beta {float(h.beta_min + h.beta_span * torch.sigmoid(h.global_beta_raw)):.4f}"
                  + (f"  skew log(a_hi/a_lo) {sk:+.4f}" if sk else "") + miss,
                  flush=True)

    report("init")
    prev = None
    for ep in range(args.fit_epochs):
        t1 = time.time()
        loss_sum = n_used = 0.0
        for _, rows in tr.index.batches(cfg.max_obs_per_batch, generator=tr.gen,
                                        crystals=tr.train_crystals):
            r = tr.step(rows.to(S["device"]))
            loss_sum += r["nll_per_obs"] * r["n_used"]
            n_used += r["n_used"]
        nll = loss_sum / max(n_used, 1)
        print(f"  epoch {ep}  nll/obs {nll:.6f}  [{time.time() - t1:.1f}s]", flush=True)
        report(f"epoch {ep}")
        if prev is not None and abs(prev - nll) < args.fit_tol:
            print(f"  converged (|d nll/obs| < {args.fit_tol})", flush=True)
            break
        prev = nll
    scale.eval()
    tr.save_best(args.out.with_suffix(".pt"))

    names = bundle.split_names or [""]
    wanted = (list(range(tr.n_conditions)) if args.all_splits
              else [names.index(args.merge_split) if args.merge_split in names
                    else int(args.merge_split)] if args.merge_split is not None
              else [0])
    hkl = prep.hkl_valid.detach().cpu().numpy().astype(int)
    for k in wanted:
        res = tr.merge(condition=(k if tr.n_conditions > 1 else None),
                       estimator=args.estimator)
        label = names[k] if k < len(names) else str(k)
        out = (args.out if len(wanted) == 1
               else args.out.with_name(f"{args.out.stem}_{label}{args.out.suffix}"))
        print(f"\nmerge [{label or 'all'}]: {res['n_crystals']:,} crystals", flush=True)
        if not res["n_crystals"] or not int(res["seen"].sum()):
            P.report_merge(res)
            print("  nothing written for this split", flush=True)
            continue
        P.report_merge(res)
        seen = res["seen"].cpu().numpy()
        if args.estimator in ("pooled", "mean"):
            P.write_hkl(out.with_suffix(".hkl"), hkl, res["I"], res["I_sd"],
                        res["n_meas"], seen)
        else:
            keep = seen & np.isfinite(res["F"].detach().cpu().numpy())
            P.write_mtz(out, hkl, res["F"], res["F_sd"],
                        cell=prep.median_cell.detach().cpu().tolist(),
                        spacegroup=bundle.spacegroup,
                        work=P.work_mask(hkl, args), keep=keep)
    tr.close()
    print(f"\ntotal {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
