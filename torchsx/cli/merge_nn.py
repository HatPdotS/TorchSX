"""``torchsx-merge-nn`` — a trained network plus the data, out comes an MTZ.

Everything about *how* to merge is read from the checkpoint: the encoder
geometry, the inner quadrature, the scatter family and miss component, the
outer-grid placement, the likelihood's background-variance term, and the data
recipe (d_min, spacegroup, gain, integration radii).  Flags override, but
nothing has to be repeated — a merge that rebuilds the network under the
*default* encoder geometry instead of the trained one loads silently and lost
0.022 in R_free the one time it happened.

Estimators.  ``posterior`` (default) is the Wilson-posterior mean of
``sqrt(I0)`` taken over the very weights the merge computed — what the
French-Wilson tables approximate, evaluated on the actual posterior instead of
interpolated, and written as F/SIGF with no intensity column so French-Wilson
cannot fire a second time downstream.  ``flat`` and ``ml`` are the same integral
without the Wilson prior.  ``pooled`` (``sum(n-bg)/sum K``) and ``mean``
(partialator's equal weighting) are shape-free moment estimators written as a
CrystFEL ``.hkl``; on the SwissFEL outer shells ``pooled`` still beats the
posterior (R_free 0.151 vs 0.230), which is why it ships.

Examples
--------
::

    torchsx-merge-nn run/best.pt dark.stream -o merged.mtz
    torchsx-merge-nn run/best.pt --split dark=dark.stream --split light=1ps.stream \\
        --all-splits -o out/merged.mtz
    torchsx-merge-nn run/best.pt dark.stream -o pooled.hkl --estimator pooled
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from torchsx.cli import _pipeline as P


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="torchsx-merge-nn",
        description="Merge with a trained scale network.",
    )
    p.add_argument("ckpt", type=Path, help="best.pt from torchsx-train")
    P.add_data_args(p, require_recipe=False)
    P.add_model_args(p)
    P.add_free_args(p)
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument("--estimator", default="posterior",
                   choices=("posterior", "flat", "ml", "pooled", "mean"))
    p.add_argument("--merge-split", default=None, metavar="NAME_OR_INDEX",
                   help="merge only this split (default: the first)")
    p.add_argument("--all-splits", action="store_true",
                   help="merge every split, writing <out stem>_<name><suffix>")
    p.add_argument("--no-half-datasets", dest="half_datasets",
                   action="store_false", default=True,
                   help="skip the half-dataset merges.  By default every split "
                        "is also merged as two random halves of its crystals, "
                        "giving CC1/2 and R_split the way partialator's "
                        ".hkl1/.hkl2 do -- a merge with no internal consistency "
                        "statistic is not a result anyone can judge.  The "
                        "halves are written next to the full merge so the "
                        "numbers can be recomputed independently.  Costs two "
                        "extra merges per split.")
    p.add_argument("--half-seed", type=int, default=0,
                   help="seed for the crystal half-split")
    p.add_argument("--unity", action="store_true",
                   help="ignore the trained weights and merge at K = 1 -- the "
                        "reference every trained run has to beat")
    return p


def main(argv: list[str] | None = None) -> None:
    # ``--split NAME=PATH`` names an input; ``--merge-split NAME`` selects
    # which of them to merge.  Two different things, two different flags.
    args = _build_parser().parse_args(argv)
    from torchsx.nn.train_amortized import AmortizedTrainer, TrainConfig

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"checkpoint {args.ckpt} (step {ckpt.get('best_step')}, "
          f"heldout {ckpt.get('best_heldout')})", flush=True)
    bundle = P.load_prep(args, recipe_defaults=ckpt.get("data") or {})
    prep, params, scale, obj, S = P.build_stack(bundle, args, ckpt=ckpt)

    cfg = TrainConfig(
        max_obs_per_batch=1_500_000, epochs=0,
        merge_target="posterior", loss="evidence",
        posterior_k_coarse=S["k_coarse"], posterior_m_fine=S["m_fine"],
        posterior_n_sigma=S["n_sigma"], posterior_max_halfwidth=S["max_halfwidth"],
        stream_batches=args.stream_batches,
        locate_windows=tuple(float(x) for x in args.locate_windows.split(","))
        if args.locate_windows else (8.0, 1.0, 0.125),
        posterior_prior=S["prior"],
        heldout_frac=0.05, device=S["device"], outdir=str(args.out.parent),
    )
    tr = AmortizedTrainer(prep, scale, obj, params, config=cfg)
    if not args.unity:
        # Checkpoints written before the data buffers were made non-persistent
        # carry the training set's own _I/_bg/_sigma, whose length is that data
        # set's observation count; drop them rather than refuse to merge a
        # different set.  Everything the model actually needs must be present.
        want = scale.state_dict()
        have = {k: v for k, v in ckpt["state_dict"].items() if k in want}
        missing = [k for k in want if k not in have]
        if missing:
            raise SystemExit(f"checkpoint is missing {len(missing)} parameters, "
                             f"first {missing[:3]}")
        scale.load_state_dict(have)
        ema = ckpt.get("ema_I0")
        if ema is not None:
            ema = ema.to(S["device"])
            for k, t in enumerate(tr.targets):
                t.ema_I0.copy_(ema if ema.dim() == 1 else ema[min(k, ema.shape[0] - 1)])
    else:
        print("  --unity: merging at K = 1 with the untrained (zero-init) heads",
              flush=True)
    tr._refresh_wilson_sigma()
    scale.eval()

    names = ckpt.get("splits") or bundle.split_names or [""]
    if tr.n_conditions != max(len(names), 1) and tr.n_conditions > 1:
        names = [str(i) for i in range(tr.n_conditions)]
    if args.all_splits:
        wanted = list(range(tr.n_conditions))
    elif args.merge_split is not None:
        wanted = [names.index(args.merge_split)
                  if args.merge_split in names else int(args.merge_split)]
    else:
        wanted = [0]

    for k in wanted:
        res = tr.merge(condition=(k if tr.n_conditions > 1 else None),
                       estimator=args.estimator)
        label = names[k] if k < len(names) else str(k)
        out = args.out
        if len(wanted) > 1:
            out = out.with_name(f"{out.stem}_{label}{out.suffix}")
        print(f"\nmerge [{label or 'all'}]: {res['n_crystals']:,} crystals", flush=True)
        if not res["n_crystals"] or not int(res["seen"].sum()):
            P.report_merge(res)
            print("  nothing written for this split", flush=True)
            continue
        P.report_merge(res)
        hkl = prep.hkl_valid.detach().cpu().numpy().astype(int)
        if args.half_datasets:
            # Two disjoint halves of THIS split's crystals, merged exactly as
            # the whole split was.  Splitting crystals (not observations) is
            # what makes the two halves independent measurements of the same
            # structure -- splitting observations would leave the same crystal's
            # scale in both and inflate the agreement.
            in_cond = (tr.condition == k).nonzero(as_tuple=True)[0]
            g = torch.Generator().manual_seed(args.half_seed)
            perm = in_cond[torch.randperm(in_cond.numel(), generator=g)]
            halves = []
            for j, part in enumerate((perm[: perm.numel() // 2],
                                      perm[perm.numel() // 2:])):
                r = tr.merge(crystals=part,
                             condition=(k if tr.n_conditions > 1 else None),
                             estimator=args.estimator)
                ho = out.with_name(f"{out.stem}_half{j + 1}{out.suffix}")
                keep_h = r["seen"].cpu().numpy() & np.isfinite(
                    r["F"].detach().cpu().numpy())
                if not keep_h.any():
                    # A half whose accumulation came back entirely non-finite
                    # leaves nothing to write, and ReflectionData rejects an
                    # all-False mask.  Report it and carry on: the split's own
                    # merge is unaffected, only its CC1/2 is lost.
                    print(f"  half{j + 1}: no usable reflections "
                          f"({r.get('n_nonfinite_rows', 0):,} non-finite rows, "
                          f"{int(r['seen'].sum()):,} seen) -- not written",
                          flush=True)
                    halves.append(None)
                    continue
                halves.append(r)
                P.write_mtz(ho, hkl, r["F"], r["F_sd"],
                            cell=prep.median_cell.detach().cpu().tolist(),
                            spacegroup=bundle.spacegroup,
                            work=P.work_mask(hkl, args, verbose=False),
                            keep=keep_h, verbose=False)
            a, b = halves
            if a is None or b is None:
                print("  half-datasets  skipped (a half had no usable "
                      "reflections)", flush=True)
                a = b = None
        if args.half_datasets and a is not None and b is not None:
            both = (a["seen"] & b["seen"]).cpu()
            I1 = a["I0"].detach().cpu()[both].double()
            I2 = b["I0"].detach().cpu()[both].double()
            ok = torch.isfinite(I1) & torch.isfinite(I2)
            I1, I2 = I1[ok], I2[ok]
            cc = float(torch.corrcoef(torch.stack([I1, I2]))[0, 1])
            # CrystFEL's definition: the 1/sqrt(2) puts R_split on the scale of
            # a full-dataset R, not of the difference between two halves.
            rs = float((2 ** -0.5) * (I1 - I2).abs().sum()
                       / (0.5 * (I1 + I2)).abs().sum())
            d = prep.d_spacing.detach().cpu()[both][ok]
            outer = d <= float(d.quantile(0.10))
            cc_o = float(torch.corrcoef(
                torch.stack([I1[outer], I2[outer]]))[0, 1])
            print(f"  half-datasets  CC1/2 {cc:.4f}  R_split {rs:.4f}  "
                  f"(outer 10 % shell d<={float(d[outer].max()):.2f} A: "
                  f"CC1/2 {cc_o:.4f})  on {int(ok.sum()):,} reflections",
                  flush=True)
        seen = res["seen"].cpu().numpy()
        if args.estimator in ("pooled", "mean"):
            P.write_hkl(out.with_suffix(".hkl"), hkl, res["I"], res["I_sd"],
                        res["n_meas"], seen)
        else:
            keep = seen & np.isfinite(res["F"].detach().cpu().numpy())
            if not keep.any():
                print("  no usable reflections for this split -- "
                      "nothing written", flush=True)
                continue
            P.write_mtz(out, hkl, res["F"], res["F_sd"],
                        cell=prep.median_cell.detach().cpu().tolist(),
                        spacegroup=bundle.spacegroup,
                        work=P.work_mask(hkl, args), keep=keep)
    tr.close()


if __name__ == "__main__":
    main()
