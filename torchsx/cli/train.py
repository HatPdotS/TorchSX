"""``torchsx-train`` — stream(s) in, trained scale network and merged MTZ out.

The amortised-scale recipe as it stands after the two-width scatter was locked
in.  One network predicts, per observation, a scale ``K`` and the width, shape
and skew of the multiplicative scatter around it; the merge integrates the
signal out against that likelihood rather than averaging corrected intensities.

Measured against partialator on the same free set: SwissFEL small molecule
R_free 0.119 (partialator 0.147), SACLA rhodopsin 0.276 (partialator 0.275,
with the low-resolution shell 0.304 vs 0.289 and the high-resolution end ours).
The outer shells are still behind a shape-free pooled estimator
(``--estimator pooled``) — that trade is deliberate and documented.

The network never sees an observation's own intensity, and never sees which
split a crystal belongs to: the scale is a function of geometry and of the
*other* observations in the crystal.  That is what keeps it from manufacturing
a difference between splits.

Examples
--------
::

    torchsx-train dark.stream --spacegroup "P 1" --int-radius 7,9,11 \\
        --gain 1.0 --dmin 0.9 -o run/

    torchsx-train --split dark=dark.stream --split light=1ps.stream \\
        --spacegroup "P 2 21 21" --int-radius 1,4,7 --gain 10.01 --dmin 1.9 \\
        --redundancy 200 --chunk-crystals 400 -o run/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from torchsx.cli import _pipeline as P


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="torchsx-train",
        description="Train the amortised scale model and merge to an MTZ.",
    )
    P.add_data_args(p)
    P.add_model_args(p)
    P.add_free_args(p)
    p.add_argument("-o", "--out", type=Path, required=True,
                   help="output directory: best.pt, metrics.jsonl, merged*.mtz")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--redundancy", type=float, default=None,
                   help="observations per reflection in a batch.  Prefer this "
                        "over --max-obs: the in-batch merge target is built "
                        "from ~max_obs/n_hkl observations, so a batch size "
                        "carried between data sets silently changes the "
                        "objective (1.5 M obs is 106 obs/reflection on a 14 k "
                        "cell and 13 on a 114 k one).")
    p.add_argument("--max-obs", type=int, default=1_500_000,
                   help="batch size in observations when --redundancy is unset")
    p.add_argument("--heldout-frac", type=float, default=0.05)
    p.add_argument("--chunk-crystals", type=int, default=0,
                   help="crystals per encoder chunk (0 = whole batch).  The "
                        "encoder is separable across crystals, so this is "
                        "exact; 400 is what makes redundancy 200 fit on 40 GB.")
    p.add_argument("--node-ckpt", action="store_true",
                   help="recompute outer quadrature nodes in backward")
    p.add_argument("--stream-batches", action="store_true",
                   help="keep the prepared observations on the host and copy "
                        "only each batch's rows to the device")
    p.add_argument("--lam0", type=float, default=10.0,
                   help="EMA prior strength on the in-batch merge target")
    p.add_argument("--lam-halflife", type=int, default=0,
                   help="steps for lam0 to halve; 0 = no decay (the safe "
                        "default: rhodopsin's in-batch target has I/sigma ~2 "
                        "even at redundancy 200 and cannot carry itself)")
    p.add_argument("--split-half-every", type=int, default=0,
                   help="run the CC1/2 / R_split merge every N epochs")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--estimator", default="posterior",
                   choices=("posterior", "flat", "ml", "pooled", "mean"),
                   help="estimator for the final merge (see torchsx-merge-nn)")
    p.add_argument("--no-merge", action="store_true",
                   help="stop after training; merge later with torchsx-merge-nn")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    from torchsx.nn.train_amortized import AmortizedTrainer, TrainConfig

    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    bundle = P.load_prep(args)
    prep, params, scale, obj, S = P.build_stack(bundle, args)
    if args.chunk_crystals > 0:
        scale.set_encoder_chunking(args.chunk_crystals)

    cfg = TrainConfig(
        max_obs_per_batch=args.max_obs,
        target_redundancy=args.redundancy,
        epochs=args.epochs, lr=args.lr,
        lam0=args.lam0, lam_halflife_steps=args.lam_halflife,
        warmup_steps=50, grad_clip=20.0, log_every=args.log_every,
        merge_target="posterior", loss="evidence",
        posterior_k_coarse=S["k_coarse"], posterior_m_fine=S["m_fine"],
        posterior_n_sigma=S["n_sigma"], posterior_max_halfwidth=S["max_halfwidth"],
        posterior_prior=S["prior"],
        posterior_checkpoint_nodes=args.node_ckpt,
        encoder_chunk_crystals=args.chunk_crystals,
        stream_batches=args.stream_batches,
        split_half_every=args.split_half_every,
        heldout_frac=args.heldout_frac,
        device=S["device"], outdir=str(args.out),
    )
    tr = AmortizedTrainer(prep, scale, obj, params, config=cfg)
    tr.data_recipe = dict(bundle.recipe)
    tr.split_names = list(bundle.split_names)
    print(f"train {tr.train_crystals.numel():,} / heldout "
          f"{tr.heldout_crystals.numel():,} crystals"
          + (f", {tr.n_conditions} splits {bundle.split_names}"
             if tr.n_conditions > 1 else ""), flush=True)

    base = tr.evaluate(tr.heldout_crystals, phase="heldout_init")
    print(f"  epoch -   heldout nll/obs {base['nll_per_obs']:.5f}  "
          f"var_z {base['var_z']:.3f}", flush=True)
    for ep in range(args.epochs):
        t1 = time.time()
        tr._refresh_feature_reference()
        tr._refresh_wilson_sigma()
        loss_sum = n_used = 0.0
        aucs = []
        for _, rows in tr.index.batches(cfg.max_obs_per_batch, generator=tr.gen,
                                        crystals=tr.train_crystals):
            r = tr.step(rows.to(S["device"]))
            # weight by n_used: a plain mean of per-obs RATIOS is not the
            # epoch's per-observation NLL when the batches differ in size
            loss_sum += r["nll_per_obs"] * r["n_used"]
            n_used += r["n_used"]
            if "condition_auc" in r:
                aucs.append(r["condition_auc"])
        h = tr.evaluate(tr.heldout_crystals)
        tr.save_best()
        auc = f"  auc {sum(aucs) / len(aucs):.3f}" if aucs else ""
        print(f"  epoch {ep:<3d} train {loss_sum / max(n_used, 1):.5f}  "
              f"heldout {h['nll_per_obs']:.5f}  "
              f"ev/hkl {h['evidence_per_hkl']:.4f}  var_z {h['var_z']:.3f}"
              f"{auc}  [{time.time() - t1:.1f}s]", flush=True)
        if args.split_half_every and (ep + 1) % args.split_half_every == 0:
            sh = tr.split_half_metrics()
            print(f"    [merge] CC1/2 full {sh['cc_half_full']:.4f}  "
                  f"Rsplit {sh['r_split_full']:.4f}  "
                  f"train {sh['cc_half_train']:.4f} val {sh['cc_half_val']:.4f}",
                  flush=True)

    # Report the parameters that scored best on held-out, not wherever training
    # happened to stop: a 300-epoch run peaked at ~230 and degraded past it.
    ckpt_path = tr.save_best(args.out / "best.pt")
    if tr.restore_best():
        print(f"restored best held-out checkpoint from step {tr.best_step} "
              f"(nll {tr.best_heldout:.5f}) -> {ckpt_path}", flush=True)
    scale.eval()

    if args.no_merge:
        tr.close()
        print(f"total {time.time() - t0:.0f}s", flush=True)
        return

    # Free the training state before merging in-process: the merge accumulates
    # (n_hkl, k_coarse) on the device, and a training run still holding its
    # optimiser state has OOM'd here before.
    tr.opt = None
    if S["device"].startswith("cuda"):
        torch.cuda.empty_cache()

    names = bundle.split_names or [""]
    for k, name in enumerate(names):
        res = tr.merge(condition=(k if tr.n_conditions > 1 else None),
                       estimator=args.estimator)
        print(f"\nmerge [{name or 'all'}]: {res['n_crystals']:,} crystals", flush=True)
        P.report_merge(res)
        _write(args, prep, res, bundle,
               args.out / (f"merged_{name}.mtz" if name else "merged.mtz"))
    tr.close()
    print(f"\ntotal {time.time() - t0:.0f}s", flush=True)


def _write(args, prep, res: dict, bundle: P.PrepBundle, out: Path) -> None:
    """One merge result → MTZ (or CrystFEL .hkl for a moment estimator)."""
    import numpy as np

    hkl = prep.hkl_valid.detach().cpu().numpy().astype(int)
    seen = res["seen"].cpu().numpy()
    if res["estimator"] in ("pooled", "mean"):
        P.write_hkl(out.with_suffix(".hkl"), hkl, res["I"], res["I_sd"],
                    res["n_meas"], seen)
        return
    keep = seen & np.isfinite(res["F"].detach().cpu().numpy())
    P.write_mtz(out, hkl, res["F"], res["F_sd"],
                cell=prep.median_cell.detach().cpu().tolist(),
                spacegroup=bundle.spacegroup,
                work=P.work_mask(hkl, args), keep=keep)


if __name__ == "__main__":
    main()
