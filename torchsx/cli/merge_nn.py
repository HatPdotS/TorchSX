"""``torchsx-merge-nn`` — a trained network plus the data, out comes an MTZ.

Everything about *how* to merge is read from the checkpoint: the encoder
geometry, the inner quadrature, the scatter family and miss component, the
outer-grid placement, the likelihood's background-variance term, and the data
recipe (d_min, spacegroup, gain, integration radii).  Flags override, but
nothing has to be repeated; a merge that rebuilt the network under different
defaults than it was trained with would load silently and merge wrongly.

Products.  The default run writes three files per split (and per half-set):

``<stem>.mtz``            the scaled posterior, F/SIGF.  The posterior is the
                          most precise readout but the shape likelihood it
                          comes from locates a skewed distribution rather than
                          estimating its mean; one smooth surface in
                          (resolution, intensity), fitted on every merged split
                          jointly, maps it onto the pooled scale.  This is the
                          product for refinement and for difference maps.
``<stem>_posterior.mtz``  the raw posterior, F/SIGF.
``<stem>_pooled.mtz``     ``sum(n - bg) / sum K`` as I/SIGI, a shape-free mean
                          estimate; French-Wilson happens downstream.

A ``<stem>.calibration.json`` sidecar per split records the surface and a
per-shell check of the mapping.  ``--no-scale-calibration`` writes only the
raw posterior under ``<stem>.mtz``.  ``--estimator flat|ml|pooled|mean|wls``
writes a single product with that estimator instead.

Examples
--------
::

    torchsx-merge-nn run/best.pt dark.stream -o merged.mtz
    torchsx-merge-nn run/best.pt --split dark=dark.stream --split light=1ps.stream \\
        --all-splits -o out/merged.mtz
    torchsx-merge-nn run/best.pt dark.stream -o pooled.mtz --estimator pooled
"""

from __future__ import annotations

import argparse
import json
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
                   choices=("posterior", "flat", "ml", "pooled", "mean", "wls"))
    p.add_argument("--all-splits", action="store_true",
                   help="merge every split, writing <out stem>_<name><suffix>")
    p.add_argument("--no-scale-calibration", dest="scale_calibration",
                   action="store_false", default=True,
                   help="write the raw posterior as the main product and skip "
                        "the pooled readout and the scaled posterior")
    p.add_argument("--calib-degree", default="9,9", metavar="NX,NY",
                   help="Chebyshev degrees of the scale-calibration surface in "
                        "s^2 and log I")
    p.add_argument("--half-seed", type=int, default=0,
                   help="seed for the crystal half-split")
    p.add_argument('--feature-reference', choices=('exact', 'legacy', 'ema'),
                   default='exact', help='Versioned checkpoint-context experiment')
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
        locate_windows=(8.0, 1.0, 0.125),
        posterior_prior=S["prior"],
        wilson_basis=S["wilson_basis"], wilson_degree=S["wilson_degree"],
        heldout_frac=0.05, device=S["device"], outdir=str(args.out.parent),
    )
    tr = AmortizedTrainer(prep, scale, obj, params, config=cfg)
    if S.get("basis") is not None:
        tr.set_reciprocal_basis(S["basis"], smooth_prior=S["wilson_basis"] == "smooth")
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
    from torchsx.nn.checkpoint_context import load_checkpoint_context
    load_checkpoint_context(tr, ckpt, args.feature_reference)
    if args.feature_reference != 'exact':
        tr._refresh_wilson_sigma()
    scale.eval()

    names = ckpt.get("splits") or bundle.split_names or [""]
    if tr.n_conditions != max(len(names), 1) and tr.n_conditions > 1:
        names = [str(i) for i in range(tr.n_conditions)]
    wanted = list(range(tr.n_conditions)) if args.all_splits else [0]

    if args.estimator == "posterior":
        merge_products(tr, prep, args, names, wanted, spacegroup=bundle.spacegroup,
                       merge_friedel=bundle.merge_friedel)
    else:
        merge_single_estimator(tr, prep, args, names, wanted, spacegroup=bundle.spacegroup,
                               merge_friedel=bundle.merge_friedel)
    tr.close()


def _split_halves(tr, k: int, seed: int):
    """Two disjoint halves of split ``k``'s crystals.

    Splitting crystals (not observations) is what makes the two halves
    independent measurements of the same structure: splitting observations
    would leave the same crystal's scale in both and inflate the agreement.
    """
    in_cond = (tr.condition == k).nonzero(as_tuple=True)[0]
    g = torch.Generator().manual_seed(seed)
    perm = in_cond[torch.randperm(in_cond.numel(), generator=g)]
    return perm[: perm.numel() // 2], perm[perm.numel() // 2:]


def _split_half_stats(a: dict, b: dict, d, *, key: str = "I0", scale=None) -> dict | None:
    """CC1/2 and R_split of two half merges on ``key`` (optionally rescaled)."""
    both = (a["seen"] & b["seen"]).cpu()
    I1 = a[key].detach().cpu().double(); I2 = b[key].detach().cpu().double()
    if scale is not None:
        I1 = I1 * torch.as_tensor(scale[0], dtype=I1.dtype)
        I2 = I2 * torch.as_tensor(scale[1], dtype=I2.dtype)
    I1, I2 = I1[both], I2[both]
    ok = torch.isfinite(I1) & torch.isfinite(I2)
    if int(ok.sum()) < 3:
        return None
    I1, I2 = I1[ok], I2[ok]
    cc = float(torch.corrcoef(torch.stack([I1, I2]))[0, 1])
    # CrystFEL's definition: the 1/sqrt(2) puts R_split on the scale of a
    # full-dataset R, not of the difference between two halves.
    rs = float((2 ** -0.5) * (I1 - I2).abs().sum() / (0.5 * (I1 + I2)).abs().sum())
    dd = torch.as_tensor(d)[both][ok]
    outer = dd <= float(dd.quantile(0.10))
    cc_o = float(torch.corrcoef(torch.stack([I1[outer], I2[outer]]))[0, 1])
    return {"cc_half": cc, "r_split": rs, "cc_half_outer": cc_o,
            "d_outer": float(dd[outer].max()), "n": int(ok.sum())}


def merge_products(tr, prep, args, names, wanted, *, spacegroup: str,
                   merge_friedel: bool = True, verbose: bool = True) -> dict:
    """Posterior, pooled and scaled posterior for every wanted split.

    Pass one merges every split (and, unless disabled, its two crystal
    halves) with both readouts.  One scale-calibration surface is then fitted
    on the full merges of all splits jointly and applied to every split and
    half.  Pass two writes, per split and half, ``<stem>.mtz`` (scaled
    posterior, the main product), ``<stem>_posterior.mtz`` (raw posterior) and
    ``<stem>_pooled.mtz`` (pooled intensities), plus a ``.calibration.json``
    sidecar per split with the surface and the per-split transfer check.
    With ``--no-scale-calibration`` only the raw posterior is written, under
    ``<stem>.mtz``.
    """
    from torchsx.merging.calibrate import ChebyshevScaleSurface, default_shell_edges

    hkl = prep.hkl_valid.detach().cpu().numpy().astype(int)
    d = prep.d_spacing.detach().cpu().numpy()
    cell = prep.median_cell.detach().cpu().tolist()
    calibrate = bool(args.scale_calibration)
    merges: dict[tuple[int, int], tuple[dict, dict | None]] = {}

    for k in wanted:
        cond = k if tr.n_conditions > 1 else None
        h1, h2 = _split_halves(tr, k, args.half_seed)
        sets = [(0, None), (1, h1), (2, h2)]
        for half, crystals in sets:
            post = tr.merge(crystals=crystals, condition=cond, estimator="posterior")
            pool = (tr.merge(crystals=crystals, condition=cond, estimator="pooled")
                    if calibrate else None)
            merges[(k, half)] = (post, pool)
        label = names[k] if k < len(names) else str(k)
        print(f"\nmerge [{label or 'all'}]: {merges[(k, 0)][0]['n_crystals']:,} crystals", flush=True)
        P.report_merge(merges[(k, 0)][0], verbose=verbose)

    surface = None
    fit_splits = []
    if calibrate:
        rows_P, rows_Q, rows_S, rows_D = [], [], [], []
        for k in wanted:
            post, pool = merges[(k, 0)]
            if not int(post["seen"].sum()):
                continue
            m = (post["seen"] & pool["seen"]).cpu().numpy()
            rows_P.append(post["I0"].detach().cpu().numpy()[m])
            rows_Q.append(pool["I"].detach().cpu().numpy()[m])
            rows_S.append(pool["I_sd"].detach().cpu().numpy()[m])
            rows_D.append(d[m])
            fit_splits.append(names[k] if k < len(names) else str(k))
        nx, ny = (int(v) for v in args.calib_degree.split(","))
        surface = ChebyshevScaleSurface(nx=nx, ny=ny)
        info = surface.fit(np.concatenate(rows_P), np.concatenate(rows_Q),
                           np.concatenate(rows_S), np.concatenate(rows_D))
        print(f"\nscale calibration (posterior -> pooled), fitted on {fit_splits}: "
              f"{info['n_coef']} coefficients on {info['n_fit']:,} reflections, "
              f"chi2/dof {info['chi2_per_dof']:.2f}, factor {info['factor_min']:.3f}-{info['factor_max']:.3f}, "
              f"mean ratio {info['mean_ratio_raw']:.3f} -> {info['mean_ratio_corrected']:.3f}", flush=True)

    edges = default_shell_edges(d)
    summary = {}
    for k in wanted:
        label = names[k] if k < len(names) else str(k)
        post, pool = merges[(k, 0)]
        if not post["n_crystals"] or not int(post["seen"].sum()):
            print(f"  [{label or 'all'}] nothing written for this split", flush=True)
            continue
        out = args.out
        if len(wanted) > 1:
            out = out.with_name(f"{out.stem}_{label}{out.suffix}")
        work = P.work_mask(hkl, args, verbose=verbose)
        factors = {}
        for half in (0, 1, 2):
            if (k, half) not in merges:
                continue
            po, pl = merges[(k, half)]
            stem = out.stem if half == 0 else f"{out.stem}_half{half}"
            F = po["F"].detach().cpu(); F_sd = po["F_sd"].detach().cpu()
            keep = po["seen"].cpu().numpy() & np.isfinite(F.numpy())
            if not keep.any():
                print(f"  half{half}: no usable reflections -- not written", flush=True)
                continue
            if surface is not None:
                f = surface.factor(d, po["I0"].detach().cpu().numpy())
                factors[half] = f
                sq = torch.from_numpy(np.sqrt(f)).to(F.dtype)
                P.write_mtz(out.with_name(f"{stem}{out.suffix}"), hkl, F * sq, F_sd * sq,
                            cell=cell, spacegroup=spacegroup, work=work, keep=keep,
                            merge_friedel=merge_friedel, verbose=verbose and half == 0)
                P.write_mtz(out.with_name(f"{stem}_posterior{out.suffix}"), hkl, F, F_sd,
                            cell=cell, spacegroup=spacegroup, work=work, keep=keep,
                            merge_friedel=merge_friedel, verbose=False)
                keep_p = pl["seen"].cpu().numpy() & np.isfinite(pl["I"].cpu().numpy()) & (pl["I_sd"].cpu().numpy() > 0)
                P.write_intensity_mtz(out.with_name(f"{stem}_pooled{out.suffix}"), hkl, pl["I"], pl["I_sd"],
                                      cell=cell, spacegroup=spacegroup, keep=keep_p,
                                      merge_friedel=merge_friedel, verbose=False)
            else:
                P.write_mtz(out.with_name(f"{stem}{out.suffix}"), hkl, F, F_sd,
                            cell=cell, spacegroup=spacegroup, work=work, keep=keep,
                            merge_friedel=merge_friedel, verbose=verbose and half == 0)
        rec = {"split": label, "products": ["scaled_posterior", "posterior", "pooled"] if surface is not None else ["posterior"]}
        if (k, 1) in merges and (k, 2) in merges:
            a, b = merges[(k, 1)][0], merges[(k, 2)][0]
            raw = _split_half_stats(a, b, d)
            if raw is not None:
                rec["split_half_posterior"] = raw
                line = (f"  half-datasets  posterior  CC1/2 {raw['cc_half']:.4f}  R_split {raw['r_split']:.4f}  "
                        f"(outer 10 % shell d<={raw['d_outer']:.2f} A: CC1/2 {raw['cc_half_outer']:.4f})  on {raw['n']:,} reflections")
                print(line, flush=True)
                if surface is not None and 1 in factors and 2 in factors:
                    sc = _split_half_stats(a, b, d, scale=(factors[1], factors[2]))
                    rec["split_half_scaled_posterior"] = sc
                    print(f"  half-datasets  scaled     CC1/2 {sc['cc_half']:.4f}  R_split {sc['r_split']:.4f}  "
                          f"(outer: CC1/2 {sc['cc_half_outer']:.4f})", flush=True)
                    pc = _split_half_stats(merges[(k, 1)][1], merges[(k, 2)][1], d, key="I")
                    if pc is not None:
                        rec["split_half_pooled"] = pc
                        print(f"  half-datasets  pooled     CC1/2 {pc['cc_half']:.4f}  R_split {pc['r_split']:.4f}", flush=True)
            else:
                print("  half-datasets  skipped (a half had no usable reflections)", flush=True)
        if surface is not None:
            m = (post["seen"] & pool["seen"]).cpu().numpy()
            check = surface.shell_check(d[m], post["I0"].detach().cpu().numpy()[m], pool["I"].detach().cpu().numpy()[m], edges)
            rec.update(calibration=surface.to_dict(), fitted_on=fit_splits, shell_check=check)
            print("  calibration check (mean posterior/pooled per shell, raw -> corrected): "
                  + "  ".join(f"{c['d_hi']:.2f}-{c['d_lo']:.2f}: {c['raw']:.3f}->{c['corrected']:.3f}" for c in check), flush=True)
            sidecar = out.with_name(f"{out.stem}.calibration.json")
            sidecar.write_text(json.dumps(rec, indent=1))
        summary[label] = rec
    return summary


def merge_single_estimator(tr, prep, args, names, wanted, *, spacegroup: str,
                           merge_friedel: bool = True) -> None:
    """One product per split for the non-posterior estimators (the original path)."""
    hkl = prep.hkl_valid.detach().cpu().numpy().astype(int)
    cell = prep.median_cell.detach().cpu().tolist()
    for k in wanted:
        cond = k if tr.n_conditions > 1 else None
        res = tr.merge(condition=cond, estimator=args.estimator)
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
        h1, h2 = _split_halves(tr, k, args.half_seed)
        halves = [tr.merge(crystals=part, condition=cond, estimator=args.estimator) for part in (h1, h2)]
        key = "I" if args.estimator in ("pooled", "mean", "wls") else "F"
        for j, r in enumerate(halves):
            keep_h = r["seen"].cpu().numpy() & np.isfinite(r[key].detach().cpu().numpy())
            if not keep_h.any():
                print(f"  half{j + 1}: no usable reflections -- not written", flush=True)
                continue
            ho = out.with_name(f"{out.stem}_half{j + 1}{out.suffix}")
            if key == "F":
                P.write_mtz(ho, hkl, r["F"], r["F_sd"], cell=cell, spacegroup=spacegroup,
                            work=P.work_mask(hkl, args, verbose=False), keep=keep_h,
                            merge_friedel=merge_friedel, verbose=False)
            else:
                P.write_intensity_mtz(ho.with_suffix(".mtz"), hkl, r["I"], r["I_sd"], cell=cell,
                                      spacegroup=spacegroup, keep=keep_h,
                                      merge_friedel=merge_friedel, verbose=False)
        st = _split_half_stats(halves[0], halves[1], prep.d_spacing.detach().cpu().numpy(),
                               key="I0" if key == "F" else "I")
        if st is not None:
            print(f"  half-datasets  CC1/2 {st['cc_half']:.4f}  R_split {st['r_split']:.4f}  "
                  f"(outer 10 % shell d<={st['d_outer']:.2f} A: CC1/2 {st['cc_half_outer']:.4f})  on {st['n']:,} reflections",
                  flush=True)
        seen = res["seen"].cpu().numpy()
        if args.estimator in ("pooled", "mean", "wls"):
            P.write_intensity_mtz(out.with_suffix(".mtz"), hkl, res["I"], res["I_sd"],
                                  cell=cell, spacegroup=spacegroup, keep=seen,
                                  merge_friedel=merge_friedel)
        else:
            keep = seen & np.isfinite(res["F"].detach().cpu().numpy())
            if not keep.any():
                print("  no usable reflections for this split -- nothing written", flush=True)
                continue
            P.write_mtz(out, hkl, res["F"], res["F_sd"], cell=cell, spacegroup=spacegroup,
                        work=P.work_mask(hkl, args), keep=keep,
                        merge_friedel=merge_friedel)


if __name__ == "__main__":
    main()
