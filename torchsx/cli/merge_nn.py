"""Export raw posterior amplitudes and their own uncertainties for every state and half.

Model and likelihood settings come from the checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from torchsx.cli import _pipeline as P


def _build_parser():
    p = argparse.ArgumentParser(prog="torchsx.merge", description="Export raw posterior full/half MTZs for every supplied state.")
    p.add_argument("ckpt", type=Path, help="checkpoint from torchsx.train")
    P.add_data_args(p)
    P.add_model_args(p)
    P.add_free_args(p)
    p.add_argument("-o", "--out", type=Path, required=True,
                   help="output MTZ, e.g. merged.mtz; with several states one file per state, "
                        "merged_STATE.mtz, each with its _half1/_half2 companions")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    import torch

    # TF32 for the encoder's float32 matmuls; the likelihood and posterior
    # integrals are elementwise and Triton code, which this does not touch.
    torch.set_float32_matmul_precision("high")
    from torchsx.nn.train_amortized import AmortizedTrainer, TrainConfig

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"checkpoint {args.ckpt} (step {ckpt.get('best_step')}, "
          f"heldout {ckpt.get('best_heldout')})", flush=True)
    if args.seed is None:
        args.seed = ((ckpt.get("data") or {}).get("training") or {}).get("config", {}).get("seed", 0)
    bundle = P.load_prep(args, recipe_defaults=ckpt.get("data") or {})
    prep, params, scale, obj, S = P.build_stack(bundle, args, ckpt=ckpt)

    cfg = TrainConfig(
        max_obs_per_batch=P.READOUT_MAX_OBS, epochs=0,
        posterior_k_coarse=S["k_coarse"], posterior_m_fine=S["m_fine"],
        posterior_n_sigma=S["n_sigma"], posterior_max_halfwidth=S["max_halfwidth"],
        stream_batches=True, encoder_chunk_crystals=400,
        locate_windows=(8.0, 1.0, 0.125),
        posterior_prior=S["prior"], wilson_degree=S["wilson_degree"],
        heldout_frac=0.05, device=S["device"], outdir=str(args.out.parent),
    )
    tr = AmortizedTrainer(prep, scale, obj, params, config=cfg)
    tr.set_reciprocal_basis(S["basis"])
    restore_wilson_surface(tr, prep, ckpt, names=bundle.split_names)
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
    # The network weights, the fitted feature standardisation and the Wilson
    # surface come from the checkpoint; everything per reflection is formed
    # from the supplied data, as in training.
    scale.eval()

    names = bundle.split_names or [""]
    if tr.n_conditions != max(len(names), 1) and tr.n_conditions > 1:
        names = [str(i) for i in range(tr.n_conditions)]
    try:
        merge_products(tr, prep, args, names, list(range(tr.n_conditions)),
                       spacegroup=bundle.spacegroup, merge_friedel=bundle.merge_friedel)
    finally:
        tr.close()


def restore_wilson_surface(tr, prep, ckpt, names=None) -> None:
    """Install the checkpoint's trained Wilson surfaces on these reflections.

    A surface belongs to the network's scale convention, so a merge evaluates
    the stored basis definition on the supplied reflections and applies the
    trained coefficients, whatever reflection set they were trained on.  Each
    input state takes the surface of the trained state of the same name
    (``names`` against the checkpoint's ``splits``), or of the same position
    when the checkpoint and the inputs carry the same unnamed states; any other
    state takes the first trained state's surface.
    """
    if tr.cfg.posterior_prior != "wilson":
        return
    surface = ckpt.get("wilson_surface")
    if surface is None:
        raise SystemExit("the checkpoint carries no Wilson surface")
    coefs, trained = list(surface["coef"]), list(ckpt.get("splits") or [])
    names = list(names or [])
    chosen = []
    for k in range(len(tr.targets)):
        name = names[k] if k < len(names) else None
        if name is not None and name in trained:
            chosen.append(coefs[trained.index(name)])
        elif not names and not trained and len(coefs) == len(tr.targets):
            chosen.append(coefs[k])
        else:
            chosen.append(coefs[0])
    from torchsx.merging.priors.reciprocal_basis import ReciprocalBasis
    phi = ReciprocalBasis.from_definition(surface["basis"]).evaluate(prep.hkl_valid)
    for t, c in zip(tr.targets, chosen):
        c = c.to(t.wilson_coef.device)
        t.set_basis(phi.to(t.wilson_coef.device))
        t.set_wilson_coef(c)
        if t.wilson_coef_param is not None:
            t.wilson_coef_param.data.copy_(c)


def _export_crystals(tr, k: int):
    """Crystals of condition ``k`` that the export merges: all of them, less an
    external held-out set (``--heldout-stream``), which is scored, never merged."""
    import torch

    in_cond = (tr.condition == k).nonzero(as_tuple=True)[0]
    ext = getattr(tr, "external_heldout", None)
    if ext is not None:
        in_cond = in_cond[~torch.isin(in_cond, ext.cpu())]
    return in_cond


def _split_halves(tr, k: int, seed: int):
    """Disjoint crystal subsets for the shared-network internal CC1/2 statistic."""
    import torch

    in_cond = _export_crystals(tr, k)
    g = torch.Generator().manual_seed(seed)
    perm = in_cond[torch.randperm(in_cond.numel(), generator=g)]
    return perm[: perm.numel() // 2], perm[perm.numel() // 2:]


def _split_half_stats(a: dict, b: dict, d, *, key: str = "I0") -> dict | None:
    """CC1/2 and R_split of two half merges on ``key``."""
    import torch

    both = (a["seen"] & b["seen"]).cpu()
    I1 = a[key].detach().cpu().double(); I2 = b[key].detach().cpu().double()
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
    """Write raw posterior amplitudes for each requested split and its halves."""
    hkl = prep.hkl_valid.detach().cpu().numpy().astype(int)
    d = prep.d_spacing.detach().cpu().numpy()
    cell = prep.median_cell.detach().cpu().tolist()
    summary = {}
    for k in wanted:
        cond = k if tr.n_conditions > 1 else None
        sets = [(0, _export_crystals(tr, k) if getattr(tr, "external_heldout", None) is not None else None)]
        h1, h2 = _split_halves(tr, k, args.seed + 1000000007)
        sets += [(1, h1), (2, h2)]
        merges = {half: tr.merge(crystals=crystals, condition=cond, estimator="posterior")
                  for half, crystals in sets}
        post = merges[0]
        label = names[k] if k < len(names) else str(k)
        print(f"\nmerge [{label or 'all'}]: {post['n_crystals']:,} crystals", flush=True)
        P.report_merge(post, verbose=verbose)
        if not post["n_crystals"] or not int(post["seen"].sum()):
            print(f"  [{label or 'all'}] nothing written for this split", flush=True)
            continue
        out = args.out
        if len(wanted) > 1:
            out = out.with_name(f"{out.stem}_{label}{out.suffix}")
        work = P.work_mask(hkl, args, verbose=verbose)
        for half, po in merges.items():
            stem = out.stem if half == 0 else f"{out.stem}_half{half}"
            F = po["F"].detach().cpu(); F_sd = po["F_sd"].detach().cpu()
            keep = po["seen"].cpu().numpy() & np.isfinite(F.numpy())
            if keep.any():
                P.write_mtz(out.with_name(f"{stem}{out.suffix}"), hkl, F, F_sd,
                            cell=cell, spacegroup=spacegroup, work=work, keep=keep,
                            merge_friedel=merge_friedel,
                            verbose=verbose and half == 0)
        rec = {"split": label, "products": ["posterior"], "prior": post["prior"]}
        stats = _split_half_stats(merges[1], merges[2], d)
        if stats is not None:
            rec["split_half_posterior"] = stats
            print(f"  half-datasets  posterior  CC1/2 {stats['cc_half']:.4f}  "
                  f"R_split {stats['r_split']:.4f} on {stats['n']:,} reflections", flush=True)
        summary[label] = rec
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    main()
