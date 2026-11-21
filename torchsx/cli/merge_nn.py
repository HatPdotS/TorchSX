"""Export merged amplitudes and their uncertainties for every state.

The product is the inverse-variance (wls) merge on the trained scales, for all
crystals and for each half. ``--estimator posterior`` writes the Wilson-posterior
amplitudes instead; its half files then hold pooled amplitudes, which share no prior.

Model and likelihood settings come from the checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from torchsx.cli import _pipeline as P
from torchsx.merge_stats import r_split


def _build_parser():
    p = argparse.ArgumentParser(prog="torchsx.merge", description="Export merged full/half MTZs for every supplied state.")
    p.add_argument("ckpt", type=Path, help="checkpoint from torchsx.train")
    P.add_data_args(p)
    P.add_model_args(p)
    P.add_free_args(p)
    P.add_export_args(p)
    p.add_argument("-o", "--out", type=Path, required=True,
                   help="output MTZ, e.g. merged.mtz; with several states one file per state, "
                        "merged_STATE.mtz, each with its _half1/_half2 companions; time and GPU "
                        "memory by phase go to merged.resources.json")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    import torch

    P.configure_torch()
    from torchsx.nn.train_amortized import AmortizedTrainer, TrainConfig

    probe = P.ResourceProbe(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    probe.begin("prepare")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"checkpoint {args.ckpt} (step {ckpt.get('best_step')}, "
          f"heldout {ckpt.get('best_heldout')})", flush=True)
    if args.seed is None:
        args.seed = ((ckpt.get("data") or {}).get("training") or {}).get("config", {}).get("seed", 0)
    bundle = P.load_prep(args, recipe_defaults=ckpt.get("data") or {})
    prep, params, scale, obj, S = P.build_stack(bundle, args, ckpt=ckpt)

    cfg = TrainConfig(
        max_obs_per_batch=P.READOUT_MAX_OBS, posterior_k_coarse=S["k_coarse"], posterior_m_fine=S["m_fine"],
        posterior_n_sigma=S["n_sigma"], posterior_max_halfwidth=S["max_halfwidth"],
        stream_batches=True,
        locate_windows=(8.0, 1.0, 0.125),
        posterior_prior=S["prior"], wilson_degree=S["wilson_degree"],
        heldout_frac=0.05, device=S["device"], outdir=str(args.out.parent),
    )
    tr = AmortizedTrainer(prep, scale, obj, params, config=cfg)
    tr.set_reciprocal_basis(S["basis"])
    restore_wilson_surface(tr, prep, ckpt, names=bundle.split_names)
    restore_intensity_modulation(tr, prep, ckpt)
    if (ckpt.get("scale") or {}).get("phase_classes"):
        from torchsx.merging.scale.pseudo_translation import PhaseClasses
        tr.set_phase_classes(PhaseClasses.from_definition(ckpt["scale"]["phase_classes"]))
    # Strict: a checkpoint from another version of the model fails here rather
    # than merging with some of its weights silently left out.
    try:
        scale.load_state_dict(ckpt["state_dict"])
    except RuntimeError as exc:
        raise SystemExit(f"checkpoint does not match this version's model; retrain it.\n{exc}")
    # The network weights, the fitted feature standardisation and the Wilson
    # surface come from the checkpoint; everything per reflection is formed
    # from the supplied data, as in training.
    scale.eval()

    names = bundle.split_names or [""]
    if tr.n_conditions != max(len(names), 1) and tr.n_conditions > 1:
        names = [str(i) for i in range(tr.n_conditions)]
    out = args.out
    try:
        probe.begin("readout")
        merge_products(tr, prep, args, names, list(range(tr.n_conditions)),
                       spacegroup=bundle.spacegroup, merge_friedel=bundle.merge_friedel)
        probe.end()
        out.with_suffix(".resources.json").write_text(json.dumps(probe.summary(), indent=2) + "\n")
    finally:
        tr.close()
    print(probe.report(), flush=True)


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


def restore_intensity_modulation(tr, prep, ckpt) -> None:
    """Install the checkpoint's prior modulation (:mod:`torchsx.merging.priors.modulation`),
    evaluated on these reflections, on the trained surfaces; nothing when it has none."""
    d = ckpt.get("intensity_modulation")
    if d is None:
        return
    import torch

    from torchsx.merging.priors.modulation import PhaseModulation
    mod = PhaseModulation.from_definition(d)
    log_mod = torch.as_tensor(mod.log_prior_offset(
        prep.hkl_valid.long().cpu().numpy(), (1.0 / prep.d_spacing.double() ** 2).cpu().numpy()),
        dtype=torch.float32)
    for t in tr.targets:
        t.set_modulation(log_mod.to(t.wilson_log_mod.device))


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


def _agree(I1, I2) -> tuple[float, float]:
    import torch

    if len(I1) < 3:
        return float("nan"), float("nan")
    return float(torch.corrcoef(torch.stack([I1, I2]))[0, 1]), r_split(I1, I2)


def _split_half_stats(a: dict, b: dict, d, *, key: str = "I0", rank=None) -> dict | None:
    """CC1/2 and R_split of two half merges on ``key``."""
    import torch

    both = (a["seen"] & b["seen"]).cpu()
    I1 = a[key].detach().cpu().double(); I2 = b[key].detach().cpu().double()
    I1, I2 = I1[both], I2[both]
    ok = torch.isfinite(I1) & torch.isfinite(I2)
    if int(ok.sum()) < 3:
        return None
    I1, I2 = I1[ok], I2[ok]
    cc, rs = _agree(I1, I2)
    dd = torch.as_tensor(d)[both][ok]
    outer = dd <= float(dd.quantile(0.10))
    cc_o, _ = _agree(I1[outer], I2[outer])
    by_shell = []
    order = torch.argsort(dd, descending=True)
    for chunk in torch.tensor_split(order, min(10, order.numel())):
        c, r = _agree(I1[chunk], I2[chunk])
        by_shell.append({"d_lo": float(dd[chunk].max()), "d_hi": float(dd[chunk].min()),
                         "cc_half": c, "r_split": r, "n": int(chunk.numel())})
    # quintiles ranked by the full merge, so neither half is favoured
    score = (0.5 * (I1 + I2) if rank is None
             else rank.detach().cpu().double()[both][ok])
    by_quintile = []
    for chunk in torch.tensor_split(torch.argsort(score), min(5, score.numel())):
        c, r = _agree(I1[chunk], I2[chunk])
        by_quintile.append({"cc_half": c, "r_split": r, "n": int(chunk.numel())})
    return {"cc_half": cc, "r_split": rs, "cc_half_outer": cc_o,
            "d_outer": float(dd[outer].max()), "n": int(ok.sum()),
            "by_shell": by_shell, "by_quintile": by_quintile}


def _amplitudes(po: dict, hkl=None, d=None, spacegroup: str | None = None):
    """Amplitude, its uncertainty and the reflections that have one.

    A posterior merge carries its own amplitudes. A moment estimator carries
    intensities, which may be negative: with ``hkl``, ``d`` and ``spacegroup``
    they go through French-Wilson, else they are truncated at zero.
    """
    import torch

    seen = po["seen"].cpu().numpy()
    if "F" in po:
        F, F_sd = po["F"].detach().cpu(), po["F_sd"].detach().cpu()
        return F, F_sd, seen & np.isfinite(F.numpy())
    I, sd = po["I"].detach().cpu(), po["I_sd"].detach().cpu()
    usable = seen & np.isfinite(I.numpy()) & np.isfinite(sd.numpy()) & (sd.numpy() > 0)
    if hkl is not None and spacegroup is not None and usable.any():
        from torchsx.merge_stats import _french_wilson
        try:
            f, s, ok = _french_wilson(hkl[usable], I.numpy()[usable], sd.numpy()[usable],
                                      np.asarray(d)[usable], spacegroup)
        except Exception as exc:  # noqa: BLE001 - too few reflections for the Wilson fit
            print(f"  French-Wilson unavailable ({exc}); amplitudes are sqrt(max(I, 0))", flush=True)
        else:
            F = torch.zeros(len(seen)); F_sd = torch.ones(len(seen))
            idx = np.flatnonzero(usable)
            F[idx] = torch.as_tensor(f, dtype=torch.float32)
            F_sd[idx] = torch.as_tensor(s, dtype=torch.float32)
            keep = np.zeros(len(seen), dtype=bool)
            keep[idx[ok]] = True
            return F, F_sd, keep
    F = I.clamp(min=0).sqrt()
    return F, sd / (2 * F).clamp(min=sd.sqrt().clamp(min=1e-30)), usable


def _report_halves(rec: dict, key: str, name: str, stats: dict | None) -> None:
    if stats is None:
        return
    rec[key] = stats
    print(f"  half-datasets  {name}  CC1/2 {stats['cc_half']:.4f}  "
          f"R_split {stats['r_split']:.4f} on {stats['n']:,} reflections", flush=True)
    print("    R_split by intensity quintile, weak to strong: "
          + "  ".join(f"{100 * q['r_split']:.1f}%" for q in stats["by_quintile"])
          + f"; outer shell {100 * stats['by_shell'][-1]['r_split']:.1f}% "
          f"(CC1/2 {stats['by_shell'][-1]['cc_half']:.3f})", flush=True)


def merge_products(tr, prep, args, names, wanted, *, spacegroup: str,
                   merge_friedel: bool = True, verbose: bool = True) -> dict:
    """Write the merged amplitudes of each requested split and of its two halves.

    ``args.estimator`` selects the product. ``wls`` (default) is the
    inverse-variance mean on the trained scales, for the full set and for each
    half, turned into amplitudes by French-Wilson. ``posterior`` is the
    Wilson-posterior amplitude; its half files hold the pooled merge, because
    two posterior halves share the prior.
    """
    product = getattr(args, "estimator", None) or P.ESTIMATORS[0]
    also = [e for e in (getattr(args, "also", None) or ()) if e != product]
    hkl = prep.hkl_valid.detach().cpu().numpy().astype(int)
    d = prep.d_spacing.detach().cpu().numpy()
    cell = prep.median_cell.detach().cpu().tolist()
    summary = {}
    for k in wanted:
        cond = k if tr.n_conditions > 1 else None
        everyone = _export_crystals(tr, k) if getattr(tr, "external_heldout", None) is not None else None
        h1, h2 = _split_halves(tr, k, args.seed + 1000000007)
        label = names[k] if k < len(names) else str(k)
        if product == "posterior":
            sets = [(0, everyone), (1, h1), (2, h2)]
            merges = {half: tr.merge(crystals=crystals, condition=cond, estimator="posterior",
                                     tether_readout=half == 0)
                      for half, crystals in sets}
            full = merges[0]
        else:
            full = tr.merge(crystals=everyone, condition=cond, estimator=product, tether_readout=True)
        print(f"\nmerge [{label or 'all'}]: {full['n_crystals']:,} crystals", flush=True)
        P.report_merge(full, hkl=hkl, verbose=verbose)
        if not full["n_crystals"] or not int(full["seen"].sum()):
            print(f"  [{label or 'all'}] nothing written for this split", flush=True)
            continue
        out = args.out
        if len(wanted) > 1:
            out = out.with_name(f"{out.stem}_{label}{out.suffix}")
        work = P.work_mask(hkl, args, verbose=verbose)
        if product == "posterior":
            halves = [tr.merge(crystals=c, condition=cond, estimator="pooled") for c in (h1, h2)]
        else:
            halves = [tr.merge(crystals=c, condition=cond, estimator=product) for c in (h1, h2)]

        def write(stem: str, res: dict, verbose_mtz: bool = False) -> None:
            F, F_sd, keep = _amplitudes(res, hkl, d, spacegroup)
            if keep.any():
                P.write_mtz(out.with_name(f"{stem}{out.suffix}"), hkl, F, F_sd,
                            cell=cell, spacegroup=spacegroup, work=work, keep=keep,
                            merge_friedel=merge_friedel, verbose=verbose_mtz)

        for half, po in enumerate((full, *halves)):
            write(out.stem if half == 0 else f"{out.stem}_half{half}", po, verbose and half == 0)
        rec = {"split": label, "products": [product], "estimator": product,
               "prior": full.get("prior", tr.cfg.posterior_prior)}
        if full.get("n_pooled") is not None:
            rec["wls_pooled"] = full["n_pooled"]
        tether = full.get("tether")
        if tether and tether.get("n_obs"):
            rec["tether"] = tether
            print("  " + P.tether_summary(tether), flush=True)
        if product == "posterior":
            rec["sigF_collapsed"] = P.collapsed_sigma(full, hkl)
            rec["top_edge"] = P.top_edge(full, hkl)
            _report_halves(rec, "split_half_pooled", "pooled, no prior",
                           _split_half_stats(*halves, d, key="I", rank=full["I0"]))
            _report_halves(rec, "split_half_posterior_shared_prior", "posterior, halves share the prior",
                           _split_half_stats(merges[1], merges[2], d, rank=full["I0"]))
        else:
            _report_halves(rec, f"split_half_{product}", f"{product}, no prior",
                           _split_half_stats(*halves, d, key="I", rank=full["I"]))
        for est in also:
            # diagnostic readouts of the same crystals, to compare with the product
            res = tr.merge(crystals=everyone, condition=cond, estimator=est)
            if est == "posterior":
                P.report_merge(res, hkl=hkl, verbose=verbose)
                rec["sigF_collapsed"] = P.collapsed_sigma(res, hkl)
                rec["top_edge"] = P.top_edge(res, hkl)
            write(f"{out.stem}_{est}", res)
            P.write_merge_table(out.with_name(f"{out.stem}_{est}.tsv"), hkl, d, res)
            rec["products"].append(est)
            print(f"  also        {est}: {int(res['seen'].sum()):,} reflections -> "
                  f"{out.stem}_{est}{out.suffix}, {out.stem}_{est}.tsv", flush=True)
        if getattr(args, "also", None):  # the product's own table, to compare the others with
            P.write_merge_table(out.with_name(f"{out.stem}_{product}.tsv"), hkl, d, full)
        summary[label] = rec
    store = getattr(tr, "_store", None)
    if verbose and store is not None:
        print(f"read from {store.path} in this run: {store.store.bytes_read / 2**30:.1f} GiB, "
              f"{store.store.wait_s:.0f} s waited", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary

if __name__ == "__main__":
    main()
