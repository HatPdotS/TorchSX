"""Train one scale network for all supplied states and export raw posterior MTZs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time
import warnings

import numpy as np

from torchsx.cli import _pipeline as P


def _positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive finite number, got {value!r}")
    return number


def _tether_terms(value):
    from torchsx.merging.tether import parse_terms
    try:
        return parse_terms(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r}: {exc}") from None


def _steps(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError(f"Expected a nonnegative update count, got {value!r}")
    return number


def _fraction(value):
    number = float(value)
    if not 0 <= number < 1:
        raise argparse.ArgumentTypeError(f"Expected a fraction in [0, 1), got {value!r}")
    return number


def _build_parser():
    p = argparse.ArgumentParser(prog="torchsx.train", description=__doc__)
    P.add_data_args(p)
    P.add_model_args(p)
    P.add_free_args(p)
    p.add_argument("-o", "--out", type=Path, required=True, help="run directory")
    p.add_argument("--steps", type=_steps, default=400,
                   help="total optimizer updates, balanced across states (default 400)")
    p.add_argument("--miss-frac", type=_fraction, default=None,
                   help="initial fraction of the learnable miss component (a predicted reflection "
                        "that recorded no signal); 0 = off (default 0.03; the merge inherits the "
                        "checkpoint)")
    p.add_argument("--outlier-frac", type=_fraction, default=None,
                   help="initial fraction of the learnable detector-outlier component "
                        "(default 0 = off; the merge inherits the checkpoint)")
    p.add_argument("--detector-noise", choices=("off", "auto", "panel"), default="off",
                   help="learned detector variance A + B (lam + sig_b2), fitted with the network. "
                        "off (default): none; auto: one global B (a gain correction on the integrated "
                        "sums) when preparation replaced the declared gain by its calibration, none "
                        "otherwise; panel: per detector panel, terms from --noise-terms. The merge "
                        "inherits the checkpoint")
    p.add_argument("--noise-terms", default="AB", choices=("AB", "A", "B"),
                   help="--detector-noise panel: which of the terms A, B are fitted (default AB; "
                        "B alone is a per-panel gain correction)")
    # Research controls: they work, but --help does not list them.
    p.add_argument("--noise-lr-mult", type=float, default=30.0, help=argparse.SUPPRESS)
    # Fold transformer: K >= 2 masked runs per crystal (default 5); the merge
    # inherits the checkpoint, as for every setting of the network.
    p.add_argument("--context-folds", type=int, default=None, help=argparse.SUPPRESS)
    # Fold transformer depth: self-attention blocks over each run (default 2) and
    # cross-attention blocks from each masked observation onto its run (default 1).
    p.add_argument("--ctx-layers", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--query-layers", type=int, default=None, help=argparse.SUPPRESS)
    # Fold transformer: correct each run's geometry (a crystal rotation and an energy
    # shift, predicted from its context) and recompute the excitation errors (default
    # on; off with --no-refine-geometry).
    p.add_argument("--refine-geometry", action=argparse.BooleanOptionalAction, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--pseudo-translation", choices=("off", "auto"), default="auto",
                   help="translational NCS or a lattice translocation defect: auto (default) finds the "
                        "structure's translations in the Patterson function of the data's plain means.  With "
                        "a dominant one, the Wilson prior's mean follows the modulation of the intensities by "
                        "the phase h.t, the scale model reads each reflection's phase, and the tether holds "
                        "every class of reflections at its sum of spots.  Without one the model is unchanged.  "
                        "off skips the search.  The merge inherits the checkpoint")
    p.add_argument("--tether", type=_tether_terms, default=None, metavar="TERMS",
                   help="directions the likelihood leaves to the model, held at the raw counts, the sum of "
                        "each reflection's spots: gauge (the rocking-curve integral of the scale is the same "
                        "at every resolution), or none, which pins the scale's smooth gauge by projection "
                        "instead.  Default: gauge.  The merge inherits the checkpoint")
    # Weight of the tether's penalty (nats^2 per observation, default 1).
    p.add_argument("--tether-weight", type=_positive, default=1.0, help=argparse.SUPPRESS)
    p.add_argument("--redundancy", type=_positive, default=100,
                   help="target observations per reflection in a batch (default 100)")
    p.add_argument("--no-merge", action="store_true", help="save checkpoints; merge separately")
    # Encoder width (default 128).
    p.add_argument("--d-model", type=int, default=None, help=argparse.SUPPRESS)
    # Held-out evaluation and checkpoint every this many updates per state.
    p.add_argument("--eval-every", type=int, default=100, help=argparse.SUPPRESS)
    # Stop after this many evaluations without a new held-out minimum (0 = never).
    p.add_argument("--patience", type=int, default=0, help=argparse.SUPPRESS)
    # Patterns of the first split scored as the held-out set: never trained on, never
    # merged into the outputs; one file across runs puts their held-out NLL on the
    # same crystals.
    p.add_argument("--heldout-stream", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--warmup", type=int, default=None,
                   help="learning-rate warm-up in updates per state (default 25)")
    p.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant",
                   help="learning rate after the warm-up: constant (default) or a cosine "
                        "decay to zero at the last update")
    p.add_argument("--average-last", type=int, default=5, metavar="N",
                   help="when the last N updates per state oscillate (each gradient points back against "
                        "the step before it), validate, checkpoint and merge the mean of the network after "
                        "each of them (default 5; 0 = the last update alone)")
    return p


def _batch(tr, step=None):
    """Sample complete crystals from one state, alternating states per update.

    ``step`` (default: the trainer's next update) fixes the draw, so a coming
    batch can be named, and read ahead, before its update runs.
    """
    import torch

    step = tr.step_i if step is None else int(step)
    k = step % tr.n_conditions
    pool = tr.train_crystals[tr.condition[tr.train_crystals] == k]
    if not pool.numel():
        raise ValueError(f"State {k} has no training crystals after observation filtering")
    g = torch.Generator().manual_seed(tr.cfg.seed + 1000003 * (step // tr.n_conditions))
    perm = pool[torch.randperm(len(pool), generator=g)]
    counts = tr.index.counts[perm].cumsum(0)
    requested = tr.cfg.max_obs_per_batch
    if getattr(tr.cfg, "target_redundancy", None) is not None:
        requested = int(round(tr.cfg.target_redundancy * tr.prep.n_valid))
    available = int(counts[-1])
    if available < requested:
        warned = getattr(tr, "_small_batch_warned_states", set())
        if k not in warned:
            warnings.warn(
                f"State {k} has {available:,} training observations, fewer than the "
                f"requested batch size of {requested:,}; using all {len(pool):,} "
                "training crystals in one batch. Validation crystals remain held out.",
                UserWarning, stacklevel=2,
            )
            warned.add(k)
            tr._small_batch_warned_states = warned
        return k, tr.index.rows(perm)
    n = max(1, int(torch.searchsorted(counts, tr.cfg.max_obs_per_batch, right=True)))
    chosen = perm[:n]
    return k, tr.index.rows(chosen)


def _save_current(tr, path):
    """Save the final/current weights without changing the best-validation snapshot."""
    keys = ("_best_state", "_best_obj_state", "_best_wilson", "best_step", "best_heldout")
    previous = {k: getattr(tr, k, None) for k in keys}
    try:
        tr._best_state = tr._best_obj_state = tr._best_wilson = None
        tr.best_step, tr.best_heldout = tr.step_i, float("nan")
        tr.save_best(path)
    finally:
        for key, value in previous.items():
            setattr(tr, key, value)


def attach_heldout_stream(bundle, name: str = "heldout"):
    """Fold the ``name`` split into the first one and return its crystal indices.

    The held-out stream is loaded as one more split so every input shares one
    reflection index; its crystals then become crystals of the first split
    (the held-out condition) and the split itself disappears.
    """
    names = list(bundle.split_names)
    if name not in names:
        raise ValueError(f"no {name!r} split to attach")
    h = names.index(name)
    oi = bundle.prep.obs_info
    lab = oi.delay_bin_idx.clone()
    ext = lab == h
    ids = ext.nonzero(as_tuple=True)[0]
    lab[ext] = 0
    lab[lab > h] -= 1
    oi.delay_bin_idx = lab
    if getattr(oi, "unique_timepoints", None) is not None:
        keep = [i for i in range(len(names)) if i != h]
        oi.unique_timepoints = oi.unique_timepoints[keep]
    bundle.split_names = [n for i, n in enumerate(names) if i != h]
    return ids


def find_used_translations(prep, spacegroup) -> tuple[list[dict], str]:
    """The data's pseudo-translations (:mod:`torchsx.merging.scale.pseudo_translation`),
    every Patterson peak reported; the ones strong enough to use, and why."""
    from torchsx.merging.scale.pseudo_translation import find_pseudo_translations, select_translations
    peaks = find_pseudo_translations(prep, spacegroup)
    used, why = select_translations(peaks)
    for i, p in enumerate(peaks):
        print(f"pseudo-translation: #{i} t ({p['t'][0]:.4f} {p['t'][1]:.4f} {p['t'][2]:.4f}) "
              f"amplitude {p['amplitude']:.3f} significance {p['significance']:.1f}"
              + ("" if p["harmonic_of"] is None else f" [harmonic {p['order']} of #{p['harmonic_of']}]")
              + (" (used)" if p in used else ""), flush=True)
    return used, why


def phase_classes(prep, spacegroup, used, why):
    """The classes of reflections the ``used`` translations make
    (:class:`~torchsx.merging.scale.pseudo_translation.PhaseClasses`)."""
    from torchsx.merging.scale.pseudo_translation import PhaseClasses
    classes = PhaseClasses.from_prep(prep, [p["t"] for p in used], spacegroup)
    cells = np.bincount(classes.cells(prep.hkl_valid.long().cpu().numpy()), minlength=classes.n_cells)
    print(f"pseudo-translation: {why}; phase features phase_c0..{classes.n_translations - 1}, tether phase "
          f"term over {classes.n_cells} cells x T0..T{classes.degree} in resolution (reflections per cell "
          f"{int(cells.min())}..{int(cells.max())})", flush=True)
    return classes


def install_structured_prior(tr, prep, spacegroup, used) -> None:
    """Give the Wilson prior the symmetry factor and the modulation of the mean intensity by the
    ``used`` translations (:mod:`torchsx.merging.priors.modulation`)."""
    from torchsx.merging.priors.modulation import PhaseModulation
    mod = PhaseModulation.from_prep(prep, [p["t"] for p in used], spacegroup)
    tr.set_intensity_modulation(mod)
    for row in mod.summary():
        print(f"structured prior: t {row['t']} amplitude {row['amplitude']} at d {row['d']} A", flush=True)
    hkl = prep.hkl_valid.long().cpu().numpy()
    g = np.exp(mod.log_factor(hkl, (1.0 / prep.d_spacing.double() ** 2).cpu().numpy()))
    print(f"structured prior: modulation q01 {np.quantile(g, 0.01):.3f} median {np.median(g):.3f} "
          f"q99 {np.quantile(g, 0.99):.3f} over {len(g):,} reflections", flush=True)


def run_updates(tr, steps, *, eval_every: int = 100, patience: int = 0, probe=None):
    """Run a fixed update budget, with the evaluation cadence per state;
    with ``patience`` > 0, stop after that many evaluations without a new
    held-out minimum.  ``probe`` (:class:`~torchsx.cli._pipeline.ResourceProbe`)
    records every update's time.  When the last ``average_last`` updates
    oscillate, each evaluation and checkpoint holds their mean
    (``tr.averaged()``), and so does the trainer when the budget ends."""
    evaluate_every = max(1, int(eval_every)) * tr.n_conditions
    stale = 0
    updates = [0] * tr.n_conditions
    has_validation = bool(tr.heldout_crystals.numel())
    upcoming = _batch(tr) if steps else None
    if upcoming is not None:
        tr.prefetch(upcoming[1])
    if has_validation:
        tr.evaluate(tr.heldout_crystals, phase="heldout_init")
    for i in range(steps):
        k, rows = upcoming
        if i + 1 < steps:
            # read the next update's observations while this one computes
            upcoming = _batch(tr, tr.step_i + 1)
            tr.prefetch(upcoming[1])
        if has_validation and ((tr.step_i + 1) % evaluate_every == 0 or i + 1 == steps):
            # and the validation set's first batch, when a validation follows
            first = next(tr.index.batches(tr.cfg.max_obs_per_batch, crystals=tr.heldout_crystals,
                                          shuffle=False), None)
            if first is not None:
                tr.prefetch(first[1])
        rec = tr.step(rows.to(tr.device))
        if rec.get("step_skipped") or not rec.get("n_used"):
            raise FloatingPointError(f"Update {tr.step_i} for state {k} was unusable; see metrics.jsonl")
        updates[k] += 1
        if probe is not None:
            probe.update(rec["sec"])
        if tr.step_i % 25 == 0 or sum(updates) == steps:
            print(f"update {tr.step_i}/{steps}: per-state {updates}, "
                  f"nll/obs {rec['nll_per_obs']:.5f}"
                  + (f", tether gauge {rec['tether_gauge']:.4f} nats" if "tether_gauge" in rec else ""),
                  flush=True)
        if tr.step_i % evaluate_every == 0 or sum(updates) == steps:
            with tr.averaged():
                if has_validation:
                    result = tr.evaluate(tr.heldout_crystals)
                    a, note = tr.averaging, ""
                    if a is not None:
                        held = f"mean of the last {a['window']} updates" if a["averaged"] else "latest update"
                        note = f"; {held} (step reversal {a['step_reversal']:+.2f})"
                    print(f"  validation state 0: nll/obs {result['nll_per_obs']:.5f}{note}", flush=True)
                    stale = 0 if tr.best_step == tr.step_i else stale + 1
                tr.save_best()
                _save_current(tr, Path(tr.cfg.outdir) / "last.pt")
            if patience > 0 and stale >= patience:
                print(f"early stop at update {tr.step_i}: {stale} evaluations without a new "
                      f"held-out minimum (best at {tr.best_step})", flush=True)
                break
    if not steps:
        tr.save_best()
        _save_current(tr, Path(tr.cfg.outdir) / "last.pt")
    tr.load_average()
    return updates


def main(argv=None):
    args = _build_parser().parse_args(argv)
    import torch

    P.configure_torch()
    from torchsx.nn.train_amortized import AmortizedTrainer, TrainConfig
    from torchsx.cli.merge_nn import merge_products

    args.seed = 0 if args.seed is None else args.seed
    random.seed(args.seed)
    np.random.seed(args.seed % 2**32)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    probe = P.ResourceProbe(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    probe.begin("prepare")
    if args.heldout_stream is not None:
        if not args.split:
            raise SystemExit("--heldout-stream needs named inputs (--split NAME=PATH)")
        args.split = list(args.split) + [f"heldout={args.heldout_stream}"]
    bundle = P.load_prep(args)
    heldout_ids = attach_heldout_stream(bundle) if args.heldout_stream is not None else None
    used, why = ([], "")
    if args.pseudo_translation == "auto":
        used, why = find_used_translations(bundle.prep, bundle.spacegroup)
    classes, tether = None, P.tether_terms(args)
    if used:
        if not tether:
            raise SystemExit("the data have a pseudo-translation, and the tether holds its classes of "
                             "reflections at their sums of spots: drop --tether none, or pass "
                             "--pseudo-translation off")
        classes = phase_classes(bundle.prep, bundle.spacegroup, used, why)
        tether = tuple(tether) + ("phase",)
    elif args.pseudo_translation == "auto":
        print(f"pseudo-translation: {why}; one intensity population and one profile for every "
              "reflection", flush=True)
    prep, params, scale, obj, settings = P.build_stack(bundle, args, phase_classes=classes)
    n_states = max(1, len(bundle.split_names))
    cfg = TrainConfig(
        target_redundancy=args.redundancy, lr=1e-3,
        warmup_steps=(25 if args.warmup is None else args.warmup) * n_states,
        lr_schedule=args.lr_schedule, total_steps=args.steps,
        average_last=max(0, args.average_last) * n_states,
        grad_clip=20.0, log_every=25,
        posterior_k_coarse=31, posterior_m_fine=129, posterior_n_sigma=6.0,
        posterior_max_halfwidth=settings["max_halfwidth"],
        locate_windows=(8.0, 1.0, 0.125), posterior_prior=settings["prior"],
        wilson_degree=settings["wilson_degree"],
        posterior_checkpoint_nodes=True,
        stream_batches=True, heldout_frac=0.05,
        device=settings["device"], outdir=str(args.out), seed=args.seed,
        heldout_crystal_ids=None if heldout_ids is None else heldout_ids.tolist(),
        noise_lr_mult=args.noise_lr_mult,
        tether=tether, lam_tether=args.tether_weight,
    )
    tr = AmortizedTrainer(prep, scale, obj, params, config=cfg)
    try:
        tr.set_reciprocal_basis(settings["basis"])
        if classes is not None:
            tr.set_phase_classes(classes)
            install_structured_prior(tr, prep, bundle.spacegroup, used)
        tr.split_names = bundle.split_names
        config = asdict(cfg)
        if heldout_ids is not None:
            config["heldout_crystal_ids"] = f"{heldout_ids.numel()} crystals of {args.heldout_stream}"
        recipe = dict(version=1, steps=args.steps, config=config,
                      checkpoint_selection="minimum validation NLL in state 0; last.pt retains endpoint"
                      + (f"; each the mean of the last {cfg.average_last} updates when they oscillate"
                         if cfg.average_last > 1 else ""),
                      validation="5% of condition-zero crystals; all observations retained for export",
                      evaluate_every=args.eval_every * n_states,
                      readout={k: v for k, v in settings.items() if k != "basis"})
        tr.data_recipe = dict(bundle.recipe, training=recipe)
        (args.out / "run.json").write_text(json.dumps(tr.data_recipe, indent=2) + "\n")
        print(f"train {tr.train_crystals.numel():,}, validation {tr.heldout_crystals.numel():,} crystals; "
              f"{args.steps} total updates across {tr.n_conditions} states", flush=True)
        probe.begin("train")
        updates = run_updates(tr, args.steps, eval_every=args.eval_every, patience=args.patience,
                              probe=probe)
        probe.end()
        summary_path = args.out / "training_summary.json"
        summary = dict(steps=args.steps, updates_per_state=updates, best_step=tr.best_step,
                       elapsed_seconds=time.time()-t0, resources=probe.summary())
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        if args.no_merge:
            print(probe.report(), flush=True)
            return
        probe.begin("readout")
        tr.restore_best()
        tr.scale.eval()
        tr.opt = None
        tr.cfg.max_obs_per_batch = P.READOUT_MAX_OBS
        if str(tr.device).startswith("cuda"):
            torch.cuda.empty_cache()
        for target in tr.targets:
            target.set_grid(settings["k_coarse"], settings["m_fine"], settings["n_sigma"])
            target.max_halfwidth = settings["max_halfwidth"]
        args.out = args.out / "merged.mtz"
        merge_products(tr, prep, args, bundle.split_names or [""], list(range(tr.n_conditions)),
                       spacegroup=bundle.spacegroup, merge_friedel=bundle.merge_friedel)
        probe.end()
        summary.update(elapsed_seconds=time.time()-t0, resources=probe.summary())
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    finally:
        tr.close()
    print(probe.report(), flush=True)
    print(f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
