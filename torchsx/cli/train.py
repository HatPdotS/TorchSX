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
import torch

from torchsx.cli import _pipeline as P


def _positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive finite number, got {value!r}")
    return number


def _steps(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError(f"Expected a nonnegative update count, got {value!r}")
    return number


def _decay(value):
    number = float(value)
    if not 0 <= number < 1:
        raise argparse.ArgumentTypeError("Expected a finite decay in [0, 1)")
    return number


def _build_parser():
    p = argparse.ArgumentParser(prog="torchsx-train", description=__doc__)
    P.add_data_args(p)
    P.add_model_args(p)
    P.add_free_args(p)
    p.add_argument("-o", "--out", type=Path, required=True, help="run directory")
    p.add_argument("--steps", type=_steps, default=800,
                   help="total optimizer updates, balanced across states (default 800)")
    p.add_argument("--outlier-frac", type=float, default=None,
                   help="initial fraction of the learnable detector-outlier component "
                        "(default 0 = off; the merge inherits the checkpoint)")
    p.add_argument("--peer-attention", type=int, default=None,
                   help="cross-attention layers from each observation onto the other observations "
                        "of its crystal (default 2; 0 = mean pool only); the merge inherits the checkpoint")
    p.add_argument("--peer-dim", default=None,
                   help="width inside each peer-attention layer (default: the encoder width), or a "
                        "comma-separated width per layer for a tapered stack, e.g. 128,128,32,32")
    p.add_argument("--context-folds", type=int, default=None,
                   help="crystal context pooling: K >= 2 folds (context from the other K-1 folds of "
                        "the crystal, default 2), or 0 = leave-one-out (all other reflections, "
                        "every context block per observation); the merge inherits the checkpoint")
    p.add_argument("--redundancy", type=_positive, default=100,
                   help="target observations per reflection in a batch (default 100)")
    p.add_argument("--no-merge", action="store_true", help="save checkpoints; merge separately")
    p.add_argument("--reference-estimator", choices=("wls", "posterior"), default="wls",
                   help="estimate averaged into the running reference (default wls)")
    p.add_argument("--reference-ema-decay", type=_decay, default=0.99,
                   help="exponential decay of the running reference (default 0.99)")
    p.add_argument("--reference-bias-correction", action="store_true",
                   help="normalise the running reference by its accumulated weight, so the "
                        "unscaled starting merge drops out at a reflection's first update")
    p.add_argument("--warmup", type=int, default=None,
                   help="learning-rate warm-up in updates per state (default 100)")
    p.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant",
                   help="learning rate after the warm-up: constant (default) or a cosine "
                        "decay to zero at the last update")
    return p


def _batch(tr):
    """Sample complete crystals from one state, alternating states per update."""
    k = tr.step_i % tr.n_conditions
    pool = tr.train_crystals[tr.condition[tr.train_crystals] == k]
    if not pool.numel():
        raise ValueError(f"State {k} has no training crystals after observation filtering")
    g = torch.Generator().manual_seed(tr.cfg.seed + 1000003 * (tr.step_i // tr.n_conditions))
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
    keys = ("_best_state", "_best_obj_state", "_best_context", "best_step", "best_heldout")
    previous = {k: getattr(tr, k, None) for k in keys}
    try:
        tr._best_state = tr._best_obj_state = tr._best_context = None
        tr.best_step, tr.best_heldout = tr.step_i, float("nan")
        tr.save_best(path)
    finally:
        for key, value in previous.items():
            setattr(tr, key, value)


def run_updates(tr, steps):
    """Run a fixed update budget, with refresh/evaluation cadence per state."""
    refresh_every, evaluate_every = 25 * tr.n_conditions, 100 * tr.n_conditions
    updates = [0] * tr.n_conditions
    has_validation = bool(tr.heldout_crystals.numel())
    if has_validation:
        tr.evaluate(tr.heldout_crystals, phase="heldout_init")
    for _ in range(steps):
        if tr.step_i % refresh_every == 0:
            tr._refresh_feature_reference()
            tr._refresh_wilson_sigma()
        k, rows = _batch(tr)
        rec = tr.step(rows.to(tr.device))
        if rec.get("step_skipped") or not rec.get("n_used"):
            raise FloatingPointError(f"Update {tr.step_i} for state {k} was unusable; see metrics.jsonl")
        updates[k] += 1
        if tr.step_i % 25 == 0 or sum(updates) == steps:
            print(f"update {tr.step_i}/{steps}: per-state {updates}, "
                  f"nll/obs {rec['nll_per_obs']:.5f}", flush=True)
        if tr.step_i % evaluate_every == 0 or sum(updates) == steps:
            # Checkpoint the exact context used at evaluation/readout.
            tr._refresh_feature_reference()
            tr._refresh_wilson_sigma()
            if has_validation:
                result = tr.evaluate(tr.heldout_crystals)
                print(f"  validation state 0: nll/obs {result['nll_per_obs']:.5f}", flush=True)
            tr.save_best()
            _save_current(tr, Path(tr.cfg.outdir) / "last.pt")
    if not steps:
        tr.save_best()
        _save_current(tr, Path(tr.cfg.outdir) / "last.pt")
    return updates


def main(argv=None):
    args = _build_parser().parse_args(argv)
    # TF32 for the encoder's float32 matmuls; the likelihood and posterior
    # integrals are elementwise and Triton code, which this does not touch.
    torch.set_float32_matmul_precision("high")
    from torchsx.nn.train_amortized import AmortizedTrainer, TrainConfig
    from torchsx.cli.merge_nn import merge_products

    args.seed = 0 if args.seed is None else args.seed
    random.seed(args.seed)
    np.random.seed(args.seed % 2**32)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    bundle = P.load_prep(args)
    prep, params, scale, obj, settings = P.build_stack(bundle, args)
    n_states = max(1, len(bundle.split_names))
    cfg = TrainConfig(
        target_redundancy=args.redundancy, epochs=0, lr=1e-3,
        lam0=10.0, lam_halflife_steps=400 * n_states,
        warmup_steps=(100 if args.warmup is None else args.warmup) * n_states,
        lr_schedule=args.lr_schedule, total_steps=args.steps,
        reference_bias_correction=args.reference_bias_correction,
        grad_clip=20.0, log_every=25, merge_target="posterior", loss="evidence",
        posterior_k_coarse=31, posterior_m_fine=129, posterior_n_sigma=6.0,
        posterior_max_halfwidth=settings["max_halfwidth"],
        locate_windows=(8.0, 1.0, 0.125), posterior_prior=settings["prior"],
        reference_estimator=args.reference_estimator,
        reference_ema_decay=args.reference_ema_decay,
        wilson_basis=settings["wilson_basis"], wilson_degree=settings["wilson_degree"],
        posterior_checkpoint_nodes=True, encoder_chunk_crystals=400,
        stream_batches=True, split_half_every=0, heldout_frac=0.05,
        device=settings["device"], outdir=str(args.out), seed=args.seed,
    )
    tr = AmortizedTrainer(prep, scale, obj, params, config=cfg)
    try:
        if settings.get("basis") is not None:
            tr.set_reciprocal_basis(settings["basis"], smooth_prior=settings["wilson_basis"] == "smooth")
        tr.split_names = bundle.split_names
        recipe = dict(version=1, steps=args.steps, config=asdict(cfg),
                      checkpoint_selection="minimum validation NLL in state 0; last.pt retains endpoint",
                      validation="5% of condition-zero crystals; all observations retained for export",
                      refresh_every=25 * n_states, evaluate_every=100 * n_states,
                      readout={k: v for k, v in settings.items() if k != "basis"})
        tr.data_recipe = dict(bundle.recipe, training=recipe)
        (args.out / "run.json").write_text(json.dumps(tr.data_recipe, indent=2) + "\n")
        print(f"train {tr.train_crystals.numel():,}, validation {tr.heldout_crystals.numel():,} crystals; "
              f"{args.steps} total updates across {tr.n_conditions} states", flush=True)
        updates = run_updates(tr, args.steps)
        (args.out / "training_summary.json").write_text(json.dumps(dict(
            steps=args.steps, updates_per_state=updates, best_step=tr.best_step,
            elapsed_seconds=time.time()-t0), indent=2) + "\n")
        if args.no_merge:
            return
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
    finally:
        tr.close()
    print(f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
