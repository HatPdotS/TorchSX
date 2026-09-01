"""Freeze one REAL training batch so the inner integral can be optimised alone.

The quantity under optimisation is the posterior accumulator

    S[h, c] = sum_{j: hkl_j = h} log INT p_pois(n_j | S + bg_j)
                                     . GG(log S - v[h,c] - base_j; a_j, b_j) d log S

with ``base_j = log K_j - log E[e^eps]_j``.  Everything it needs is seven
per-observation vectors plus the outer node grid, so it can be lifted out of
the trainer entirely and iterated on in seconds instead of 20-minute epochs.

It must be a real batch, not a synthetic one: the cost and the accuracy are
both driven by the *spread* of (n_ph, bg, alpha, beta) -- a strong reflection
needs 30x the node density of a weak one, and this data set is 32.5 % negative
with median I/sigma 0.847, so a uniform synthetic batch would flatter any
candidate that places nodes badly.

Usage (slurm, gpu partition)::

    python dump_batch.py --steps 200 --out batch.pt
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from torchsx.stream import SFXStream
from torchsx.merging.config import MergeConfig
from torchsx.merging.hkl.filter import filter_resolution
from torchsx.merging.observations import prepare_observations
from torchsx.merging.parameters import Parameters
from torchsx.merging.scatter_shape import FreePerHKLShape
from torchsx.merging.likelihood import Poisson
from torchsx.merging.quadrature.fixed_log_grid import FixedLogGrid
from torchsx.merging.scale import AmortizedScale
from torchsx.nn.objective import MarginalNLL
from torchsx.nn.train_amortized import AmortizedTrainer, TrainConfig

ROOT = Path("/das/work/units/LBR-FEL/p17489/Peter/2025-01-31_SwissFEL_SFX/"
            "svd_time_signal_test/abismal")
DMIN = 0.9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200,
                    help="training steps to run first, so (K, w, beta) are "
                         "realistic rather than at their zero-init values")
    ap.add_argument("--obs", type=int, default=1_500_000)
    ap.add_argument("--k-coarse", type=int, default=21)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "batch.pt")
    args = ap.parse_args()

    t0 = time.time()
    stream = SFXStream.from_stream(
        ROOT / "streams" / "dark.stream", spacegroup="P 1",
        cache=ROOT / "bench" / "dark.sfx", n_workers=8, merge_friedel=True,
    )
    stream = SFXStream(filter_resolution(stream._core, d_min=DMIN, d_max=999.0))
    prep = prepare_observations(stream, MergeConfig(gain=1.0))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    prep = prep.to(dev)
    print(f"[{time.time()-t0:6.1f}s] {prep.n_obs_total:,} obs  "
          f"{prep.n_valid:,} hkl", flush=True)

    params = Parameters(
        log_I0_init=prep.mu_init.clone(),
        scatter_shape=FreePerHKLShape(prep.alpha_init.log(), prep.beta_perhkl.log()),
    ).to(dev)
    scale = AmortizedScale.from_prep(
        prep, d_model=128, n_layers=5, include_hkl=False,
        predict_shape=False, beta_init=1.0,
    )
    obj = MarginalNLL(
        FixedLogGrid(S_min=math.exp(-34.5), S_max=math.exp(34.5),
                     n_points=2001).to(dev),
        Poisson(),
    )
    cfg = TrainConfig(
        max_obs_per_batch=args.obs, epochs=1, lr=1e-3,
        lam0=10.0, lam_halflife_steps=400, lam_spread=0.0, log_every=10**9,
        warmup_steps=100, grad_clip=20.0,
        merge_target="posterior", loss="evidence",
        posterior_k_coarse=args.k_coarse, posterior_m_fine=129,
        posterior_n_sigma=6.0, posterior_prior="wilson",
        heldout_frac=0.05, device=dev, outdir=str(args.out.parent),
    )
    tr = AmortizedTrainer(prep, scale, obj, params, config=cfg)

    done = 0
    while done < args.steps:
        for _, rows in tr.index.batches(cfg.max_obs_per_batch, generator=tr.gen,
                                        crystals=tr.train_crystals):
            tr.step(rows.to(dev))
            done += 1
            if done >= args.steps:
                break
        tr._refresh_wilson_sigma()
    print(f"[{time.time()-t0:6.1f}s] {done} warm-up steps done", flush=True)

    # One batch, exactly as _forward() would see it.
    _, rows = next(iter(tr.index.batches(cfg.max_obs_per_batch, generator=tr.gen,
                                         crystals=tr.train_crystals)))
    rows = rows.to(dev)
    scale.eval()
    with torch.no_grad():
        meta = prep.obs_info.gather(rows)
        pred = scale.prime(meta)
        n_ph, bg = tr.n_ph[rows], prep.bg[rows]
        # Same grid placement the trainer uses: measure the curve, then sit on it.
        centre, sd = tr.target.locate(
            obj, scale.scatter, params, pred, n_ph, bg, meta.hkl_idx, meta,
        )
        v_nodes = centre[:, None] + sd[:, None] * tr.target.u_coarse[None, :]

    torch.save({
        "base": (pred.log_K - pred.log_e_s).cpu(),
        "n_ph": n_ph.cpu(), "bg": bg.cpu(),
        "log_alpha": pred.log_alpha.cpu(), "log_beta": pred.beta.log().cpu(),
        "hkl_id": meta.hkl_idx.cpu(), "v_nodes": v_nodes.cpu(),
        "n_hkl": int(prep.n_valid), "steps": done,
        "grid": {"lo": -34.5, "hi": 34.5, "n_points": 2001},
    }, args.out)

    a = pred.log_alpha.exp()
    print(f"\nwrote {args.out}  ({args.out.stat().st_size/1e6:.0f} MB)")
    print(f"  M {rows.numel():,}   n_hkl {prep.n_valid:,}   C {v_nodes.shape[1]}")
    print(f"  n_ph    q01 {n_ph.quantile(.01):9.2f}  med {n_ph.median():9.2f}"
          f"  q99 {n_ph.quantile(.99):9.2f}  max {n_ph.max():9.2f}")
    print(f"  bg      med {bg.median():9.2f}")
    print(f"  alpha   q01 {a.quantile(.01):9.3f}  med {a.median():9.3f}"
          f"  q99 {a.quantile(.99):9.3f}")
    print(f"  beta    med {pred.beta.median():9.3f}")
    print(f"  base    sd  {(pred.log_K - pred.log_e_s).std():9.3f}")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
