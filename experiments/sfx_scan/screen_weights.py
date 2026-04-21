#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u


#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -p gpu-day
#SBATCH -o sfx_scan/screen_weights.out

"""Screen weighting parameters: L-BFGS from calibration with EM reweighting."""

import itertools
from pathlib import Path

import torch

from torchsx.stream import SFXStream
from torchsx.scaling import AbsoluteScale, BScale, prepare_scaler_inputs
from torchsx.target._kl import KLScalingTarget
from torchsx.merge._stats import match_halves, cc_half, r_split

STREAM_DIR = Path('/das/work/p17/p17490/Peter/torchsx/sfx_scan') / "streams"
DELAYS = [-91, -136, -182, -227, -273]
SPACEGROUP = "P 1"
MERGE_FRIEDEL = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
n_workers = 16

CENTER_LAM = 10.0
N_CYCLES = 10  # 10 L-BFGS steps, 1 per weight update


def run_lbfgs(inp, device, n_cycles=N_CYCLES, w_min=0.1, cutoff=2.0, steepness=2.0,
              use_weights=True):
    """Calibrate + L-BFGS with EM reweighting. Returns merged dict + final loss."""
    abs_scale = AbsoluteScale(inp.n_crystals).to(device)
    b_scale = BScale(inp.n_crystals).to(device)
    target = KLScalingTarget(inp.n_unique, inp.n_shells, lam=1.0).to(device)

    cid = inp.crystal_id.long()

    with torch.no_grad():
        total_scale_init = torch.ones_like(inp.I_obs)
    target.calibrate(
        I_obs=inp.I_obs, total_scale=total_scale_init,
        hkl_id=inp.hkl_id, shell=inp.shell, crystal_id=inp.crystal_id,
    )

    params = (
        list(abs_scale.parameters())
        + list(b_scale.parameters())
        + list(target.parameters())
    )
    optimizer = torch.optim.LBFGS(
        params, lr=0.5, max_iter=40, line_search_fn="strong_wolfe",
    )

    weight = None
    for cycle in range(n_cycles):
        # Compute weights from current model
        if use_weights:
            with torch.no_grad():
                ts = torch.exp(abs_scale.log_G[cid] - 2.0 * b_scale.B[cid] * inp.s_sq)
            weight = target.compute_crystal_weights(
                inp.I_obs, ts, inp.hkl_id, inp.crystal_id,
                w_min=w_min, cutoff=cutoff, steepness=steepness,
            )

        def closure():
            optimizer.zero_grad()
            log_G = abs_scale.log_G[cid]
            log_B = -2.0 * b_scale.B[cid] * inp.s_sq
            ts = torch.exp(log_G + log_B)
            loss = target(inp.I_obs, ts, inp.hkl_id, weight=weight)
            loss = loss + CENTER_LAM * abs_scale.log_G.mean() ** 2
            loss.backward()
            return loss

        loss = optimizer.step(closure)

    merged = target.to_dict()
    merged["hkl"] = inp.unique_hkl

    # Collect weight stats
    w_info = ""
    if weight is not None:
        mean_w = weight.mean().item()
        n_down = (weight < 0.5 * mean_w).sum().item()
        w_info = f" w=[{weight.min().item():.3f},{weight.max().item():.3f}] down={100*n_down/weight.numel():.1f}%"

    return {k: v.cpu() for k, v in merged.items()}, loss.item(), w_info


# -- 1. Load and concatenate --
streams = []
for d in DELAYS:
    stem = STREAM_DIR / f"delay_{d}.stream"
    cache = stem.with_suffix(".sfx")
    print(f"Loading {stem.name} ...")
    s = SFXStream.from_stream(stem, spacegroup=SPACEGROUP, cache=cache,
                               n_workers=n_workers, merge_friedel=MERGE_FRIEDEL)
    print(f"  {s.n_crystals} crystals, {s.n_observations} obs")
    streams.append(s)

combined = SFXStream.concat(streams)
print(f"\nCombined: {combined.n_crystals} crystals, {combined.n_observations} obs")

# -- 2. Split halves --
half1, half2 = combined.split_random(n_parts=2, seed=42)
print(f"Half 1: {half1.n_crystals} crystals, {half1.n_observations} obs")
print(f"Half 2: {half2.n_crystals} crystals, {half2.n_observations} obs")

median_cell = combined._core.cell.median(dim=0).values

inp1 = prepare_scaler_inputs(half1._core, n_shells=20).to(device)
inp2 = prepare_scaler_inputs(half2._core, n_shells=20).to(device)

# -- 3. Baseline (no weighting) --
print("\n=== Baseline (no weights, 10 L-BFGS steps) ===")
merged1_base, loss1_base, _ = run_lbfgs(inp1, device, use_weights=False)
merged2_base, loss2_base, _ = run_lbfgs(inp2, device, use_weights=False)

matched = match_halves(merged1_base["hkl"], merged1_base["F"],
                       merged2_base["hkl"], merged2_base["F"],
                       median_cell, d_min=0.9, d_max=10.0)
cc_base = cc_half(matched)
r_base = r_split(matched)
print(f"  Baseline: CC1/2 = {cc_base:.4f}  R_split = {r_base:.4f}  loss = {(loss1_base+loss2_base)/2:.4f}")
print(f"  F range: [{merged1_base['F'].min():.2f}, {merged1_base['F'].max():.2f}]")

# -- 4. Screen weight parameters --
W_MINS = [0.01, 0.05, 0.1, 0.2]
CUTOFFS = [0.5, 1.0, 1.5, 2.0, 3.0]
STEEPNESSES = [1.0, 2.0, 4.0]

results = []
n_combos = len(W_MINS) * len(CUTOFFS) * len(STEEPNESSES)
print(f"\n=== Screening {n_combos} combinations ===\n")
print(f"{'#':>3} {'w_min':>6} {'cutoff':>7} {'steep':>6} {'CC1/2':>8} {'R_split':>8} {'dCC':>7} {'loss':>8} {'weights':>30}")
print("-" * 95)

for i, (w_min, cutoff, steepness) in enumerate(itertools.product(W_MINS, CUTOFFS, STEEPNESSES), 1):
    try:
        merged1, loss1, w1 = run_lbfgs(inp1, device, w_min=w_min, cutoff=cutoff, steepness=steepness)
        merged2, loss2, w2 = run_lbfgs(inp2, device, w_min=w_min, cutoff=cutoff, steepness=steepness)

        matched = match_halves(merged1["hkl"], merged1["F"],
                               merged2["hkl"], merged2["F"],
                               median_cell, d_min=0.9, d_max=10.0)
        cc = cc_half(matched)
        r = r_split(matched)
        delta_cc = cc - cc_base
        avg_loss = (loss1 + loss2) / 2

        results.append((w_min, cutoff, steepness, cc, r, delta_cc, avg_loss))
        sign = "+" if delta_cc >= 0 else ""
        print(f"{i:3d} {w_min:6.2f} {cutoff:7.1f} {steepness:6.1f} {cc:8.4f} {r:8.4f} {sign}{delta_cc:6.4f} {avg_loss:8.4f}{w1}")

    except Exception as e:
        print(f"{i:3d} {w_min:6.2f} {cutoff:7.1f} {steepness:6.1f}  FAILED: {e}")

    # Free GPU cache between runs
    torch.cuda.empty_cache()

# -- 5. Summary --
if results:
    results.sort(key=lambda x: -x[3])  # sort by CC½ descending
    print(f"\n{'='*60}")
    print(f"Top 10 by CC1/2 (baseline CC1/2 = {cc_base:.4f}, R_split = {r_base:.4f}):")
    print(f"{'w_min':>6} {'cutoff':>7} {'steep':>6} {'CC1/2':>8} {'R_split':>8} {'dCC':>7}")
    print("-" * 50)
    for w_min, cutoff, steepness, cc, r, delta_cc, _ in results[:10]:
        sign = "+" if delta_cc >= 0 else ""
        print(f"{w_min:6.2f} {cutoff:7.1f} {steepness:6.1f} {cc:8.4f} {r:8.4f} {sign}{delta_cc:6.4f}")

    results.sort(key=lambda x: x[4])  # sort by R-split ascending
    print(f"\nTop 10 by R_split:")
    print(f"{'w_min':>6} {'cutoff':>7} {'steep':>6} {'CC1/2':>8} {'R_split':>8}")
    print("-" * 50)
    for w_min, cutoff, steepness, cc, r, delta_cc, _ in results[:10]:
        print(f"{w_min:6.2f} {cutoff:7.1f} {steepness:6.1f} {cc:8.4f} {r:8.4f}")
