#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u


#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -p gpu-day
#SBATCH -o sfx_scan/debug_optimizer.out

"""Diagnose why Adam works and L-BFGS doesn't: test centering methods + optimizers."""

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


def run_trial(inp, device, optimizer_type="adam", n_steps=1000,
              centering="hard", center_lam=10.0, lr=None, label=""):
    """Run a single optimization trial.

    centering: "hard" (subtract mean after each step),
               "soft" (add penalty to loss),
               "none" (no centering)
    """
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

    if optimizer_type == "adam":
        if lr is None:
            lr = 0.005
        optimizer = torch.optim.Adam(params, lr=lr)

        for step in range(n_steps):
            optimizer.zero_grad()
            log_G = abs_scale.log_G[cid]
            log_B = -2.0 * b_scale.B[cid] * inp.s_sq
            ts = torch.exp(log_G + log_B)
            loss = target(inp.I_obs, ts, inp.hkl_id)
            if centering == "soft":
                loss = loss + center_lam * abs_scale.log_G.mean() ** 2
            loss.backward()
            optimizer.step()
            if centering == "hard":
                with torch.no_grad():
                    abs_scale.log_G -= abs_scale.log_G.mean()

    elif optimizer_type == "lbfgs":
        if lr is None:
            lr = 0.5
        optimizer = torch.optim.LBFGS(
            params, lr=lr, max_iter=40, line_search_fn="strong_wolfe",
        )

        for step in range(n_steps):
            def closure():
                optimizer.zero_grad()
                log_G = abs_scale.log_G[cid]
                log_B = -2.0 * b_scale.B[cid] * inp.s_sq
                ts = torch.exp(log_G + log_B)
                loss = target(inp.I_obs, ts, inp.hkl_id)
                if centering == "soft":
                    loss = loss + center_lam * abs_scale.log_G.mean() ** 2
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            if centering == "hard":
                with torch.no_grad():
                    abs_scale.log_G -= abs_scale.log_G.mean()

    # Diagnostics
    with torch.no_grad():
        mean_logG = abs_scale.log_G.mean().item()
        std_logG = abs_scale.log_G.std().item()
        F = target.I_true.clamp(min=0).sqrt()
        F_min = F.min().item()
        F_max = F.max().item()
        F_med = F.median().item()

    merged = target.to_dict()
    merged["hkl"] = inp.unique_hkl

    print(f"  [{label}] loss={loss.item():.4f}  mean(logG)={mean_logG:.4f}  "
          f"std(logG)={std_logG:.4f}  F=[{F_min:.2f}, {F_med:.2f}, {F_max:.2f}]")

    return {k: v.cpu() for k, v in merged.items()}


# -- Load data --
streams = []
for d in DELAYS:
    stem = STREAM_DIR / f"delay_{d}.stream"
    cache = stem.with_suffix(".sfx")
    s = SFXStream.from_stream(stem, spacegroup=SPACEGROUP, cache=cache,
                               n_workers=n_workers, merge_friedel=MERGE_FRIEDEL)
    streams.append(s)
    print(f"  {stem.name}: {s.n_crystals} crystals")

combined = SFXStream.concat(streams)
half1, half2 = combined.split_random(n_parts=2, seed=42)
median_cell = combined._core.cell.median(dim=0).values

inp1 = prepare_scaler_inputs(half1._core, n_shells=20).to(device)
inp2 = prepare_scaler_inputs(half2._core, n_shells=20).to(device)

print(f"\nHalf 1: {half1.n_crystals} crystals, {half1.n_observations} obs")
print(f"Half 2: {half2.n_crystals} crystals, {half2.n_observations} obs")


def evaluate(m1, m2, label):
    matched = match_halves(m1["hkl"], m1["F"], m2["hkl"], m2["F"],
                           median_cell, d_min=0.9, d_max=10.0)
    cc = cc_half(matched)
    r = r_split(matched)
    print(f"  => {label}: CC1/2 = {cc:.4f}  R_split = {r:.4f}\n")
    return cc, r


# -- Trials --
trials = [
    # (optimizer, n_steps, centering, center_lam, lr, label)
    ("adam",  1000, "hard", None, 0.005, "Adam 1000 hard"),
    ("adam",  1000, "soft", 10,   0.005, "Adam 1000 soft lam=10"),
    ("adam",  1000, "soft", 1000, 0.005, "Adam 1000 soft lam=1000"),
    ("adam",  1000, "none", None, 0.005, "Adam 1000 no centering"),
    ("lbfgs", 10,  "hard", None, 0.5,   "LBFGS 10 hard"),
    ("lbfgs", 10,  "soft", 10,   0.5,   "LBFGS 10 soft lam=10"),
    ("lbfgs", 10,  "soft", 100,  0.5,   "LBFGS 10 soft lam=100"),
    ("lbfgs", 10,  "soft", 1000, 0.5,   "LBFGS 10 soft lam=1000"),
    ("lbfgs", 10,  "none", None, 0.5,   "LBFGS 10 no centering"),
    ("lbfgs", 50,  "hard", None, 0.5,   "LBFGS 50 hard"),
    ("lbfgs", 50,  "soft", 10,   0.5,   "LBFGS 50 soft lam=10"),
    ("lbfgs", 50,  "soft", 1000, 0.5,   "LBFGS 50 soft lam=1000"),
    ("lbfgs", 10,  "hard", None, 0.1,   "LBFGS 10 hard lr=0.1"),
    ("lbfgs", 10,  "hard", None, 1.0,   "LBFGS 10 hard lr=1.0"),
]

print(f"\n{'='*80}")
print(f"Running {len(trials)} trials")
print(f"{'='*80}\n")

results = []
for opt, n_steps, centering, clam, lr, label in trials:
    print(f"--- {label} ---")
    try:
        m1 = run_trial(inp1, device, opt, n_steps, centering, clam or 0, lr, f"h1 {label}")
        m2 = run_trial(inp2, device, opt, n_steps, centering, clam or 0, lr, f"h2 {label}")
        cc, r = evaluate(m1, m2, label)
        results.append((label, cc, r))
    except Exception as e:
        print(f"  FAILED: {e}\n")
        results.append((label, float('nan'), float('nan')))
    torch.cuda.empty_cache()

# -- Summary --
print(f"\n{'='*80}")
print(f"{'Trial':<35} {'CC1/2':>8} {'R_split':>8}")
print("-" * 55)
for label, cc, r in results:
    print(f"{label:<35} {cc:8.4f} {r:8.4f}")
