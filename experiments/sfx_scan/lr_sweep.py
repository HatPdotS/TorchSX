#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u
"""Quick lr/reg sweep for Adam + partiality. Full merge only, no split-half."""

from pathlib import Path
import torch

from torchsx.stream import SFXStream
from torchsx.scaling import AbsoluteScale, BScale, EwaldPartiality, prepare_scaler_inputs
from torchsx.target._self_consistent import SelfConsistentTarget

STREAM_DIR = Path('/das/work/p17/p17490/Peter/torchsx/sfx_scan') / "streams"
DELAYS = [-91, -136, -182, -227, -273]
SPACEGROUP = "P 1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load once
streams = []
for d in DELAYS:
    stem = STREAM_DIR / f"delay_{d}.stream"
    cache = stem.with_suffix(".sfx")
    s = SFXStream.from_stream(stem, spacegroup=SPACEGROUP, cache=cache, n_workers=16, merge_friedel=True)
    streams.append(s)
combined = SFXStream.concat(streams)
print(f"Combined: {combined.n_crystals} crystals, {combined.n_observations} obs\n")


def run_test(lr, dw_reg, dw_lr_scale, sigma_lr_scale, n_cycles=8, warmup=200, refine=100, label=""):
    prefix = f"[{label}] "
    inp = prepare_scaler_inputs(combined._core, n_shells=20).to(device)
    I_bg = inp.I_bg if inp.I_bg is not None else torch.zeros_like(inp.I_obs)

    abs_scale = AbsoluteScale(inp.n_crystals).to(device)
    b_scale = BScale(inp.n_crystals).to(device)
    target = SelfConsistentTarget(inp.n_unique, inp.n_shells).to(device)
    partiality = EwaldPartiality(inp.n_crystals, inp.UB, inp.wavelength).to(device)

    cid = inp.crystal_id.long()
    hkl_obs_float = inp.hkl_obs.float()

    target.calibrate(inp.I_obs, torch.ones_like(inp.I_obs), inp.hkl_id, inp.shell, inp.crystal_id)

    optimizer = torch.optim.Adam([
        {"params": [abs_scale.log_G], "lr": lr},
        {"params": [b_scale.B], "lr": lr},
        {"params": [target.log_I_true], "lr": lr},
        {"params": [partiality.delta_omega], "lr": lr * dw_lr_scale},
        {"params": [partiality.log_sigma_0, partiality.log_B_relp], "lr": lr * sigma_lr_scale},
    ])
    all_params = list(abs_scale.parameters()) + list(b_scale.parameters()) + list(target.parameters()) + list(partiality.parameters())

    weight = None
    for cycle in range(n_cycles):
        if cycle > 0:
            with torch.no_grad():
                log_p = partiality.log_scale_per_obs(hkl_obs_float, inp.crystal_id, s_sq=inp.s_sq)
                ts = torch.exp(abs_scale.log_G[cid] - 2.0 * b_scale.B[cid] * inp.s_sq + log_p)
            target.update_sigma(inp.I_obs, ts, inp.hkl_id)
            weight = target.compute_crystal_weights(
                inp.I_obs, ts, inp.hkl_id, inp.crystal_id,
                w_min=0.01, cutoff=0.5, steepness=5.0,
            )

        n_steps = warmup if cycle == 0 else refine
        for step in range(n_steps):
            optimizer.zero_grad()
            log_G = abs_scale.log_G[cid]
            log_B = -2.0 * b_scale.B[cid] * inp.s_sq
            log_p = partiality.log_scale_per_obs(hkl_obs_float, inp.crystal_id, s_sq=inp.s_sq)
            log_total = (log_G + log_B + log_p).clamp(-20, 20)
            total_scale = torch.exp(log_total)
            loss = target(inp.I_obs, total_scale, inp.hkl_id, I_bg, weight=weight)
            loss = loss + 100.0 * abs_scale.log_G.mean() ** 2
            loss = loss + 1e-3 * (abs_scale.log_G ** 2).mean()
            loss = loss + 1e-3 * (b_scale.B ** 2).mean()
            loss = loss + dw_reg * (partiality.delta_omega ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 10.0)
            optimizer.step()

        if torch.isnan(loss):
            print(f"  {prefix}cycle {cycle}: NaN! Aborting.")
            del abs_scale, b_scale, target, partiality, optimizer, inp
            torch.cuda.empty_cache()
            return

    with torch.no_grad():
        log_p = partiality.log_scale_per_obs(hkl_obs_float, inp.crystal_id, s_sq=inp.s_sq)
        p = torch.exp(log_p)
        dw_deg = partiality.delta_omega.norm(dim=-1) * (180.0 / 3.14159)
        sm = torch.exp(partiality.log_sigma_0).item()
        sd = torch.exp(partiality.log_B_relp).item()

    print(f"  {prefix}loss={loss.item():.4f}, "
          f"δω med={dw_deg.median().item():.2f}° max={dw_deg.max().item():.2f}°, "
          f"σ_0={sm:.5f} B_relp={sd:.3f}, "
          f"p=[{p.min().item():.3f}, {p.median().item():.3f}, {p.max().item():.3f}]")

    del abs_scale, b_scale, target, partiality, optimizer, inp
    torch.cuda.empty_cache()


# Sweep: vary lr and dw_reg
configs = [
    # (lr,    dw_reg, dw_lr_scale, sigma_lr_scale, label)
    (0.01,   1.0,    0.1,   0.01,  "lr=0.01 reg=1.0 (baseline)"),
    (0.03,   1.0,    0.1,   0.01,  "lr=0.03 reg=1.0"),
    (0.05,   1.0,    0.1,   0.01,  "lr=0.05 reg=1.0"),
    (0.1,    1.0,    0.1,   0.01,  "lr=0.1  reg=1.0"),
    (0.03,   10.0,   0.1,   0.01,  "lr=0.03 reg=10"),
    (0.05,   10.0,   0.1,   0.01,  "lr=0.05 reg=10"),
    (0.1,    10.0,   0.1,   0.01,  "lr=0.1  reg=10"),
    (0.1,    50.0,   0.1,   0.01,  "lr=0.1  reg=50"),
    (0.1,    10.0,   0.1,   0.1,   "lr=0.1  reg=10 σ_lr=0.1"),
    (0.1,    10.0,   0.1,   1.0,   "lr=0.1  reg=10 σ_lr=1.0"),
]

for lr, dw_reg, dw_lr_scale, sigma_lr_scale, label in configs:
    print(f"\n--- {label} ---")
    run_test(lr, dw_reg, dw_lr_scale, sigma_lr_scale, label=label)
