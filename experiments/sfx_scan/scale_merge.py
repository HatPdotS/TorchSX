#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u


#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -p gpu-day
#SBATCH -o sfx_scan/scale_merge.out

"""Scale & merge 5 negative-delay SFX streams to MTZ, with split-half validation."""

from pathlib import Path

import torch

from torchsx.stream import SFXStream
from torchsx.scaling import AbsoluteScale, BScale, EwaldPartiality, prepare_scaler_inputs
from torchsx.target._self_consistent import SelfConsistentTarget

STREAM_DIR = Path('/das/work/p17/p17490/Peter/torchsx/sfx_scan') / "streams"
DELAYS = [-91, -136, -182, -227, -273]
SPACEGROUP = "P 1"
MERGE_FRIEDEL = True
OUTPUT_MTZ = Path('/das/work/p17/p17490/Peter/torchsx/sfx_scan') / "neg_delay_merged.mtz"
PLOT_DIR = Path('/das/work/p17/p17490/Peter/torchsx/sfx_scan')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

n_workers = 16


def run_scaling(stream: SFXStream, device: torch.device,
                n_cycles: int = 15, warmup_steps: int = 500,
                refine_steps: int = 200,
                lr: float = 0.1, label: str = "") -> dict:
    """Scale a stream and return merged dict with keys: hkl, I, F, sigma_LN.

    Uses Adam with EM-style per-crystal confidence reweighting.
    Cycle 0 runs warmup_steps to converge without weights.
    Cycles 1+ run refine_steps each with updated weights.
    """
    prefix = f"[{label}] " if label else ""

    inp = prepare_scaler_inputs(stream._core, n_shells=20)
    inp = inp.to(device)

    # Use zeros for I_bg if not available from stream
    I_bg = inp.I_bg if inp.I_bg is not None else torch.zeros_like(inp.I_obs)

    abs_scale = AbsoluteScale(inp.n_crystals).to(device)
    b_scale = BScale(inp.n_crystals).to(device)
    target = SelfConsistentTarget(inp.n_unique, inp.n_shells).to(device)

    cid = inp.crystal_id.long()

    with torch.no_grad():
        total_scale_init = torch.ones_like(inp.I_obs)
    target.calibrate(
        I_obs=inp.I_obs,
        total_scale=total_scale_init,
        hkl_id=inp.hkl_id,
        shell=inp.shell,
        crystal_id=inp.crystal_id,
    )

    # --- Partiality model (jointly optimized with G, B, I_true) ---
    partiality = EwaldPartiality(inp.n_crystals, inp.UB, inp.wavelength).to(device)
    hkl_obs_float = inp.hkl_obs.float()

    optimizer = torch.optim.Adam([
        {"params": [abs_scale.log_G], "lr": lr},
        {"params": [b_scale.B], "lr": lr},
        {"params": [target.log_I_true], "lr": lr},
        {"params": [partiality.delta_omega], "lr": lr * 0.1},
        {"params": [partiality.log_sigma_0, partiality.log_B_relp], "lr": lr * 0.01},
    ], lr=lr)

    all_params = (
        list(abs_scale.parameters())
        + list(b_scale.parameters())
        + list(target.parameters())
        + list(partiality.parameters())
    )

    print(f"{prefix}Optimizing ({inp.n_crystals} crystals, {inp.n_unique} unique) ...")
    weight = None  # start unweighted (cycle 0)
    total_steps = 0

    for cycle in range(n_cycles):
        # Update sigma and reweight after first cycle
        if cycle > 0:
            with torch.no_grad():
                log_p = partiality.log_scale_per_obs(hkl_obs_float, inp.crystal_id, s_sq=inp.s_sq)
                ts = torch.exp(abs_scale.log_G[cid] - 2.0 * b_scale.B[cid] * inp.s_sq + log_p)
            target.update_sigma(inp.I_obs, ts, inp.hkl_id)
            sLN = target.sigma_LN_prior
            print(f"  {prefix}cycle {cycle}: sigma_LN=[{sLN.min().item():.4f}, {sLN.max().item():.4f}]")

            weight = target.compute_crystal_weights(
                inp.I_obs, ts, inp.hkl_id, inp.crystal_id,
                w_min=0.01, cutoff=0.5, steepness=5.0,
            )
            mean_w = weight.mean().item()
            n_down = (weight < 0.5 * mean_w).sum().item()
            pct_down = 100.0 * n_down / weight.numel()
            print(f"  {prefix}cycle {cycle}: reweight w=[{weight.min().item():.3f}, {weight.max().item():.3f}], "
                  f"mean={mean_w:.3f}, {n_down}/{weight.numel()} obs downweighted ({pct_down:.1f}%)")

            # Log partiality diagnostics
            with torch.no_grad():
                dw_deg = partiality.delta_omega.norm(dim=-1) * (180.0 / 3.14159)
                s0 = torch.exp(partiality.log_sigma_0).item()
                br = torch.exp(partiality.log_B_relp).item()
                p = torch.exp(log_p)
            print(f"  {prefix}cycle {cycle}: δω median={dw_deg.median().item():.3f}° max={dw_deg.max().item():.3f}°, "
                  f"σ_0={s0:.5f} B_relp={br:.3f} Å⁻¹, "
                  f"p=[{p.min().item():.3f}, {p.median().item():.3f}, {p.max().item():.3f}]")

        n_steps = warmup_steps if cycle == 0 else refine_steps
        for step in range(n_steps):
            optimizer.zero_grad()
            log_G = abs_scale.log_G[cid]
            log_B = -2.0 * b_scale.B[cid] * inp.s_sq
            log_p = partiality.log_scale_per_obs(hkl_obs_float, inp.crystal_id, s_sq=inp.s_sq)
            log_total = (log_G + log_B + log_p).clamp(-20, 20)
            total_scale = torch.exp(log_total)
            loss = target(inp.I_obs, total_scale, inp.hkl_id, I_bg, weight=weight)
            # Soft centering penalty
            loss = loss + 100.0 * abs_scale.log_G.mean() ** 2
            # Regularisation on scale params
            loss = loss + 1e-3 * (abs_scale.log_G ** 2).mean()
            loss = loss + 1e-3 * (b_scale.B ** 2).mean()
            # Tikhonov on orientation perturbations
            loss = loss + 50.0 * (partiality.delta_omega ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 10.0)
            optimizer.step()
            total_steps += 1

        print(f"  {prefix}cycle {cycle}: loss = {loss.item():.4f}  (step {total_steps})")

    merged = target.to_dict()
    merged["hkl"] = inp.unique_hkl

    # Free GPU memory
    del abs_scale, b_scale, target, partiality, optimizer, inp
    torch.cuda.empty_cache()

    return {k: v.cpu() for k, v in merged.items()}


# -- 1. Load streams on CPU (with .sfx caching) --   
streams = []
for d in DELAYS:
    stem = STREAM_DIR / f"delay_{d}.stream"
    cache = stem.with_suffix(".sfx")
    print(f"Loading {stem.name} ...")
    s = SFXStream.from_stream(stem, spacegroup=SPACEGROUP, cache=cache, n_workers=n_workers, merge_friedel=MERGE_FRIEDEL)
    print(f"  {s.n_crystals} crystals, {s.n_observations} obs")
    streams.append(s)

# -- 2. Concatenate (still on CPU) --
combined = SFXStream.concat(streams)
print(f"\nCombined: {combined.n_crystals} crystals, {combined.n_observations} obs")

# -- 3. Full merge --
print("\n=== Full merge ===")
merged = run_scaling(combined, device, label="full")

I_merged = merged["I"]
F_merged = merged["F"]
sigma_LN = merged["sigma_LN"]
# sigma(F) ~ F * sigma_LN / 2 (error propagation from I to F)
F_sigma = F_merged * sigma_LN / 2.0

# F-value diagnostics
print(f"\nF-value stats: min={F_merged.min():.2f}, max={F_merged.max():.2f}, "
      f"median={F_merged.median():.2f}, dynamic range={F_merged.max()/F_merged.min().clamp(min=1e-6):.0f}")
for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    val = F_merged.quantile(pct / 100.0).item()
    print(f"  F percentile {pct:2d}%: {val:.2f}")
print(f"  I range: {I_merged.min():.2f} to {I_merged.max():.2f}, median={I_merged.median():.2f}")

# -- 4. Write MTZ --
from torchref import Cell, SpaceGroup, ReflectionData

median_cell = combined._core.cell.median(dim=0).values
cell = Cell(median_cell, device="cpu")
sg = SpaceGroup(SPACEGROUP)

hkl_int = merged["hkl"].cpu().int()
n_refl = hkl_int.shape[0]
rfree_flags = torch.rand(n_refl) < 0.05

data = ReflectionData.from_tensors(
    hkl=hkl_int,
    F=F_merged,
    F_sigma=F_sigma,
    cell=cell,
    spacegroup=sg,
    rfree_flags=rfree_flags,
    device="cpu",
)

data.write_mtz(str(OUTPUT_MTZ))
print(f"\nWrote {OUTPUT_MTZ}  ({n_refl} unique reflections)")

# -- 5. Split-half validation --
print("\n=== Split-half validation ===")
half1, half2 = combined.split_random(n_parts=2, seed=42)
print(f"Half 1: {half1.n_crystals} crystals, {half1.n_observations} obs")
print(f"Half 2: {half2.n_crystals} crystals, {half2.n_observations} obs")

merged1 = run_scaling(half1, device, label="half-1")
merged2 = run_scaling(half2, device, label="half-2")

from torchsx.merge._stats import match_halves, cc_half, r_split, shell_stats, format_shell_table, write_shell_csv
from torchsx.merge._plots import plot_cc_vs_resolution, plot_rsplit_vs_resolution, plot_f_scatter

matched = match_halves(
    merged1["hkl"], merged1["F"],
    merged2["hkl"], merged2["F"],
    median_cell, d_min=0.9, d_max=10.0
)
print(f"\nMatched reflections: {matched.F1.numel()}")

overall_cc = cc_half(matched)
overall_r = r_split(matched)
shells = shell_stats(matched, n_shells=20)

print(f"\n{format_shell_table(shells, overall_cc, overall_r)}")

plot_cc_vs_resolution(shells, PLOT_DIR / "cc_half_vs_resolution.png")
plot_rsplit_vs_resolution(shells, PLOT_DIR / "rsplit_vs_resolution.png")
plot_f_scatter(matched, PLOT_DIR / "f_scatter_halves.png")
print(f"\nPlots saved to {PLOT_DIR}")
write_shell_csv(shells, PLOT_DIR / "shell_stats.csv")
    