#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -p gpu-day
#SBATCH -o sfx_scan/independent_compare.out

"""Compare two independent delay streams merged separately — torchsx vs partialator."""

from pathlib import Path
import torch

from torchsx.stream import SFXStream
from torchsx.scaling import AbsoluteScale, BScale, prepare_scaler_inputs
from torchsx.target._self_consistent import SelfConsistentTarget

STREAM_DIR = Path('/das/work/p17/p17490/Peter/torchsx/sfx_scan') / "streams"
SPACEGROUP = "P 1"
MERGE_FRIEDEL = True
PART_DIR = Path('/das/work/p17/p17490/Peter/torchsx/sfx_scan/partialator')
PLOT_DIR = Path('/das/work/p17/p17490/Peter/torchsx/sfx_scan')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CELL = torch.tensor([14.97, 18.85, 18.89, 89.37, 84.94, 67.84])


def run_scaling(stream, device, label=""):
    prefix = f"[{label}] " if label else ""
    inp = prepare_scaler_inputs(stream._core, n_shells=20)
    inp = inp.to(device)
    I_bg = inp.I_bg if inp.I_bg is not None else torch.zeros_like(inp.I_obs)

    abs_scale = AbsoluteScale(inp.n_crystals).to(device)
    b_scale = BScale(inp.n_crystals).to(device)
    target = SelfConsistentTarget(inp.n_unique, inp.n_shells).to(device)
    cid = inp.crystal_id.long()

    with torch.no_grad():
        total_scale_init = torch.ones_like(inp.I_obs)
    target.calibrate(inp.I_obs, total_scale_init, inp.hkl_id, inp.shell, inp.crystal_id)

    params = list(abs_scale.parameters()) + list(b_scale.parameters()) + list(target.parameters())
    optimizer = torch.optim.LBFGS(params, lr=1.0, max_iter=20, history_size=10,
                                   line_search_fn="strong_wolfe")

    print(f"{prefix}Optimizing ({inp.n_crystals} crystals, {inp.n_unique} unique) ...")
    weight = None
    for cycle in range(5):
        if cycle > 0:
            with torch.no_grad():
                ts = torch.exp(abs_scale.log_G[cid] - 2.0 * b_scale.B[cid] * inp.s_sq)
            weight = target.compute_crystal_weights(
                inp.I_obs, ts, inp.hkl_id, inp.crystal_id,
                w_min=0.01, cutoff=0.5, steepness=5.0,
            )
        n_steps = 50 if cycle == 0 else 20
        for step in range(n_steps):
            current_loss = [None]
            def closure():
                optimizer.zero_grad()
                log_G = abs_scale.log_G[cid]
                log_B = -2.0 * b_scale.B[cid] * inp.s_sq
                total_scale = torch.exp(log_G + log_B)
                loss = target(inp.I_obs, total_scale, inp.hkl_id, I_bg, weight=weight)
                loss = loss + 100.0 * abs_scale.log_G.mean() ** 2
                loss = loss + 1e-3 * (abs_scale.log_G ** 2).mean()
                loss = loss + 1e-3 * (b_scale.B ** 2).mean()
                loss.backward()
                current_loss[0] = loss.item()
                return loss
            optimizer.step(closure)
        print(f"  {prefix}cycle {cycle}: loss = {current_loss[0]:.4f}")

    merged = target.to_dict()
    merged["hkl"] = inp.unique_hkl
    del abs_scale, b_scale, target, optimizer, inp
    torch.cuda.empty_cache()
    return {k: v.cpu() for k, v in merged.items()}


# -- Load two independent streams --
print("Loading delay_-91 ...")
s1 = SFXStream.from_stream(STREAM_DIR / "delay_-91.stream", spacegroup=SPACEGROUP,
                            cache=STREAM_DIR / "delay_-91.sfx", n_workers=16, merge_friedel=MERGE_FRIEDEL)
print(f"  {s1.n_crystals} crystals, {s1.n_observations} obs")

print("Loading delay_-136 ...")
s2 = SFXStream.from_stream(STREAM_DIR / "delay_-136.stream", spacegroup=SPACEGROUP,
                            cache=STREAM_DIR / "delay_-136.sfx", n_workers=16, merge_friedel=MERGE_FRIEDEL)
print(f"  {s2.n_crystals} crystals, {s2.n_observations} obs")

# -- Merge independently --
print("\n=== TorchSX independent merge ===")
m1 = run_scaling(s1, device, label="delay-91")
m2 = run_scaling(s2, device, label="delay-136")

from torchsx.merge._stats import match_halves, cc_half, r_split, shell_stats, format_shell_table

matched_tx = match_halves(m1["hkl"], m1["F"], m2["hkl"], m2["F"], CELL, d_min=0.9, d_max=10.0)
print(f"\nTorchSX matched: {matched_tx.F1.numel()}")
overall_cc_tx = cc_half(matched_tx)
overall_r_tx = r_split(matched_tx)
shells_tx = shell_stats(matched_tx, n_shells=20)
print(f"\n{format_shell_table(shells_tx, overall_cc_tx, overall_r_tx)}")

# -- Partialator independent merge (if available) --
def parse_hkl(path):
    h_list, k_list, l_list, I_list = [], [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('CrystFEL', 'Symmetry', 'h ', 'End')):
                continue
            parts = line.split()
            if len(parts) < 4: continue
            try:
                h_list.append(int(parts[0])); k_list.append(int(parts[1])); l_list.append(int(parts[2]))
                I_list.append(float(parts[3]))
            except ValueError: continue
    hkl = torch.tensor(list(zip(h_list, k_list, l_list)), dtype=torch.int16)
    I = torch.tensor(I_list, dtype=torch.float32)
    F = I.clamp(min=0).sqrt()
    return hkl, F

try:
    from torchref.symmetry import canonicalize_hkl
    print("\n=== Partialator independent merge ===")
    hkl_p1, F_p1 = parse_hkl(PART_DIR / "ind_half1.hkl")
    hkl_p2, F_p2 = parse_hkl(PART_DIR / "ind_half2.hkl")
    hkl_p1, _, _, s1 = canonicalize_hkl(hkl_p1.int(), "P -1", include_friedel=True)
    hkl_p2, _, _, s2 = canonicalize_hkl(hkl_p2.int(), "P -1", include_friedel=True)
    F_p1, F_p2 = F_p1[s1], F_p2[s2]
    hkl_p1, hkl_p2 = hkl_p1.short(), hkl_p2.short()

    matched_part = match_halves(hkl_p1, F_p1, hkl_p2, F_p2, CELL, d_min=0.9, d_max=10.0)
    print(f"Partialator matched: {matched_part.F1.numel()}")
    overall_cc_part = cc_half(matched_part)
    overall_r_part = r_split(matched_part)
    shells_part = shell_stats(matched_part, n_shells=20)
    print(f"\n{format_shell_table(shells_part, overall_cc_part, overall_r_part)}")
except FileNotFoundError:
    print("\nPartialator independent HKL files not ready yet — run compare later.")
