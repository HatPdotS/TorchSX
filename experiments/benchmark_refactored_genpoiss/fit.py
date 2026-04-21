#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -p gpu
#SBATCH -t 4:00:00
#SBATCH -o /das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/benchmark_refactored_genpoiss/fit.out

"""Fit all methods on the 3 stream dataset and save results for evaluation.

Methods:
  - EM Unified (new Merger, single LBFGS)
  - EM Poisson (baseline fit_em)
  - GenGauss   (fit_gengauss)
  - Partialator unity   (parsed from pre-computed CrystFEL .hkl)
  - Partialator xsphere (parsed from pre-computed CrystFEL .hkl)
"""

from __future__ import annotations

from pathlib import Path

import torch

from torchsx.stream import SFXStream
from torchsx.merge import fit_em, EMConfig, ScaleBCorrection as OldScaleB
from torchsx.merge._distribution import fit_gengauss
from torchsx.merging_clean import Merger, MergeConfig, ScaleBCorrection as NewScaleB

# ── Configuration ────────────────────────────────────────────
STREAM_DIR = Path("/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams")
PART_DIR = Path("/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/partialator")
OUTDIR = Path("/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/benchmark_refactored_genpoiss")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Same 3 streams the user asked for
DELAYS = [-91, -136, -182, -227]
CELL = torch.tensor([14.97, 18.85, 18.89, 89.37, 84.94, 67.84], dtype=torch.float32)
SEED = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}", flush=True)


# ── Helpers ──────────────────────────────────────────────────
def parse_crystfel_hkl(path):
    h, k, l, I_ = [], [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("CrystFEL", "Symmetry", "h ", "End")):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                h.append(int(parts[0])); k.append(int(parts[1]))
                l.append(int(parts[2])); I_.append(float(parts[3]))
            except ValueError:
                continue
    hkl = torch.tensor(list(zip(h, k, l)), dtype=torch.int16)
    return hkl, torch.tensor(I_, dtype=torch.float32)


def save_result(result, path: Path) -> None:
    """Save a dict or MergedResult in the unified schema expected by evaluate.py."""
    if hasattr(result, "hkl"):  # MergedResult
        torch.save({
            "hkl": result.hkl,
            "I0": result.I0,
            "alpha": result.alpha,
            "beta": result.beta,
            "d_spacing": result.d_spacing,
            "n_obs": result.n_obs,
        }, path)
    else:
        torch.save(result, path)


# ══════════════════════════════════════════════════════════════
# Load and split
# ══════════════════════════════════════════════════════════════
print("── Loading streams ──", flush=True)
streams = [SFXStream.load(STREAM_DIR / f"delay_{d}.sfx") for d in DELAYS]
s = SFXStream.concat(streams)
h1, h2 = s.split_random(n_parts=2, seed=SEED)
print(f"Total: {s.n_crystals} crystals, split: {h1.n_crystals}/{h2.n_crystals}", flush=True)

torch.save(CELL, OUTDIR / "cell.pt")

# ══════════════════════════════════════════════════════════════
# Partialator (parse pre-computed results)
# ══════════════════════════════════════════════════════════════
from torchref.symmetry import canonicalize_hkl

print("\n── Partialator ──", flush=True)
for label, h1_path, h2_path, merge_path in [
    ("unity", "ind_half1.hkl", "ind_half2.hkl", "neg.hkl"),
    ("xsphere", "xsphere.hkl1", "xsphere.hkl2", "xsphere.hkl"),
]:
    p1 = PART_DIR / h1_path
    p2 = PART_DIR / h2_path
    pm = PART_DIR / merge_path
    if not p1.exists():
        print(f"  skip {label} (missing {p1})", flush=True)
        continue
    for tag, path in [("half1", p1), ("half2", p2), ("merge", pm)]:
        hkl, I = parse_crystfel_hkl(path)
        hkl, _, _, si = canonicalize_hkl(hkl.int(), "P -1", include_friedel=True)
        F = I[si].clamp(min=0).sqrt()
        torch.save({"hkl": hkl.to(torch.int16), "F": F},
                   OUTDIR / f"part_{label}_{tag}.pt")
        print(f"  Saved part_{label}_{tag}.pt ({len(F)} refl)", flush=True)

# ══════════════════════════════════════════════════════════════
# GenGauss (log-space IRLS)
# ══════════════════════════════════════════════════════════════
print("\n── GenGauss ──", flush=True)
for label, stream in [("half1", h1), ("half2", h2), ("full", s)]:
    out = OUTDIR / f"gg_{label}.pt"
    if out.exists():
        print(f"  {label}: skip (already have {out.name})", flush=True)
        continue
    print(f"  {label}...", flush=True)
    dist = fit_gengauss(
        stream,
        min_obs=20, n_smooth_iter=15, smooth_lambda=0.3,
        n_em_cycles=10, scale_crystals=True,
    )
    save_result(dist, out)
    print(f"    {len(dist.I0)} HKLs  med(α)={dist.alpha.median():.4f}", flush=True)

# ══════════════════════════════════════════════════════════════
# EM Poisson (baseline) — does not chunk; OOMs on ~22 GB GPUs for full dataset.
# ══════════════════════════════════════════════════════════════
print("\n── EM Poisson ──", flush=True)
em_cfg = EMConfig(n_em_cycles=10, min_obs=5, alpha_prior_lambda=0.0)
for label, stream in [("half1", h1), ("half2", h2), ("full", s)]:
    out = OUTDIR / f"em_{label}.pt"
    if out.exists():
        print(f"  {label}: skip (already have {out.name})", flush=True)
        continue
    print(f"  {label}...", flush=True)
    dist = fit_em(stream, config=em_cfg, corrections=[OldScaleB()], device=device)
    save_result(dist, out)
    print(f"    {len(dist.I0)} HKLs  med(α)={dist.alpha.median():.4f}", flush=True)

# ══════════════════════════════════════════════════════════════
# EM Unified (new single-LBFGS merger)
# ══════════════════════════════════════════════════════════════
print("\n── EM Unified ──", flush=True)
# All penalty defaults now baked into MergeConfig (see torchsx/merging_clean/config.py):
#   lam_α_smooth = lam_β_smooth = 500   (rational 10× rule for N_h_max ≈ 1000)
#   lam_α_anchor = lam_β_anchor = 0     (anchors dropped — shell-fit biases β)
#   lam_sigma_cauchy = 50, τ = 1.0      (half-Cauchy width shrinkage prior)
#   sig_lo_override = 1e-4              (sig_lo sweep plateau bottom)
#   obs_chunk_size = 1_000_000          (tuned from bench_closure)
u_cfg = MergeConfig(
    min_obs=5,
    max_iter=150,
    history_size=10,
)

# Fit FULL first — more data per HKL → better-determined (α, β).  Then seed
# the halves from the full result so all three end up in the same basin
# (diagnosing whether the full/half β discrepancy was a local-minimum artifact).
print(f"  full... (no seed)", flush=True)
full_merger = Merger(s, config=u_cfg, corrections=[NewScaleB], device=device)
full_result = full_merger.merge()
save_result(full_result, OUTDIR / "unified_full.pt")
print(
    f"    {len(full_result.I0)} HKLs  med(α)={full_result.alpha.median():.4f}  "
    f"med(β)={full_result.beta.median():.4f}  "
    f"final_nll={full_result.diagnostics.get('final_data_nll', 'n/a')}",
    flush=True,
)

for label, stream in [("half1", h1), ("half2", h2)]:
    print(f"  {label}... (seeded from full)", flush=True)
    merger = Merger(
        stream, config=u_cfg, corrections=[NewScaleB],
        device=device, init_from=full_result,
    )
    result = merger.merge()
    save_result(result, OUTDIR / f"unified_{label}.pt")
    print(
        f"    {len(result.I0)} HKLs  med(α)={result.alpha.median():.4f}  "
        f"med(β)={result.beta.median():.4f}  "
        f"final_nll={result.diagnostics.get('final_data_nll', 'n/a')}",
        flush=True,
    )

print("\nDone.", flush=True)
