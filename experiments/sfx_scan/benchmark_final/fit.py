#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -p gpu-day
#SBATCH -t 4:00:00
#SBATCH -o /das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/benchmark_final/fit.out

"""Fit all methods and save results for later evaluation."""

from __future__ import annotations

from pathlib import Path

import torch

from torchsx.stream import SFXStream
from torchsx.merge import (
    fit_em, EMConfig, ScaleBCorrection,
)
from torchsx.merge._distribution import fit_gengauss

# ── Configuration ────────────────────────────────────────────
STREAM_DIR = Path("/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams")
PART_DIR = Path("/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/partialator")
OUTDIR = Path("/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/benchmark_final")
OUTDIR.mkdir(parents=True, exist_ok=True)

DELAYS = [-91, -136, -182, -227]
CELL = torch.tensor([14.97, 18.85, 18.89, 89.37, 84.94, 67.84], dtype=torch.float32)
SEED = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


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


# ══════════════════════════════════════════════════════════════
# Load and split
# ══════════════════════════════════════════════════════════════
print("── Loading streams ──")
streams = [SFXStream.load(STREAM_DIR / f"delay_{d}.sfx") for d in DELAYS]
s = SFXStream.concat(streams)
h1, h2 = s.split_random(n_parts=2, seed=SEED)
print(f"Total: {s.n_crystals} crystals, split: {h1.n_crystals}/{h2.n_crystals}")

# ══════════════════════════════════════════════════════════════
# Partialator (parse pre-computed results)
# ══════════════════════════════════════════════════════════════
from torchref.symmetry import canonicalize_hkl

print("\n── Partialator ──")
for label, h1_path, h2_path, merge_path in [
    ("unity", "ind_half1.hkl", "ind_half2.hkl", "neg.hkl"),
    ("xsphere", "xsphere.hkl1", "xsphere.hkl2", "xsphere.hkl"),
]:
    p1 = PART_DIR / h1_path
    p2 = PART_DIR / h2_path
    pm = PART_DIR / merge_path
    if not p1.exists():
        continue
    for tag, path in [("half1", p1), ("half2", p2), ("merge", pm)]:
        hkl, I = parse_crystfel_hkl(path)
        hkl, _, _, si = canonicalize_hkl(hkl.int(), "P -1", include_friedel=True)
        F = I[si].clamp(min=0).sqrt()
        torch.save({"hkl": hkl.to(torch.int16), "F": F},
                    OUTDIR / f"part_{label}_{tag}.pt")
        print(f"  Saved part_{label}_{tag}.pt ({len(F)} refl)")

# ══════════════════════════════════════════════════════════════
# GenGauss (log-space IRLS) — halves + full
# ══════════════════════════════════════════════════════════════
print("\n── GenGauss ──")
FIT_KW = dict(min_obs=20, n_smooth_iter=15, smooth_lambda=0.3)

for label, stream in [("half1", h1), ("half2", h2), ("full", s)]:
    print(f"  {label}...")
    dist = fit_gengauss(stream, n_em_cycles=10, scale_crystals=True, **FIT_KW)
    torch.save({"hkl": dist.hkl, "I0": dist.I0, "alpha": dist.alpha,
                "beta": dist.beta, "d_spacing": dist.d_spacing,
                "n_obs": dist.n_obs},
               OUTDIR / f"gg_{label}.pt")
    print(f"  Saved gg_{label}.pt ({len(dist.I0)} HKLs, med(α)={dist.alpha.median():.4f})")

# ══════════════════════════════════════════════════════════════
# EM Poisson — halves + full
# ══════════════════════════════════════════════════════════════
print("\n── EM Poisson ──")
cfg = EMConfig(n_em_cycles=10, min_obs=5, alpha_prior_lambda=0.0)

for label, stream in [("half1", h1), ("half2", h2), ("full", s)]:
    print(f"  {label}...")
    dist = fit_em(stream, config=cfg, corrections=[ScaleBCorrection()], device=device)
    torch.save({"hkl": dist.hkl, "I0": dist.I0, "alpha": dist.alpha,
                "beta": dist.beta, "d_spacing": dist.d_spacing,
                "n_obs": dist.n_obs},
               OUTDIR / f"em_{label}.pt")
    print(f"  Saved em_{label}.pt ({len(dist.I0)} HKLs, med(α)={dist.alpha.median():.4f})")

torch.save(CELL, OUTDIR / "cell.pt")
print("\nDone.")
