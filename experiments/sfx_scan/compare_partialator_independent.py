#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u
"""Compare CrystFEL partialator split-half results with torchsx metrics."""

from pathlib import Path

import torch

from torchsx.merge._stats import (
    MatchedHalves,
    cc_half,
    cc_star,
    r_split,
    shell_stats,
    format_shell_table,
    match_halves,
    write_shell_csv
)
from torchsx.merge._plots import (
    plot_cc_vs_resolution,
    plot_rsplit_vs_resolution,
    plot_f_scatter,
)

PART_DIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/partialator")
PLOT_DIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan")

# Median cell from our data (Å, degrees) — triclinic P -1
CELL = torch.tensor([14.97, 18.85, 18.89, 89.37, 84.94, 67.84])


def parse_crystfel_hkl(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Parse a CrystFEL .hkl file → (hkl, F) tensors.

    Returns sqrt(I) as F (intensities → amplitudes).
    """
    h_list, k_list, l_list, I_list = [], [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("CrystFEL") or line.startswith("Symmetry"):
                continue
            if line.startswith("h ") or line.startswith("End"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                h_list.append(int(parts[0]))
                k_list.append(int(parts[1]))
                l_list.append(int(parts[2]))
                I_list.append(float(parts[3]))
            except ValueError:
                continue

    hkl = torch.tensor(list(zip(h_list, k_list, l_list)), dtype=torch.int16)
    I = torch.tensor(I_list, dtype=torch.float32)
    F = I.clamp(min=0).sqrt()
    return hkl, F


# -- Parse partialator half-datasets --
print("Parsing partialator HKL files ...")
hkl1, F1 = parse_crystfel_hkl(PART_DIR / "ind_half1.hkl")
hkl2, F2 = parse_crystfel_hkl(PART_DIR / "ind_half2.hkl")
print(f"  Half 1: {hkl1.shape[0]} reflections")
print(f"  Half 2: {hkl2.shape[0]} reflections")

# Canonicalize HKL to ensure consistency with torchsx convention
from torchref.symmetry import canonicalize_hkl

hkl1, _, _, sort1 = canonicalize_hkl(hkl1.int(), "P -1", include_friedel=True)
hkl2, _, _, sort2 = canonicalize_hkl(hkl2.int(), "P -1", include_friedel=True)
F1 = F1[sort1]
F2 = F2[sort2]
hkl1 = hkl1.to(torch.int16)
hkl2 = hkl2.to(torch.int16)

# -- Match and compute stats --
matched = match_halves(hkl1, F1, hkl2, F2, CELL, d_min=0.9, d_max=10.0)
print(f"  Matched: {matched.F1.numel()} reflections")

overall_cc = cc_half(matched)
overall_r = r_split(matched)
shells = shell_stats(matched, n_shells=20)

print(f"\n=== CrystFEL partialator split-half (unity model) ===\n")
print(format_shell_table(shells, overall_cc, overall_r))

# -- Plots --
plot_cc_vs_resolution(shells, PLOT_DIR / "partialator_cc_vs_resolution_ind.png")
plot_rsplit_vs_resolution(shells, PLOT_DIR / "partialator_rsplit_vs_resolution_ind.png")
plot_f_scatter(matched, PLOT_DIR / "partialator_f_scatter_ind.png")
print(f"\nPlots saved to {PLOT_DIR}")

write_shell_csv(shells, PLOT_DIR / "partialator_shell_stats_ind.csv")