#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH -o /das/work/p17/p17490/Peter/torchsx/sfx_scan/split_half_distribution.out
#SBATCH -c 16
#SBATCH -p day
#SBATCH -t 2:00:00

"""Split-half validation: Partialator vs GenGauss vs GenGauss+EM vs GenGauss+EM+Scale."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from torchsx.stream import SFXStream
from torchsx.merge._distribution import (
    HKLDistribution,
    fit_gengauss,
    compare_distributions,
    format_dist_table,
)
from torchsx.merge._stats import (
    match_halves, cc_half, cc_star, r_split, spearman_cc,
    shell_stats, format_shell_table, MatchedHalves,
)

# ── Configuration ──────────────────────────────────────────────
STREAM_DIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/streams")
PART_DIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/partialator")
DELAYS = [-91, -136, -182, -227, -273]
SPACEGROUP = "P-1"
OUTDIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/resolution_study")
N_WORKERS = 16
SEED = 42
CELL = torch.tensor([14.97, 18.85, 18.89, 89.37, 84.94, 67.84])
N_EM = 10
FIT_KW = dict(min_obs=20, n_shells=20, n_smooth_iter=15, smooth_lambda=0.3)


# ── Helpers ────────────────────────────────────────────────────
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


def to_matched(f1, f2, cell, d_min=0.9, d_max=10.0):
    """HKLDistribution pair → MatchedHalves (F = sqrt(I0))."""
    return match_halves(f1.hkl, f1.I0.clamp(min=0).sqrt(),
                        f2.hkl, f2.I0.clamp(min=0).sqrt(),
                        cell, d_min=d_min, d_max=d_max)


def d_to_x(d):
    return 1.0 / np.asarray(d) ** 2


def setup_res_axis(ax, xlabel=True):
    tick_d = np.array([10, 5, 3, 2, 1.5, 1.2, 1.0, 0.9])
    tick_d = tick_d[(tick_d >= 0.9) & (tick_d <= 10.0)]
    ax.set_xticks(d_to_x(tick_d))
    ax.set_xticklabels([f"{d:g}" for d in tick_d])
    ax.set_xlim(d_to_x(10.0), d_to_x(0.9))
    if xlabel:
        ax.set_xlabel("Resolution (Å)")


def cc_I_vs_partialator(fit, hkl_ref, I_ref, cell):
    """CC between GenGauss I0 and partialator I for matched HKLs."""
    matched = match_halves(fit.hkl, fit.I0, hkl_ref, I_ref,
                           cell, d_min=0.9, d_max=10.0)
    if matched.F1.numel() < 2:
        return 0.0, matched
    return cc_half(matched), matched


# ══════════════════════════════════════════════════════════════
# 1. Load data
# ══════════════════════════════════════════════════════════════
streams = []
for d in DELAYS:
    stem = STREAM_DIR / f"delay_{d}.stream"
    cache = stem.with_suffix(".sfx")
    print(f"Loading {stem.name} ...")
    sx = SFXStream.from_stream(stem, spacegroup=SPACEGROUP,
                               cache=cache, n_workers=N_WORKERS)
    print(f"  {sx.n_crystals} crystals, {sx.n_observations} obs")
    streams.append(sx)

s = SFXStream.concat(streams)
print(f"\nTotal: {s.n_crystals} crystals, {s.n_observations} observations")

# ══════════════════════════════════════════════════════════════
# 2. Parse partialator results
# ══════════════════════════════════════════════════════════════
from torchref.symmetry import canonicalize_hkl

def load_partialator(half1_path, half2_path, merge_path=None):
    """Load partialator split-half + optional merge, canonicalize HKLs."""
    hkl1, I1 = parse_crystfel_hkl(half1_path)
    hkl2, I2 = parse_crystfel_hkl(half2_path)
    F1, F2 = I1.clamp(min=0).sqrt(), I2.clamp(min=0).sqrt()
    hkl1, _, _, s1 = canonicalize_hkl(hkl1.int(), "P -1", include_friedel=True)
    hkl2, _, _, s2 = canonicalize_hkl(hkl2.int(), "P -1", include_friedel=True)
    F1, F2 = F1[s1], F2[s2]
    hkl1, hkl2 = hkl1.to(torch.int16), hkl2.to(torch.int16)
    matched = match_halves(hkl1, F1, hkl2, F2, CELL, d_min=0.9, d_max=10.0)
    hkl_m, I_m = None, None
    if merge_path is not None and merge_path.exists():
        hkl_m, I_m = parse_crystfel_hkl(merge_path)
        hkl_m, _, _, sm = canonicalize_hkl(hkl_m.int(), "P -1", include_friedel=True)
        I_m = I_m[sm]; hkl_m = hkl_m.to(torch.int16)
    return matched, hkl_m, I_m


# ── Partialator unity ──
print(f"\n── Partialator unity ──")
part_matched, hkl_merge, I_merge = load_partialator(
    PART_DIR / "ind_half1.hkl", PART_DIR / "ind_half2.hkl",
    PART_DIR / "neg.hkl",
)
part_cc = cc_half(part_matched)
part_r = r_split(part_matched)
part_sp = spearman_cc(part_matched)
part_shells = shell_stats(part_matched, n_shells=20)
print(f"  {part_matched.F1.numel()} matched, CC1/2={part_cc:.4f}, Sp={part_sp:.4f}, R={part_r:.4f}")
print(f"  Full merge: {hkl_merge.shape[0]} reflections")

# ── Partialator xsphere (if available) ──
# Use internal split (xsphere.hkl1/hkl2) — independent half runs crash on this data
xsph_cc, xsph_r, xsph_sp, xsph_shells = 0.0, 0.0, 0.0, []
xsph_merge_hkl, xsph_merge_I = None, None
HAS_XSPHERE = (PART_DIR / "xsphere.hkl1").exists()
if HAS_XSPHERE:
    print(f"\n── Partialator xsphere (internal split) ──")
    xsph_matched, xsph_merge_hkl, xsph_merge_I = load_partialator(
        PART_DIR / "xsphere.hkl1", PART_DIR / "xsphere.hkl2",
        PART_DIR / "xsphere.hkl",
    )
    xsph_cc = cc_half(xsph_matched)
    xsph_r = r_split(xsph_matched)
    xsph_sp = spearman_cc(xsph_matched)
    xsph_shells = shell_stats(xsph_matched, n_shells=20)
    print(f"  {xsph_matched.F1.numel()} matched, CC1/2={xsph_cc:.4f}, Sp={xsph_sp:.4f}, R={xsph_r:.4f}")
else:
    print(f"\n  (xsphere results not yet available — run partialator/run_xsphere.sh)")

# ══════════════════════════════════════════════════════════════
# 3. Split into halves
# ══════════════════════════════════════════════════════════════
h1, h2 = s.split_random(2, seed=SEED)
print(f"\nSplit: half1={h1.n_crystals} crystals, half2={h2.n_crystals} crystals")

# ══════════════════════════════════════════════════════════════
# 4. Method A: Single-pass GenGauss
# ══════════════════════════════════════════════════════════════
print(f"\n── (A) Single-pass GenGauss ──")
fA1 = fit_gengauss(h1, n_em_cycles=0, **FIT_KW)
fA2 = fit_gengauss(h2, n_em_cycles=0, **FIT_KW)
mA = to_matched(fA1, fA2, CELL)
ccA, rA, spA = cc_half(mA), r_split(mA), spearman_cc(mA)
shA = shell_stats(mA, n_shells=20)
print(f"  CC1/2={ccA:.4f}, Sp={spA:.4f}, R={rA:.4f}, β={fA1.beta.median():.3f}")

# Full-dataset fit for partialator comparison
fA_full = fit_gengauss(s, n_em_cycles=0, **FIT_KW)
ccA_part, _ = cc_I_vs_partialator(fA_full, hkl_merge, I_merge, CELL)
print(f"  CC(I0 vs partialator) = {ccA_part:.4f}")

# ══════════════════════════════════════════════════════════════
# 5. Method B: EM-refined GenGauss (no scaling)
# ══════════════════════════════════════════════════════════════
print(f"\n── (B) GenGauss + EM×{N_EM} ──")
fB1 = fit_gengauss(h1, n_em_cycles=N_EM, scale_crystals=False, **FIT_KW)
fB2 = fit_gengauss(h2, n_em_cycles=N_EM, scale_crystals=False, **FIT_KW)
mB = to_matched(fB1, fB2, CELL)
ccB, rB, spB = cc_half(mB), r_split(mB), spearman_cc(mB)
shB = shell_stats(mB, n_shells=20)
print(f"  CC1/2={ccB:.4f}, Sp={spB:.4f}, R={rB:.4f}")

fB_full = fit_gengauss(s, n_em_cycles=N_EM, scale_crystals=False, **FIT_KW)
ccB_part, _ = cc_I_vs_partialator(fB_full, hkl_merge, I_merge, CELL)
print(f"  CC(I0 vs partialator) = {ccB_part:.4f}")

# ══════════════════════════════════════════════════════════════
# 6. Method C: EM + per-crystal scale + B-factor
# ══════════════════════════════════════════════════════════════
print(f"\n── (C) GenGauss + EM×{N_EM} + crystal scaling ──")
fC1 = fit_gengauss(h1, n_em_cycles=N_EM, scale_crystals=True, **FIT_KW)
fC2 = fit_gengauss(h2, n_em_cycles=N_EM, scale_crystals=True, **FIT_KW)
mC = to_matched(fC1, fC2, CELL)
ccC, rC, spC = cc_half(mC), r_split(mC), spearman_cc(mC)
shC = shell_stats(mC, n_shells=20)
print(f"  CC1/2={ccC:.4f}, Sp={spC:.4f}, R={rC:.4f}")

fC_full = fit_gengauss(s, n_em_cycles=N_EM, scale_crystals=True, **FIT_KW)
ccC_part, _ = cc_I_vs_partialator(fC_full, hkl_merge, I_merge, CELL)
print(f"  CC(I0 vs partialator) = {ccC_part:.4f}")

# ══════════════════════════════════════════════════════════════
# 6b. Method D: C + multiplicity-weighted smoothing
# ══════════════════════════════════════════════════════════════
print(f"\n── (D) GenGauss + EM×{N_EM} + scale + weighted smooth ──")
fD1 = fit_gengauss(h1, n_em_cycles=N_EM, scale_crystals=True, weighted_smooth=True, **FIT_KW)
fD2 = fit_gengauss(h2, n_em_cycles=N_EM, scale_crystals=True, weighted_smooth=True, **FIT_KW)
mD = to_matched(fD1, fD2, CELL)
ccD, rD, spD = cc_half(mD), r_split(mD), spearman_cc(mD)
shD = shell_stats(mD, n_shells=20)
print(f"  CC1/2={ccD:.4f}, Sp={spD:.4f}, R={rD:.4f}")

fD_full = fit_gengauss(s, n_em_cycles=N_EM, scale_crystals=True, weighted_smooth=True, **FIT_KW)
ccD_part, _ = cc_I_vs_partialator(fD_full, hkl_merge, I_merge, CELL)
print(f"  CC(I0 vs partialator) = {ccD_part:.4f}")

# ══════════════════════════════════════════════════════════════
# 6c. Method E: C + crystal quality weighting
# ══════════════════════════════════════════════════════════════
print(f"\n── (E) GenGauss + EM×{N_EM} + scale + crystal quality ──")
fE1 = fit_gengauss(h1, n_em_cycles=N_EM, scale_crystals=True, crystal_quality_weights=True, **FIT_KW)
fE2 = fit_gengauss(h2, n_em_cycles=N_EM, scale_crystals=True, crystal_quality_weights=True, **FIT_KW)
mE = to_matched(fE1, fE2, CELL)
ccE, rE, spE = cc_half(mE), r_split(mE), spearman_cc(mE)
shE = shell_stats(mE, n_shells=20)
print(f"  CC1/2={ccE:.4f}, Sp={spE:.4f}, R={rE:.4f}")

fE_full = fit_gengauss(s, n_em_cycles=N_EM, scale_crystals=True, crystal_quality_weights=True, **FIT_KW)
ccE_part, _ = cc_I_vs_partialator(fE_full, hkl_merge, I_merge, CELL)
print(f"  CC(I0 vs partialator) = {ccE_part:.4f}")

# ══════════════════════════════════════════════════════════════
# 6d. Method F: C + crystal quality cutoff (reject bottom 10%)
# ══════════════════════════════════════════════════════════════
CUTOFF = 0.10
print(f"\n── (F) GenGauss + EM×{N_EM} + scale + quality cutoff {CUTOFF:.0%} ──")
fF1 = fit_gengauss(h1, n_em_cycles=N_EM, scale_crystals=True, crystal_quality_cutoff=CUTOFF, **FIT_KW)
fF2 = fit_gengauss(h2, n_em_cycles=N_EM, scale_crystals=True, crystal_quality_cutoff=CUTOFF, **FIT_KW)
mF = to_matched(fF1, fF2, CELL)
ccF, rF, spF = cc_half(mF), r_split(mF), spearman_cc(mF)
shF = shell_stats(mF, n_shells=20)
print(f"  CC1/2={ccF:.4f}, Sp={spF:.4f}, R={rF:.4f}")

fF_full = fit_gengauss(s, n_em_cycles=N_EM, scale_crystals=True, crystal_quality_cutoff=CUTOFF, **FIT_KW)
ccF_part, _ = cc_I_vs_partialator(fF_full, hkl_merge, I_merge, CELL)
print(f"  CC(I0 vs partialator) = {ccF_part:.4f}")

# ══════════════════════════════════════════════════════════════
# 6e. Method G: C + posterior crystal weighting
# ══════════════════════════════════════════════════════════════
print(f"\n── (G) GenGauss + EM×{N_EM} + scale + posterior P(good|c) ──")
fG1 = fit_gengauss(h1, n_em_cycles=N_EM, scale_crystals=True, crystal_posterior=True, **FIT_KW)
fG2 = fit_gengauss(h2, n_em_cycles=N_EM, scale_crystals=True, crystal_posterior=True, **FIT_KW)
mG = to_matched(fG1, fG2, CELL)
ccG, rG, spG = cc_half(mG), r_split(mG), spearman_cc(mG)
shG = shell_stats(mG, n_shells=20)
print(f"  CC1/2={ccG:.4f}, Sp={spG:.4f}, R={rG:.4f}")

fG_full = fit_gengauss(s, n_em_cycles=N_EM, scale_crystals=True, crystal_posterior=True, **FIT_KW)
ccG_part, _ = cc_I_vs_partialator(fG_full, hkl_merge, I_merge, CELL)
print(f"  CC(I0 vs partialator) = {ccG_part:.4f}")

# ══════════════════════════════════════════════════════════════
# 6f. Method H: ScaleB + PanelCorrection (constant)
# ══════════════════════════════════════════════════════════════
from torchsx.merge._corrections import ScaleBCorrection, PanelCorrection

print(f"\n── (H) GenGauss + EM×{N_EM} + ScaleB + Panel (32 params) ──")
corr_H = [ScaleBCorrection(), PanelCorrection(fit_gradient=False)]
fH1 = fit_gengauss(h1, n_em_cycles=N_EM, corrections=corr_H, **FIT_KW)
corr_H2 = [ScaleBCorrection(), PanelCorrection(fit_gradient=False)]
fH2 = fit_gengauss(h2, n_em_cycles=N_EM, corrections=corr_H2, **FIT_KW)
mH = to_matched(fH1, fH2, CELL)
ccH, rH, spH = cc_half(mH), r_split(mH), spearman_cc(mH)
shH = shell_stats(mH, n_shells=20)
print(f"  CC1/2={ccH:.4f}, Sp={spH:.4f}, R={rH:.4f}")

corr_H_full = [ScaleBCorrection(), PanelCorrection(fit_gradient=False)]
fH_full = fit_gengauss(s, n_em_cycles=N_EM, corrections=corr_H_full, **FIT_KW)
ccH_part, _ = cc_I_vs_partialator(fH_full, hkl_merge, I_merge, CELL)
print(f"  CC(I0 vs partialator) = {ccH_part:.4f}")

# ══════════════════════════════════════════════════════════════
# 6g. Method I: ScaleB + PanelCorrection (with gradient)
# ══════════════════════════════════════════════════════════════
print(f"\n── (I) GenGauss + EM×{N_EM} + ScaleB + Panel+grad (96 params) ──")
corr_I = [ScaleBCorrection(), PanelCorrection(fit_gradient=True)]
fI1 = fit_gengauss(h1, n_em_cycles=N_EM, corrections=corr_I, **FIT_KW)
corr_I2 = [ScaleBCorrection(), PanelCorrection(fit_gradient=True)]
fI2 = fit_gengauss(h2, n_em_cycles=N_EM, corrections=corr_I2, **FIT_KW)
mI = to_matched(fI1, fI2, CELL)
ccI, rI, spI = cc_half(mI), r_split(mI), spearman_cc(mI)
shI = shell_stats(mI, n_shells=20)
print(f"  CC1/2={ccI:.4f}, Sp={spI:.4f}, R={rI:.4f}")

corr_I_full = [ScaleBCorrection(), PanelCorrection(fit_gradient=True)]
fI_full = fit_gengauss(s, n_em_cycles=N_EM, corrections=corr_I_full, **FIT_KW)
ccI_part, _ = cc_I_vs_partialator(fI_full, hkl_merge, I_merge, CELL)
print(f"  CC(I0 vs partialator) = {ccI_part:.4f}")

# ══════════════════════════════════════════════════════════════
# 6h. Method J: JOINT Poisson+LogNormal (fit_joint_em)
# ══════════════════════════════════════════════════════════════
from torchsx.merge._noise import PoissonNoiseModel
from torchsx.merge._em import fit_joint_em, JointConfig

JOINT_CFG = JointConfig(n_em_cycles=5, reg_lambda=0.1, max_iter=30, warmup_iter=5,
                        min_obs=20, n_shells=20)

print(f"\n── (J) JOINT: ScaleB + Poisson (λ={JOINT_CFG.reg_lambda}) ──")
noise1 = PoissonNoiseModel()
fJ1 = fit_joint_em(h1, config=JOINT_CFG, corrections=[ScaleBCorrection()],
                   poisson_noise=noise1)
noise2 = PoissonNoiseModel()
fJ2 = fit_joint_em(h2, config=JOINT_CFG, corrections=[ScaleBCorrection()],
                   poisson_noise=noise2)
mJ = to_matched(fJ1, fJ2, CELL)
ccJ, rJ, spJ = cc_half(mJ), r_split(mJ), spearman_cc(mJ)
shJ = shell_stats(mJ, n_shells=20)
print(f"  CC1/2={ccJ:.4f}, Sp={spJ:.4f}, R={rJ:.4f}")
print(f"  med(α): half1={fJ1.alpha.median():.4f}, half2={fJ2.alpha.median():.4f}")
print(f"  I₀ < 1: half1={int((fJ1.I0 < 1).sum())}, half2={int((fJ2.I0 < 1).sum())}")

noise_full = PoissonNoiseModel()
fJ_full = fit_joint_em(s, config=JOINT_CFG, corrections=[ScaleBCorrection()],
                       poisson_noise=noise_full)
ccJ_part, _ = cc_I_vs_partialator(fJ_full, hkl_merge, I_merge, CELL)
print(f"  CC(I0 vs partialator) = {ccJ_part:.4f}")

# ══════════════════════════════════════════════════════════════
# 7. Summary tables
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*98}")
print(f"{'Method':>35s} {'CC1/2':>7s} {'Spear':>7s} {'CC*':>7s} {'R_split':>8s} {'CC(part)':>8s}")
print(f"{'-'*98}")
print(f"{'Partialator unity':>35s} {part_cc:7.4f} {part_sp:7.4f} {cc_star(part_cc):7.4f} {part_r:8.4f} {'—':>8s}")
if HAS_XSPHERE:
    print(f"{'Partialator xsphere':>35s} {xsph_cc:7.4f} {xsph_sp:7.4f} {cc_star(xsph_cc):7.4f} {xsph_r:8.4f} {'—':>8s}")
print(f"{'(A) GenGauss':>35s} {ccA:7.4f} {spA:7.4f} {cc_star(ccA):7.4f} {rA:8.4f} {ccA_part:8.4f}")
print(f"{'(B) GenGauss + EM':>35s} {ccB:7.4f} {spB:7.4f} {cc_star(ccB):7.4f} {rB:8.4f} {ccB_part:8.4f}")
print(f"{'(C) GenGauss + EM + scale':>35s} {ccC:7.4f} {spC:7.4f} {cc_star(ccC):7.4f} {rC:8.4f} {ccC_part:8.4f}")
print(f"{'(D) C + weighted smooth':>35s} {ccD:7.4f} {spD:7.4f} {cc_star(ccD):7.4f} {rD:8.4f} {ccD_part:8.4f}")
print(f"{'(E) C + crystal quality':>35s} {ccE:7.4f} {spE:7.4f} {cc_star(ccE):7.4f} {rE:8.4f} {ccE_part:8.4f}")
print(f"{'(F) C + quality cutoff 10%':>35s} {ccF:7.4f} {spF:7.4f} {cc_star(ccF):7.4f} {rF:8.4f} {ccF_part:8.4f}")
print(f"{'(G) C + posterior P(good|c)':>35s} {ccG:7.4f} {spG:7.4f} {cc_star(ccG):7.4f} {rG:8.4f} {ccG_part:8.4f}")
print(f"{'(H) ScaleB + Panel':>35s} {ccH:7.4f} {spH:7.4f} {cc_star(ccH):7.4f} {rH:8.4f} {ccH_part:8.4f}")
print(f"{'(I) ScaleB + Panel+grad':>35s} {ccI:7.4f} {spI:7.4f} {cc_star(ccI):7.4f} {rI:8.4f} {ccI_part:8.4f}")
print(f"{'(J) LINEAR + ScaleB + Poisson':>35s} {ccJ:7.4f} {spJ:7.4f} {cc_star(ccJ):7.4f} {rJ:8.4f} {ccJ_part:8.4f}")
print(f"{'='*98}")

# Per-shell comparison: Pearson CC
print(f"\nPearson CC₁/₂ per shell:")
print(f"{'d_max':>7s} {'d_min':>6s}  {'CC_u':>6s}", end="")
if HAS_XSPHERE:
    print(f" {'CC_x':>6s}", end="")
print(f" {'CC_C':>6s} {'CC_I':>6s} {'CC_J':>6s}")
print("-" * (62 if HAS_XSPHERE else 55))
n_sh = min(len(part_shells), len(shC), len(shI), len(shJ))
for i in range(n_sh):
    p, c, ii, j = part_shells[i], shC[i], shI[i], shJ[i]
    print(f"{p.d_max:7.2f} {p.d_min:6.2f}  {p.cc_half:6.3f}", end="")
    if HAS_XSPHERE and i < len(xsph_shells):
        print(f" {xsph_shells[i].cc_half:6.3f}", end="")
    print(f" {c.cc_half:6.3f} {ii.cc_half:6.3f} {j.cc_half:6.3f}")

# Per-shell comparison: Spearman CC
print(f"\nSpearman CC₁/₂ per shell:")
print(f"{'d_max':>7s} {'d_min':>6s}  {'Sp_u':>6s}", end="")
if HAS_XSPHERE:
    print(f" {'Sp_x':>6s}", end="")
print(f" {'Sp_C':>6s} {'Sp_I':>6s} {'Sp_J':>6s}")
print("-" * (62 if HAS_XSPHERE else 55))
for i in range(n_sh):
    p, c, ii, j = part_shells[i], shC[i], shI[i], shJ[i]
    print(f"{p.d_max:7.2f} {p.d_min:6.2f}  {p.spearman:6.3f}", end="")
    if HAS_XSPHERE and i < len(xsph_shells):
        print(f" {xsph_shells[i].spearman:6.3f}", end="")
    print(f" {c.spearman:6.3f} {ii.spearman:6.3f} {j.spearman:6.3f}")

# ══════════════════════════════════════════════════════════════
# 8. Plots
# ══════════════════════════════════════════════════════════════
OUTDIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# (a) CC1/2 vs resolution
ax = axes[0, 0]
plot_methods = [
    (part_shells, "Part. unity", "C3", "D"),
]
if HAS_XSPHERE and xsph_shells:
    plot_methods.append((xsph_shells, "Part. xsphere", "C4", "P"))
plot_methods += [
    (shA, "(A) GenGauss", "C0", "o"),
    (shB, f"(B) +EM×{N_EM}", "C2", "s"),
    (shC, f"(C) +EM+scale", "C1", "^"),
    (shD, f"(D) +wt.smooth", "C5", "v"),
    (shE, f"(E) +cryst.qual", "C6", "X"),
    (shF, f"(F) +qual.cut10%", "C7", "d"),
    (shG, f"(G) +posterior", "C8", "h"),
]
for shells, label, color, marker in plot_methods:
    d_mid = [0.5 * (sh.d_min + sh.d_max) for sh in shells]
    ax.plot(d_to_x(d_mid), [sh.cc_half for sh in shells],
            f"{marker}-", color=color, markersize=4, label=label)
ax.set_ylabel("CC₁/₂"); ax.set_ylim(-0.1, 1.05)
ax.set_title("Split-half CC₁/₂"); ax.legend(fontsize=6)
setup_res_axis(ax)

# (b) R_split vs resolution
ax = axes[0, 1]
for shells, label, color, marker in plot_methods:
    d_mid = [0.5 * (sh.d_min + sh.d_max) for sh in shells]
    ax.plot(d_to_x(d_mid), [sh.r_split for sh in shells],
            f"{marker}-", color=color, markersize=4, label=label)
ax.set_ylabel("R_split"); ax.set_title("R_split"); ax.legend(fontsize=6)
setup_res_axis(ax)

# (c) I0 vs partialator merged I — scatter
ax = axes[0, 2]
_, matched_part = cc_I_vs_partialator(fC_full, hkl_merge, I_merge, CELL)
if matched_part.F1.numel() > 0:
    x = torch.log10(matched_part.F1.clamp(min=1e-3)).numpy()  # GenGauss I0
    y = torch.log10(matched_part.F2.clamp(min=1e-3)).numpy()  # partialator I
    ax.scatter(x, y, s=1, alpha=0.15, rasterized=True)
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, "k--", lw=0.8)
ax.set_xlabel("log₁₀(I0 GenGauss+EM+scale)")
ax.set_ylabel("log₁₀(I partialator)")
ax.set_title(f"vs partialator merge (CC={ccC_part:.3f})")
ax.set_aspect("equal")

# (d) I0 half1 vs half2 — method C (EM+scale)
ax = axes[1, 0]
compC = compare_distributions(fC1, fC2, n_shells=20)
pos = (compC.I0_1 > 0) & (compC.I0_2 > 0)
if pos.sum() > 0:
    sc = ax.scatter(torch.log10(compC.I0_1[pos]).numpy(),
                    torch.log10(compC.I0_2[pos]).numpy(),
                    c=d_to_x(compC.d_spacing[pos].numpy()),
                    s=1, alpha=0.2, cmap="viridis", rasterized=True)
    lims = [min(torch.log10(compC.I0_1[pos]).min().item(),
                torch.log10(compC.I0_2[pos]).min().item()),
            max(torch.log10(compC.I0_1[pos]).max().item(),
                torch.log10(compC.I0_2[pos]).max().item())]
    ax.plot(lims, lims, "k--", lw=0.8)
ax.set_xlabel("log₁₀(I0) half 1"); ax.set_ylabel("log₁₀(I0) half 2")
ax.set_title(f"(C) EM+scale I0 (CC={compC.cc_I0:.3f})")
ax.set_aspect("equal")

# (e) CC(I0) vs resolution — all GenGauss methods
ax = axes[1, 1]
compA = compare_distributions(fA1, fA2, n_shells=20)
compB = compare_distributions(fB1, fB2, n_shells=20)
for comp, label, color in [
    (compA, "(A) GenGauss", "C0"),
    (compB, f"(B) +EM", "C2"),
    (compC, f"(C) +EM+scale", "C1"),
]:
    if comp.shells:
        d_mid = [0.5 * (sh.d_min + sh.d_max) for sh in comp.shells]
        ax.plot(d_to_x(d_mid), [sh.cc_I0 for sh in comp.shells],
                "o-", color=color, markersize=3, label=label)
ax.set_ylabel("CC(I0)"); ax.set_title("Distribution CC(I0) vs resolution")
ax.legend(fontsize=7)
setup_res_axis(ax)

# (f) β from both halves (method C)
ax = axes[1, 2]
for fit, label, color in [(fC1, "Half 1", "C0"), (fC2, "Half 2", "C1")]:
    n_bins = 20
    rank = fit.d_spacing.argsort().argsort()
    bins = (rank.float() / len(rank) * n_bins).long().clamp(max=n_bins - 1)
    for b_idx in range(n_bins):
        bm = bins == b_idx
        if bm.sum() > 5:
            ax.plot(d_to_x(fit.d_spacing[bm].median().item()),
                    fit.beta[bm].median().item(),
                    "o", color=color, markersize=4, alpha=0.8)
    ax.plot([], [], "o", color=color, label=label)
ax.axhline(2, ls=":", color="k", alpha=0.3, label="β=2")
ax.set_ylabel("β"); ax.set_title("(C) β vs resolution")
ax.legend(fontsize=8)
setup_res_axis(ax)

fig.suptitle("Method comparison: Partialator vs GenGauss variants", fontsize=13)
fig.tight_layout()
fig.savefig(str(OUTDIR / "split_half_comparison.png"), dpi=150)
plt.close(fig)
print(f"\nSaved {OUTDIR / 'split_half_comparison.png'}")
