#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH -o /das/work/p17/p17490/Peter/torchsx/sfx_scan/neural_correction_split_half.out
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 4:00:00
#SBATCH --mem=64G

"""Train-Correct-Compare: test whether Ewald-sphere-based neural corrections
improve split-half statistics (CC1/2, R_split).

Workflow:
  1. Load streams
  2. Run inference with saved no-peak model (or retrain if missing)
  3. Apply corrections: I, background, sigma all divided by c_j so Poisson noise rescales
  4. Split original and corrected data into halves (same seed)
  5. Fit GenGauss+EM on each pair of halves
  6. Compare CC1/2, R_split with and without corrections

The model uses NO peak/background features — only physics-based inputs:
  excitation_error, q_lab, s², fs, ss, panel_id, profile_rad
"""

import sys
import os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "/das/work/p17/p17490/Peter/torchsx")

from dataclasses import replace
from pathlib import Path
import math
import torch

from torchsx.stream import SFXStream
from torchsx.merge._distribution import fit_gengauss
from torchsx.merge._noise import PoissonNoiseModel
from torchsx.merge._stats import match_halves, cc_half, r_split, spearman_cc, shell_stats
from torchsx.merge._neural_model import (
    CrystalTransformerCorrection,
    CorrectionFeatures,
    extract_features,
)

# ── Configuration ──
STREAM_DIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/streams")
DELAYS = [-91, -136, -182, -227, -273, -45]
SPACEGROUP = "P-1"
OUTDIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/neural_correction")
MODEL_PATH = OUTDIR / "neural_v2_nopeak.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CELL = torch.tensor([14.97, 18.85, 18.89, 89.37, 84.94, 67.84])
SEED = 42
FIT_KW = dict(min_obs=20, n_shells=20, n_smooth_iter=15, smooth_lambda=0.3)
N_EM = 10

print(f"Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════
# 1. Load streams
# ══════════════════════════════════════════════════════════════
print("\n── Loading streams ──")
streams = []
for d in DELAYS:
    p = STREAM_DIR / f"delay_{d}.stream"
    print(f"  {p.name} ...", end=" ", flush=True)
    st = SFXStream.from_stream(str(p), spacegroup=SPACEGROUP, n_workers=1)
    print(f"{st.n_crystals} crystals, {st.n_observations} obs")
    streams.append(st)

s = SFXStream.concat(streams)
core = s._core
print(f"\nTotal: {s.n_crystals} crystals, {s.n_observations} observations")

# ══════════════════════════════════════════════════════════════
# 2. Extract physics features (no peak/background)
# ══════════════════════════════════════════════════════════════
print("\n── Extracting features ──")
features_all = extract_features(core, wavelength=0.9567)

# Sort by crystal for batch inference
crystal_id_all = core.crystal_id.long()
sort_idx = crystal_id_all.argsort(stable=True)
crystal_id_sorted = crystal_id_all[sort_idx]
sort_cpu = sort_idx.cpu()

# Zero-out peak and background
feat_sorted = CorrectionFeatures(
    peak=torch.zeros(core.n_observations),
    background=torch.zeros(core.n_observations),
    s_sq=features_all.s_sq[sort_cpu],
    excitation_error=features_all.excitation_error[sort_cpu],
    q_lab=features_all.q_lab[sort_cpu],
    fs=features_all.fs[sort_cpu],
    ss=features_all.ss[sort_cpu],
    panel_id=features_all.panel_id[sort_cpu],
    profile_rad=features_all.profile_rad[sort_cpu],
    crystal_id=crystal_id_sorted,
)
print(f"  excitation_error: med={features_all.excitation_error.median():.4f} Å⁻¹")

# Crystal offsets for batching
n_cryst = int(crystal_id_sorted.max().item()) + 1
obs_per_crystal = torch.zeros(n_cryst, dtype=torch.long)
obs_per_crystal.scatter_add_(0, crystal_id_sorted,
                              torch.ones(core.n_observations, dtype=torch.long))
crystal_offsets = torch.zeros(n_cryst + 1, dtype=torch.long)
crystal_offsets[1:] = obs_per_crystal.cumsum(0)

# ══════════════════════════════════════════════════════════════
# 3. Load model and run inference
# ══════════════════════════════════════════════════════════════
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"No saved model at {MODEL_PATH}. Run diagnose_peak_leakage.py first."
    )

print(f"\n── Loading model from {MODEL_PATH.name} ──")
checkpoint = torch.load(str(MODEL_PATH), map_location=DEVICE)
model = CrystalTransformerCorrection(
    n_panels=32, panel_embed_dim=8,
    d_model=128, n_heads=4, n_layers=2, dropout=0.1,
).to(DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()
print(f"  Saved baseline: {checkpoint['baseline']:.4f}")
print(f"  Saved best free NLL: {checkpoint['best_free']:.4f}")
print(f"  Improvement: {checkpoint['baseline'] - checkpoint['best_free']:.4f}")

# Run inference: predict log_c for ALL observations (in sorted order)
print(f"\n── Running inference on {core.n_observations} observations ──")
BATCH_CRYSTALS = 256
log_c_sorted = torch.zeros(core.n_observations)

with torch.no_grad():
    for ci_start in range(0, n_cryst, BATCH_CRYSTALS):
        ci_end = min(ci_start + BATCH_CRYSTALS, n_cryst)
        s_off = crystal_offsets[ci_start].item()
        e_off = crystal_offsets[ci_end].item()
        if e_off <= s_off:
            continue

        obs_cpu = torch.arange(s_off, e_off)

        cids_chunk = crystal_id_sorted[s_off:e_off]
        unique_cids, inverse = cids_chunk.unique(return_inverse=True)
        n_per_eval = torch.zeros(len(unique_cids), dtype=torch.long)
        n_per_eval.scatter_add_(0, inverse,
                                 torch.ones(e_off - s_off, dtype=torch.long))

        batch_feat = CorrectionFeatures(
            peak=feat_sorted.peak[obs_cpu].to(DEVICE),
            background=feat_sorted.background[obs_cpu].to(DEVICE),
            s_sq=feat_sorted.s_sq[obs_cpu].to(DEVICE),
            excitation_error=feat_sorted.excitation_error[obs_cpu].to(DEVICE),
            q_lab=feat_sorted.q_lab[obs_cpu].to(DEVICE),
            fs=feat_sorted.fs[obs_cpu].to(DEVICE),
            ss=feat_sorted.ss[obs_cpu].to(DEVICE),
            panel_id=feat_sorted.panel_id[obs_cpu].to(DEVICE),
            profile_rad=feat_sorted.profile_rad[obs_cpu].to(DEVICE),
            crystal_id=inverse.to(DEVICE),
        )

        lc = model(batch_feat, n_per_eval.to(DEVICE))
        log_c_sorted[s_off:e_off] = lc.cpu()

# Unsort back to original observation order
unsort_idx = sort_idx.argsort()
log_c = log_c_sorted[unsort_idx]

c = log_c.exp()
print(f"  log_c: min={log_c.min():.3f}, median={log_c.median():.3f}, "
      f"max={log_c.max():.3f}")
print(f"  c: min={c.min():.3f}, median={c.median():.3f}, max={c.max():.3f}")
print(f"  std(log_c): {log_c.std():.3f}")

# ══════════════════════════════════════════════════════════════
# 4. Build corrected SFXStream
# ══════════════════════════════════════════════════════════════
print("\n── Applying corrections ──")
# Divide I, background, AND sigma all by c_j.
# Background enters the Poisson variance formula σ² = g²·(I + bg·n_pk·(1+n_pk/n_bg)),
# so it must scale together with I for the recomputed noise to reflect the correction.
# sigma (CrystFEL reported) scales by the same factor for consistency.
c_safe = c.clamp(min=0.05, max=20.0)  # clip extreme corrections

n_clipped = ((c < 0.05) | (c > 20.0)).sum().item()
print(f"  Clipped {n_clipped} extreme corrections (c<0.05 or c>20)")
print(f"  I: median {core.I.median():.2f} → {(core.I/c_safe).median():.2f}")

corrected_core = replace(
    core,
    I=core.I / c_safe,
    background=core.background / c_safe,
    sigma=core.sigma / c_safe,
)
s_corrected = SFXStream(corrected_core)

# ══════════════════════════════════════════════════════════════
# 5. Split into halves (same seed for fair comparison)
# ══════════════════════════════════════════════════════════════
print(f"\n── Splitting into halves (seed={SEED}) ──")
h1_orig, h2_orig = s.split_random(2, seed=SEED)
h1_corr, h2_corr = s_corrected.split_random(2, seed=SEED)
print(f"  Half 1: {h1_orig.n_crystals} crystals, {h1_orig.n_observations} obs")
print(f"  Half 2: {h2_orig.n_crystals} crystals, {h2_orig.n_observations} obs")

# ══════════════════════════════════════════════════════════════
# 6. Fit both pairs
# ══════════════════════════════════════════════════════════════
def fit_and_match(h1, h2, label, n_em=N_EM):
    print(f"\n── {label} ──")
    f1 = fit_gengauss(h1, n_em_cycles=n_em, **FIT_KW)
    f2 = fit_gengauss(h2, n_em_cycles=n_em, **FIT_KW)
    matched = match_halves(
        f1.hkl, f1.I0.clamp(min=0).sqrt(),
        f2.hkl, f2.I0.clamp(min=0).sqrt(),
        CELL, d_min=0.9, d_max=10.0,
    )
    cc = cc_half(matched)
    r = r_split(matched)
    sp = spearman_cc(matched)
    print(f"  {matched.F1.numel()} matched HKLs")
    print(f"  CC1/2={cc:.4f}  Sp={sp:.4f}  R_split={r:.4f}")
    print(f"  med(α)={f1.alpha.median():.4f}  med(β)={f1.beta.median():.4f}")
    shells = shell_stats(matched, n_shells=10)
    print(f"\n  Shell statistics:")
    print(f"  {'d_max':>6}  {'d_min':>6}  {'CC1/2':>8}  {'R':>8}  {'n_refl':>7}")
    for sh in shells:
        print(f"  {sh.d_max:6.2f}  {sh.d_min:6.2f}  {sh.cc_half:8.4f}  {sh.r_split:8.4f}  {sh.n_matched:7d}")
    return cc, r, sp, f1, f2, matched

# A: no corrections + EM  (current baseline)
cc_orig, r_orig, sp_orig, f1o, f2o, m_orig = fit_and_match(
    h1_orig, h2_orig, "A: original + EM (baseline)", n_em=N_EM)
# B: neural corrections + no EM  (fair comparison: one or the other)
cc_corr, r_corr, sp_corr, f1c, f2c, m_corr = fit_and_match(
    h1_corr, h2_corr, "B: neural corrections, no EM", n_em=0)
# C: neural corrections + EM  (previous double-counting test)
cc_both, r_both, sp_both, f1b, f2b, m_both = fit_and_match(
    h1_corr, h2_corr, "C: neural corrections + EM (double-counting)", n_em=N_EM)

# ══════════════════════════════════════════════════════════════
# 7. Summary
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"  {'':30}  {'CC1/2':>8}  {'Sp':>8}  {'R_split':>8}  {'med(α)':>8}")
print(f"  {'-'*66}")
print(f"  {'A: original + EM':30}  {cc_orig:8.4f}  {sp_orig:8.4f}  {r_orig:8.4f}  {f1o.alpha.median():8.4f}")
print(f"  {'B: neural, no EM':30}  {cc_corr:8.4f}  {sp_corr:8.4f}  {r_corr:8.4f}  {f1c.alpha.median():8.4f}")
print(f"  {'C: neural + EM':30}  {cc_both:8.4f}  {sp_both:8.4f}  {r_both:8.4f}  {f1b.alpha.median():8.4f}")
print(f"{'='*70}")
dB = cc_corr - cc_orig
dC = cc_both - cc_orig
print(f"  B vs A (fair):         CC1/2 {dB:+.4f}")
print(f"  C vs A (double-count): CC1/2 {dC:+.4f}")
