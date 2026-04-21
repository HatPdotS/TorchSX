"""Diagnose outliers: is the loss dominated by a few bad crystals/ROIs?

Processes ALL batches and computes:
- Per-ROI NLL distribution + cumulative contribution
- Per-crystal NLL distribution + cumulative contribution
- Outlier flagging thresholds
"""

import numpy as np
import torch

from torchsx import SFXStream
from torchsx.scaling import (
    CrystalOrientation, AbsoluteScale, BScale, prepare_scaler_inputs,
)
from torchsx.merge._structure_factor import MergedStructureFactor
from torchsx.merge._hkl import compute_hkl_index
from torchsx.forward import (
    RefinableGeometry, RelpProfile, PCASpectrum,
    PixelForwardModel, PixelLikelihood, precompute_spectral_overlap,
)
from torchsx.forward.batch import ROIBatchLoader
from torchsx.io._roi_store import ROIStore

# ── Load ─────────────────────────────────────────────────────────────────
print("Loading...")
stream = SFXStream.load("sfx_scan/streams/delay_-45.sfx")
stream = stream.remap_filenames(
    r"/sf/alvra/data/p21957/raw/([^/]+)/data/",
    "/das/work/p17/p17490/Peter/torchsx/h5files" + r"/\1/",
)
core = stream._core
inp = prepare_scaler_inputs(core, n_shells=20)
hkl_idx = compute_hkl_index(core.hkl)

roi_store = ROIStore("delay_-45_rois_11x11.safetensors")
grids = precompute_spectral_overlap(
    np.zeros(1), np.zeros((1, 1)),
    cache_path="delay_-45_pca_grids.npz",
)
crystal_to_shot = np.arange(core.n_crystals) % 12194

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
geometry = RefinableGeometry(core.geometry)
orientation = CrystalOrientation(inp.UB)
crystal_scale = AbsoluteScale(core.n_crystals)
crystal_bfactor = BScale(core.n_crystals)
sf = MergedStructureFactor.from_merge(core)
relp = RelpProfile(n_unique=hkl_idx.n_unique, sigma_r_init=0.0005, sigma_t_init=0.002)
pca_spectrum = PCASpectrum(grids, core.n_crystals, crystal_to_shot)

model = PixelForwardModel(
    structure_factors=sf, orientation=orientation,
    crystal_scale=crystal_scale, crystal_bfactor=crystal_bfactor,
    relp_profile=relp, spectrum=pca_spectrum,
    geometry=geometry, wavelength=inp.wavelength,
).to(device)

loader = ROIBatchLoader(
    roi_store=roi_store, core=core, hkl_index=hkl_idx,
    geometry=geometry, n_crystals_per_batch=500, device=device,
)

# ── Process all batches ──────────────────────────────────────────────────
print("Processing all batches...")

all_roi_nll = []       # total NLL per ROI
all_roi_npix = []      # valid pixel count per ROI
all_roi_obs_max = []   # max observed count per ROI
all_roi_crystal = []   # crystal_id per ROI
all_roi_integral = []  # center-pixel integral per ROI
all_roi_signal_max = []  # max signal per ROI

n_batches = 0
for batch in loader:
    batch = batch.to(device)
    B, ph, pw = batch.patches.shape
    c = ph // 2

    with torch.no_grad():
        result = model.forward_debug(batch)

    obs = batch.patches.float().cpu().numpy()
    pred = result.n_pred.cpu().numpy()
    valid = (batch.valid > 0).cpu().numpy()
    signal = result.n_signal.cpu().numpy()
    integral = result.integral.cpu().numpy()
    crystal_id = batch.crystal_id.cpu().numpy()

    for i in range(B):
        m = valid[i]
        o = obs[i][m]
        p = pred[i][m]
        nll_i = (0.5 * (o - p)**2 / np.clip(p, 1e-3, None)
                 + 0.5 * np.log(np.clip(p, 1e-3, None))).sum()
        all_roi_nll.append(nll_i)
        all_roi_npix.append(m.sum())
        all_roi_obs_max.append(o.max() if len(o) > 0 else 0)
        all_roi_crystal.append(crystal_id[i])
        all_roi_integral.append(integral[i, c, c])
        all_roi_signal_max.append(signal[i].max() if m.any() else 0)

    n_batches += 1
    if n_batches % 20 == 0:
        print(f"  batch {n_batches}...")

all_roi_nll = np.array(all_roi_nll)
all_roi_npix = np.array(all_roi_npix)
all_roi_obs_max = np.array(all_roi_obs_max)
all_roi_crystal = np.array(all_roi_crystal)
all_roi_integral = np.array(all_roi_integral)
all_roi_signal_max = np.array(all_roi_signal_max)

N_roi = len(all_roi_nll)
all_roi_nll_pp = all_roi_nll / np.clip(all_roi_npix, 1, None)

print(f"\nTotal: {N_roi:,} ROIs from {n_batches} batches")

# ══════════════════════════════════════════════════════════════════════════
# 1. ROI-level NLL distribution
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("1. ROI-LEVEL NLL DISTRIBUTION")
print("="*70)

pcts = np.percentile(all_roi_nll_pp, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100])
print(f"  NLL/pixel per ROI:")
print(f"    p0={pcts[0]:.1f}  p1={pcts[1]:.1f}  p5={pcts[2]:.1f}  p10={pcts[3]:.1f}")
print(f"    p25={pcts[4]:.1f}  p50={pcts[5]:.1f}  p75={pcts[6]:.1f}  p90={pcts[7]:.1f}")
print(f"    p95={pcts[8]:.1f}  p99={pcts[9]:.1f}  p100={pcts[10]:.1f}")

# Cumulative NLL contribution
sorted_nll = np.sort(all_roi_nll)[::-1]  # descending
cum_nll = np.cumsum(sorted_nll)
total_nll = cum_nll[-1]

print(f"\n  Total NLL: {total_nll:,.0f}")
for frac in [0.01, 0.05, 0.10, 0.20, 0.50]:
    n = max(1, int(N_roi * frac))
    print(f"  Top {frac*100:5.1f}% ROIs ({n:>7,}): {cum_nll[n-1]/total_nll*100:5.1f}% of total NLL")

# ══════════════════════════════════════════════════════════════════════════
# 2. Crystal-level NLL distribution
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("2. CRYSTAL-LEVEL NLL DISTRIBUTION")
print("="*70)

unique_crystals = np.unique(all_roi_crystal)
n_crystals = len(unique_crystals)
crystal_nll = np.zeros(n_crystals)
crystal_npix = np.zeros(n_crystals)
crystal_nroi = np.zeros(n_crystals, dtype=int)

crystal_map = {c: i for i, c in enumerate(unique_crystals)}
for j in range(N_roi):
    ci = crystal_map[all_roi_crystal[j]]
    crystal_nll[ci] += all_roi_nll[j]
    crystal_npix[ci] += all_roi_npix[j]
    crystal_nroi[ci] += 1

crystal_nll_pp = crystal_nll / np.clip(crystal_npix, 1, None)

pcts = np.percentile(crystal_nll_pp, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100])
print(f"  NLL/pixel per crystal ({n_crystals:,} crystals):")
print(f"    p0={pcts[0]:.1f}  p1={pcts[1]:.1f}  p5={pcts[2]:.1f}  p10={pcts[3]:.1f}")
print(f"    p25={pcts[4]:.1f}  p50={pcts[5]:.1f}  p75={pcts[6]:.1f}  p90={pcts[7]:.1f}")
print(f"    p95={pcts[8]:.1f}  p99={pcts[9]:.1f}  p100={pcts[10]:.1f}")

# Cumulative
sorted_crystal_nll = np.sort(crystal_nll)[::-1]
cum_crystal = np.cumsum(sorted_crystal_nll)
total_crystal_nll = cum_crystal[-1]

print(f"\n  Total NLL: {total_crystal_nll:,.0f}")
for frac in [0.01, 0.05, 0.10, 0.20, 0.50]:
    n = max(1, int(n_crystals * frac))
    print(f"  Top {frac*100:5.1f}% crystals ({n:>6,}): "
          f"{cum_crystal[n-1]/total_crystal_nll*100:5.1f}% of total NLL")

# ROIs per crystal distribution
pcts_nroi = np.percentile(crystal_nroi, [0, 25, 50, 75, 95, 100])
print(f"\n  ROIs per crystal: p0={pcts_nroi[0]:.0f}  p25={pcts_nroi[1]:.0f}  "
      f"p50={pcts_nroi[2]:.0f}  p75={pcts_nroi[3]:.0f}  p95={pcts_nroi[4]:.0f}  "
      f"p100={pcts_nroi[5]:.0f}")

# ══════════════════════════════════════════════════════════════════════════
# 3. Characterize worst crystals
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("3. WORST 20 CRYSTALS")
print("="*70)

worst_crystals = np.argsort(crystal_nll_pp)[::-1][:20]
print(f"  {'crystal':>8s}  {'NLL/pix':>8s}  {'n_ROI':>6s}  {'n_pix':>8s}  {'total_NLL':>12s}")
for ci in worst_crystals:
    print(f"  {unique_crystals[ci]:8d}  {crystal_nll_pp[ci]:8.1f}  "
          f"{crystal_nroi[ci]:6d}  {crystal_npix[ci]:8.0f}  {crystal_nll[ci]:12.0f}")

# ══════════════════════════════════════════════════════════════════════════
# 4. Characterize worst ROIs
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("4. WORST 20 ROIs")
print("="*70)

worst_rois = np.argsort(all_roi_nll_pp)[::-1][:20]
print(f"  {'roi':>8s}  {'NLL/pix':>8s}  {'obs_max':>8s}  {'int_ctr':>10s}  "
      f"{'sig_max':>8s}  {'crystal':>8s}")
for ri in worst_rois:
    print(f"  {ri:8d}  {all_roi_nll_pp[ri]:8.1f}  {all_roi_obs_max[ri]:8.0f}  "
          f"{all_roi_integral[ri]:10.6f}  {all_roi_signal_max[ri]:8.1f}  "
          f"{all_roi_crystal[ri]:8d}")

# ══════════════════════════════════════════════════════════════════════════
# 5. What fraction of NLL is "trivially bad" (zero integral)?
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("5. ZERO-INTEGRAL ROIs")
print("="*70)

for threshold in [1e-6, 1e-5, 1e-4, 1e-3]:
    mask = all_roi_integral < threshold
    n = mask.sum()
    nll_frac = all_roi_nll[mask].sum() / total_nll if total_nll > 0 else 0
    print(f"  integral < {threshold:.0e}: {n:>7,} ROIs ({n/N_roi*100:5.1f}%), "
          f"{nll_frac*100:5.1f}% of total NLL")

# ══════════════════════════════════════════════════════════════════════════
# 6. Effect of outlier removal on mean NLL
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("6. EFFECT OF OUTLIER REMOVAL")
print("="*70)

print("\n  --- ROI-level filtering ---")
for cutoff_pct in [99, 95, 90, 80]:
    threshold = np.percentile(all_roi_nll_pp, cutoff_pct)
    keep = all_roi_nll_pp <= threshold
    n_keep = keep.sum()
    mean_nll = all_roi_nll[keep].sum() / np.clip(all_roi_npix[keep].sum(), 1, None)
    print(f"  Keep p<={cutoff_pct}: {n_keep:>7,} ROIs, mean NLL/pix = {mean_nll:.2f} "
          f"(threshold = {threshold:.1f})")

print("\n  --- Crystal-level filtering ---")
for cutoff_pct in [99, 95, 90, 80]:
    threshold = np.percentile(crystal_nll_pp, cutoff_pct)
    bad_crystals = set(unique_crystals[crystal_nll_pp > threshold])
    keep = np.array([all_roi_crystal[j] not in bad_crystals for j in range(N_roi)])
    n_keep_roi = keep.sum()
    n_keep_cryst = n_crystals - len(bad_crystals)
    mean_nll = all_roi_nll[keep].sum() / np.clip(all_roi_npix[keep].sum(), 1, None)
    print(f"  Keep p<={cutoff_pct}: {n_keep_cryst:>6,} crystals, {n_keep_roi:>7,} ROIs, "
          f"mean NLL/pix = {mean_nll:.2f}")

# ══════════════════════════════════════════════════════════════════════════
# 7. NLL by obs_max bucket (per ROI)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("7. ROI NLL BY PEAK INTENSITY")
print("="*70)

buckets = [(0, 5, "0-5"), (5, 20, "5-20"), (20, 100, "20-100"),
           (100, 500, "100-500"), (500, 2000, "500-2k"), (2000, 1e8, "2k+")]

for lo, hi, label in buckets:
    m = (all_roi_obs_max >= lo) & (all_roi_obs_max < hi)
    if m.sum() == 0:
        continue
    mean_nll = all_roi_nll[m].sum() / np.clip(all_roi_npix[m].sum(), 1, None)
    med_integral = np.median(all_roi_integral[m])
    nll_frac = all_roi_nll[m].sum() / total_nll * 100
    print(f"  peak {label:8s}: {m.sum():>7,} ROIs ({m.sum()/N_roi*100:5.1f}%)  "
          f"NLL/pix={mean_nll:7.1f}  NLL_frac={nll_frac:5.1f}%  "
          f"med_integral={med_integral:.6f}")

print("\nDone.")
