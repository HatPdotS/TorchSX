"""Diagnose residuals: where is the loss coming from?"""

import time
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

# ── Get one batch and run forward ────────────────────────────────────────
batch = next(iter(loader)).to(device)
B, ph, pw = batch.patches.shape
print(f"Batch: {B} obs, {ph}×{pw}")

with torch.no_grad():
    result = model.forward_debug(batch)

obs = batch.patches.float().cpu().numpy()
pred = result.n_pred.cpu().numpy()
valid = (batch.valid > 0).cpu().numpy()
bg = batch.background.cpu().numpy()
signal = result.n_signal.cpu().numpy()
integral = result.integral.cpu().numpy()
F_sq = result.F_sq.cpu().numpy()
scale = result.scale.cpu().numpy()

# ══════════════════════════════════════════════════════════════════════════
# 1. Global pixel statistics
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("1. GLOBAL PIXEL STATISTICS")
print("="*70)

obs_v = obs[valid]
pred_v = pred[valid]

print(f"  Observed:  min={obs_v.min():.0f}  median={np.median(obs_v):.1f}  "
      f"mean={obs_v.mean():.1f}  max={obs_v.max():.0f}")
print(f"  Predicted: min={pred_v.min():.2f}  median={np.median(pred_v):.1f}  "
      f"mean={pred_v.mean():.1f}  max={pred_v.max():.2f}")
print(f"  Background: min={bg.min():.1f}  median={np.median(bg):.1f}  "
      f"mean={bg.mean():.1f}  max={bg.max():.1f}")

residual = obs_v - pred_v
print(f"  Residual:  mean={residual.mean():.2f}  std={residual.std():.1f}  "
      f"RMS={np.sqrt((residual**2).mean()):.1f}")

# NLL per pixel breakdown
nll = 0.5 * (obs_v - pred_v)**2 / np.clip(pred_v, 1e-3, None) + 0.5 * np.log(np.clip(pred_v, 1e-3, None))
print(f"  NLL/pixel: mean={nll.mean():.2f}  median={np.median(nll):.2f}")

resid_term = 0.5 * (obs_v - pred_v)**2 / np.clip(pred_v, 1e-3, None)
log_term = 0.5 * np.log(np.clip(pred_v, 1e-3, None))
print(f"  Residual term: mean={resid_term.mean():.2f}  median={np.median(resid_term):.2f}")
print(f"  Log term:      mean={log_term.mean():.2f}  median={np.median(log_term):.2f}")

# ══════════════════════════════════════════════════════════════════════════
# 2. Histograms of observed and predicted
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("2. PIXEL COUNT DISTRIBUTIONS")
print("="*70)

for label, arr in [("Observed", obs_v), ("Predicted", pred_v)]:
    pcts = np.percentile(arr, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    print(f"  {label:10s}: p0={pcts[0]:.1f}  p1={pcts[1]:.1f}  p5={pcts[2]:.1f}  "
          f"p25={pcts[3]:.1f}  p50={pcts[4]:.1f}  p75={pcts[5]:.1f}  "
          f"p95={pcts[6]:.1f}  p99={pcts[7]:.1f}  p100={pcts[8]:.1f}")

# ══════════════════════════════════════════════════════════════════════════
# 3. Center vs border pixels
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("3. CENTER vs BORDER PIXELS")
print("="*70)

# Center 3×3
c = ph // 2
center_mask = np.zeros((ph, pw), dtype=bool)
center_mask[c-1:c+2, c-1:c+2] = True
border_mask = ~center_mask

for label, pmask in [("Center 3×3", center_mask), ("Border", border_mask)]:
    m = valid & pmask[None, :, :]
    o = obs[m]
    p = pred[m]
    if len(o) == 0:
        continue
    r = o - p
    nll_m = 0.5 * (o - p)**2 / np.clip(p, 1e-3, None) + 0.5 * np.log(np.clip(p, 1e-3, None))
    print(f"  {label:12s}: n={len(o):,}  obs_mean={o.mean():.1f}  pred_mean={p.mean():.1f}  "
          f"resid_RMS={np.sqrt((r**2).mean()):.1f}  NLL/pix={nll_m.mean():.2f}")

# ══════════════════════════════════════════════════════════════════════════
# 4. Signal component magnitudes
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("4. SIGNAL COMPONENTS (per obs)")
print("="*70)

print(f"  F²:        min={F_sq.min():.2f}  median={np.median(F_sq):.1f}  "
      f"mean={F_sq.mean():.1f}  max={F_sq.max():.1f}")
print(f"  Scale:     min={scale.min():.4f}  median={np.median(scale):.4f}  "
      f"mean={scale.mean():.4f}  max={scale.max():.4f}")

# Integral at center pixel
int_center = integral[:, c, c]
print(f"  Integral (center): min={int_center.min():.6f}  median={np.median(int_center):.6f}  "
      f"mean={int_center.mean():.6f}  max={int_center.max():.6f}")

# Signal at center pixel
sig_center = signal[:, c, c]
print(f"  Signal (center):   min={sig_center.min():.2f}  median={np.median(sig_center):.2f}  "
      f"mean={sig_center.mean():.2f}  max={sig_center.max():.2f}")

# Signal max across patch
sig_max = signal.reshape(B, -1).max(axis=1)
print(f"  Signal (max/ROI):  min={sig_max.min():.2f}  median={np.median(sig_max):.2f}  "
      f"mean={sig_max.mean():.2f}  max={sig_max.max():.2f}")

# Total predicted = bg + signal
pred_center = pred[:, c, c]
obs_center = obs[:, c, c]
print(f"  Pred (center):  median={np.median(pred_center):.1f}  mean={pred_center.mean():.1f}")
print(f"  Obs (center):   median={np.median(obs_center):.1f}  mean={obs_center.mean():.1f}")

# ══════════════════════════════════════════════════════════════════════════
# 5. NLL by observed count bucket
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("5. NLL BY OBSERVED COUNT BUCKET")
print("="*70)

buckets = [(0, 1, "0-1"), (1, 5, "1-5"), (5, 20, "5-20"), (20, 100, "20-100"),
           (100, 500, "100-500"), (500, 10000, "500+")]

for lo, hi, label in buckets:
    m = (obs_v >= lo) & (obs_v < hi)
    if m.sum() == 0:
        continue
    o = obs_v[m]
    p = pred_v[m]
    nll_m = 0.5 * (o - p)**2 / np.clip(p, 1e-3, None) + 0.5 * np.log(np.clip(p, 1e-3, None))
    frac = m.sum() / len(obs_v)
    print(f"  obs {label:8s}: n={m.sum():>8,} ({frac*100:5.1f}%)  "
          f"pred_mean={p.mean():8.1f}  resid_RMS={np.sqrt(((o-p)**2).mean()):8.1f}  "
          f"NLL/pix={nll_m.mean():8.2f}  NLL_total_frac={nll_m.sum()/nll.sum()*100:5.1f}%")

# ══════════════════════════════════════════════════════════════════════════
# 6. Worst ROIs — which observations have highest loss?
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("6. WORST 20 ROIs (highest NLL)")
print("="*70)

roi_nll = np.zeros(B)
roi_npix = np.zeros(B)
for i in range(B):
    m = valid[i]
    o = obs[i][m]
    p = pred[i][m]
    roi_nll[i] = (0.5 * (o - p)**2 / np.clip(p, 1e-3, None) + 0.5 * np.log(np.clip(p, 1e-3, None))).sum()
    roi_npix[i] = m.sum()

roi_nll_pp = roi_nll / np.clip(roi_npix, 1, None)
worst = np.argsort(roi_nll_pp)[::-1][:20]

print(f"  {'idx':>6s}  {'NLL/pix':>8s}  {'obs_max':>8s}  {'pred_max':>8s}  "
      f"{'bg':>6s}  {'F²':>8s}  {'scale':>8s}  {'sig_max':>8s}  {'int_ctr':>10s}")
for i in worst:
    print(f"  {i:6d}  {roi_nll_pp[i]:8.1f}  {obs[i][valid[i]].max():8.0f}  "
          f"{pred[i][valid[i]].max():8.1f}  {bg[i]:6.1f}  {F_sq[i]:8.1f}  "
          f"{scale[i]:8.4f}  {sig_max[i]:8.1f}  {int_center[i]:10.6f}")

# ══════════════════════════════════════════════════════════════════════════
# 7. Sample good and bad ROIs (obs vs pred side by side)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("7. SAMPLE ROIs (best vs worst)")
print("="*70)

best = np.argsort(roi_nll_pp)[:5]
for label, indices in [("BEST", best), ("WORST", worst[:5])]:
    print(f"\n  --- {label} ---")
    for i in indices:
        print(f"  ROI {i}: NLL/pix={roi_nll_pp[i]:.1f}  bg={bg[i]:.1f}  "
              f"F²={F_sq[i]:.1f}  scale={scale[i]:.4f}")
        print(f"    Obs  center row: {obs[i, c, :].astype(int)}")
        print(f"    Pred center row: {np.round(pred[i, c, :], 1)}")
        print(f"    Sig  center row: {np.round(signal[i, c, :], 2)}")

print("\nDone.")
