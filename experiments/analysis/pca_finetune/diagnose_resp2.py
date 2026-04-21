#!/usr/bin/env python
"""Check F² and signal magnitude from the merge."""

import torch
import numpy as np
from torchsx import SFXStream
from torchsx.scaling import prepare_scaler_inputs
from torchsx.merge._structure_factor import MergedStructureFactor
from torchsx.merge._hkl import compute_hkl_index

stream = SFXStream.load("sfx_scan/streams/delay_-45.sfx")
core = stream._core

inp = prepare_scaler_inputs(core, n_shells=20)
hkl_idx = compute_hkl_index(core.hkl)

sf = MergedStructureFactor.from_merge(core)
F_sq = torch.exp(sf.log_F * 2).detach()

print(f"Crystals: {core.n_crystals:,}")
print(f"Unique reflections: {hkl_idx.n_unique:,}")
print(f"\n--- F² statistics ---")
print(f"  min:    {F_sq.min():.2f}")
print(f"  25%:    {F_sq.quantile(0.25):.2f}")
print(f"  median: {F_sq.median():.2f}")
print(f"  75%:    {F_sq.quantile(0.75):.2f}")
print(f"  max:    {F_sq.max():.2f}")
print(f"  mean:   {F_sq.mean():.2f}")

print(f"\n--- log_F statistics ---")
print(f"  log_global_scale: {sf.log_global_scale.item():.4f}")
print(f"  log_F min:   {sf.log_F.min():.4f}")
print(f"  log_F median:{sf.log_F.median():.4f}")
print(f"  log_F max:   {sf.log_F.max():.4f}")

# What's the typical observed intensity?
I_obs = core.I.float()
bg_obs = core.background.float()
peak_obs = core.peak.float()

print(f"\n--- Observed intensity statistics ---")
print(f"  I:    median={I_obs.median():.1f}, mean={I_obs.mean():.1f}")
print(f"  bg:   median={bg_obs.median():.1f}, mean={bg_obs.mean():.1f}")
print(f"  peak: median={peak_obs.median():.1f}, mean={peak_obs.mean():.1f}")
print(f"  I-bg: median={(I_obs-bg_obs).median():.1f}")

# Signal at sparse pixels: n_signal = scale * F² * overlap * corrections
# With scale=1 (log_G=0), B=0, polarization~1, solid_angle~1:
# n_signal ≈ F² * overlap
# At σ_r=0.0002, overlap at ds_r=0 ≈ 0.57
# So n_signal ≈ F² * 0.57
# Background at sparse pixels is typically... let's estimate
# From relp analysis: median I ≈ 45, median bg ≈ 11 at excitation_error ≈ 0
# So signal ≈ 34, bg ≈ 11, resp ≈ 34/(34+11) ≈ 75%

# But that's for INTEGRATED intensity. Pixel-level is different.
# A Bragg spot spreads over ~10-20 pixels, so per-pixel signal ≈ F²*overlap/n_pixels

# What's the typical number of sparse pixels per reflection?
# From the finetune output: 5 pairs/refl
# So per-pixel signal ≈ F² * 0.57 / 5

print(f"\n--- Expected per-pixel signal (σ_r=0.0002) ---")
typical_overlap = 0.57
typical_pairs = 5
per_pixel = F_sq.median() * typical_overlap / typical_pairs
print(f"  F²_median * overlap / pairs_per_refl = {F_sq.median():.2f} * {typical_overlap} / {typical_pairs} = {per_pixel:.4f}")
print(f"  Typical bg per pixel: ~10-30 (from radial profile)")
print(f"  Expected resp: {per_pixel / (per_pixel + 15):.6f}")

# With σ_r=0.005 (previous setting):
typical_overlap_big = 0.996
per_pixel_big = F_sq.median() * typical_overlap_big / typical_pairs
print(f"\n--- Expected per-pixel signal (σ_r=0.005) ---")
print(f"  F²_median * overlap / pairs_per_refl = {F_sq.median():.2f} * {typical_overlap_big} / {typical_pairs} = {per_pixel_big:.4f}")
print(f"  Expected resp: {per_pixel_big / (per_pixel_big + 15):.6f}")
