# TorchRef: Pixel-Level SFX Forward Model Specification

**Version:** 0.1 — DRAFT  
**Date:** March 2026  
**Author:** Peter, Paul Scherrer Institut  

---

## 1. Motivation

Current serial femtosecond crystallography (SFX) processing pipelines decompose the path from raw detector pixels to refined structure into sequential, independently optimized stages: spot finding → integration → partiality correction → merging → refinement. Each stage discards information and introduces model assumptions that are not propagated downstream. The fundamental inconsistency is that refinement treats merged structure factor amplitudes as delta-function observations, while the raw data clearly shows that reciprocal lattice points (relps) have finite extent — evidenced by the continuous distribution of partial intensities, the ability to apply I/σ cutoffs, and the resolution-dependent observation variance.

This specification defines a differentiable forward model that predicts raw detector pixel counts directly from the estimated strucutre factors/

---

## 2. Scope

The model targets room-temperature TR-SFX data collected at the Alvra endstation of SwissFEL using a JUNGFRAU detector. The initial dataset comprises ~800K indexed crystal frames of a photoactive palladium coordination complex with azoarene chromophores.

### 2.1 What This Model Replaces

- Post-hoc partiality correction (replaced by spectral integration over the relp profile)
- Intensity integration with box summation (replaced by pixel-level likelihood)
- Background subtraction from annular estimates (replaced by per-ROI precomputed background)
- Merging to unique reflections (replaced by optimization of reflections)

### 2.2 What This Model Retains From Upstream

- CrystFEL indexing solutions (orientation matrices, unit cell) as initial values
- CrystFEL geometry files (detector panel positions) as initial values
- Spot positions for ROI extraction (updated iteratively)

---

## 3. Coordinate System

All pixel positions are described in angular coordinates (2θ, φ) relative to the beam axis, where:

- **2θ** is the scattering angle (radial distance from beam center in reciprocal space)
- **φ** is the azimuthal angle around the beam axis

This choice makes centrosymmetric contributions (solvent scatter, HEC rings, air scatter) purely functions of 2θ, simplifying background estimation.

Reciprocal lattice point profiles are parameterized in **spherical coordinates relative to the scattering vector**:

- **Radial (r̂):** along **h**/|**h**|, pointing from origin to the relp center. Captures bandwidth broadening, crystal size effects, and lattice strain.
- **Tangential (t̂):** perpendicular to r̂. Captures mosaicity and beam divergence. Treated as isotropic in the tangential plane for the initial implementation (single σ_t).

---

## 4. Forward Model

For pixel **p** in ROI **i** belonging to frame **j**, with nearest reflection **hkl**, the predicted detector count is:

```
n_pred(p) = G · [ bg_i + S_j · V_j · exp(-B_j · d*² / 2) · P(2θ_p, φ_p) · Ω(2θ_p)
              · |F(hkl)|² · ∫ w_j(λ) · R(s_p(λ) − h; σ_r, σ_t) dλ ]
```

### 4.1 Component Definitions

#### 4.1.1 Structure Factor — F(hkl)

The squared amplitude of the structure factor for reflection hkl. Either refined as free parameters per unique reflection, or computed from an atomic model (coordinates, ADPs, occupancies). This is the primary quantity of interest.

#### 4.1.2 Relp Profile — R(Δs; σ_r, σ_t)

The reciprocal lattice point intensity profile, modeled as an anisotropic Gaussian in spherical coordinates:

```
R(Δs; σ_r, σ_t) = exp( -Δs_r² / (2·σ_r²) - Δs_t² / (2·σ_t²) )
```

where:

- Δs_r = (s_p − h) · ĥ is the radial component (along the scattering vector)
- Δs_t = |( s_p − h) − Δs_r · ĥ| is the tangential component
- ĥ = h / |h| is the unit vector from the origin to the relp center

**Parameters:** σ_r(hkl) and σ_t(hkl) per unique reflection (2 × N_unique ≈ 80K parameters).

**Physical content:**
- σ_r encodes bandwidth broadening, crystal size along beam direction, lattice strain (d-spacing variation)
- σ_t encodes mosaicity, beam divergence, lattice rotational disorder
- For SASE XFEL data: expect σ_r >> σ_t (radially elongated "streaky" relps)

#### 4.1.3 Spectral Integration — ∫ w_j(λ) · R(…) dλ

Each wavelength λ in the beam spectrum maps pixel p to a different reciprocal space coordinate s_p(λ). The predicted intensity is the integral of the relp profile weighted by the spectral power over all wavelengths present in the shot. This integral replaces the traditional partiality correction entirely.

**Spectral model:** The SASE spectrum for frame j is reconstructed from a low-dimensional PCA basis:

```
w_j(λ) = w_mean(λ) + Σ_k  c_k^(j) · w_basis_k(λ)     (k = 1, 2, 3)
```

The PCA basis is learned from ~10% of frames that have spectrometer measurements (Alvra inline spectrometer). The three components capture: mean energy shift, effective bandwidth variation, and dominant asymmetry/bimodality. For the remaining ~90% of frames, only the three coefficients c_k are refined against diffraction data.

**Parameters:** 3 spectral PCA coefficients per crystal (3 × N_crystals). PCA basis vectors are fixed after initial estimation from spectrometer data.

**Validation:** For the 10% of frames with spectrometer data, the diffraction-refined spectral coefficients can be compared against the spectrometer-derived values as an independent model quality check.

#### 4.1.4 Orientation — U_j

Rotation matrix for crystal j, parameterized as a small perturbation from the CrystFEL indexing solution:

```
U_j = δR(ε_1, ε_2, ε_3) · U_CrystFEL_j
```

where δR is a rotation parameterized by Rodrigues vector or quaternion, initialized at identity. The orientation determines the mapping from pixel coordinates to reciprocal space via:

```
s_p(λ) = (1/λ) · (ŝ_out(p) − ŝ_in)
```

where ŝ_out is the unit vector from sample to pixel, determined by detector geometry, and ŝ_in is the incident beam direction.

**Parameters:** 3 DOF per crystal (3 × N_crystals).

#### 4.1.5 Per-Crystal Scale — S_j

Overall scale factor absorbing beam intensity fluctuation, crystal illuminated volume, and any other shot-level multiplicative factors.

**Parameters:** 1 per crystal.

#### 4.1.6 Interaction Volume — V_j

The overlap between the X-ray beam and the crystal. Varies with crystal position in the liquid jet relative to beam center and with crystal size. Partially degenerate with S_j; may be absorbed into S_j as a single combined scale or separated if beam profile is modeled explicitly.

**Parameters:** 1 per crystal (or absorbed into S_j).

#### 4.1.7 Per-Crystal B-factor — B_j

Isotropic displacement parameter per crystal, capturing crystal-level resolution-dependent intensity falloff due to overall disorder, thermal motion, and crystal quality:

```
D_j(hkl) = exp(-B_j · d*²_hkl / 2)
```

**Parameters:** 1 per crystal.

**Physical content:** Distribution of B_j across the dataset characterizes crystal quality variation. Bimodal distributions may indicate distinct crystal populations. Systematic differences between light/dark frames in TR-SFX may indicate photoreaction-induced disorder.

#### 4.1.8 Polarization Correction — P(2θ, φ)

SwissFEL produces horizontally polarized X-rays. The correction is:

```
P(2θ, φ) = 1 − sin²(2θ) · cos²(φ)
```

where φ is measured relative to the horizontal polarization plane. No free parameters; fully determined by geometry / might be refined within the dataset. 

#### 4.1.9 Solid Angle Correction — Ω(2θ)

Each pixel subtends a different solid angle depending on distance and angle from the sample:

```
Ω(p) ∝ cos³(α)
```

where α is the angle between the beam axis and the sample-to-pixel vector. No free parameters; fully determined by geometry.

#### 4.1.10 Background — bg_i

Per-ROI scalar estimated during data extraction as the **median** of border pixels in the excised patch. Median is preferred over mean for robustness against contamination from tails of neighboring spots.

**Parameters:** None (precomputed and frozen).

#### 4.1.11 Detector Gain — G

Conversion factor from photons to ADU, per detector panel. Required for correct Poisson likelihood if working in ADU. Can be calibrated offline from flat-field exposures or refined as a nuisance parameter.

**Parameters:** 1 per detector panel.

#### 4.1.12 Detector Geometry

Panel positions (x, y, z) and tilts (3 Euler angles) mapping pixel indices to laboratory-frame scattering vectors. Initialized from CrystFEL .geom file, optionally refined as global parameters.

**Parameters:** ~6 per panel (small corrections to initial geometry).

---


## 5. Parameter Summary

### 5.1 Per-Crystal Parameters (8 DOF × ~800K crystals)

| Parameter | DOF | Initialization | Constraint |
|-----------|-----|----------------|------------|
| Orientation U_j | 3 | CrystFEL indexing | Small perturbation |
| Scale S_j | 1 | Wilson statistics | Positive |
| B-factor B_j | 1 | Wilson plot | Positive |
| Spectral PCA coefficients c_k | 3 | Spectrometer (10%) / zeros (90%) | PCA subspace |
| **Total per crystal** | **8** | | |
| **Total all crystals** | **~6.4M** | | |

### 5.2 Dataset-Level Parameters

| Parameter | Count | Initialization | Constraint |
|-----------|-------|----------------|------------|
| Structure factors F(hkl) or atomic model | ~40K (free) or ~10K (model) | Molecular replacement | Geometry restraints (if model) |
| σ_r(hkl) | ~40K | Resolution-binned estimate | Positive |
| σ_t(hkl) | ~40K | Resolution-binned estimate | Positive |
| Detector panel geometry | ~6 × N_panels | CrystFEL .geom | Small perturbation |
| Detector gain | N_panels | Flat-field calibration | Positive |
| **Total dataset-level** | **~120K + model** | | |

### 5.3 Precomputed (Frozen)

| Quantity | Count | Source |
|----------|-------|--------|
| Background bg_i | 1 per ROI | Median of border pixels |
| Polarization P(2θ, φ) | Per pixel | Geometry |
| Solid angle Ω(2θ) | Per pixel | Geometry |
| Spectral PCA basis | 3 vectors × N_spectral_bins | 10% spectrometer data |
| Pixel → (2θ, φ) mapping | Per pixel | Geometry |

### 5.4 Overdetermination

| Quantity | Approximate Value |
|----------|-------------------|
| Indexed frames | 800,000 |
| Spots per frame | ~1,000 |
| Pixels per ROI (20×20) | 400 |
| Total pixel observations | ~320 billion |
| Total free parameters | ~6.5M |
| **Overdetermination ratio** | **~50,000 : 1** |

---

## 6. Likelihood Function

### 6.1 Poisson Likelihood

For photon-counting detectors (JUNGFRAU in appropriate gain mode), the natural likelihood is Poisson:

```
ln P(n_obs | n_pred) = n_obs · ln(n_pred) − n_pred − ln(n_obs!)
```

summed over all pixels in all ROIs.

### 6.2 Gaussian Approximation

For pixels with counts above ~20, the Gaussian approximation is adequate and computationally cheaper:

```
ln P(n_obs | n_pred) = −(n_obs − n_pred)² / (2 · n_pred) − 0.5 · ln(n_pred)
```

where the variance equals the predicted counts (Poisson regime).

### 6.3 Masked Pixels

Dead pixels, pixels flagged by the detector, and pixels in known artifact regions (panel edges, gaps) are excluded from the likelihood by a precomputed binary mask.

---

## 7. Data Architecture

### 7.1 Offline Extraction (One-Time)

Source H5 files are processed into memory-mapped arrays for fast random access during refinement.

**Step 1: Predict ROI positions** from CrystFEL indexing solutions. For each indexed frame and each predicted reflection, compute the detector pixel coordinate.

**Step 2: Extract oversized patches** (40×40 pixels) centered on predicted positions, providing ±10 pixel margin for orientation refinement. Store as contiguous memmap:

```
roi_patches:   memmap[N_rois, 40, 40]   dtype=uint16
roi_metadata:  memmap[N_rois]            structured array:
    frame_id:    uint32
    hkl:         int16 × 3
    panel_id:    uint8
    center_x:    float32
    center_y:    float32
    two_theta:   float32
    phi:         float32
    background:  float32     (median of border pixels)
```

**Storage estimate:** 800K frames × 100 spots × roi_large × roi_large × 4 bytes

**Step 3: Apply matched filter** using predicted spot shapes to detect weak high-resolution spots invisible to conventional spot finders. Re-extract additional ROIs.

For detector edges we just extract the roi to the edge. The idea is that we can just model the pixels that are there and ignore the missing pixels. 


### 7.2 Online Refinement (Per Batch)

Each refinement iteration:

1. Load a batch of roi_large*roi_large ROI patches from memmap into pinned CPU memory
2. Compute predicted spot centers from current geometry (orientation, detector, wavelength)
3. Extract roi_small*roi_large sub-patches at predicted centers — pure GPU tensor indexing, no disk IO
4. Evaluate forward model on sub-patches
5. Compute Poisson/Gaussian NLL
6. Backpropagate and update parameters

**Batch size:** ~50K ROIs × roi_large × roi_large × 4 bytes

roi large and roi small are hyper parameter. The goal is to pick a parameter where the spot is well characterized but no second spot is in the roi

**Out-of-bounds handling:** If refined orientation shifts predicted center beyond the ±10 pixel margin, flag the ROI for re-extraction from the original H5 file. This should be rare (<1% of ROIs) and is batched periodically rather than every iteration.

---

## 8. Refinement Strategy

### 8.1 Stage 1 — Bootstrap

**Fix:** Orientations (CrystFEL), spectral coefficients for frames with spectrometer data.  
**Refine:** S_j, B_j, F(hkl), σ_r(hkl), σ_t(hkl).  
**Purpose:** Establish relp profiles and structure factors with minimal risk of overfitting. Uses the 10% of frames with known spectra as high-confidence anchors.

### 8.2 Stage 2 — Spectral Refinement

**Fix:** Orientations.  
**Release:** Spectral PCA coefficients for all frames.  
**Purpose:** With relp profiles known, pixel patterns strongly constrain the spectrum — spectral spikes illuminate specific radial slices through the relp.

### 8.3 Stage 3 — Joint Refinement

**Release:** All parameters (orientations, spectra, structure factors, relp profiles, detector geometry).  
**Purpose:** Full joint optimization. The 10% with spectrometer data act as anchors preventing orientation–spectrum degeneracy.

### 8.4 Stage 4 — Weak Spot Recovery

Apply matched filter with refined model to detect additional reflections. Re-extract ROIs. Re-run Stage 3 with expanded dataset. Iterate if resolution extends significantly.

---

## 9. Diagnostics

The factored parameter structure enables rich diagnostics:

### 9.1 Per-Crystal B-factor Distribution

Histogram of B_j across the dataset reveals crystal quality variation. Bimodal distributions indicate distinct populations. Systematic B_j differences between light/dark frames indicate photoreaction-induced disorder. High-B_j outliers can be downweighted or rejected.

### 9.2 Relp Profile Landscape

σ_r(hkl) and σ_t(hkl) plotted as a function of resolution (d*) and position in reciprocal space:

- Smooth resolution dependence is expected from Ewald sphere geometry
- Deviations flag specific reflections with anomalous behavior
- Elevated σ at specific 2θ rings indicates ice contamination (HEC rings)
- Discontinuities correlated with detector panel boundaries indicate geometry errors
- σ_r/σ_t ratio quantifies the relative contribution of bandwidth vs. mosaicity

### 9.3 Spectral Validation

For the 10% of frames with spectrometer data: compare diffraction-refined PCA coefficients against spectrometer-derived values. Strong correlation validates the model; systematic offsets indicate model deficiencies.

### 9.4 Residual Maps

Per-pixel residuals (n_obs − n_pred) aggregated by:

- Detector position: reveals panel-level systematics, gain errors
- Resolution shell: reveals resolution-dependent model failures
- Frame: reveals shot-level outliers

---

## 10. Potential Degeneracies

| Degeneracy | Mitigation |
|------------|------------|
| S_j ↔ V_j (scale vs. interaction volume) | Absorb V_j into S_j unless beam profile is modeled explicitly |
| B_j ↔ resolution-dependent σ(hkl) | Fix ⟨B_j⟩ = 0 by convention; Wilson B lives in the atomic model ADPs |
| Orientation ↔ spectrum (rotation vs. energy shift) | Rotation moves spots tangentially, energy shifts radially; many spots per frame break the degeneracy; spectrometer data for 10% provide anchors |
| σ_r ↔ spectral bandwidth | Spectral PCA constrains bandwidth shape; σ_r absorbs residual |
| Gain G ↔ S_j | Fix gain from offline calibration or constrain ⟨S_j⟩ per run |

---

## 11. Computational Considerations

### 11.1 GPU Memory

The forward model evaluates per-pixel predicted counts for a batch of ROIs. The dominant memory cost is storing the pixel-level tensors:

- 50K ROIs × 20 × 20 pixels × float32 ≈ 80 MB (observations)
- Same for predicted counts, gradients: ~240 MB total
- Per-crystal parameters for batch: negligible
- Dataset-level parameters (F, σ): ~1 MB

Total GPU memory per batch: < 1 GB. Easily fits on a single GPU.

### 11.2 Spectral Integration

The integral over the spectrum is discretized over N_bins wavelength points (typically 50–100). For each pixel, N_bins reciprocal space coordinates are computed and the relp profile is evaluated at each. This is the inner loop and dominates compute. Separable Gaussian evaluation and vectorization over spectral bins keep it tractable.

### 11.3 IO Bandwidth

Memmap reads are sequential if ROIs are sorted by frame index. Expected throughput from GPFS: ~1–5 GB/s. Batch load time (160 MB): ~30–160 ms. Forward/backward pass likely dominates.

---

## 12. Extensions (Future Work)

- **Explicit crystal shape transform:** Replace Gaussian relp profile with physical model based on crystal dimensions (parallelepiped or ellipsoidal shape transform)
- **Per-crystal unit cell:** Allow unit cell variation across crystals to model lattice strain from photoreaction
- **Anisotropic tangential profile:** Split σ_t into two orthogonal components for anisotropic mosaicity
- **Detector PSF:** Convolve predicted spot with JUNGFRAU charge-sharing kernel

---

## 13. Notation Reference

| Symbol | Meaning |
|--------|---------|
| n_pred(p) | Predicted photon count (or ADU) at pixel p |
| F(hkl) | Structure factor amplitude for reflection hkl |
| R(Δs; σ_r, σ_t) | Reciprocal lattice point profile (anisotropic Gaussian) |
| σ_r(hkl) | Radial relp width for reflection hkl |
| σ_t(hkl) | Tangential relp width for reflection hkl |
| w_j(λ) | Spectral power distribution for frame j |
| c_k^(j) | k-th PCA coefficient for spectrum of frame j |
| U_j | Orientation matrix for crystal j |
| S_j | Scale factor for crystal j |
| V_j | Interaction volume for crystal j |
| B_j | Isotropic B-factor for crystal j |
| P(2θ, φ) | Polarization correction |
| Ω(2θ) | Solid angle correction |
| bg_i | Background estimate for ROI i |
| G | Detector gain (ADU per photon) |
| d* | Reciprocal resolution (1/d = 2 sin θ / λ) |
| ĥ | Unit vector from origin to relp center |
| s_p(λ) | Reciprocal space coordinate of pixel p at wavelength λ |