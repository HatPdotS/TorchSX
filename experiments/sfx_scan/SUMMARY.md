# SFX Neural Correction: Summary of Findings

## 1. Dataset

- **Source**: Dark-frame time-resolved SFX at SwissFEL (12,960 eV, lambda=0.9567 A)
- **Data**: 6 delay-time streams (P-1 spacegroup)
- **Scale**: 47,502 crystals, 9.8M observations, 15,279 unique HKLs
- **Detector**: JUNGFRAU 4M, 32 panels

## 2. Observation Model

The intensity of observation j of reflection h follows a multiplicative model:

```
I_obs(j) = c_j * I_0(h) * exp(epsilon_j) + noise_j
```

where:
- `I_0(h)` = true structure factor intensity for reflection h
- `c_j` = per-observation scale correction (partiality, crystal scale, detector response)
- `epsilon_j ~ GenGauss(0, alpha_h, beta_h)` = stochastic multiplicative error in log-space
- `noise_j ~ N(0, sigma^2_poi)` = additive Poisson/counting noise

### 2.1 Generalized Gaussian (GenGauss) Prior

The log-space error distribution uses a generalized Gaussian:

```
p(epsilon) = beta / (2 * alpha * Gamma(1/beta)) * exp(-|epsilon/alpha|^beta)
```

**Fitted parameters (median over HKLs):**
- beta = 1.40 (sub-Gaussian: heavier tails than Gaussian, lighter than Laplace)
- alpha = 1.40 (log-space width: this is large)

### 2.2 Noise Components

**Additive Poisson noise:**
```
sigma^2_poi = g^2 * (I_net + bg * n_peak * (1 + n_peak/n_bg))
```
- Gain factor g = 0.878 (calibrated from negative-intensity tail)
- Integration: r_peak=7px (154 px), r_bg=9-11px (126 px)

**Log-space width (sigma_LN) per HKL:**
- Median: 1.56 (enormous: exp(1.56) = 4.8x multiplicative scatter)
- Resolution-dependent: sigma_LN = 2.25 - 1.16/d^2
- Low resolution has wider spread (more variable partiality)

### 2.3 Variance Decomposition

For `log(c_j) = log(I_obs/I_0)` (deviation of each observation from merged value):

| Source | Variance | Std |
|--------|----------|-----|
| Total var(log c_j) | 2.04 | 1.43 |
| Expected from GenGauss epsilon | 2.60 | 1.61 |
| Excess (scale + systematic) | -0.56 | -- |

**Key finding**: The GenGauss model *already accounts for more variance than exists*.
The observation-to-observation variation is entirely consistent with stochastic noise.
There is no detectable systematic partiality/scale component beyond what epsilon captures.

## 3. Merging Performance

### 3.1 Distribution Model Comparison (split-half validation)

| Method | CC1/2 | R_split |
|--------|-------|---------|
| Single GenGauss fit | 0.988 | 0.040 |
| GenGauss + 10 EM cycles | **0.990** | **0.038** |
| + Crystal scale factors | 0.989 | 0.039 |
| + Per-panel corrections | 0.979 | 0.040 |
| + Panel + gradient (96 params) | diverges | -- |

**Takeaway**: Simple GenGauss + EM is optimal. Adding per-crystal or per-panel
corrections overfits. The stochastic model is already capturing essentially all
the useful information.

### 3.2 Optimizer Choice

| Optimizer | CC1/2 | R_split |
|-----------|-------|---------|
| LBFGS (hard constraint) | 0.52 | -- |
| LBFGS + soft L2 (lambda=10) | 0.83 | 0.08 |
| **Adam (1000 iters)** | **0.97** | **0.07** |

Adam substantially outperforms LBFGS for this problem (non-convex, many HKLs).

## 4. Neural Correction Experiments

### 4.1 Architecture

CrystalTransformerCorrection: 289k parameters
- Per-observation features: peak, background, s^2, excitation_error, q_lab (3D),
  fs, ss, panel_id (embedded), profile_rad
- Crystal-context: observations within one crystal attend to each other
- Output: log(c_j) per observation, bounded by tanh * 4*alpha

### 4.2 Results

| Model | Best free NLL | Delta vs baseline | Likelihood ratio |
|-------|--------------|-------------------|-----------------|
| Baseline (c=1) | 6.77 | -- | -- |
| Without peak/bg | 4.62 | 1.99 | 7.3x |
| With all features | **3.41** | **3.37** | **29x** |

### 4.3 Peak Height Leakage Analysis

**The problem**: CrystFEL's `peak` field (max pixel value) is strongly correlated
with I_obs (r = 0.79 in log-space). The model can use peak as a proxy for I_obs
and learn `log(c_j) ~ log(peak) - f(resolution)`, which absorbs both the true
correction AND the stochastic noise epsilon into c_j.

**Evidence:**
- r(log peak, log c_j) = 0.49, R^2 = 0.24
- With resolution binning: R^2 = 0.39
- Peak accounts for **1.21 NLL** of the 3.37 improvement (~36%)
- But even **without peak**, the model still achieves 1.99 NLL improvement (7.3x)

### 4.4 Interpretation

The 1.99 NLL improvement without peak comes from:
1. **Excitation error** (Ewald sphere offset) — direct partiality predictor
2. **Detector position** (fs, ss, panel_id) — position-dependent response
3. **q_lab** (3D scattering vector) — orientation-dependent effects
4. **Crystal context** (transformer attention) — per-crystal systematic patterns

However, even these "physics" features face the same fundamental issue:
- The model has no way to distinguish "this observation is partial" from
  "this observation happened to scatter strongly" without knowing I_0(h)
- Since I_0 varies hugely within any resolution shell (structure factor variation),
  the model cannot disentangle partiality from I_0 variation

The fast convergence and early stopping at epoch 2 (without peak) suggests the
model quickly learns a systematic detector/position correction and then has
nothing more to extract.

## 5. Other Characterizations

### 5.1 Angular Modulation

Spherical harmonic analysis (l_max=1-6) of intensity variation vs. scattering
direction: only **0.4% variance reduction**. The illumination is essentially
isotropic — no evidence for beam inhomogeneity.

### 5.2 Ewald Sphere Widths

| Resolution (A) | sigma_ewald (1/A) | sigma_eff (1/A) |
|----------------|-------------------|-----------------|
| 1.0-0.9 | 0.0025 | 1.17 |
| 1.2-1.0 | 0.0019 | 1.35 |
| 1.5-1.2 | 0.0013 | 1.53 |

The empirical reciprocal-lattice point widths (sigma_eff) are enormous compared
to the Ewald sphere curvature (sigma_ewald), suggesting that the data is
"still-frame-like" with minimal partiality correction needed.

### 5.3 Feature Importance (gradient-boosted trees)

Nonlinear R^2 = 49% for predicting log(c_j):
1. peak: 1.81 (dominates — but this IS the intensity)
2. background: 0.45 (correlated with peak)
3. d_spacing: 0.10
4. s_sq: 0.06

## 6. Key Conclusions

1. **The GenGauss model works well**: beta~1.4 captures the heavy-tailed nature
   of SFX intensity variation. Simple EM-refined GenGauss gives CC1/2=0.99.

2. **No detectable systematic corrections remain**: The variance decomposition
   shows that GenGauss epsilon already accounts for all observation-to-observation
   variation. There is no "room" for a deterministic correction.

3. **Neural corrections largely absorb noise**: The massive NLL improvement
   (29x) from the neural model is dominated by peak-height leakage. The model
   learns c_j ~ I_obs/I_0, absorbing stochastic noise rather than correcting
   systematic effects.

4. **Physics features give modest signal**: Without peak, excitation error and
   detector position still provide 7.3x improvement. This could reflect real
   detector response variations, but needs further validation (e.g., does it
   improve R_split?).

5. **For this dataset, simple merging is near-optimal**: CC1/2=0.99, R_split=0.04
   with plain GenGauss+EM. The data quality is already very high. Neural
   corrections add complexity without clear improvement in merged data quality.

6. **Adam >> LBFGS** for joint I_0 + distribution parameter optimization.
