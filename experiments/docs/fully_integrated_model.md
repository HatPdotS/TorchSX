# TorchRef: Partiality-Aware Self-Consistent Scaling

**Design Document — March 2026 | P. Kolenko, PSI**

---

## 1. Motivation

The current `SelfConsistentTarget` merging model uses a per-crystal scale and isotropic B-factor to relate observed intensities to true structure factor intensities:

```
I_pred = I_true[hkl] * G * exp(-2 B s²)
```

This model implicitly assumes every observation fully samples the Bragg reflection. In serial femtosecond crystallography (SFX), each still-shot exposure records only a partial slice through the reciprocal lattice point (relp). The fraction recorded — the partiality — depends on the excitation error, crystal mosaicity, beam divergence, and spectral bandwidth.

Without explicit partiality modelling, the scale parameters must absorb partiality effects. This works reasonably at high resolution where relps are narrow relative to the Ewald sphere curvature. At low resolution, however, partiality variation is larger and poorly described by an isotropic B-factor, leading to biased intensity estimates. Current benchmarks show TorchRef outperforming Partialator at high resolution (CC½ 0.57 vs 0.09 at 0.9 Å) but underperforming at low resolution.

This document describes a partiality-aware extension that models observed intensities in observation space — never dividing by partiality — so that the heteroscedastic noise model naturally downweights poorly-sampled reflections.

---

## 2. Forward Model

### 2.1 Intensity Prediction

The extended forward model predicts the observed intensity for observation *j* of reflection *hkl* from crystal *c*:

```
log I_pred = log_I_true[hkl] + log_G[c] - 2 B[c] s² + log_p[j]

I_pred = exp(log I_pred)
```

where:

- `log_I_true[hkl]` — log of the true merged intensity for unique reflection hkl
- `log_G[c]` — log absolute scale for crystal c
- `B[c]` — isotropic B-factor for crystal c
- `log_p[j]` — log partiality for observation j (derived from the Ewald offset model, see §2.2)

The entire forward model is a sum of additive terms in log space, with a single `exp()` at the end. This ensures gradients are *relative* — proportional to I_pred itself — eliminating the dynamic-range problem where strong reflections dominate weak ones in linear space. L-BFGS conditioning is dramatically improved.

### 2.2 Ewald Offset Partiality Model

The partiality for each observation is derived from the excitation error — the signed distance from the reciprocal lattice point to the Ewald sphere surface.

#### 2.2.1 Excitation Error Computation

Given the oriented reciprocal lattice point position:

```
q_hkl = R(δω) · UB_indexed · [h, k, l]ᵀ
```

the excitation error is:

```
r_hkl = |q_hkl + s₀| - 1/λ
```

where `s₀` is the incident beam wavevector (|s₀| = 1/λ) and `R(δω)` is a small rotation perturbation applied to the indexed orientation matrix UB. The entire computation — matrix multiply, vector norm, subtraction — is composed of standard differentiable operations.

#### 2.2.2 Partiality from Excitation Error

Assuming a Gaussian relp profile:

```
log_p = -r_hkl² / (2 σ_eff²)
```

where the effective relp width combines mosaicity and beam divergence in quadrature:

```
σ_eff² = σ_mosaicity² + σ_divergence²
```

Note that `log_p` is computed directly, without ever forming `exp(-r²/2σ²)` and then taking the log. This avoids numerical underflow for reflections with large excitation errors.

#### 2.2.3 Bandwidth Contribution

Spectral bandwidth effectively thickens the Ewald sphere into a shell. For a monochromatic beam with bandwidth Δλ/λ, the shell half-thickness at scattering vector s is approximately:

```
Δr_bw ≈ s · (Δλ/λ)
```

This can be incorporated as a resolution-dependent additive term to σ_eff², or by evaluating the integrated partiality over the bandwidth shell analytically (trapezoidal approximation over the wavelength range). For the initial implementation we assume a fixed spectrum from beamline metadata and a simple shell-thickness model.

#### 2.2.4 Lorentz-Type Geometric Correction

Relps at different scattering geometries intersect the Ewald sphere at different incidence angles. A relp that grazes the sphere tangentially samples a different effective path length than one crossing it perpendicularly. This introduces a geometric factor:

```
L ∝ 1 / cos(α)
```

where α is the angle between the relp velocity (through the Ewald sphere during a hypothetical rotation) and the sphere normal. In still-shot geometry this factor modulates the effective integrated volume of the relp that falls within the Ewald sphere. It is folded into the partiality calculation as a multiplicative (additive in log space) correction.

---

## 3. Noise Model Integration

The heteroscedastic Gaussian NLL loss is retained unchanged:

```
L = mean[ (I_obs - I_pred)² / σ² + log(σ²) ]
```

with predicted variance:

```
σ² = I_pred + 2 I_bg + I_pred² · σ_LN(shell)²
```

The partiality enters naturally through I_pred. For a reflection with small partiality:

- **Poisson term** (I_pred): scales linearly with p — less signal, less shot noise
- **Background term** (2 I_bg): unchanged — background is independent of crystal diffraction
- **Lognormal term** (I_pred² · σ_LN²): scales as p² — crystal-to-crystal variation is multiplicative on the full intensity

As p decreases, σ² becomes dominated by the constant background term, making the observation contribute less to the gradient:

```
dL/d(log_I_true) ∝ (I_obs - I_pred) / σ² · I_pred
```

A low-partiality observation naturally self-downweights. **We never divide by partiality**, so there is no noise amplification for barely-sampled reflections. This is fundamentally cleaner than the conventional correct-then-merge approach.

---

## 4. Parameters

### 4.1 Optimised Parameters

| Parameter | Shape | Per | Description |
|---|---|---|---|
| `log_I_true` | (n_unique,) | Reflection | Log merged intensity |
| `log_G` | (n_crystals,) | Crystal | Log absolute scale |
| `B` | (n_crystals,) | Crystal | Isotropic B-factor |
| `δω` | (n_crystals, 3) | Crystal | **NEW:** Orientation perturbation (Rodrigues vector) |
| `σ_mosaicity` | (1,) | Global | **NEW:** Crystal mosaicity (optionally per-crystal later) |
| `σ_divergence` | (1,) | Global | **NEW:** Beam divergence |

### 4.2 Fixed Inputs (from Indexing / Beamline)

| Input | Shape | Description |
|---|---|---|
| `UB_indexed` | (n_crystals, 3, 3) | Orientation matrix from indexing (includes cell) |
| `λ` (or spectrum) | (1,) or per-crystal | Wavelength / bandwidth from beamline |
| `s₀` | (3,) | Incident beam direction |
| `h, k, l` | (n_obs, 3) | Miller indices per observation |
| `s²` | (n_obs,) | Resolution (0.25/d²) per observation |
| `I_obs` | (n_obs,) | Observed integrated intensity (may be negative) |
| `I_bg` | (n_obs,) | Per-observation background from integration |
| `t_shot` | (n_crystals,) | Timestamp per shot (for geometry restraints, optional) |

### 4.3 Re-estimated Each Cycle (Buffers)

| Buffer | Shape | Description |
|---|---|---|
| `σ_LN_prior` | (n_shells,) | Per-shell lognormal spread (unchanged) |
| `shell_per_hkl` | (n_unique,) | Shell assignment per reflection (unchanged) |
| `crystal_weights` | (n_crystals,) | CC-based per-crystal weights (unchanged) |

---

## 5. Computation Graph

The full differentiable forward pass for a single observation *j* from crystal *c*, reflection *hkl*:

```
                   ┌───────────────────────────────────────┐
                   │  Orientation Block                    │
  [h,k,l] ────┐   │                                       │
               ├──>│  q = R(δω) · UB · [h,k,l]            │
  UB_indexed ──┘   │  r = |q + s₀| - 1/λ                  │
  δω ─────────────>│  log_p = -r² / (2 σ_eff²)            │
  σ_m, σ_d ───────>│  σ_eff² = σ_m² + σ_d²                │
                   └───────────────┬───────────────────────┘
                                   │
                              log_p│
                                   │
  ┌────────────────────────┬───────┴─────────────────────┐
  │  Scaling Block         │                             │
  │                        │                             │
  │  log_I_pred = log_I_true + log_G - 2Bs² + log_p     │
  │  I_pred = exp(log_I_pred)                            │
  └────────────────────────┬─────────────────────────────┘
                           │
                      I_pred│
                           │
  ┌────────────────────────┴─────────────────────────────┐
  │  Loss Block                                          │
  │                                                      │
  │  σ² = I_pred.detach() + 2 I_bg + I_pred² · σ_LN²    │
  │  L = (I_obs - I_pred)² / σ² + log(σ²)               │
  └──────────────────────────────────────────────────────┘
```

Gradient flow: loss → I_pred (through residual only, not through σ²) → log_I_pred → all additive terms (log_I_true, log_G, B, log_p → δω, σ_m, σ_d). The detached variance prevents the optimizer from reducing loss by inflating noise.

---

## 6. Optimisation Strategy

### 6.1 Staged Approach

Refinement proceeds in stages to avoid coupling poorly-conditioned parameters early:

**Stage 1: Orientation Refinement (pre-step)**

- Refine only `δω` (3 params per crystal) against the current merged I_true held fixed
- Each crystal has ~200 observations constraining 3 parameters — strongly overdetermined
- Objective: maximise likelihood of crystal's observations given partiality model
- Decouples "where is the relp" from "how bright is it"
- Initialize δω = 0 (trust indexed orientation as starting point)

**Stage 2: Partiality-Aware Scaling**

- Compute partialities from refined orientations + Gaussian relp model
- Fold `log_p` into the existing scaling as an additional additive term
- Re-run full L-BFGS scaling with partialities (log_I_true, log_G, B)
- Same cycle structure as current: calibrate → warmup → iterative refinement

**Stage 3: Joint Refinement (optional)**

- Refine orientations jointly with scale parameters
- Optionally promote σ_mosaicity to per-crystal (1 extra param per crystal)
- Requires careful learning rate tuning — orientation parameters may need smaller step size

### 6.2 Optimiser Details

- **Stages 1 & 2:** L-BFGS, lr=1.0, strong_wolfe, history_size=10 (same as current)
- **Stage 1:** 50 L-BFGS steps per crystal batch (orientation only)
- **Stage 2:** Same cycle structure: 50-step warmup + 15 cycles of 20 steps
- **Stage 3:** Per-parameter learning rates via parameter groups

### 6.3 Regularisation

Existing regularisation is unchanged:

- `100 · mean(log_G)²` — soft centering of absolute scales
- `1e-3 · mean(log_G²) + 1e-3 · mean(B²)` — light L2 on scale parameters

New regularisation for partiality parameters:

- `1e-2 · mean(δω²)` — Tikhonov prior: orientations stay near indexed values
- Optional: time-based smoothness restraint on geometry via `L2(d(geometry)/dt)`

---

## 7. Implementation Plan

| Phase | Task | Complexity | Priority |
|---|---|---|---|
| 1 | Excitation error computation (q_hkl, r_hkl) from UB + hkl + λ | Low — linear algebra | Critical |
| 1 | Rodrigues rotation R(δω) differentiable | Low — standard formula | Critical |
| 1 | Gaussian relp partiality: log_p = -r²/2σ² | Low | Critical |
| 2 | Integrate log_p into SelfConsistentTarget forward model | Low — one additive term | Critical |
| 2 | Stage 1 orientation refinement loop | Medium | Critical |
| 2 | Lorentz geometric correction | Medium | High |
| 3 | Bandwidth shell integration | Medium | High |
| 3 | Per-crystal mosaicity refinement | Low — one param | Medium |
| 4 | Time-restrained geometry smoothing | Medium | Low (future) |

---

## 8. Expected Impact

### 8.1 Low-Resolution Improvement

Partiality variation is largest at low resolution (large d-spacing → relp width is a smaller fraction of Ewald sphere curvature). Explicit partiality modelling should:

- Reduce systematic bias in low-resolution intensity estimates
- Close the gap with Partialator at low resolution
- Maintain or improve the already-superior high-resolution performance

### 8.2 Noise Model Consistency

Because partiality enters multiplicatively in the forward model:

- Low-partiality observations are naturally downweighted through the noise model
- No division by partiality → no noise amplification (unlike conventional `I_obs / p` correction)
- Negative I_obs values remain handled correctly (linear-space residual)

### 8.3 Orientation Refinement Bonus

Even without the full partiality model, small orientation corrections propagate to better excitation errors and hence better intensity estimates. The gradient signal from ~200 reflections per crystal is sufficient to correct orientation errors of tenths of a degree.

---

## 9. Computational Cost

The partiality computation adds per-observation work:

- Matrix multiply: `R(δω) · UB · [h,k,l]` — one 3×3 @ 3×1 matmul per observation (~8M obs)
- Norm + subtraction: `|q + s₀| - 1/λ` — trivial
- Gaussian: `-r² / (2σ²)` — trivial

Total additional cost per L-BFGS evaluation: one batched matmul over 8M vectors. On GPU this is comfortably sub-millisecond. The dominant cost remains the structure factor and gradient computation. The partiality overhead should be negligible (<1% of total cycle time).

Memory overhead: 3 floats per crystal for δω (~120k params for 40k crystals) and one float per observation for the cached log_p. For 8M observations, that is ~32MB — negligible relative to the existing observation arrays.

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Poor indexed orientations cause partiality model to diverge | Orientation refinement pushes δω far from zero, overfitting noise | Tikhonov regularisation on δω; stage 1 refinement against fixed I_true; partiality floor at p_min = 0.01 |
| Mosaicity varies strongly crystal-to-crystal | Single global σ_m underfits; per-crystal adds 40k params | Start global; promote to per-crystal only if residuals show systematic trends. Well-conditioned: 200 obs per param |
| Gaussian relp model is too simple | Real relp profiles are not Gaussian (Lorentzian tails, anisotropy) | Gaussian is a good starting point; can upgrade to Lorentzian or Voigt later. The key physics is r²/σ² |
| Cell parameter errors bias excitation errors systematically | Shifts all relps for a crystal, creating correlated partiality bias | Monitor per-crystal residual trends vs resolution. Can add 6 cell perturbation params later if needed |
| L-BFGS coupling between orientation and intensity parameters | Slow convergence, oscillation | Staged optimisation (stage 1 decouples them); parameter-group learning rates in stage 3 |

---

## Appendix A: Rodrigues Rotation Formula

The orientation perturbation δω is parameterised as a Rodrigues rotation vector. Given `δω = [ω₁, ω₂, ω₃]` with angle `θ = |δω|` and axis `k = δω/θ`:

```
R(δω) = I + sin(θ) [k]× + (1 - cos(θ)) [k]×²
```

where `[k]×` is the skew-symmetric cross-product matrix. For small angles (|δω| ≪ 1), this simplifies to `R ≈ I + [δω]×`, which is useful for understanding the gradient but the full formula should be used in implementation for correctness at larger corrections. PyTorch's `torch.matrix_exp` on the skew-symmetric matrix is an equivalent and numerically stable implementation.