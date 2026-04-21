# SFX Scaler: Composite Lognormal-Gaussian Loss Function
## Design Document

---

## 1. Motivation

Standard SFX merging with partialator using unity partiality achieves ~8% Rsplit
on this dataset despite >1000σ single-spot SNR and 80k patterns. The gap between
the photon statistics floor (~0.4%) and the observed Rsplit is entirely systematic.
Three identified sources drive this floor:

1. **Lognormal observation statistics** — multiplicative error chain
   (partiality × scale × extinction × absorption) produces a lognormal intensity
   distribution. Linear-space merging is a biased estimator of |F|² under lognormal
   noise, and is particularly unstable under sparse sampling because the heavy
   right tail is inconsistently sampled across even/odd splits.


3. **Shot noise heterogeneity** — with intensities spanning 50 to 1×10⁵ photons,
   the relative shot noise varies by 45×. A single likelihood model treating all
   observations equally is suboptimal across this dynamic range.

The composite loss addresses points 1 and 3 directly. Point 2 is addressed
separately via κ-refinement of Pd in the dark structure (see companion document).

---

## 2. Physical Model

Per-observation predicted intensity:

```
I_pred(h, i) = G_i × exp(-2 B_i s²) × |F(h)|²
```

where:

| Symbol | Description |
|--------|-------------|
| `h` | Miller index |
| `i` | Crystal index |
| `G_i` | Per-crystal scale factor |
| `B_i` | Per-crystal isotropic Wilson B-factor |
| `s(h) = \|h\| / 2` | Resolution coordinate (Å⁻¹) |
| `\|F(h)\|²` | Merged structure factor intensity, refinable |

All three per-crystal parameters (`G_i`, `B_i`) plus `|F(h)|²` are jointly
optimized. Sequential optimization — scale then merge — is biased when
partiality and scale are correlated, which they always are for plate-shaped
crystals. Joint optimization eliminates this bias.

Working in log space for numerical stability:

```
log I_pred(h, i) = log|F(h)|² + log G_i - 2 B_i s²
```

---

## 3. Noise Model

The observation model has two additive contributions in consistent χ² units:

### 3.1 Shot Noise (Gaussian)

```
σ_shot²(h, i) = I_obs(h, i)          [Poisson → Gaussian approximation]
```

Valid for I_obs >> 1, which holds throughout the dynamic range of this dataset
(minimum ~50 photons).

### 3.2 Multiplicative Systematic (Lognormal)

The multiplicative error chain:

```
I_obs = P × G × exp(δ_extinction) × exp(δ_absorption) × |F|²
```

produces a lognormal distribution by the CLT in log space. The systematic
width σ_LN is **calibrated empirically per resolution shell from negative
time delay observations** (see Section 5), requiring no model assumptions.

In log space the lognormal becomes Gaussian with width σ_LN(shell):

```
log I_obs ~ N(log I_pred, σ_LN²)     [systematic component]
```

### 3.3 Regime Crossover

The two noise sources have opposite intensity dependence:

```
Relative shot noise:        σ_shot / I_obs = 1/√I_obs    → decreases with I
Relative systematic noise:  σ_LN                          → constant
```

Crossover intensity where they are equal:

```
I_crossover(shell) = 1 / σ_LN(shell)²
```

For σ_LN ~ 0.05–0.10 this gives I_crossover ~ 100–400 photons, within the
observed dynamic range. The loss function must handle both regimes correctly.

---

## 4. Loss Function

### 4.1 Per-Observation Loss

```
L(h, i) = w_gauss(h, i) × (I_pred - I_obs)²
         + strong(h, i) × w_log(h, i) × (log I_pred - log I_obs)²
```

### 4.2 Weights

**Gaussian weight** — all observations:

```
w_gauss(h, i) = 1 / I_obs(h, i)
```

**Lognormal weight** — strong observations only:

```
w_log(h, i) = 1 / σ_LN(shell(h))²
```

**Hard threshold mask**:

```
strong(h, i) = I_obs(h, i) > I_min(shell(h))

where I_min(shell) = I_crossover(shell) = 1 / σ_LN(shell)²
```

The threshold is the 1σ crossover point — below it, shot noise exceeds the
systematic contribution and the log term adds noise rather than signal. The
mask has no dependence on optimized parameters, so gradients are clean.

### 4.3 Behavior by Regime

| Intensity | Active terms | Effective estimator |
|-----------|-------------|-------------------|
| I < I_min | Gaussian only | Weighted mean in linear space |
| I ≈ I_min | Both, equal weight | Transition |
| I >> I_min | Lognormal dominates | Geometric mean in log space |

The geometric mean (log-space mean) is an **unbiased estimator of |F|²** under
lognormal noise:

```
E[log I_obs] = log|F|²   →   exp(mean(log I_obs)) = |F|²
```

whereas the linear mean is biased upward by exp(σ_LN²/2) — approximately
0.25–1% for σ_LN ~ 0.05–0.10, which is non-negligible relative to the 8%
Rsplit floor.

### 4.4 Total Loss

```
L_total = Σ_{h,i} L(h, i)
```

Summed over all observations across all crystals and all time points.

---

## 5. σ_LN Calibration From Negative Time Data

Negative time delay observations (t < 0, nominally t = −250 fs) sample the
dark structure only. The true signal is stationary — any variation is pure
noise. This provides a direct empirical noise characterization with no model
dependence.

### 5.1 Calibration Procedure

```python
def calibrate_sigma_LN(I_obs_neg, log_G_neg, shells, I_min_abs=50.0):
    """
    Estimate per-shell lognormal systematic width from negative time data.
    
    Parameters
    ----------
    I_obs_neg : Tensor (N,)   observed intensities at t < 0
    log_G_neg : Tensor (N,)   per-crystal log scale at t < 0
    shells    : Tensor (N,)   resolution shell index per observation
    I_min_abs : float         absolute minimum intensity to include
    """
    sigma_LN = torch.zeros(n_shells)
    
    for s in range(n_shells):
        mask = (shells == s) & (I_obs_neg > I_min_abs)
        I_s = I_obs_neg[mask]
        G_s = log_G_neg[mask]
        
        # Log residuals after removing per-crystal scale
        log_resid = torch.log(I_s) - G_s
        
        # Total log-space variance
        sigma_total_sq = log_resid.var()
        
        # Shot noise contribution in log space (delta method: σ_log ≈ 1/√I)
        sigma_shot_sq = (1.0 / I_s).mean()
        
        # Systematic component — clamp to zero to avoid sqrt of negative
        sigma_LN[s] = torch.sqrt((sigma_total_sq - sigma_shot_sq).clamp(min=0))
    
    return sigma_LN
```

### 5.2 Diagnostic Value

Beyond calibration, the per-shell σ_LN distribution is a diagnostic for
systematic effects:

| Pattern | Interpretation |
|---------|---------------|
| σ_LN elevated at low resolution | Extinction, Pd IAM errors |
| σ_LN varies with crystal orientation class | Partiality model misspecification |
| σ_LN varies across negative time bins | Beam condition drift, crystal quality variation |
| σ_LN flat across shells | Clean systematics, well-calibrated |

---

## 6. PyTorch Implementation

```python
class SFXScaler(nn.Module):
    def __init__(self, n_reflections, n_crystals, n_shells):
        super().__init__()
        # Refinable parameters
        self.log_F_sq = nn.Parameter(torch.zeros(n_reflections))
        self.log_G    = nn.Parameter(torch.zeros(n_crystals))
        self.B        = nn.Parameter(torch.zeros(n_crystals))
        
        # Calibrated noise model — not refined, set from negative time data
        self.register_buffer('sigma_LN', torch.ones(n_shells) * 0.05)

    def predict(self, h, crystal_idx, s_sq):
        log_I_pred = (self.log_F_sq[h]
                    + self.log_G[crystal_idx]
                    - 2.0 * self.B[crystal_idx] * s_sq)
        return log_I_pred

    def loss(self, I_obs, h, shell, crystal_idx, s_sq):
        log_I_pred = self.predict(h, crystal_idx, s_sq)
        I_pred     = torch.exp(log_I_pred)

        # Gaussian term — all observations
        w_gauss    = 1.0 / I_obs
        gauss_term = w_gauss * (I_pred - I_obs)**2

        # Lognormal term — strong observations only
        I_min  = 1.0 / self.sigma_LN[shell]**2        # per-shell threshold
        strong = (I_obs > I_min).float()               # hard mask, no gradient
        w_log  = 1.0 / self.sigma_LN[shell]**2
        log_term = w_log * (log_I_pred - torch.log(I_obs))**2

        return (gauss_term + strong * log_term).sum()
```

---

## 7. Parameter Count and Overdetermination

With >100 spots per crystal:

| Parameter set | Count | Observations / parameter |
|--------------|-------|--------------------------|
| `log_F_sq` (n_reflections) | ~10,000 | >>100 (heavily redundant) |
| `log_G` (n_crystals) | 80,000 | ~100 per crystal |
| `B` (n_crystals) | 80,000 | ~100 per crystal |
| **Total** | ~170,000 | well-determined |

The per-crystal parameters are independent across crystals — the sparse
structure allows efficient block-diagonal Jacobian computation.

---

## 8. Expected Impact on Rsplit

The composite loss improves **accuracy** rather than internal consistency
directly. However, Rsplit can improve indirectly through:

1. **Removal of intensity-dependent bias** — linear merging over-weights
   strong low-resolution Pd-dominated reflections; lognormal correction
   removes this bias and the cross-shell contamination it causes.

2. **Stable estimation under sparse sampling** — log-space variance scales
   as σ_LN²/N vs the lognormal tail-dominated linear-space variance, which
   scales as exp(σ_LN²)/N. For sparse time bins (long delays, logarithmic
   coverage) the improvement in estimator variance is:

   ```
   Variance ratio = exp(σ_LN²) - 1 / σ_LN²  ≈  1 + σ_LN²/2 + ...
   ```

   For σ_LN ~ 0.1 this is ~0.5% improvement per shell. For σ_LN ~ 0.3
   (worst-affected reflections) this is ~5% — non-negligible.

3. **CC½ improvement** — CC½ is more sensitive than Rsplit to
   intensity-dependent bias and should show a more direct improvement.

The primary downstream benefit is cleaner phases for difference map
computation and more reliable extrapolated structure factors for the
azoarene TR-SFX signal extraction.

---

## 10. Open Questions

- **Partiality**: plate geometry produces rod-shaped reciprocal lattice points.
  Ellipsoidal partiality correction (σ_ab, σ_c per crystal from orientation
  matrix) would add 2 parameters per crystal and directly address the xsphere
  failure mode. Not included in current design to maintain 2-parameter
  per-crystal budget and avoid scaling bias risk.

- **Activation fraction weighting**: pumped observations have additional
  variance from crystal-to-crystal activation heterogeneity. Current model
  treats this as absorbed into σ_LN for pumped time bins. A more principled
  treatment would model the activation fraction distribution explicitly and
  propagate it into per-observation weights.

- **Reflection multiplicity weighting**: for very low multiplicity reflections
  (n < 5) even the log-space estimator has elevated variance. An additional
  multiplicity-dependent prior on |F(h)|² could stabilize these reflections
  without introducing bias.