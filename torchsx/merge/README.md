# torchsx.merge

EM-based merging of unmerged SFX intensities with heteroscedastic noise modelling.

## Generative Model

```
I_obs_j = I0_h * exp(epsilon_j) + eta_j

epsilon ~ GenGauss(0, alpha_h, beta_h)   # multiplicative scatter
eta     ~ N(0, sigma2_poi_j)             # Poisson shot noise
```

The EM loop alternates:
- **S-step**: fit per-crystal corrections (scale G, B-factor, panel gain) via ridge OLS
- **D-step**: fit per-HKL I0 via quadrature marginal likelihood, alpha via log-space MLE

## Modules

| Module | Description |
|--------|-------------|
| `_em.py` | `fit_em()` — main EM loop with warm-up schedule and correction pipeline |
| `_dstep.py` | `fit_I0_quadrature()`, `fit_I0_newton()`, `fit_alpha()`, `d_step()`, `fit_joint()` |
| `_distribution.py` | `HKLDistribution`, `fit_gengauss()`, shell beta fitting, Laplacian smoothing |
| `_corrections.py` | `Correction` ABC, `ScaleBCorrection`, `PanelCorrection`, `FixedCorrection`, `CorrectionPipeline` |
| `_noise.py` | `PoissonNoiseModel`, `compute_sigma2_LN()` |
| `_hkl.py` | `compute_hkl_index()`, `merge_intensities()` |
| `_stats.py` | `shell_stats()`, `cc_half()`, `cc_star()`, `r_split()`, completeness, multiplicity |
| `_filter.py` | Resolution, outlier, crystal, and image filtering |
| `_structure_factor.py` | `MergedStructureFactor` nn.Module (log F per HKL) |
| `_neural_model.py` | Neural network correction model |
| `_neural_correction.py` | Crystal-context transformer for learned corrections |
| `_train_correction.py` | Training loop for neural correction network |

## Usage

```python
from torchsx.merge import fit_em, EMConfig, ScaleBCorrection

config = EMConfig(n_em_cycles=15, min_obs=20)
dist = fit_em(stream, config=config, corrections=[ScaleBCorrection()])

# dist.I0    — per-HKL intensity
# dist.alpha — per-HKL GenGauss width
# dist.beta  — per-HKL GenGauss shape
```

## Quadrature D-step

The I0 estimator uses exact marginal likelihood via quadrature over the GenGauss
distribution. The quadrature range adapts to beta so that heavy-tailed distributions
(beta < 1) are not truncated. For beta = 1.3 (typical SFX), the grid uses ~29
points; for beta = 0.8, ~88 points.

A Newton fallback (`i0_method="newton"`) uses the Gaussian variance approximation
and is immune to quadrature truncation.
