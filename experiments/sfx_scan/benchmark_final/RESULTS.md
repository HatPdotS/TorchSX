# Benchmark: Poisson-GenGauss EM vs Partialator

## Summary

We developed and validated a Poisson marginal likelihood approach for merging
serial femtosecond crystallography (SFX) data.  The method models each
observation as:

    n_peak ~ Poisson(I0 * scale * exp(eps) + bg_total)
    eps ~ GenGauss(0, alpha, beta)

where `bg_total = bg_per_pixel * n_peak_pixels` is the total background in the
integration region.  I0 is estimated by maximising the marginal likelihood via
LBFGS in log(I0) space, with numerical integration over eps using an adaptive
softplus quadrature grid.

## Data

- 4 negative-delay dark streams from SwissFEL SFX (P-1, 12960 eV)
- 31648 crystals, 6.5M observations, ~15700 unique HKLs
- Resolution: 0.9-10.0 Angstrom
- Detector: JUNGFRAU 4M, CrystFEL ring integration (r_peak=7, r_bg=11)

## Methods compared

| Method | Description |
|--------|-------------|
| **EM Poisson** | `fit_em()` with Poisson marginal MLE, ScaleBCorrection, 10 EM cycles |
| GenGauss | `fit_gengauss()` log-space IRLS, positive obs only, ScaleBCorrection |
| Part. unity | CrystFEL `partialator --model=unity -n 10` |
| Part. xsphere | CrystFEL `partialator --model=xsphere -n 3` |

All comparisons use split-half validation (seed=42) with strict 0.9-10.0
Angstrom resolution cutoff and 14100 matched reflections.

## Results

### Global metrics

| Method | CC1/2 | CC* | R_split | N |
|--------|-------|-----|---------|---|
| EM Poisson | 0.9872 | 0.9968 | 0.0605 | 14100 |
| GenGauss | 0.9869 | 0.9967 | 0.0435 | 14085 |
| Part. unity | 0.9850 | 0.9962 | 0.0762 | 14100 |
| Part. xsphere | 0.9600 | 0.9897 | 0.1161 | 14100 |

### Key findings

1. **EM Poisson achieves the highest CC1/2** (0.987) of all methods, beating
   partialator unity (0.985) and GenGauss (0.987 -- tied within noise).

2. **R_split is 20% lower than partialator** (0.06 vs 0.08) across all
   resolution shells.  GenGauss achieves the lowest R_split (0.04) but
   excludes weak/negative reflections.

3. **The Poisson model correctly handles negative observations**.  Unlike
   GenGauss (log-space, positive only), EM Poisson uses all observations
   including negatives, giving unbiased I0 estimates for weak reflections.

4. **Alpha is resolution-dependent and larger than log-space estimates**.
   EM Poisson: med(alpha) = 1.65.  GenGauss: med(alpha) = 1.25.  The
   difference arises because GenGauss only sees positive observations
   (truncated distribution), underestimating the true spread.

5. **Per-crystal ScaleBCorrection reduces alpha by ~0.06** (1.67 -> 1.61),
   confirming that scale corrections remove real inter-crystal variation.

## Technical details

### Poisson noise model

CrystFEL's ring integration sums n_peak pixels in the peak region and
estimates background from n_bg pixels in the annulus:

    I_obs = peak_counts - bg_estimate
    bg_estimate = (annulus_counts / n_bg) * n_peak

The total background entering the Poisson model is `bg_total = bg_per_pixel *
n_peak` (not the per-pixel value).  Getting this wrong (using per-pixel bg)
caused a 2.3x systematic bias in I0 -- the single most important bug fix.

### Quadrature grid

The softplus quadrature grid maps uniform t-spacing to signal via
`signal = softplus(t)`, giving log-spacing at small signal and linear spacing
at large signal.  Key parameters:

- `n_quad = 51` points per observation
- `signal_lo = max(0.5 * sigma_floor, 0.01)` where `sigma_floor = sqrt(bg_mean / n_obs)`
- `signal_max = n_ph - bg + 5 * sqrt(n_ph)`

The grid lower bound is set by the Poisson detection floor -- the weakest
signal distinguishable from background noise given the multiplicity.

### Truncation bias correction

The grid floor at signal_lo biases I0 upward for reflections near the
detection limit.  After the EM loop converges, we apply an analytical
correction using the inverse Mills ratio of the truncated normal:

    eps_min = log(signal_lo / I0)
    z = eps_min / alpha
    bias = alpha * phi(z) / Phi_c(z)
    I0_corrected = I0 * exp(-bias)

This shifts weak reflections down without affecting strong ones (< 1%
correction for I0 > 30, ~15% for I0 = 3-10, ~55% for I0 = 0.5-1).

### Detection limit

The per-HKL detection limit depends only on background and multiplicity:

    sigma_floor = sqrt(bg_total / n_obs)

For this dataset: bg_total ~ 92 photons, median n_obs = 385, giving
sigma_floor ~ 0.49.  Reflections with I0 below this are indistinguishable
from noise.

### Code changes from initial version

1. **Replaced `fit_I0_quadrature`** (old, buggy Gauss-GenGauss grid) with
   `fit_I0_poisson` (validated softplus grid with proper Poisson PMF).

2. **Fixed per-pixel vs total background bug** in `_em.py`: `bg_all` must be
   `bg_per_pixel * n_peak`, not `bg_per_pixel`.

3. **Fixed NaN from negative lambda** in `build_softplus_grid`: clamped
   `lam = (signal + bg).clamp(min=1e-6)`.

4. **Added Poisson floor grid**: `signal_lo = 0.5 * sqrt(bg_mean / n_obs)` per
   HKL, replacing the heuristic `signal_max * 0.01`.

5. **Added post-EM truncation correction** via inverse Mills ratio.

6. **Removed deprecated code**: `fit_I0_newton`, `fit_I0_quadrature`,
   `fit_I0_log`, `fit_joint`, `JointConfig`, and the joint mode in
   `fit_gengauss`.

## Reproducibility

```bash
# Fit all methods (GPU, ~6 min)
sbatch experiments/sfx_scan/benchmark_final/fit.py

# Evaluate and generate plots (CPU, ~30s)
python experiments/sfx_scan/benchmark_final/evaluate.py
```

Output files in `experiments/sfx_scan/benchmark_final/`:
- `benchmark_final.png` -- CC1/2 and R_split vs resolution
- `scatter_self_consistency.png` -- F(half1) vs F(half2) with detection limit
- `scatter_cross_method.png` -- cross-method F comparisons
- `scatter_ratio.png` -- relative difference vs intensity
- `scatter_alpha.png` -- alpha self-consistency and resolution dependence
- `benchmark_final_shells.csv` -- per-shell statistics
