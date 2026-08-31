# Heteroscedastic Noise Model Validation

Synthetic tests for the D-step estimators under the exact generative model:

```
epsilon ~ GenGauss(0, alpha, beta)
I_obs   = I0 * exp(epsilon) + N(0, sigma2_poi)
```

## Test Modules

| Module | Description |
|--------|-------------|
| `generate.py` | Synthetic data generator with ground-truth parameters |
| `test_recovery.py` | Basic I0 and alpha recovery for quadrature, Newton, and full d_step |
| `test_multiplicity.py` | Precision vs observations-per-HKL (5 to 200) |
| `test_beta_screening.py` | Recovery vs GenGauss shape beta (0.6 to 2.0), quadrature range validation |

## Running

```bash
# All heteroscedastic tests
python -m pytest tests/test_heteroscedastic/ -v -s

# Just multiplicity screening
python -m pytest tests/test_heteroscedastic/test_multiplicity.py -v -s

# Just beta screening
python -m pytest tests/test_heteroscedastic/test_beta_screening.py -v -s
```

## What the Tests Validate

1. **I0 recovery**: Both quadrature and Newton recover I0 within <10% median
   relative error at 50 obs/HKL, and the two methods agree within 5%.

2. **Alpha recovery**: Per-HKL GenGauss width alpha is recovered with
   correlation >0.85 when alpha varies across HKLs.

3. **Multiplicity scaling**: Error decreases with more observations per HKL.
   Even 5 obs/HKL gives correlation >0.9 with ground truth.

4. **Beta robustness**: Quadrature adapts its integration range to heavy-tailed
   distributions (beta < 1) and maintains recovery precision.

5. **Sigma calibration**: Reported sigma(I0) provides honest coverage
   (~68% of true values within 1 sigma).
