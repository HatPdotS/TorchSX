# torchsx.partiality

Partiality modelling for serial crystallography: excitation error computation,
partiality profile fitting, and rank-concordance orientation refinement.

## Architecture

Three layers compose into a full partiality model:

```
ExcitationError  ──>  |r| per observation
       +
PartialityProfile ──>  log(p) = f(|r|, 1/d)
       =
PartialityCorrection ──>  log-space correction for EM merge
```

## Modules

| Module | Description |
|--------|-------------|
| `_excitation.py` | `ExcitationError` — per-crystal delta_omega (2 DOF) + delta_lambda |
| `_profile.py` | `GaussianProfile`, `GenLorentzProfile` — partiality vs excitation error |
| `_concordance.py` | `RankConcordanceLoss` — model-free pairwise concordance loss |
| `_pairs.py` | `PairData`, `build_pairs()`, `fit_BG()` — pair construction for refinement |
| `_refine.py` | `refine_orientations()` — Adam optimisation with soft L2 restraints |
| `_fit.py` | `fit_partiality_model()` — heteroscedastic Gaussian NLL fitting |
| `_correction.py` | `PartialityCorrection` — adapter to `torchsx.merge.Correction` interface |
| `_pipeline.py` | `run_partiality_pipeline()` — iterative EM + orientation + partiality |

## GenLorentz Profile

The generalised Lorentzian fits SFX data better than Gaussian, exponential, or
standard Lorentzian profiles:

```
log(p) = a0(s) - log(1 + (|r| / gamma(s))^alpha(s))
```

where gamma, alpha, a0 are polynomials in 1/d (resolution). Typical values:
alpha ~ 0.6 (low-res) to 1.6 (high-res), gamma ~ 0.01-0.9 x 10^-3 A^-1.

## Pipeline

```python
from torchsx.partiality import run_partiality_pipeline, PipelineConfig

config = PipelineConfig(
    n_outer_cycles=4,
    refinement=RefinementConfig(restraint_omega=300.0),
)
result = run_partiality_pipeline(stream, config)
```

Each outer cycle:
1. EM merge with current corrections (ScaleB + partiality)
2. Rank-concordance orientation refinement (cold start each cycle)
3. Fit GenLorentz partiality profile
4. Diagnostics (rho(I, |r|), convergence monitoring)
