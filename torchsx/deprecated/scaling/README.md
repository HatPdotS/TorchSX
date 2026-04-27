# torchsx.scaling

Per-crystal scale factors and composite loss for gradient-based SFX scaling.

## Modules

| Module | Description |
|--------|-------------|
| `_absolute.py` | `AbsoluteScale` — per-crystal log(G) scale factor |
| `_bscale.py` | `BScale` — per-crystal isotropic B-factor correction |
| `_partiality.py` | `EwaldPartiality` — Gaussian relp partiality with refineable orientation |
| `_orientation.py` | `CrystalOrientation` — per-crystal Rodrigues rotation |
| `_loss.py` | `CompositeScalingLoss` — lognormal + Gaussian loss with per-shell noise |
| `_inputs.py` | `ScalerInputs`, `prepare_scaler_inputs()` — observation-level data for scaling |

## Scale Model

Per-observation log-scale is the sum of independent components:

```
log_scale = log_G[crystal] - 2*B[crystal]*s_sq + log_p(|r|, 1/d)
```

Each component is an `nn.Module` with refineable parameters optimised
via Adam on a composite loss (see `torchsx.target` for loss functions).

## Usage

```python
from torchsx.scaling import AbsoluteScale, BScale, prepare_scaler_inputs

inp = prepare_scaler_inputs(stream._core, n_shells=20)
abs_scale = AbsoluteScale.from_core(stream._core)
b_scale = BScale.from_core(stream._core)

log_scale = (abs_scale.log_scale_per_obs(inp.crystal_id)
             + b_scale.log_scale_per_obs(inp.crystal_id, inp.s_sq))
```

## Note

For partiality modelling beyond the Gaussian relp, use `torchsx.partiality`
which provides `GenLorentzProfile` with resolution-dependent parameters.
