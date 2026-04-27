# torchsx.target

Scaling targets (loss functions) for joint refinement of per-crystal scale
factors and merged structure factors.

## Modules

| Module | Description |
|--------|-------------|
| `_kl.py` | `KLScalingTarget` — log-space NLL + truncated Gaussian + floor penalty |
| `_self_consistent.py` | `SelfConsistentTarget` — self-consistent scaling target |

## KLScalingTarget

Unified loss with three components:

1. **Positive observations** (log-space NLL): penalises deviation of log(I_obs) from log(I_true)
2. **Non-positive observations** (truncated Gaussian): constrains I_true from above for weak reflections
3. **Floor penalty**: anchors noise parameters to calibrated priors to prevent noise absorption

Key methods:
- `calibrate()` — initialise from data (per-crystal scale removal, per-shell noise estimation)
- `forward()` — compute total loss
- `compute_crystal_weights()` — EM-style per-crystal confidence weights

## Usage

```python
from torchsx.target import KLScalingTarget

target = KLScalingTarget(n_unique, n_shells)
target.calibrate(I_obs, total_scale, hkl_id, shell, crystal_id)
loss = target(I_obs, total_scale, hkl_id, weight)
```
