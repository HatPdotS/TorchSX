# torchsx.forward

Pixel-level differentiable forward model for serial crystallography.

Predicts pixel intensities from crystal parameters (orientation, unit cell,
B-factor) through a differentiable path, enabling gradient-based refinement
of all parameters jointly.

## Modules

| Module | Description |
|--------|-------------|
| `forward_model.py` | `PixelForwardModel` — full differentiable path: parameters to pixel counts |
| `predict.py` | `PredictedReflections` — reflection prediction and diagnostics |
| `spectrum.py` | `GaussianSpectrum`, `PCASpectrum` — spectral models for polychromatic beams |
| `batch.py` | `ROIBatch`, `ROIBatchLoader` — batch processing for regions-of-interest |
| `sparse_batch.py` | `SparseBatch`, `SparseBatchLoader` — sparse data batching |
| `frame_batch.py` | `FrameBatchLoader` — frame-level batch processing |
| `coords.py` | `RefinableGeometry` — refinable detector geometry parameters |
| `relp.py` | `RelpProfile` — reciprocal lattice point profile model |
| `corrections.py` | Polarization correction, solid angle correction |
| `likelihood.py` | `PixelLikelihood` — pixel-level likelihood |
| `_censored_poisson.py` | Censored Poisson distribution for truncated data |
| `_soft_assignment.py` | Soft assignment of pixels to reflections |
| `_pixel_matching.py` | Pixel-to-reflection matching |
| `_attraction_loss.py` | Attraction loss for orientation refinement |
| `_profile_fitted.py` | Profile fitting utilities |

## Usage

```python
from torchsx.forward import PixelForwardModel, predict_reflections

model = PixelForwardModel(geometry, spectrum)
predicted = predict_reflections(stream, model)
```
