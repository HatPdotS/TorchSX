# TorchSX

GPU-friendly stacked tensors for unmerged serial crystallography data.

TorchSX provides a PyTorch-native framework for processing, merging, and
scaling Serial Femtosecond Crystallography (SFX) data. It reads CrystFEL
`.stream` files into stacked tensors at three levels (observations, crystals,
images) and provides differentiable merging, scaling, and partiality correction.

## Amortised scale model

`torchsx.nn` trains one network to predict a per-observation scale, merging
through a posterior integral over `I0` rather than a point estimate, so `I` and
`sigma_I` come from the same integral. See
`experiments/docs/amortized_scale_plan.md` for the design and
`experiments/optimize_integration/README.md` for the measured state of the
integration — the node settings that should be used, the background likelihood,
and the measurement traps that produced two wrong conclusions along the way.

Validated against refinement rather than only CC1/2, since a merge can be
systematically wrong and perfectly self-consistent — and on this data it
repeatedly is. **R_free 0.1269 (`nll`) / 0.1252 (`ml`)** against 0.1825 for
partialator + French–Wilson on the same model, cell and free set, with the
weak-end compression slope at 1.02 / 0.95 where the uncompensated merge gives
0.16 / 0.18.

That the two must be checked separately is not a formality here: the
single best CC1/2 of any run trained (0.9982) came from a configuration whose
merged intensities are **not monotone** in `F_model` and whose R_free is the
worst of the trained set.

## Command line

Three commands cover the amortised-scale workflow end to end.  The integration
radii are required — CrystFEL does not record them in the stream and the
background scales with `n_peak` — and everything else a merge must not guess is
carried in the checkpoint.

```bash
# stream(s) -> trained network + merged MTZ
python -m torchsx.cli.train dark.stream --spacegroup "P 1" \
    --int-radius 7,9,11 --gain 1.0 --dmin 0.9 -o run/

# trained network + data -> MTZ (recipe read from the checkpoint)
python -m torchsx.cli.merge_nn run/best.pt dark.stream -o merged.mtz

# no weighting at all: K = 1 with one global width / shape / skew fitted
# to the same evidence -- the reference a trained run has to beat
python -m torchsx.cli.merge_global dark.stream --spacegroup "P 1" \
    --int-radius 7,9,11 --gain 1.0 --dmin 0.9 -o unity.mtz
```

Crystals can be divided into named **splits** (`--split dark=dark.stream
--split light=1ps.stream`, or `--split-file labels.pt` for one stream): batches
are then split-pure, each split keeps its own merge target, and one MTZ is
written per split.  The network never sees the label.

On the SwissFEL small molecule, same model and free set: R_free **0.118**
(trained) / **0.123** (global fit, no weighting) / 0.147 (partialator).

## Installation

```bash
pip install -e .
```

Requires Python >= 3.10 and `torchref >= 0.4` (provides torch, numpy, gemmi).

## Quick Start

```python
from torchsx import SFXStream

# Load data
stream = SFXStream.from_stream("data.stream", spacegroup="P 1")
stream = stream.filter_resolution(d_min=2.0).filter_cell(tolerance=0.1)

# Merge with EM (heteroscedastic noise model)
from torchsx.merge import fit_em, EMConfig, ScaleBCorrection
dist = fit_em(stream, config=EMConfig(), corrections=[ScaleBCorrection()])

# Statistics
from torchsx.merge import shell_stats
for s in shell_stats(stream, n_bins=20):
    print(f"d={s.d_min:.1f}-{s.d_max:.1f}: CC1/2={s.cc_half:.3f}")
```

## Package Structure

```
torchsx/
├── stream.py          SFXStream — public API (load, filter, merge, save)
├── _core.py           _SFXStreamCore — stacked-tensor dataclass
├── geometry.py        Detector geometry (CrystFEL .geom format)
│
├── io/                I/O: .stream parsing, .sfx safetensors, write-back
├── merge/             EM-based merging with heteroscedastic noise model
├── scaling/           Per-crystal scale factors (G, B, partiality)
├── target/            Scaling loss functions (KL, self-consistent)
├── partiality/        Partiality modelling and orientation refinement
└── forward/           Pixel-level differentiable forward model
```

## Submodules

### `torchsx.io` — I/O

Parse CrystFEL `.stream` files, save/load `.sfx` binary format (safetensors),
extract pixel patches from HDF5 frames.

### `torchsx.merge` — Merging

EM-based intensity merging under a generalized Gaussian multiplicative noise
model with Poisson shot noise. The D-step uses exact marginal likelihood via
adaptive quadrature. Corrections (scale, B-factor, panel gain, partiality)
are modular and compose via a `CorrectionPipeline`.

### `torchsx.scaling` — Scaling

Per-crystal absolute scale (`G`), B-factor, and Ewald-offset partiality as
differentiable `nn.Module`s. Optimised via Adam on composite lognormal-Gaussian
loss or the KL scaling target.

### `torchsx.target` — Scaling Targets

`KLScalingTarget`: unified log-space NLL with truncated Gaussian for negatives
and floor penalty to anchor noise to calibrated priors.

### `torchsx.partiality` — Partiality

Rank-concordance orientation refinement (model-free, differentiable) and
generalised Lorentzian partiality profile fitting. The iterative pipeline
alternates EM merging, orientation refinement, and partiality estimation.

### `torchsx.forward` — Forward Model

Pixel-level differentiable forward model: predicts pixel intensities from
crystal parameters through a fully differentiable path for joint refinement.

## Data Model

TorchSX uses a three-level stacked-tensor representation:

| Level | Count | Key Fields |
|-------|-------|------------|
| Observations | N_obs | hkl, I, sigma, background, crystal_id |
| Crystals | N_crystals | astar, bstar, cstar, cell, image_id |
| Images | N_images | beam_energy, det_shift_x, det_shift_y |

Join keys: `crystal_id` (obs -> crystal), `image_id` (crystal -> image).

All filtering operations return new immutable objects.

## Tests

```bash
# Fast tests (no data files needed)
python -m pytest tests/ -m "not slow" -v

# Full suite (needs merged_368.stream)
python -m pytest tests/ -v
```

## Experiments

Exploratory scripts, benchmarks, and one-off analyses live in `experiments/`.
These are not part of the installable package.
