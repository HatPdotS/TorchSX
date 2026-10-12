<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/HatPdotS/TorchSX/master/assets/torchsx-banner-dark.png">
    <img src="https://raw.githubusercontent.com/HatPdotS/TorchSX/master/assets/torchsx-banner-light.png" alt="TorchSX" width="100%">
  </picture>
</p>

<h1 align="center">A Py<ins>Torch</ins>-based scaling and merging library for serial crystallography (<ins>SX</ins>)</h1>

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![TorchRef](https://img.shields.io/badge/TorchRef-0.7-f97316.svg)](https://github.com/HatPdotS/TorchRef)
[![CUDA](https://img.shields.io/badge/CUDA-supported-76b900.svg)](https://developer.nvidia.com/cuda-zone)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

In serial crystallography every reflection is seen once per crystal, with an
unknown partiality, incident intensity and crystal volume. Established programs
scale each crystal with a few parameters and average. TorchSX instead trains one
neural network per dataset that predicts, for every observation, a scale factor
and the parameters of a multiplicative scatter distribution from that
observation's crystal- and image-level context.

Merging is then a posterior integral rather than an average: for each reflection
the latent intensity is integrated under a Wilson prior over a per-observation
likelihood that convolves the learned scatter law with a Poisson detector law,
including the variance of the background estimate. The amplitude and its
uncertainty come out of the same posterior, and the network is trained on the
marginal likelihood of the data alone — never against a reference structure.

> **Scope.** TorchSX starts from integrated observations; peak finding and
> indexing belong to CrystFEL or Probixi, and refinement to TorchRef.
> Training length remains an empirical setting: validation NLL is not established
> as an optimum for independent difference-map precision.

## Where it sits

TorchSX occupies the scaling and merging step. TorchRef sits on both sides of it
— it is the tensor, structure-factor and GPU substrate TorchSX is built on, and
it provides the refinement and difference-map tools that consume the merge.

```
  CrystFEL / Probixi           TorchSX                    TorchRef
  -----------------------  ->  ---------------------  ->  ---------------
  peak finding, indexing,      amortized scaling,         refinement,
  integration                  posterior merging          difference maps

  .stream                      F, sigma(F) per state      model, maps
```

## Key features

- **Amortized scaling.** One network predicts a per-observation scale and scatter
  law from crystal- and image-level context, rather than fitting a handful of
  parameters per crystal.

- **Posterior merging.** The merged amplitude and its standard uncertainty are
  moments of one posterior integral, not an average and a propagated error.

- **Reference-free objective.** Training maximizes the marginal likelihood of the
  observations. No reference structure, target set or external intensity enters
  the objective.

- **Joint multi-state training.** Light and dark (or any set of labelled states)
  share one network while remaining separate merge targets. The state label is
  not a network feature.

- **Gain calibrated from the data.** The detector gain is estimated from the input
  and reconciled with its declared value, because converting to photons wrongly
  misspecifies the detector law.

- **GPU quadrature.** The posterior integral runs as a Triton kernel with an
  analytic backward pass; encoder chunking bounds GPU memory without reducing the
  requested statistical batch support.

- **Built on PyTorch.** PyTorch throughout, so the model, likelihood and
  quadrature are ordinary differentiable modules that compose with the rest of a
  PyTorch workflow.

## Quickstart

```bash
pip install torchsx

torchsx.train indexed.stream --spacegroup 'P 63' --dmin 2.0 -o run/
```

That writes checkpoints and merged MTZs into `run/`. Everything below expands on
it: separate readout of a checkpoint, joint light/dark states, difference maps,
and the observation statistics that motivate the model.

## Install

```bash
pip install torchsx
```

Python >=3.10. torchsx depends on TorchRef 0.7, which brings torch, triton and
gemmi. `.stream` files are read and written by `torchsx-native`, a small Rust
package built from `native/` in this repository and released with torchsx;
`pip install torchsx` installs the prebuilt wheel for your platform. Optional
extras are `duckdb` (the existing Probixi reader) and `dev`.

PyPI's default torch build needs a recent NVIDIA driver. If
`torch.cuda.is_available()` is false on a GPU machine, install the torch build
for your driver first (see [pytorch.org](https://pytorch.org/get-started/locally/)),
within TorchRef's bound, for example for a CUDA 12 driver:

```bash
pip install "torch<2.14" --index-url https://download.pytorch.org/whl/cu128
pip install torchsx
```

From a checkout, build the native package first (needs a Rust toolchain, see
[native/README.md](https://github.com/HatPdotS/TorchSX/blob/master/native/README.md)), then install torchsx against it:

```bash
pip install ./native
pip install -e .
```

If `torchsx_native` cannot be imported, torchsx reads and writes streams with
slower pure-Python code and warns once.

## Train and merge

```bash
torchsx.train indexed.stream --spacegroup 'P 63' --dmin 2.0 -o run/
torchsx.merge run/best.pt indexed.stream -o merged.mtz
```

Training writes `best.pt`, `last.pt`, `run.json`, `metrics.jsonl`,
`training_summary.json` and full/half merged MTZs. `--no-merge` stops after saving
checkpoints, for workflows that schedule readout separately.

`--steps` is the total optimizer-update budget, independent of dataset size
(default 400). Training alternates state-pure batches: 400 updates over two
states gives 200 per state. An uneven budget differs by at most one update between states. Training
length remains empirical. `best.pt` selects minimum validation NLL in state zero;
`last.pt` preserves the endpoint. Validation NLL is not established as an optimum
for independent difference-map precision. Validation uses 5% of state-zero
crystals; all supplied observations remain eligible for the final merge.

The statistical batch size is controlled by `--redundancy` (default 100 target
observations per reflection). A small state uses all its available training
crystals. Data stay on the host; encoder chunking and quadrature recomputation
bound GPU memory without reducing the requested statistical batch support.
Parser workers respect the available CPU allocation, capped at eight.

Measurement policy, gain selection and the full public option surface —
`--gain`, `--int-radius`, `--free-from`, `--cache`, `--seed` and the rest — are
in the [command-line reference](https://github.com/HatPdotS/TorchSX/blob/master/docs/cli.md).

## Joint light/dark inputs

Train both states in one network; the merge targets remain separate. The label
is not a network feature.

```bash
torchsx.train --split dark=dark.stream --split light=light.stream \
  --spacegroup 'P 63' --dmin 2.0 -o joint/
torchsx.merge joint/best.pt \
  --split dark=dark.stream --split light=light.stream -o merged.mtz
```

A combined stream can instead use a CrystFEL custom-split file with
`filename event dataset` columns:

```bash
torchsx.train combined.stream --split-file states.split \
  --spacegroup 'P 63' --dmin 2.0 -o joint/
```

`--split-file` also accepts an integer label tensor (`.pt`), one label per
crystal after the resolution filter. Every crystal must be labelled. State names
in text custom-split files are sorted; repeated `--split NAME=PATH` inputs retain
argument order. Separate checkpoint readout accepts any number of bins and uses
their supplied names. References and merge targets are initialized from the
current input data; network weights and feature standardization are restored
from the checkpoint. All supplied states are exported automatically. For changed
inputs or split assignments, supply a new `--cache` path.

In text split files, repeat a `filename event` pair with different dataset names
to include the frame (and all its crystals) in multiple bins. Repeated identical
rows count only once per bin.

## Products and difference maps

Both commands use one writer. Each state produces the raw posterior amplitude
and its own uncertainty (F/SIGF), plus two disjoint crystal-half products:
`merged_dark.mtz`, `merged_dark_half1.mtz`, `merged_dark_half2.mtz`, and the
corresponding light files. The `.metrics.json` sidecar records internal CC1/2
and R_split. These halves share the trained network; they are not independent
network runs. There is no posterior-to-pooled correction or companion output.

For difference maps, use each merge's own uncertainty and TorchRef's CLIs:

```bash
torchref.difference-map -dm dark.pdb \
  -dsf merged_dark.mtz -lsf merged_light.mtz -o difference.mtz
torchref.mtz2map -sf difference.mtz -csf dF -cw W_Q -cphi PHDELWT -o difference.ccp4
```

`dF` is the observed light-minus-dark amplitude difference on the dark model's
phases, and `W_Q` its per-reflection weight. A light model (`-lm`) with
`--fraction` adds the extrapolated map. Compare signed differences and maps
between two independent jointly trained light/dark runs, with disjoint pattern
sets; do not impose an extra reserved-HKL subset. Use TorchSX amplitude/sigma
MTZs and Partialator intensity/sigma MTZs, letting TorchRef perform its own
intensity-to-amplitude conversion.

## Raw-distribution figures

```bash
torchsx.observation-stats indexed.stream --spacegroup 'P 63' -o figures/
```

Writes **PNG only**: raw intensities, reflection skewness, normalized mean–median
gaps and individual reflection examples. The default samples 50K crystals in
complete images; use `--sample-crystals all` for the full dataset. The cell comes
from the input. Named inputs and CrystFEL custom splits are supported.

These figures describe the observations that motivate scaling and likelihood
modeling. They complement the independent precision and accuracy comparisons.
[Plot interpretation, CLI controls and cache replay](https://github.com/HatPdotS/TorchSX/blob/master/docs/stats.md).

## Merge statistics

```bash
torchsx.merge-stats A/merged_dark.mtz B/merged_dark.mtz --name dark \
  --dataset light A/merged_light.mtz B/merged_light.mtz -o stats/
```

Scores merged MTZ, CrystFEL `.hkl` and mmCIF files. A dataset is one merge or two
independent halves:

- **One merge:** completeness, ⟨I/σ(I)⟩, multiplicity and the Wilson falloff.
- **Two halves:** CC, R_split, RMS z and ln N/S after scaling B onto A.
- **Differences:** every dataset is scored against the reference (default: the first
  dataset), as per-shell CC and ln N/S of ΔF between the two arms, plus information in
  bits over resolution × σ(ΔF) bins.

Writes CSV and PNG by default (`--format csv json plots`).
[Definitions, inputs and outputs](https://github.com/HatPdotS/TorchSX/blob/master/docs/merge_stats.md).

## Datasource API

```python
from torchsx import SFXStream, SourceInput

stream = SFXStream.read('indexed.stream', spacegroup='P 63')
print(stream.measurement)
probixi = SourceInput('probixi', {'database': 'run.duckdb'})
```

CrystFEL is implemented through the measurement adapter. The Probixi
measurement adapter remains a reserved seam; the existing `SFXStream.from_duckdb`
reader is available separately. Filtering and storage retain source identities
and measurement provenance.

## Documentation

| Document | Contents |
|---|---|
| [docs/cli.md](https://github.com/HatPdotS/TorchSX/blob/master/docs/cli.md) | Measurement policy, gain selection, the full public option surface, caching and reproducibility |
| [docs/stats.md](https://github.com/HatPdotS/TorchSX/blob/master/docs/stats.md) | `torchsx.observation-stats` plot interpretation, CLI controls and cache replay |
| [docs/merge_stats.md](https://github.com/HatPdotS/TorchSX/blob/master/docs/merge_stats.md) | `torchsx.merge-stats` inputs, statistic definitions and outputs |

Design notes recording why the model looks the way it does:
[scaling_merging](https://github.com/HatPdotS/TorchSX/blob/master/docs/scaling_merging.md),
[optimize_integration](https://github.com/HatPdotS/TorchSX/blob/master/docs/optimize_integration.md),
[sparse_poisson_loss](https://github.com/HatPdotS/TorchSX/blob/master/docs/sparse_poisson_loss.md) and
[fused_gaussian_mixture](https://github.com/HatPdotS/TorchSX/blob/master/docs/fused_gaussian_mixture.md).

## License

MIT. See [LICENSE](https://github.com/HatPdotS/TorchSX/blob/master/LICENSE).
