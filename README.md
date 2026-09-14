<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/torchsx-banner-dark.png">
    <img src="assets/torchsx-banner-light.png" alt="TorchSX" width="100%">
  </picture>
</p>

<h1 align="center">A Py<ins>Torch</ins>-based scaling and merging library for serial crystallography (<ins>SX</ins>)</h1>

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![TorchRef](https://img.shields.io/badge/TorchRef-0.6.4+-f97316.svg)](https://github.com/HatPdotS/TorchRef)
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
  PyTorch workflow, maybe at some point we differentiate the model against all the data? 

## Quickstart

```bash
pip install -e .

torchsx-train indexed.stream --spacegroup 'P 63' --dmin 2.0 --steps 800 -o run/
```

That writes checkpoints and merged MTZs into `run/`. Everything below expands on
it: separate readout of a checkpoint, joint light/dark states, difference maps,
and the observation statistics that motivate the model.

## Install

Python >=3.10 and TorchRef >=0.6.4 are required; TorchRef brings torch, triton
and gemmi. Optional extras are `native` (for building the stream parser),
`duckdb` (the existing Probixi reader), `stats` (Matplotlib for distribution
plots) and `dev`, installed as `pip install -e '.[stats,native]'`.

## Train and merge

```bash
torchsx-train indexed.stream --spacegroup 'P 63' --dmin 2.0 --steps 800 -o run/
torchsx-merge-nn run/best.pt indexed.stream -o merged.mtz
```

Training writes `best.pt`, `last.pt`, `run.json`, `metrics.jsonl`,
`training_summary.json` and full/half merged MTZs. `--no-merge` stops after saving
checkpoints, for workflows that schedule readout separately.

`--steps` is the total optimizer-update budget, independent of dataset size.
Training alternates state-pure batches: 800 updates over two states gives 400 per
state. An uneven budget differs by at most one update between states. Training
length remains empirical. `best.pt` selects minimum validation NLL in state zero;
`last.pt` preserves the endpoint. Validation NLL is not established as an optimum
for independent difference-map precision. Validation uses 5% of state-zero
crystals; all supplied observations remain eligible for the final merge.

The statistical batch size is controlled by `--redundancy` (default 25 target
observations per reflection). A small state uses all its available training
crystals. Data stay on the host; encoder chunking and quadrature recomputation
bound GPU memory without reducing the requested statistical batch support.
Parser workers respect the available CPU allocation, capped at eight.

Measurement policy, gain selection and the full public option surface —
`--gain`, `--int-radius`, `--free-from`, `--cache`, `--seed` and the rest — are
in the [command-line reference](docs/cli.md).

## Joint light/dark inputs

Train both states in one network; the merge targets remain separate. The label
is not a network feature.

```bash
torchsx-train --split dark=dark.stream --split light=light.stream \
  --spacegroup 'P 63' --dmin 2.0 --steps 800 -o joint/
torchsx-merge-nn joint/best.pt \
  --split dark=dark.stream --split light=light.stream -o merged.mtz
```

A combined stream can instead use a CrystFEL custom-split file with
`filename event dataset` columns:

```bash
torchsx-train combined.stream --split-file states.split \
  --spacegroup 'P 63' --dmin 2.0 --steps 800 -o joint/
```

`--split-file` also accepts an integer label tensor (`.pt`), one label per
crystal after the resolution filter. Every crystal must be labelled. State names
in text custom-split files are sorted; repeated `--split NAME=PATH` inputs retain
argument order. Supply the same states in the same order for checkpoint readout.
All supplied states are exported automatically.

## Products and difference maps

Both commands use one writer. Each state produces the raw posterior amplitude
and its own uncertainty (F/SIGF), plus two disjoint crystal-half products:
`merged_dark.mtz`, `merged_dark_half1.mtz`, `merged_dark_half2.mtz`, and the
corresponding light files. The `.metrics.json` sidecar records internal CC1/2
and R_split. These halves share the trained network; they are not independent
network runs. There is no posterior-to-pooled correction or companion output.

For difference maps, use each merge's own uncertainty and the native TorchRef CLI:

```bash
torchref.phased-difference-map -dm reference.pdb -lm reference.pdb \
  -dsf merged_dark.mtz -lsf merged_light.mtz --fraction 0.5 --dmin 2.0 \
  -o difference.mtz
torchref.mtz2map -sf difference.mtz -csf WDF -cphi PHIC_dark -o difference.ccp4
```

With the same unchanged reference model, WDF contains the observed weighted
light-minus-dark difference. The fraction controls additional extrapolated
columns. Compare signed WDF and maps between two independent jointly trained
light/dark runs, with disjoint pattern sets; do not impose an extra reserved-HKL
subset. Use TorchSX amplitude/sigma MTZs and Partialator intensity/sigma MTZs,
letting TorchRef perform its own intensity-to-amplitude conversion.

## Raw-distribution figures

```bash
pip install -e '.[stats]'
torchsx-stats indexed.stream --spacegroup 'P 63' -o figures/
```

Writes **PNG only**: raw intensities, reflection skewness, normalized mean–median
gaps and individual reflection examples. The default samples 50K crystals in
complete images; use `--sample-crystals all` for the full dataset. The cell comes
from the input. Named inputs and CrystFEL custom splits are supported.

These figures describe the observations that motivate scaling and likelihood
modeling. They complement the independent precision and accuracy comparisons.
[Plot interpretation, CLI controls and cache replay](docs/stats.md).

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
| [docs/cli.md](docs/cli.md) | Measurement policy, gain selection, the full public option surface, caching and reproducibility |
| [docs/stats.md](docs/stats.md) | `torchsx-stats` plot interpretation, CLI controls and cache replay |

Design notes recording why the model looks the way it does:
[scaling_merging](docs/scaling_merging.md),
[optimize_integration](docs/optimize_integration.md),
[sparse_poisson_loss](docs/sparse_poisson_loss.md),
[fused_gaussian_mixture](docs/fused_gaussian_mixture.md) and
[crystfel_ingestion_validation](docs/crystfel_ingestion_validation.md). The
remaining files in `docs/` are working plans and drafts, kept as a record of the
decisions rather than as current interface documentation.

## License

MIT. See [LICENSE](LICENSE).
