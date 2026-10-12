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
scale each crystal with a few parameters and average. TorchSX instead works
from a generative model of every observation. The signal that reached the
detector aperture of observation *j* of reflection *h* is

    S_j = K_j · I_h · exp(ε_j)

where I_h is the reflection's true intensity, a latent variable with a Wilson
prior; K_j is a scale predicted for this observation; and ε_j is a random
multiplicative scatter, with E[exp(ε_j)] = 1, whose distribution is also
predicted for this observation (a mixture of two-width Gaussians in log space).
The detector then records Poisson counts of S_j on top of the background. One
neural network per dataset predicts K_j and the scatter distribution from the
observation's geometry, its crystal's metadata and the crystal's other
observations. Merging is a posterior integral rather than an average: each
merged amplitude and its uncertainty are moments of the posterior of I_h given
all its observations. The network is trained on random batches of crystals
sized to a given multiplicity, maximising the marginal likelihood of their
observations with I_h integrated out.

TorchSX starts from integrated CrystFEL `.stream` files and writes merged MTZs.
We are working on integration with dials and integration of Probixi. 

TorchSX is under active development, if you find something behaving unexpectedly please create an issue on GitHub. 

Refinement and difference maps in this repo were done with
[TorchRef](https://github.com/HatPdotS/TorchRef). TorchSX output should be compatible with any classical refinement or map calculation program. 

## Install

```bash
pip install torchsx
```

Python ≥ 3.10. This brings TorchRef 0.7, torch, triton, gemmi and
`torchsx-native`, the compiled `.stream` reader for your platform. TorchSX is intended to be run on a GPU; its core algorithms are Triton kernels, and we have not investigated CPU runtime or even functionality. GPU memory should not exceed 10 GB. It scales with the redundancy requested for training (default 100) and the number of unique HKLs, not with the amount of data. Host RAM does scale with the amount of data, at roughly half the size of the CrystFEL stream; streaming from disk is being worked on but is not implemented yet.

From a checkout, build the native package first (needs Rust, see
[native/README.md](https://github.com/HatPdotS/TorchSX/blob/master/native/README.md)):
`pip install ./native`, then `pip install -e .`.

## Usage

```bash
torchsx.train indexed.stream --spacegroup 'P 63' --dmin 2.0 -o run/
```

This trains the network (400 updates by default) and merges the data with it.
`run/` then contains:

```
run/
├── merged.mtz              merged amplitudes from all crystals
├── merged_half1.mtz        the same from one half of the crystals
├── merged_half2.mtz        ... and from the other half
├── merged.metrics.json     in-run precision: CC1/2 and R_split between the two halves
├── best.pt                 the network at the lowest validation loss; it made the merges
├── last.pt                 the network after the last update
├── prepared.pt             the parsed observations, reused by later runs on the same input
├── run.json                the resolved settings
├── training_summary.json   updates, best step and run time
└── metrics.jsonl           training log: one line per update and per validation
```

Each MTZ holds:

| Column | MTZ type | Content |
|---|---|---|
| `H`, `K`, `L` | H | Miller index; Friedel mates merged |
| `F-obs` | F | posterior mean of the amplitude \|F\| |
| `SIGF-obs` | Q | posterior standard deviation of the amplitude |
| `R-free-flags` | I | refinement free set: 0 = free (5 %), 1 = work |

There are no intensity columns. `F-obs` is already the posterior estimate of
the amplitude, so it must not go through French–Wilson again; refinement
programs use it as it is. With `--no-merge-friedel` the mates stay apart and the
files carry `F-obs(+)`, `SIGF-obs(+)`, `F-obs(-)` and `SIGF-obs(-)`.
`--free-from earlier.mtz` takes the free set from an existing file.

The two halves split the crystals, not the training: both are merged with the
same network, so the CC1/2 and R_split in `merged.metrics.json` measure the
in-run precision, not the agreement of independent merges. For independent
halves, as in the comparison below, train two runs on disjoint sets of
patterns.

### Light and dark

```bash
torchsx.train --split dark=dark.stream --split light=light.stream \
  --spacegroup 'P 63' --dmin 2.0 -o run/
```

One network is trained on both states, and each state is merged on its own:
`merged_dark.mtz` and `merged_light.mtz`, each with its two halves. A single
stream with a CrystFEL custom-split file works too (`--split-file`).

## How it works, by analogy

Every crystal is a witness that saw each reflection once, under its own
conditions: how much of the reflection the Ewald sphere caught, how bright the
pulse was, how big the crystal was (partiality, incident intensity and crystal
volume, none of them known per observation). Established programs correct each
witness with a few parameters per crystal and then average the testimonies
(per-crystal scale and B factor, then a weighted mean).

TorchSX instead learns, from the whole data set, how reliable each kind of
testimony is. One network reads every observation together with the other
spots on the same image and predicts two things: how bright this observation
should be, and how far from that it may plausibly fall (a scale factor K and a
scatter distribution, a mixture of two-width Gaussians in log space, for every
observation).

Merging then does not average. It asks which true intensities are consistent
with all the testimonies given their predicted reliabilities, and reports the
centre of that range as the merged value and its width as the uncertainty (the
posterior of each reflection's intensity under the product of per-observation
likelihoods and a Wilson prior.).

The network is trained the way a weather forecaster is scored: rewarded for
confidence where the data agree with it, penalised for confidence where they do
not, so the predicted scatter stays honest (maximising the marginal
likelihood, with each reflection's true intensity integrated out).

## Compared with Partialator

On the bacteriorhodopsin (bR) data of Weinert et al.<sup>[1](#references)</sup>:

<p align="center">
  <img src="https://raw.githubusercontent.com/HatPdotS/TorchSX/master/assets/readme-br-rfree.png" alt="bR R-free against dark patterns per merge, TorchSX and Partialator" width="65%">
</p>

R<sub>free</sub> of models refined from one starting model against each dark
merge, on the reflections common to both programs at each size, as a function
of the number of dark patterns per merge. Points, mean of two independent
merges; bars, their range. Initial models were refined against a larger merge
with the respective method. Refinement used TorchRef with one protocol for all
merges.

<p align="center">
  <img src="https://raw.githubusercontent.com/HatPdotS/TorchSX/master/assets/readme-br-maps.png" alt="bR light-minus-dark difference maps from two independent halves, TorchSX and Partialator" width="100%">
</p>

Light-minus-dark difference maps from two independent merges of disjoint
patterns (7,800 per state each), phased with the dark model. Shown: the end of helix E, the EF loop and helix F of the dark model
(grey) and of the open state 6RPH<sup>[1](#references)</sup> (orange). The
number under each program is the correlation of its two maps over helix F.

## Other commands

- `torchsx.merge run/best.pt more.stream -o merged.mtz` merges new or additional
  data with a trained network.
- `torchsx.merge-stats A/merged.mtz B/merged.mtz -o stats/` compares merges:
  completeness, CC1/2, R_split and, for light/dark pairs, the information in
  ΔF ([docs/merge_stats.md](https://github.com/HatPdotS/TorchSX/blob/master/docs/merge_stats.md)).
- `torchsx.observation-stats indexed.stream --spacegroup 'P 63' -o figures/`
  plots the raw observations: intensity distributions, skewness, examples
  ([docs/stats.md](https://github.com/HatPdotS/TorchSX/blob/master/docs/stats.md)).

All options, gain and measurement settings, caching and reproducibility are in
the [command-line reference](https://github.com/HatPdotS/TorchSX/blob/master/docs/cli.md).

In Python, `torchsx.SFXStream.read('indexed.stream', spacegroup='P 63')` reads a
stream with its geometry and measurement settings.

## References

1. Weinert, T. et al. Proton uptake mechanism in bacteriorhodopsin captured by
   serial synchrotron crystallography. *Science* **365**, 61–65 (2019).
   [doi:10.1126/science.aaw8634](https://doi.org/10.1126/science.aaw8634)
2. Seidel, H.-P., Standfuss, J. & Weinert, T. TorchRef: an open-source PyTorch
   framework for crystallographic refinement. *J. Appl. Cryst.*, accepted (2026).

## License

MIT. See [LICENSE](https://github.com/HatPdotS/TorchSX/blob/master/LICENSE).
