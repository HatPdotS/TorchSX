# TorchSX

GPU-friendly stacked tensors for unmerged serial crystallography data.

TorchSX merges and scales Serial Femtosecond Crystallography (SFX) data. It
reads indexed reflections into stacked tensors at three levels (observations,
crystals, images) and provides differentiable merging, scaling and partiality
correction on top of them.

It does **not** do data reduction. Pixels, peak finding and indexing belong to
[probixi](https://github.com/ryan-odea/probixi) or CrystFEL; TorchSX starts
where observations of the same reflection first meet.

## Amortised scale model

`torchsx.nn` trains one network to predict a per-observation scale, merging
through a posterior integral over `I0` rather than a point estimate, so `I` and
`sigma_I` come from the same integral. See `docs/amortized_scale_plan.md` for
the design and `docs/optimize_integration.md` for the measured state of the
integration — the node settings that should be used, the background likelihood,
and the measurement traps that produced two wrong conclusions along the way.

Validated against refinement rather than only CC1/2, since a merge can be
systematically wrong and perfectly self-consistent — and on this data it
repeatedly is. **R_free 0.1269 (`nll`) / 0.1252 (`ml`)** against 0.1825 for
partialator + French–Wilson on the same model, cell and free set, with the
weak-end compression slope at 1.02 / 0.95 where the uncompensated merge gives
0.16 / 0.18.

The likelihood is exactly invariant under `K(h) -> K(h) g(h)`, `I0 -> I0 / g(h)`
for any smooth function `g` of reciprocal space, and the encoder sees resolution,
so without a pin the merged fall-off random-walks along that direction as
training proceeds (measured on BR: a bend of 1.6 log units against partialator
after ~900 steps, costing 0.01-0.03 in R_free). Two things now hold it. The
Wilson prior's `Sigma(h)` is a smooth symmetry-adapted surface (Chebyshev in
`s^2` plus the point-group-invariant anisotropy tensor, fitted by Wilson maximum
likelihood; `--wilson-basis smooth`, the default) instead of resolution shells,
and the scale model projects out the smooth common mode of `log K` in the same
basis after fitting a per-crystal constant, so genuine per-crystal scale survives
(`--gauge-anchor basis`, the default). `--wilson-basis shells --gauge-anchor
median` recovers the previous behaviour; checkpoints written before these
options existed are rebuilt as they were trained.

That the two must be checked separately is not a formality here: the single best
CC1/2 of any run trained (0.9982) came from a configuration whose merged
intensities are **not monotone** in `F_model` and whose R_free is the worst of
the trained set.

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
pip install -e .              # merging + scaling
pip install -e '.[duckdb]'    # + read probixi runs
```

Requires Python >= 3.10 and `torchref >= 0.6.4`, which provides torch, triton
and gemmi. The remaining extras are optional because each is imported lazily:

| Extra | Enables |
|---|---|
| `duckdb` | `SFXStream.from_duckdb` — read probixi runs |
| `patches` | `excise_patches` / `excise_rois` — pixel patches from HDF5 |
| `gating` | `merging/gating`'s background-significance split |
| `native` | JIT-building the native `.stream` parser extension |

## Quick Start

Two ways in, depending on who indexed the data.

**From probixi** (a DuckDB run — one row per frame, reflections keyed by frame):

```python
from torchsx import SFXStream

stream = SFXStream.from_duckdb("run.duckdb", spacegroup="P 43 21 2")
```

**From CrystFEL** (an `indexamajig` `.stream`):

```python
stream = SFXStream.from_stream("indexed.stream", spacegroup="P 43 21 2",
                               cache="indexed.sfx")
```

Then merge, the same either way:

```python
stream = stream.filter_resolution(d_min=2.0).filter_cell(tolerance=0.1)

loss = stream.make_loss()          # log-space merging loss
loss.refine()                      # LBFGS
result = loss.posterior()          # I, sigma_I from one posterior integral
```

Filtering is immutable — every operation returns a new object, with the join
keys renumbered onto the surviving rows.

## Package structure

```
torchsx/
├── stream.py       SFXStream — public API (load, filter, merge, save)
├── _core.py        the three-level stacked-tensor container
├── levels.py       Level / ColumnSpec — the columnar primitive
├── geometry.py     detector geometry (CrystFEL .geom)
│
├── merging/        the merge engine (see below)
├── nn/             amortised per-observation scale network
├── io/             .stream parsing, .sfx safetensors, DuckDB ingest
└── cli/            merge, filter_ewald
```

### `torchsx.merging`

Organised as the layers of the generative model:

```
IntensityPrior → ScaleModel × ScatterModel → Quadrature → DetectionLikelihood
```

Each is an `nn.Module` behind a narrow ABC, aggregated by a single `LossState`
into one scalar for LBFGS. Subpackages: `priors/`, `scale/`, `scatter/`,
`quadrature/`, `likelihood/`, `targets/`, `regularizers/`, `hkl/`, `gating/`.

`hkl/` is the reflection-bookkeeping layer — HKL indexing, filtering, shell
statistics, and the Poisson noise model whose `calibrate_sigma2_of_bg` derives
detector gain from off-Bragg (predicted-zero-signal) observations via
Ewald-offset gating. This is the **default** (`gain="auto"`); the alternatives
are `"stream"` (photon energy × ADU/eV), `"empirical"` (negatives-Mandel, biased
high) or an explicit float.

### `torchsx.io`

Three formats, three jobs:

| Format | Role | Entry point |
|---|---|---|
| DuckDB | ingest from probixi | `SFXStream.from_duckdb` |
| `.stream` | ingest from CrystFEL | `SFXStream.from_stream` |
| `.sfx` | working cache (safetensors + JSON sidecar) | `SFXStream.load` / `.save` |

DuckDB is the interchange format and `.sfx` the working one, because probixi's
schema is fixed DDL while `Level`/`ColumnSpec` takes arbitrary columns —
timepoints, split labels, derived caches. Load once from the database, then live
in `.sfx`.

The two key/unit traps in that conversion are documented in `io/_duckdb.py`.

## Data model

| Level | Count | Key fields |
|---|---|---|
| Observations | N_obs | hkl, I, sigma, background, peak, fs, ss, panel_id, crystal_id |
| Crystals | N_crystals | astar, bstar, cstar, cell, profile_rad, resolution, n_refl, image_id |
| Images | N_images | beam_energy, det_shift_x, det_shift_y, n_peaks, camera_length |

Join keys: `crystal_id` (obs → crystal), `image_id` (crystal → image). These are
**dense row positions**, not identities — they are rewritten on every filter so
gather/scatter stays a plain index op. A probixi `frame_id` is preserved
separately in `metadata["crystal"]["frame_id"]`, since that is the key any
write-back must use.

## Tests

```bash
python -m pytest tests/ -m "not slow" -v      # no data files needed
python -m pytest tests/ -v                    # full suite (needs merged_368.stream)
```

Note `tests/test_consistency_loss.py::test_recovers_scale_width` is very slow
and not yet marked `slow`; deselect it for a quick run.

## Docs

`docs/` holds the design and measurement record: `amortized_scale_plan.md`
(design), `optimize_integration.md` (measured integration state),
`specsheet.md`, `scaling_merging.md`, `fully_integrated_model.md`,
`sparse_poisson_loss.md`, `implementation_plan_stream_class.md`.
