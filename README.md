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
```

Crystals can be divided into named **splits** (`--split dark=dark.stream
--split light=1ps.stream`, or `--split-file labels.pt` for one stream): batches
are then split-pure, each split keeps its own merge target, and one set of
files is written per split.  The network never sees the label.

The scatter on the scale is a two-width generalised Gaussian in log space by
default.  `--scatter-family mixture` replaces it with a mixture of such
components whose weights, offsets and side widths come from a small
conditioner on the predicted width, shape and skew; the mixture is centred so
its mean multiplicative factor is one, so `K * I0` is the mean of the
observation by construction (`torchsx/merging/scatter/mixture.py`).  The
family is recorded in the checkpoint and rebuilt by `merge_nn` without flags.

`merge_nn` writes three products per split and per half-set: `<stem>.mtz`,
the **scaled posterior** (F/SIGF) used for refinement and difference maps;
`<stem>_posterior.mtz`, the raw posterior; and `<stem>_pooled.mtz`, the
shape-free pooled intensities (I/SIGI, French-Wilson downstream).  The
posterior is the most precise readout, but the shape likelihood locates a
skewed observation distribution rather than estimating its mean, which is
what refinement compares against; the scaled posterior maps it onto the pooled
scale with one smooth Chebyshev surface in resolution and intensity, fitted on
all merged splits jointly (`torchsx/merging/calibrate.py`, recorded in a
`.calibration.json` sidecar).  `--no-scale-calibration` writes the raw
posterior only.

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
stream = SFXStream.read("indexed.stream", spacegroup="P 43 21 2",
                        cache="indexed.sfx")
print(stream.measurement)  # declared units, integration recipe and provenance
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

## Source measurement contract

CrystFEL ingestion retains raw intensity, background and sigma values. Its
immutable `MeasurementSpec` records the producer/version, integration method,
units and definitions, recorded `--int-radius`, gain declarations, unresolved
fields and their provenance. Both `--option=value` and `--option value` are
read from the recorded command as text. No command in an input file is executed.

The `train` and `merge_nn` commands resolve settings in this order: **explicit flags,
checkpoint recipe, source metadata**. With a recorded ring recipe and gain,
`--int-radius` and `--gain` can be omitted. Space group and resolution cuts
remain user choices. `adu_per_photon = 1` means gain 1; `adu_per_eV` is
multiplied by the mean recorded image energy. Missing or invalid declarations
require an explicit gain or a requested `auto`/`empirical` calibration. Their
existing algorithms are unchanged. Photon conversion happens once, in shared
observation preparation. The resolved numeric gain and its origin are saved
with the preparation recipe and checkpoint.
The existing Probixi reader and explicitly requested legacy `gain="stream"`
recipes retain their geometry-based gain behavior with a warning until their
source semantics are supplied.

One integration recipe is assumed per run, including named splits. Profile
integration, gradient backgrounds and other incompatible semantics are rejected
before training. Unknown integration methods/modifiers require an explicit
historical ring recipe where compatible. The exact integer ring counts are
**nominal aperture counts**, not effective counts after masking or clipping.
This boundary does not introduce a new background-bias correction or calibration
policy; those remain scientific qualification work.

`stream.observed_hkl` retains indices before canonicalization. Source filenames,
events and lattice indices survive filtering, concatenation and `.sfx` storage,
along with the measurement record and Friedel state. Ewald-offset computation
continues to use the existing symmetry-image calculation. Legacy caches without
a measurement record need an explicit or checkpoint recipe; `--trust-cache-prep`
still loads a moved prepared cache without reading the original inputs. An old
`.sfx` without a recorded HKL state cannot establish whether its indices were
already canonicalized; `merge_friedel=False` preserves the historical assumption
with a warning. Re-parsing the source recovers observed indices.

`from_stream`, `from_duckdb` and `load` remain available. `SFXStream.read` accepts
a path or named files through `SourceInput`:

```python
from torchsx import SourceInput
stream = SFXStream.read(SourceInput("crystfel", {"stream": "indexed.stream"}),
                        spacegroup="P 43 21 2")
# Reserved descriptors; their measurement adapters are not implemented yet:
probixi = SourceInput("probixi", {"database": "run.duckdb"})
dials = SourceInput("dials", {"experiments": "run.expt", "reflections": "run.refl"})
```

Probixi continues to work through `from_duckdb`; its next step is to supply the
measurement record. A future DIALS adapter must select and document intensity,
variance, background and upstream-correction semantics before entering shared
preparation. Neither requires a second tensor container or a plugin registry.

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
└── cli/            train, merge_nn
```

### `torchsx.merging`

Organised as the layers of the generative model:

```
ScaleModel × ScatterModel → Quadrature → DetectionLikelihood
```

Each is an `nn.Module` behind a narrow ABC. The latent per-reflection intensity
is integrated out by the quadrature rather than estimated, so what the network
trains against is a marginal evidence (`nn/objective.py`). Subpackages:
`scale/`, `scatter/`, `quadrature/`, `likelihood/`, `priors/`, `hkl/`.

`quadrature/` holds two grids. `SplitLogGrid` places nodes per observation and
carries the two-width and mixture scatter paths; it is what training integrates
against. `FixedLogGrid` shares one wide grid across observations, which is what
the posterior readout wants. Each has a Triton fast path and an eager fallback
that must agree.

`hkl/` is the reflection-bookkeeping layer — HKL indexing, filtering, shell
statistics, and the Poisson noise model whose `calibrate_sigma2_of_bg` derives
detector gain from off-Bragg (predicted-zero-signal) observations via
Ewald-offset gating when `gain="auto"` is requested. The default is `"stream"`
(declared ADU/photon, or photon energy × ADU/eV), with a recorded checkpoint
recipe taking precedence in the CLI. Other explicit choices are `"empirical"`
(negatives-Mandel, biased high) or a numeric gain.

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
