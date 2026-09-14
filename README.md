# TorchSX

TorchSX scales and merges indexed serial crystallography observations with a
shared neural network and a posterior likelihood. It starts from integrated
observations; peak finding and indexing belong to CrystFEL, Probixi or DIALS.

## Install

```bash
pip install -e .
```

Python >=3.10 and TorchRef >=0.6.4 are required. Optional extras are `native`
(for building the stream parser), `duckdb` (the existing Probixi reader) and `dev`.

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

## Measurement and gain

CrystFEL loading retains raw intensities, backgrounds, sigmas and integration
metadata. Ring radii resolve from an explicit `--int-radius`, then the checkpoint,
then the input's recorded integration command. Missing radii need an explicit
value, for example `--int-radius 3,4,7`. Unsupported integration semantics are
rejected. Nominal aperture counts do not account for masked or clipped pixels.

`--gain` accepts:

| Value | Behavior |
|---|---|
| `auto` (default) | Compare input and calibrated gains; replace input when the relative discrepancy exceeds 50%. |
| `trust_input` | Use the datasource declaration without calibration. |
| `calibrate` | Always use a valid calibrated estimate; fail if unavailable. |
| Positive finite number | Explicit ADU/photon gain. |

Auto retains a valid input gain when calibration is unavailable and uses valid
calibration when input gain is missing. It fails when neither is usable. Invalid
arguments report the received value and accepted choices. The resolved gain,
selection reason and calibration diagnostics are recorded; checkpoint readout
reuses the selected numeric gain. The legacy `stream` spelling remains a Python
recipe alias for `trust_input`.

New training uses the annulus background-mean correction inferred from integration
metadata and a four-component Gaussian mixture. Gain conversion, background-mean
correction and posterior uncertainty are distinct operations. Existing checkpoints
retain their recorded scatter family and background convention.

## Public controls and reproducibility

The shared options are `--out`, `--split`, `--split-file`, `--spacegroup`,
`--dmin`, `--dmax`, `--gain`, `--int-radius`, `--no-friedel`, `--free-from`,
`--device`, `--cache` and `--seed`. Training adds `--steps`, `--redundancy` and
`--no-merge`. Symmetry and Friedel behavior are unchanged by this CLI cleanup.

Use `--free-from existing.mtz` to preserve a refinement free set. Otherwise the
existing default 5% free-flag export policy applies. Refinement free flags are
export annotations, distinct from validation crystals and the diagnostic halves;
no reflections are removed from the difference calculation by these flags.

`--seed` controls initialization and batch sampling, with a derived diagnostic
half seed. Merge reuses the saved run seed unless overridden. Refinement free-flag
assignment retains its separate fixed seed, so changing the training seed does
not deliberately change the refinement comparison.

`--cache` names a prepared cache (default `RUN/prepared.pt`; merge uses the
checkpoint directory). Input contents, labels, measurement policy and preparation
settings are validated. Identical files may move without invalidating the cache;
source hashes cost one sequential input read. Inputs must remain available for
validation. Changed data/settings and older cache schemas require a new cache
path. There is no bypass flag. Original input paths remain recorded as provenance.

Model, likelihood and quadrature settings are internal and recorded in each run.
Training uses the 31/129/6 outer grid; readout uses 63/257/24, with inner 192 and
locating windows 8/1/0.125. The resolved recipe records learning rates, per-state
warmup/decay, resource policy and evaluation cadence. Research variants use the
Python API rather than another public configuration language.

Historical experiment scripts using removed flags or the retired correction need
their original source revision. Checkpoints with recorded inference context remain
readable; older checkpoints without it need the original versioned recipe or
retraining. This cleanup does not assert new accuracy or data-efficiency results.

## Datasource API

```python
from torchsx import SFXStream, SourceInput

stream = SFXStream.read('indexed.stream', spacegroup='P 63')
print(stream.measurement)
probixi = SourceInput('probixi', {'database': 'run.duckdb'})
dials = SourceInput('dials', {'experiments': 'run.expt', 'reflections': 'run.refl'})
```

CrystFEL is implemented through the measurement adapter. Probixi and DIALS
measurement adapters remain reserved seams; the existing `SFXStream.from_duckdb`
reader is available separately. Filtering and storage retain source identities
and measurement provenance.
