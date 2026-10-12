# Command-line reference

Measurement policy, gain selection, the public option surface and the
reproducibility rules for `torchsx.train` and `torchsx.merge`. The README
covers the common path; this document is the lookup reference.

## Measurement and gain

CrystFEL loading retains raw intensities, backgrounds, sigmas and integration
metadata. Ring radii resolve from an explicit `--int-radius`, then the checkpoint,
then the input's recorded integration command. Omitted CrystFEL options inherit
its defaults ([indexamajig manual](https://www.desy.de/~twhite/crystfel/manual-indexamajig.html)):
ring integration without centering, saturation or gradient options, and radii
`4,5,7` if not declared. Defaults are recorded in measurement provenance;
explicit declarations are preserved. Unsupported integration semantics are
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

Calibration uses background-binned off-Bragg variance when the input predicts
reflections far from the Ewald sphere. CrystFEL streams do not, so calibration
then falls back to a truncated normal fitted to the negative integrated
intensities, cut to the core of their distribution (above -3 robust standard
deviations, the cut modelled in the fit) against over-subtracted outliers. That
estimate is biased, so `auto` replaces a declared gain only past the 50%
tolerance. If neither
estimator is usable, `auto` retains a valid declared gain, while `calibrate`
fails. New `auto`/`calibrate` requests require prepared caches built with the
current gain policy; checkpoint readout still reuses its recorded numeric gain.

`--detector-noise auto` refines a gain the calibration selected (by `auto` past
the tolerance, or `calibrate`) in training by one global B, the counting variance
B (lam + sig_b2), which is exactly a gain correction on the integrated sums; with
the input gain kept, no detector term is fitted. `--detector-noise panel` fits
A and/or B per detector panel (`--noise-terms`). The default, `off`, fits none.
Merges inherit the checkpoint's terms.

New training uses the annulus background-mean correction inferred from integration
metadata and a four-component Gaussian mixture. Gain conversion, background-mean
correction and posterior uncertainty are distinct operations. Existing checkpoints
retain their recorded scatter family and background convention.

## Training length and checkpoints

`--steps` is the total optimizer-update budget, independent of dataset size
(default 400). Training alternates state-pure batches: 400 updates over two
states gives 200 per state; an uneven budget differs by at most one update
between states. `best.pt` is the network at minimum validation NLL in state
zero, and the merges written by `torchsx.train` use it; `last.pt` preserves the
endpoint. Validation uses 5% of state-zero crystals; all supplied observations
remain eligible for the final merge. Training length remains an empirical
setting: validation NLL is not established as an optimum for independent
difference-map precision. `--no-merge` stops after saving checkpoints, for
workflows that merge separately with `torchsx.merge`.

Data stay on the host; encoder chunking and quadrature recomputation bound GPU
memory without reducing the requested statistical batch support. Parser workers
respect the available CPU allocation, capped at eight.

## States

Light and dark (or any set of labelled states) are trained in one network; the
merge targets remain separate, and the label is not a network feature.

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
argument order. In text split files, repeat a `filename event` pair with
different dataset names to include the frame (and all its crystals) in multiple
bins; repeated identical rows count only once per bin.

Separate checkpoint readout accepts any number of bins and uses their supplied
names. References and merge targets are initialized from the current input
data; network weights and feature standardization are restored from the
checkpoint. All supplied states are exported automatically. For changed inputs
or split assignments, supply a new `--cache` path.

## Public controls and reproducibility

The shared options are `--out`, `--split`, `--split-file`, `--spacegroup`,
`--dmin`, `--dmax`, `--gain`, `--int-radius`, `--merge-friedel`/`--no-merge-friedel`, `--free-from`,
`--device`, `--cache`, `--seed` and `--prior`. Training adds `--steps` (default
400), `--redundancy`, `--no-merge`, `--miss-frac`, `--outlier-frac`,
`--detector-noise`, `--noise-terms`, `--peer-attention`, `--warmup` and
`--lr-schedule`. The defaults are the production configuration. A few research
options (encoder geometry, evaluation cadence, early stopping, an external
held-out set) are accepted but not listed by `--help`; see
`torchsx/cli/train.py`. Friedel mates are merged by default; checkpoint readout inherits
the recorded choice unless overridden. `--no-merge-friedel` keeps acentric mates
separate and writes (+)/(−) columns. Bijvoet mates share refinement free flags.

`--redundancy` sets the target observations per reflection in a training batch
(default 100).
If a state's available training data is smaller than the requested batch,
training warns once for that state and uses all its training crystals in one
batch. Validation crystals remain held out.

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

A separate merge takes the checkpoint's network
weights, fitted feature standardization and Wilson surface, and forms everything
per reflection from the supplied observations exactly as training does, so a
merge of the training data reproduces the training run's own readout. Any number
of bins can be exported under their current labels.

### Scale encoder

Each observation's scale is predicted from its own features and those of the
other observations of its crystal. `--peer-attention N` sets the number of
cross-attention layers onto these peers (default 2; `0` pools by mean only).
Each crystal is split into two folds, and an observation takes its context from
the other fold. Checkpoint readout inherits both.

`--miss-frac F` adds a learnable miss component, a predicted reflection that
recorded no signal, starting at fraction F (default 0.03; `0` turns it off).
`--outlier-frac F` adds a learnable detector-outlier component, starting at
fraction F (default 0, off). Both fractions lie in [0, 1); readout inherits
them.

### Training schedule

The learning rate warms up over `--warmup` updates per state (default 25) and
then stays constant. `--lr-schedule cosine` decays it to zero at the last
update instead.

### No per-reflection running state

Nothing is carried from one batch to the next except the network and its
fitted parameters. Each batch places its own integration grid, around its
weighted mean at the network's scale (floored 4 nats below the Wilson
surface); a merge starts from the unity merge of its inputs with the same
floor and refines in three windows.

### Peer features

The context of an observation carries, for every peer, its observed scale as
its intensity over the plain (unweighted) mean of its reflection's
observations, and the log of that mean. The mean is computed once from the
input observations, all states pooled.

### Wilson surface

With `--prior wilson` each state's prior mean `Σ(h)` is a smooth surface
`log Σ(h) = φ(h)·c` on a reciprocal-space basis (Chebyshev polynomials in
resolution plus the space group's anisotropy terms). The coefficients start at
the Wilson fit to the unity merge and are trained with the network by the same
marginal likelihood (type-II maximum likelihood), at ten times the base learning
rate. The checkpoint stores the basis definition and each state's
coefficients, so a separate merge evaluates the same surface on any reflection
set.

### Intensity prior

`torchsx.train ... --prior wilson` is the default. `--prior none` removes the
Wilson intensity density and integrates the observation likelihood with respect
to positive intensity (`dI`, including the Jacobian on the log-intensity grid).
It does not select a maximum-likelihood point estimate or a flat-log prior.
Training's integrated likelihood, validation, and full/half posterior readouts all
use the selected measure. I, F, their uncertainties and integration-edge
statistics come from those same weights. The scale anchor remains enabled.

The choice is saved in `posterior_grid.prior` and the resolved training recipe.
`torchsx.merge best.pt ...` inherits it; an explicit `--prior none` or
`--prior wilson` overrides it for a readout experiment without retraining.
The resolved prior is printed and saved for each state in `.metrics.json`.
Historical `flat_I0` checkpoints are read as `none`; historical `flat_log`
checkpoints retain their original measure. Neither legacy spelling is a CLI option.

`none` uses the finite numerical integration range. In particular, the existing
missed-spot likelihood has an intensity-independent component, so a uniform prior
over an unbounded positive intensity axis would not yield a proper posterior.
The resulting integrated-likelihood score is not normalized Bayesian evidence;
its scale and the weak-reflection moments can depend on the grid bounds. Inspect
edge mass and range sensitivity before interpreting results. Wilson and `none`
validation scores should not be compared as interchangeable model evidences.
