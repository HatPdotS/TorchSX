# Command-line reference

Measurement policy, gain selection, the public option surface and the
reproducibility rules for `torchsx-train` and `torchsx-merge-nn`. The README
covers the common path; this document is the lookup reference.

## Measurement and gain

CrystFEL loading retains raw intensities, backgrounds, sigmas and integration
metadata. Ring radii resolve from an explicit `--int-radius`, then the checkpoint,
then the input's recorded integration command. Omitted CrystFEL options inherit
its defaults: ring integration without centering, saturation or gradient options,
and radii `4,5,7` if not declared. Defaults are recorded in measurement provenance;
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

## Public controls and reproducibility

The shared options are `--out`, `--split`, `--split-file`, `--spacegroup`,
`--dmin`, `--dmax`, `--gain`, `--int-radius`, `--merge-friedel`/`--no-merge-friedel`, `--free-from`,
`--device`, `--cache` and `--seed`. Training adds `--steps`, `--redundancy` and
`--no-merge`. Friedel mates are merged by default; checkpoint readout inherits
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

Historical experiment scripts using removed flags or the retired correction need
their original source revision. A separate merge takes the checkpoint's network
weights, fitted feature standardization and Wilson surface, and forms everything
per reflection from the supplied observations exactly as training does, so a
merge of the training data reproduces the training run's own readout. Any number
of bins can be exported under their current labels.

### Scale encoder

Each observation's scale is predicted from its own features and those of the
other observations of its crystal. `--peer-attention N` sets the number of
cross-attention layers onto these peers (default 2; `0` pools by mean only).
`--peer-dim` sets the width inside each layer, by default the encoder width; a
comma-separated list gives one width per layer, for example `128,128,32,32`.
`--context-folds K` splits each crystal into K folds and takes an observation's
context from the other K−1 (default 2); `0` takes it from all other reflections
of the crystal. Checkpoint readout inherits all three.

`--outlier-frac F` adds a learnable detector-outlier component to the
likelihood, starting at fraction F (default 0, off). Readout inherits it.

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

`torchsx-train ... --prior wilson` is the default. `--prior none` removes the
Wilson intensity density and integrates the observation likelihood with respect
to positive intensity (`dI`, including the Jacobian on the log-intensity grid).
It does not select a maximum-likelihood point estimate or a flat-log prior.
Training's integrated likelihood, validation, and full/half posterior readouts all
use the selected measure. I, F, their uncertainties and integration-edge
statistics come from those same weights. The scale anchor remains enabled.

The choice is saved in `posterior_grid.prior` and the resolved training recipe.
`torchsx-merge-nn best.pt ...` inherits it; an explicit `--prior none` or
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
