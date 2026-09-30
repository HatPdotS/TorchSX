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
their original source revision. Separate merge initializes its intensity reference
and merge targets from a unity merge of the supplied observations, retaining the
checkpoint's network weights and fitted feature standardization. Any number of
bins can be exported under their current labels. Exact checkpoint replay through
the Python API still requires recorded inference context and matching data.

### Running intensity reference

`torchsx-train --reference-estimator posterior --reference-ema-decay 0.99`
reuses each training batch's integrated posterior mean for the running intensity
reference. For reflections observed in the batch, the update is
`reference = 0.99 * reference + 0.01 * posterior_mean`. Absent reflections retain
their previous value. This is an exponential average with no fixed window; its
e-folding horizon is about 100 updates **per observed reflection**. The initial
unity merge supplies the starting value.

The reference is detached from autograd and is used for network context, Wilson
prior fitting and integration-grid placement. Posterior densities are not reused
as priors. Validation does not update any running reference. Checkpoints record
the estimator and decay as well as the running state. The default estimator
remains `wls` for comparison; both estimators use decay 0.99 by default.

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

### Shared Wilson surface

`--prior joined-wilson` fits one Wilson surface from all conditions and uses it
for training and integration. Conditions retain separate intensity running means
and integration widths. Each observed condition contributes equally to the fit,
regardless of its reflection count; untouched running estimates are excluded.
Before any updates the shared pooled initialization supplies the fit. Smooth and
shell bases are supported; unsupported shells retain their previous prior.
Use `--reference-estimator posterior` to fit from posterior running means
(the reference estimator default remains `wls`).

Separate checkpoint readout fits the Wilson surface from the current input's
unity reference, including when overriding `wilson` or `joined-wilson`. All bins
start from this common reference, and full/half readouts use the same fitted
surface. Separate readout can therefore differ from training's immediate exports,
which use the trained per-state running references.
