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
