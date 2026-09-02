# optimize_integration

Correctness and speed of the posterior integration used by the amortised scale
model (`torchsx.nn.posterior_target.PosteriorMergeTarget`).

The merge is a nested integral — an inner marginal over `log S` per observation
and an outer integral over `log I0` per reflection — and it is ~57 % of a
training step. This directory exists because neither its accuracy nor its cost
had ever been measured: both were argued from first principles, and several of
those arguments turned out to be wrong.

## What is here

| file | purpose |
|---|---|
| `oracle.py` | dense float64 oracle for **both** integrals, plus `make_problem` (generates from the model, so `I0_true` is known) and `REAL_LIKE` (parameters fitted to a real dark-stream batch) |
| `verify_oracle.py` | certifies the oracle: FFT vs direct, closed-form derivatives vs autograd, convergence, and recovery of `I0_true` |
| `score.py` | scores `PosteriorMergeTarget` against the oracle — values *and* gradients — over node counts, grid extent and likelihood |
| `bench.py` / `ops.py` | same for the inner accumulator alone, on a **real** batch (`dump_batch.py` writes it) |
| `roofline.py` | is the Triton kernel SFU-bound? (no: 17.5 % of peak) |
| `fused_check.py`, `fused_tune.py`, `fused_c.py` | the fused (M, C) kernel: correctness, tiling, and scaling with node count |
| `positivity_bias.py`, `fix_check.py` | ground-truth tests of the weak-end bias and of candidate fixes |
| `bg_forms.py` | the exact marginal over the annulus's Gamma posterior, and every candidate cheap form against it |
| `bg_recover.py`, `bg_sweep.py` | ground-truth I0 recovery per likelihood, and the width-vs-shape sweep over the coefficient |
| `bg_verify.py`, `bg_time.py` | kernel correctness (bit-exact when off, gradients vs autograd) and what the branch costs |
| `match_real.py` | fits `REAL_LIKE` to the real batch's marginals |

Three further measurements live in the analysis tree next to the run
scripts (`abismal/bench/`), because they need the stream rather than this
package: `effective_nbg.py` (background-ring occlusion from spot
positions), `calib_bg.py` (off-Bragg sigma^2(bg)) and
`logmean_sensitivity.py` (does a likelihood change flatten log p in
log_mean).

Run anything via `gpu.sbatch`; `verify.sbatch` runs the oracle certification.

## Settled

**The oracle is trustworthy.** Convergence floor ~3e-9 on values and 1e-9 on
gradients: halving the inner spacing, doubling the outer nodes, or widening
either truncation moves nothing above that. It reaches `CC(I0_mean, I0_true) =
0.99999` at the real multiplicity, so it is checked statistically as well as
numerically. Restriction: `beta >= 1`, because the GenGauss tail needs a grid
`30^(1/beta)` wide.

**Node allocation.** The shipped `k_coarse=21, n_points=2001` was inherited, not
chosen, and is off the Pareto front in both directions. Use **`k_coarse=31,
n_points=1355`** (same budget, 2.7x better merge values), or the same split at
0.7x the budget, which is still 2.2x better than today. Above ~1000 inner nodes
the merge value is limited entirely by the *outer* count; inner nodes buy only
gradient accuracy. Keep the extent at **±34.5** — narrower truncates the
GenGauss tail, and that damage is invisible to an inner-accumulator-only test.
Hard floor: `n_points >= 1000`; at 591 it collapses (edge_mass 1.00).

**Kernel.** Aux statistics `S1..S4` are ~47 % of the kernel and are dead weight
unless the matching input requires grad — now gated by `constexpr`, bit-exact,
1.6–1.9x. The Poisson factor is formed about its own peak
(`n log(lam/n) - (lam-n) + logpeak`) rather than as `n log lam - lam - log n!`,
whose three ~3e6 terms cancel to -6.4 and cost ~0.4 nats in float32: 25x better
values on the real batch for +5 % runtime.

**Fused (M, C) kernel** (`_fused_posterior_triton.py`) is correct and 2.08x at
C = 32, but its closed-form shift costs dynamic range and its gradient
direction is ~5° off. Default **off**.

## Settled: the background estimate's error

`bg` is not measured, it is estimated from the 9-11 px annulus, so it carries
`sigma_b^2 = (n_peak/n_bg)*bg` -- 11.6 counts against a median signal of 7.3.
The plain Poisson treats it as exact, and the error *rectifies* (an upward
fluctuation makes the apparent signal negative, which `S = K I0 exp(eps) > 0`
cannot represent), inflating weak reflections 2.4-3.4x on ground truth.  It does
not average down with multiplicity.

The direct symptom is a width error: measured against the exact marginal over
the annulus's Gamma posterior, the plain Poisson's marginal is **0.70x** as wide
as it should be, and **0.32x** in the weakest quartile.

**The fix is the functional form, and the first one was wrong.**
`bg_var_mode="second_order"` expanded `int Poisson(n|S+b) N(b|bg,sigma_b^2) db`
about `b = bg`.  That expansion is unbounded as `lam` leaves `n`, so it needed a
+-10 clamp -- which fires on **16 % of the grid points that carry mass** -- and it
moves the mode.  `bg_var_mode="deviance"` (now the default) instead scales the
Poisson deviance by the dispersion `phi = 1 + sigma_b^2/lam`, the saddlepoint
density of the overdispersed law:

    log p = -D(n, lam) / (2 phi) - 0.5 log phi + logpeak(n)

Mode at `lam = n`, variance `lam + sigma_b^2`, exact Poisson at `sigma_b^2 = 0`,
bounded, no clamp.  One reciprocal and one `log2` per grid point, which measures
(`bg_time.py`, M = 1.05e6, K = 1355) as 12.63 -> 14.40 ms without gradients and
15.73 -> 19.03 ms with them: **+14 % / +21 %** on a kernel that is ~57 % of a
training step, so ~8-12 % on the step.  With `bg_var_factor = 0` the kernel is
bit-identical to before (`bg_verify.py` test 1: max |diff| exactly 0).

Median |error| against the exact marginal, over grid points within 15 nats of
the peak (`bg_forms.py`, real `(n, bg)` pairs):

| form | med nats | max nats | sd/exact | weak-quartile E[S] (exact 2.9) |
|---|---|---|---|---|
| plain Poisson | 1.48 | 24.8 | 0.70 | 0.3 |
| second_order (clamped) | 0.89 | 15.0 | 1.48 | **15.9** |
| **deviance** | **0.13** | 5.2 | 1.02 | 3.6 |
| negative binomial | 0.10 | 2.4 | 1.01 | 3.2 |
| Gaussian, `sigma^2 = lam + sigma_b^2` | 0.19 | 20.7 | 1.04 | 4.2 |

Ground-truth recovery through the production merge (`bg_recover.py`,
`bg_sweep.py`; ceiling = the same problem with an exact background, 1.018):

    poisson                 3.03      second_order   0.82  (overcorrected)
    deviance x1             1.21      deviance x2    1.05
    gaussian + bg_var       1.03

**Trained, 100 epochs, same config, only the likelihood differing:**

|  | plain Poisson | + deviance bg_var |
|---|---|---|
| CC1/2 | 0.9932 | **0.9949** |
| R_split | 0.0779 | **0.0742** |
| var_z | 0.591 | **1.085** |
| median predicted `w` | 1.605 | **1.390** |

`w` falling is the result to read.  `var_z` is *not* independent evidence of it:
its denominator is `max(n, bg) + (K I0)^2 (e^{w^2}-1)` and does not include the
background term, so a smaller `w` mechanically raises `var_z` with the residuals
unchanged.  The two rows are one fact, not two.  What makes that fact
meaningful is that CC1/2 and R_split -- which are computed from a split-half
merge and know nothing about `w` -- improve at the same time.  (`_posterior_step`
now adds `bg_var_factor*bg` to that denominator, so runs from 2026-09-02 onward
report a `var_z` that is not comparable with the table above.)  The unity-scale evidence
scan agrees -- best `w` moves 1.65 -> 1.45 the moment the term is modelled.  An
earlier claim that `w` did *not* move was measured with the second-order form.

A negative binomial moment-matched to `(lam, lam + sigma_b^2)` is the better
approximation pointwise (0.10 nats against 0.13) and IS evaluable in float32 --
the direct form cancels to 1e14 but substituting `p = n sigma_b^2/lam^2`,
`q = sigma_b^2/lam`, `g(x) = log(1+x) - x` removes the cancellation exactly
(`nb_precision.py`: 1.7e-5 nats where mass lives).  It is **not worth the two
extra transcendentals**: ground-truth recovery is 1.196 against the deviance
form's 1.207.  Settled; do not revisit.

**Do not tune the coefficient -- three independent criteria agree on the
nominal `n_peak/n_bg = 1.164`.**

1. *Real-data evidence*, each coefficient at its own best `w`: `0 ->
   -15,968,119`; **`1.164 -> -15,660,228` (max)**; `1.746 -> -15,666,785`;
   `2.328 -> -15,700,485`; `3.492 -> -15,797,496`.
2. *Measured from the residuals* (`var_decomp.py`).  In the low-signal cells,
   where the multiplicative scatter contributes least, the excess variance over
   the counting terms `mu + bg` is a constant multiple of `bg` across a **40x
   range** of bg (0.99 to 1.62, median **1.35**; ~1.2 after removing the fitted
   `mu^2` term).  This is the number `calibrate_sigma2_of_bg` was meant to give
   before its off-Bragg gate turned out to be selecting Bragg peaks.
   Read it off the low-signal cells, never off the joint fit: that fit is
   dominated by the `mu^2` term (top cell `Var = 2e5` against `30` at the
   bottom) and returns `c ~ 6.9`.
3. *Ground-truth recovery* prefers 1.5-2x -- but that is an estimator criterion
   buying a smaller bias with extra width, and it is the odd one out.

Real-data merge at matched `n_sigma = 12` (`bg_factor.sbatch`):

| coefficient | median F | F/sigF | edge_mass >1% | weakest decile F |
|---|---|---|---|---|
| 0 | 8.361 | 23.49 | 0.085 % | 4.868 |
| **1.164 (deviance)** | 7.938 | 19.41 | 0.149 % | 4.053 |
| 1.746 | 7.828 | 18.17 | 0.177 % | 3.718 |
| 2.328 | 7.734 | 17.22 | 0.191 % | 3.418 |
| 1.164 (second_order) | **4.971** | **12.07** | 0.474 % | **0.700** |

The last row is the failure: median `F` down 40 % and the weakest decile at
0.700 against a refined model's ~2.95.  The deviance family moves gently and
monotonically, and pays ~17 % of `F/sigF` for admitting the background's
uncertainty -- which is the honest price, not a defect.

### The two fixes together

`ewald_offset` (below) and the background term are independent and both help.
100 epochs, same config, 2x2:

| ewald_offset | bg_var | CC1/2 | R_split | median `w` | within-crystal share | train-val gap |
|---|---|---|---|---|---|---|
| broken | off | 0.9932 | 0.0779 | 1.605 | 0.115 | +0.0018 |
| broken | deviance | 0.9949 | 0.0742 | 1.390 | 0.053 | +0.0021 |
| fixed | off | 0.9974 | 0.0540 | 1.258 | 0.493 | +0.0020 |
| **fixed** | **deviance** | **0.9982** | **0.0536** | **0.895** | 0.483 | **+0.0002** |

`median w = 0.895` against the `ScaleBCorrection` baseline of 1.61 is the
acceptance number the design doc set, and it is now beaten by 45 % rather than
by the 0.3 % that `ScaleBPerCrystal` managed.  The train-val gap collapsing to
+0.0002 says the width is no longer being spent on fitting noise.

Note CC1/2 and R_split are *precision* measures and are structurally blind to
the bias this correction removes -- both halves inflate identically, so they
agree better for it.  That they improve anyway is the necessary condition, not
the evidence; R_free and the weak-end floor are where the correction has to
prove itself.

## Two artefacts that produced false conclusions

**`ewald_offset` was computed from the Friedel-canonicalised Miller index --
and this cost more than the likelihood ever did.**
`symmetrize()` negates about half of them, so half the excitation errors were
those of `-q`: |offset| ran to 0.89 A^-1 (q75 = 0.43) instead of 0.0068 A^-1 max.
`calibrate_sigma2_of_bg` gates on `|r| > 0.25` for a signal-free sample and
therefore selected 36 % of observations whose mean I was 68 photons -- Bragg
peaks -- so its `intercept = 29.65` and `mu_z = +1.63 sigma` were meaningless.
After the fix there are no off-Bragg observations at all, and the truncnorm
fallback gives `mu_z = +0.061 sigma`: **there is no systematic background bias.**
`ewald_offset` is also a *network input feature*, and features are standardised,
so the artefact spanned the whole range and compressed the genuine +-0.007 A^-1
partiality signal into ~1.5 % of it.  With `--fix-profile-radius` having already
removed `profile_radius` as a source of partiality information, this was the
geometric channel that was left.  Trained, 100 epochs, same config:

|  | broken feature | fixed feature |
|---|---|---|
| CC1/2 | 0.9932 | **0.9974** |
| R_split | 0.0779 | **0.0540** |
| median predicted `w` | 1.605 | 1.258 |
| **within-crystal share of var(log K)** | 0.115 | **0.493** |

The last row is the one that matters: the premise of an amortised scale is that
the *within*-crystal part -- partiality -- is real and predictable, and a
per-crystal G+B model is 100 % between.  At 0.115 the network was mostly
reproducing G.  At 0.493 it is predicting partiality.

Checkpoints trained before the fix cannot be merged with the current code.

**`n_sigma` is not a constant, and a merge at one value is not comparable with
a merge at another.**  `edge_mass` is the mass in the outer 5 % of nodes at each
end, on a grid spanning `+- n_sigma * sd` where `sd` is read off a coarse scan.

At 6, the *plain Poisson* unity merge already exceeds 0.01 for 23 % of
reflections; at 12 it is 0.085 %.  Every reference merge in `bench/refine/` used
`--n-sigma 12`, and a deviance merge run at the old default 6 reported 27.7 %,
which read as the correction blowing up the grid.  It was the grid.

**The requirement grows as the scale model improves**, which is the
counter-intuitive part.  `sd` is a *local* width, and a sharper posterior has a
smaller `sd` without proportionally lighter tails, so `+- n_sigma * sd` covers
less of the mass.  Measured on the same checkpoint (`edge_probe.sbatch`):

| checkpoint | median `w` | bg_var | n_sigma | edge_mass >1% |
|---|---|---|---|---|
| evplain | 1.258 | on | 12 | 0.000 % |
| evdev | 0.895 | **off** | 12 | 20.97 % |
| evdev | 0.895 | on | 12 | 11.98 % |
| evdev | 0.895 | on | 24 | 0.057 % |
| evdev | 0.895 | on | 40 | 0.000 % |

Note the correction is not the cause -- switching it *off* on the same
checkpoint makes truncation worse.  Widening does not move `F` (1.872 -> 1.866
-> 1.859 across 12/24/40) but it does move `F/sigF` (29.59 / 29.45 / 27.93):
past ~24 the fixed `m_fine = 129` nodes are spread too thin and the posterior
width is over-estimated.  **Use 24 for a trained merge, 12 for unity, and read
`edge_mass` every time rather than trusting either.**

Check `sacct -j <id> -o SubmitLine%200` before trusting a baseline from an
earlier run.

## Result: correction in training, with the width prior reversed

`lam_spread` had to go **negative**.  The only change between `evdev` and
`evdevup10` is `lam_spread` 0.05 -> -0.10.  All merges at `n_sigma = 24`,
`edge_mass = 0.000 %`, same free set, `--xray-mode nll`:

| merge | R_free | weak-quartile slope | strong-quartile slope | floor (model 2.95) |
|---|---|---|---|---|
| unity, plain Poisson | 0.1533 | 0.562 | 1.020 | 12.18 |
| trained, broken ewald, plain | 0.1384 | 0.282 | 0.975 | 5.55 |
| trained, fixed ewald, plain | 0.1350 | 0.158 | 0.986 | 4.70 |
| trained WITH deviance, `w` collapsed | 0.1513 | 1.148 | 0.958 | 5.23 (non-monotone) |
| **trained WITH deviance, width pulled up** | **0.1292** | **1.070** | **1.010** | **3.62** |

That is the best R_free of any merge in this project (previous best 0.1348) with
the weak-end compression **gone** -- both slopes within 7 % of 1.  `evdevup25`
(`lam_spread = -0.25`) reproduces the training numbers (CC1/2 0.9978, R_split
0.0535, `w` 1.111), so the result is not perched on one coefficient.

Caveat on the floor column: it bins on `I_model` from a model refined against a
*different* merge, and model decile correlates with resolution, so a difference
in overall B between two merges tilts the profile.  Read the weakest decile and
the slopes; do not over-read the middle deciles.

## Why the width prior had to be reversed

All trained, all `edge_mass` ~ 0, all `--xray-mode nll`, same free set.  The
profile is `median I_obs` per decile of `I_model` from an independently refined
model, so it is not circular with our own merge:

| merge | R_free | weak-end floor (model 2.95) | profile monotone in I_model? |
|---|---|---|---|
| broken ewald, plain Poisson | 0.1384 | 5.55 | yes |
| fixed ewald, plain Poisson | **0.1350** | 4.70 | yes |
| fixed ewald, **deviance at merge only** | 0.1374 | **3.97** | yes |
| fixed ewald, **trained with deviance** | 0.1513 | 5.23 | **no** |

Merging with the correction, on a network trained without it, costs +0.0024 in
R_free -- about twice the refinement's repeat noise -- and closes 40 % of the
remaining floor, leaving the strong end untouched (`20.47/17.73`, `38.28/30.33`,
`103.60/86.43`, identical to the uncorrected merge).

Training *with* it does something different and worse.  Its merged intensities
stop increasing with the model: `5.23, 6.56, 5.11, 4.06, 4.73, 6.24, 9.19`
across the seven weakest deciles.  That is not a floor, it is a loss of
ordering, and it costs +0.016 in R_free.  **Meanwhile its CC1/2 is the best of
every run at 0.9982** -- the clearest demonstration available that CC1/2 measures
precision and cannot see this at all.

The suspected mechanism is the width head: trained with the correction it goes
to a median `w` of **0.895**, far below the **1.45** the unity-scale evidence
scan makes optimal under the same likelihood.  A narrow scatter lets a single
observation dominate a reflection.

**`lam_spread` is not a shrinkage toward a target -- it is `lam_spread *
mean(log w)`, one-sided, and pushes `w` DOWN.**  Raising it therefore pushes
further the wrong way, which is what the first two attempts measured:

| run | lam_spread | median `w` | CC1/2 | R_split |
|---|---|---|---|---|
| evplain | +0.05 | 1.258 | 0.9974 | 0.0540 |
| evdev | +0.05 | 0.895 | **0.9982** | **0.0536** |
| evdevw25 | +0.25 | **0.069** | 0.9888 | 0.0758 |
| evdevw50 | +0.50 | 1.505 | 0.9844 | 0.0914 |

(+0.50 landing above +0.25 is not a monotone response -- that run's
within-crystal share fell to 0.212, so it is in a different regime, not a
stronger version of the same one.)  A **negative** coefficient is what pulls `w`
up, and it cannot run away: the GenGauss normaliser carries `-log(alpha)`, so
widening by `c` already costs `log c` per observation and any
`|lam_spread| < 1` leaves that restoring force intact.  `evdevup10.py` /
`evdevup25.py` test that.

## Open: the residual weak-end floor

With the background handled, ground-truth recovery is 1.21 against a 1.02
ceiling -- most of the bias is gone, some is not.  Against a common refined
reference model, the weakest decile's merged intensity reads 11.3 (plain
Poisson), 7.9 (deviance), 5.3 (trained scale, plain Poisson) and 0.9
(second-order, overcorrected), where the model says 2.85.  Note the reference
model was itself refined against one of these merges, so it is a proxy, not
truth.

Ruled out by measurement -- do not re-litigate: the Wilson prior (it pulls weak
reflections *down* 11 %, and the prior-free ML estimate still floors); outer-grid
truncation; positivity of `I0 = exp(v)` (with an exact background, true-zero
returns 0.01); global-`w` double-counting *as an explanation for the
second-order form's failure*; an oversized coefficient (the evidence peaks at
nominal); a crowded annulus (nearest-neighbour spot distance is 150 px median,
**0.0 %** within 18 px, effective `n_bg` = 128.0); a systematic background bias
(`mu_z = +0.061 sigma`); and the hypothesis that the correction flattens `log p`
in `log_mean` (`bench/logmean_sensitivity.py`: the drop from peak to +-6 nats is
-5.17 nats without and -5.20 with, i.e. unchanged).

## Accuracy validation

`../../../bench/` holds the refinement-based check — CC1/2 measures precision,
R_free measures accuracy. Trained scale **R_free 0.1397** vs unity 0.1648 vs
partialator + French-Wilson 0.1825, identical model, target, cell and free set.
Refinement repeat noise is ~0.001, so differences below ~0.005 are not callable.
