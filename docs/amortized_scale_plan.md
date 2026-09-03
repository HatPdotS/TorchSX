# Amortised scale model for torchsx — design and plan

Status: design agreed 2026-08-31. **Phases 0-3 implemented and tested**
2026-08-31 — see "Implementation status" at the end. Two decisions in this
document were revised during implementation and are marked **REVISED** in
place; the revisions are the authoritative version.
Working dir this was developed in:
`/das/work/p17/p17489/Peter/2025-01-31_SwissFEL_SFX/svd_time_signal_test/abismal/`
(the abismal investigation that led here is in `notes/CONTEXT.md` there).

---

## 1. Why

torchsx already has the correct statistical model and beats partialator:

| method | CC1/2 | CC* | R_split |
|---|---|---|---|
| **EM Poisson** | **0.9872** | 0.9968 | 0.0605 |
| GenGauss (log-space) | 0.9869 | 0.9967 | 0.0435 |
| partialator unity | 0.9850 | 0.9962 | 0.0762 |
| partialator xsphere | 0.9600 | 0.9897 | 0.1161 |

What it does **not** have is an amortised scale model. `ScaleBCorrection`
(per-crystal G + B) only moves median alpha 1.67 -> 1.61, i.e. it removes ~4% of
the inter-crystal variation.

Measurements made on this dataset that explain why a per-crystal scale is weak,
and why amortisation should help:

| quantity | value |
|---|---|
| observations with \|ipred\|/sigma < 1 (likelihood ~flat in the scale) | **84.1%** |
| reflections per crystal | 177 |
| **effective** reflections determining that crystal's scale (participation ratio) | **4.3** |
| share of Fisher information in the top 5% of observations | 97.8% |
| precision on a per-crystal scale, sigma_G/G | median **6.95%** (10th 1.45%, 90th 32%) |
| usable reflections per image, \|ipred\|/sigma > 1 | 16 |
| fraction of total variance held by I/sigma > 20 observations | 99.42% |

So: the information is *concentrated* (N_eff 4.3) but not *scarce* (sigma_G/G 7%,
median total Fisher info 207 per crystal). A per-crystal fit spends ~200 units of
information on one number. An amortised model fits the scale *function* across
millions of observations instead.

Also decisive for this dataset: integration used `--fix-profile-radius=4e7`, so
`profile_radius` is **constant across all crystals** in the stream. The one
geometric parameter that determines partiality carries zero information. Mosaic
state can only be inferred from the pattern of a crystal's *other* reflections,
which is exactly what a leave-one-out context encoder does.

## 2. Direction: port abismal's scaler INTO torchsx

Not the reverse. abismal is TensorFlow, torchsx is PyTorch, and **you cannot
backprop from torchsx's loss into a TF network** — so "use abismal inside
torchsx" cannot mean joint training. Port-or-nothing.

The port is small and the asymmetry is clear:

| | torchsx | abismal |
|---|---|---|
| likelihood | Poisson/Gaussian on counts, negatives handled | positive-only posterior |
| scatter | GenGauss in log space, per-hkl alpha/beta | unbounded scale, saturates to 1e-12 |
| marginalisation | 12 quadrature schemes + Triton kernels | 32 MC samples, noisy |
| **scale model** | per-crystal G, B | **amortised network** |

Only the last row favours abismal, and it is ~500 lines of MLP. Everything else
favours torchsx and is ~8,000 lines of tuned numerics.

Much of abismal's bulk is ragged-tensor plumbing that vanishes in torchsx's flat
tensor + `crystal_id` index representation.

## 3. The central idea: leave-one-out at every level

This is the architectural principle the rest of the design follows from.

**Every quantity used to score observation `j` is computed from everything except
`j`.** Applied consistently, it turns the training objective itself into a
cross-fitted, out-of-sample prediction — which is what stops the whole scheme
becoming a tautology.

It appears at two levels in the forward pass, and both are the same one-line trick:

| level | what is excluded | what it prevents |
|---|---|---|
| **context pooling** | `j` is excluded from its own crystal's pooled embedding | the network reading `j`'s own intensity to set `j`'s own scale — pure circularity |
| **merge target** | `j` is excluded from `ybar` for the reflection | the network lowering the loss by dragging the merged estimate toward `j` |

Both are exact and essentially free — sum-minus-self on an accumulator that has
to be computed anyway:

```python
# context (per crystal)
sums = zeros(n_crystals, D).index_add_(0, crystal_id, h)
cnts = bincount(crystal_id).clamp(min=2)
ctx  = (sums[crystal_id] - h) / (cnts[crystal_id] - 1)[:, None]

# merge target (per reflection)
num = zeros(n_hkl).index_add_(0, hkl_id, w*y);  den = zeros(n_hkl).index_add_(0, hkl_id, w)
ybar_loo = (num[hkl_id] - w*y) / (den[hkl_id] - w)
```

Why it is the important part: without it, both halves of this design are
degenerate. A scale predicted from an observation's own intensity can fit that
observation perfectly and mean nothing. A merge that includes the observation
being scored can always be pulled toward it. With leave-one-out, the only way to
lower the loss is to make observations **mutually predictable** — to predict `j`
from the crystal's *other* reflections and have it agree with what the *other*
crystals say about that reflection. That is a genuine generalisation objective,
not a fitting objective, and it is enforced inside the loss rather than by a
held-out split.

It also explains why the heads are per-observation (section 4): with a
leave-one-out context, each observation is an independent prediction problem, so
there is nothing to gain from tying predictions together per crystal.

Two caveats worth carrying:

- **The guarantee is per forward pass, not per parameter.** The shared network
  weights were trained on `j` in earlier steps. That residual leakage is what the
  held-out *crystals* diagnostic in section 9 is for.
- **"Independent" is doing real work.** Observations of the same reflection from
  the same crystal share that crystal's scale error, so their agreement is
  inflated. Rare here, but check for systematic sources (Friedel mates mapping to
  the same ASU index, multi-crystal frames indexed twice).

## 4. The model

One amortised network is the **sole** scale source. `ScaleBPerCrystal` is
removed — two competing scale sources (one parametric, one flexible) would be an
identifiability mess.

Per observation the network predicts:

```
log K   scale               (this is the ScaleModel.log_k contract)
log s   scatter width
beta    scatter shape       (via sigmoid onto ~[0.3, 4])
```

**Reparameterise (alpha, beta) -> (w, beta)** where `w` is the actual SD
(implemented as `log_alpha_from_log_width`; sanity points beta=2 -> alpha=w*sqrt2,
beta=1 -> alpha=w/sqrt2):

```
log s = log alpha + 0.5*[log Gamma(3/beta) - log Gamma(1/beta)]
```

because `Var(GenGauss) = alpha^2 Gamma(3/beta)/Gamma(1/beta)` — alpha and beta
are degenerate (both set the width) and would compete. With (s, beta) the width
and the shape are nearly orthogonal: beta is then identified only by the *shape*
of the residual distribution, which is a genuinely different signal.

**No explicit penalty on s.** The GenGauss normaliser already contains `-log alpha`:

```
log GG(e; alpha, beta) = log beta - log 2 - log alpha - log Gamma(1/beta) - |e/alpha|^beta
```

Widening by a factor c costs exactly `log c` per observation. Automatic,
correctly scaled, no hyperparameter.

**No free parameters anywhere except network weights.** Every per-observation
quantity is predicted from features by a shared network, which is regularisation
by construction. This is what makes it safe where global refined coefficients
were not — see section 10.

## 5. The loss  **[REVISED — the marginal kernel, not the closed form]**

**Original decision:** a closed-form Gaussian agreement loss, "two scatter-adds,
no quadrature at all", accepting that beta becomes unidentifiable.

**Revised decision: use the marginal kernels.** The reason is a measurement, not
a preference. `marginalize_signal_triton` already has this signature:

```python
marginalize_signal_triton(log_mean, n_ph, bg, log_alpha_obs, log_beta_obs, log_S_grid, log_w)
#   d log_p / d log_alpha_obs = +beta*S3 - 1
#   d log_p / d log_beta_obs  = -beta*S4 + 1 + psi(1/beta)/beta
```

alpha and beta are **already per observation** with exact analytic gradients,
and the `(M, K)` integrand is never materialised — the kernel saves four aux
statistics, ~5 floats per observation. At 1.46M observations per batch the
quadrature costs ~30 MB against several GB of network activations. So the
marginal is the *cheap* option here, and it keeps beta identifiable, which the
Gaussian closed form cannot.

Per observation:

```
log_mean_j = log I0_h^(-j) + log K_j
log p_j    = log INT p_det(n_j | S, bg_j) . GenGauss(log S - log_mean_j; a_j, b_j) d log S
loss       = -SUM_j log p_j  +  anchor(mean log K = 0)
```

The Cox-Reid term goes with the closed form. **Leave-one-out does that job**:
every observation is scored against a target it did not contribute to, which
addresses the same incidental-parameter leakage more directly than a correction
term does.

The closed form survives as `AgreementNLL` behind the objective seam — grid-free
and exactly differentiable, so it is a fast dev loop and a cross-check (the two
must agree at beta=2 in the high-count limit).

### The merge target

```
y_j     = (n_j - bg_j) / K_j
tau_j^2 = sigma_j^2/K_j^2 + I0_ref^2 (e^{w_j^2} - 1)
I0^(-j) = LOO inverse-variance mean of y over the reflection, plus an EMA prior
```

computed under `no_grad` from a **detached** prediction. Leave-one-out blocks an
observation from pulling its own target; detachment blocks the network from
pulling the whole merge toward itself collectively. The EMA enters as a
pseudo-observation on the accumulator (`prior_num`, `prior_den`), which is exact
and preserves leave-one-out because a term independent of `j` cannot leak into
`j`. lambda -> inf is the fixed-target EM fallback, lambda -> 0 full in-batch.

**No merged intensity is a parameter.** No I0 vector, no EM step, no LBFGS.

## 6. Architecture

Deep Sets, ported from abismal's `ImageScaler`:

```
per observation:  metadata -> Linear(D)
                     |
        +------------+------------+
        | IMAGE branch            | SCALE branch
        | (sees n, sigma)         | (metadata only)
        | L residual FF blocks    | Linear(D)
        | LEAVE-ONE-OUT mean      |
        |   over crystal_id  -----+--> added as bias
        +------------+------------+
                     | L residual FF blocks
                     | -> 3 heads: log K, log s, beta
```

- residual FF blocks are ResNet-v2 style (RMS pre-norm, second Dense zero-init
  so each block starts as identity)
- leave-one-out mean via `index_add` on `crystal_id`, then `sums[cid] - x` over
  `(counts[cid] - 1)`. ~5 lines; abismal's ragged version is much uglier.
- drops in as a `ScaleModel` subclass: `log_k(params, meta, obs_range) -> (n_obs,)`

### Inputs

`ObsInfo` already carries: `crystal_id`, `s_sq`, `hkl_idx`, `panel_id`, `fs`,
`ss`, `peak`, `background`, `ewald_offset`.

**hkl goes in for now** — but see the monitor in section 9.

**The context should carry the crystal's other residuals** (`n_i/(I0_h K_i)`),
which is far more informative than raw counts — "this crystal's other
reflections run 20% high" *is* the scale. Use **previous-cycle residuals with
stop-gradient** to break the K -> residual -> I0 -> K feedback loop.

**Condition-blindness is a hard constraint.** The scale model must NOT see
`delay_bin_idx` or `unique_timepoints`. Any degree of freedom that can
distinguish light from dark will absorb the difference signal. Assert this in
code, do not leave it to convention. Conditions are handled partialator-style:
one shared network, separate merge targets per condition.

## 7. Training

- **Adam everywhere. No LBFGS.**
- **Gradient checkpointing on by default.**
- **8192 crystals per batch** (~1.46M observations, ~77 observations per
  reflection in-batch).
- Global scale is unidentifiable (multiply all K by c, divide all ybar by c ->
  identical loss). **Anchor `mean(log K) = 0` per batch from step 0.**
- Size the batch in *observations* rather than crystals so memory stays flat
  (crystals have a variable observation count; 178 is the median).

### Memory (D = model width, L = layers per stack, 2L blocks total)

Per-observation activations ~ `2L * 6 * D * 4` bytes.

| config | per obs | per crystal | per crystal, checkpointed |
|---|---|---|---|
| D=256, L=5 | 61 KB | 10.9 MB | **1.8 MB** |
| D=128, L=5 | 31 KB | 5.5 MB | 0.91 MB |
| D=128, L=3 | 18 KB | 3.3 MB | 0.55 MB |

At D=256, L=5, checkpointed, 8192 crystals is ~15 GB — comfortable on a 40 GB
A100, easy on the 80 GB H100 in `gpu-large`.

Parameters are negligible (~2.7M weights + Adam moments = ~32 MB). All cost is
activations. **Start at D=128** — abismal's class default is 32 and its CLI
default is 256, a factor of 8 apart, with the CLI value coming from a benchmark
config that was subsequently reverted. Scale up only if held-out likelihood says
capacity binds.

No MC samples: the scale is deterministic and the uncertainty lives in the
predicted scatter. abismal multiplies its activations by `mc_samples=32`.

## 8. Initialisation

```
1. seed I0_EMA  <- existing EM Poisson merge              (one pass)
2. empirical scatter -> bias for the log s head           (one pass)
   (benchmark puts median alpha at 1.65 Poisson / 1.25 log-space-truncated;
    use 1.65, the unbiased one)
3. zero-init all three output heads; biases = [0, log 1.65, beta0]
4. lambda starts high, decays on a schedule
5. anchor mean(log K) = 0 from step 0
```

Zero-init gives `K == 1` at step 0, so the first merge is automatically a unity
merge — and `partialator --model=unity` already reaches CC1/2 0.985 on this data,
so that is a serviceable start with no special-casing.

Better: blend the in-batch merge with a running EMA seeded from the existing
merge:

```
I0 = (den_batch*I0_batch + lambda*den_EMA*I0_EMA) / (den_batch + lambda*den_EMA)
```

Start lambda high so the network first learns K against a **stationary**
reference, then decay. This is both the cold start and the mitigation for
co-adaptation. It also spans both modes with one knob: lambda -> infinity is the
EM fallback (fixed target, re-merge once per epoch), lambda -> 0 is full in-batch
merging.

Early diagnostic: initial loss must be finite and sensible, and the first
gradients on log K should be dominated by crystals whose brightness is furthest
from the mean. Enormous or NaN initial loss => the scatter initialisation is
wrong, not the architecture.

## 9. Diagnostics and acceptance

**Primary, falsifiable:** median `s` should drop materially below the 1.61
equivalent of the current `ScaleBCorrection` baseline. If an amortised scale with
crystal context cannot beat it, the residual scatter is not predictable from the
features and the approach is dead — a cheap, clean result either way.

**Held-out CRYSTALS, not observations.** Score observations from crystals the
network never trained on, against a merge built from training crystals. If
in-batch loss falls while held-out loss does not, that is co-adaptation through
the merge rather than learning. This diagnostic was missing throughout the
abismal investigation and would have caught several dead ends early.

**hkl permutation monitor.** Permute the hkl columns across observations and
measure how much of predicted `log K` variance moves. It was 0.3% in abismal. A
scale that depends on hkl is degenerate with I0 and is the one channel that can
manufacture a difference peak. If this number grows as the model strengthens,
drop hkl before any light/dark work.

**Calibration.** Normalised residuals `(y_j - ybar)/tau_j` should have unit
variance per bin. This pins the *level* of the scatter as a constraint rather
than a fitted parameter — which is what Evans intends for Sdfac and what both
known failures (section 9) were missing.

**Same-crystal pairs.** "Independent" is doing real work in the agreement
objective. Observations of the same reflection from the same crystal share the
crystal's scale error, so their agreement is inflated. Rare at 178 reflections /
19,030 reflections, but check for systematic sources (Friedel mates mapping to
the same ASU index, multi-crystal frames). Cheap test: exclude same-crystal pairs
from the chi^2 and see whether the loss moves.

**Second-order:** predicted scatter should show real spread and correlate with
excitation error — that is the partiality signal `--fix-profile-radius` erased
from the geometry. If beta is reintroduced and depends on excitation error, that
is a direct measurement of where the partiality model is mis-specified.

## 10. Ruled out, with reasons (do not revisit)

**Free global noise coefficients are not identifiable from the data NLL.** Two
independent demonstrations:
- torchsx's own `GaussianPolynomialLikelihood` trainable mode converges to a
  non-physical `g1 ~ 14` and is NaN-prone (documented in
  `experiments/weak_reflections/README.md`, T9 run set).
- abismal's Ev11 `Sdfac` ran to 4.75 in 11 epochs, inflating every sigma ~5x and
  collapsing predictive power below the no-scale-model floor.

Note this is not a runaway: minimising `log sigma + r^2/2 sigma^2` gives
`sigma = r`, so it is the MLE honestly reporting that the model misfits by ~5x.
The defect is that ONE global number is forced to describe every observation when
scale determinacy varies >20x across crystals. Predicting it per observation from
features is the fix; refining it as a free scalar is not.

**Prior pressure on the scale is the wrong knob.** `--scale-kl-weight` 10 and 100
in abismal gave healthy-looking scale distributions (CV ~0.8) and destroyed
predictive power (val_CCpred 0.031 and 0.007 against a 0.107 no-scale floor).
Weak prior gives saturation, strong prior gives no adaptation.

**Log-space merging biases the scatter low.** The benchmark measured alpha 1.25
(log-space, positives only) vs 1.65 (Poisson, all observations). 32.5% of
observations are negative; any merge that takes logs discards them and
underestimates the spread. This is why the merge must be a linear
inverse-variance regression.

**Positive-only posteriors compress weak reflections.** abismal reports zero
negative merged intensities in every shell where partialator has up to 6.4% in
the outer shell. For refinement this trades bias for variance and *improves*
R_free; for a difference map it pulls F_light and F_dark toward the same prior
and shrinks away the signal. Amplitudes are the wrong currency — keep intensities
with support on the whole real line and do French-Wilson after the difference.

**CCpred is not a selection metric.** Configurations with well-behaved scale
distributions scored ~0.10 while saturated ones scored 0.40, against a
no-scale-model floor of 0.107. Select on merged-output quality.

**"Differentiate through the merge" as a separate feature.** *[CORRECTED — the
implementation does NOT do this.]*  The in-batch merge target is computed under
``no_grad`` from a detached prediction, so no gradient flows through it.  The
model optimises "fit the current merge", not "improve the merge" — an
alternating scheme, not end-to-end differentiation.

That is defensible rather than merely convenient: the target is the weighted
least-squares argmin over I0 given K, and at an argmin ``dL/dI0 = 0``, so by
the envelope theorem the gradient through I0 vanishes to first order.  But the
argument is approximate on three counts — the target optimises the *Gaussian*
variance model while the objective is the GenGauss marginal, the lambda-blend
deliberately pulls the target toward the EMA and away from the batch argmin, and
leave-one-out means it is not the full-batch argmin either.

Whether letting the gradient through helps is therefore an open, testable
question, not a settled one.  The risk it guards against is real: leave-one-out
blocks an observation from pulling *its own* target, but nothing stops the
network lowering the loss by dragging the whole merge collectively.

## 11. Data facts for this dataset

- **dmin = 0.9 A.** The detector edge is at ~0.9 and only part of the detector
  reaches 0.86: 1.00-0.95 A 1,728 refl; 0.95-0.90 A 2,140; 0.90-0.86 A 1,965;
  **0.86-0.82 A only 332**. Merging past 0.9 drags a junk shell into every
  comparison.
- Space group **P1**; `-y -1` in partialator is the **Laue class** (Friedel pairs
  merged), not the space group. Passing "P -1" to merging software would impose
  centric statistics on an acentric structure.
- ~19,030 unique reflections, 178 observations per crystal.
- Sigmas are correct after the gain fix: SigI spans 1.0 to 2,269 with 9,753
  distinct values, I/sigma median 0.847, 12.9% above 5.
  `SigI/sqrt(I+1)` median 3.5 — expected for rings-cen integration where
  background-subtraction variance dominates weak reflections, not a defect.

## 12. Open

1. **Arbitrary sigmas.** Decided: use CrystFEL sigmas for now. `PreparedObs` has
   no sigma field — `_OBS_FIELDS` includes `"sigma"` and `SFXStream.sigma` exists,
   but it stops at `prepare_observations`. Needs a field + a
   `SuppliedSigmaLikelihood(GaussianLikelihood)` whose `sigma_squared_obs`
   returns it (~15 lines; the base docstring already anticipates "a lookup from
   external calibration"). Triton kernel already takes a per-obs `var` tensor.
   Open: replace the polynomial, or fold supplied sigma^2 into `g0` and keep the
   multiplicative term. Needs the split-half comparison, not an argument.
2. **beta** — deferred until residuals show it is needed (section 5).
3. **lambda decay schedule** for the EMA blend.
4. **Whether the context residuals help or destabilise** — biggest single
   unknown in the design.

---

## 13. Implementation status (2026-08-31)

### Built and tested

| file | what |
|---|---|
| `torchsx/nn/pooling.py` | `leave_one_out_mean`, `loo_weighted_mean` (+ EMA prior, cancellation guard) |
| `torchsx/nn/deepsets.py` | `ResidualFF`, `DeepSetsEncoder` (abismal port), `ScaleEncoder` protocol |
| `torchsx/nn/heads.py` | `ScaleHeads` -> `ScalePrediction(log_K, log_w, beta, log_alpha)` |
| `torchsx/nn/obs_features.py` | `ObsFeatureSpec` — the self/context tiering and both guards |
| `torchsx/nn/merge_target.py` | `LOOInBatchTarget` with the lambda-blend |
| `torchsx/nn/objective.py` | `MarginalNLL` (primary), `AgreementNLL` (reference) |
| `torchsx/nn/metrics.py` | the monitor suite + `MetricLogger` (JSONL) |
| `torchsx/nn/train_amortized.py` | `CrystalIndex`, `TrainConfig`, `AmortizedTrainer` |
| `torchsx/merging/scale/amortized.py` | `AmortizedScale(ScaleModel)` |
| `torchsx/merging/scatter/predicted.py` | `PredictedGenGaussScatter` |

Tests: `tests/test_loo_pooling.py` (22), `tests/test_amortized_scale.py` (12),
`tests/test_amortized_training.py` (3).

### The structural change

`ScatterModel.log_alpha_beta_obs(params, meta)` is now the single seam where
per-HKL alpha/beta are gathered. Overriding it supplies **per-observation**
network predictions instead, and the eager path, both Triton kernels, the
posterior moments, `compute_residuals` and `Merger` all accept that unchanged.
`ObsInfo` gained `sigma`, `obs_id` and `gather(idx)` — `obs_id` is what lets a
per-observation side-channel resolve under a non-contiguous crystal-batched
gather, where `obs_range` is meaningless.

Verified behaviour-preserving: the merging suite scored identically (19 failed /
69 passed) before and after. Those 19 were pre-existing breakage against an
older `Parameters` signature; 17 are now fixed as collateral, leaving 2
unrelated failures.

### Numerics: float32, and what leave-one-out actually guarantees

Sum-minus-self is exact in real arithmetic but not in floating point — the group
sum includes `x[j]`, so perturbing `x[j]` perturbs how the sum rounds. Measured
with a gross (1e3) perturbation: float32 accumulation leaks 1.7e-5, float64
accumulation is bit-exact, and **the gradient w.r.t. self is exactly zero in
every case**. That last one is the guarantee that matters — an optimiser moves
only along gradients, so training cannot *learn* to read an observation's own
value. Everything runs float32.

For the merge target the hazard is cancellation, not leakage. On realistic
groups (77 obs/reflection, heavy-tailed weights) float32 holds a median relative
error of 1e-7 against float64. It fails only when one observation carries
essentially the whole group weight — and there the guard marks the observation
**invalid** rather than returning silent garbage (without it, a true target of
1.0 comes back as 0.0 and is reported as valid). That is also the statistically
right call: such a group carries no independent peer information.

### Validation: the central claim, demonstrated

Synthetic Poisson data with a hidden per-crystal brightness `G`, and **no
feature encoding it** — the only route is the leave-one-out context.

```
before training   spearman(predicted per-crystal log K, true log G) = +0.027
after training                                                      = +0.965
sd(log K) = 0.43   vs true sd(log G) = 0.46
hkl-permutation sensitivity = 0.00000   (control: hkl excluded from features)
held-out NLL at fixed lambda tracks train — no co-adaptation blow-up
```

Codified as `test_leave_one_out_context_recovers_a_hidden_per_crystal_scale`.

### What the metrics caught immediately

Training NLL oscillated. The metrics named the cause without any extra
instrumentation: `median_w` swings 1.65 -> 0.15 -> 0.45 -> 0.20 while `var_z`
moves in **anti-phase** (0.13 -> 3.1 -> 1.0 -> 1.5). That is the width head
hunting — buying fit with uncertainty and paying it back — which is exactly the
"one term paying for another" failure that is invisible in the total loss and
is how `Sdfac` reached 4.75. It damps (`kurtosis` 45 -> 0.9, `frac_gt3`
0.080 -> 0.011) but the width-head learning rate is the knob to look at first on
real data.

Second finding: **the training NLL is not comparable across the lambda
schedule**, because lambda changes the merge target itself — as lambda falls the
target moves from a stationary EMA to a noisier in-batch merge and the loss
rises for that reason alone. Held-out scoring therefore pins `cfg.eval_lam`.

### Decisions taken with the user during implementation

- **Split protocol.** Production = train one network on all crystals. Validation
  = train `net_A` on half A and `net_B` on half B *independently*, apply each to
  its own half, merge separately. No cross-half leakage at all, not even through
  weights, so CC1/2 is directly comparable to partialator. Yields a metric
  nothing else gives: `CC(log K_A, log K_B)` on common observations.
- **float32 everywhere**; float64 only as a test/diagnostic option.

### Remaining

1. Real-data run on the dark streams (dmin 0.9 A) — the acceptance gate is
   median `w` materially below the 1.61 of `ScaleBCorrection`.
2. Split-half CC1/2 under the two-network protocol; epoch-level merge metrics
   (group E of the metrics plan) are not wired yet.
3. `SuppliedSigmaLikelihood` for CrystFEL sigmas — `ObsInfo.sigma` is now
   populated and the Gaussian Triton kernel already takes a per-obs `var`, but
   `sigma_squared_obs(n, bg)` needs `meta` threaded to it (the `fixed_log_grid`
   dispatch site already has `meta` in scope; the eager
   `log_prob_given_signal` does not).
4. Context residuals `n_i/(I0_h K_i)` with stop-gradient — biggest single
   unknown in the design; the raw `(I, sigma)` context is what ships now.

---

## 14. First real-data runs (dark stream, 2026-08-31)

`streams/dark.stream` → 70,338 crystals, 12.23 M observations at d ≥ 0.9 Å,
14,128 unique reflections. **Runtime 8.6 s/epoch** (8 steps of 1.5 M obs) on an
A100-40GB with the Triton path; the full 26 M-observation dark would be ~18 s,
matching the synthetic benchmark.

Real-data statistics match the design assumptions almost exactly — negative
fraction 32.4 % (assumed 32.5 %), I/σ median 0.846 (assumed 0.847), 12.3 %
above 5 (assumed 12.9 %), 174 obs/crystal (assumed 178).

### Two configuration errors this surfaced

**`gain` must be pinned to 1.0, not read from the stream.** The geometry says
`adu_per_eV = 1` at 12960 eV, so `MergeConfig(gain="stream")` computes
gain = 12960 ADU/photon and divides every intensity by it, handing the Poisson
likelihood counts of ~1e-4. The DAQ already divided by photon energy, so the
stored data *are* photon counts. Note I/σ is gain-invariant, so the statistics
table looks correct either way — this error does not announce itself.

**`filter_resolution` was broken** — it passed a per-observation `(N, 6)` cell to
`get_d_spacing`, which only accepts `(6,)`, so it raised for any stream with more
than one crystal. Fixed to use the median cell, which is also what
`prepare_observations` uses for `prep.d_spacing`: cutting on anything else would
use a different resolution scale than every downstream shell and bin, and a
consistent hkl set is wanted regardless.

### Degeneracy 1: hkl — found, closed

With hkl in the features, `hkl_permutation_sensitivity` reached **0.586**
(abismal: 0.003), `sd(log K)` blew up to 14.5, and 80 % of that variance sat
*within* crystal. The scale had become I0 by another name.

The failure signature matters: training converged normally (`var_z` → 1.02,
held-out 5.10 at epoch 18) and collapsed only afterwards (held-out 6.33,
`var_z` 12.5 by epoch 28). **The network learns the legitimate scale first and
discovers the shortcut second**, so a short run looks healthy.

`include_hkl` now defaults **False**, and the trainer warns above
`cfg.hkl_permutation_warn` (0.02). With hkl removed the monitor reads 0.00000.

### Degeneracy 2: the mean anchor is not robust

With hkl gone, `sd(log K)` was still 12.9. The distribution was bimodal: the
bulk in `[-5.5, +0.2]` with ~1.5 % of observations at `log K ≈ +68`. Under a
mean anchor those outliers contribute ≈ +1.0 to the mean and **drag the other
98.5 % down by ≈ 1.0** — the bulk median came out at −2.4. The gauge is
arbitrary, so any location statistic is valid; the median cannot be hijacked.
`anchor_stat` now defaults to `"median"`, and `sd(log K)` fell 12.9 → 0.69.

### Open: the misfit finds a new outlet

With both closed, the *width* head now runs away instead: median `w` 6.75
(against a 1.61 target), `var_z` swinging 0.004–1.53, training loss unstable
after ~epoch 22. This is the group-A failure — buying fit with uncertainty —
and it is being caught, but not yet prevented.

Every run so far is stable for ~20 epochs and then finds whichever unbounded
parameter is still available. That pattern says there is an underlying misfit
looking for an outlet, and the candidates are:

1. **Bound the heads architecturally**, as `beta` already is — `log w` to
   roughly [0.05, 5] and `log K` to [−8, +4]. This is a parameterisation
   bound, not prior pressure on the scale (which §10 rules out on evidence); it
   does not fight the data inside the range. abismal needed Student-t `-t 32`
   for the same reason: 0.19 % of reflections hold 46.9 % of the χ².
2. **λ schedule** — λ ≈ 2 by epoch 25; a slower decay keeps the target
   stationary for longer.
3. **Separate (lower) learning rate for the width head**, which was already
   seen hunting in anti-phase with `var_z` on synthetic data.

(1) is the most likely to be both necessary and sufficient, but it is a design
decision about the scale distribution and should be taken deliberately.

---

## 15. The mean/median bias — the actual cause of the width runaway

Every configuration ran away in the width head: w reached 6.75 (narrow fixed
grid), 26.96 (wide fixed grid) and 8.32 (trapezoidal), with beta pinned at its
4.0 ceiling — a near-uniform density, i.e. the model opting out of predicting
anything. Two wrong diagnoses first, both settled by measurement:

**Not the normalisation.** The fixed grid does truncate the scatter (`Sum_n
p(n)` = 0.941 at w=1.65, 0.586 at w=6.75), but dividing by the in-grid mass is
*not* the fix: the conditioning event moves with the parameters, so
`log p - log mass` runs to `+inf` as the mass goes to zero and the width head is
actively rewarded for outgrowing the grid. It drove the loss to 0 and NaN-ed the
network in one epoch. `gengauss_grid_log_mass` is kept as a **diagnostic only**.

**Not the quadrature.** A side-by-side sweep of NLL vs w over Trapezoidal, Wide
and Fixed grids agrees to <0.01 nat and puts the minimum in the same place for
all three. Grid truncation only set *how far* the runaway got (it penalised
alpha above ~2.5, an accidental brake — which is why widening the grid made
things worse, not better).

**The cause: the merge target and the likelihood disagreed about what
``log_mean`` means.** The target must be a *linear* estimator, because 32.4 % of
observations are negative and any log-space merge discards them. A linear
estimator estimates the **mean**. But ``GenGaussLogScatter`` centres its density
on ``log_mean``, so ``log_mean`` is a **median**. The two differ by
``E[exp(eps)]``:

| w | 0.3 | 0.8 | 1.25 | 1.65 | 2.5 |
|---|---|---|---|---|---|
| E[S] | 1.05 | 1.38 | 2.18 | **3.90** | **22.8** |

Because the factor **grows with the width**, it closes a loop: wider scatter ->
target biased higher -> worse fit -> wider scatter. That is the runaway.

`GenGaussLogScatterRecentered` already implements this shift, but from a single
global `(alpha_ref, beta_ref)` fixed at init, and its docstring called a
per-HKL version "the natural extension". It is not an extension — it is
load-bearing. A frozen shift leaves exactly the width-dependent residual that
drives the loop, so with a fitted width it must be per observation.
`log_e_s_gengauss(log_alpha, beta)` computes it on a shared standardised
`u`-grid (differentiable in both arguments; matches the closed form `w^2/2` for
beta=2 to 5 decimals), and `ScalePrediction.log_e_s` is subtracted when forming
`log_mean = log I0 + log K - log E[exp(eps)]`.

### Result on the real dark (30 epochs, trapezoidal grid)

| | before | after |
|---|---|---|
| held-out NLL | 5.10 then diverged | **4.995, stable to epoch 30** |
| median w | 8.32 | **2.01** |
| sd(log K) | 4.30 | **1.11** |
| var_z | 0.08 - 178 | **0.31, steady** |
| beta | 4.000 (ceiling) | 3.82 |
| hkl-permutation | 0.000 | 0.000 |

### What this now exposes

1. **`between_frac` = 0.973.** With the runaway gone, the scale is almost
   entirely per-crystal — the network has rediscovered G and the within-crystal
   (partiality) component is ~0.03. This is the number the whole approach rests
   on, and as it stands the amortised model does not yet earn its keep over
   `ScaleBPerCrystal`.
2. **median w = 2.01 against the 1.61 acceptance target, with var_z = 0.31.**
   Both say the same thing: tau is ~1.8x too large, so w should be falling to
   ~1.7 and is not. beta also still sits near its 4.0 ceiling. The shape head
   appears to be compensating for something the width head should be giving
   back.

### Also fixed along the way

* Chunking in `MarginalNLL` saved no memory — concatenating chunk outputs kept
  every chunk's graph alive to backward, which OOM'd a 40 GB A100. Chunks are
  now recomputed in backward (`checkpoint_chunks`, exact in value and gradient).
* `TrapezoidalLogGrid` is the pipeline default and is what should be used; the
  Triton kernels in this build (905b36f) are wired only to `FixedLogGrid`, so
  trapezoidal runs eager — 9.6 s/epoch vs 8.6, immaterial.

---

## 16. Why the peers beat the reflection's own geometry

The design doc originally motivated the leave-one-out context as a workaround
for missing geometry (``--fix-profile-radius=4e7`` makes ``profile_radius``
constant). That undersells it. The peers are a **better measurement than ``j``'s
own geometry can be**:

* ``j``'s excitation error is computed from a **refined orientation that is
  itself in error**, so the nominal offset is a corrupted input, not merely an
  incomplete one. This is why ``spearman(log_K, ewald_offset)`` is only 0.100.
* The offset carries **no information about the pulse spectrum**, which at SASE
  jitters shot to shot and directly sets partiality.

The other reflections *measure the crystal's state*: which of them came out
strong or weak, given where they nominally sat, constrains the orientation error
and the spectral content — and those are what actually determine ``j``'s
partiality. The context is therefore a **regression over the peer set**: fit the
crystal's scale function from ``(geometry, observed scale)`` pairs, evaluate at
``j``.

### `obs_scale` — the channel that carries partiality

Raw intensity conflates "intrinsically strong reflection" (I0) with "well
positioned reflection" (partiality). Dividing by the reference merge separates
them, with no K in it so no circularity:

    obs_scale_i = n'_i / I0_{h(i)}  =  K_i . S_i

Linear, so the 32.4 % negative observations survive. Added to
`CONTEXT_ONLY_FEATURES`; the reference is refreshed from the target's EMA each
epoch, and **the normalisation statistics must be recomputed with it** — frozen
stats against a stale reference dropped the synthetic recovery from rho = 0.96
to 0.11.

Result on the real dark (beta = 1 both arms):

| | without `obs_scale` | with `obs_scale` |
|---|---|---|
| median w | 1.384 | **1.146** |
| within_frac | 0.098 | 0.020 |
| between_frac | 0.902 | 0.980 |

**median w = 1.146 clears the 1.61 acceptance target** — the plan's primary
falsifiable metric is met. But `within_frac` went *down*: the gain is entirely
crystal-level. Mean pooling can sharpen "how bright is this crystal"; it cannot
do the per-reflection inference, because a single pooled mean destroys the
locality the regression needs.

### The token/attention formulation

One embedding per reflection covering everything (hkl, geometry, Ewald offset,
detector position, intensity, sigma). For ``j``'s own prediction the query copy
has its measurement channels zeroed **and the attention diagonal is masked**.

Both halves are required. Zeroing the query alone is not enough: if ``j``
attends to itself its intensity returns through the *value* and the objective
collapses to the tautology ``K_j = n'_j / I0_j``. With the diagonal masked,
``j``'s geometry still enters as the query — which is how it localises — while
its intensity cannot reach its own output by any path.

This subsumes Deep Sets (mean pooling is attention with uniform weights and a
masked diagonal) and moves the leave-one-out contract from "two divergent
feature lists" into the mask. The constructor guard therefore relaxes under
`token_mode` and is **re-asserted at every `build()`** instead.

`PeerAttentionEncoder` satisfies the existing `ScaleEncoder` protocol, so it is
a one-argument switch: `AmortizedScale.from_prep(..., encoder="attention")`.
Cost on the real dark: 26 s/epoch at 400 k observations per batch, against
9.6 s for Deep Sets at 1.5 M — attention is O(L^2) so the batch is smaller.

### Monitor correction

`hkl_permutation_sensitivity` now permutes **only the self branch**. Once the
context carries `obs_scale = n'/I0`, permuting the whole meta also permutes the
*peers'* I0 lookup, and the monitor then reports legitimate use of
per-reflection I0 as leakage — it read 0.33 on a model with no hkl feature at
all. The degeneracy that matters is whether `log_K[j]` responds to `j`'s *own*
reflection identity, which is a self-branch property. `AmortizedScale.
predict_split(meta_self, meta_ctx)` exists for this.

---

## 17. Marginalising I0 instead of fitting it

The leave-one-out target is the argmin of a **Gaussian WLS surrogate** while the
objective is the **GenGauss marginal**.  Different optima — and that mismatch is
the *only* reason detaching the target is approximate.  Were I0 the argmax of
the same objective, `dL/dI0 = 0` and the envelope theorem would make detaching
exact.  `PosteriorMergeTarget` removes the mismatch from the other end: it never
fits I0.

### Geometry

On the log-intensity axis each observation is a bump.  `log K_j` **slides** it
(a multiplicative scale is a *translation* in log space — this is why the whole
model is parameterised in logs) and `alpha_j` **blurs** it.  Merging is sliding
every observation of a reflection into a common frame and asking how well the
bumps stack; the marginal likelihood is the **area under their product**.

Because the GenGauss is normalised, blurring also *flattens* — so agreement
cannot be bought by widening the scatter.  That is the geometric form of the
`-log alpha` normaliser, and it is what the grid-truncation bug was defeating.

Laplace on the integral gives `log Z ~ log(peak) - 1/2 log j`, so marginalising
is profiling *plus* the Cox-Reid term already implemented in
`_ChunkedDataNLL_REML`.  The `-1/2 log j` is an Occam factor: you are charged
for how much freedom fitting I0 consumed.

### What makes it affordable

1. **Individual observation curves are broad; only their product is narrow.**
   `f_j(mu)` has the width of one observation's uncertainty; the posterior is
   ~sqrt(n) narrower.  So evaluate `f_j` **coarsely** and interpolate.
   Measured requirement: ~25-33 coarse points (rel. err 5.6e-3 at 25, 8.2e-4 at
   33) — which is why `merging/posterior.py` uses `K_coarse = 31`.
2. **Linear interpolation commutes with the sum over observations**, so
   accumulate on the coarse grid *first* — `(n_hkl, K_coarse)` — and interpolate
   that.  The `(n_obs, M_fine)` tensor never exists.
3. **Triton never materialises the `(M, K_inner)` integrand** (it saves six
   tensors of size M), and the inner node count is nearly free (K 501 -> 2001
   costs +4 %).  So a **wide** `FixedLogGrid` gives an untruncated inner
   integral at almost no cost — measured in-grid mass 1.000000 up to w = 6.75,
   against 0.94 at w=1.65 on the narrow grid.

Measured on an A100: full step 836 ms, one inner quadrature 36.5 ms (**4.4 %**
of a step), so 31 outer passes is ~**2.3x** the step.  The wide Triton grid is
itself *faster* than trapezoidal (7.3 vs 10.4 s/epoch), so the net landing point
is ~17 s/epoch.

**Future work:** the 31 outer evaluations each re-launch the kernel and re-read
`n`, `bg`, `alpha`, `beta`.  A fused kernel looping the outer grid internally
would read those once.  Since the kernel is memory-bound (K 501 -> 10001 costs
only +26 %), that should recover most of the 31x.

### What is and is not differentiable

Differentiable: the integrand at every node, the accumulation, the coarse->fine
interpolation, the `logsumexp`, the posterior mean.  Verified — the evidence
loss reaches the encoder and both heads.

**Detached: the grid geometry only** (centre and width).  For a converged
quadrature the answer does not depend on where the nodes sit, so any gradient
through node placement is *discretisation error* — propagating it injects noise.
That argument holds only while the grid covers the mass, which makes
`edge_mass` the **validity condition for the detachment**, not merely a quality
check.

### Grid placement was the hard part

Not resolution — **width**.  The p90 error was flat in node count
(0.543 / 0.512 / 0.514 at 17 / 33 / 65 points): ~10 % of reflections had >50 %
error because the grid did not cover their posterior at all.  Gauss-Hermite was
*worse* than the trapezoid for the same reason — it commits harder to the
assumed sigma, and the Laplace sigma (from the Gaussian variance model) is wrong
for weak reflections whose posteriors are wide and skewed.

Measuring the width instead of predicting it works, but a naive follow-the-
measurement loop **oscillates** (edge_mass 0.29 -> 0.03 -> 0.52 -> 0.01 -> 0.53):
a clipped grid under-reports its own width, the correction overshoots, the now
wide grid measures a small sd, and it clips again.  A **ratchet** — widen hard
when clipping, shrink only slowly when not — settles: edge_mass max 0.927 ->
~0.011 by the second pass.

### Config

`merge_target="posterior"` + `loss="evidence"` (guarded: evidence requires the
posterior target).  With the evidence loss there is no per-observation target at
all, so no leave-one-out bookkeeping is needed on the grid — what protects it is
the **feature contract** (the network could align every bump by setting
`K_j ~ n'_j`, and cannot only because it never sees `n'_j`), not target exclusion.

---

## 18. Why the merge must be an integral, not an estimator

Measured at **K = 1** — no scale model at all, same stream, same protocol, same
split — purely changing the estimator:

| merge | CC1/2 (full data) | R_split | shallow val |
|---|---|---|---|
| WLS, `sum(K n'/V) / sum(K^2/V)` | 0.9871 | 0.0682 | 0.79 |
| **posterior mean** | **0.9944** | **0.0591** | **0.93** |

### The general reason

Every point estimator encodes an assumed distribution, and none of the
available shapes fits this data:

* **arithmetic mean** — additive, symmetric error.
* **inverse-variance weighting** — Gaussian with *value-independent* variance.
  CrystFEL's sigma is derived from the measured I, so the weight correlates
  with the value and systematically upweights downward fluctuations at low
  I/sigma.  Measured cost: outer-shell CC1/2 **0.622** with `1/sigma^2` weights
  against **0.880** with near-uniform weights, on identical data at K = 1.
* **geometric mean / log space** — lognormal, and discards the 32.4 % negative
  observations.
* **WLS with multiplicative scatter** — degenerates.  With
  `V = sigma^2 + (K I0)^2 (e^{w^2}-1)` and the scatter term dominating
  (measured `var_frac_scatter = 0.99`), the weights cancel:

      a_j = K n'/(K^2 I0^2 c) ,  b_j = 1/(I0^2 c)  =>  I0 = (1/n) sum_j n'_j/K_j

  an unweighted mean of **data divided by the scale** — the one thing the
  design set out to avoid, arrived at not by writing it but by the estimator
  collapsing into it.  With 2.9 % of observations below `log K = -5`, those
  terms are amplified ~150x.

The true distribution is a convolution: a positive latent, multiplicative
GenGauss scatter, observed through Poisson detection with background
subtraction that makes negatives routine.  There is no closed-form sufficient
statistic, so there is no estimator to choose.  And the shape is
**heterogeneous per observation** — alpha, beta, sigma and K all differ — so
even the best single choice is wrong for most of the data.  Integrating handles
that automatically, and yields `sigI` as the posterior width rather than as a
propagated error model carrying the same wrong assumption.

### Process note

This was resisted for a long time in favour of patching the WLS estimator: the
mean/median (`E[exp(eps)]`) correction, a float32 cancellation guard, and the
weighting.  Each patch fixed one shape assumption and left the next standing.
The pattern is worth recognising early — repeated local fixes to an estimator
are evidence the estimator is the wrong object, not that it needs another fix.

### What made it practical

Three failed *formulas* for the integration grid width (too narrow, edge_mass
0.93; an oscillating follow-the-measurement loop, 0.29/0.03/0.52; too wide,
sigma pinned at its bound so the merge returned the grid centre and the data
contributed nothing) were replaced by **measuring the curve**: sweep a generous
grid once, then read the peak and curvature off it, `s = 1/sqrt(-S'')`.  No
distributional assumption, nothing that can pin at a bound.  Resulting
`edge_mass` ~2e-13.
