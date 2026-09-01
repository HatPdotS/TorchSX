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

## Open: the weak-end floor

Merged `F` cannot approach zero (`min F / median F` = 0.36; partialator reaches
-0.24), and the compression slope at the weak end is 0.65 rather than 1.

Established by ground truth: the background is *estimated* from the 9–11 px
annulus and carries `sigma_b^2 = (n_peak/n_bg)*bg` — 11.6 counts against a
median signal of 7.3. The Poisson marginal takes `bg` as exact, and that error
**rectifies** (an upward fluctuation makes the apparent signal negative, which
`S = K I0 exp(eps) > 0` cannot represent), inflating weak reflections 2.4–3.4x.
It does not average down with multiplicity.

Ruled out by measurement — do not re-litigate: the Wilson prior (it pulls weak
reflections *down* 11 %, and the prior-free ML estimate still floors); outer-grid
truncation (fixed, slope moved 0.651 -> 0.677 only); positivity of `I0 = exp(v)`
(with an exact background, true-zero returns 0.01); global-`w` double-counting
(evidence-optimal `w` is 1.65 either way); an oversized coefficient; and a
crowded annulus (measured: nearest-neighbour spot distance is 150 px median,
**0.0 %** within 18 px, effective `n_bg` = 128.0 — this cell is *small*, so the
reciprocal cell is large and spots are far apart).

**Still unexplained:** `Poisson(bg_var_factor=n_peak/n_bg)` is correctly sized
and removes the compression exactly (slope 0.562 -> 1.004), yet degrades the
merge, and in training drives CC1/2 from 0.99 to 0.51 while the loss falls
monotonically and `var_z` converges to 1.004. The leading suspicion is that the
correction is a function of the grid point, not of `log_mean`, so where it is
large and positive it adds a plateau that flattens `log p`'s dependence on the
scale. `bg_var_factor` defaults to 0 everywhere; nothing is affected until this
is settled.

## Accuracy validation

`../../../bench/` holds the refinement-based check — CC1/2 measures precision,
R_free measures accuracy. Trained scale **R_free 0.1397** vs unity 0.1648 vs
partialator + French-Wilson 0.1825, identical model, target, cell and free set.
Refinement repeat noise is ~0.001, so differences below ~0.005 are not callable.
