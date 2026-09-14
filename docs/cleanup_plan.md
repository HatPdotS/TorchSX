# Deletion plan

Goal: get `src/torchsx` into a state a reviewer can read end to end.  The
package is 30,454 lines of source (excluding tests) in 130 modules, plus
~9,800 lines of tests.  This plan removes **9,184 lines (30 %) by dead-code
elimination alone**, leaving a 21,270-line core in 67 modules, and then lists
the further cuts that need a decision rather than an analysis.

Nothing here changes numerical behaviour of `torchsx-train` or
`torchsx-merge-nn`.  Stages 1–2 are provably behaviour-preserving; stages 3–5
are not, and each names what it changes.

---

## 1. What "necessary" means here

Two definitions were considered and one rejected.

*Rejected:* "reachable by import from anywhere in the package."  This marks
118 of 130 modules live, because `merging/__init__.py`, `merging/quadrature/__init__.py`
and the other package `__init__` files eagerly re-export every variant.
Importing `torchsx.merging.config` executes `merging/__init__.py`, which imports
`Merger`, which imports the whole LBFGS engine.  The measure is useless.

*Adopted:* "reachable from the two commands that were actually executed."
Across 314 recorded `argv` invocations in `experiments/`, exactly two torchsx
entry points appear: `python -m torchsx.cli.train` (28×) and
`python -m torchsx.cli.merge_nn` (33×).  `torchsx-merge`, `torchsx-merge-global`,
`torchsx-stats` and `torchsx-filter-ewald` were never run, in any experiment,
under either the module or the console-script spelling.

Reachability was computed with a symbol-resolving closure: `from pkg import Name`
resolves `Name` through `pkg/__init__.py` to its defining module and pulls in
*only* that module, not the package's other re-exports.  Every borderline case
below was then confirmed by hand — a name with 1–2 apparent references almost
always turned out to be a docstring mention, not a call.

**Caveat on the figure campaigns.** `figures/README.md` in the paper repo costs
six main figures.  None of them invokes `torchsx-merge`, `torchsx-merge-global`
or `torchsx-stats`; the unweighted baseline in Fig 1 comes from partialator, not
from `merge_global`.  If that changes, `merge_global` moves from stage 1 to the
deferred list — it is the only stage-1 item with a plausible claim on the paper.

---

## 2. Discovery schedule, by job

The audit was run in this order and the plan is organised the same way.  The
order matters: each stage makes the next one measurable.

| # | Job | Modules | Status |
|---|-----|---------|--------|
| 1 | **Entry points** — what is actually invoked | `cli/` | done |
| 2 | **Data loading** — stream → `PreparedObs` | `io/`, `stream.py`, `_core.py`, `levels.py`, `measurement.py`, `geometry.py` | done, one open question (§5.3) |
| 3 | **Neural network** — features, encoder, heads | `nn/` minus training loop | done |
| 4 | **Merging and loss** — likelihood, quadrature, scatter, priors | `merging/` | done, largest cut |
| 5 | **Training loop and readout** — checkpoints, metrics, merge targets | `nn/train_amortized.py`, `nn/*_target.py`, `nn/metrics.py` | done |
| 6 | **Writing out corrections** — MTZ/HKL products, calibration | `cli/_pipeline.py`, `io/_writeback.py`, `merging/calibrate.py` | done, one open question (§6) |
| 7 | **Tests** — follow the source | `tests/` | done |
| 8 | **Prose** — docstrings and comments | all | scoped, not yet applied |

---

## 3. Stage 1 — explicit deletions (1,973 lines + two methods)

Files with no importer reachable from `train` or `merge_nn`, and no recorded
invocation.

| Lines | Path | Evidence |
|------:|------|----------|
| 332 | `cli/merge.py` | never invoked; the "May-2026 recipe" LBFGS CLI |
| 271 | `stats.py` | only importer is `cli/stats.py` |
| 269 | `nn/consistency_loss.py` | `ScaleConsistencyNLL` appears only in docstrings of `nn/features.py` |
| 480 | `nn/features.py` | `ReflectionDataset` appears only in a docstring of `nn/obs_features.py`; superseded by `nn/obs_features.py` |
| 162 | `nn/batch.py` | `StreamBatcher`/`GroupedBatch` appear only in docstrings of `nn/features.py` and `nn/deepsets.py` |
| 160 | `cli/merge_global.py` | never invoked (see caveat in §1) |
| 113 | `cli/filter_ewald.py` | never invoked; the Ewald-offset filter now lives in `merging/observations.py` |
| 99 | `merging/gating/background_significance.py` | no reference outside its own package `__init__` |
| 78 | `cli/stats.py` | never invoked |
| 9 | `merging/gating/__init__.py` | package becomes empty |

Plus two methods on `SFXStream` — these are the keystones:

| Lines | Location | What it holds alive |
|------:|----------|---------------------|
| ~4 | `stream.py:900` `SFXStream.merge()` | `from torchsx.merging import Merger` → `merger.py`, `posterior.py`, `targets/`, `regularizers/`, `priors/wilson*`, `result.py`, `scale/*`, `scatter/gamma` |
| ~76 | `stream.py:909` `SFXStream.make_loss()` | `from torchsx.merging.loss import LogSpaceLoss` → `loss.py` and the same subtree |

`SFXStream.merge()` has **no caller anywhere** — not in the package, not in any
experiment harness, not in tests.  `make_loss` has exactly one: `cli/merge.py:247`,
deleted above.  Between them these ~80 lines are the sole reason the entire
legacy LBFGS merging engine is still reachable.

---

## 4. Stage 2 — cascade (7,211 lines, 53 modules, automatic)

Deleting stage 1 makes all of the following unreachable.  No judgement is
involved; this is the transitive closure.

**The LBFGS merging engine (3,058 lines).**
`merging/posterior.py` (1,094), `merging/merger.py` (864), `merging/loss.py` (726),
`merging/result.py` (32), `merging/__init__.py` (52), plus
`merging/targets/` entirely (6 files, 491) — `DataNLLTarget`, `IntensityPriorTarget`,
`RegularizerTarget`, `ScaleRegTarget` and their base.

**Alternative quadratures (2,207 lines).**
`precomputed.py` (565), `laplace_saddle.py` (359), `hierarchical_mc.py` (276),
`adaptive_hybrid_log.py` (153), `hybrid_log.py` (145), `mixed_delta.py` (139),
`wide_log_grid.py` (131), `adaptive_signif.py` (128), `moment_matched.py` (124),
`delta_at_es.py` (97), **`trapezoidal.py` (156)**, and `quadrature/__init__.py` (27).

> `trapezoidal.py` was on the keep list in the earlier audit.  That was wrong:
> its only importers are `posterior.py` and `merger.py`, both of which go.  The
> production path uses `SplitLogGrid` (training, via `cli/_pipeline.py:815`) and
> `FixedLogGrid` (readout, via `nn/objective.py:92` and `nn/posterior_target.py:499`)
> and nothing else.

**Priors and regularizers (748 lines).**
`priors/wilson.py` (120), `priors/wilson_aniso.py` (125), `priors/wilson_match.py` (126),
`priors/flat.py` (21), `priors/base.py` (28), `priors/__init__.py` (22);
`regularizers/` entirely (7 files, 360) — `MillerGraphSmoothness`, `GaussianAnchor`,
`SigmaCauchy`, `ScaleBMeanRegularizer`, `PoissonNullShrinkage`.
`priors/reciprocal_basis.py` (266) **survives** — it is the live Wilson prior,
used by `nn/posterior_target.py:390` and `nn/train_amortized.py:1647`.

**Scale and scatter variants (444 lines).**
`scale/per_crystal.py` (78), `scale/panel.py` (73), `scale/composite.py` (58),
`scale/fixed.py` (37), `scale/__init__.py` (23);
`scatter/gengauss_recentered.py` (103), `scatter/gamma.py` (53), `scatter/__init__.py` (23).

**Likelihood variants (342 lines).**
`likelihood/gaussian_polynomial.py` (166), `likelihood/poisson_bg_mix.py` (102),
`likelihood/gaussian_bg_mix.py` (74).  `likelihood/poisson.py`,
`likelihood/gaussian_base.py` and `likelihood/base.py` survive.

**Package `__init__` shells (412 lines).**
`merging/hkl/__init__.py` (76), `nn/__init__.py` (54), `io/__init__.py` (31),
`likelihood/__init__.py` (23), `scatter/__init__.py` (23), `scale/__init__.py` (23),
`priors/__init__.py` (22), `targets/__init__.py` (18), `regularizers/__init__.py` (25),
`io/_parse/__init__.py` (9), `torchsx/__init__.py` (9), `cli/__init__.py` (1).

> Rewrite these rather than delete the file where the package must survive.
> **The eager re-export is the reason the codebase resisted auditing**; every
> `__init__.py` that stays should re-export nothing, or at most the two or three
> names the production path actually imports through it.  Doing this first, as a
> separate commit, makes every subsequent stage verifiable by import alone.

### Surviving core after stage 2 — 67 modules, 21,270 lines

| Subsystem | Lines | Largest survivors |
|---|---:|---|
| `merging/` | 7,013 | `observations.py` 797, `scale/amortized.py` 487, `_fixed_log_grid_triton.py` 434, `hkl/noise.py` 419, `fixed_log_grid.py` 366, `split_log_grid.py` 344 |
| `nn/` | 5,519 | `train_amortized.py` 1,726, `posterior_target.py` 735, `obs_features.py` 485, `pooling.py` 460, `heads.py` 398, `deepsets.py` 379 |
| `io/` | 4,188 | `_parse/orchestrator.py` 589, `_excise.py` 572, `_writeback.py` 482, `times.py` 439 |
| top level | 2,911 | `stream.py` 990, `geometry.py` 744, `_core.py` 604 |
| `cli/` | 1,639 | `_pipeline.py` 989, `merge_nn.py` 376, `train.py` 274 |

---

## 5. Stage 3 — config-dead branches inside survivors

These are live code by import but unreachable under every configuration ever
run.  Each needs a one-line decision, not an investigation.

### 5.1 CLI flags never passed (31 of 70)

`--beta-init`, `--bg-var-factor`, `--bg-var-mode`, `--calib-degree`,
`--crystals-file`, `--free-frac`, `--free-from`, `--free-seed`, `--gauge-anchor`,
`--lam0`, `--lam-halflife`, `--locate-windows`, `--max-crystals`,
`--max-halfwidth`, `--max-obs`, `--merge-split`, `--miss`, `--no-friedel`,
`--no-half-datasets`, `--no-scale-calibration`, `--no-skew`, `--out`,
`--posterior-prior`, `--sfx-cache`, `--skew`, `--split-names`, `--unity`,
`--warmup`, `--wilson-basis`, `--wilson-degree`, `--w-init`.

**Split these into two groups before touching anything.**  A flag that was never
*passed* may still have a *default that every run used* — removing the flag is
then a cosmetic change, but removing the code behind it changes behaviour.
`--free-frac`, `--free-seed`, `--no-half-datasets`, `--no-scale-calibration` and
`--calib-degree` are in that category and only lose the flag, with the default
inlined.  `--unity`, `--miss`, `--skew`/`--no-skew`, `--w-init`, `--beta-init`,
`--gauge-anchor`, `--wilson-basis`, `--wilson-degree`, `--locate-windows`,
`--max-halfwidth`, `--posterior-prior` are exploratory knobs and lose both.

**`--bg-var-mode` is unconditional.** Its only non-default value reaches
`merging/quadrature/split_log_grid.py:303`, which is
`raise NotImplementedError("second_order background mode is not supported")`.
The flag can only crash.  Remove flag, branch and the raise.

### 5.2 Dispatch branches with no live selector

- `merging/merger.py` and `merging/loss.py` dispatch tables go with stage 2.
- `merging/config.py` is the LBFGS-era `MergeConfig`, still imported by
  `cli/_pipeline.py:173`.  It carries fields its own comments call dead:
  `reg_scale_ref_obs` ("DEPRECATED … no longer consulted"), `lam_I0_floor` /
  `log_I0_floor_target` ("kept for back-compat; new runs set to 0"), the four
  `adaptive_*` fields ("settings for AdaptiveHybridLogGrid experiments" — deleted
  in stage 2), and the `lam_*_smooth` / `lam_*_anchor` / `lam_sigma_cauchy` /
  `lam_frac_wilson` / `lam_wilson_match` / `lam_wilson_aniso*` weights, which only
  ever fed regularizers and priors that stage 2 removes.  Audit which fields
  `_pipeline.py` and `observations.py` actually read and delete the rest; the
  dataclass should shrink from 174 lines to the observation-filtering, shell-setup,
  gain and posterior-grid fields.
- `merging/scale/amortized.py:165` accepts `encoder in ("deepsets", "attention")`.
  `"attention"` is not plumbed to any CLI flag and no checkpoint selects it.
  Cutting it removes `nn/attention.py` (207) and `tests/test_peer_attention.py` (94),
  and simplifies `nn/obs_features.py`'s two-mode token/flat feature build.
  **Decision needed** — this is a documented design claim ("attention subsumes
  Deep Sets"), not just a stale branch.

### 5.3 Optional data-loading paths

`stream.py` exposes four loaders and two pixel-extraction methods.  Production
uses CrystFEL `.stream` / `.sfx`.  Never invoked in any recorded run:

| Lines | Surface | Note |
|------:|---------|------|
| 572 | `io/_excise.py` via `SFXStream.excise_patches` | needs `h5py`+`bitshuffle`; pixel patches are not in any figure |
| 279 | `io/_roi_store.py` via `SFXStream.excise_rois` | same |
| 259 | `io/_duckdb.py` via `SFXStream.from_duckdb` | probixi ingest; no recorded use |

851 further lines, but this is the one place where "never run in `experiments/`"
is weak evidence — these are ingest paths for other people's data.  **Decision
needed.**  `io/times.py` (439) stays: `--split` was passed 58 times.

---

## 6. Stage 4 — explicitly legacy compatibility surfaces

Distinct from stage 3: these are live, exercised code paths that exist only to
read older inputs.

- `io/_safetensors.py:140` `_load_legacy` — pre-columnar flat-key `.sfx` reader,
  ~35 lines plus `_LEGACY_TENSOR_FIELDS` and the `_legacy_hkl_state` flag
  threaded through `_core.py`.  Removable once no `.sfx` in the campaign inputs
  is pre-columnar; check before cutting.
- `measurement.py:172` `legacy_stream_gain` — "legacy geometry (unverified)" gain,
  used by `merging/observations.py:501` when a source declares no measurement.
  This one is a correctness surface, not a convenience: it silently substitutes an
  unverified gain.  Keeping it is defensible; keeping it *silent* is not.
- `_core.py` "legacy keyword constructor" (`ShoeboxStreamCore`, `__post_init__`,
  the field→level routing table) — the pre-`levels` construction path, still used
  by `_load_legacy`.  Dies with it.
- `cli/_pipeline.py` `legacy_cache` branch — now unreachable, because the new
  `observation_filter_version` gate raises `SystemExit` before it for any cache
  lacking the key.  Delete the branch (see the validity-fix review).

---

## 7. Stage 5 — tests

Delete outright (1,616 lines), their subject is gone:

`test_merging_priors.py` (307), `test_consistency_loss.py` (279),
`test_merging_posterior.py` (267), `test_merging_priors_aniso.py` (251),
`test_merging_chunked_data_nll.py` (184), `test_nn_features.py` (144),
`test_stats.py` (118), `test_batch.py` (66).

Edit, do not delete — they test surviving behaviour but import a deleted name:

`test_cli_pipeline.py` (`cli.merge_global`), `test_duckdb_load.py`
(`from torchsx.merging import …`), `test_bg_mean_offset.py`
(`quadrature.moment_matched`), `test_amortized_scale.py`,
`test_amortized_training.py`, `test_merging_layers.py`, `test_merging_scale.py`,
`test_measurement_sources.py`, `test_hkl.py`, `test_posterior_smooth_sigma.py`
(all reach a deleted module through a package `__init__`).

Conditional on §5.3: `test_excise.py` (263), `test_roi_store.py` (176),
`test_duckdb_load.py` (595).  Conditional on §5.2: `test_peer_attention.py` (94).

---

## 8. Stage 6 — prose pass

28 % of the source is docstrings and comments: **8,627 of 30,454 lines**
(`merging/` 30 %, `nn/` 37 %, `io/` 20 %, `cli/` 15 %).  Much of that is
narrative history, which the rule here forbids: **comments describe what the code
does and why it is correct; they do not record what it used to do.**

Three things to remove, in decreasing order of how much they mislead a reader:

**(a) Result numbers inside source.**  These date the file and go stale silently.
- `cli/train.py:11` — "SwissFEL small molecule R_free 0.119 (partialator 0.147),
  SACLA rhodopsin 0.276 …".  Both predate the Ewald-offset and background-mean
  fixes and are no longer reproducible from this build.
- `cli/merge.py:4` — "validated on tubulin (T7, R-free = 0.2285) and BR
  (R-free = 0.180, 1.24 pp better than partialator unity)".  File is deleted anyway.
- `nn/posterior_target.py:179` — "27 % of reflections sat on such a grid; R_free …";
  `:335` — "collapsed to ~0 and R_free was 0.351 against 0.322".

**(b) Changelog prose.**  Narrates a past state of the code.
- `cli/_pipeline.py:6` — "The bench scripts each grew their own copy of this; the
  copies drifted, and one of them merged a fold-encoder checkpoint under the
  default encoder geometry and silently lost 0.022 in R_free."
- `cli/merge_global.py:10` — the whole "Why fit rather than fix" paragraph
  ("A frozen `w = 1.65` silently absorbs … median F down 40 %, F/sigF halved …
  This replaces the manual `w` × background-variance grid").
- `nn/posterior_target.py:283` — "the merge does, and previously called `nodes()`
  instead, which still used the old closed-form width."
- `nn/train_amortized.py:774` — "It previously aliased `loss_total` here while …";
  `:300` — "Set to a single value to recover the old one-pass behaviour."
- `merging/config.py` — "DEPRECATED", "kept for back-compat", "legacy behaviour",
  and the pointer to `experiments/synthetic_gengauss/integrand_diag/fix_bias`.

**(c) References to experiment directories.**  A source file should not depend on
an `experiments/` path to be understood; those directories are not shipped and
several are already superseded.

Keyword scan (`used to`, `previously`, `no longer`, `formerly`, `legacy`,
`deprecat`, `R_free`, `drifted`, `TODO`/`FIXME`/`XXX`/`HACK`, …) finds 133 hits
across 40 files, worst: `cli/_pipeline.py` 16, `merging/loss.py` 9,
`io/_safetensors.py` 9, `measurement.py` 8, `nn/train_amortized.py` 7,
`nn/posterior_target.py` 7, `_core.py` 7.  The scan **undercounts** — the
`merge_global` paragraph above contains none of those words — so treat it as
a starting list, not the specification.  Do this pass *after* stages 1–2, when
half the prose has already been deleted with its module.

What stays: statements of invariant and rationale that are still true, e.g.
`cli/_pipeline.py`'s "**No intensity column in an MTZ**" contract and
`merging/observations.py`'s alignment refusal.  Rewrite those in the present
tense without the incident that motivated them.

---

## 9. Ledger

| Stage | Source | Tests | Behaviour change | Gate |
|---|---:|---:|---|---|
| 1 explicit | 1,973 + ~80 | — | none | none |
| 2 cascade | 7,211 | — | none | none |
| **subtotal** | **9,184** | | | |
| 5 tests | — | 1,616 | none | follows 1–2 |
| 3.1 flags | ~200? | — | removes unused knobs; defaults inlined | list split first |
| 3.2 `MergeConfig` | ~100 | — | none if fields are truly unread | field-by-field audit |
| 3.2 attention | 301 | 94 | drops a documented design option | **decision** |
| 3.3 duckdb/excise/ROI | 851 | 1,034 | drops three ingest paths | **decision** |
| 4 legacy readers | ~200 | — | drops pre-columnar `.sfx` | check inputs |
| 6 prose | ~1,000? | — | none | after 1–2 |

Floor: 21,270 lines.  With every decision resolved toward deletion: ~19,000,
from 30,454 — a 38 % reduction, in 60-odd modules.

---

## 10. Verification

Stages 1–2 claim exact behaviour preservation.  The check is cheap and should be
run as the gate on each commit:

1. `python -c "import torchsx.cli.train, torchsx.cli.merge_nn"` — no ImportError.
2. `pytest src/torchsx/tests` — after stage 5, green.
3. A short `torchsx-train` run (2 global steps, the existing pilot config) and a
   `torchsx-merge-nn` readout from the resulting checkpoint, before and after,
   with **bit-identical** MTZ/HKL output.  Anything that moves means a stage-1 or
   stage-2 claim was wrong.
4. `--trust-cache-prep` against one of the existing prepared caches, to confirm
   the cache signature path still resolves.

Step 3 is the real gate; steps 1–2 only catch import breakage.

---

## 11. Suggested commit order

1. `__init__.py` de-export (§4, "Package `__init__` shells").  No deletions.
   Makes everything after it verifiable by import.
2. `stream.py`: drop `merge()` and `make_loss()`; `cli/merge.py` delete.
3. Cascade: the LBFGS engine, `targets/`, `regularizers/`, `priors/wilson*`.
4. Cascade: alternative quadratures, scale/scatter/likelihood variants.
5. Remaining stage-1 files (`stats`, `filter_ewald`, `merge_global`, `gating`,
   `nn/features`, `nn/batch`, `nn/consistency_loss`).
6. Tests (§7).
7. `--bg-var-mode` and the `second_order` raise (§5.1).
8. Everything else, once decided.
