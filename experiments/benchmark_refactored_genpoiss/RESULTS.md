# Benchmark: unified LBFGS merge vs EM Poisson + partialator

Mirrors `experiments/sfx_scan/benchmark_final/` with one extra method:
**EM Unified** (`torchsx.merging_clean.Merger`) — a single LBFGS
minimization of the Poisson-marginal GenGauss NLL with explicit
Laplacian and ridge regularizers in place of the old D/S alternation.

## Run order

```bash
# 1. Closure-only sanity + memory/timing sweep (~10 min on A100)
sbatch experiments/benchmark_refactored_genpoiss/bench_closure.py

# 2. Full fit across all 5 methods (~1 h on A100)
sbatch experiments/benchmark_refactored_genpoiss/fit.py

# 3. Evaluate (CPU, ~30 s) — produces CSV + plots
experiments/benchmark_refactored_genpoiss/evaluate.py
```

## Data

- 3 negative-delay dark streams: `delay_-91`, `delay_-136`, `delay_-182`
  (subset of the 4 streams in `benchmark_final/`).
- Seed 42, split-half via `SFXStream.split_random(2, seed=42)`.

## Methods compared

| Prefix | Display name | Source |
|---|---|---|
| `unified_` | EM Unified | `torchsx.merging_clean.Merger` — single joint LBFGS |
| `em_` | EM Poisson | `torchsx.merge.fit_em` (alternating D/S, baseline) |
| `gg_` | GenGauss | `torchsx.merge._distribution.fit_gengauss` (log-space only) |
| `part_unity_` | Part. unity | CrystFEL `partialator --model=unity` (pre-computed) |
| `part_xsphere_` | Part. xsphere | CrystFEL `partialator --model=xsphere` (pre-computed) |

## Success criteria for EM Unified

- CC½ ≥ EM Poisson overall and shell-by-shell (no shell drops > 0.005).
- R_split ≤ EM Poisson overall and in every shell.
- Per-HKL I0 CC vs EM Poisson ≥ 0.999.
- Runtime within 2× of `fit_em` on the same GPU.

If any of these fail, tune penalty weights / chunk size / LBFGS settings
(see `MergeConfig` defaults) before declaring the new pipeline a
replacement for `fit_em`.

## Outputs

- `{method}_{half1,half2,full}.pt` — per-method results.
- `benchmark_refactored_genpoiss.png` — CC½ / R_split vs resolution.
- `scatter_self_consistency.png` — F-scatter per method.
- `scatter_alpha.png` — α vs resolution.
- `benchmark_refactored_genpoiss_shells.csv` — per-shell table.
- `bench_closure.out`, `fit.out` — SLURM logs.
