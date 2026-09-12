# CrystFEL ingestion validation — 2026-09-11

The implementation is in `worktrees/torchsx-crystfel-ingest`, branch
`codex/crystfel-ingest`, based on `b2fe17e` plus the existing uncommitted
background-mean correction. The original `src/torchsx` tracked diff and
`test_bg_mean_offset.py` were checked against the starting snapshot and are
unchanged. No experiment checkout or shared Python installation was replaced.

The boundary reuses the parser, level columns, metadata serializer and exact
integer ring-count helper. It adds three small modules: `measurement.py` for
semantics/resolution, `io/sources.py` for dispatch, and `io/_crystfel.py` for
header interpretation. Duplicate CLI radii/gain parsing and format branches
were removed. There is no new tensor container, adapter hierarchy or registry.

Validation used Python 3.11.16, Torch 2.9.1+cu128, CPU with two Torch threads,
and the existing TorchRef checkout. Pytest 9.1.1 and DuckDB 1.5.5 were installed
under `/tmp/torchsx-ingest-testdeps` for testing. The native C++ parser compiled
and passed equivalence checks against the Python reader.

- The broad regression run passed **130 tests**, covering measurement ingestion,
  metadata, levels, persistence, stream operations, Probixi, Ewald offsets,
  CLI/checkpoint behavior and the existing background-mean correction.
- The final focused measurement suite passed **19 tests**, including additional
  multi-lattice/source provenance, calibration-mode and checkpoint-cache reuse
  cases. This suite overlaps the broad regression run.
- Four CUDA tests and one test comparing against an unavailable upstream
  Probixi source were skipped. Two slow historical full-stream tests were not
  run. Three CLI tests needing the unavailable historical stream were instead
  run on a real local subset; all three passed.
- `git diff --check` passed.

The numerical check used the first 64 small-molecule crystals (55 chunks,
11,273 observations) from `input/prototype_10k_dark_20260911/streams/`.
Preparation from the original checkout with explicit radii `7,9,11` and gain
`1` was compared with preparation from the new reader using source metadata.
**All 27 prepared/observation-info tensors matched exactly**, including Ewald
offsets. The synthetic posterior comparison passed at `rtol=1e-5, atol=1e-6`.
Full refinement, accuracy/precision qualification and GPU performance were
not rerun by this ingestion change.

Header inspection of the three prototype datasets gave:

| Dataset | Radii | Declared gain in ADU/photon | Integration declaration |
| --- | --- | --- | --- |
| Small molecule | 7,9,11 | 1 | rings-cen |
| BR | 3,4,7 | 1.00002268 | absent; explicit ring recipe required |
| TD1 | 2,3,6 | 3.996985 | absent; explicit ring recipe required |

The TD1 experimental gain of 43 continues to override source metadata.
Reconciling that disagreement, qualifying automatic background calibration,
effective masked apertures and broader code pruning remain roadmap tasks.
One homogeneous integration recipe is assumed per run; mixed-recipe handling
is outside this implementation. Probixi/DIALS measurement adapters are future
work; the existing Probixi reader remains available.

The portable contract checks are in `tests/test_measurement_sources.py` and
can be run with `python -m pytest tests/test_measurement_sources.py`. The local
real-data comparison driver is `/tmp/torchsx_ingest_probe.py`; local run logs
are `/tmp/torchsx-ingest-regression.log`, `/tmp/torchsx-ingest-contract-final.log`
and `/tmp/torchsx-ingest-real-probe.log`.
