# Changelog

## 0.2.0 (2026-11-04)

Nothing is backwards compatible with 0.1.0: retrain, and rebuild prepared caches.

- Fold transformer scale encoder with per-run geometry correction (#9, #16)
- Sum-of-spots tether: merged intensities follow the sum of the spots at every resolution (#12)
- Pseudo-translation search on by default: structured Wilson prior and phase classes for tNCS and lattice translocation defects (#14)
- Mean of the last updates when training oscillates (`--average-last`, #15)
- Four-component Gaussian-mixture scatter model (#10)
- probixi DuckDB runs as input (Ryan O'Dea, #7)
- On-disk prepared store: training and merging read batches from disk (#3)
- `torchsx.merge-stats` takes run directories and reports flat CC1/2 (#4, #11)

## 0.1.0 (2026-10-12)

Initial public release.
