# Changelog

## v0.2.3

- torch 2.7 to 2.14 instead of 2.9 only: on 2.11 to 2.13 the fold encoder compiles with inductor's coalescing analysis and mix-order reduction off, the two passes that fail on it.
- The fold encoder is compiled on its first block before training or merging; where torch cannot compile it, it runs uncompiled with a warning, and `training_summary.json` records `compiled`.

## v0.2.2

- Pin torch 2.9, the version the compiled fold encoder is known to train with. 0.2.1 only excluded torch 2.13, so it installs torch 2.12.1; torch 2.11 fails as well. On 0.2.1, `pip install "torch==2.9.*"` pins it by hand.

## v0.2.1

- Exclude torch 2.13: its compiler fails on the fold encoder's backward pass (`InductorError: CantSplit ... not divisible by ...`) on any GPU. Without the upgrade, `TORCHSX_PEER_COMPILE=0` avoids it, at some speed.

## v0.2.0

Nothing is backwards compatible with 0.1.0: retrain, and rebuild prepared caches.

- Fold transformer scale encoder with per-crystal geometry correction (#9, #16)
- Sum-of-spots tether: merged intensities follow the sum of the spots at every resolution (#12)
- Pseudo-translation search on by default: structured Wilson prior and phase classes for tNCS and lattice translocation defects (#14)
- Mean of the last updates when training oscillates (`--average-last`, #15)
- Four-component Gaussian-mixture scatter model (#10)
- probixi DuckDB runs as input (Ryan O'Dea, #7)
- On-disk prepared store: training and merging read batches from disk (#3)
- `torchsx.merge-stats` takes run directories and reports flat CC1/2 (#4, #11)

## 0.1.0 (2026-10-12)

Initial public release.
