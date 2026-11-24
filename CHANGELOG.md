# Changelog

## Unreleased

- `--warmup` counts updates of the run, not updates per state: the default is 25 updates however many states there are. Before, the warm-up was 25 updates times the number of states while `--steps` is the total over all states, so it took a growing share of the run, and all of the default 400 updates from 16 states on, where the learning rate never reached its full value. Runs with one state are unchanged; with several states the warm-up is shorter, and `--warmup N` gives the old length with N = 25 × states.
- The flat CC1/2 of `torchsx.merge-stats` takes each reflection's local trend from a plane fitted through its 32 nearest reflections in reciprocal space, instead of the mean of its 32 nearest in direction within a fine resolution shell. Values change. Halves that share a prior and agree to a few percent now read close to zero in every shell, the lowest and the highest included, and with reflections missing; before, they read 0.3 to 0.7. Where the halves carry signal the values move by a few hundredths.

## v0.2.4

- `--prior none` is removed (#24). Runs with the default Wilson prior are unaffected. The flat prior was never a usable option: its posterior has no upper bound, so for weak reflections it could run off the top of the integration grid, with F at the grid ceiling and a standard deviation near zero. `--prior` accepts only `wilson` and stays for existing scripts. `torchsx.merge` refuses a checkpoint trained with `--prior none`.
- TorchRef 0.8 (`>=0.8.1,<0.9`). torchsx's use of it is unchanged: 0.8 drops the `verbose` arguments of its readers and of `ReflectionData`, which torchsx never passed, and reports through `logging` instead of `print` and `warnings`. The one visible difference is that writing the merged MTZ no longer prints TorchRef's "✓ Wrote … MTZ" summary; torchsx's own summary stays.
- A state can come from several streams: `--split` names repeat, a `--split` path may be a glob pattern, and several unnamed streams are one data set.
- `--split-file` labels the frames of any number of streams, and an event of `*` names every frame of a file.
- Patterns a split file assigns to no state no longer stop the run: by default they train the network, with their share of the updates, and are merged into nothing; `--unassigned drop` or `error` chooses otherwise.
- The prepared store carries no states; they are assigned when it is opened. Other state names, another split file or another `--unassigned` reuse the store, so `torchsx.merge` can merge a run's data into new states without preparing it again. Prepared caches of earlier versions are rebuilt.
- A frame in several states is stored once and is a member of each, instead of being prepared once per state.
- The merge's report of the sum-of-spots tether is opt-in (`--tether-report`, off by default). It fits all of a state's observations at once, at about 100 bytes of device memory each, and set the GPU peak of large merges; the merged intensities never depended on it.
- The compile dry run of the fold encoder holds one pass at a time, and an evaluation on flex attention dry-runs only the checkpointed pass it uses. Before, the dry run of the first evaluation held a plain pass with its autograd graph while it ran the checkpointed one, and set the device memory peak of training.
- `SFXStream.completeness()` and `.multiplicity()` work again: they imported a TorchRef module that does not exist and failed on every call. Both now count unique reflections in the stream's space group with Friedel mates merged, whether or not the stream was symmetrized; completeness leaves out systematic absences, and the shells are equal in `1/d^3` over the observed resolution range in the median cell.

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
