# Changelog

## Unreleased

- `merged_half1.mtz` and `merged_half2.mtz` always hold the same estimator as `merged.mtz`. With `--estimator posterior` they held the pooled merge. `merged.metrics.json` records `split_half_posterior` in place of `split_half_pooled` and `split_half_posterior_shared_prior`.
- `merged.metrics.json` records the plain and the flat CC1/2 of the half files as `split_half_files`, overall and per shell, as `torchsx.merge-stats` computes them.
- The moment estimators (`wls`, `pooled`, `mean`) keep a reflection measured once. They dropped reflections with fewer than two observations, so the `wls` product held fewer reflections than the posterior product. `mean` has no sigma for such a reflection and its amplitude is not written.
- The `wls` and `pooled` variances take the detector variance terms the checkpoint fitted (`--detector-noise`), A + B times the counting variance. Before, only the posterior product used them.
- `--gain auto` with several probixi databases uses the recorded noise scale only where every input records the same one. An input without a scale took the scale of another input. Where the scales differ the log says so before the gain is estimated.
- The log of the enrichment gate counts the lattices without `enrich_p` separately. probixi leaves `enrich_p` empty for a lattice it did not integrate, and the gate drops it.
- An `.sfx` saved from a probixi database is a training cache: it holds the crystals that passed the filter in force when the database was read, and its measurement provenance records that filter (`enrich_gate`, `crystals_where`). `torchsx.train` and `torchsx.merge` use such a file as saved and print the recorded filter. `--enrich-alpha` and `--crystals-where` are refused for it. An `.sfx` that records no filter is taken as unfiltered.
- `torchsx.merge-stats` skips a cross-correlation it cannot compute (merges with no common reflections) with a message, and writes the other statistics.
- The `wls` product reports the sum-of-spots tether only with `--tether-report`, like the posterior product.
- Training and merging read their batches into a few host buffers that the store reuses (`torchsx.merging.store.RecordBuffers`). A buffer returns to the store when the last tensor on it is released, in the thread that releases it. Before, a read-ahead thread allocated each batch buffer and the consumer released it; the mimalloc allocator inside torch (2.13 on aarch64) kept such a buffer resident, so host memory grew by one batch per update (1.5 GB per update at 28 million observations per batch). On hosts with `zone_reclaim_mode` set, an update then took minutes instead of seconds once the NUMA node of the process was full. The merged values do not change.
- The `wls` merge takes the pooled merge, `sum(n - bg) / sum(K)`, for a reflection with more than half of its observations at the scatter limit of the network (a log-scatter variance of 20). The weights of such a reflection gave the plain mean of `(n - bg) / K` with an error far above the counting error, and French-Wilson returned the prior mean of the shell. The log counts these reflections and `merged.metrics.json` records `wls_pooled`.
- `--detector-noise resolution` fits the detector variance terms per resolution bin (ten bins of equal reflection count) instead of per panel; `--noise-terms B` makes it a scale of the counting variance that changes with resolution. The merge inherits the bins from the checkpoint.

## v0.3.0

- TorchSX names the probixi it is paired with: `torchsx.SUPPORTED_DB_SCHEMA` (2, the newest probixi database schema the reader understands) and `torchsx.PROBIXI_VERSION` (the first probixi release that writes it). probixi holds the counterpart (`probixi.SCHEMA_VERSION`, `probixi.TORCHSX_VERSION`). Neither package depends on the other. Where probixi is installed, the reader warns if that probixi writes a newer schema. The error for a database with a newer schema names the writer and the probixi release this reader mirrors. Schema 1 databases read as before.
- `--enrich-alpha auto` reads the second and later lattices of a frame (probixi `--max-lattices` above one) only with `enrich_p` at most 1e-10. The first lattice keeps the alpha the database records. A number given to `--enrich-alpha` applies to every lattice.
- `--gain auto` on a probixi database that records its noise scale (`integration.bg_model_var_scale`, probixi 0.10.3 and later) uses the declared gain times that scale instead of estimating the gain from the integrated sums. Prepared caches made under `auto` are rebuilt. `--gain calibrate` keeps the estimate.
- The merge written to `merged.mtz` and its halves is the `wls` estimator, the inverse-variance mean of `(n - bg) / K` on the trained scales, with French-Wilson amplitudes. `--estimator posterior` writes the Wilson-posterior amplitudes as before.
- The gain calibration whitens each observation with its recorded disk where the source has one (probixi databases). With the nominal disk it returned the detector gain times the ratio of the disk areas for databases whose disk grows with resolution. Prepared caches are rebuilt.
- `--gain auto` takes the calibrated gain whenever a valid calibration exists (it kept a declared gain within 50 % of the estimate). Prepared caches made under `auto` are rebuilt.
- probixi databases are read through the enrichment gate by default: lattices with `enrich_p` above alpha are dropped (`--enrich-alpha auto|off|ALPHA`, `enrich_alpha=` in `load_duckdb`, `SFXStream.from_duckdb` and `SFXStream.read`). `auto` takes `integration.bg_model_var_alpha` when the database records it, else 1e-3. Prepared caches of probixi databases are rebuilt.
- `--crystals-where SQL` reads only the crystals of a probixi database that the predicate accepts.
- A posterior that sits on one grid node or at a grid edge is re-placed on a narrower or shifted grid before the readout; the merge report counts the reflections re-placed and those whose sigF stays below 0.001 of the median.
- With `--estimator posterior`, `merged_half1.mtz` and `merged_half2.mtz` hold the pooled, prior-free merge, and the log and `merged.metrics.json` report the pooled and the shared-prior half statistics.
- `--also posterior|flat|ml|pooled|mean|wls` writes further estimators of the full merge as MTZ and as a table.
- The merge report lists the multiplicity of the reflections whose posterior reaches the top grid edge.
- `torchsx.merge-stats` reports the cross-correlation with a reference dataset per shell, against the value expected from the two CC1/2.
- probixi databases with the `.db` suffix are accepted.

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
