# Changelog

## Unreleased

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
