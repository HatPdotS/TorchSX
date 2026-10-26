# Changelog

## 0.2.0 (unreleased)

### Upgrading from 0.1.0

- **Prepared caches from 0.1.0 are refused.** `torchsx.train` stops with
  "`--cache` … was built with different settings"; delete the cache or point
  `--cache` elsewhere and it is rebuilt. The cache signature records new
  settings (disk size, background, per-image k0) and the on-disk store record
  gained the disk pixel count.
- **`torchsx.merge-stats` writes tab-separated tables.** `--format csv` is gone:
  use `--format tsv`, and read `summary.tsv`, `shells_*.tsv` and
  `bins_difference.tsv` instead of the `.csv` files.
- **R_split has one definition everywhere**, √2 Σ|I_A − I_B| / Σ|I_A + I_B|: the
  training statistics, `STEM.metrics.json` and `torchsx.merge-stats`. Values
  change where measured intensities are negative (they used to be inflated, or
  even negative); for positive intensities nothing changes.
- **`duckdb` is a required dependency**; the `torchsx[duckdb]` extra is gone.
- Checkpoints from 0.1.0 record no feature tiers and are rebuilt with the
  default tiers, which are the 0.1.0 ones.

### Added

- **probixi databases as input.** A probixi DuckDB run (schema 1) is accepted
  wherever a stream is (`SFXStream.read`, `read_source`, `inspect_source` and
  the CLIs). What the database records is applied without flags: each disk's
  recorded pixel count, probixi's background model and its measured variance,
  probixi's crystal diagnostics as crystal features, and per-image Ewald
  offsets when photon energies vary between frames. Without `--dmin` the
  resolution cut is the 10th percentile of the crystals' diffraction limits.
  Disks clipped by the mask, the frame edge or a neighbour are dropped unless
  `clipped=True`. Streams and `.sfx` inputs prepare as before. (Ryan O'Dea, #7)
- **`torchsx.train --pseudo-translation auto`** for crystals with translational
  NCS, a lattice translocation defect or stacking faults. It finds the dominant
  translation in the Patterson function of the normalised intensities and gives
  each reflection a core and a broad profile component. The merge reports the
  total intensity, core and broad part, and `torchsx.merge` restores the
  profiles from the checkpoint. Off by default. (#6)
- **On-disk prepared store** (`RUN/prepared/`). Streams are converted disk to
  disk, and training and merging read each batch from disk while the previous
  one computes, so per-observation data is never held whole in host or GPU
  memory. Each update logs `io_wait`. (#3)
- The merge's split-half record carries CC1/2 and R_split for ten equal-count
  resolution shells and five intensity quintiles, and the merge prints the
  quintile R_split and the outer shell. (Ryan O'Dea, #7)
- `torchsx.merge-stats` takes a `torchsx.train`/`torchsx.merge` output
  directory (or runs inside one with no inputs), compares every exported
  state's merge and halves, reports CC1/2 and R_split for the halves, and
  writes to `DIR/merge_stats` unless `-o` is given. (#4)
- Checkpoints record the scale network's feature tiers, so `torchsx.merge`
  rebuilds the encoder that was trained. (Ryan O'Dea, #7)
- Every console script answers `--help` without importing torch. (#4)

### Changed

- A CrystFEL stream integrated with a gradient background (`rings-grad`) runs
  with a warning instead of failing. Other unsupported measurement settings
  still raise, now naming each field with its found and expected value. (#2)
- A missing crystal feature takes the column mean instead of zero. (#7)

### Fixed

- R_split with negative measured intensities (see *Upgrading*). (#4)
- Building a prepared store no longer deletes an existing `CACHE.raw` or
  `CACHE.partial` directory it did not create; it stops with an error and
  leaves the contents as they are. (#3)

### Dependencies

- `duckdb>=1.0` is required (tested with 2.0.0).
- `torchsx-native` stays at 0.1.0: the native build has not changed.

## 0.1.0 (2026-10-12)

Initial public release.
