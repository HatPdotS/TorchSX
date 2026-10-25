# Changelog

TorchSX is released as two PyPI packages from this repository: `torchsx` and
`torchsx-native`, the compiled `.stream` reader/writer. `torchsx-native` gets a
new version only when `native/` changes; a release that leaves it unchanged
says so below.

## 0.1.1

`torchsx-native` is unchanged (0.1.0).

### Added

- On-disk prepared store. `torchsx.train` and `torchsx.merge` parse the streams
  block by block into a store on disk (default `RUN/prepared/`, about 52 bytes
  per observation) and read it back one batch of crystals at a time, the next
  batch while the current one is computed. Host memory no longer grows with
  the number of observations.
- `torchsx.merge-stats` reads a run directory: `torchsx.merge-stats run/`, or
  no input from inside one, finds the merges and halves of every state and
  writes `run/merge_stats/`. With several states, each is scored against the
  first.
- `torchsx.merge-stats` reports CC1/2 and R_split per resolution shell for two
  halves, with the highest-resolution shell in the headline (`cc_half_outer`,
  `r_split_outer`).
- `torchsx.merge_stats.r_split` and `split_half_stats`.

### Changed

- `--cache` names a directory (default `RUN/prepared`) instead of a file
  (`RUN/prepared.pt`). A run prepared by 0.1.0 keeps its `prepared.pt`, and
  `torchsx.merge` still reads it.
- R_split has one definition throughout torchsx (training log,
  `STEM.metrics.json` and `torchsx.merge-stats`):
  √2 Σ|I_A − I_B| / Σ|I_A + I_B|. The denominator sums the magnitude of each
  pair, so negative intensities of weak reflections no longer cancel positive
  ones; values can differ from those 0.1.0 reported.
- `torchsx.merge-stats` writes TSV instead of CSV (`summary.tsv`,
  `shells_*.tsv`, `bins_difference.tsv`); `--format csv` is now `--format tsv`.
  The halves tables carry `cc_half` and `r_split` in place of `cc_I`, `cc_F`,
  `rsplit_I`, `r_F`, `rms_z`, `frac_z_gt3` and the ln N/S columns.
- `torchsx.merge-stats -o` is optional for a run directory.
- Streams integrated with a gradient background (`--integration=rings-grad`)
  give a warning instead of an error; the mean-background noise model is used
  as an approximation.
- Unsupported measurement semantics name the input file, the value found and
  the value expected.
- `import torchsx` no longer imports torch; the public names load on first use,
  and the command-line tools answer `--help` without torch.

### Fixed

- An interrupted build of the prepared store no longer blocks the next run: its
  staging directories (`CACHE.raw`, `CACHE.partial`) are replaced. A directory
  of that name that TorchSX did not create stops the run before parsing and is
  left untouched.

## 0.1.0

Initial public release.
