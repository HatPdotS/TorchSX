# Raw-distribution figures

`torchsx-stats` describes integrated observations before scaling or merging. It
writes PNG figures only, retaining negative and zero intensities and the complete
finite tails. There is no gain calibration, posterior correction or training in
this command. Install plotting support with `pip install -e '.[stats]'`.

```bash
# Quick look: random complete images totaling at least 50K crystals, or all available.
torchsx-stats indexed.stream --spacegroup 'P 63' -o figures

# Final figures with explicit input selection and full observation counts.
torchsx-stats combined.stream --spacegroup 'P 63' \
  --split-file custom.split --select dark light3 \
  --sample-crystals all -o figures_full

# Separate inputs are plotted separately, with shared raw-intensity bins and resolution axes.
torchsx-stats --split dark=dark.stream --split light=light.stream \
  --spacegroup 'P 63' --sample-crystals all -o light_dark

# Replot without reparsing or requiring the original sources.
torchsx-stats --from-cache figures_full-cache --plots reflection-skew skew-examples \
  --target-skew 0 1 2 4 --min-observations 50 --min-frames 30 -o figures_selected
```

`python -m torchsx.cli.stats` runs the same command from a source checkout.

## What the plots show

| Plot | Interpretation |
| --- | --- |
| `raw-intensity` | Signed unscaled ADU across all finite observations. The asinh intensity axis preserves negatives and the region near zero; counts have a log axis. Unequal ADU bin widths mean this is counts per transformed bin, not a probability density. |
| `reflection-skew` | Adjusted sample skewness of repeated observations of each symmetry-unique HKL versus resolution, colored by observation count. Positive values indicate a longer positive tail. |
| `mean-median-gap` | `(mean − median) / (Q75 − Q25)` per HKL. This shows separation of two location summaries relative to the middle-50% width. The orange curve is the median in each resolution bin with at least 20 eligible HKLs. |
| `skew-examples` | Histograms for distinct HKLs nearest the requested skewnesses, with their means, medians, achieved skewness, image counts and resolution. Writes an overview and an individual PNG for each selected HKL. |

For a motivation figure, the per-HKL examples make asymmetry concrete, and the
skewness/gap panels show its prevalence across reflections and resolution. The
aggregate intensity histogram provides dynamic-range and negative-observation
context: mixing weak and strong reflections alone produces a broad distribution.
Per-HKL raw distributions also combine counting noise, crystal scales,
partiality and other heterogeneity. They do not isolate a likelihood residual or
establish that TorchSX improves accuracy. Use the independent merge and difference
comparisons for that claim. The mean–median gap is a descriptive statistic, not an
estimate of estimator bias.

Examples are selected for proximity to the requested skewnesses, with observation
count and HKL ordering breaking ties. They are explicitly labeled as selected
examples, not a random or representative sample. Displayed skewness may differ
from its target. Shape panels exclude systematic absences, undefined skewness,
invalid resolution and insufficiently observed HKLs. Gap panels also require a
nonzero interquartile range. Their captions record the eligibility thresholds.
The raw-intensity histogram includes all finite observations, including systematic
absences. No outlier clipping is applied.

## CLI review

The recovered command has **14 optional controls**, reduced from the original 20.

| Controls | Default and purpose |
| --- | --- |
| `-o` / `--out` | Required new PNG directory; one subdirectory per input/split label. |
| `--split NAME=PATH` | Repeat for named inputs. Bare positional inputs use their filename stems. Labels must be unique and safe directory names. |
| `--split-file`, `--select` | CrystFEL `filename event label` assignments; optionally select labels. Every sampled image must have an assignment. Conflicting duplicate assignments fail. |
| `--spacegroup` | Required for `.stream`; inherited from `.sfx` metadata unless overridden. |
| `--cell A B C ALPHA BETA GAMMA` | Optional resolution reference, in angstrom/degrees; otherwise the median valid cell of the sampled input. The resolved cell is recorded. |
| `--sample-crystals N\|all` | Default 50000 per input, seed 42. Random complete images are retained until the crystal target is reached, so the final image may exceed it. `all` uses the full input. |
| `--seed` | Sampling seed. |
| `--merge-friedel` / `--no-merge-friedel` | Merge by default; otherwise separate acentric mates using the production loader. Centric mates remain one HKL. Legacy caches lacking observed indices cannot recover already-merged mates. |
| `--from-cache` | Regenerate PNGs from a statistics cache. Accepts plot and eligibility controls, rejects input overrides. |
| `--plots` | All four plot families by default. |
| `--target-skew` | Example targets: `-1 0 1 2 4 6`. |
| `--min-observations`, `--min-frames` | Shape eligibility: at least 30 observations and 20 distinct images per HKL. Multiple crystals from one image count as one image. |

The former `--full` is `--sample-crystals all`. `--output` becomes `--out`, and
`--no-friedel` becomes `--no-merge-friedel`, matching the production CLI.
Parser workers follow the CPU allocation, capped at eight. The cache automatically
lives beside the output in `OUT-cache`. Reference-structure cell extraction,
custom cache placement, titles and intensity-unit relabeling no longer add public
flags; explicit cells and the Python rendering API cover those needs. Existing
cache unit labels are respected when replayed.

Sampling happens before custom-split selection: 50K crystals from a combined
stream does not mean 50K in each selected label. Actual sampled crystal/image
counts are printed and recorded; finite observation counts and eligible HKLs are
reported per label. `.stream` sampling uses the existing parser; `.sfx` loads the
cached tensors before sampling complete images, so its peak read memory is still
that of the full cache. Run substantial datasets on an allocated CPU node.

## Reproducibility and output

`OUT` contains PNGs only, never PDF files. `OUT-cache` contains compressed grouped
observations and a manifest recording source paths, sizes, timestamps, SFX sidecar
checksum, split-file checksum, resolved cells/groups, sample counts, settings and
stats/CLI source hashes. Cached arrays have checksums verified on replay. This
cache preserves the observations used to render; source files are not reread on
replay. Source size/timestamp records are not full source-content hashes.

Each rendering records its plot choices, target skewnesses and eligibility
thresholds beside the cached arrays. Existing output/cache directories are not
overwritten. Shared plotting bins/axes apply within one command invocation.

This CLI was restored for dataset characterization and paper motivation figures;
its usefulness is independent of whether a training job invokes it.
