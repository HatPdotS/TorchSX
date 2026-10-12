# Raw-distribution figures

`torchsx.observation-stats` plots integrated observations before any scaling
or merging: how the intensities of each reflection are distributed across
crystals. It keeps negative and zero intensities and the full tails, and
writes PNG figures.

```bash
# Quick look: random complete images totalling at least 50,000 crystals.
torchsx.observation-stats indexed.stream --spacegroup 'P 63' -o figures

# Every crystal, for selected states of a custom-split file.
torchsx.observation-stats combined.stream --spacegroup 'P 63' \
  --split-file custom.split --select dark light3 \
  --sample-crystals all -o figures_full

# Separate inputs, plotted separately on shared bins and axes.
torchsx.observation-stats --split dark=dark.stream --split light=light.stream \
  --spacegroup 'P 63' --sample-crystals all -o light_dark

# Replot from the cache, without the original inputs.
torchsx.observation-stats --from-cache figures_full-cache --plots reflection-skew skew-examples \
  --target-skew 0 1 2 4 --min-observations 50 --min-frames 30 -o figures_selected
```

## Plots

| Plot | Shows |
| --- | --- |
| `raw-intensity` | Histogram of all finite observations in ADU, on an asinh axis that keeps negative values and the region near zero; counts on a log axis. The bins are uneven in ADU, so this is counts per bin, not a density. |
| `reflection-skew` | Skewness of the repeated observations of each unique reflection against resolution, coloured by the number of observations. Positive values mean a longer tail towards high intensities. |
| `mean-median-gap` | `(mean − median) / (Q75 − Q25)` per reflection against resolution, with the median per resolution bin (bins with at least 20 reflections) in orange. |
| `skew-examples` | Histograms of individual reflections whose skewness is closest to each `--target-skew`, with mean, median, skewness, number of images and resolution. One overview PNG and one PNG per reflection. |

The per-reflection distributions combine counting noise, crystal scale,
partiality and every other source of variation; they describe the data, they
do not separate these sources. The examples are chosen by their skewness, not
at random. The shape plots leave out systematic absences and reflections below
`--min-observations` or `--min-frames`; the gap plot also leaves out reflections
with a zero interquartile range. The raw-intensity histogram includes every
finite observation.

## Options

| Option | Default | |
| --- | --- | --- |
| `-o`, `--out DIR` | | New directory for the PNGs, one subdirectory per input or state. The cache goes to `DIR-cache`. |
| `STREAM`, `--split NAME=PATH` | | Inputs. A bare input is labelled by its file name. |
| `--split-file FILE`, `--select LABEL ...` | all labels | CrystFEL custom-split file (`filename event label`), and the labels to plot. |
| `--spacegroup SG` | | Required for `.stream` input. |
| `--cell A B C ALPHA BETA GAMMA` | median cell of the input | Cell used for resolution. |
| `--sample-crystals N\|all` | 50000 | Whole images are drawn at random until N crystals are reached; `all` uses every crystal. |
| `--seed N` | 42 | Sampling seed. |
| `--merge-friedel` / `--no-merge-friedel` | merge | Keep acentric Friedel mates apart with `--no-merge-friedel`. |
| `--from-cache DIR` | | Replot from a cache; input options are not accepted with it. |
| `--plots` | all four | Which plots to draw. |
| `--target-skew` | `-1 0 1 2 4 6` | Skewness targets for the examples. |
| `--min-observations`, `--min-frames` | 30, 20 | Minimum observations and distinct images per reflection for the shape plots. Several crystals on one image count as one image. |

Images are sampled before a custom split is applied, so 50,000 crystals from a
combined stream are 50,000 in total, not per state. The command prints the
sampled counts and the observations and eligible reflections per label.

## Cache

`DIR-cache` holds the grouped observations behind the figures and a manifest
of the inputs (paths, sizes, timestamps, split-file checksum), the cell,
sample counts and settings. `--from-cache` replots from it without reading the
inputs again, with checksums verified. Existing output and cache directories
are never overwritten.
