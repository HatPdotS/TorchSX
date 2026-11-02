# Command-line reference

`torchsx.train` trains the scaling network on one or more streams and merges
them with it. `torchsx.merge` merges streams with an already trained network.
Both take the same input and measurement options.

## torchsx.train

```bash
torchsx.train STREAM --spacegroup SG -o RUN [options]
torchsx.train --split NAME=PATH --split NAME=PATH --spacegroup SG -o RUN [options]
```

### Inputs

| Option | Default | |
|---|---|---|
| `STREAM` | | CrystFEL `.stream`, a prepared `.sfx`, or a probixi `.duckdb` run (see [probixi input](#probixi-input)). |
| `--split NAME=PATH` | | A named state, one stream each; repeatable. Each state is merged on its own. |
| `--split-file FILE` | | States for a single stream (see [States](#states)). |
| `--spacegroup SG` | | Space group, e.g. `'P 63'`. Required for `.stream` input. |
| `--dmin`, `--dmax` | all | High- and low-resolution cut in Å. |
| `--merge-friedel` / `--no-merge-friedel` | merge | Keep Friedel mates apart with `--no-merge-friedel`; the MTZs then carry (+) and (−) columns. |

### Measurement

| Option | Default | |
|---|---|---|
| `--int-radius R_PEAK,R_GAP,R_BG` | from the stream | The integration radii in pixels, as given to indexamajig. |
| `--gain` | `auto` | Detector gain in ADU per photon (see [Gain](#gain)). |
| `--detector-noise` | `off` | Learned detector variance A + B (λ + σ<sub>b</sub>²): `auto` fits one global B when the gain was calibrated, `panel` fits it per detector panel. |
| `--noise-terms` | `AB` | With `--detector-noise panel`: which of A and B are fitted. B alone is a per-panel gain correction. |

### Training

| Option | Default | |
|---|---|---|
| `--steps N` | 400 | Optimizer updates in total, divided evenly over the states. |
| `--redundancy N` | 100 | Target observations per reflection in a batch. |
| `--warmup N` | 25 | Learning-rate warm-up, in updates per state. |
| `--lr-schedule` | `constant` | Learning rate after the warm-up: `constant`, or `cosine` decay to zero at the last update. |
| `--miss-frac F` | 0.03 | Initial weight of the miss component, a predicted reflection that recorded no signal; 0 turns it off. |
| `--outlier-frac F` | 0 | Initial weight of a detector-outlier component; 0 turns it off. |
| `--prior` | `wilson` | Intensity prior: `wilson`, or `none` (see [Intensity prior](#intensity-prior)). |
| `--tether TERMS` | `gauge` | Directions the likelihood leaves to the model, held at the raw counts; `none` turns it off (see [Sum-of-spots tether](#sum-of-spots-tether)). |
| `--pseudo-translation` | `auto` | Finds translational pseudo-symmetry in the data and, with a dominant translation, handles its classes of reflections; `off` skips the search (see [Pseudo-translation](#pseudo-translation)). |
| `--seed N` | 0 | Network initialization and batch sampling. |
| `--device` | `cuda` if available | `cuda` or `cpu`. |

### Output

| Option | Default | |
|---|---|---|
| `-o`, `--out RUN` | | Run directory. |
| `--no-merge` | | Stop after saving the checkpoints, for a separate `torchsx.merge`. |
| `--free-from FILE` | 5 % new | Copy the free set from an `.mtz` (`R-free-flags` or `FreeR_flag`). |
| `--cache DIR` | `RUN/prepared` | Where the prepared observations are stored. |

`torchsx.train --help` lists the same options.

### How a run proceeds

1. The streams are parsed block by block, converted to photons with the
   selected gain and written to the store `prepared/`, one crystal's
   observations after the other.
2. 5 % of the crystals of the first state are held out for validation.
3. The network is trained for `--steps` updates. Each batch holds crystals of
   one state, and the states take turns: 400 updates over two states give 200
   per state.
4. `best.pt` keeps the network at the lowest validation loss, `last.pt` the
   network after the last update.
5. With `best.pt`, all crystals of each state are merged into `merged.mtz`, and
   each half of the crystals into `merged_half1.mtz` and `merged_half2.mtz`.
   `merged.metrics.json` records CC<sub>1/2</sub> and R<sub>split</sub> between
   the halves.

The halves share the network. For independent merges, train two runs on
disjoint sets of patterns.

## torchsx.merge

```bash
torchsx.merge RUN/best.pt STREAM -o merged.mtz
torchsx.merge RUN/best.pt --split dark=dark.stream --split light=light.stream -o merged.mtz
```

Merges the given streams with a trained network. The checkpoint supplies the
network, the Wilson surface, the measurement settings (radii, gain, detector
noise) and the seed for the halves; `--int-radius`, `--gain`, `--prior` and
`--merge-friedel` override them. The streams can be the training data, more
data from the same experiment, or other states. Merging the training data
reproduces the merge written by `torchsx.train`.

With one state the output is `merged.mtz` with its two halves and
`merged.metrics.json`. With several states each gets its own set:
`merged_dark.mtz`, `merged_dark_half1.mtz`, and so on.

## States

States are subsets of the data that share the network but are merged
separately, typically light and dark. The state label is not an input to the
network.

```bash
torchsx.train --split dark=dark.stream --split light=light.stream \
  --spacegroup 'P 63' --dmin 2.0 -o joint/
```

A single stream can be divided with `--split-file`, either a CrystFEL
custom-split file (the one partialator takes as `--custom-split`) or a `.pt`
tensor with one integer label per crystal after the resolution cut:

```bash
torchsx.train combined.stream --split-file states.split \
  --spacegroup 'P 63' --dmin 2.0 -o joint/
```

A custom-split file has three columns, `filename event dataset`. Every crystal
must be labelled. A frame listed under several dataset names goes into each of
those states; a repeated identical row counts once. States from a split file
are ordered by name, states from `--split` in the order given. The first state
supplies the validation crystals.

## probixi input

A probixi database carries what a stream cannot, and its presence switches
this on; a checkpoint keeps what it was trained with:

- the integration radii and gain of the run, from its `integration` and
  `calibration` tables; the calibration's probixi version, schema version and
  noise-model settings are recorded as provenance, and a database whose schema
  is newer than this TorchSX supports is refused;
- every predicted reflection, including the disks the stream writer drops
  because the mask, the frame edge or a neighbouring disk clipped them, with
  the background summed over the pixels actually integrated (`n_pixels`),
  which the scale network also sees;
- the noise model's background under each disk (`background_model`) in place
  of the annulus mean, with its measured variance in place of the ring's
  Poisson law and no mean correction; the intensity is re-referenced so that
  intensity plus background stays the measured count;
- the run's per-crystal diagnostics in the network's crystal head: the frame's
  fluence scale, enrichment, bright and indexed peak counts, the indexing
  rmsd, mosaicity, profile radius and peak resolution;
- when the frames' photon energies differ, the Ewald offsets at each image's
  own wavelength rather than the median one.

Without `--dmin`, a probixi database is cut at the 10th percentile of its
crystals' diffraction limits, so shells with no signal do not enter the merge
or its statistics.

A custom-split file names a database's frames as CrystFEL does, `filename //event`.

## Gain

TorchSX models the data in photons, so the gain matters: a wrong gain
misstates the counting noise. `--gain` takes:

| Value | |
|---|---|
| `auto` | Estimate the gain from the data and compare it with the gain in the stream. Keep the stream's gain unless the two differ by more than 50 %. |
| `trust_input` | Use the stream's gain. |
| `calibrate` | Use the estimate; fail if there is none. |
| a number | Use this gain in ADU per photon. |

The estimate fits a truncated normal to the core of the negative integrated
intensities, whose spread is set by the counting noise. It is biased, hence
the 50 % tolerance. `auto` uses the stream's gain when no estimate is possible and the
estimate when the stream has no gain. The selected gain and the reason for it
are printed and recorded in `run.json`.

Integration radii come from `--int-radius`, else the checkpoint, else the
indexamajig command recorded in the stream. Omitted indexamajig options take
their CrystFEL defaults ([indexamajig manual](https://www.desy.de/~twhite/crystfel/manual-indexamajig.html)),
including radii `4,5,7`. Ring integration without centring is supported; other
integration modes are rejected. With `grad`, CrystFEL subtracts a background
plane fitted to the background ring instead of the ring's mean; TorchSX then
prints a warning and uses the mean-background noise model as an approximation.

## Free set

Each MTZ carries `R-free-flags` (0 = free, 1 = work). By default 5 % of the
reflections are drawn with a fixed seed, independent of `--seed`. To compare
refinements of different merges, give them the same free set with
`--free-from earlier.mtz`. Friedel mates always share a flag.

## Intensity prior

With `--prior wilson` each state has a Wilson prior whose mean intensity is a
smooth surface over reciprocal space: Chebyshev polynomials in resolution plus
the anisotropy terms of the space group. Its coefficients start from a Wilson
fit to the data and are trained with the network on the same marginal
likelihood.

`--prior none` integrates the likelihood over positive intensity without a
prior density. The result then depends on the bounds of the integration grid
for weak reflections, and its validation loss is not comparable with that of
a Wilson run.

`torchsx.merge` uses the prior of the checkpoint unless `--prior` is given.

## Sum-of-spots tether

The likelihood sees an observation through the product of its scale and the
reflection's intensity, so a smooth rescaling of every reflection's scale
against its intensity leaves the fit unchanged. Something has to fix it. The
raw counts do: the observations of a reflection in still images sample the
rocking curve evenly, so the mean of their scales is the rocking-curve integral,
the sum of the reflection's spots, which depends on no model. The tether holds
the model to it, as a term in the training loss computed from each batch alone.
Unlike the intensity prior it does not weaken as the data grow; at its minimum
every term is where the counts put it.

| Term | Holds |
|---|---|
| `gauge` | The rocking-curve integral of the scale is the same at every resolution and in every direction. The merged intensities keep the fall-off of the integrated intensities. |

With `--tether none` the smooth common mode of the log scale is projected out of
every batch instead, which pins its geometric rather than its arithmetic mean.

`merged.metrics.json` reports every term over each state's merged observations
(`tether`), whichever terms the network was trained with. `torchsx.merge` takes
the setting from the checkpoint.

## Model

Each observation's scale and scatter distribution are predicted from its own
geometry and from the other observations of its crystal. The crystal's
observations are split into five folds, and the crystal is encoded once per
fold with the measurements of that fold hidden: its observations attend to
each other, the crystal's orientation and photon energy are corrected from
them, and every hidden observation is read out from the encoding in which it
is hidden. For every observation whose measurement it sees, the network reads
its intensity relative to the mean of its reflection over the whole input, and
the log of that mean. The scatter distribution is a mixture of four Gaussians in
log scale, plus the miss component (`--miss-frac`). The detector term is
Poisson counting on the spot and the background estimate.

## Pseudo-translation

For crystals with translational NCS, a lattice translocation defect or
stacking faults. A translation `t` relating parts of the crystal makes a
reflection's mean intensity, and how it is recorded, depend on its phase `h·t`.
In each resolution shell the reflections then form classes: some much stronger
than the shell's mean and some much weaker, some recorded as sharp spots and
some spread along the fault direction.

By default (`--pseudo-translation auto`) training first looks for such
translations in the data: the Patterson function of the normalised plain
means, its significant off-origin peaks refined, harmonics `n·t` of a stronger
translation placed exactly. The run log lists each translation with the
amplitude of the intensity modulation it implies. A translation is used when it
dominates: an amplitude of at least 0.1 and three times that of any other
translation that is not one of its harmonics; its harmonics with an amplitude
of at least 0.1 are used with it. A copy of the whole structure gives such a
peak; the interatomic vectors of a small structure give several peaks of
similar height, and none is used. Data with fewer than 1,000 observed
reflections are not searched.

With a translation in use, three things follow the classes it makes, with
`c(h) = cos 2π h·R t` averaged over the point group:

- **The prior.** The Wilson prior's mean is the smooth surface times a
  modulation `g(h)`, fitted to the plain means before training: per resolution
  shell, the amplitude of each translation's `c(h)`, then a non-decreasing
  link.
- **The network.** The scale model reads each observation's `c(h)`, so each
  class can have its own rocking curve, and each crystal its own class pattern.
- **The tether.** A `phase` term holds the arithmetic mean of K in every class
  (equal-count bins of the `c(h)`) and along low-order trends in resolution, so
  the brightness of every class is its sum of spots, whatever the network makes
  of the features.

The three belong together: with class features, a prior whose mean ignores the
classes pulls every class toward the shell's mean through the scatter shape,
and the tether's phase term needs the tether (`--tether none` is refused for
such data). Without a translation the model is unchanged. `torchsx.merge` takes
the modulation and the classes from the checkpoint.

## Cache and reproducibility

The prepared observations are stored in `--cache` (default `RUN/prepared`;
`torchsx.merge` uses the checkpoint's directory). A later run with the same
inputs and preparation settings reuses it: the store records the input
contents and settings, and a mismatch is an error rather than a silent reparse.
Moved but identical files are accepted. For changed inputs or split
assignments, give a new `--cache` path. A run prepared before the store
existed keeps its `prepared.pt`, and `torchsx.merge` reads it.

The store holds the observations on disk; host memory holds the crystal and
reflection tables and a few batches. Preparation reads the streams in blocks
into a temporary file next to the store, and training and merging read one
batch of crystals at a time, the next while the current one is computed. The
store needs about 56 bytes per observation of disk space, and the temporary
file about 50 more while it is written.

While it is built, the store stages in `CACHE.raw` and `CACHE.partial` next to
it. A directory of either name that an interrupted build left behind is
replaced; any other directory of that name stops the run before parsing, and
its contents are left as they are.

`run.json` records the resolved settings: inputs, measurement, gain and its
reason, model, learning rates and schedule. `--seed` fixes network
initialization and batch sampling; `torchsx.merge` reuses the run's seed for
the halves unless `--seed` is given. Parsing uses up to eight CPU workers
within the allocation.
