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
| `STREAM` | | CrystFEL `.stream`, a prepared `.sfx`, or a probixi `.duckdb` run (see [probixi input](#probixi-input)). Several streams form one data set. |
| `--split NAME=PATH` | | A stream of the state `NAME`; repeatable. Streams of the same name form one state, and `PATH` may be a quoted glob pattern. Each state is merged on its own. |
| `--split-file FILE` | | States for the frames of the streams (see [States](#states)). |
| `--unassigned` | `train` | Patterns `--split-file` assigns to no state: `train` on them without merging them, `drop` them from both, or refuse the input with `error`. |
| `--spacegroup SG` | | Space group, e.g. `'P 63'`. Required for `.stream` input. |
| `--dmin`, `--dmax` | all | High- and low-resolution cut in Å. |
| `--merge-friedel` / `--no-merge-friedel` | merge | Keep Friedel mates apart with `--no-merge-friedel`; the MTZs then carry (+) and (−) columns. |
| `--enrich-alpha ALPHA` | `auto` | probixi database only: the enrichment gate (see [probixi input](#probixi-input)). `off` reads every lattice. An `.sfx` saved from a database is used as saved. |
| `--crystals-where SQL` | | probixi database only: an SQL predicate on its `crystals` table. Only the crystals it accepts are read. It applies on top of the gate. |

### Measurement

| Option | Default | |
|---|---|---|
| `--int-radius R_PEAK,R_GAP,R_BG` | from the stream | The integration radii in pixels, as given to indexamajig. |
| `--gain` | `auto` | Detector gain in ADU per photon (see [Gain](#gain)). |
| `--detector-noise` | `off` | Learned detector variance A + B (λ + σ<sub>b</sub>²): `auto` fits one global B when the gain was estimated from the data, `panel` fits it per detector panel, `resolution` per resolution bin (ten bins of equal reflection count). |
| `--noise-terms` | `AB` | With `--detector-noise panel` or `resolution`: which of A and B are fitted. B alone is a gain correction per panel or bin. |

### Training

| Option | Default | |
|---|---|---|
| `--steps N` | 400 | Optimizer updates in total, divided evenly over the states (see [States](#states) for patterns of no state). |
| `--redundancy N` | 100 | Target observations per reflection in a batch. |
| `--warmup N` | 25 | Learning-rate warm-up, in updates per state. |
| `--lr-schedule` | `constant` | Learning rate after the warm-up: `constant`, or `cosine` decay to zero at the last update. |
| `--average-last N` | 5 | When the last N updates per state oscillate, validate, checkpoint and merge the mean of the network after each of them; 0 uses the last update alone. |
| `--miss-frac F` | 0.03 | Initial weight of the miss component, a predicted reflection that recorded no signal; 0 turns it off. |
| `--prior` | `wilson` | Intensity prior: `wilson`, the only one (see [Intensity prior](#intensity-prior)). |
| `--tether TERMS` | `gauge` | Directions the likelihood leaves to the model, held at the raw counts; `none` turns it off (see [Sum-of-spots tether](#sum-of-spots-tether)). |
| `--pseudo-translation` | `auto` | Finds translational pseudo-symmetry in the data and, with a dominant translation, handles its classes of reflections; `off` skips the search (see [Pseudo-translation](#pseudo-translation)). |
| `--seed N` | 0 | Network initialization and batch sampling. |
| `--device` | `cuda` if available | `cuda` or `cpu`. |

### Output

| Option | Default | |
|---|---|---|
| `-o`, `--out RUN` | | Run directory. |
| `--no-merge` | | Stop after saving the checkpoints, for a separate `torchsx.merge`. |
| `--tether-report` | off | Report the tether's terms over each state's merged observations in `merged.metrics.json` (see [Sum-of-spots tether](#sum-of-spots-tether)). |
| `--free-from FILE` | 5 % new | Copy the free set from an `.mtz` (`R-free-flags` or `FreeR_flag`). |
| `--cache DIR` | `RUN/prepared` | Where the prepared observations are stored. The store depends on the inputs and the measurement settings, not on the states. |

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
   network at the end of training. At a constant learning rate, consecutive
   updates scatter around the optimum. When the last `--average-last` updates
   per state do (each gradient points back against the step before it), both
   checkpoints hold the mean of the network after each of them, so the merge
   does not inherit that scatter; otherwise they hold the latest update. The
   validation line in the log says which. Training itself continues from the
   latest update.
5. With `best.pt`, all crystals of each state are merged into `merged.mtz`, and
   each half of the crystals into `merged_half1.mtz` and `merged_half2.mtz`.
   `merged.metrics.json` records CC<sub>1/2</sub> and R<sub>split</sub> between
   the halves.

The merge is the `wls` estimator: for each reflection, the inverse-variance
mean of `(n - bg) / K` over its observations, with the scales K and the
per-observation variance from the trained network. That variance is the
counting variance, with the detector terms of `--detector-noise` where the
checkpoint fitted them, plus the scatter the network predicts. A reflection
measured once is merged with the sigma of its one observation. The
intensities, which can be negative, become amplitudes by French-Wilson. The
merge rests on the trained scales, the error model and the French-Wilson
prior.

The network limits the scatter it predicts for an observation at a log-scatter
variance of 20. A reflection with more than half of its observations at that
limit takes the pooled merge, `sum(n - bg) / sum(K)`, with the pooled sigma.
The log counts these reflections and `merged.metrics.json` records the count as
`wls_pooled`.

`--estimator posterior` writes the Wilson-posterior amplitudes of the shape
likelihood instead.

The half files hold the same estimator as `merged.mtz`, on each half of the
crystals. `merged.metrics.json` records the agreement of the half intensities
before any conversion to amplitudes as `split_half_wls` or
`split_half_posterior`, and the statistics of the half files as
`split_half_files`: CC<sub>1/2</sub> and the flat CC<sub>1/2</sub> of
[`torchsx.merge-stats`](merge_stats.md), overall and per shell. The halves
share the network and the form of the prior, French-Wilson fitted per half or
Wilson. Where the data are weak a plain CC<sub>1/2</sub> then measures the
prior. The flat CC<sub>1/2</sub> divides out the running mean and spread of
each half and falls to zero there. For independent merges, train two runs on
disjoint sets of patterns.

`--also ESTIMATOR ...` merges all crystals with further estimators and writes
`merged_ESTIMATOR.mtz` and `merged_ESTIMATOR.tsv` for each, plus the table of
the product. `flat` and `ml` read the posterior curve without the prior.
`pooled`, `mean` and `wls` are moment estimators on the trained scales; their
tables keep negative intensities.

## torchsx.merge

```bash
torchsx.merge RUN/best.pt STREAM -o merged.mtz
torchsx.merge RUN/best.pt --split dark=dark.stream --split light=light.stream -o merged.mtz
```

Merges the given streams with a trained network. The checkpoint supplies the
network, the Wilson surface, the measurement settings (radii, gain, detector
noise) and the seed for the halves; `--int-radius`, `--gain` and
`--merge-friedel` override them. `--tether-report` adds the tether's report. The streams can be the training data, more
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

A state can come from several streams: repeat its name, or give a glob
pattern, quoted so that the shell leaves it to TorchSX. Several streams without
a name are one data set.

```bash
torchsx.train --split dark=run1.stream --split dark=run2.stream \
  --split light='light_runs/*.stream' --spacegroup 'P 63' --dmin 2.0 -o joint/
torchsx.train run1.stream run2.stream --spacegroup 'P 63' --dmin 2.0 -o pooled/
```

The frames of one or more streams can instead be divided with `--split-file`,
either a CrystFEL custom-split file (the one partialator takes as
`--custom-split`) or a `.pt` tensor with one integer label per crystal after
the resolution cut (−1: no state):

```bash
torchsx.train run1.stream run2.stream --split-file states.split \
  --spacegroup 'P 63' --dmin 2.0 -o joint/
```

A custom-split file has three columns, `filename event dataset`:

```
/data/run1.h5  //0  dark
/data/run1.h5  //1  light
/data/run2.h5  *    light      # every frame of run2.h5
```

An event of `*` names every frame of the file. A frame listed under several
dataset names is a member of each of those states: it is stored once, drawn
into the batches of each and merged into each, so states may overlap or
contain one another (every light frame next to each delay, say). A repeated
identical row counts once. States from a split file are ordered by name,
states from `--split` in the order first given. The first state supplies the
validation crystals, which are held out of the training of every state.

Patterns the split file assigns to no state are kept by default
(`--unassigned train`): the state label is not an input to the network, so
they train it like any other pattern, and they are merged into no MTZ. Their
batches take the share of the updates that they have of the training crystals,
spread evenly over the run; the states take turns over the rest. With 400
updates, two states and a tenth of the crystals unassigned, that is 40 updates
on the unassigned patterns and 180 per state. `--unassigned drop` neither
trains on them nor merges them, and `--unassigned error` stops when there are
any. `torchsx.merge` takes the choice from the checkpoint.

The prepared store holds every crystal of the inputs once and no state; the
states are assigned each time it is opened. The same streams with other names,
another split file or another `--unassigned` therefore reuse the store, in
`torchsx.train` and in `torchsx.merge`:

```bash
torchsx.merge joint/best.pt combined.stream --split-file by_delay.split -o by_delay/merged.mtz
```

This merges the training data into the states of `by_delay.split` without
preparing it again. A state the network was not trained with takes the Wilson
surface of the first trained state. Because the store holds all crystals,
dropped patterns still count in what is formed from the whole store before
training: the reflection list, the gain calibration and the start values.

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

A probixi database keeps every lattice, including those the indexer found by
chance. probixi scores each lattice with `enrich_p`, the probability that its
predicted spots land on signal as often as they do by chance. TorchSX reads
only the lattices with `enrich_p` at most alpha. `--enrich-alpha auto` takes the
alpha the database records (`integration.bg_model_var_alpha`), else 1e-3, the
default of probixi's own `--enrich-alpha`. `--enrich-alpha off` reads every
lattice. The log prints the gate and the number of lattices it keeps, and the
recipe records it. A lattice without `enrich_p` was not integrated by probixi
and does not pass the gate. The log counts such lattices separately.

With probixi's `--max-lattices` above one, a frame can hold several lattices.
The second and later ones are indexed on the peaks the first left over. Under
`auto` they are read only with `enrich_p` at most 1e-10. A number given to
`--enrich-alpha` applies to every lattice alike.

An `.sfx` saved from a probixi database is a training cache. It holds the
crystals that passed the filter in force when the database was read
(`enrich_alpha` and `crystals_where` of `SFXStream.from_duckdb` or
`SFXStream.read`), and its sidecar records that filter. Training and merging
use the file as saved: the log prints the recorded filter, the recipe keeps it,
and `--enrich-alpha` and `--crystals-where` are refused, since they apply only
where a database is read. An `.sfx` that records no filter is taken as
unfiltered. To change the filter, read the database again.

Without `--dmin`, a probixi database is cut at the 10th percentile of its
crystals' diffraction limits, so shells with no signal do not enter the merge
or its statistics.

A custom-split file names a database's frames as CrystFEL does, `filename //event`.

TorchSX and probixi name each other's version. `torchsx.SUPPORTED_DB_SCHEMA` is
the newest database schema the reader understands, and `torchsx.PROBIXI_VERSION`
is the first probixi release that writes it. probixi holds
`probixi.SCHEMA_VERSION` and `probixi.TORCHSX_VERSION`, the first TorchSX
release that reads its schema. Neither package depends on the other. Where
probixi is installed, TorchSX warns when that probixi writes a newer schema.
Older schemas read as before.

## Gain

TorchSX models the data in photons, so the gain matters: a wrong gain
misstates the counting noise. `--gain` takes:

| Value | |
|---|---|
| `auto` | For a probixi database that records its noise scale, use the declared gain times that scale. Otherwise estimate the gain from the data and use the estimate. Use the stream's gain when no estimate is possible. |
| `trust_input` | Use the stream's gain. |
| `calibrate` | Use the estimate; fail if there is none. |
| a number | Use this gain in ADU per photon. |

The estimate fits a truncated normal to the core of the negative integrated
intensities, whose spread is set by the counting noise. It is biased, but a
declared gain that understates the noise does more harm: at several thousand
observations per reflection, a variance one third too small already appears
as intensity in the weak shells. The selected gain and the reason for it are
printed and recorded in `run.json`.

probixi (0.10.3 and later) fits the variance of the background under its disks
against the Poisson law at the declared gain and records the ratio in
`integration.bg_model_var_scale`. The declared gain times this ratio is the
gain at which the photon counts carry Poisson noise, measured on the frames.
`auto` uses it and makes no estimate of its own; `calibrate` still does.
With several databases `auto` uses the ratio only where every input records
the same one. Otherwise it says so in the log and estimates the gain.

Both estimates divide the scatter of signal-free sums by the counting noise of
the integration disk. A probixi database records the pixels of every disk, and
the estimate uses them. A stream states one disk in its header. For a stream
whose reflections were integrated with larger disks, the estimate returns the
detector gain multiplied by the ratio of the disk areas, and the counting noise
of those reflections is understated. probixi writes such streams with `--disk
shell`; its default stream uses one disk.

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

Each state has a Wilson prior whose mean intensity is a
smooth surface over reciprocal space: Chebyshev polynomials in resolution plus
the anisotropy terms of the space group. Its coefficients start from a Wilson
fit to the data and are trained with the network on the same marginal
likelihood.

`--prior wilson` is the only choice; the flag stays so existing scripts run.
The flat prior, `--prior none`, was removed in 0.2.4: its posterior has no
upper bound, so for weak reflections it could run off the top of the finite
integration grid ([#24](https://github.com/HatPdotS/TorchSX/issues/24)). The
Wilson prior closes the posterior off and is not affected. `torchsx.merge` refuses
a checkpoint trained with it; retrain, or merge it with torchsx 0.2.3.

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

With `--tether-report`, `merged.metrics.json` reports every term over each
state's merged observations (`tether`), whichever terms the network was trained
with. The report is a check only; the merged intensities do not depend on it.
It fits all of a state's observations at once, at about 100 bytes of device
memory each, so it is off by default. `torchsx.merge` takes the terms from the
checkpoint.

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
assignments, give a new `--cache` path.

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
