# Merge statistics

`torchsx-merge-stats` scores merged reflection files. Each **dataset** is either one
merge or two independent halves. A single merge gets its summary statistics, and two
halves get A/B statistics. Every dataset other than the **reference** is also scored as
a difference against it, provided both have two halves. The reference is the first
dataset unless `--reference` names another.

```bash
# One merge: completeness, <I/sigma(I)>, multiplicity, Wilson falloff
torchsx-merge-stats merged.mtz -o stats

# Two independent halves of one dataset
torchsx-merge-stats A/merged_dark.mtz B/merged_dark.mtz -o stats

# Dark reference plus light datasets, each as two independent arms
torchsx-merge-stats A/merged_dark.mtz B/merged_dark.mtz --name dark \
  --dataset t1 A/merged_t1.mtz B/merged_t1.mtz \
  --dataset t2 A/merged_t2.mtz B/merged_t2.mtz -o stats

# CrystFEL halves (no cell in the file) and mmCIF, all formats in one call
torchsx-merge-stats dark.hkl1 dark.hkl2 --cell 61.5 90.8 150.9 90 90 90 \
  --spacegroup 'C 2 2 21' --dataset light light.hkl1 light.hkl2 -o stats
```

The positional files come first; each `--dataset NAME FILE [FILE]` adds one more
dataset. `python -m torchsx.cli.merge_stats` runs the same command from a source
checkout. CPU only.

## Inputs

| Format | Read by | Notes |
| --- | --- | --- |
| MTZ | TorchRef `MTZReader` | Column choice follows TorchRef's priority list; `--labels F=FP SIGF=SIGFP` (keys F, SIGF, I, SIGI) pins it. A count column labelled N, NOBS, NMEAS, MULT or MULTIPLICITY gives the multiplicity. |
| CrystFEL `.hkl`, `.hkl1`, `.hkl2` | TorchRef `HKLReader` | The file has no cell or space group: `--cell`/`--spacegroup`, else those of the first MTZ or mmCIF input. `nmeas` gives the multiplicity. |
| mmCIF structure factors | TorchRef `ReflectionCIFReader` | `_refln` amplitudes or intensities; cell and space group from the file. |

Each merge is mapped onto the canonical CCP4 asymmetric unit with Friedel mates
merged. Rows that share an index are averaged with inverse-variance weights. All
inputs must share one Laue class, and `--dmin`/`--dmax` cut every input.

Amplitudes stored in the file are used as given. For a file with intensities only,
`--amplitudes` chooses how amplitudes are made:

- `french-wilson` (the default) uses TorchRef's French-Wilson posterior.
- `sqrt` uses F = √I and σ(F) = σ(I)/2F, for I > 0 only.

A reflection without an amplitude (a French-Wilson rejection, or I ≤ 0 under `sqrt`)
still counts for the summary of a single merge. It is left out of the A/B and
difference comparisons, which need amplitudes.

Intensity statistics use the measured intensities when every merge in the comparison
carries them, and F² otherwise. The `intensities` field records which was used.

## Statistics

All shells are equal-count resolution shells over the reflections in the comparison;
there are `--shells` of them (default 20).

**Summary of one merge**, per shell and overall:

- **Completeness.** The fraction of the unique, not systematically absent reflections of the space group that the merge holds. The lowest shell reaches out to `--dmax`, or to all low-order reflections when no cut is given. `n_absent` counts merged reflections that are systematically absent.
- **⟨I⟩ and ⟨I/σ(I)⟩.**
- **Mean multiplicity.** Only where the file carries counts.
- **Wilson falloff.** `wilson_B` is the slope of ln⟨I/ε⟩ against s² = 1/4d² over the shells beyond 4 Å. There is no scattering-factor normalisation, so it compares merges of one crystal; it is not an absolute B.

**Two halves A, B.** B is first scaled onto A on their common reflections. The scale
model is one log scale plus six anisotropic terms on hkl/max|hkl|. It is fitted by
σ-weighted soft-L1 least squares on amplitudes. Per shell and overall:

| Column | Definition |
| --- | --- |
| `cc_I`, `cc_F` | Pearson CC of I and of F |
| `rsplit_I` | √2 Σ\|I_A − I_B\| / Σ(I_A + I_B) |
| `r_F` | Σ\|F_A − F_B\| / ½Σ(F_A + F_B) |
| `rms_z`, `frac_z_gt3` | RMS and tail fraction of (F_A − F_B)/√(σ_A² + σ_B²); 1 and 0.3 % for honest σ |
| `S_I`, `N_I`, `lnsr_I`, `lnsr_F` | S = Cov(x_A, x_B), the variance of the common signal; N = Var(x_A − x_B)/2, the error variance of one half; ln N/S |

CC = 1/(1 + N/S) saturates once N ≪ S, while `lnsr` keeps resolving the error: a
change of −0.1 means 10 % less error variance. The scale coefficients and the
isotropic ΔB of B against A are reported alongside. `lowres_drift` is the unscaled
median ln(I_B/I_A) of the lowest-resolution 2 % of reflections minus that of the
middle 40 %. It is the smooth falloff difference that the anisotropic model cannot
absorb.

**Difference of dataset X against the reference R.** Both datasets come as two
independent arms A and B. The difference is computed on the reflections common to the
four merges:

1. In each arm, X is scaled onto R.
2. Arm B is scaled onto arm A using the reference merges, and the same correction is applied to X_B.
3. Each arm gives ΔF = F_X − F_R, and its σ(ΔF) = hypot(σ_R, σ_X).

Per shell, the outputs are:

- `cc_dF`: the Pearson CC of ΔF between the two arms.
- `lnsr_dF`: ln N/S of ΔF.
- `bits`: information in bits.

Within each resolution shell, the reflections are split into `--sigma-bins` quantile
bins (default 5) of hypot(σ_ΔF,A, σ_ΔF,B). The information is

    bits = Σ_b n_b · (−½ log₂(1 − CC_b))

For x_A = s + e_A and x_B = s + e_B, each term −½ log₂(1 − CC) = ½ log₂(1 + S/N) is
the information one arm carries about the common signal, per reflection. CC is
clipped to ±0.999 and keeps its sign, so noise around zero averages out. A bin with
fewer than three reflections contributes nothing.

Binning by σ keeps two effects from counting as information: a re-weighting of noisy
reflections, and a truncation that raises a pooled CC. That is why no pooled cc_dF is
reported. `bits_res_only` uses the shell CC alone. The absolute number of bits depends
on the binning, so compare datasets at a fixed `--shells` and `--sigma-bins`.

## Outputs

`--format` chooses what is written to the new `-o` directory; the default is `csv plots`.
A short table always goes to stdout.

| File | Contents |
| --- | --- |
| `summary.csv` | One headline row per dataset (`kind` summary or halves) and per difference (`kind` difference, with its `reference`). |
| `shells_summary.csv`, `shells_halves.csv`, `shells_difference.csv` | One row per dataset and shell; written for the kinds present. |
| `bins_difference.csv` | One row per difference, shell and σ bin: ranges, n, cc_dF and bits. |
| `stats.json` | Everything above plus scale coefficients and settings (`--format json`). |
| `summary.png`, `halves.png`, `difference.png` | Per-shell curves with one line per dataset, written for the kinds present (`--format plots`). |
