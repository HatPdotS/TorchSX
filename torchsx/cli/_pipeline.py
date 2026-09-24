"""Shared input preparation, checkpoint reconstruction and MTZ export.

Source metadata resolves integration semantics and detector gain. Posterior
products contain F/SIGF; alternate research estimators use the Python API.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


READOUT_MAX_OBS = 1_500_000

# ── data ───────────────────────────────────────────────────────────────


@dataclass
class PrepBundle:
    """A prepared data set plus everything needed to reproduce it."""

    prep: object                      # PreparedObs
    recipe: dict                      # d_min, spacegroup, gain, radii, ...
    split_names: list[str] = field(default_factory=list)
    bg_var_factor: float = 0.0

    @property
    def spacegroup(self) -> str:
        return self.recipe.get("spacegroup") or "P 1"

    @property
    def merge_friedel(self) -> bool:
        """False when ``hkl`` holds signed ASU indices keeping Bijvoet mates apart."""
        return bool(self.recipe.get("merge_friedel", True))


def add_data_args(p: argparse.ArgumentParser, *, require_recipe: bool = True) -> None:
    """Stream inputs and the detector recipe.

    ``require_recipe=False`` for commands that read the recipe off a
    checkpoint; the flags then act as overrides.
    """
    p.add_argument("stream", type=Path, nargs="*",
                   help=".stream or .sfx input.  Omit when every input is "
                        "given as --split NAME=PATH.")
    p.add_argument("--split", action="append", default=[], metavar="NAME=PATH",
                   help="a named crystal split, one stream each; repeatable. "
                        "Splits are merged separately and never pooled in an "
                        "in-batch merge target.  No physical meaning is "
                        "assumed -- the name is just a label.")
    p.add_argument("--split-file", type=Path, default=None,
                   help="crystal splits for a single stream, either a "
                        "CrystFEL custom-split file (three whitespace columns, "
                        "'filename event dataset', the same file partialator "
                        "takes as --custom-split) or a .pt int64 label per "
                        "crystal after the d_min filter.  Every crystal must "
                        "be labelled. Repeat a filename/event with different "
                        "dataset names to include a frame in multiple splits.")
    p.add_argument("--spacegroup", default=None,
                   help='e.g. "P 1", "P 2 21 21".  Stamped into the MTZ and '
                        "used for centric detection.")
    p.add_argument("--int-radius", default=None, metavar="R_PEAK,R_GAP,R_BG",
                   help="indexamajig integration radii in pixels, e.g. 7,9,11. "
                        "Default: checkpoint recipe, then recorded source metadata. "
                        "Required if neither records a usable ring recipe.")
    p.add_argument("--gain", default=None, type=_gain_arg,
                   help="ADU/photon: auto (default), trust_input, calibrate, or a "
                        "positive number. Auto replaces the input gain when "
                        "calibration differs by more than 50%%.")
    p.add_argument("--dmin", type=float, default=None, help="high-resolution cut (A)")
    p.add_argument("--dmax", type=float, default=None, help="low-resolution cut (A)")
    p.add_argument("--cache", dest="cache_prep", type=Path, default=None,
                   help="prepared cache file; default RUN/prepared.pt. "
                        "Input content and preparation settings are validated.")
    p.add_argument("--merge-friedel", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="merge Friedel mates into a single reflection. "
                        "--no-merge-friedel keeps mates apart and writes (+)/(-) "
                        "columns. Default: the checkpoint recipe, then merge.")


# Preserve these import names for callers while keeping one resolver.
from torchsx.measurement import (parse_gain, parse_radii, GAIN_POLICY_VERSION,
                                 DEFAULT_GAIN_TOLERANCE)

def _gain_arg(value):
    try:
        return parse_gain(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc

def _workers():
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return max(1, min(8, available, int(os.environ.get("SLURM_CPUS_PER_TASK", available))))



def _comparable_prep_signature(signature: dict | None, recipe: dict) -> dict:
    """Compare effective preparation settings, retaining provenance in storage.

    A checkpoint freezes a previously resolved numeric gain. It can reuse the
    cache that estimated/read that gain, provided all source/settings checks
    still match. A new calibration request is not replaced by a cached number.
    """
    from copy import deepcopy
    result = deepcopy(signature or {})
    measurement = result.get("measurement")
    if measurement is not None:
        if measurement.get("source"):
            measurement["source"].pop("source_files", None)
        measurement.pop("resolved", None)  # radii/gain already occur at top level
        numeric = (recipe.get("measurement") or {}).get("gain_adu_per_photon")
        if numeric is not None:
            result["gain"] = numeric
            result.pop("gain_policy_version", None)
            result.pop("gain_tolerance", None)
    return result


def _splits_from_args(args) -> list[tuple[str, Path]]:
    """``[(name, path)]`` in split order; a bare stream is one unnamed split."""
    out: list[tuple[str, Path]] = []
    for entry in args.split or []:
        if "=" not in entry:
            raise SystemExit(f"--split wants NAME=PATH, got {entry!r}")
        name, _, path = entry.partition("=")
        if not name.strip() or not path.strip() or Path(name.strip()).name != name.strip() or name.strip() in (".", ".."):
            raise SystemExit(f"--split wants a nonempty NAME=PATH with a filename-safe name, got {entry!r}")
        out.append((name.strip(), Path(path.strip())))
    for path in args.stream or []:
        out.append(("", Path(path)))
    if not out:
        raise SystemExit("no input: give a stream path or --split NAME=PATH")
    if len(out) > 1 and any(not n for n, _ in out):
        raise SystemExit("mixing --split with unnamed streams is ambiguous; "
                         "name every input")
    if len({name for name, _ in out}) != len(out):
        raise SystemExit("Input split names must be unique")
    if args.split_file is not None and len(out) != 1:
        raise SystemExit("--split-file labels one input; use --split NAME=PATH for separate inputs")
    return out


def load_prep(args, *, recipe_defaults: dict | None = None, verbose: bool = True) -> PrepBundle:
    """Streams (or a cached blob) → ``PrepBundle``.

    All splits go through **one** ``prepare_observations`` call, so they share a
    single reflection index — which is what makes a difference between two
    splits meaningful at all.
    """
    from torchsx.merging.config import MergeConfig
    from torchsx.merging.hkl.filter import filter_resolution
    from torchsx.merging.hkl.noise import PoissonNoiseModel
    from torchsx.merging.observations import OBSERVATION_FILTER_VERSION, prepare_observations
    from torchsx.stream import SFXStream
    from torchsx.io.sources import inspect_source
    from torchsx.measurement import combine_measurements, resolve_measurement, validate_measurement

    d = dict(recipe_defaults or {})
    d_min = args.dmin if args.dmin is not None else d.get("d_min")
    spacegroup = args.spacegroup or d.get("spacegroup")
    merge_friedel = (bool(args.merge_friedel) if args.merge_friedel is not None
                     else bool(d.get("merge_friedel", True)))
    args.dmax = args.dmax if args.dmax is not None else d.get("d_max", 999.0)
    if d_min is None:
        raise SystemExit("--dmin is required")
    if not (np.isfinite(d_min) and np.isfinite(args.dmax) and 0 < d_min < args.dmax):
        raise SystemExit("Resolution cuts must be finite and satisfy 0 < dmin < dmax")

    splits = _splits_from_args(args)

    cache = args.cache_prep
    if cache is None and getattr(args, "out", None) is not None:
        run = args.ckpt.parent if getattr(args, "ckpt", None) else args.out
        cache = run / "prepared.pt"
    blob = torch.load(cache, weights_only=False) if cache is not None and cache.exists() else None
    if blob is not None:
        if (blob.get("signature") or {}).get("observation_filter_version") != OBSERVATION_FILTER_VERSION:
            raise SystemExit(
                f"Prepared cache {cache} uses an incompatible observation filter. "
                "Rebuild it from the source using a new --cache path; "
                "Cache validation cannot be bypassed."
            )
    def _stamp(path, *, precise=False):
        if path is None:
            return None
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
        return [Path(path).stat().st_size, digest.hexdigest()]

    signature = {
        "observation_filter_version": OBSERVATION_FILTER_VERSION,
        "cache_schema": 2,
        "splits": [n for n, _ in splits],
        "d_min": float(d_min), "d_max": float(args.dmax),
        "spacegroup": spacegroup,
        "merge_friedel": merge_friedel,
        "split_file": _stamp(args.split_file),
    }
    if args.split_file is not None:
        signature["split_file_version"] = 2  # Preserve overlapping assignments.

    sources = [path for _, path in splits]
    try:
        specs = [inspect_source(path) for path in sources]
        measurement = combine_measurements(specs)
        resolved = resolve_measurement(measurement, radii=args.int_radius,
                                       gain=args.gain, defaults=d)
        explicit_rings = args.int_radius is not None or d.get("radii") is not None
        for spec in specs:
            validate_measurement(spec, explicit_rings=explicit_rings)
        radii, gain = resolved.radii, resolved.gain
        if gain in ("auto", "calibrate"):
            signature["gain_policy_version"] = GAIN_POLICY_VERSION
        if gain == "auto":
            signature["gain_tolerance"] = DEFAULT_GAIN_TOLERANCE
        signature.update(radii=list(radii), gain=gain,
                         measurement={"source": measurement.to_dict() if measurement else None,
                                      "resolved": resolved.to_dict()},
                         source_files=[_stamp(path, precise=True) for _, path in splits])
        if verbose:
            origins = dict(resolved.provenance)
            print(f"  measurement: radii={radii} [{origins['radii']}], "
                  f"gain={gain} [{origins['gain']}]", flush=True)
            for override in resolved.overrides:
                print(f"    override: {override}", flush=True)
            if measurement and measurement.unresolved:
                print(f"    unresolved source fields: {', '.join(measurement.unresolved)}", flush=True)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Cannot resolve input measurement: {exc}") from exc

    if blob is not None:
        got = blob.get("signature")
        comparable_got = _comparable_prep_signature(got, {})
        comparable_signature = _comparable_prep_signature(signature, {})
        if isinstance(signature.get("gain"), (float, int)):
            comparable_got = _comparable_prep_signature(got, blob["recipe"])
        if comparable_got != comparable_signature:
            differ = sorted(
                k for k in set(comparable_signature) | set(comparable_got)
                if comparable_got.get(k) != comparable_signature.get(k))
            raise SystemExit(
                f"--cache {cache} was built with different settings "
                f"({', '.join(differ) if differ else 'no signature recorded'}); "
                f"delete it or point --cache somewhere else.")
        if verbose:
            print(f"prep from cache {cache} "
                  f"({blob['prep'].n_valid:,} reflections, "
                  f"{blob['prep'].I_obs.numel():,} obs, "
                  f"{len(blob.get('split_names', []))} splits)", flush=True)
        return PrepBundle(blob["prep"], blob["recipe"], blob.get("split_names", []),
                          blob["bg_var_factor"])

    parts = []
    for name, path in splits:
        # --spacegroup reaches the .sfx loader too, and is applied before
        # canonicalization: an .sfx cache carries whatever group it was parsed
        # with, and reducing a P 2 21 21 data set as P 1 marks every reflection
        # acentric and gives the wrong Wilson prior for 9 % of them.
        s = SFXStream.read(
            path, spacegroup=spacegroup,
            n_workers=_workers(), merge_friedel=merge_friedel,
        )
        s = SFXStream(filter_resolution(s._core, d_min=d_min, d_max=args.dmax))
        if verbose:
            print(f"  {name or path.name}: {s.n_crystals:,} crystals, "
                  f"{s.n_observations:,} obs (d >= {d_min} A)", flush=True)
        parts.append(s)
    stream = parts[0] if len(parts) == 1 else SFXStream.concat(parts)
    n_per_split = [p.n_crystals for p in parts]

    # ── splits → the per-crystal label the trainer already consumes ──
    names: list[str] = []
    if args.split_file is not None:
        if str(args.split_file).endswith((".pt", ".pth")):
            labels = torch.load(args.split_file).long().reshape(-1)
        else:
            # CrystFEL custom-split file: "filename event dataset", the unit
            # being a frame, not a crystal. All crystals of a frame belong
            # to each of its named splits.
            lookup = {}
            for raw in open(args.split_file):
                line = raw.split("#", 1)[0].split()
                if not line:
                    continue
                if len(line) < 3:
                    raise SystemExit(
                        f"{args.split_file}: expected 'filename event dataset', "
                        f"got {raw.rstrip()!r}")
                lookup.setdefault((line[0], line[1]), set()).add(line[2])
            names = sorted({name for bins in lookup.values() for name in bins})
            index = {n: i for i, n in enumerate(names)}
            fnames, evts = stream.filenames, stream.events
            if not fnames or not evts:
                raise SystemExit(
                    "the stream carries no image filename/event metadata, so a "
                    "custom-split file cannot be matched to it; re-parse from "
                    "the .stream, or pass a .pt label tensor")
            memberships = [sorted(index[name] for name in lookup.get((f, e), ()))
                           for f, e in zip(fnames, evts)]
            per_image = torch.tensor([bins[0] if bins else -1 for bins in memberships],
                                     dtype=torch.long)
            labels = per_image[stream.image_id.long()]
            if verbose:
                miss = int((labels < 0).sum())
                print(f"  custom split {args.split_file.name}: {len(lookup):,} "
                      f"frames, {len(names)} datasets, {miss:,} crystals "
                      f"unmatched", flush=True)
            # The trainer consumes one label per crystal. Add another copy
            # of the frame's crystals/observations for each extra membership.
            # Sets above prevent duplicate rows from double-counting a bin.
            parts, label_parts = [stream], [labels]
            for slot in range(1, max(map(len, memberships), default=0)):
                extra = torch.tensor([bins[slot] if len(bins) > slot else -1
                                      for bins in memberships], dtype=torch.long)
                extra = extra[stream.image_id.long()]
                mask = extra >= 0
                if mask.any():
                    parts.append(stream.filter_crystals(mask))
                    label_parts.append(extra[mask])
            stream = SFXStream.concat(parts)
            labels = torch.cat(label_parts)
        if labels.numel() != stream.n_crystals:
            raise SystemExit(
                f"--split-file has {labels.numel():,} labels but the data has "
                f"{stream.n_crystals:,} crystals (after the resolution filter)")
        if int((labels < 0).sum()):
            raise SystemExit(f"--split-file: {int((labels < 0).sum()):,} unlabelled "
                             "crystals (-1); every crystal must belong to a split")
        # A custom-split file defines state names even if filtering empties a state.
        n_splits = len(names) if names else int(labels.max()) + 1
        if not names:
            names = [str(i) for i in range(n_splits)]
        if len(names) != n_splits:
            raise SystemExit(f"{len(names)} split names for {n_splits} labels")
    elif len(splits) > 1:
        labels = torch.repeat_interleave(
            torch.arange(len(splits)), torch.tensor(n_per_split))
        names = [n for n, _ in splits]
    else:
        labels = None

    noise = PoissonNoiseModel(r_peak=radii[0], r_gap=radii[1], r_bg=radii[2],
                              gain=(gain if isinstance(gain, float) else 1.0))
    try:
        prep = prepare_observations(stream, MergeConfig(gain=gain), poisson_noise=noise,
                                    resolved_measurement=resolved)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    bvf = noise.n_peak / noise.n_bg


    if labels is not None:
        if labels.numel() != prep.obs_info.n_crystals:
            raise SystemExit(
                f"{labels.numel():,} split labels but prepare_observations kept "
                f"{prep.obs_info.n_crystals:,} crystals")
        prep.obs_info.delay_bin_idx = labels
        prep.obs_info.unique_timepoints = torch.arange(len(names), dtype=torch.float64)
        if verbose:
            counts = torch.bincount(labels, minlength=len(names)).tolist()
            print(f"  splits {names} -> crystals {counts}", flush=True)

    recipe = {"d_min": float(d_min), "d_max": float(args.dmax),
              "spacegroup": stream._core.spacegroup or spacegroup or "P 1",
              "gain": gain, "radii": list(radii),
              "gain_tolerance": DEFAULT_GAIN_TOLERANCE,
              "inputs": [[name, str(path.resolve())] for name, path in splits],
              "n_peak": noise.n_peak, "n_bg": noise.n_bg,
              "merge_friedel": merge_friedel,
              "measurement": {**resolved.to_dict(), "gain_adu_per_photon": noise.gain,
                              "n_peak": noise.n_peak, "n_bg": noise.n_bg,
                              "gain_selection": noise.gain_selection,
                              "pixel_count_definition": "nominal_integer_ring_mask",
                              "source": measurement.to_dict() if measurement else None}}
    previous_selection = (d.get("measurement") or {}).get("gain_selection")
    if previous_selection and dict(resolved.provenance)["gain"] == "checkpoint":
        recipe["measurement"]["gain_selection"] = dict(previous_selection, reused_from_checkpoint=True)
    if verbose:
        print(f"prepared: {prep.n_valid:,} reflections, {prep.I_obs.numel():,} obs, "
              f"{prep.obs_info.n_crystals:,} crystals, n_peak/n_bg={bvf:.4f}, "
              f"{100 * float(prep.is_centric.float().mean()):.1f} % centric", flush=True)
    bundle = PrepBundle(prep, recipe, names, bvf)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(cache.suffix + ".partial")
        torch.save({"prep": prep, "recipe": recipe, "split_names": names,
                    "bg_var_factor": bvf, "signature": signature}, tmp)
        # rename only once the write completed, so an interrupted run cannot
        # leave a truncated blob that the next one would load as valid
        tmp.replace(cache)
        if verbose:
            print(f"  prep cached -> {cache} "
                  f"({cache.stat().st_size / 2**30:.1f} GiB)", flush=True)
    return bundle


# ── free flags ─────────────────────────────────────────────────────────


def add_free_args(p: argparse.ArgumentParser) -> None:
    p.set_defaults(free_frac=0.05, free_seed=20260902, no_free=False)
    p.add_argument("--seed", type=int, default=None, help="random seed (default 0; merge reuses the run seed)")
    p.add_argument("--free-from", type=Path, default=None,
                   help="take the free set from an existing .mtz (FreeR_flag / "
                        "R-free-flags) or a .pkl of {(h,k,l): flag} on the "
                        "Friedel-canonical index.  Sharing the free set is the "
                        "only thing that makes two merges' R_free comparable, "
                        "so prefer this over generating a new one.")


def canonical_hkl(hkl: np.ndarray) -> np.ndarray:
    """Friedel-canonical Miller index: first non-zero component positive."""
    h = np.asarray(hkl, dtype=int).copy()
    for col in (0, 1, 2):
        undecided = np.ones(len(h), dtype=bool)
        for c in range(col):
            undecided &= h[:, c] == 0
        flip = undecided & (h[:, col] < 0)
        h[flip] *= -1
    return h


def work_mask(hkl: np.ndarray, args, *, verbose: bool = True) -> np.ndarray | None:
    """``True = working set, False = free`` (CCP4's 0 = free convention).

    Reflections absent from a shared free set go to the WORKING set: inventing
    them into the test set would score the model on data no other merge held
    out.
    """
    if args.no_free:
        return None
    n = len(hkl)
    if args.free_from is not None:
        if str(args.free_from).endswith(".pkl"):
            flags = pickle.load(open(args.free_from, "rb"))
            canon = canonical_hkl(hkl)
            work = np.array([int(flags.get(tuple(k), 1)) != 0 for k in canon], dtype=bool)
        else:
            import gemmi
            mtz = gemmi.read_mtz_file(str(args.free_from))
            lab = next((c.label for c in mtz.columns
                        if c.label in ("FreeR_flag", "R-free-flags", "FREE")), None)
            if lab is None:
                raise SystemExit(f"{args.free_from}: no free-flag column found")
            src = {tuple(int(v) for v in row[:3]): int(row[3])
                   for row in np.array(mtz, dtype=float)[:, [0, 1, 2, mtz.column_labels().index(lab)]]}
            canon = canonical_hkl(hkl)
            work = np.array([src.get(tuple(k), 1) != 0 for k in canon], dtype=bool)
        if verbose:
            print(f"  free set from {args.free_from.name}: "
                  f"{int((~work).sum()):,} of {n:,} held out "
                  f"({100 * float((~work).mean()):.2f} %)", flush=True)
        return work
    g = torch.Generator().manual_seed(args.free_seed)
    free = torch.zeros(n, dtype=torch.bool)
    free[torch.randperm(n, generator=g)[: int(round(args.free_frac * n))]] = True
    # Separated Friedel mates are two rows sharing one canonical key, and
    # splitting them across work and free leaks the held-out reflection.  Let the
    # first member of each group decide for both.  A no-op when mates are merged,
    # where the fold is injective and every group is a single row, so the draw is
    # bit-for-bit what it was before anomalous merging existed.
    _, first, group_id = np.unique(canonical_hkl(hkl), axis=0,
                                   return_index=True, return_inverse=True)
    free = free[torch.from_numpy(first[np.asarray(group_id).reshape(-1)]).long()]
    if verbose:
        print(f"  free set generated (seed {args.free_seed}): "
              f"{int(free.sum()):,} of {n:,} held out", flush=True)
    return (~free).numpy()


# ── output ─────────────────────────────────────────────────────────────


def write_mtz(path: Path, hkl: np.ndarray, F, F_sd, *, cell, spacegroup: str,
              work: np.ndarray | None, keep: np.ndarray, merge_friedel: bool = True,
              verbose: bool = True) -> None:
    """Write F/SIGF (+ free flags) and prove no intensity column landed.

    With ``merge_friedel=False`` the indices are signed, and ``ReflectionData``
    pivots the mates onto one ASU row carrying ``F-obs(+)``/``F-obs(-)``.
    """
    import gemmi
    from torchref import ReflectionData
    from torchref.symmetry import Cell, SpaceGroup

    path.parent.mkdir(parents=True, exist_ok=True)
    kt = torch.from_numpy(keep)
    data = ReflectionData.from_tensors(
        hkl=torch.from_numpy(hkl[keep]).int(),
        F=F.detach().cpu()[kt].float().clamp(min=1e-6),
        F_sigma=F_sd.detach().cpu()[kt].float().clamp(min=1e-6),
        cell=Cell(list(cell)),
        spacegroup=SpaceGroup(spacegroup),
        rfree_flags=(None if work is None else torch.from_numpy(work[keep])),
        friedel_merged=merge_friedel,
    )
    # Deliberately NOT setting data.I / data.I_sigma -- see the module docstring.
    data.write_mtz(str(path))

    mtz = gemmi.read_mtz_file(str(path))
    labels = [c.label for c in mtz.columns]
    bad = [l for l in labels if l.upper() in ("I", "IMEAN", "SIGI", "SIGIMEAN", "I-OBS")]
    if bad:
        raise SystemExit(f"REFUSING: intensity columns {bad} present in {path}; "
                         "ReflectionData.load would apply French-Wilson")
    if verbose:
        print(f"wrote {path}\n  reflections {int(keep.sum()):,}  columns {labels}",
              flush=True)
        free_label = next((l for l in ("FreeR_flag", "R-free-flags") if l in labels), None)
        if free_label is not None:
            fr = np.array(mtz.column_with_label(free_label), dtype=int)
            print(f"  readback free {100 * (fr == 0).mean():.2f} % "
                  f"({int((fr == 0).sum()):,} of {len(fr):,})", flush=True)


def write_intensity_mtz(path: Path, hkl: np.ndarray, I, I_sd, *, cell, spacegroup: str,
                        keep: np.ndarray, n_meas=None, merge_friedel: bool = True,
                        verbose: bool = True) -> None:
    """Write merged intensities as ``I-obs`` / ``SIGI-obs`` (no free flags).

    For the moment estimators, whose product is an intensity; French-Wilson
    happens downstream.  ``n_meas`` goes out as ``N``, aimless's spelling for the
    multiplicity.  With ``merge_friedel=False`` the signed indices are pivoted
    onto ``I-obs(+)``/``I-obs(-)``/``N(+)``/``N(-)``, since a negative ASU index
    in the file is not what a reader takes for a Bijvoet mate.
    """
    import pandas as pd
    from torchref.io.mtz import write

    def _np(t, dtype=np.float64):
        return np.asarray(t.detach().cpu().numpy() if hasattr(t, "detach") else t,
                          dtype=dtype)

    path.parent.mkdir(parents=True, exist_ok=True)
    I, I_sd = _np(I), _np(I_sd)
    keep = np.asarray(keep, dtype=bool) & np.isfinite(I) & np.isfinite(I_sd) & (I_sd > 0)

    cols = {"I-obs": I[keep], "SIGI-obs": I_sd[keep]}
    if n_meas is not None:
        cols["N"] = _np(n_meas, np.int32)[keep]
    if merge_friedel:
        rows = hkl[keep]
    else:
        rows, cols = _bijvoet_columns(hkl[keep], cols, spacegroup)
        # A mate nobody measured has no intensity, but its multiplicity is 0.
        for name in ("N(+)", "N(-)"):
            if name in cols:
                cols[name] = np.nan_to_num(cols[name]).astype(np.int32)
    df = pd.DataFrame({"H": rows[:, 0], "K": rows[:, 1], "L": rows[:, 2], **cols})
    write(df, [float(c) for c in cell], spacegroup, str(path))
    if verbose:
        print(f"wrote {path}\n  reflections {len(df):,}  "
              f"columns {list(df.columns)}", flush=True)


def _bijvoet_columns(hkl: np.ndarray, cols: dict,
                     spacegroup: str) -> tuple[np.ndarray, dict]:
    """Pivot signed ASU indices onto one row per reflection.

    A row is the I(-) member exactly when its index is outside the ASU, which is
    what :meth:`SFXStream.symmetrize` encodes by negating it.  Each column splits
    into a ``(+)``/``(-)`` pair, NaN where a mate was not measured.
    """
    import gemmi

    asu = gemmi.ReciprocalAsu(gemmi.SpaceGroup(spacegroup))
    plus = np.array([asu.is_in(row.tolist()) for row in hkl], dtype=bool)
    rows, inverse = np.unique(np.where(plus[:, None], hkl, -hkl), axis=0,
                              return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)

    out = {}
    for sel, suffix in ((plus, "(+)"), (~plus, "(-)")):
        for name, values in cols.items():
            col = out.setdefault(f"{name}{suffix}", np.full(len(rows), np.nan))
            col[inverse[sel]] = values[sel]
    return rows, out


def report_merge(res: dict, *, verbose: bool = True) -> None:
    """The diagnostics that have actually caught bugs in this pipeline."""
    if not verbose:
        return
    q = lambda t, p: float(t.quantile(torch.tensor(p, dtype=t.dtype, device=t.device)))
    seen = res["seen"]
    if not int(seen.sum()):
        # Nothing survived the merge.  Say so and return: quantile() on an
        # empty tensor raises, and a split whose grid missed every reflection
        # once took a whole run's remaining merges down with it.
        print(f"  estimator {res['estimator']}: NO reflections merged "
              f"({res['n_crystals']:,} crystals) -- the outer grid found no "
              f"support; check edge_mass and the locating rounds", flush=True)
        return
    if res["estimator"] in ("pooled", "mean"):
        I, sd = res["I"][seen], res["I_sd"][seen]
        ratio = I / sd.clamp(min=1e-12)
        print(f"  estimator {res['estimator']}: {int(seen.sum()):,} reflections, "
              f"median I/sig {float(ratio.median()):.2f}, "
              f"negative {100 * float((I < 0).float().mean()):.1f} %", flush=True)
        return
    if res.get("n_nonfinite_rows"):
        print(f"  NOTE {res['n_nonfinite_rows']:,} reflection-batch rows were "
              f"non-finite and were dropped from the accumulation", flush=True)
    F, S, E = res["F"][seen], res["F_sd"][seen], res["edge_mass"][seen]
    ratio = F / S.clamp(min=1e-12)
    # MEDIAN of the per-reflection ratio: F is heavily right-skewed, so
    # <F>/<sigF> is dominated by a few strong reflections and flatters the data.
    print(f"  F           median {float(F.median()):.3f}   <F> {float(F.mean()):.3f}")
    print(f"  F/sigF      median {float(ratio.median()):.2f}  "
          f"q10 {q(ratio, 0.1):.2f}  q90 {q(ratio, 0.9):.2f}")
    if res.get("loc_sd") is not None:
        sd = res["loc_sd"]
        print(f"  outer grid  sd med {float(sd.median()):.3f}  "
              f"q90 {float(sd.quantile(0.90)):.3f}  frac at clamp "
              f"{float((sd >= 0.999 * res['scan_halfwidth']).float().mean()):.4f}")
    # edge_mass is the truncation sentinel: a posterior clipped low is a
    # reflection indistinguishable from zero, clipped high means the grid is too
    # narrow for a real peak.  They call for opposite fixes.
    print(f"  edge split  low {float(res['edge_low'].median()):.2e}  "
          f"high {float(res['edge_high'].median()):.2e}  | "
          f"frac low>0.01 {float((res['edge_low'] > 0.01).float().mean()):.4f}  "
          f"frac high>0.01 {float((res['edge_high'] > 0.01).float().mean()):.4f}")
    print(f"  edge_mass   median {float(E.median()):.2e}  q99 {q(E, 0.99):.2e}  "
          f"max {float(E.max()):.2e}  | frac>0.01 "
          f"{100 * float((E > 0.01).double().mean()):.3f} %", flush=True)


# ── model stack ────────────────────────────────────────────────────────


#: The configuration the shipped numbers were measured with.  Two-width scatter
#: is ON: it took SwissFEL R_free 0.121 -> 0.119 (lowest shell 0.115 -> 0.103)
#: and rhodopsin 0.289 -> 0.276 with the low-resolution shell 0.344 -> 0.304.
#: The outer shells are still behind a shape-free pooled estimator; that trade
#: is deliberate.  ``lam_spread`` is absent on purpose -- the -0.25 carried by
#: every scored run was a silent no-op (``_lam_spread`` returns 0 for <= 0), so
#: shipping it as if it did something would be a lie in the config.
DEFAULTS = dict(
    d_model=128, n_layers=5, include_hkl=False, predict_shape=False,
    beta_init=1.0, w_init=1.65, context_folds=2, obs_context_layers=2,
    obs_self_layers=2, skew=True, miss=0.03, outlier=0.0, inner_points=192,
    k_coarse=63, m_fine=257, n_sigma=24.0, max_halfwidth=12.0,
    wilson_basis="smooth", wilson_degree=4, gauge_anchor="basis",
    scatter_family="mixture", scatter_components=4, scatter_shape="gaussian",
)


def add_model_args(p: argparse.ArgumentParser) -> None:
    """Flags shared by every command that builds the merge stack."""
    p.add_argument("--device", default=None, help="cuda | cpu")
    p.add_argument("--prior", choices=("none", "wilson", "joined-wilson"), default=None,
                   help="intensity prior: wilson (separate, default), joined-wilson (shared), or none "
                        "(flat in positive intensity); merge inherits the checkpoint unless set")


# Checkpoint ``kind`` markers for the corrected detection stack.  The first of
# each pair is what this package writes; the second is the out-of-tree
# experiment wrapper (``annulus_mean_only_v1``, jobs 205123/205124) whose
# trained models predate the parameter and are still worth merging.
_BG_MEAN_QUAD_KINDS = ("SplitLogGrid+bgmean", "AnnulusMeanOnlySplitLogGridV1")
_BG_MEAN_LIK_KINDS = ("Poisson+bgmean", "AnnulusMeanOnlyPoissonV1")


def bg_mean_offset_from_ckpt(ck: dict, bvf: float) -> float:
    """The background-mean offset a checkpoint was trained with.

    Both ``kind`` markers have to agree: a checkpoint carrying one alone is a
    write we do not know how to interpret, and guessing would merge the model
    at the wrong background convention -- silently, at exactly the scale that
    motivated the correction.
    """
    qkind = (ck.get("quadrature") or {}).get("kind")
    lkind = (ck.get("likelihood") or {}).get("kind")
    corrected = (qkind in _BG_MEAN_QUAD_KINDS, lkind in _BG_MEAN_LIK_KINDS)
    if any(corrected) and not all(corrected):
        raise SystemExit(
            f"checkpoint markers disagree: quadrature {qkind!r} vs likelihood "
            f"{lkind!r}; one says the background mean was corrected and the "
            "other does not")
    stored = (ck.get("likelihood") or {}).get("bg_mean_offset")
    if stored is not None:
        return float(stored)
    # The experiment wrapper recorded no offset; it always used c itself.
    return float(bvf) if any(corrected) else 0.0


def build_stack(bundle: PrepBundle, args, *, ckpt: dict | None = None,
                verbose: bool = True):
    """Build the production model, preserving recorded checkpoint geometry."""
    from torchsx.merging.likelihood import Poisson
    from torchsx.merging.quadrature.split_log_grid import SplitLogGrid
    from torchsx.merging.scale import AmortizedScale
    from torchsx.nn.objective import MarginalNLL

    ck = ckpt or {}
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    prep = bundle.prep  # Host residency bounds device memory independently of dataset size.
    # The predicted scatter families carry every per-observation shape parameter;
    # there are no per-HKL parameters to fit, so the objective's ``params`` slot
    # is empty.
    params = None

    enc = dict(ck.get("encoder") or {})
    sc = dict(ck.get("scale") or {})
    scat = dict(ck.get("scatter") or {})
    skew = bool(scat.get("predict_skew", DEFAULTS["skew"]) if scat
                else DEFAULTS["skew"])
    w_init = sc.get("w_init", DEFAULTS["w_init"])
    # scatter family: the checkpoint's own family wins over the defaults (a
    # checkpoint written before the key existed is the single family)
    scatter_family = scat.get("family", "gengauss") if ck else DEFAULTS["scatter_family"]
    n_components = int(scat.get("n_components") or DEFAULTS["scatter_components"])
    scatter_shape = str(scat.get("component_shape") or ("shared" if ck else DEFAULTS["scatter_shape"]))
    beta_init = sc.get("beta_init", DEFAULTS["beta_init"])
    kw = dict(
        d_model=enc.get("d_model") or sc.get("d_model") or DEFAULTS["d_model"],
        n_layers=enc.get("n_layers") or sc.get("n_layers") or DEFAULTS["n_layers"],
        include_hkl=bool(sc.get("include_hkl", DEFAULTS["include_hkl"])),
        predict_shape=bool(sc.get("predict_shape", DEFAULTS["predict_shape"])),
        beta_init=beta_init, w_init=w_init,
        context_folds=enc.get("context_folds", DEFAULTS["context_folds"] if getattr(args, "context_folds", None) is None
                              else int(args.context_folds)),
        obs_context_layers=enc.get("obs_context_layers", DEFAULTS["obs_context_layers"]),
        obs_self_layers=enc.get("obs_self_layers", DEFAULTS["obs_self_layers"]),
        scatter_family=scatter_family, n_components=n_components,
        component_shape=scatter_shape,
    )
    # A checkpoint's own head geometry wins over the defaults: rebuilding it
    # with different heads would fail the state_dict load, or load a subset.
    kw.update(
        predict_width=bool(sc.get("predict_width", True)),
        anchor_log_k=bool(sc.get("anchor_log_k", True)),
        anchor_stat=sc.get("anchor_stat",
                            "median" if ck else DEFAULTS["gauge_anchor"]),
        predict_skew=bool(skew),
    )
    scale = AmortizedScale.from_prep(prep, **kw).to(dev)
    scale.set_encoder_chunking(400)

    bvf = (ck.get("likelihood") or {}).get("bg_var_factor", bundle.bg_var_factor)
    bg_mean_offset = bg_mean_offset_from_ckpt(ck, bvf) if ck else float(bvf)
    if bg_mean_offset and float(bvf) <= 0.0:
        raise SystemExit(
            f"Background mean offset {bg_mean_offset:g} requires a positive "
            "background variance coefficient in the measurement/checkpoint recipe")
    quad = dict(ck.get("quadrature") or {})
    n_points = quad.get("n_points") or DEFAULTS["inner_points"]
    if quad and quad.get("kind") not in (None, "SplitLogGrid", *_BG_MEAN_QUAD_KINDS):
        raise SystemExit(f"checkpoint was trained with {quad['kind']}; these CLIs "
                         "support SplitLogGrid checkpoints only")
    miss = (DEFAULTS["miss"] if scat.get("miss_logit") is None and not scat
            else (0.5 if scat.get("miss_logit") is not None else 0.0))
    if ck:
        outlier = 0.5 if scat.get("outlier_logit") is not None else 0.0
    else:
        outlier = (getattr(args, "outlier_frac", None) if getattr(args, "outlier_frac", None) is not None
                   else DEFAULTS["outlier"])
    obj = MarginalNLL(SplitLogGrid(n_points=int(n_points)).to(dev),
                      Poisson(bg_var_factor=float(bvf),
                              bg_mean_offset=bg_mean_offset),
                      miss_frac=float(miss), outlier_frac=float(outlier)).to(dev)
    if scat.get("miss_logit") is not None and obj.miss_logit is not None:
        with torch.no_grad():
            obj.miss_logit.fill_(float(scat["miss_logit"]))
        miss = float(obj.miss_logit.detach().sigmoid())
    if scat.get("outlier_logit") is not None and obj.outlier_logit is not None:
        with torch.no_grad():
            obj.outlier_logit.fill_(float(scat["outlier_logit"]))
        outlier = float(obj.outlier_logit.detach().sigmoid())

    # The outer grid is deliberately NOT inherited from the checkpoint.  A
    # training run may integrate on a cheaper grid than a merge should: at
    # n_sigma 6 a quarter of the reflections put >1 % of their posterior against
    # the grid edge in a full merge, a defect invisible during training because
    # edge_mass is computed per batch and reads ~2e-3 there.  Every comparison
    # that has been scored was merged at 24 / 63 / 257, so that is the default
    # and the checkpoint's training grid is reported, not applied.
    grid = dict(ck.get("posterior_grid") or {})
    prior = getattr(args, "prior", None)
    if prior is None:
        prior = str(grid.get("prior", "wilson"))
    if prior == "flat_I0":  # Historical checkpoint spelling.
        prior = "none"
    if prior not in ("none", "wilson", "joined-wilson", "flat_log"):
        raise ValueError(f"Invalid checkpoint prior {prior!r}; expected 'none', 'wilson' or 'joined-wilson' "
                         "(legacy 'flat_I0' and 'flat_log' are also readable)")
    # A checkpoint decides its own prior and anchor; one written before the
    # smooth basis existed (no "wilson_basis" key) is rebuilt as it was trained,
    # in shells / median mode, not with today's defaults.
    legacy_ckpt = bool(ck) and "wilson_basis" not in grid
    wilson_basis = grid.get(
        "wilson_basis", "shells" if legacy_ckpt else DEFAULTS["wilson_basis"])
    wilson_degree = int(grid.get("wilson_degree", DEFAULTS["wilson_degree"]))
    # One smooth reciprocal-space basis for the prior's Sigma(h) and the scale's
    # gauge pin, built from the prepared reflections, the median cell and the
    # space group -- so a merge rebuilds exactly the basis the network trained on.
    basis = None
    if wilson_basis == "smooth" or getattr(scale, "anchor_stat", None) == "basis":
        from torchsx.merging.priors.reciprocal_basis import ReciprocalBasis
        basis = ReciprocalBasis.from_hkl(
            prep.hkl_valid.long(), prep.median_cell, bundle.spacegroup, degree=wilson_degree,
        )
    settings = dict(
        device=dev, max_obs=READOUT_MAX_OBS,
        k_coarse=int(DEFAULTS["k_coarse"]),
        m_fine=int(DEFAULTS["m_fine"]),
        n_sigma=float(DEFAULTS["n_sigma"]),
        max_halfwidth=float(DEFAULTS["max_halfwidth"]),
        prior=prior,
        bg_var_factor=float(bvf),
        bg_mean_offset=float(bg_mean_offset), inner_points=int(n_points),
        skew=bool(skew), miss=float(miss), outlier=float(outlier),
        wilson_basis=wilson_basis, wilson_degree=wilson_degree, basis=basis,
        scatter_family=scatter_family, scatter_components=int(n_components), scatter_shape=scatter_shape,
    )
    if verbose:
        print(f"intensity prior: {prior}"
              + (" (overriding checkpoint prior " + str(grid.get("prior", "wilson")) + ")"
                 if ck and getattr(args, "prior", None) is not None else ""), flush=True)
        if prior == "none":
            print("  none: integrate likelihood with dI on the finite intensity grid; "
                  "the missed-spot component prevents an unbounded flat posterior. "
                  "Check edge mass and integration-range sensitivity.", flush=True)
        print(f"stack: amortised scale, "
              f"{'two-width' if skew else 'symmetric'} "
              f"{f'{n_components}-component {scatter_shape}-shape mixture ' if scatter_family == 'mixture' else ''}scatter, "
              f"miss {miss:.3f}, " + (f"outlier {outlier:.4f}, " if outlier else "")
              + f"SplitLogGrid({n_points}), "
              f"outer {settings['k_coarse']}/{settings['m_fine']} "
              f"n_sigma {settings['n_sigma']:g}, "
              f"Poisson(bg_var={bvf:.4f}"
              + (f", bg_mean+{bg_mean_offset:.4f}" if bg_mean_offset else "")
              + f"), device {dev}", flush=True)
        if bg_mean_offset:
            print(f"  detection background mean is bg+{bg_mean_offset:.4f} "
                  f"({bg_mean_offset / float(bvf):.3f} x n_peak/n_bg); variance "
                  f"stays {bvf:.4f}*max(bg,0); raw bg still feeds the encoder, "
                  "the pooled estimator and the calibration", flush=True)
        if basis is not None:
            print(f"  smooth reciprocal basis: Chebyshev degree {wilson_degree} + "
                  f"{basis.n_aniso} invariant anisotropy tensor(s) ({bundle.spacegroup}), "
                  f"p = {basis.p}; prior Sigma {wilson_basis}, log K anchor "
                  f"{getattr(scale, 'anchor_stat', None)}", flush=True)
        if grid:
            print(f"  (checkpoint trained on outer grid "
                  f"{grid.get('k_coarse')}/{grid.get('m_fine')} "
                  f"n_sigma {grid.get('n_sigma')}, prior {grid.get('prior')})",
                  flush=True)
    return prep, params, scale, obj, settings
