"""Shared internals for ``torchsx-train`` / ``torchsx-merge-nn`` /
``torchsx-merge-global``.

Everything the three commands have in common and nothing else: turning stream
paths into a :class:`PreparedObs`, the detector recipe that goes with it, the
free-flag policy, and writing the result out.  The bench scripts each grew their
own copy of this; the copies drifted, and one of them merged a fold-encoder
checkpoint under the default encoder geometry and silently lost 0.022 in R_free.

Two contracts are enforced here rather than left to the caller:

**The measurement recipe must be resolved.** Radii and gain come from explicit
flags, a checkpoint, or recorded source metadata, in that order. Since
``bg = bg_per_pixel * n_peak``, substituting a default aperture can silently
rescale every background. Missing required settings remain explicit.

**No intensity column in a posterior MTZ.**  ``ReflectionData.load`` applies
French-Wilson whenever an intensity column is present and exposes no flag to
turn it off, so an ``I`` column would apply the Wilson prior a second time on
top of the posterior that already integrated against it.  The guarantee is
structural: F and SIGF are written, and the file is read back with gemmi to
prove it.  The moment estimators (``pooled``, ``mean``) are the deliberate
exception: their product *is* an intensity, so they are written as ``I-obs``
and take French-Wilson downstream like any other unmerged-derived intensity.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


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
                        "be labelled.")
    p.add_argument("--min-split-crystals", type=int, default=0, metavar="N",
                   help="drop any split with fewer than N crystals, from the "
                        "DATA and not just from the merge.  A split of a few "
                        "hundred crystals cannot support a merge worth making "
                        "(this project's own scan measured the signal peaking "
                        "at ~10 k crystals per bin), and it still gets its own "
                        "in-batch merge target during training, where the "
                        "per-reflection support is what keeps the target "
                        "finite.  0 keeps everything.")
    p.add_argument("--spacegroup", default=None,
                   help='e.g. "P 1", "P 2 21 21".  Stamped into the MTZ and '
                        "used for centric detection.")
    p.add_argument("--int-radius", default=None, metavar="R_PEAK,R_GAP,R_BG",
                   help="indexamajig integration radii in pixels, e.g. 7,9,11. "
                        "Default: checkpoint recipe, then recorded source metadata. "
                        "Required if neither records a usable ring recipe.")
    p.add_argument("--gain", default=None,
                   help="ADU/photon: 'stream' (declared adu_per_photon or "
                        "adu_per_eV x beam energy), 'auto', 'empirical', or a number. "
                        "Default: checkpoint recipe, then source declaration.")
    p.add_argument("--dmin", type=float, default=None, help="high-resolution cut (A)")
    p.add_argument("--dmax", type=float, default=999.0)
    p.add_argument("--cache-prep", type=Path, default=None,
                   help="read the prepared observations from here if the file "
                        "exists, else write them.  prepare_observations is ~10 "
                        "min on 133 M observations; nobody should pay it twice.")
    p.add_argument("--trust-cache-prep", action="store_true", default=False,
                   help="load --cache-prep without comparing its recorded "
                        "settings to the command line.  The signature stamps "
                        "the absolute stream and split-file paths, so a cache "
                        "moved to another machine is otherwise refused; the "
                        "inputs need not be present at all.  The recipe still "
                        "has to be passed for the network to see the same "
                        "settings, and it is the caller's job that it matches.")
    p.add_argument("--n-workers", type=int, default=8, help="stream parser workers")
    p.add_argument("--merge-friedel", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="merge Friedel mates into a single reflection (the "
                        "default).  --no-merge-friedel keeps them apart: the "
                        "merge is then over the point group rather than the "
                        "Laue class, and the products carry I(+)/I(-).  "
                        "Default: the checkpoint recipe, then merge.")
    p.add_argument('--experiment-partition', type=Path, default=None)
    p.add_argument('--experiment-half', type=int, choices=(0, 1), default=None)


# Preserve these import names for callers while keeping one resolver.
from torchsx.measurement import parse_gain, parse_radii


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
        measurement.pop("resolved", None)  # radii/gain already occur at top level
        numeric = (recipe.get("measurement") or {}).get("gain_adu_per_photon")
        if numeric is not None:
            result["gain"] = numeric
    return result


def _splits_from_args(args) -> list[tuple[str, Path]]:
    """``[(name, path)]`` in split order; a bare stream is one unnamed split."""
    out: list[tuple[str, Path]] = []
    for entry in args.split or []:
        if "=" not in entry:
            raise SystemExit(f"--split wants NAME=PATH, got {entry!r}")
        name, _, path = entry.partition("=")
        out.append((name.strip(), Path(path.strip())))
    for path in args.stream or []:
        out.append(("", Path(path)))
    if not out:
        raise SystemExit("no input: give a stream path or --split NAME=PATH")
    if len(out) > 1 and any(not n for n, _ in out):
        raise SystemExit("mixing --split with unnamed streams is ambiguous; "
                         "name every input")
    return out


def load_prep(args, *, recipe_defaults: dict | None = None, verbose: bool = True) -> PrepBundle:
    """Streams (or a cached blob) → ``PrepBundle``.

    All splits go through **one** ``prepare_observations`` call, so they share a
    single reflection index — which is what makes a difference between two
    splits meaningful at all.
    """
    from torchsx.merging.config import MergeConfig
    from torchsx.merging.hkl.filter import filter_crystals, filter_resolution
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
    if d_min is None:
        raise SystemExit("--dmin is required")

    splits = _splits_from_args(args)

    cache = args.cache_prep
    blob = torch.load(cache, weights_only=False) if cache is not None and cache.exists() else None
    if blob is not None:
        if (blob.get("signature") or {}).get("observation_filter_version") != OBSERVATION_FILTER_VERSION:
            raise SystemExit(
                f"Prepared cache {cache} uses an incompatible observation filter. "
                "Rebuild it from the source using a new --cache-prep path; "
                "--trust-cache-prep cannot bypass measurement filtering."
            )
    if blob is not None and getattr(args, "trust_cache_prep", False):
        if verbose:
            print(f"prep from cache {cache} (settings not checked): "
                  f"{blob['prep'].n_valid:,} reflections, "
                  f"{blob['prep'].I_obs.numel():,} obs, "
                  f"{len(blob.get('split_names', []))} splits", flush=True)
        from torchsx.nn.experiment_subset import maybe_subset
        return maybe_subset(PrepBundle(blob["prep"], blob["recipe"], blob.get("split_names", []),
                                       blob["bg_var_factor"]), args)

    # Everything that changes what prepare_observations returns.  A cache is
    # only reusable when all of it matches: the blob carries the split labels
    # baked into obs_info, so a cache built from a different --split-file would
    # silently merge the wrong crystals together.
    def _stamp(path, *, precise=False):
        if path is None:
            return None
        st = Path(path).stat()
        return [str(path), st.st_size, st.st_mtime_ns if precise else int(st.st_mtime)]

    signature = {
        "observation_filter_version": OBSERVATION_FILTER_VERSION,
        "splits": [[n, str(q)] for n, q in splits],
        "d_min": float(d_min), "d_max": float(args.dmax),
        "spacegroup": spacegroup,
        "merge_friedel": merge_friedel,
        "split_file": _stamp(args.split_file),
        "min_split_crystals": int(args.min_split_crystals),
    }

    legacy_cache = blob is not None and "measurement" not in (blob.get("signature") or {})
    if legacy_cache:
        # Old caches are reusable with their explicit/recorded recipe, without
        # pretending that their source semantics were validated retroactively.
        radii = args.int_radius if args.int_radius is not None else d.get("radii")
        if radii is None:
            raise SystemExit("Legacy prepared cache requires --int-radius or a checkpoint recipe.")
        gain = parse_gain(args.gain if args.gain is not None else d.get("gain"))
        signature.update(radii=list(parse_radii(radii)), gain=gain)
    else:
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
                f"--cache-prep {cache} was built with different settings "
                f"({', '.join(differ) if differ else 'no signature recorded'}); "
                f"delete it or point --cache-prep somewhere else.")
        if verbose:
            print(f"prep from cache {cache} "
                  f"({blob['prep'].n_valid:,} reflections, "
                  f"{blob['prep'].I_obs.numel():,} obs, "
                  f"{len(blob.get('split_names', []))} splits)", flush=True)
        from torchsx.nn.experiment_subset import maybe_subset
        return maybe_subset(PrepBundle(blob["prep"], blob["recipe"], blob.get("split_names", []),
                                       blob["bg_var_factor"]), args)

    parts = []
    for name, path in splits:
        # --spacegroup reaches the .sfx loader too, and is applied before
        # canonicalization: an .sfx cache carries whatever group it was parsed
        # with, and reducing a P 2 21 21 data set as P 1 marks every reflection
        # acentric and gives the wrong Wilson prior for 9 % of them.
        s = SFXStream.read(
            path, spacegroup=spacegroup,
            n_workers=args.n_workers, merge_friedel=merge_friedel,
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
            # being a frame, not a crystal -- multi-crystal frames put all
            # their crystals in the same split, which is what partialator
            # does with the same file.
            lookup = {}
            for raw in open(args.split_file):
                line = raw.split("#", 1)[0].split()
                if not line:
                    continue
                if len(line) < 3:
                    raise SystemExit(
                        f"{args.split_file}: expected 'filename event dataset', "
                        f"got {raw.rstrip()!r}")
                lookup[(line[0], line[1])] = line[2]
            names = sorted(set(lookup.values()))
            index = {n: i for i, n in enumerate(names)}
            fnames, evts = stream.filenames, stream.events
            if not fnames or not evts:
                raise SystemExit(
                    "the stream carries no image filename/event metadata, so a "
                    "custom-split file cannot be matched to it; re-parse from "
                    "the .stream, or pass a .pt label tensor")
            per_image = torch.tensor(
                [index.get(lookup.get((f, e), None), -1)
                 for f, e in zip(fnames, evts)], dtype=torch.long)
            labels = per_image[stream.image_id.long()]
            if verbose:
                miss = int((labels < 0).sum())
                print(f"  custom split {args.split_file.name}: {len(lookup):,} "
                      f"frames, {len(names)} datasets, {miss:,} crystals "
                      f"unmatched", flush=True)
        if labels.numel() != stream.n_crystals:
            raise SystemExit(
                f"--split-file has {labels.numel():,} labels but the data has "
                f"{stream.n_crystals:,} crystals (after --dmin/--crystals-file)")
        if int((labels < 0).sum()):
            raise SystemExit(f"--split-file: {int((labels < 0).sum()):,} unlabelled "
                             "crystals (-1); every crystal must belong to a split")
        # A custom-split file defines the datasets; a label tensor only implies
        # them.  A dataset may hold no crystal at all, which --min-split-crystals
        # then drops below -- the count must not be read off the labels present.
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

    # ── drop splits too small to merge ─────────────────────────────────
    if labels is not None and args.min_split_crystals > 0:
        counts = torch.bincount(labels, minlength=len(names))
        keep_split = counts >= args.min_split_crystals
        if not bool(keep_split.any()):
            raise SystemExit(
                f"--min-split-crystals {args.min_split_crystals:,} drops every "
                f"split; largest has {int(counts.max()):,} crystals")
        if not bool(keep_split.all()):
            gone = [(names[i], int(counts[i]))
                    for i in range(len(names)) if not keep_split[i]]
            keep_obs = keep_split[labels]
            if verbose:
                print(f"  dropping {len(gone)} split(s) under "
                      f"{args.min_split_crystals:,} crystals "
                      f"({int((~keep_obs).sum()):,} crystals): "
                      + ", ".join(f"{n} ({c:,})" for n, c in gone), flush=True)
            # Out of the DATA, not merely out of the merge: a split of a few
            # hundred crystals also gets its own in-batch merge target, and a
            # target with no support per reflection is what produced the
            # non-finite batch that killed a full run.
            stream = SFXStream(filter_crystals(stream._core, keep_obs))
            renumber = torch.cumsum(keep_split.long(), 0) - 1
            labels = renumber[labels[keep_obs]]
            names = [n for n, k in zip(names, keep_split.tolist()) if k]

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
              "n_peak": noise.n_peak, "n_bg": noise.n_bg,
              "merge_friedel": merge_friedel,
              "measurement": {**resolved.to_dict(), "gain_adu_per_photon": noise.gain,
                              "n_peak": noise.n_peak, "n_bg": noise.n_bg,
                              "pixel_count_definition": "nominal_integer_ring_mask",
                              "source": measurement.to_dict() if measurement else None}}
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
    from torchsx.nn.experiment_subset import maybe_subset
    return maybe_subset(bundle, args)


# ── free flags ─────────────────────────────────────────────────────────


def add_free_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--stream-batches", action="store_true",
                   help="keep the prepared observations on the host and copy "
                        "only each batch's rows to the device.  Required for "
                        "any data set that does not fit the card: 794 M "
                        "observations is ~38 GB before a single step.  Merging "
                        "needs it exactly as much as training does.")
    p.add_argument("--free-from", type=Path, default=None,
                   help="take the free set from an existing .mtz (FreeR_flag / "
                        "R-free-flags) or a .pkl of {(h,k,l): flag} on the "
                        "Friedel-canonical index.  Sharing the free set is the "
                        "only thing that makes two merges' R_free comparable, "
                        "so prefer this over generating a new one.")
    p.add_argument("--free-frac", type=float, default=0.05,
                   help="fraction held out when generating a new free set")
    p.add_argument("--free-seed", type=int, default=20260902)
    p.add_argument("--no-free", action="store_true",
                   help="write no R-free column")


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
    obs_self_layers=2, skew=True, miss=0.03, inner_points=192,
    k_coarse=63, m_fine=257, n_sigma=24.0, max_halfwidth=12.0,
    wilson_basis="smooth", wilson_degree=4, gauge_anchor="basis",
    scatter_family="gengauss", scatter_components=3, scatter_hidden=8, scatter_shape="shared",
)


def add_model_args(p: argparse.ArgumentParser) -> None:
    """Flags shared by every command that builds the merge stack."""
    p.add_argument("--scatter-family", default=None, choices=("gengauss", "mixture"),
                   help="scatter on the scale: one two-width GenGauss, or a mixture of "
                        "them centred on its mean with parameters from a small conditioner "
                        f"on the predicted width, shape and skew [default {DEFAULTS['scatter_family']}]")
    p.add_argument("--scatter-components", type=int, default=None,
                   help=f"mixture components [default {DEFAULTS['scatter_components']}]")
    p.add_argument("--scatter-hidden", type=int, default=None,
                   help=f"hidden units of the mixture conditioner [default {DEFAULTS['scatter_hidden']}]")
    p.add_argument("--scatter-shape", default=None, choices=("shared", "gaussian", "free", "partiality"),
                   help="component shapes of the mixture: the base beta for all, Gaussian (beta 2) for all, "
                        "one beta per component, or a partiality grid (Gaussian components at fixed support "
                        f"points of log p below the edge, one residual width) [default {DEFAULTS['scatter_shape']}]")
    p.add_argument("--inner-points", type=int, default=None,
                   help="SplitLogGrid nodes for the inner integral")
    p.add_argument("--k-coarse", type=int, default=None)
    p.add_argument("--m-fine", type=int, default=None)
    p.add_argument("--n-sigma", type=float, default=None,
                   help="outer-grid half-width in located sigmas")
    p.add_argument("--bg-mean-offset", default=None, metavar="C|auto",
                   help="add C to the background the DETECTION mean uses, "
                        "correcting the annulus estimate's bias E[B|A] = bg + "
                        "n_peak/n_bg; 'auto' uses n_peak/n_bg.  Raw bg still "
                        "feeds the encoder, the pooled estimator and the "
                        "calibration.  Default: off, or whatever the "
                        "checkpoint was trained with")
    p.add_argument("--device", default=None, help="cuda | cpu")


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


def resolve(args, ckpt: dict | None, key: str, *ckpt_path: str):
    """A setting from the flag, else the checkpoint, else ``DEFAULTS``."""
    v = getattr(args, key.replace("-", "_"), None)
    if v is not None:
        return v
    node = ckpt or {}
    for step in ckpt_path:
        if not isinstance(node, dict) or step not in node or node[step] is None:
            node = None
            break
        node = node[step]
    if node is not None and not isinstance(node, dict):
        return node
    return DEFAULTS.get(key)


def build_stack(bundle: PrepBundle, args, *, ckpt: dict | None = None,
                verbose: bool = True):
    """``(prep, params, scale, objective, settings)`` on the compute device.

    One builder for all three commands: the *only* difference between the
    trained merge and the global fit is the scale model handed back here.
    """
    from torchsx.merging.likelihood import Poisson
    from torchsx.merging.parameters import Parameters
    from torchsx.merging.quadrature.split_log_grid import SplitLogGrid
    from torchsx.merging.scale import AmortizedScale
    from torchsx.merging.scatter_shape import FreePerHKLShape
    from torchsx.nn.objective import MarginalNLL

    ck = ckpt or {}
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # --stream-batches means the per-observation state STAYS ON THE HOST and
    # only a batch's gathered rows cross the bus; the trainer's ``_batch``
    # reads ``self.prep.bg.device`` to decide that, so moving prep here made
    # the flag a no-op that copied device-to-device.  Harmless while the data
    # fit the card, fatal at 794 M observations: the move alone asks for ~38 GB
    # before a single step.  The per-reflection parameters still go to the
    # device -- they are what the optimiser touches.
    prep = bundle.prep if getattr(args, "stream_batches", False) else bundle.prep.to(dev)
    params = Parameters(
        log_I0_init=prep.mu_init.clone(),
        scatter_shape=FreePerHKLShape(prep.alpha_init.log(), prep.beta_perhkl.log()),
    ).to(dev)

    enc = dict(ck.get("encoder") or {})
    sc = dict(ck.get("scale") or {})
    scat = dict(ck.get("scatter") or {})
    skew = bool(scat.get("predict_skew", DEFAULTS["skew"]) if scat
                else DEFAULTS["skew"])
    w_init = sc.get("w_init", DEFAULTS["w_init"])
    # scatter family: the checkpoint's own family wins over the defaults (a
    # checkpoint written before the key existed is the single family)
    scatter_family = (args.scatter_family if args.scatter_family is not None
                      else (scat.get("family", "gengauss") if ck else DEFAULTS["scatter_family"]))
    n_components = (args.scatter_components if args.scatter_components is not None
                    else int(scat.get("n_components") or DEFAULTS["scatter_components"]))
    scatter_hidden = (args.scatter_hidden if args.scatter_hidden is not None
                      else int(scat.get("conditioner_hidden") or DEFAULTS["scatter_hidden"]))
    scatter_shape = (args.scatter_shape if args.scatter_shape is not None
                     else str(scat.get("component_shape") or DEFAULTS["scatter_shape"]))
    beta_init = sc.get("beta_init", DEFAULTS["beta_init"])
    kw = dict(
        d_model=enc.get("d_model") or sc.get("d_model") or DEFAULTS["d_model"],
        n_layers=enc.get("n_layers") or sc.get("n_layers") or DEFAULTS["n_layers"],
        include_hkl=bool(sc.get("include_hkl", DEFAULTS["include_hkl"])),
        predict_shape=bool(sc.get("predict_shape", DEFAULTS["predict_shape"])),
        beta_init=beta_init, w_init=w_init,
        context_folds=enc.get("context_folds", DEFAULTS["context_folds"]),
        obs_context_layers=enc.get("obs_context_layers", DEFAULTS["obs_context_layers"]),
        obs_self_layers=enc.get("obs_self_layers", DEFAULTS["obs_self_layers"]),
        scatter_family=scatter_family, n_components=n_components, conditioner_hidden=scatter_hidden,
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

    bvf = (ck.get("likelihood") or {}).get("bg_var_factor", bundle.bg_var_factor)
    # CLI beats the checkpoint beats OFF.  Deliberately NOT bundle-derived:
    # silence has to mean the legacy detection model, so every checkpoint
    # written before this parameter re-merges bit-exactly.
    if args.bg_mean_offset is None:
        bg_mean_offset = bg_mean_offset_from_ckpt(ck, bvf)
    elif str(args.bg_mean_offset).lower() == "auto":
        bg_mean_offset = float(bvf)
    else:
        bg_mean_offset = float(args.bg_mean_offset)
    if bg_mean_offset and float(bvf) <= 0.0:
        raise SystemExit(
            f"--bg-mean-offset {bg_mean_offset:g} needs a non-zero "
            "--bg-var-factor: both are n_peak/n_bg and describe the same "
            "annulus estimate, so correcting its mean without its variance is "
            "worse than correcting neither")
    quad = dict(ck.get("quadrature") or {})
    n_points = (args.inner_points if args.inner_points is not None
                else quad.get("n_points") or DEFAULTS["inner_points"])
    if quad and quad.get("kind") not in (None, "SplitLogGrid", *_BG_MEAN_QUAD_KINDS):
        raise SystemExit(f"checkpoint was trained with {quad['kind']}; these CLIs "
                         "ship SplitLogGrid only (see bench/to_mtz.py --quadrature)")
    miss = (DEFAULTS["miss"] if scat.get("miss_logit") is None and not scat
            else (0.5 if scat.get("miss_logit") is not None else 0.0))
    obj = MarginalNLL(SplitLogGrid(n_points=int(n_points)).to(dev),
                      Poisson(bg_var_factor=float(bvf),
                              bg_mean_offset=bg_mean_offset),
                      miss_frac=float(miss)).to(dev)
    if scat.get("miss_logit") is not None and obj.miss_logit is not None:
        with torch.no_grad():
            obj.miss_logit.fill_(float(scat["miss_logit"]))

    # The outer grid is deliberately NOT inherited from the checkpoint.  A
    # training run may integrate on a cheaper grid than a merge should: at
    # n_sigma 6 a quarter of the reflections put >1 % of their posterior against
    # the grid edge in a full merge, a defect invisible during training because
    # edge_mass is computed per batch and reads ~2e-3 there.  Every comparison
    # that has been scored was merged at 24 / 63 / 257, so that is the default
    # and the checkpoint's training grid is reported, not applied.
    grid = dict(ck.get("posterior_grid") or {})
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
        device=dev,
        k_coarse=int(args.k_coarse or DEFAULTS["k_coarse"]),
        m_fine=int(args.m_fine or DEFAULTS["m_fine"]),
        n_sigma=float(args.n_sigma or DEFAULTS["n_sigma"]),
        max_halfwidth=float(DEFAULTS["max_halfwidth"]),
        prior="wilson",
        bg_var_factor=float(bvf),
        bg_mean_offset=float(bg_mean_offset), inner_points=int(n_points),
        skew=bool(skew), miss=float(miss),
        wilson_basis=wilson_basis, wilson_degree=wilson_degree, basis=basis,
        scatter_family=scatter_family, scatter_components=int(n_components), scatter_shape=scatter_shape,
    )
    if verbose:
        print(f"stack: amortised scale, "
              f"{'two-width' if skew else 'symmetric'} "
              f"{f'{n_components}-component {scatter_shape}-shape mixture ' if scatter_family == 'mixture' else ''}scatter, "
              f"miss {miss:.3f}, SplitLogGrid({n_points}), "
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
