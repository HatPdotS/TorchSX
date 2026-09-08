"""Shared internals for ``torchsx-train`` / ``torchsx-merge-nn`` /
``torchsx-merge-global``.

Everything the three commands have in common and nothing else: turning stream
paths into a :class:`PreparedObs`, the detector recipe that goes with it, the
free-flag policy, and writing the result out.  The bench scripts each grew their
own copy of this; the copies drifted, and one of them merged a fold-encoder
checkpoint under the default encoder geometry and silently lost 0.022 in R_free.

Two contracts are enforced here rather than left to the caller:

**Integration radii are required.**  CrystFEL does not record them in the
stream, and ``bg = bg_per_pixel * n_peak`` — running the 7/9/11 default against
a 1/4/7 integration scales every background by 30x.  Nothing downstream can
detect that.

**No intensity column in an MTZ.**  ``ReflectionData.load`` applies
French-Wilson whenever an intensity column is present and exposes no flag to
turn it off, so an ``I`` column would apply the Wilson prior a second time on
top of the posterior that already integrated against it.  The guarantee is
structural: F and SIGF are written, and the file is read back with gemmi to
prove it.  The moment estimators (``pooled``, ``mean``) are the deliberate
exception and are written as a CrystFEL ``.hkl`` instead, so they take exactly
the French-Wilson path partialator's own output takes.
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
    p.add_argument("--split-names", default=None,
                   help="comma-separated names for --split-file labels, in "
                        "label order (default: 0, 1, ...)")
    p.add_argument("--spacegroup", default=None,
                   help='e.g. "P 1", "P 2 21 21".  Stamped into the MTZ and '
                        "used for centric detection.")
    p.add_argument("--int-radius", default=None, metavar="R_PEAK,R_GAP,R_BG",
                   help="indexamajig integration radii in pixels, e.g. 7,9,11. "
                        "REQUIRED: they are not in the stream, and bg scales "
                        "with n_peak." + ("" if require_recipe else "  Default: the checkpoint's."))
    p.add_argument("--gain", default=None,
                   help="ADU/photon: 'stream' (adu_per_eV x beam energy), "
                        "'auto', 'empirical', or a number.  Default 'stream'; "
                        "pass 1.0 when the DAQ already wrote photon counts.")
    p.add_argument("--dmin", type=float, default=None, help="high-resolution cut (A)")
    p.add_argument("--dmax", type=float, default=999.0)
    p.add_argument("--cache-prep", type=Path, default=None,
                   help="read the prepared observations from here if the file "
                        "exists, else write them.  prepare_observations is ~10 "
                        "min on 133 M observations; nobody should pay it twice.")
    p.add_argument("--sfx-cache", type=Path, default=None,
                   help="parsed-stream cache (.sfx) for a .stream input")
    p.add_argument("--n-workers", type=int, default=8, help="stream parser workers")
    p.add_argument("--no-friedel", dest="merge_friedel", action="store_false",
                   default=True, help="do not merge Friedel mates on load")
    p.add_argument("--crystals-file", type=Path, default=None,
                   help="bool mask over crystals (after the d_min filter) to keep")
    p.add_argument("--max-crystals", type=int, default=None,
                   help="parse only a subset of the stream (testing)")
    p.add_argument('--experiment-partition', type=Path, default=None)
    p.add_argument('--experiment-half', type=int, choices=(0, 1), default=None)


def parse_radii(spec: str) -> tuple[float, float, float]:
    parts = [float(x) for x in str(spec).replace(":", ",").split(",")]
    if len(parts) != 3:
        raise ValueError(f"--int-radius wants three numbers, got {spec!r}")
    if not parts[0] < parts[1] <= parts[2]:
        raise ValueError(f"--int-radius must be r_peak < r_gap <= r_bg, got {spec!r}")
    return parts[0], parts[1], parts[2]


def parse_gain(spec) -> str | float:
    if spec is None:
        return "stream"
    if isinstance(spec, (int, float)):
        return float(spec)
    if str(spec) in ("stream", "auto", "empirical"):
        return str(spec)
    return float(spec)


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
    from torchsx.merging.observations import prepare_observations
    from torchsx.stream import SFXStream

    d = dict(recipe_defaults or {})
    d_min = args.dmin if args.dmin is not None else d.get("d_min")
    spacegroup = args.spacegroup or d.get("spacegroup")
    radii = parse_radii(args.int_radius) if args.int_radius else (
        tuple(d["radii"]) if d.get("radii") else None)
    gain = parse_gain(args.gain) if args.gain is not None else d.get("gain", "stream")
    if radii is None:
        raise SystemExit(
            "--int-radius is required (r_peak,r_gap,r_bg from the indexamajig "
            "run).  They are not recorded in the stream and the background "
            "scales with n_peak.")
    if d_min is None:
        raise SystemExit("--dmin is required")

    splits = _splits_from_args(args)

    # Everything that changes what prepare_observations returns.  A cache is
    # only reusable when all of it matches: the blob carries the split labels
    # baked into obs_info, so a cache built from a different --split-file would
    # silently merge the wrong crystals together.
    def _stamp(path):
        if path is None:
            return None
        st = Path(path).stat()
        return [str(path), st.st_size, int(st.st_mtime)]

    signature = {
        "splits": [[n, str(q)] for n, q in splits],
        "d_min": float(d_min), "d_max": float(args.dmax),
        "spacegroup": spacegroup, "gain": gain, "radii": list(radii),
        "merge_friedel": bool(args.merge_friedel),
        "split_file": _stamp(args.split_file),
        "split_names": args.split_names,
        "min_split_crystals": int(args.min_split_crystals),
        "crystals_file": _stamp(args.crystals_file),
        "max_crystals": args.max_crystals,
    }

    cache = args.cache_prep
    if cache is not None and cache.exists():
        blob = torch.load(cache, weights_only=False)
        got = blob.get("signature")
        if got != signature:
            differ = sorted(
                k for k in set(signature) | set(got or {})
                if (got or {}).get(k) != signature.get(k))
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
        if str(path).endswith(".sfx"):
            s = SFXStream.load(path, merge_friedel=args.merge_friedel)
        else:
            s = SFXStream.from_stream(
                path, spacegroup=spacegroup, cache=args.sfx_cache,
                n_workers=args.n_workers, merge_friedel=args.merge_friedel,
                max_crystals=args.max_crystals,
            )
        if spacegroup and (s._core.spacegroup or "P 1") != spacegroup:
            # An .sfx cache carries whatever space group it was parsed with, and
            # prepare_observations reads centric reflections off it.  Silently
            # merging a P 2 21 21 data set as P 1 marks every reflection
            # acentric and gives the wrong Wilson prior for 9 % of them.
            print(f"    spacegroup {s._core.spacegroup!r} -> {spacegroup!r} "
                  f"(from --spacegroup)", flush=True)
            s = SFXStream(s._core.replace(spacegroup=spacegroup))
        s = SFXStream(filter_resolution(s._core, d_min=d_min, d_max=args.dmax))
        if verbose:
            print(f"  {name or path.name}: {s.n_crystals:,} crystals, "
                  f"{s.n_observations:,} obs (d >= {d_min} A)", flush=True)
        parts.append(s)
    stream = parts[0] if len(parts) == 1 else SFXStream.concat(parts)
    n_per_split = [p.n_crystals for p in parts]

    if args.crystals_file is not None:
        mask = torch.load(args.crystals_file)
        stream = SFXStream(filter_crystals(stream._core, mask))
        if len(parts) > 1:
            raise SystemExit("--crystals-file with several splits is ambiguous")
        n_per_split = [stream.n_crystals]
        if verbose:
            print(f"  crystals: {int(mask.sum()):,} of {int(mask.numel()):,} kept",
                  flush=True)

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
        n_splits = int(labels.max()) + 1
        if args.split_names:
            names = [n.strip() for n in args.split_names.split(",")]
        elif not names:
            names = [str(i) for i in range(n_splits)]
        if len(names) != n_splits:
            raise SystemExit(f"--split-names has {len(names)} names for {n_splits} labels")
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
    prep = prepare_observations(stream, MergeConfig(gain=gain), poisson_noise=noise)
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
              "merge_friedel": bool(args.merge_friedel)}
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
    p.add_argument("--locate-windows", default=None, metavar="W1,W2,...",
                   help="scan half-widths (nats) for the merge's locating "
                        "passes, one data pass each.  Default 8,1,0.125: a "
                        "single wide pass mis-centres the outer grid by ~0.59 "
                        "nats at any multiplicity, which is harmless below "
                        "~1e3 counts per reflection and fatal above ~1e4 "
                        "(all-NaN F on a 422 k-crystal split).  Pass a single "
                        "value for the old one-pass behaviour.")
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
    if verbose:
        print(f"  free set generated (seed {args.free_seed}): "
              f"{int(free.sum()):,} of {n:,} held out", flush=True)
    return (~free).numpy()


# ── output ─────────────────────────────────────────────────────────────


def write_mtz(path: Path, hkl: np.ndarray, F, F_sd, *, cell, spacegroup: str,
              work: np.ndarray | None, keep: np.ndarray, verbose: bool = True) -> None:
    """Write F/SIGF (+ free flags) and prove no intensity column landed."""
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
        if "FreeR_flag" in labels:
            fr = np.array(mtz.column_with_label("FreeR_flag"), dtype=int)
            print(f"  readback free {100 * (fr == 0).mean():.2f} % "
                  f"({int((fr == 0).sum()):,} of {len(fr):,})", flush=True)


def write_hkl(path: Path, hkl: np.ndarray, I, I_sd, n_meas, keep: np.ndarray,
              *, point_group: str = "-1", verbose: bool = True) -> None:
    """CrystFEL reflection list, for the identical French-Wilson path
    partialator's own output takes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    I = I.detach().cpu().numpy()
    I_sd = I_sd.detach().cpu().numpy()
    n_meas = n_meas.detach().cpu().numpy()
    n_written = 0
    with open(path, "w") as f:
        f.write("CrystFEL reflection list version 2.0\n")
        f.write(f"Symmetry: {point_group}\n")
        f.write("   h    k    l          I    phase   sigma(I)   nmeas\n")
        for (h, k, l), Ii, ss, nn, ok in zip(hkl, I, I_sd, n_meas, keep):
            if ok and np.isfinite(Ii) and ss > 0:
                f.write(f"{h:4d} {k:4d} {l:4d} {Ii:10.2f}        - "
                        f"{ss:10.2f} {int(nn):7d}\n")
                n_written += 1
        f.write("End of reflections\n")
    if verbose:
        print(f"wrote {path}: {n_written:,} reflections", flush=True)


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
)


def add_model_args(p: argparse.ArgumentParser) -> None:
    """Flags shared by every command that builds the merge stack."""
    p.add_argument("--skew", dest="skew", action="store_true", default=None,
                   help="two-width (asymmetric) GenGauss scatter [default on]")
    p.add_argument("--no-skew", dest="skew", action="store_false",
                   help="symmetric scatter")
    p.add_argument("--miss", type=float, default=None,
                   help=f"miss-component fraction, 0 = off "
                        f"[default {DEFAULTS['miss']}]")
    p.add_argument("--w-init", type=float, default=None)
    p.add_argument("--beta-init", type=float, default=None)
    p.add_argument("--inner-points", type=int, default=None,
                   help="SplitLogGrid nodes for the inner integral")
    p.add_argument("--k-coarse", type=int, default=None)
    p.add_argument("--m-fine", type=int, default=None)
    p.add_argument("--n-sigma", type=float, default=None,
                   help="outer-grid half-width in located sigmas")
    p.add_argument("--max-halfwidth", type=float, default=None)
    p.add_argument("--posterior-prior", default=None,
                   choices=("wilson", "flat_I0"),
                   help="prior on the outer (signal) integral [default wilson]")
    p.add_argument("--bg-var-factor", type=float, default=None,
                   help="background-estimate variance coefficient; default "
                        "n_peak/n_bg from --int-radius")
    p.add_argument("--bg-var-mode", choices=("deviance", "second_order"), default=None)
    p.add_argument("--device", default=None, help="cuda | cpu")


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
                global_scale: bool = False, verbose: bool = True):
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
    skew = resolve(args, ck, "skew") if args.skew is not None else bool(
        scat.get("predict_skew", scat.get("global_skew", DEFAULTS["skew"]))
        if scat else DEFAULTS["skew"])
    w_init = args.w_init if args.w_init is not None else sc.get("w_init", DEFAULTS["w_init"])
    beta_init = (args.beta_init if args.beta_init is not None
                 else sc.get("beta_init", DEFAULTS["beta_init"]))
    kw = dict(
        d_model=enc.get("d_model") or sc.get("d_model") or DEFAULTS["d_model"],
        n_layers=enc.get("n_layers") or sc.get("n_layers") or DEFAULTS["n_layers"],
        include_hkl=bool(sc.get("include_hkl", DEFAULTS["include_hkl"])),
        predict_shape=bool(sc.get("predict_shape", DEFAULTS["predict_shape"])),
        beta_init=beta_init, w_init=w_init,
        context_folds=enc.get("context_folds", DEFAULTS["context_folds"]),
        obs_context_layers=enc.get("obs_context_layers", DEFAULTS["obs_context_layers"]),
        obs_self_layers=enc.get("obs_self_layers", DEFAULTS["obs_self_layers"]),
    )
    if global_scale:
        # No per-observation content at all: K == 1 and one global scalar each
        # for width, shape and skew.  Nothing reads the embedding, so
        # AmortizedScale.predict skips the features and the encoder outright.
        kw.update(predict_width=False, predict_shape=False, anchor_log_k=False,
                  unity_k=True, predict_skew=False, global_skew=bool(skew))
    else:
        # A checkpoint's own head geometry wins over the defaults: rebuilding a
        # globally-fitted checkpoint with per-observation heads would fail the
        # state_dict load, and rebuilding it the other way would load a subset.
        kw.update(
            predict_width=bool(sc.get("predict_width", True)),
            unity_k=bool(sc.get("unity_k", False)),
            anchor_log_k=bool(sc.get("anchor_log_k", True)),
            anchor_stat=sc.get("anchor_stat", "median"),
            global_skew=bool(scat.get("global_skew", False)),
            predict_skew=bool(skew) and not scat.get("global_skew", False),
        )
    scale = AmortizedScale.from_prep(prep, **kw).to(dev)

    bvf = (args.bg_var_factor if args.bg_var_factor is not None
           else (ck.get("likelihood") or {}).get("bg_var_factor", bundle.bg_var_factor))
    mode = (args.bg_var_mode
            or (ck.get("likelihood") or {}).get("bg_var_mode") or "deviance")
    quad = dict(ck.get("quadrature") or {})
    n_points = (args.inner_points if args.inner_points is not None
                else quad.get("n_points") or DEFAULTS["inner_points"])
    if quad and quad.get("kind") not in (None, "SplitLogGrid"):
        raise SystemExit(f"checkpoint was trained with {quad['kind']}; these CLIs "
                         "ship SplitLogGrid only (see bench/to_mtz.py --quadrature)")
    miss = args.miss if args.miss is not None else (
        DEFAULTS["miss"] if scat.get("miss_logit") is None and not scat
        else (0.5 if scat.get("miss_logit") is not None else 0.0))
    obj = MarginalNLL(SplitLogGrid(n_points=int(n_points)).to(dev),
                      Poisson(bg_var_factor=float(bvf), bg_var_mode=mode),
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
    settings = dict(
        device=dev,
        k_coarse=int(args.k_coarse or DEFAULTS["k_coarse"]),
        m_fine=int(args.m_fine or DEFAULTS["m_fine"]),
        n_sigma=float(args.n_sigma or DEFAULTS["n_sigma"]),
        max_halfwidth=float(args.max_halfwidth or DEFAULTS["max_halfwidth"]),
        prior=args.posterior_prior or "wilson",
        bg_var_factor=float(bvf), bg_var_mode=mode, inner_points=int(n_points),
        skew=bool(skew), miss=float(miss),
    )
    if verbose:
        print(f"stack: {'global (K=1)' if global_scale else 'amortised'} scale, "
              f"{'two-width' if skew else 'symmetric'} scatter, "
              f"miss {miss:.3f}, SplitLogGrid({n_points}), "
              f"outer {settings['k_coarse']}/{settings['m_fine']} "
              f"n_sigma {settings['n_sigma']:g}, "
              f"Poisson(bg_var={bvf:.4f}, {mode}), device {dev}", flush=True)
        if grid:
            print(f"  (checkpoint trained on outer grid "
                  f"{grid.get('k_coarse')}/{grid.get('m_fine')} "
                  f"n_sigma {grid.get('n_sigma')}, prior {grid.get('prior')})",
                  flush=True)
    return prep, params, scale, obj, settings
