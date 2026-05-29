"""Times-file loader and filename-derived timepoint helpers.

Two ways to attach per-image timepoints to a stream:

1. **Times file** (3-col whitespace, ``filename event timepoint``) —
   see :func:`load_times_file` + :func:`assign_timepoints_to_images`.
2. **Filename-derived** — one ``.sfx`` file per timepoint, with the
   timepoint embedded in the filename.  See
   :func:`load_streams_with_filename_timepoints`.

Times-file format
-----------------
Three whitespace-separated columns: ``filename event timepoint``.
Comment lines (``#``-prefixed) and blank lines are skipped.  No header.

Example::

    # crystal-by-crystal pump–probe timestamps (fs)
    run123_001.cxi   //0   -45.0
    run123_001.cxi   //1   -45.0
    run123_002.cxi   //0   -91.0

Public entry points
-------------------
``load_times_file(path)`` → ``dict[(filename, event), float]``
``assign_timepoints_to_images(filenames, events, lookup)`` →
    ``(timepoints, report)``.
``load_streams_with_filename_timepoints(paths, pattern=...)`` →
    one labelled :class:`SFXStream` with each image carrying the
    timepoint parsed from its source filename.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


@dataclass
class TimesAssignmentReport:
    """Counts and per-timepoint breakdown for an image→timepoint assignment.

    Useful to print after a stream has been labelled so the user can
    see how many images were matched and how many of the times-file
    entries went unused (typical when the .stream is a subset of the
    full experiment).
    """

    n_times_rows: int                    # rows in the times file
    n_images: int                        # images in the stream
    n_matched: int                       # images that received a timepoint
    per_timepoint_counts: dict[float, int] = field(default_factory=dict)
    n_unused_entries: int = 0            # times-file rows whose key never matched

    @property
    def n_unmatched_images(self) -> int:
        return self.n_images - self.n_matched

    @property
    def match_fraction(self) -> float:
        return self.n_matched / self.n_images if self.n_images else 0.0

    def format(self, path: str | Path | None = None) -> str:
        lines = []
        src = f" from {path}" if path is not None else ""
        lines.append(f"Times file: {self.n_times_rows} rows{src}")
        lines.append(f"Images in stream: {self.n_images}")
        pct = 100.0 * self.match_fraction
        lines.append(f"Matched: {self.n_matched} / {self.n_images} ({pct:.1f}%)")
        lines.append(f"Unmatched (no entry in times file): {self.n_unmatched_images}")
        if self.per_timepoint_counts:
            sorted_t = sorted(self.per_timepoint_counts.keys())
            lines.append(f"Unique timepoints found: {sorted_t}")
            counts_repr = ", ".join(
                f"{t}: {self.per_timepoint_counts[t]}" for t in sorted_t
            )
            lines.append(f"Per-timepoint counts: {{{counts_repr}}}")
        else:
            lines.append("Unique timepoints found: []")
        lines.append(f"Times-file entries not used: {self.n_unused_entries}")
        return "\n".join(lines)


def load_times_file(path: str | Path) -> dict[tuple[str, str], float]:
    """Parse a 3-column whitespace-separated ``filename event timepoint`` file.

    Returns a ``(filename, event) → timepoint`` dict.  Duplicate keys
    are overwritten by the last-seen value (a warning is printed).
    Malformed rows raise ``ValueError`` with the offending line number.
    """
    path = Path(path)
    lookup: dict[tuple[str, str], float] = {}
    with open(path, "r") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(
                    f"{path}:{line_no}: expected 3 whitespace-separated "
                    f"columns 'filename event timepoint', got {len(parts)}: "
                    f"{raw.rstrip()!r}"
                )
            filename = parts[0]
            event = parts[1]
            try:
                tp = float(parts[2])
            except ValueError as e:
                raise ValueError(
                    f"{path}:{line_no}: third column not parseable as "
                    f"float: {parts[2]!r}"
                ) from e
            key = (filename, event)
            if key in lookup and lookup[key] != tp:
                print(
                    f"  [times] {path}:{line_no}: duplicate key {key!r} "
                    f"(previous={lookup[key]}, new={tp}); keeping new"
                )
            lookup[key] = tp
    return lookup


def assign_timepoints_to_images(
    filenames: Iterable[str],
    events: Iterable[str],
    lookup: dict[tuple[str, str], float],
) -> tuple[list[float], TimesAssignmentReport]:
    """Look up a timepoint per image; unmatched images get ``NaN``.

    Both inputs must be aligned per-image lists.  Matching is exact on
    the ``(filename, event)`` tuple — no fuzzy matching, no event-string
    canonicalisation.  Unused times-file rows (no image matched them)
    are counted in the report.
    """
    fnames = list(filenames)
    evts = list(events)
    if len(fnames) != len(evts):
        raise ValueError(
            f"filenames and events must align: got {len(fnames)} vs {len(evts)}"
        )
    n_images = len(fnames)

    timepoints: list[float] = []
    used_keys: set[tuple[str, str]] = set()
    per_t: dict[float, int] = {}
    n_matched = 0

    for i in range(n_images):
        key = (fnames[i], evts[i])
        tp = lookup.get(key)
        if tp is None:
            timepoints.append(float("nan"))
        else:
            timepoints.append(float(tp))
            used_keys.add(key)
            per_t[tp] = per_t.get(tp, 0) + 1
            n_matched += 1

    return timepoints, TimesAssignmentReport(
        n_times_rows=len(lookup),
        n_images=n_images,
        n_matched=n_matched,
        per_timepoint_counts=per_t,
        n_unused_entries=len(lookup) - len(used_keys),
    )


def unique_finite_timepoints(timepoints: Iterable[float]) -> list[float]:
    """Sorted unique non-NaN timepoints — convenient for kinetic-model grids."""
    out: set[float] = set()
    for t in timepoints:
        if t is None:
            continue
        if isinstance(t, float) and math.isnan(t):
            continue
        out.add(float(t))
    return sorted(out)


# ── Filename-derived timepoints ──────────────────────────────────────────


_DEFAULT_TIMEPOINT_PATTERN = r"(?P<t>[+-]?\d+(?:\.\d+)?)"


def parse_timepoint_from_filename(
    name: str,
    pattern: str = _DEFAULT_TIMEPOINT_PATTERN,
) -> float:
    """Parse a float timepoint out of ``name`` using a regex.

    Parameters
    ----------
    name :
        Filename (or full path — only ``Path(name).name`` is searched).
    pattern :
        Regex with a named group ``(?P<t>...)`` capturing the timepoint.
        The default matches the **first** signed decimal in the
        filename, which works for names like ``delay-45.sfx``,
        ``run_t100.sfx``, ``run-001_+1.5ps.sfx``.  Override when the
        first number is something else (run number, event id, etc.).

    Returns
    -------
    float

    Raises
    ------
    ValueError if the pattern matches no number in the filename.
    """
    stem = Path(name).name
    m = re.search(pattern, stem)
    if m is None:
        raise ValueError(
            f"timepoint pattern {pattern!r} did not match in filename {stem!r}"
        )
    try:
        # Prefer the named group if present, else group(0) / group(1).
        gd = m.groupdict()
        if "t" in gd and gd["t"] is not None:
            raw = gd["t"]
        else:
            raw = m.group(1) if m.groups() else m.group(0)
        return float(raw)
    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"failed to parse float from regex match {m.group(0)!r} "
            f"in filename {stem!r}: {exc}"
        ) from exc


def load_streams_with_filename_timepoints(
    paths: Sequence[str | Path],
    *,
    pattern: str = _DEFAULT_TIMEPOINT_PATTERN,
    merge_friedel: bool = True,
    spacegroup: str | None = None,
    verbose: bool = True,
):
    """Load multiple ``.sfx`` files, label each by a filename-derived timepoint, concatenate.

    Each input file is assumed to represent one timepoint of a
    time-resolved experiment.  The timepoint is parsed out of the
    filename via the regex ``pattern`` (see
    :func:`parse_timepoint_from_filename`).  After loading, every image
    in the source stream receives that timepoint as its
    ``metadata['image']['timepoint']`` entry, and all streams are
    concatenated via :meth:`SFXStream.concat` to a single labelled
    stream.

    Parameters
    ----------
    paths :
        Iterable of ``.sfx`` paths.  Order is irrelevant — the parsed
        timepoint determines the temporal ordering of the resulting
        stream's metadata.
    pattern :
        Regex matched against ``Path(p).name`` for each entry.  Must
        capture a float-parseable string (named group ``t`` or group 1
        or group 0).  Default: first signed decimal in the filename.
    merge_friedel :
        Forwarded to :meth:`SFXStream.load`.
    verbose :
        Print a per-file summary line and the unique timepoints found.

    Returns
    -------
    SFXStream
        A single concatenated stream with every image carrying a
        finite ``timepoint`` metadata entry.
    """
    from torchsx.stream import SFXStream
    import dataclasses
    from copy import deepcopy

    if len(paths) == 0:
        raise ValueError("paths must contain at least one .sfx file")

    parsed: list[tuple[float, Path]] = []
    for p in paths:
        p = Path(p)
        t = parse_timepoint_from_filename(p.name, pattern)
        parsed.append((t, p))
    # Sort by parsed timepoint so the concatenated stream is naturally ordered.
    parsed.sort(key=lambda pt: pt[0])

    if verbose:
        print(
            f"[filename-timepoints] {len(parsed)} files, "
            f"pattern={pattern!r}",
            flush=True,
        )

    labelled_streams = []
    for t, p in parsed:
        # If the caller overrides the spacegroup, load with the requested
        # merge-Friedel setting; for P1 / centrosymmetric P-1 / etc. the
        # appropriate Friedel behaviour is the caller's call.
        s = SFXStream.load(p, merge_friedel=merge_friedel)
        n_images = s.n_images
        new_meta = deepcopy(s._core.metadata) if s._core.metadata else {}
        if "image" not in new_meta:
            new_meta["image"] = {}
        new_meta["image"]["timepoint"] = [float(t)] * n_images
        replace_kwargs: dict[str, object] = {"metadata": new_meta}
        if spacegroup is not None:
            replace_kwargs["spacegroup"] = spacegroup
        new_core = dataclasses.replace(s._core, **replace_kwargs)
        labelled_streams.append(SFXStream(new_core))
        if verbose:
            print(
                f"  t={t:+.4g}  ←  {p.name}  "
                f"(n_images={n_images}, n_crystals={s.n_crystals})",
                flush=True,
            )

    combined = SFXStream.concat(labelled_streams)
    if verbose:
        uniq = unique_finite_timepoints(combined.image_timepoints)
        print(
            f"[filename-timepoints] combined: n_images={combined.n_images}, "
            f"n_crystals={combined.n_crystals}, unique timepoints={uniq}",
            flush=True,
        )
    return combined
