"""Times-file loader: assign per-image timepoints from a 3-column whitespace file.

Format
------
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
    ``(timepoints, report)`` with permissive matching and a printable
    summary report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


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
