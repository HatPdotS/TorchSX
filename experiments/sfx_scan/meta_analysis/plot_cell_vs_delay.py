#!/usr/bin/env python
"""Plot unit cell parameters and metadata vs. pump–probe delay."""

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
SFX_SCAN_DIR = SCRIPT_DIR.parent

sys.path.insert(0, str(SFX_SCAN_DIR.parent))
from torchsx.io._parser import parse_stream

STREAM_DIR = SFX_SCAN_DIR / "streams"
CACHE_FILE = SCRIPT_DIR / "cell_vs_delay_cache.json"
OUT_PNG = SCRIPT_DIR / "cell_vs_delay.png"

CELL_LABELS = ["a (Å)", "b (Å)", "c (Å)", "α (°)", "β (°)", "γ (°)"]
MERGE_TOL = 5  # merge delays within this many units of each other


def estimate_total_frames(core) -> int:
    """Estimate total frames from max event index per unique file.

    Streams use --no-non-hits-in-stream, so only hits are stored.
    Events are //0..//N per HDF5 file, so total = sum(max_event+1) per file.
    """
    filenames = core.metadata["image"]["filename"]
    events = core.metadata["image"]["event"]
    max_event = {}
    for fn, ev in zip(filenames, events):
        idx = int(ev.strip("/"))
        if fn not in max_event or idx > max_event[fn]:
            max_event[fn] = idx
    return sum(v + 1 for v in max_event.values())


def extract_delay(path: Path) -> float:
    m = re.search(r"delay_([-\d.]+)\.stream$", path.name)
    if m is None:
        raise ValueError(f"Cannot parse delay from {path.name}")
    return float(m.group(1))


def parse_and_cache(streams):
    """Parse all streams, caching per-stream statistics to JSON."""
    # Load existing cache
    cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        print(f"Loaded cache with {len(cache)} entries")

    raw_delays = []
    raw_data = []
    dirty = False

    for i, p in enumerate(streams):
        delay = extract_delay(p)
        key = p.name

        if key in cache:
            d = cache[key]
            raw_delays.append(delay)
            raw_data.append({
                "cell_med": torch.tensor(d["cell_med"]),
                "cell_q25": torch.tensor(d["cell_q25"]),
                "cell_q75": torch.tensor(d["cell_q75"]),
                "res_med": d["res_med"],
                "res_q25": d["res_q25"],
                "res_q75": d["res_q75"],
                "n_total": d["n_total"],
                "n_hits": d["n_hits"],
                "n_indexed": d["n_indexed"],
            })
            print(f"  [{i+1}/{len(streams)}] delay={delay:>8.0f}  (cached)")
            continue

        core = parse_stream(p, n_workers=16)
        n_crystals = core.n_crystals
        n_hits = core.n_images
        n_indexed = core.image_id.unique().numel() if n_crystals > 0 else 0
        n_total = estimate_total_frames(core)

        cell = core.cell
        cell_med = cell.median(dim=0).values
        cell_q25 = cell.quantile(0.25, dim=0)
        cell_q75 = cell.quantile(0.75, dim=0)

        res = core.resolution
        res_med = res.median().item()
        res_q25 = res.quantile(0.25).item()
        res_q75 = res.quantile(0.75).item()

        raw_delays.append(delay)
        raw_data.append({
            "cell_med": cell_med,
            "cell_q25": cell_q25,
            "cell_q75": cell_q75,
            "res_med": res_med,
            "res_q25": res_q25,
            "res_q75": res_q75,
            "n_total": n_total,
            "n_hits": n_hits,
            "n_indexed": n_indexed,
        })

        # Add to cache
        cache[key] = {
            "cell_med": cell_med.tolist(),
            "cell_q25": cell_q25.tolist(),
            "cell_q75": cell_q75.tolist(),
            "res_med": res_med,
            "res_q25": res_q25,
            "res_q75": res_q75,
            "n_total": n_total,
            "n_hits": n_hits,
            "n_indexed": n_indexed,
        }
        dirty = True

        print(f"  [{i+1}/{len(streams)}] delay={delay:>8.0f}  "
              f"frames={n_total}  hits={n_hits}  indexed={n_indexed}  "
              f"hit={100*n_hits/n_total:.1f}%  idx={100*n_indexed/n_total:.1f}%")

    if dirty:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=1)
        print(f"Cache saved → {CACHE_FILE}")

    return raw_delays, raw_data


def merge_nearby_delays(raw_delays, raw_data):
    """Group entries whose delays differ by <= MERGE_TOL."""
    merged_delays = []
    merged_data = []
    i = 0
    while i < len(raw_delays):
        group_delays = [raw_delays[i]]
        group_data = [raw_data[i]]
        j = i + 1
        while j < len(raw_delays) and raw_delays[j] - raw_delays[i] <= MERGE_TOL:
            group_delays.append(raw_delays[j])
            group_data.append(raw_data[j])
            j += 1
        merged_delays.append(sum(group_delays) / len(group_delays))
        n = len(group_data)
        if n == 1:
            merged_data.append(group_data[0])
        else:
            # Weighted average of pre-computed statistics by n_hits
            weights = torch.tensor([d["n_hits"] for d in group_data], dtype=torch.float32)
            w = weights / weights.sum()
            merged_data.append({
                "cell_med": sum(d["cell_med"] * wi for d, wi in zip(group_data, w)),
                "cell_q25": sum(d["cell_q25"] * wi for d, wi in zip(group_data, w)),
                "cell_q75": sum(d["cell_q75"] * wi for d, wi in zip(group_data, w)),
                "res_med": sum(d["res_med"] * wi for d, wi in zip(group_data, w)).item(),
                "res_q25": sum(d["res_q25"] * wi for d, wi in zip(group_data, w)).item(),
                "res_q75": sum(d["res_q75"] * wi for d, wi in zip(group_data, w)).item(),
                "n_total": sum(d["n_total"] for d in group_data),
                "n_hits": sum(d["n_hits"] for d in group_data),
                "n_indexed": sum(d["n_indexed"] for d in group_data),
            })
            print(f"  Merged delays {group_delays} → {merged_delays[-1]:.0f}")
        i = j
    return merged_delays, merged_data


def main():
    streams = sorted(STREAM_DIR.glob("delay_*.stream"), key=extract_delay)
    print(f"Found {len(streams)} stream files")

    raw_delays, raw_data = parse_and_cache(streams)

    delays, data = merge_nearby_delays(raw_delays, raw_data)
    print(f"After merging: {len(delays)} delay points (from {len(raw_delays)} streams)")

    # Collect statistics
    cell_med, cell_q25, cell_q75 = [], [], []
    hit_rate, indexing_rate = [], []
    res_med, res_q25, res_q75 = [], [], []

    for d in data:
        cell_med.append(d["cell_med"])
        cell_q25.append(d["cell_q25"])
        cell_q75.append(d["cell_q75"])

        n_total = d["n_total"]
        hit_rate.append(100 * d["n_hits"] / n_total if n_total > 0 else 0)
        indexing_rate.append(100 * d["n_indexed"] / n_total if n_total > 0 else 0)

        res_med.append(d["res_med"])
        res_q25.append(d["res_q25"])
        res_q75.append(d["res_q75"])

    import numpy as np

    delay_vals = delays  # python list of floats
    cell_med = torch.stack(cell_med)
    cell_q25 = torch.stack(cell_q25)
    cell_q75 = torch.stack(cell_q75)

    # Plot by dataset index (evenly spaced) with delay labels on x-axis
    x = np.arange(len(delay_vals))
    # Pick ~15 tick positions spread across the range
    n_ticks = min(15, len(x))
    tick_idx = np.linspace(0, len(x) - 1, n_ticks, dtype=int)
    tick_labels = [f"{delay_vals[i]:.0f}" for i in tick_idx]

    def setup_xaxis(ax):
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)

    # --- Figure: 3 rows × 3 cols ---
    # Row 1-2: cell params (2×3)
    # Row 3: resolution (left), hit/indexing rate (right, wide)
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    # Cell parameters (first 6 panels)
    for j in range(6):
        ax = fig.add_subplot(gs[j // 3, j % 3])
        ax.fill_between(x, cell_q25[:, j], cell_q75[:, j], alpha=0.25)
        ax.plot(x, cell_med[:, j], lw=1.2)
        ax.set_ylabel(CELL_LABELS[j])
        ax.ticklabel_format(axis="y", useOffset=False)
        setup_xaxis(ax)

    # Resolution
    ax = fig.add_subplot(gs[2, 0])
    ax.fill_between(x, res_q25, res_q75, alpha=0.25, color="C1")
    ax.plot(x, res_med, lw=1.2, color="C1")
    ax.set_ylabel("Resolution (Å)")
    ax.set_xlabel("Delay")
    setup_xaxis(ax)

    # Hit rate & indexing rate (span 2 cols)
    ax = fig.add_subplot(gs[2, 1:])
    ax.plot(x, hit_rate, lw=1.5, color="C2", label="Hit rate")
    ax.plot(x, indexing_rate, lw=1.5, color="C3", label="Indexing rate")
    ax.set_ylabel("Rate (%)")
    ax.set_xlabel("Delay")
    ax.set_ylim(10, 100)
    ax.set_yscale("log")

    mean_hit = np.mean(hit_rate)
    mean_idx = np.mean(indexing_rate)
    ax.axhline(mean_hit, ls="--", lw=1, color="C2", alpha=0.6)
    ax.axhline(mean_idx, ls="--", lw=1, color="C3", alpha=0.6)
    ax.text(x[-1], mean_hit, f" {mean_hit:.1f}%", va="top", fontsize=8, color="C2")
    ax.text(x[-1], mean_idx, f" {mean_idx:.1f}%", va="top", fontsize=8, color="C3")

    ax.legend(fontsize=9)
    setup_xaxis(ax)

    fig.suptitle("Stream metadata vs. delay", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Saved → {OUT_PNG}")


if __name__ == "__main__":
    main()
