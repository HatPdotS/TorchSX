#!/usr/bin/env python
"""Visualize time-delay scan points for SFX experiment."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
from pathlib import Path
import re

# --- extract delays from filenames ---
stream_dir = Path(__file__).parent / "streams"
delays = sorted(
    int(m.group(1))
    for f in stream_dir.glob("delay_*.stream")
    if (m := re.match(r"delay_(-?\d+)\.stream", f.name))
)
delays = np.array(delays, dtype=float)

# --- detect linear / log transition ---
diffs = np.diff(delays)
median_early = np.median(diffs[:20])
transition_idx = np.argmax(diffs > 1.3 * median_early)
t_transition = delays[transition_idx]

# --- colors ---
c_neg = "#3B7DD8"    # blue   – negative / pre-T0
c_lin = "#2CA02C"    # green  – linear
c_log = "#D94F04"    # red    – logarithmic
c_bg  = "#FAFAFA"

regions = np.where(delays < 0, "neg",
          np.where(delays <= t_transition, "lin", "log"))
colors = np.array([{"neg": c_neg, "lin": c_lin, "log": c_log}[r] for r in regions])

# ── Figure ──
fig = plt.figure(figsize=(13, 7), facecolor="white")

# Layout: top-left = step size, top-right = zoom inset, bottom = timeline
ax_step = fig.add_axes([0.08, 0.60, 0.55, 0.30])   # step size plot
ax_zoom = fig.add_axes([0.72, 0.60, 0.25, 0.30])   # zoom of early region
ax_main = fig.add_axes([0.08, 0.07, 0.89, 0.40])   # main timeline

fig.text(0.5, 0.96, "Time-delay scan coverage", ha="center", fontsize=15,
         fontweight="bold", va="top")

# ═══ Top-left: step spacing ═══
midpoints = 0.5 * (delays[:-1] + delays[1:])
ax_step.semilogy(midpoints / 1e3, diffs, "-", color="0.6", lw=0.7, zorder=1)
ax_step.scatter(midpoints / 1e3, diffs, c=colors[:-1], s=14, zorder=2,
                edgecolors="none")
ax_step.set_ylabel("Step size (fs)", fontsize=11)
ax_step.set_xlabel("Delay (ps)", fontsize=10)
ax_step.set_title("Step spacing", fontsize=11, fontweight="semibold", pad=6)
ax_step.axvline(t_transition / 1e3, ls="--", color="0.55", lw=0.8, alpha=0.6)
ax_step.text(t_transition / 1e3 + 1.0, 30,
             f"transition\n≈ {t_transition:.0f} fs", fontsize=8.5,
             color="0.4", va="top")
for sp in ["top", "right"]:
    ax_step.spines[sp].set_visible(False)

# ═══ Top-right: zoomed early region ═══
zoom_max = t_transition + 200  # fs
mask_zoom = delays <= zoom_max
d_zoom = delays[mask_zoom]
c_zoom = colors[mask_zoom]

for i, (d, c) in enumerate(zip(d_zoom, c_zoom)):
    ax_zoom.plot([d, d], [0, 0.6], lw=1.2, color=c, alpha=0.6)
ax_zoom.scatter(d_zoom, np.full_like(d_zoom, 0.6), c=c_zoom, s=22,
                zorder=3, edgecolors="white", linewidths=0.4)
ax_zoom.axhline(0, color="0.3", lw=1)
ax_zoom.axvline(0, ls=":", color="0.5", lw=0.9)
ax_zoom.text(0, -0.15, "T₀", ha="center", fontsize=10, fontweight="bold",
             color="0.3")
ax_zoom.set_xlabel("Delay (fs)", fontsize=10)
ax_zoom.set_title("Early region (zoom)", fontsize=11, fontweight="semibold",
                   pad=6)
ax_zoom.set_xlim(d_zoom[0] - 40, zoom_max + 40)
ax_zoom.set_ylim(-0.25, 1.0)
ax_zoom.set_yticks([])
for sp in ["left", "top", "right"]:
    ax_zoom.spines[sp].set_visible(False)

# Region labels in zoom panel
n_neg = (delays < 0).sum()
n_lin = ((delays >= 0) & (delays <= t_transition)).sum()
ax_zoom.text(delays[delays < 0].mean(), 0.85,
             f"pre-T₀\n{n_neg} pts", ha="center", fontsize=8.5,
             color=c_neg, fontweight="semibold")
ax_zoom.text((0 + t_transition) / 2, 0.85,
             f"linear\n{n_lin} pts  Δt ≈ {median_early:.0f} fs",
             ha="center", fontsize=8.5, color=c_lin, fontweight="semibold")

# ═══ Bottom: full timeline ═══
n_log = (delays > t_transition).sum()

for i, (d, c) in enumerate(zip(delays, colors)):
    ax_main.plot([d / 1e3, d / 1e3], [0, 0.45], lw=0.6, color=c, alpha=0.5)
ax_main.scatter(delays / 1e3, np.full_like(delays, 0.45), c=colors, s=16,
                zorder=3, edgecolors="white", linewidths=0.3)
ax_main.axhline(0, color="0.3", lw=1.2)
ax_main.axvline(0, ls=":", color="0.5", lw=0.9)
ax_main.text(0, -0.15, "T₀", ha="center", fontsize=11, fontweight="bold",
             color="0.3")

# Region shading
ax_main.axvspan(delays[0] / 1e3, 0, alpha=0.06, color=c_neg, zorder=0)
ax_main.axvspan(0, t_transition / 1e3, alpha=0.06, color=c_lin, zorder=0)
ax_main.axvspan(t_transition / 1e3, delays[-1] / 1e3, alpha=0.04, color=c_log, zorder=0)

# Labels on main timeline — placed higher to avoid crowding
ax_main.text((t_transition / 1e3 + delays[-1] / 1e3) / 2, 0.72,
             f"logarithmic\n{n_log} pts", ha="center", fontsize=9.5,
             color=c_log, fontweight="semibold")

# Formatting
ax_main.set_xlabel("Delay (ps)", fontsize=12)
ax_main.set_xlim(delays[0] / 1e3 - 0.5, delays[-1] / 1e3 + 1.5)
ax_main.set_ylim(-0.25, 0.95)
ax_main.set_yticks([])
for sp in ["left", "top", "right"]:
    ax_main.spines[sp].set_visible(False)

# Stats box
ax_main.text(0.99, 0.97,
             f"{len(delays)} time points  ·  "
             f"{delays[0]/1e3:.2f} to {delays[-1]/1e3:.1f} ps",
             transform=ax_main.transAxes, ha="right", va="top",
             fontsize=10, color="0.35",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8",
                        alpha=0.85))

# Legend
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=c_neg,
           markersize=7, label="Pre-T₀ (negative delays)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=c_lin,
           markersize=7, label=f"Linear (Δt ≈ {median_early:.0f} fs)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=c_log,
           markersize=7, label="Logarithmic"),
]
ax_main.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.9,
               edgecolor="0.8")

plt.savefig(stream_dir.parent / "time_delay_scan.png", dpi=200,
            bbox_inches="tight", facecolor="white")
plt.savefig(stream_dir.parent / "time_delay_scan.pdf",
            bbox_inches="tight", facecolor="white")
print(f"Saved to {stream_dir.parent / 'time_delay_scan.png'}")
print(f"  {len(delays)} delays: {delays[0]:.0f} to {delays[-1]:.0f} fs")
print(f"  {n_neg} pre-T0 + {n_lin} linear (Δ≈{median_early:.0f} fs) + {n_log} logarithmic")
