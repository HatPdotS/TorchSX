"""Split-half validation plots (matplotlib with Agg backend for SLURM)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from torchsx.merge._stats import MatchedHalves, ShellStats


def plot_cc_vs_resolution(shells: list[ShellStats], path: str | Path) -> None:
    """CC_half and CC* vs resolution shell."""
    if not shells:
        return
    d_mid = [0.5 * (s.d_min + s.d_max) for s in shells]
    inv_d2 = [1.0 / d**2 for d in d_mid]
    cc_h = [s.cc_half for s in shells]
    cc_s = [s.cc_star for s in shells]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(inv_d2, cc_h, "o-", label=r"CC$_{1/2}$")
    ax.plot(inv_d2, cc_s, "s--", label="CC*")
    ax.set_xlabel(r"$1/d^2$ ($\AA^{-2}$)")
    ax.set_ylabel("Correlation coefficient")
    ax.set_ylim(-0.1, 1.05)
    ax.legend()
    ax.set_title(r"CC$_{1/2}$ and CC* vs resolution")

    # Add d-spacing tick labels on top
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_d = [s.d_min for s in shells[::max(1, len(shells) // 6)]]
    tick_inv_d2 = [1.0 / d**2 for d in tick_d]
    ax2.set_xticks(tick_inv_d2)
    ax2.set_xticklabels([f"{d:.1f}" for d in tick_d])
    ax2.set_xlabel(r"d ($\AA$)")

    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def plot_rsplit_vs_resolution(shells: list[ShellStats], path: str | Path) -> None:
    """R_split vs resolution shell."""
    if not shells:
        return
    d_mid = [0.5 * (s.d_min + s.d_max) for s in shells]
    inv_d2 = [1.0 / d**2 for d in d_mid]
    rs = [s.r_split for s in shells]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(inv_d2, rs, "o-", color="tab:red")
    ax.set_xlabel(r"$1/d^2$ ($\AA^{-2}$)")
    ax.set_ylabel(r"$R_\mathrm{split}$")
    ax.set_title(r"$R_\mathrm{split}$ vs resolution")

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_d = [s.d_min for s in shells[::max(1, len(shells) // 6)]]
    tick_inv_d2 = [1.0 / d**2 for d in tick_d]
    ax2.set_xticks(tick_inv_d2)
    ax2.set_xticklabels([f"{d:.1f}" for d in tick_d])
    ax2.set_xlabel(r"d ($\AA$)")

    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def plot_f_scatter(matched: MatchedHalves, path: str | Path) -> None:
    """F1 vs F2 scatter plot with identity line."""
    F1 = matched.F1.cpu().numpy()
    F2 = matched.F2.cpu().numpy()
    if len(F1) == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(F1, F2, s=1, alpha=0.3, rasterized=True)
    lo = min(F1.min(), F2.min())
    hi = max(F1.max(), F2.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="y = x")
    ax.set_xlabel(r"$F_1$")
    ax.set_ylabel(r"$F_2$")
    ax.set_title("Split-half structure factor comparison")
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
