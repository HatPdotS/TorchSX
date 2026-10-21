"""Resolution-binned statistics: completeness, multiplicity, split-half metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from torchsx._core import _SFXStreamCore
from torchsx.merge_stats import r_split as _r_split


def completeness(core: _SFXStreamCore, n_bins: int = 20) -> torch.Tensor:
    """Resolution-binned completeness (fraction of unique HKLs observed)."""
    from torchref.base.reciprocal.hkl import get_d_spacing, generate_possible_hkl
    from torchref.base.alignment.normalization import (
        compute_radial_shells,
        assign_to_shells,
    )

    median_cell = core.cell.median(dim=0).values
    d_min_val = core.resolution.min().item()
    d_max_val = core.resolution.max().item()

    possible_hkl = generate_possible_hkl(median_cell, d_min_val)
    d_possible = get_d_spacing(possible_hkl.float(), median_cell.unsqueeze(0))

    shell_edges, _ = compute_radial_shells(
        d_min_val, d_max_val, n_bins, device=core.hkl.device
    )

    obs_hkl_unique = core.hkl.unique(dim=0)
    obs_cell = median_cell.unsqueeze(0).expand(obs_hkl_unique.shape[0], -1)
    d_obs = get_d_spacing(obs_hkl_unique.float(), obs_cell)

    s_possible = 1.0 / d_possible
    s_obs = 1.0 / d_obs

    bins_possible = assign_to_shells(s_possible, shell_edges)
    bins_obs = assign_to_shells(s_obs, shell_edges)

    comp = torch.zeros(n_bins)
    for i in range(n_bins):
        n_poss = (bins_possible == i).sum().item()
        n_obs = (bins_obs == i).sum().item()
        comp[i] = n_obs / max(n_poss, 1)

    return comp


def multiplicity(core: _SFXStreamCore, n_bins: int = 20) -> torch.Tensor:
    """Resolution-binned mean multiplicity."""
    from torchref.base.reciprocal.hkl import get_d_spacing
    from torchref.base.alignment.normalization import (
        compute_radial_shells,
        assign_to_shells,
    )

    median_cell = core.cell.median(dim=0).values
    d_min_val = core.resolution.min().item()
    d_max_val = core.resolution.max().item()

    obs_cell = core.cell[core.crystal_id.long()]
    d_obs = get_d_spacing(core.hkl.float(), obs_cell)

    shell_edges, _ = compute_radial_shells(
        d_min_val, d_max_val, n_bins, device=core.hkl.device
    )
    s_obs = 1.0 / d_obs
    bins = assign_to_shells(s_obs, shell_edges)

    mult = torch.zeros(n_bins)
    for i in range(n_bins):
        mask = bins == i
        if mask.sum() == 0:
            continue
        n_total = mask.sum().item()
        n_unique = core.hkl[mask].unique(dim=0).shape[0]
        mult[i] = n_total / max(n_unique, 1)

    return mult


# ── Split-half comparison ──────────────────────────────────────────────


@dataclass
class MatchedHalves:
    """Paired reflections present in both halves."""

    F1: torch.Tensor
    F2: torch.Tensor
    d_spacing: torch.Tensor
    hkl: torch.Tensor


def match_halves(
    hkl1: torch.Tensor,
    F1: torch.Tensor,
    hkl2: torch.Tensor,
    F2: torch.Tensor,
    cell: torch.Tensor,
    d_min: float = 0.0,
    d_max: float = float("inf"),
) -> MatchedHalves:
    """Match two merged HKL sets and compute d-spacing."""
    from torchref.base.reciprocal.hkl import get_d_spacing

    hkl2_long = hkl2.long()
    lookup2: dict[tuple[int, int, int], int] = {}
    for i in range(hkl2_long.shape[0]):
        key = (hkl2_long[i, 0].item(), hkl2_long[i, 1].item(), hkl2_long[i, 2].item())
        lookup2[key] = i

    idx1_list: list[int] = []
    idx2_list: list[int] = []
    hkl1_long = hkl1.long()
    for i in range(hkl1_long.shape[0]):
        key = (hkl1_long[i, 0].item(), hkl1_long[i, 1].item(), hkl1_long[i, 2].item())
        j = lookup2.get(key)
        if j is not None:
            idx1_list.append(i)
            idx2_list.append(j)

    if not idx1_list:
        empty = torch.zeros(0)
        return MatchedHalves(
            F1=empty, F2=empty, d_spacing=empty,
            hkl=torch.zeros(0, 3, dtype=torch.int16),
        )

    idx1 = torch.tensor(idx1_list, dtype=torch.long)
    idx2 = torch.tensor(idx2_list, dtype=torch.long)

    matched_hkl = hkl1[idx1]
    matched_F1 = F1[idx1]
    matched_F2 = F2[idx2]
    d = get_d_spacing(matched_hkl.float(), cell)

    if d_min > 0.0 or d_max < float("inf"):
        mask = (d >= d_min) & (d <= d_max)
        matched_hkl = matched_hkl[mask]
        matched_F1 = matched_F1[mask]
        matched_F2 = matched_F2[mask]
        d = d[mask]

    return MatchedHalves(F1=matched_F1, F2=matched_F2, d_spacing=d, hkl=matched_hkl)


def cc_half(matched: MatchedHalves) -> float:
    if matched.F1.numel() < 2:
        return 0.0
    stacked = torch.stack([matched.F1.double(), matched.F2.double()])
    return torch.corrcoef(stacked)[0, 1].item()


def _rankdata(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    order = x.argsort()
    ranks = torch.empty_like(x, dtype=torch.double)
    ranks[order] = torch.arange(1, n + 1, dtype=torch.double)
    sorted_x = x[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        if j > i + 1:
            avg_rank = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman_cc(matched: MatchedHalves) -> float:
    if matched.F1.numel() < 2:
        return 0.0
    r1 = _rankdata(matched.F1)
    r2 = _rankdata(matched.F2)
    stacked = torch.stack([r1, r2])
    return torch.corrcoef(stacked)[0, 1].item()


def cc_star(cc_half_value: float) -> float:
    if cc_half_value <= 0:
        return 0.0
    return math.sqrt(2.0 * cc_half_value / (1.0 + cc_half_value))


def r_split(matched: MatchedHalves) -> float:
    """R_split of the matched halves (:func:`torchsx.merge_stats.r_split`); 0 when empty."""
    if matched.F1.numel() == 0:
        return 0.0
    r = _r_split(matched.F1.double(), matched.F2.double())
    return 0.0 if math.isnan(r) else r


@dataclass
class ShellStats:
    d_min: float
    d_max: float
    cc_half: float
    cc_star: float
    r_split: float
    n_matched: int
    spearman: float = 0.0


def shell_stats(matched: MatchedHalves, n_shells: int = 20) -> list[ShellStats]:
    """Bin matched reflections into equal-count shells sorted by resolution."""
    d = matched.d_spacing
    if d.numel() == 0:
        return []

    order = d.argsort(descending=True)
    bins = torch.zeros(d.numel(), dtype=torch.long)
    chunk_size = d.numel() // n_shells
    remainder = d.numel() % n_shells
    start = 0
    for i in range(n_shells):
        end = start + chunk_size + (1 if i < remainder else 0)
        bins[order[start:end]] = i
        start = end

    shells: list[ShellStats] = []
    for i in range(n_shells):
        mask = bins == i
        n = mask.sum().item()
        if n == 0:
            continue
        d_shell = d[mask]
        sub = MatchedHalves(
            F1=matched.F1[mask],
            F2=matched.F2[mask],
            d_spacing=d_shell,
            hkl=matched.hkl[mask],
        )
        cc = cc_half(sub)
        sp = spearman_cc(sub)
        shells.append(ShellStats(
            d_min=d_shell.min().item(),
            d_max=d_shell.max().item(),
            cc_half=cc,
            cc_star=cc_star(cc),
            r_split=r_split(sub),
            n_matched=n,
            spearman=sp,
        ))
    return shells


def format_shell_table(
    shells: list[ShellStats], overall_cc: float, overall_r: float
) -> str:
    lines = [
        f"{'d_max':>8s} {'d_min':>8s} {'N':>7s} "
        f"{'CC1/2':>8s} {'CC*':>8s} {'R_split':>8s}",
        "-" * 55,
    ]
    for s in shells:
        lines.append(
            f"{s.d_max:8.2f} {s.d_min:8.2f} {s.n_matched:7d} "
            f"{s.cc_half:8.4f} {s.cc_star:8.4f} {s.r_split:8.4f}"
        )
    lines.append("-" * 55)
    lines.append(
        f"{'Overall':>17s} {sum(s.n_matched for s in shells):7d} "
        f"{overall_cc:8.4f} {cc_star(overall_cc):8.4f} {overall_r:8.4f}"
    )
    return "\n".join(lines)


def write_shell_csv(
    shells: list[ShellStats],
    path: str | Path,
    overall_cc: float | None = None,
    overall_r: float | None = None,
) -> None:
    path = Path(path)
    lines = ["d_max,d_min,n_matched,cc_half,cc_star,r_split"]
    for s in shells:
        lines.append(
            f"{s.d_max:.4f},{s.d_min:.4f},{s.n_matched},"
            f"{s.cc_half:.6f},{s.cc_star:.6f},{s.r_split:.6f}"
        )
    if overall_cc is not None and overall_r is not None:
        n_total = sum(s.n_matched for s in shells)
        lines.append(
            f",,{n_total},{overall_cc:.6f},{cc_star(overall_cc):.6f},{overall_r:.6f}"
        )
    path.write_text("\n".join(lines) + "\n")
