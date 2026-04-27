"""Pixel-to-reflection matching for sparse pixel-level refinement.

Matches sparse observed pixels to nearby predicted reflections using
distance thresholding. Produces PixelReflectionPairs with CSR indexing
for efficient GPU scatter_add operations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class PixelReflectionPairs:
    """Matched (reflection, pixel) pairs for sparse forward evaluation.

    All arrays are flat (no grid structure). CSR indexing via pair_ptr
    enables scatter_add aggregation on GPU.
    """

    # ── Per-reflection (N_refl) ──────────────────────────
    crystal_id: Tensor    # (N_refl,) int32
    hkl_id: Tensor        # (N_refl,) int64
    hkl: Tensor           # (N_refl, 3) int16
    panel_id: Tensor      # (N_refl,) int16 — panel of predicted center
    ss_pred: Tensor       # (N_refl,) float — predicted center position
    fs_pred: Tensor       # (N_refl,) float

    # ── Per-pixel (N_pixels) ─────────────────────────────
    pixel_ss: Tensor      # (N_pixels,) float — detector coordinates
    pixel_fs: Tensor      # (N_pixels,) float
    pixel_panel: Tensor   # (N_pixels,) int16
    pixel_counts: Tensor  # (N_pixels,) int32 — observed photon counts
    pixel_bg: Tensor      # (N_pixels,) float — background from radial profile

    # ── Per-pair (N_pairs) ───────────────────────────────
    refl_idx: Tensor      # (N_pairs,) int64 — index into reflection arrays
    pixel_idx: Tensor     # (N_pairs,) int64 — index into pixel arrays

    # ── CSR indexing ─────────────────────────────────────
    # Reflection i owns pairs pair_ptr[i]:pair_ptr[i+1]
    pair_ptr: Tensor      # (N_refl + 1,) int64

    # ── Precomputed geometry (set by forward model) ──────
    pixel_s_hat: Tensor | None = None      # (N_pixels, 3)
    pixel_two_theta: Tensor | None = None  # (N_pixels,)
    pixel_pol: Tensor | None = None        # (N_pixels,)
    pixel_solid_angle: Tensor | None = None  # (N_pixels,)

    @property
    def n_refl(self) -> int:
        return self.crystal_id.shape[0]

    @property
    def n_pixels(self) -> int:
        return self.pixel_ss.shape[0]

    @property
    def n_pairs(self) -> int:
        return self.refl_idx.shape[0]

    def to(self, device: torch.device) -> PixelReflectionPairs:
        """Move all tensors to device."""
        fields = {}
        for name in self.__dataclass_fields__:
            val = getattr(self, name)
            if isinstance(val, Tensor):
                fields[name] = val.to(device)
            else:
                fields[name] = val
        return PixelReflectionPairs(**fields)


@dataclass
class SparseForwardResult:
    """Result of forward_pairs() evaluation."""

    n_signal_pairs: Tensor   # (N_pairs,) — predicted signal per pair
    n_pred_pixels: Tensor    # (N_pixels,) — total predicted rate per pixel
    F_sq: Tensor             # (N_refl,) — structure factor squared
    scale: Tensor            # (N_refl,) — crystal scale × B-factor


def match_pixels_to_reflections(
    refl_ss: Tensor,
    refl_fs: Tensor,
    refl_frame: Tensor,
    pixel_ss: Tensor,
    pixel_fs: Tensor,
    pixel_frame: Tensor,
    pixel_counts: Tensor,
    pixel_bg: Tensor,
    search_radius: float = 15.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Match sparse pixels to nearby reflections within each frame.

    Matches purely by Euclidean distance in (ss, fs) detector space —
    no panel constraint, since reflections near panel boundaries can
    have signal on adjacent panels.

    Parameters
    ----------
    refl_ss, refl_fs : (N_refl,) float — predicted reflection centers
    refl_frame : (N_refl,) int — frame index per reflection
    pixel_ss, pixel_fs : (N_pixels,) float — sparse pixel positions
    pixel_frame : (N_pixels,) int — frame index per pixel
    pixel_counts : (N_pixels,) int32 — observed counts
    pixel_bg : (N_pixels,) float — background
    search_radius : float — max distance in pixels

    Returns
    -------
    refl_idx : (N_pairs,) int64 — index into reflection arrays
    pixel_idx : (N_pairs,) int64 — index into pixel arrays
    pair_ptr : (N_refl + 1,) int64 — CSR pointers
    """
    device = refl_ss.device
    N_refl = refl_ss.shape[0]
    N_pixels = pixel_ss.shape[0]

    if N_refl == 0 or N_pixels == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.zeros(N_refl + 1, dtype=torch.long, device=device),
        )

    # Sort reflections and pixels by frame for grouped processing
    refl_order = refl_frame.argsort()
    pixel_order = pixel_frame.argsort()

    refl_frame_sorted = refl_frame[refl_order]
    pixel_frame_sorted = pixel_frame[pixel_order]

    # Find unique frames and their boundaries
    refl_unique, refl_counts = refl_frame_sorted.unique_consecutive(return_counts=True)
    pixel_unique, pixel_counts_per_frame = pixel_frame_sorted.unique_consecutive(return_counts=True)

    # Build frame offset tables
    refl_ends = refl_counts.cumsum(0)
    refl_starts = refl_ends - refl_counts
    pixel_ends = pixel_counts_per_frame.cumsum(0)
    pixel_starts = pixel_ends - pixel_counts_per_frame

    # Build pixel frame lookup (frame_id → index in pixel_unique)
    pixel_frame_map = {}
    for i in range(pixel_unique.shape[0]):
        pixel_frame_map[pixel_unique[i].item()] = i

    # Process each frame: find pairs within search_radius
    all_refl_idx = []
    all_pixel_idx = []

    # Precompute sorted positions for distance computation
    refl_ss_sorted = refl_ss[refl_order]
    refl_fs_sorted = refl_fs[refl_order]
    pixel_ss_sorted = pixel_ss[pixel_order]
    pixel_fs_sorted = pixel_fs[pixel_order]

    for fi in range(refl_unique.shape[0]):
        frame_id = refl_unique[fi].item()
        rs, re = refl_starts[fi].item(), refl_ends[fi].item()

        if frame_id not in pixel_frame_map:
            continue
        pi = pixel_frame_map[frame_id]
        ps, pe = pixel_starts[pi].item(), pixel_ends[pi].item()

        # Compute pairwise distances — no panel constraint
        r_ss = refl_ss_sorted[rs:re]  # (n_r,)
        r_fs = refl_fs_sorted[rs:re]
        p_ss = pixel_ss_sorted[ps:pe]  # (n_p,)
        p_fs = pixel_fs_sorted[ps:pe]

        # Pairwise: (n_r, n_p)
        dss = r_ss[:, None] - p_ss[None, :]
        dfs = r_fs[:, None] - p_fs[None, :]
        dist_sq = dss ** 2 + dfs ** 2

        valid = dist_sq <= search_radius ** 2

        # Collect pairs
        r_local, p_local = torch.where(valid)
        if r_local.numel() > 0:
            # Map back to original (unsorted) indices
            all_refl_idx.append(refl_order[rs + r_local])
            all_pixel_idx.append(pixel_order[ps + p_local])

    if all_refl_idx:
        refl_idx_cat = torch.cat(all_refl_idx)
        pixel_idx_cat = torch.cat(all_pixel_idx)
    else:
        refl_idx_cat = torch.empty(0, dtype=torch.long, device=device)
        pixel_idx_cat = torch.empty(0, dtype=torch.long, device=device)

    # Build CSR pointers: sort pairs by refl_idx, then cumsum
    if refl_idx_cat.numel() > 0:
        sort_order = refl_idx_cat.argsort()
        refl_idx_cat = refl_idx_cat[sort_order]
        pixel_idx_cat = pixel_idx_cat[sort_order]

        # Count pairs per reflection
        counts = torch.zeros(N_refl, dtype=torch.long, device=device)
        counts.scatter_add_(0, refl_idx_cat, torch.ones_like(refl_idx_cat))
        pair_ptr = torch.zeros(N_refl + 1, dtype=torch.long, device=device)
        pair_ptr[1:] = counts.cumsum(0)
    else:
        pair_ptr = torch.zeros(N_refl + 1, dtype=torch.long, device=device)

    return refl_idx_cat, pixel_idx_cat, pair_ptr
