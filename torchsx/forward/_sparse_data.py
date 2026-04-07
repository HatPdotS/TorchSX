"""Load sparse compressed files and provide per-frame lookups.

Sparse compressed data format (HDF5 / safetensors):
- profiles: (n_frames, n_bins) radial background profiles
- signal_idx: (n_total,) int32 flat good-pixel index
- signal_frame: (n_total,) int32 frame index
- signal_val: (n_total,) int32 photon count
- bin_edges: (n_bins+1,) radial bin edges
- Metadata: z_threshold, n_good_pixels, detector_shape
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


@dataclass
class SparseFrameStore:
    """Loaded sparse data for one file."""

    profiles: Tensor        # (n_frames, n_bins) — radial bg profiles
    signal_idx: Tensor      # (n_total,) int32 — flat good-pixel index
    signal_frame: Tensor    # (n_total,) int32 — frame index
    signal_val: Tensor      # (n_total,) int32 — photon count
    bin_edges: Tensor       # (n_bins+1,)
    z_threshold: float
    n_good_pixels: int
    detector_shape: tuple[int, int]

    @property
    def n_frames(self) -> int:
        return self.profiles.shape[0]

    @property
    def n_bins(self) -> int:
        return self.profiles.shape[1]

    def to(self, device: torch.device) -> SparseFrameStore:
        """Move all tensors to device."""
        return SparseFrameStore(
            profiles=self.profiles.to(device),
            signal_idx=self.signal_idx.to(device),
            signal_frame=self.signal_frame.to(device),
            signal_val=self.signal_val.to(device),
            bin_edges=self.bin_edges.to(device),
            z_threshold=self.z_threshold,
            n_good_pixels=self.n_good_pixels,
            detector_shape=self.detector_shape,
        )


def load_sparse(path: Path) -> SparseFrameStore:
    """Load sparse compressed data from safetensors + JSON sidecar.

    Parameters
    ----------
    path : Path to .safetensors file. JSON sidecar at path.with_suffix('.json').

    Returns
    -------
    SparseFrameStore with all tensors on CPU.
    """
    import json
    from safetensors.torch import load_file

    path = Path(path)
    tensors = load_file(str(path))

    # Load metadata from JSON sidecar
    json_path = path.with_suffix(".json")
    with open(json_path) as f:
        meta = json.load(f)

    return SparseFrameStore(
        profiles=tensors["profiles"],
        signal_idx=tensors["signal_idx"],
        signal_frame=tensors["signal_frame"],
        signal_val=tensors["signal_val"],
        bin_edges=tensors["bin_edges"],
        z_threshold=float(meta["z_threshold"]),
        n_good_pixels=int(meta["n_good_pixels"]),
        detector_shape=tuple(meta["detector_shape"]),
    )


class PixelIndexMapper:
    """Map (ss, fs) detector coords <-> flat good-pixel index.

    Precomputes a dense lookup table: good_to_flat[ss, fs] -> flat index
    (-1 for bad pixels). This enables O(1) per-pixel queries.

    Parameters
    ----------
    good_mask_or_lookup : (total_ss, total_fs) bool mask, or pre-built
        int64 lookup table (e.g. already on GPU).
    device : target device for the lookup table. Ignored when passing
        a pre-built lookup (device inferred from the tensor).
    """

    def __init__(
        self,
        good_mask_or_lookup: Tensor,
        device: torch.device | None = None,
    ):
        if good_mask_or_lookup.dtype == torch.bool:
            # Build flat index from mask
            good_mask = good_mask_or_lookup
            flat = good_mask.long()
            flat_idx = torch.cumsum(flat.reshape(-1), dim=0).reshape(good_mask.shape) - 1
            flat_idx[~good_mask] = -1
            target_device = device or good_mask.device
            self._lookup = flat_idx.to(target_device)
            self.n_good = good_mask.sum().item()
        else:
            # Pre-built lookup table (already on desired device)
            self._lookup = good_mask_or_lookup
            self.n_good = (good_mask_or_lookup >= 0).sum().item()

        self._shape = self._lookup.shape

        # Reverse lookup: flat good-pixel index → (ss, fs)
        # torch.where on CPU (one-time cost), then move to device
        mask_cpu = (self._lookup >= 0).cpu()
        good_ss, good_fs = torch.where(mask_cpu)
        self._good_ss = good_ss.to(torch.int32).to(self._lookup.device)
        self._good_fs = good_fs.to(torch.int32).to(self._lookup.device)

    def flat_to_ss_fs(self, flat_idx: Tensor) -> tuple[Tensor, Tensor]:
        """Convert flat good-pixel indices to (ss, fs) detector coordinates.

        Parameters
        ----------
        flat_idx : (N,) int — flat good-pixel index

        Returns
        -------
        ss : (N,) int32 — slow-scan coordinates
        fs : (N,) int32 — fast-scan coordinates
        """
        idx = flat_idx.long()
        return self._good_ss[idx], self._good_fs[idx]

    def grid_to_flat(
        self,
        ss_grid: Tensor,
        fs_grid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Map (B, ph, pw) pixel grids to flat good-pixel indices.

        Parameters
        ----------
        ss_grid : (B, ph, pw) float — slow-scan pixel coordinates
        fs_grid : (B, ph, pw) float — fast-scan pixel coordinates

        Returns
        -------
        flat_idx : (B, ph, pw) int64 — flat good-pixel index (0 if bad)
        on_good : (B, ph, pw) bool — True if pixel is good
        """
        ss_int = ss_grid.long()
        fs_int = fs_grid.long()

        total_ss, total_fs = self._shape

        # Bounds check
        on_det = (
            (ss_int >= 0) & (ss_int < total_ss)
            & (fs_int >= 0) & (fs_int < total_fs)
        )

        # Clamp for safe indexing (out-of-bounds → will be masked)
        ss_safe = ss_int.clamp(0, total_ss - 1)
        fs_safe = fs_int.clamp(0, total_fs - 1)

        flat_idx = self._lookup[ss_safe, fs_safe]
        on_good = on_det & (flat_idx >= 0)
        flat_idx = flat_idx.where(on_good, torch.zeros_like(flat_idx))

        return flat_idx, on_good
