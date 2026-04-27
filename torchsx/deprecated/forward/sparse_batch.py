"""SparseBatch and SparseBatchLoader for refinement against sparse compressed data."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from queue import Queue
from threading import Thread
from typing import Iterator

import torch
from torch import Tensor

from torchsx._core import _SFXStreamCore
from torchsx.deprecated.merge._hkl import HKLIndex
from torchsx.deprecated.forward.batch import _build_pixel_grids
from torchsx.deprecated.forward._censored_poisson import compute_censoring_threshold
from torchsx.deprecated.forward._sparse_data import SparseFrameStore, PixelIndexMapper


@dataclass
class SparseBatch:
    """ROIBatch augmented with sparse pixel annotations.

    Duck-types ROIBatch for the forward model: same grid fields (fs_grid,
    ss_grid, pid_grid), same crystal/hkl fields. The forward model doesn't
    need ``patches`` — it only produces predictions. The loss function handles
    the comparison via the sparse annotation fields.
    """

    # Core fields (shared with ROIBatch)
    crystal_id: Tensor       # (B,) int32
    hkl_id: Tensor           # (B,) int64
    hkl: Tensor              # (B, 3) int16
    panel_id: Tensor         # (B,) int16
    fs_center: Tensor        # (B,) float32
    ss_center: Tensor        # (B,) float32
    obs_idx: Tensor | None   # (B,) int64

    # Pixel grids (for forward model geometry)
    fs_grid: Tensor          # (B, ph, pw) float32
    ss_grid: Tensor          # (B, ph, pw) float32
    pid_grid: Tensor         # (B, ph, pw) long

    # Per-pixel background from radial profile
    background: Tensor       # (B, ph, pw) float32

    # Sparse annotations
    is_sparse: Tensor        # (B, ph, pw) bool
    sparse_counts: Tensor    # (B, ph, pw) int32
    threshold: Tensor        # (B, ph, pw) float32
    valid: Tensor            # (B, ph, pw) bool — on-detector + good pixel

    # Precomputed geometry (set by forward model's _ensure_geometry)
    s_hat_out: Tensor | None = None
    two_theta: Tensor | None = None
    phi: Tensor | None = None
    polarization: Tensor | None = None
    solid_angle: Tensor | None = None

    # Dummy patches for forward_debug compatibility (uses shape for ph, pw)
    @property
    def patches(self) -> Tensor:
        """Dummy patches tensor — shape used by forward_debug for ph, pw."""
        return self.background

    def to(self, device: torch.device) -> SparseBatch:
        """Move all tensors to device."""
        fields = {}
        for name in self.__dataclass_fields__:
            val = getattr(self, name)
            if isinstance(val, Tensor):
                fields[name] = val.to(device)
            else:
                fields[name] = val
        return SparseBatch(**fields)


def build_frame_to_obs(
    core: _SFXStreamCore,
    filenames: list[str],
    events: list[str],
    h5_offsets: dict[str, tuple[int, int]],
) -> dict[int, Tensor]:
    """Vectorized construction of global_frame_idx → observation indices.

    Parameters
    ----------
    core : stream core with crystal_id, image_id
    filenames : per-image filenames (from metadata)
    events : per-image event strings (from metadata)
    h5_offsets : h5_path → (global_frame_offset, n_frames)

    Returns
    -------
    dict mapping global frame index → tensor of observation indices
    """
    from torchsx.io._excise import _parse_event_id

    # Step 1: build per-image global frame index (vectorized over images)
    n_images = len(filenames)
    image_global_frame = torch.full((n_images,), -1, dtype=torch.long)

    for img_id in range(n_images):
        fn = filenames[img_id]
        if fn not in h5_offsets:
            continue
        offset, n_frames = h5_offsets[fn]
        frame_idx = _parse_event_id(events[img_id])
        if frame_idx < n_frames:
            image_global_frame[img_id] = offset + frame_idx

    # Step 2: map observations → global frame via crystal_id → image_id
    obs_image_id = core.image_id[core.crystal_id.long()]  # (n_obs,)
    obs_global_frame = image_global_frame[obs_image_id]     # (n_obs,)

    # Step 3: group by frame (vectorized sort + split)
    valid_mask = obs_global_frame >= 0
    valid_obs = valid_mask.nonzero(as_tuple=True)[0]
    valid_frames = obs_global_frame[valid_obs]

    sort_order = valid_frames.argsort()
    sorted_obs = valid_obs[sort_order]
    sorted_frames = valid_frames[sort_order]

    unique_frames, counts = sorted_frames.unique_consecutive(return_counts=True)
    splits = counts.tolist()
    obs_groups = sorted_obs.split(splits)

    return {
        fi.item(): obs_t
        for fi, obs_t in zip(unique_frames, obs_groups)
    }


class SparseBatchLoader:
    """Batch iterator for sparse compressed data, grouped by frames.

    All sparse lookups are fully vectorised — no per-frame Python loops.
    Signal membership is resolved via a single searchsorted on composite
    keys ``(frame_id << 32) | pixel_idx``.

    Parameters
    ----------
    sparse_store : SparseFrameStore
        Loaded sparse compressed data.
    frame_to_obs : dict[int, Tensor]
        Map from frame index → observation indices in core.
    core : _SFXStreamCore
        Stream data (observations, crystals).
    hkl_index : HKLIndex
        HKL index mapping.
    pixel_mapper : PixelIndexMapper
        Maps (ss, fs) → flat good-pixel index.
    bin_assign : Tensor
        (n_good_pixels,) int64 — radial bin index for each good pixel.
    patch_size : int
        ROI patch size (must be odd).
    frames_per_batch : int
        Number of frames per batch.
    device : torch.device
        Target device.
    z_threshold : float
        Anscombe z-score threshold for censoring.
    prefetch : int
        Number of batches to prefetch.
    """

    def __init__(
        self,
        sparse_store: SparseFrameStore,
        frame_to_obs: dict[int, Tensor],
        core: _SFXStreamCore,
        hkl_index: HKLIndex,
        pixel_mapper: PixelIndexMapper,
        bin_assign: Tensor,
        patch_size: int = 11,
        frames_per_batch: int = 500,
        device: torch.device | None = None,
        z_threshold: float | None = None,
        prefetch: int = 2,
    ):
        self._frame_to_obs = frame_to_obs
        self.core = core
        self.hkl_index = hkl_index
        self.patch_size = patch_size
        self.frames_per_batch = frames_per_batch
        self.device = device or torch.device("cpu")
        self.z_threshold = z_threshold or sparse_store.z_threshold
        self.prefetch = prefetch
        self._n_good = sparse_store.n_good_pixels

        # Frame indices that have observations
        self._frame_keys: list[int] = sorted(frame_to_obs.keys())

        # Move all static data to GPU — batch construction happens entirely
        # on device, eliminating CPU→GPU transfer bottleneck.
        dev = self.device
        self._profiles = sparse_store.profiles.to(dev)       # (n_frames, n_bins)
        self._bin_assign = bin_assign.to(dev)                 # (n_good_pixels,)
        self._mapper = PixelIndexMapper(pixel_mapper._lookup.to(dev))

        # Core observation data on GPU for fast indexing
        self._crystal_id = core.crystal_id.to(dev)
        self._hkl = core.hkl.to(dev)
        self._hkl_id = hkl_index.hkl_id.to(dev)
        self._panel_id = core.panel_id.to(dev)
        self._fs = core.fs.float().to(dev)
        self._ss = core.ss.float().to(dev)

        # Build per-frame signal index on CPU (memory-efficient)
        self._build_frame_signal_index(sparse_store)

    def _build_frame_signal_index(self, store: SparseFrameStore):
        """Build per-frame offset table for signal entries.

        Keeps signal data on CPU. For each batch, only the relevant frames'
        entries are sliced and moved to GPU. This scales to billions of signal
        pixels without GPU memory pressure.
        """
        # Sort by frame on CPU
        sig_frame = store.signal_frame.long()
        order = sig_frame.argsort()
        self._sig_frame_sorted = sig_frame[order]       # (n_total,) on CPU
        self._sig_idx_sorted = store.signal_idx.long()[order]  # (n_total,) on CPU
        self._sig_val_sorted = store.signal_val[order]         # (n_total,) on CPU

        # Build frame → (start, end) offset table
        n_total = sig_frame.numel()
        if n_total == 0:
            self._frame_offsets: dict[int, tuple[int, int]] = {}
            return

        unique_frames, counts = self._sig_frame_sorted.unique_consecutive(
            return_counts=True,
        )
        ends = counts.cumsum(0)
        starts = ends - counts
        self._frame_offsets = {
            f.item(): (s.item(), e.item())
            for f, s, e in zip(unique_frames, starts, ends)
        }

    def update_core(
        self,
        core: _SFXStreamCore,
        hkl_index: HKLIndex,
        frame_to_obs: dict[int, Tensor],
    ):
        """Rebuild after re-prediction."""
        self.core = core
        self.hkl_index = hkl_index
        self._frame_to_obs = frame_to_obs
        self._frame_keys = sorted(frame_to_obs.keys())

        # Update GPU copies of core data
        dev = self.device
        self._crystal_id = core.crystal_id.to(dev)
        self._hkl = core.hkl.to(dev)
        self._hkl_id = hkl_index.hkl_id.to(dev)
        self._panel_id = core.panel_id.to(dev)
        self._fs = core.fs.float().to(dev)
        self._ss = core.ss.float().to(dev)

    @property
    def n_observations(self) -> int:
        return sum(t.numel() for t in self._frame_to_obs.values())

    def __len__(self) -> int:
        return ceil(len(self._frame_keys) / self.frames_per_batch)

    def __iter__(self) -> Iterator[SparseBatch]:
        n_frames = len(self._frame_keys)
        if n_frames == 0:
            return

        frame_order = torch.randperm(n_frames)

        batch_groups = []
        for start in range(0, n_frames, self.frames_per_batch):
            end = min(start + self.frames_per_batch, n_frames)
            batch_groups.append(frame_order[start:end])

        prefetch_queue: Queue[SparseBatch | None] = Queue(maxsize=self.prefetch)

        def _producer():
            try:
                for group in batch_groups:
                    batch = self._make_batch(group)
                    if batch is not None:
                        prefetch_queue.put(batch)
            except Exception as e:
                prefetch_queue.put(e)
            finally:
                prefetch_queue.put(None)

        thread = Thread(target=_producer, daemon=True)
        thread.start()

        while True:
            item = prefetch_queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    def _make_batch(self, frame_group: Tensor) -> SparseBatch | None:
        """Build one SparseBatch from a set of frames.

        Signal data is sliced per-batch from CPU and moved to GPU.
        Only the frames in this batch contribute to the searchsorted,
        keeping GPU memory proportional to batch size, not dataset size.
        """
        n_good = self._n_good
        ph = pw = self.patch_size
        dev = self.device

        # ── Gather observations, frame ids, and signal slices (CPU) ─
        obs_lists = []
        frame_ids = []
        sig_slices = []
        batch_frames = set()
        for gi in frame_group:
            fk = self._frame_keys[gi.item()]
            obs = self._frame_to_obs[fk]
            if obs.numel() > 0:
                obs_lists.append(obs)
                frame_ids.extend([fk] * obs.numel())
                batch_frames.add(fk)

            # Collect signal entries for this frame
            if fk in self._frame_offsets:
                s, e = self._frame_offsets[fk]
                sig_slices.append((fk, s, e))

        if not obs_lists:
            return None

        obs_idx = torch.cat(obs_lists).to(dev)
        B = obs_idx.numel()
        if B == 0:
            return None

        frame_ids_t = torch.tensor(frame_ids, dtype=torch.long, device=dev)

        # ── Build batch-local signal index on GPU ────────────────
        # Gather only signal entries for frames in this batch
        if sig_slices:
            idx_parts = []
            val_parts = []
            frame_parts = []
            for fk, s, e in sig_slices:
                idx_parts.append(self._sig_idx_sorted[s:e])
                val_parts.append(self._sig_val_sorted[s:e])
                frame_parts.append(
                    torch.full((e - s,), fk, dtype=torch.long)
                )
            batch_sig_idx = torch.cat(idx_parts).to(dev)
            batch_sig_val = torch.cat(val_parts).to(dev)
            batch_sig_frame = torch.cat(frame_parts).to(dev)

            # Composite key and sort (small — only this batch's frames)
            batch_key = batch_sig_frame * n_good + batch_sig_idx
            del batch_sig_idx, batch_sig_frame
            order = batch_key.argsort()
            batch_key_sorted = batch_key[order]
            batch_val_sorted = batch_sig_val[order]
            del batch_key, batch_sig_val, order
        else:
            batch_key_sorted = torch.empty(0, dtype=torch.long, device=dev)
            batch_val_sorted = torch.empty(0, dtype=torch.long, device=dev)

        # ── Core data (GPU indexing) ───────────────────────────────
        crystal_id = self._crystal_id[obs_idx]
        hkl = self._hkl[obs_idx]
        hkl_id = self._hkl_id[obs_idx]
        panel_id = self._panel_id[obs_idx]
        fs_center = self._fs[obs_idx]
        ss_center = self._ss[obs_idx]

        # ── Pixel grids (GPU) ──────────────────────────────────────
        fs_grid, ss_grid, pid_grid = _build_pixel_grids(
            fs_center, ss_center, panel_id, ph, pw,
        )

        # ── Map pixel grids to flat good-pixel indices (GPU) ───────
        flat_idx, on_good = self._mapper.grid_to_flat(ss_grid, fs_grid)

        # ── Per-pixel background (GPU) ─────────────────────────────
        flat_idx_safe = flat_idx.clamp(min=0)
        bin_idx = self._bin_assign[flat_idx_safe]
        frame_exp = frame_ids_t[:, None, None].expand(B, ph, pw)
        background = self._profiles[frame_exp, bin_idx]
        del bin_idx

        # ── Sparse annotation: searchsorted on batch-local index ──
        query_flat = (frame_exp * n_good + flat_idx_safe).reshape(-1)
        del frame_exp, flat_idx_safe, flat_idx

        if batch_key_sorted.numel() > 0:
            pos = torch.searchsorted(batch_key_sorted, query_flat)
            pos.clamp_(max=batch_key_sorted.numel() - 1)
            hit = batch_key_sorted[pos] == query_flat
            del query_flat

            is_sparse = hit.reshape(B, ph, pw) & on_good
            sparse_counts = batch_val_sorted[pos].reshape(B, ph, pw) * is_sparse.int()
            del hit, pos, batch_key_sorted, batch_val_sorted
        else:
            del query_flat, batch_key_sorted, batch_val_sorted
            is_sparse = torch.zeros(B, ph, pw, dtype=torch.bool, device=dev)
            sparse_counts = torch.zeros(B, ph, pw, dtype=torch.int32, device=dev)

        # ── Censoring threshold (GPU) ──────────────────────────────
        threshold = compute_censoring_threshold(background, self.z_threshold)

        return SparseBatch(
            crystal_id=crystal_id,
            hkl_id=hkl_id,
            hkl=hkl,
            panel_id=panel_id,
            fs_center=fs_center,
            ss_center=ss_center,
            obs_idx=obs_idx,
            fs_grid=fs_grid,
            ss_grid=ss_grid,
            pid_grid=pid_grid,
            background=background,
            is_sparse=is_sparse,
            sparse_counts=sparse_counts,
            threshold=threshold,
            valid=on_good,
        )
