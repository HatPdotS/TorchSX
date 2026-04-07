"""FrameBatchLoader: GPU-ready batches from FrameStore with on-the-fly extraction."""

from __future__ import annotations

from math import ceil
from queue import Queue
from threading import Thread
from typing import Iterator

import numpy as np
import torch
from torch import Tensor

from torchsx._core import _SFXStreamCore
from torchsx.io._excise import _parse_event_id
from torchsx.io._frame_store import FrameStore
from torchsx.io._frame_extract import ROIExtractor
from torchsx.merge._hkl import HKLIndex
from torchsx.forward.batch import ROIBatch, _build_pixel_grids
from torchsx.forward.corrections import estimate_background


class FrameBatchLoader:
    """Batch iterator grouped by frames (not crystals).

    Produces ROIBatch objects compatible with PixelForwardModel.
    Uses ROIExtractor with blosc2-internal threading for decompression.
    A background prefetch thread overlaps CPU decompress+extract with
    GPU forward/backward (CUDA ops release the GIL).

    Parameters
    ----------
    frame_store : FrameStore
        Compressed frame archive.
    core : _SFXStreamCore
        Stream data (observations, crystals, images).
    hkl_index : HKLIndex
        HKL index mapping.
    patch_size : int
        Patch width/height (must be odd).
    frames_per_batch : int
        Number of frames per batch.
    device : torch.device
        Target GPU device for batches.
    obs_mask : Tensor | None
        Boolean mask selecting observations to include.
    mask : np.ndarray | None
        Detector bad-pixel mask (True = bad).
    nthreads : int
        blosc2 decompression threads (scales ~linearly up to 16).
    prefetch : int
        Number of batches to prefetch.
    """

    def __init__(
        self,
        frame_store: FrameStore,
        core: _SFXStreamCore,
        hkl_index: HKLIndex,
        patch_size: int = 11,
        frames_per_batch: int = 500,
        device: torch.device | None = None,
        obs_mask: Tensor | None = None,
        mask: np.ndarray | None = None,
        nthreads: int = 8,
        prefetch: int = 2,
    ):
        self._store = frame_store
        self.core = core
        self.hkl_index = hkl_index
        self.patch_size = patch_size
        self.frames_per_batch = frames_per_batch
        self.device = device or torch.device("cpu")
        self.obs_mask = obs_mask
        self.prefetch = prefetch

        self._extractor = ROIExtractor(
            frame_store, patch_size=patch_size, mask=mask, nthreads=nthreads,
        )

        self._build_frame_obs_map()

    def _build_frame_obs_map(self):
        """Map each frame key → list of observation indices.

        Uses core.crystal_id → core.image_id → metadata (filename, event).
        Filters by obs_mask if provided.
        """
        core = self.core
        meta = core.metadata
        filenames = meta["image"]["filename"]
        events = meta["image"]["event"]

        # Select observations
        if self.obs_mask is not None:
            obs_indices = self.obs_mask.nonzero(as_tuple=True)[0].long()
        else:
            obs_indices = torch.arange(core.n_observations, dtype=torch.long)

        # Map obs → image_id
        image_ids = core.image_id[core.crystal_id[obs_indices]]

        # Group observations by (filename, event)
        frame_obs: dict[tuple[str, int], list[int]] = {}
        image_ids_np = image_ids.numpy()

        for i, obs_i in enumerate(obs_indices.numpy()):
            img_id = image_ids_np[i]
            fn = filenames[img_id]
            ev = _parse_event_id(events[img_id])
            key = (fn, ev)
            if key not in self._store:
                continue
            frame_obs.setdefault(key, []).append(int(obs_i))

        # Convert to ordered lists
        self._frame_keys: list[tuple[str, int]] = list(frame_obs.keys())
        self._frame_obs: list[Tensor] = [
            torch.tensor(frame_obs[k], dtype=torch.long)
            for k in self._frame_keys
        ]

    def update_core(
        self,
        core: _SFXStreamCore,
        hkl_index: HKLIndex,
        obs_mask: Tensor | None = None,
    ):
        """Rebuild frame-obs mapping after re-prediction.

        Called when predict_reflections() adds/removes/moves observations.
        No ROIStore re-excision needed — patches extracted on-the-fly.
        """
        self.core = core
        self.hkl_index = hkl_index
        self.obs_mask = obs_mask
        self._build_frame_obs_map()

    @property
    def n_observations(self) -> int:
        """Total observations across all frames."""
        return sum(t.numel() for t in self._frame_obs)

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return ceil(len(self._frame_keys) / self.frames_per_batch)

    def __iter__(self) -> Iterator[ROIBatch]:
        """Yield ROIBatch objects with prefetching."""
        n_frames = len(self._frame_keys)
        if n_frames == 0:
            return

        frame_order = torch.randperm(n_frames)

        # Partition into batch groups
        batch_groups = []
        for start in range(0, n_frames, self.frames_per_batch):
            end = min(start + self.frames_per_batch, n_frames)
            batch_groups.append(frame_order[start:end])

        # Prefetch with background thread
        prefetch_queue: Queue[ROIBatch | None] = Queue(maxsize=self.prefetch)

        def _producer():
            try:
                for group in batch_groups:
                    batch = self._make_batch(group)
                    if batch is not None:
                        prefetch_queue.put(batch)
            except Exception as e:
                # Put the exception so consumer can re-raise
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

    def _make_batch(self, frame_indices: Tensor) -> ROIBatch | None:
        """Build one ROIBatch from a set of frames."""
        core = self.core
        ph = pw = self.patch_size

        # Gather observation indices for all frames in this group
        obs_lists = [self._frame_obs[i.item()] for i in frame_indices]
        frame_keys = [self._frame_keys[i.item()] for i in frame_indices]

        # Filter empty frames
        nonempty = [(k, o) for k, o in zip(frame_keys, obs_lists) if o.numel() > 0]
        if not nonempty:
            return None

        frame_keys_ne, obs_lists_ne = zip(*nonempty)
        obs_idx = torch.cat(obs_lists_ne)

        if obs_idx.numel() == 0:
            return None

        # Pixel centers
        fs = core.fs[obs_idx].numpy()
        ss = core.ss[obs_idx].numpy()
        fs_int = np.round(fs).astype(np.int64)
        ss_int = np.round(ss).astype(np.int64)

        # Build per-frame center arrays
        centers_per_frame = []
        offset = 0
        for obs_list in obs_lists_ne:
            n = obs_list.numel()
            centers = np.column_stack([ss_int[offset:offset + n],
                                       fs_int[offset:offset + n]])
            centers_per_frame.append(centers)
            offset += n

        # Serial decompress + extract (blosc2 internal threading)
        patches_np, valid_np = self._extractor.extract_batch(
            list(frame_keys_ne), centers_per_frame,
        )

        # Build ROIBatch on target device
        dev = self.device
        patches_t = torch.from_numpy(patches_np).float().to(dev)
        valid_t = torch.from_numpy(valid_np).to(dev)
        crystal_id = core.crystal_id[obs_idx].to(dev)
        hkl = core.hkl[obs_idx].to(dev)
        hkl_id = self.hkl_index.hkl_id[obs_idx].to(dev)
        panel_id = core.panel_id[obs_idx].to(dev)
        fs_center = core.fs[obs_idx].float().to(dev)
        ss_center = core.ss[obs_idx].float().to(dev)

        bg = estimate_background(patches_t, valid_t)

        fs_grid, ss_grid, pid_grid = _build_pixel_grids(
            fs_center, ss_center, panel_id, ph, pw,
        )

        return ROIBatch(
            patches=patches_t,
            valid=valid_t,
            crystal_id=crystal_id,
            hkl_id=hkl_id,
            hkl=hkl,
            panel_id=panel_id,
            fs_center=fs_center,
            ss_center=ss_center,
            background=bg,
            obs_idx=obs_idx.to(dev),
            fs_grid=fs_grid,
            ss_grid=ss_grid,
            pid_grid=pid_grid,
        )

    def close(self):
        """Clean up resources."""
        self._extractor.close()
