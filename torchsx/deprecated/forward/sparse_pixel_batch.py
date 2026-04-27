"""SparsePixelBatchLoader: produces PixelReflectionPairs batches grouped by frames."""

from __future__ import annotations

from math import ceil
from queue import Queue
from threading import Thread
from typing import Iterator

import torch
from torch import Tensor

from torchsx._core import _SFXStreamCore
from torchsx.deprecated.merge._hkl import HKLIndex
from torchsx.deprecated.forward._sparse_data import SparseFrameStore, PixelIndexMapper
from torchsx.deprecated.forward._pixel_matching import (
    PixelReflectionPairs,
    match_pixels_to_reflections,
)


class SparsePixelBatchLoader:
    """Batch iterator producing PixelReflectionPairs from sparse compressed data.

    Groups observations by frame, matches sparse pixels to nearby reflections,
    and produces flat pair-indexed tensors for the forward model.

    Parameters
    ----------
    sparse_store : SparseFrameStore
    frame_to_obs : dict[int, Tensor]
    core : _SFXStreamCore
    hkl_index : HKLIndex
    pixel_mapper : PixelIndexMapper
    bin_assign : Tensor — (n_good_pixels,) int64 radial bin index
    search_radius : float — max distance in pixels for matching
    frames_per_batch : int
    device : torch.device
    prefetch : int — number of batches to prefetch
    """

    def __init__(
        self,
        sparse_store: SparseFrameStore,
        frame_to_obs: dict[int, Tensor],
        core: _SFXStreamCore,
        hkl_index: HKLIndex,
        pixel_mapper: PixelIndexMapper,
        bin_assign: Tensor,
        search_radius: float = 15.0,
        frames_per_batch: int = 50,
        device: torch.device | None = None,
        prefetch: int = 2,
    ):
        self._frame_to_obs = frame_to_obs
        self.core = core
        self.hkl_index = hkl_index
        self.search_radius = search_radius
        self.frames_per_batch = frames_per_batch
        self.device = device or torch.device("cpu")
        self.prefetch = prefetch
        self._n_good = sparse_store.n_good_pixels

        self._frame_keys: list[int] = sorted(frame_to_obs.keys())

        dev = self.device

        # Static data on GPU
        self._profiles = sparse_store.profiles.to(dev)
        self._bin_assign = bin_assign.to(dev)
        self._mapper = PixelIndexMapper(pixel_mapper._lookup.to(dev))

        # Core observation data on GPU
        self._crystal_id = core.crystal_id.to(dev)
        self._hkl = core.hkl.to(dev)
        self._hkl_id = hkl_index.hkl_id.to(dev)
        self._panel_id = core.panel_id.to(dev)
        self._fs = core.fs.float().to(dev)
        self._ss = core.ss.float().to(dev)

        # Panel map for sparse pixel → panel_id
        self._panel_map = core.geometry.build_panel_map().to(dev)

        # Build per-frame signal index (CPU, sliced per batch)
        self._build_frame_signal_index(sparse_store)

    def _build_frame_signal_index(self, store: SparseFrameStore):
        """Build per-frame offset table for signal entries on CPU."""
        sig_frame = store.signal_frame.long()
        order = sig_frame.argsort()
        self._sig_frame_sorted = sig_frame[order]
        self._sig_idx_sorted = store.signal_idx.long()[order]
        self._sig_val_sorted = store.signal_val[order]

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

    def __iter__(self) -> Iterator[PixelReflectionPairs]:
        n_frames = len(self._frame_keys)
        if n_frames == 0:
            return

        frame_order = torch.randperm(n_frames)

        batch_groups = []
        for start in range(0, n_frames, self.frames_per_batch):
            end = min(start + self.frames_per_batch, n_frames)
            batch_groups.append(frame_order[start:end])

        prefetch_queue: Queue[PixelReflectionPairs | None | Exception] = Queue(
            maxsize=self.prefetch
        )

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

    def _make_batch(self, frame_group: Tensor) -> PixelReflectionPairs | None:
        """Build one PixelReflectionPairs batch from a set of frames."""
        dev = self.device
        n_good = self._n_good

        # ── Gather observations and signal slices ──────────────────
        obs_lists = []
        frame_ids_per_obs = []
        sig_slices = []

        for gi in frame_group:
            fk = self._frame_keys[gi.item()]
            obs = self._frame_to_obs[fk]
            if obs.numel() > 0:
                obs_lists.append(obs)
                frame_ids_per_obs.extend([fk] * obs.numel())

            if fk in self._frame_offsets:
                s, e = self._frame_offsets[fk]
                sig_slices.append((fk, s, e))

        if not obs_lists:
            return None

        obs_idx = torch.cat(obs_lists).to(dev)
        N_refl = obs_idx.numel()
        if N_refl == 0:
            return None

        refl_frame = torch.tensor(frame_ids_per_obs, dtype=torch.long, device=dev)

        # ── Reflection data ────────────────────────────────────────
        crystal_id = self._crystal_id[obs_idx]
        hkl = self._hkl[obs_idx]
        hkl_id = self._hkl_id[obs_idx]
        panel_id = self._panel_id[obs_idx]
        fs_pred = self._fs[obs_idx]
        ss_pred = self._ss[obs_idx]

        # ── Gather sparse pixels for these frames ──────────────────
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
            sig_flat_idx = torch.cat(idx_parts).to(dev)
            sig_val = torch.cat(val_parts).to(dev)
            sig_frame = torch.cat(frame_parts).to(dev)
        else:
            return None  # no sparse data for these frames

        N_pixels = sig_flat_idx.shape[0]
        if N_pixels == 0:
            return None

        # Convert flat good-pixel index → (ss, fs)
        pixel_ss, pixel_fs = self._mapper.flat_to_ss_fs(sig_flat_idx)
        pixel_ss = pixel_ss.float()
        pixel_fs = pixel_fs.float()

        # Panel ID from ss coordinate
        pixel_panel = self._panel_map[pixel_ss.long()]

        # Background from radial profiles
        bin_idx = self._bin_assign[sig_flat_idx.long()]
        pixel_bg = self._profiles[sig_frame, bin_idx]

        pixel_counts = sig_val.int()

        # ── Match pixels to reflections ────────────────────────────
        refl_idx, pixel_idx, pair_ptr = match_pixels_to_reflections(
            refl_ss=ss_pred,
            refl_fs=fs_pred,
            refl_frame=refl_frame,
            pixel_ss=pixel_ss,
            pixel_fs=pixel_fs,
            pixel_frame=sig_frame,
            pixel_counts=pixel_counts,
            pixel_bg=pixel_bg,
            search_radius=self.search_radius,
        )

        if refl_idx.numel() == 0:
            return None

        return PixelReflectionPairs(
            crystal_id=crystal_id,
            hkl_id=hkl_id,
            hkl=hkl,
            panel_id=panel_id,
            ss_pred=ss_pred,
            fs_pred=fs_pred,
            pixel_ss=pixel_ss,
            pixel_fs=pixel_fs,
            pixel_panel=pixel_panel,
            pixel_counts=pixel_counts,
            pixel_bg=pixel_bg,
            refl_idx=refl_idx,
            pixel_idx=pixel_idx,
            pair_ptr=pair_ptr,
        )
