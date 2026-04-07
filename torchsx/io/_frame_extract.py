"""ROIExtractor: decompress + patch extraction from FrameStore.

Uses blosc2's internal multi-threading for decompression (nthreads=8–16
scales nearly linearly). ThreadPoolExecutor is not used because
blosc2.decompress_chunk holds the GIL, so Python-level threading gives
zero speedup. The prefetch thread in FrameBatchLoader still provides
useful overlap: while GPU kernels run (GIL released), the next batch's
decompress+extract can proceed.
"""

from __future__ import annotations

import numpy as np

from torchsx.io._frame_store import FrameStore
from torchsx.io._excise import _extract_frame_patches, _clip_patch


class ROIExtractor:
    """Extract ROI patches from compressed frames.

    Decompression uses blosc2 internal threading (``nthreads``).
    Frames are decompressed serially; parallelism comes from blosc2
    spawning ``nthreads`` C-level threads per chunk.

    Usage
    -----
    >>> extractor = ROIExtractor(store, patch_size=11, mask=det_mask, nthreads=8)
    >>> patches, valid = extractor.extract("run.h5", 42,
    ...     centers=np.array([[100, 200], [300, 400]]))
    >>> patches, valid = extractor.extract_batch(frame_keys, centers_per_frame)
    """

    def __init__(
        self,
        store: FrameStore,
        patch_size: int = 11,
        mask: np.ndarray | None = None,
        nthreads: int = 8,
    ):
        import blosc2

        self._store = store
        self._hh = patch_size // 2
        self._hw = patch_size // 2
        self._patch_size = patch_size
        self._mask = mask

        # Single schunk handle; blosc2 internal threading handles parallelism.
        self._schunk = blosc2.open(str(store._schunk_path), mode="r")
        self._schunk.update_dparams(blosc2.DParams(nthreads=nthreads))

    def extract(
        self,
        filename: str,
        event: int,
        centers: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Single-frame extraction.

        Parameters
        ----------
        filename, event : frame key
        centers : (N, 2) array of [ss, fs] integer pixel centers

        Returns
        -------
        patches : (N, ph, pw) float32
        valid : (N, ph, pw) bool
        """
        chunk_id = self._store.key_to_id((filename, event))
        return self._extract_one_frame(chunk_id, centers[:, 0], centers[:, 1])

    def extract_batch(
        self,
        frame_keys: list[tuple[str, int]],
        centers_per_frame: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Multi-frame extraction (serial decompress, blosc2-internal threading).

        Parameters
        ----------
        frame_keys : list of (filename, event) tuples
        centers_per_frame : list of (N_i, 2) arrays of [ss, fs] centers

        Returns
        -------
        patches : (N_total, ph, pw) float32
        valid : (N_total, ph, pw) bool
        """
        n_total = sum(c.shape[0] for c in centers_per_frame)
        ph = 2 * self._hh + 1
        pw = 2 * self._hw + 1

        if n_total == 0:
            return (
                np.empty((0, ph, pw), dtype=np.float32),
                np.empty((0, ph, pw), dtype=bool),
            )

        patches_out = np.empty((n_total, ph, pw), dtype=np.float32)
        valid_out = np.empty((n_total, ph, pw), dtype=bool)

        offset = 0
        for key, centers in zip(frame_keys, centers_per_frame):
            n = centers.shape[0]
            if n == 0:
                continue
            chunk_id = self._store.key_to_id(key)
            p, v = self._extract_one_frame(
                chunk_id, centers[:, 0], centers[:, 1],
            )
            patches_out[offset : offset + n] = p
            valid_out[offset : offset + n] = v
            offset += n

        return patches_out, valid_out

    def _extract_one_frame(
        self,
        chunk_id: int,
        centers_ss: np.ndarray,
        centers_fs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decompress one frame and extract patches."""
        buf = self._schunk.decompress_chunk(chunk_id)
        frame = np.frombuffer(buf, dtype=self._store.dtype).reshape(
            self._store.frame_shape
        )

        patches, valid = _extract_frame_patches(
            frame, centers_ss, centers_fs, self._hh, self._hw, 0.0,
        )

        if self._mask is not None:
            self._apply_mask(patches, valid, centers_ss, centers_fs)

        return patches, valid

    def _apply_mask(
        self,
        patches: np.ndarray,
        valid: np.ndarray,
        centers_ss: np.ndarray,
        centers_fs: np.ndarray,
    ) -> None:
        """Apply detector bad-pixel mask to extracted patches (in-place)."""
        mask = self._mask
        for j in range(len(centers_ss)):
            ss_src, ss_dst = _clip_patch(centers_ss[j], self._hh, mask.shape[0])
            fs_src, fs_dst = _clip_patch(centers_fs[j], self._hw, mask.shape[1])
            if ss_src.start < ss_src.stop and fs_src.start < fs_src.stop:
                mask_patch = mask[ss_src, fs_src]
                valid[j, ss_dst, fs_dst] &= ~mask_patch

    def close(self):
        """No-op (no thread pool to shut down)."""
        pass
