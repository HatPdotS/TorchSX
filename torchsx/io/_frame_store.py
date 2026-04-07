"""FrameStore: compressed frame archive backed by blosc2 SChunk.

On-disk format: directory containing:
  - frames.blosc2    — one chunk per frame, LZ4+bitshuffle, typesize=2
  - index.json       — maps (filename, event) → chunk_id + metadata
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from torchsx._core import _SFXStreamCore
from torchsx.io._excise import _parse_event_id


class FrameStore:
    """Compressed frame archive backed by blosc2 SChunk.

    Usage
    -----
    >>> store = FrameStore.from_core(core, "frame_store/", n_threads=8)
    >>> store = FrameStore("frame_store/")
    >>> frame = store["run001.h5", 42]   # (H, W) int16 ndarray
    >>> len(store)                        # number of frames
    """

    def __init__(self, directory: str | Path):
        import blosc2

        self._dir = Path(directory)
        self._schunk_path = self._dir / "frames.blosc2"
        self._index_path = self._dir / "index.json"

        if not self._schunk_path.exists():
            raise FileNotFoundError(f"No frames.blosc2 in {self._dir}")
        if not self._index_path.exists():
            raise FileNotFoundError(f"No index.json in {self._dir}")

        self._schunk = blosc2.open(str(self._schunk_path), mode="r")

        with open(self._index_path) as f:
            index_data = json.load(f)

        self._frame_shape = tuple(index_data["frame_shape"])
        self._dtype = np.dtype(index_data["dtype"])

        # key → chunk_id mapping: store as dict with string keys
        # "(filename, event)" → chunk_id
        self._key_to_id: dict[tuple[str, int], int] = {}
        self._id_to_key: list[tuple[str, int]] = []
        for entry in index_data["frames"]:
            key = (entry["filename"], entry["event"])
            cid = entry["chunk_id"]
            self._key_to_id[key] = cid
            # Ensure list is large enough
            while len(self._id_to_key) <= cid:
                self._id_to_key.append(("", -1))
            self._id_to_key[cid] = key

    @classmethod
    def from_core(
        cls,
        core: _SFXStreamCore,
        output_dir: str | Path,
        data_path: str | None = None,
        clevel: int = 5,
        n_threads: int = 8,
    ) -> FrameStore:
        """Build store from all unique images in an _SFXStreamCore.

        Enumerates unique (filename, event) pairs, reads each frame from H5,
        compresses with blosc2 LZ4+bitshuffle, and appends to an SChunk.
        """
        # Resolve data path from geometry if not given
        if data_path is None:
            if core.geometry is not None and hasattr(core.geometry, "data_path"):
                data_path = core.geometry.data_path
            else:
                data_path = "/data/data"

        meta = core.metadata
        filenames = meta["image"]["filename"]
        events = meta["image"]["event"]
        n_images = core.beam_energy.shape[0]

        # Collect unique (filename, event) pairs
        pairs: list[tuple[str, int]] = []
        seen: set[tuple[str, int]] = set()
        for i in range(n_images):
            fn = filenames[i]
            ev = _parse_event_id(events[i])
            key = (fn, ev)
            if key not in seen:
                seen.add(key)
                pairs.append(key)

        # Sort by (filename, event) for sequential H5 reads
        pairs.sort()

        return cls.from_h5_files(
            pairs, output_dir, data_path=data_path,
            clevel=clevel, n_threads=n_threads,
        )

    @classmethod
    def from_h5_files(
        cls,
        file_event_pairs: list[tuple[str, int]],
        output_dir: str | Path,
        data_path: str = "/data/data",
        clevel: int = 5,
        n_threads: int = 8,
    ) -> FrameStore:
        """Build store from explicit (filename, event) pairs."""
        import blosc2
        import h5py

        try:
            import bitshuffle.h5  # noqa: F401
        except ImportError:
            pass

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        schunk_path = output_dir / "frames.blosc2"
        index_path = output_dir / "index.json"

        # Group by file for sequential I/O
        from collections import OrderedDict
        file_groups: OrderedDict[str, list[int]] = OrderedDict()
        for fn, ev in file_event_pairs:
            file_groups.setdefault(fn, []).append(ev)

        # Read first frame to determine shape
        first_fn, first_ev = file_event_pairs[0]
        with h5py.File(first_fn, "r") as f:
            dset = f[data_path]
            frame_shape = dset.shape[1:]  # (H, W)
            dtype = np.int16  # force int16

        frame_nbytes = int(np.prod(frame_shape)) * np.dtype(dtype).itemsize

        # Create SChunk with LZ4 + bitshuffle
        cparams = blosc2.CParams(
            codec=blosc2.Codec.LZ4,
            clevel=clevel,
            filters=[blosc2.Filter.BITSHUFFLE],
            typesize=2,
            nthreads=n_threads,
        )
        schunk = blosc2.SChunk(
            chunksize=frame_nbytes,
            mode="w",
            urlpath=str(schunk_path),
            cparams=cparams,
        )

        # Read frames and compress
        index_entries: list[dict] = []
        chunk_id = 0
        n_total = len(file_event_pairs)

        current_h5 = None
        current_fn = None

        for fn, events_for_file in file_groups.items():
            events_sorted = sorted(events_for_file)
            try:
                current_h5 = h5py.File(fn, "r")
                dset = current_h5[data_path]
            except (OSError, FileNotFoundError, KeyError) as e:
                warnings.warn(f"Cannot open {fn}: {e}")
                if current_h5 is not None:
                    current_h5.close()
                    current_h5 = None
                continue

            for ev in events_sorted:
                try:
                    frame = np.asarray(dset[ev], dtype=dtype)
                except Exception as e:
                    warnings.warn(f"Cannot read event {ev} from {fn}: {e}")
                    continue

                if frame.shape != frame_shape:
                    warnings.warn(
                        f"Frame shape mismatch: {frame.shape} vs {frame_shape} "
                        f"for {fn} event {ev}"
                    )
                    continue

                schunk.append_data(frame)
                index_entries.append({
                    "filename": fn,
                    "event": ev,
                    "chunk_id": chunk_id,
                })
                chunk_id += 1

                if chunk_id % 500 == 0:
                    print(f"  [{chunk_id}/{n_total}] frames compressed")

            current_h5.close()

        print(f"  [{chunk_id}/{n_total}] frames compressed (done)")

        # Write index
        index_data = {
            "frame_shape": list(frame_shape),
            "dtype": str(np.dtype(dtype)),
            "n_frames": chunk_id,
            "frames": index_entries,
        }
        with open(index_path, "w") as f:
            json.dump(index_data, f)

        return cls(output_dir)

    def __getitem__(self, key: tuple[str, int]) -> np.ndarray:
        """Decompress and return frame as (H, W) array."""
        chunk_id = self._key_to_id[key]
        return self.decompress_by_id(chunk_id)

    def get_compressed(self, key: tuple[str, int]) -> bytes:
        """Return raw compressed chunk bytes."""
        chunk_id = self._key_to_id[key]
        return self._schunk.get_chunk(chunk_id)

    def decompress_by_id(self, chunk_id: int) -> np.ndarray:
        """Decompress by chunk_id (faster, skips key lookup)."""
        buf = self._schunk.decompress_chunk(chunk_id)
        return np.frombuffer(buf, dtype=self._dtype).reshape(self._frame_shape)

    def key_to_id(self, key: tuple[str, int]) -> int:
        """Look up chunk_id for a (filename, event) key."""
        return self._key_to_id[key]

    def id_to_key(self, chunk_id: int) -> tuple[str, int]:
        """Look up (filename, event) for a chunk_id."""
        return self._id_to_key[chunk_id]

    @property
    def frame_shape(self) -> tuple[int, int]:
        return self._frame_shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def __len__(self) -> int:
        return len(self._key_to_id)

    def __contains__(self, key: tuple[str, int]) -> bool:
        return key in self._key_to_id
