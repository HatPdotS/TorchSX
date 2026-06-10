"""Read SwissFEL JUNGFRAU detector frames from HDF5.

A thin source over the per-acquisition ``acq*.JF06T32V07.h5`` files written by the
SwissFEL DAQ.  The detector data lives at ``/data/JF06T32V07/data`` with shape
``(n_frames, 16448, 1030)`` int32 and is **already gain-corrected to photons**
(``meta/conversion=True``) — no pedestal/gain correction is applied here.

The data is stored with the bitshuffle HDF5 filter (id 32008); importing
:mod:`hdf5plugin` at module load registers that filter so frame *values* can be
read (without it, only shapes are readable).
"""

from __future__ import annotations

from pathlib import Path

import hdf5plugin  # noqa: F401  — registers the bitshuffle (32008) HDF5 filter
import h5py
import numpy as np
import torch

from torchsx._core import abspath_str

# JUNGFRAU JF06T32V07: 32 modules stacked along the slow-scan axis, each 514×1030.
JF_N_PANELS = 32
JF_PANEL_SS = 514
JF_PANEL_FS = 1030

# Default location of the detector cube inside each acq file (matches the .geom
# ``data = /data/JF06T32V07/data`` entry).
DEFAULT_DATA_PATH = "/data/JF06T32V07/data"


class JungfrauReader:
    """Random-access reader for one ``acq*.JF06T32V07.h5`` file.

    Parameters
    ----------
    path : str | Path
        Path to an ``acq*.JF06T32V07.h5`` file.
    data_path : str, optional
        HDF5 path to the detector cube.  Defaults to
        :data:`DEFAULT_DATA_PATH`; pass ``geom.data_path`` to follow the .geom.

    Notes
    -----
    Per-frame metadata (``pulse_id``, ``is_good_frame``) and the detector
    ``pixel_mask`` / ``good_pixels_fraction`` are read from sibling datasets
    under the same group (``/data/JF06T32V07``).
    """

    def __init__(self, path: str | Path, data_path: str = DEFAULT_DATA_PATH):
        # Absolute so the path recorded in the stream (image filename, builder
        # key) resolves regardless of the working directory at reload time.
        self.path = abspath_str(str(path))
        self.data_path = data_path
        self._group_path = data_path.rsplit("/", 1)[0]  # e.g. /data/JF06T32V07
        self._h5: h5py.File | None = h5py.File(self.path, "r")
        self._data = self._h5[data_path]
        if self._data.ndim != 3:
            raise ValueError(
                f"expected a 3-D detector cube at {data_path!r}, "
                f"got shape {self._data.shape}"
            )
        self._gpu_decoders: dict = {}  # device-str -> Bslz4GpuDecoder | None (lazy)

    # ── lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __enter__(self) -> "JungfrauReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __len__(self) -> int:
        return self.n_frames

    def _grp(self) -> h5py.Group:
        return self._h5[self._group_path]

    # ── shape / metadata ──────────────────────────────────────────────────

    @property
    def n_frames(self) -> int:
        return int(self._data.shape[0])

    @property
    def detector_shape(self) -> tuple[int, int]:
        return (int(self._data.shape[1]), int(self._data.shape[2]))

    @property
    def n_panels(self) -> int:
        return self.detector_shape[0] // JF_PANEL_SS

    def pulse_id(self, i: int) -> int:
        return int(np.asarray(self._grp()["pulse_id"][i]).ravel()[0])

    def is_good_frame(self, i: int) -> bool:
        ds = self._grp().get("is_good_frame")
        if ds is None:
            return True
        return bool(np.asarray(ds[i]).ravel()[0])

    def pixel_mask(self) -> np.ndarray:
        """Detector bad-pixel mask, ``(H, W)`` bool — **True = good pixel**."""
        return np.asarray(self._grp()["meta/pixel_mask"][:], dtype=bool)

    # ── frame access ──────────────────────────────────────────────────────

    def frame(self, i: int) -> torch.Tensor:
        """Read frame ``i`` as a ``(H, W)`` int32 torch tensor (photons)."""
        arr = np.asarray(self._data[i])
        return torch.from_numpy(arr)

    def panels(self, i: int) -> torch.Tensor:
        """Frame ``i`` reshaped to the module stack ``(n_panels, 514, 1030)``."""
        return self.frame(i).reshape(self.n_panels, JF_PANEL_SS, self.detector_shape[1])

    def _gpu_decoder(self, device):
        """Lazily build (and cache) a :class:`Bslz4GpuDecoder` for ``device``.

        Returns ``None`` (cached) if GPU decode is unavailable — the dataset is not
        bslz4 / one-chunk-per-frame, the dtype is not int32, or nvcomp/Triton are
        missing — so callers transparently fall back to the CPU h5py path.
        """
        key = str(device)
        if key in self._gpu_decoders:
            return self._gpu_decoders[key]
        decoder = None
        try:
            from ._bslz4_gpu import Bslz4GpuDecoder, dataset_is_bslz4, parse_bslz4_chunk
            if str(self._data.dtype) == "int32" and dataset_is_bslz4(self._data):
                _m, raw = self._data.id.read_direct_chunk((0, 0, 0))
                _t, block_nbytes, _b = parse_bslz4_chunk(bytes(raw))
                total_ne = self.detector_shape[0] * self.detector_shape[1]
                decoder = Bslz4GpuDecoder(total_ne, block_nbytes // 4, device)
        except Exception:  # pragma: no cover - missing nvcomp/triton/CUDA → CPU fallback
            decoder = None
        self._gpu_decoders[key] = decoder
        return decoder

    def panels_gpu(self, i: int, device) -> torch.Tensor:
        """Frame ``i`` as an int32 ``(n_panels, 514, 1030)`` tensor decoded on ``device``.

        On a CUDA device with a bslz4 dataset this reads the raw compressed chunk and
        decodes it on the GPU (nvcomp LZ4 + Triton inverse-bitshuffle, ~9 ms/frame vs
        ~35 ms for the CPU h5py decode, and only ~6.6 MB crosses PCIe instead of
        ~68 MB). Falls back to ``panels(i).to(device)`` on CPU or when GPU decode is
        unavailable. Cast to float32 at the call site, as with :meth:`panels`.
        """
        import torch
        device = torch.device(device)
        decoder = self._gpu_decoder(device) if device.type == "cuda" else None
        if decoder is None:
            return self.panels(i).to(device)
        _m, raw = self._data.id.read_direct_chunk((i, 0, 0))
        flat = decoder.decode(bytes(raw))  # (total_ne,) int32 on device
        return flat.reshape(self.n_panels, JF_PANEL_SS, self.detector_shape[1])

    def mask_panels(self) -> torch.Tensor:
        """Pixel mask reshaped to ``(n_panels, 514, 1030)`` (True = good)."""
        m = torch.from_numpy(self.pixel_mask())
        return m.reshape(self.n_panels, JF_PANEL_SS, self.detector_shape[1])


def list_run(data_dir: str | Path, pattern: str = "*.JF06T32V07.h5") -> list[Path]:
    """Sorted list of detector files in an acquisition directory."""
    return sorted(Path(data_dir).glob(pattern))
