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
        self.path = str(path)
        self.data_path = data_path
        self._group_path = data_path.rsplit("/", 1)[0]  # e.g. /data/JF06T32V07
        self._h5: h5py.File | None = h5py.File(self.path, "r")
        self._data = self._h5[data_path]
        if self._data.ndim != 3:
            raise ValueError(
                f"expected a 3-D detector cube at {data_path!r}, "
                f"got shape {self._data.shape}"
            )

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

    def mask_panels(self) -> torch.Tensor:
        """Pixel mask reshaped to ``(n_panels, 514, 1030)`` (True = good)."""
        m = torch.from_numpy(self.pixel_mask())
        return m.reshape(self.n_panels, JF_PANEL_SS, self.detector_shape[1])


def list_run(data_dir: str | Path, pattern: str = "*.JF06T32V07.h5") -> list[Path]:
    """Sorted list of detector files in an acquisition directory."""
    return sorted(Path(data_dir).glob(pattern))
