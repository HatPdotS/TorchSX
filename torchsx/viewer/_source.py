"""Read per-frame results out of a torchsx ``SFXStream`` (the viewer's cache).

The stream is the result cache: it stores, per image, the indexed reflections
(``hkl`` + detector position + intensity), the crystal (cell + reciprocal basis),
and the raw-file identity (``filenames``/``events``) — but never the frame pixels,
which the server reads from the raw ``.h5`` separately.

``StreamResults`` loads a stream once, precomputes an image→observation index for
O(1) per-frame slicing, and exposes plain-Python views the server serializes to
JSON. Column access goes through ``stream._core`` (the columnar facade), since
``SFXStream`` only surfaces some columns as properties.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class StreamResults:
    def __init__(self, stream_path: str | Path, data_root: str | Path | None = None):
        from torchsx.stream import SFXStream

        self.path = str(stream_path)
        # Bases for resolving relative frame paths stored in the stream: an explicit
        # --data-root, the stream file's own directory, then the CWD.
        self._roots = [Path(r) for r in (data_root, Path(self.path).resolve().parent, Path.cwd()) if r]
        self.stream = SFXStream.load(self.path)
        core = self.stream._core

        # obs columns → numpy (once)
        self._hkl = core.hkl.cpu().numpy()                      # (Nobs, 3) int16
        self._fs = core.fs.cpu().numpy().astype(np.float32)     # in-panel fast scan
        self._ss = core.ss.cpu().numpy().astype(np.float32)     # assembled slow scan
        self._panel = core.panel_id.cpu().numpy().astype(np.int64)
        self._I = core.I.cpu().numpy().astype(np.float32)
        self._obs_crystal = core.crystal_id.cpu().numpy().astype(np.int64)
        self._prob = (core.prob.cpu().numpy().astype(np.float32)
                      if "prob" in core.levels["obs"].columns else None)

        # crystal columns
        self._cell = core.cell.cpu().numpy().astype(np.float32)        # (Ncryst, 6)
        self._astar = core.astar.cpu().numpy().astype(np.float32)
        self._bstar = core.bstar.cpu().numpy().astype(np.float32)
        self._cstar = core.cstar.cpu().numpy().astype(np.float32)
        self._resolution = core.resolution.cpu().numpy().astype(np.float32)
        self._n_refl = core.n_refl.cpu().numpy().astype(np.int64)
        self._cryst_image = core.image_id.cpu().numpy().astype(np.int64)

        # image columns
        self._n_peaks = core.n_peaks.cpu().numpy().astype(np.int64)
        self.n_images = int(self.stream.n_images)
        self.filenames = list(self.stream.filenames)
        self.events = list(self.stream.events)

        # peak level (raw found peaks for every hit frame, indexed or not) — only
        # present in streams built with hits; joins to image via peak_image_id.
        self._has_peaks = "peak" in core.levels and core.levels["peak"].n > 0
        if self._has_peaks:
            pk = core.levels["peak"].columns
            self._pk_fs = pk["peak_fs"].cpu().numpy().astype(np.float32)
            self._pk_ss = pk["peak_ss"].cpu().numpy().astype(np.float32)
            self._pk_panel = pk["peak_panel"].cpu().numpy().astype(np.int64)
            self._pk_I = pk["peak_intensity"].cpu().numpy().astype(np.float32)
            self._pk_prob = (pk["peak_prob"].cpu().numpy().astype(np.float32)
                             if "peak_prob" in pk else None)
            self._pk_image = pk["peak_image_id"].cpu().numpy().astype(np.int64)

        self._build_index()

    # ── per-image index ──────────────────────────────────────────────────────
    def _build_index(self):
        """image → crystal rows, and image → obs indices (via crystal_id)."""
        self._crystals_of_image: list[list[int]] = [[] for _ in range(self.n_images)]
        for c, img in enumerate(self._cryst_image):
            if 0 <= img < self.n_images:
                self._crystals_of_image[img].append(c)
        # obs → image, then group obs indices by image
        obs_image = self._cryst_image[self._obs_crystal] if self._obs_crystal.size else np.empty(0, np.int64)
        self._obs_of_image: list[np.ndarray] = [np.empty(0, np.int64)] * self.n_images
        if obs_image.size:
            order = np.argsort(obs_image, kind="stable")
            sorted_img = obs_image[order]
            bounds = np.searchsorted(sorted_img, np.arange(self.n_images + 1))
            for i in range(self.n_images):
                self._obs_of_image[i] = order[bounds[i]:bounds[i + 1]]

        # peak → image grouping (peak_image_id joins straight to image)
        self._peaks_of_image: list[np.ndarray] = [np.empty(0, np.int64)] * self.n_images
        if self._has_peaks and self._pk_image.size:
            order = np.argsort(self._pk_image, kind="stable")
            bounds = np.searchsorted(self._pk_image[order], np.arange(self.n_images + 1))
            for i in range(self.n_images):
                self._peaks_of_image[i] = order[bounds[i]:bounds[i + 1]]

    # ── frame identity ─────────────────────────────────────────────────────────
    def frame_source(self, i: int) -> tuple[str, int]:
        """(h5 path, frame index) for image ``i`` from filenames/events.

        The stream may store a relative path; resolve it against the configured
        roots (--data-root / stream dir / CWD) and return the first that exists.
        """
        fn = self.filenames[i] if i < len(self.filenames) else ""
        ev = self.events[i] if i < len(self.events) else "//0"
        try:
            frame_index = int(str(ev).lstrip("/"))
        except ValueError:
            frame_index = 0
        p = Path(fn)
        if not (p.is_absolute() and p.exists()):
            for root in self._roots:
                cand = root / fn
                if cand.exists():
                    return str(cand), frame_index
        return fn, frame_index

    @property
    def geometry(self):
        return self.stream.geometry

    def detector_mask(self) -> np.ndarray | None:
        """Embedded good-pixel mask as ``(P, nss, nfs)`` uint8 (1=good, 0=masked),
        or None when the stream carries no mask. A per-run constant."""
        m = self.stream.detector_mask
        if m is None:
            return None
        return m.cpu().numpy().astype(bool).astype(np.uint8)

    # ── per-frame views ──────────────────────────────────────────────────────
    def manifest_rows(self) -> list[dict]:
        rows = []
        for i in range(self.n_images):
            crows = self._crystals_of_image[i]
            cell = self._cell[crows[0]].tolist() if crows else None
            rows.append({
                "idx": i,
                "n_peaks": int(self._n_peaks[i]) if i < len(self._n_peaks) else 0,
                "n_found": int(self._peaks_of_image[i].size),
                "n_refl": int(self._obs_of_image[i].size),
                "n_crystals": len(crows),
                "cell_lengths": cell[:3] if cell else None,
                "success": len(crows) > 0,
            })
        return rows

    def reflections(self, i: int) -> list[dict]:
        idx = self._obs_of_image[i]
        out = []
        for j in idx:
            out.append({
                "panel": int(self._panel[j]),
                "ss": float(self._ss[j]),
                "fs": float(self._fs[j]),
                "intensity": float(self._I[j]),
                "prob": float(self._prob[j]) if self._prob is not None else None,
                "hkl": [int(x) for x in self._hkl[j]],
            })
        return out

    def found_peaks(self, i: int) -> list[dict]:
        """Raw peak-finder hits for image ``i`` (the ``peak`` level), indexed or
        not. Empty when the stream carries no peak level."""
        idx = self._peaks_of_image[i] if self._has_peaks else np.empty(0, np.int64)
        out = []
        for j in idx:
            out.append({
                "panel": int(self._pk_panel[j]),
                "ss": float(self._pk_ss[j]),
                "fs": float(self._pk_fs[j]),
                "intensity": float(self._pk_I[j]),
                "prob": float(self._pk_prob[j]) if self._pk_prob is not None else None,
            })
        return out

    def crystals(self, i: int) -> list[dict]:
        out = []
        for c in self._crystals_of_image[i]:
            out.append({
                "cell_lengths": self._cell[c][:3].tolist(),
                "cell_angles": self._cell[c][3:].tolist(),
                "astar": self._astar[c].tolist(),
                "bstar": self._bstar[c].tolist(),
                "cstar": self._cstar[c].tolist(),
                "resolution": float(self._resolution[c]),
                "n_refl": int(self._n_refl[c]),
            })
        return out
