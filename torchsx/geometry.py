"""Detector geometry from CrystFEL .stream or .geom files.

Single source of truth for all geometry-related operations:
panel definitions, coordinate transforms, radial maps, panel maps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import torch


@dataclass
class Panel:
    """Single detector panel definition."""

    name: str
    min_fs: int
    max_fs: int
    min_ss: int
    max_ss: int
    corner_x: float  # pixels
    corner_y: float  # pixels
    fs_vec: tuple[float, float, float]  # fast-scan direction
    ss_vec: tuple[float, float, float]  # slow-scan direction
    coffset: float  # camera length offset (m)

    @property
    def n_fs(self) -> int:
        return self.max_fs - self.min_fs + 1

    @property
    def n_ss(self) -> int:
        return self.max_ss - self.min_ss + 1

    def corners_mm(self, res: float) -> np.ndarray:
        """Compute 4 corners of this panel in physical (x, y) mm.

        Parameters
        ----------
        res : pixels per meter

        Returns
        -------
        corners : (4, 2) ndarray of (x_mm, y_mm), ordered for a closed polygon.
        """
        cx, cy = self.corner_x, self.corner_y
        fsx, fsy, _ = self.fs_vec
        ssx, ssy, _ = self.ss_vec

        corners_fs = np.array([0, self.n_fs, self.n_fs, 0], dtype=np.float64)
        corners_ss = np.array([0, 0, self.n_ss, self.n_ss], dtype=np.float64)

        x_px = cx + corners_fs * fsx + corners_ss * ssx
        y_px = cy + corners_fs * fsy + corners_ss * ssy

        x_mm = x_px / res * 1000
        y_mm = y_px / res * 1000

        return np.column_stack([x_mm, y_mm])


@dataclass
class Geometry:
    """CrystFEL detector geometry.

    Parses the geometry section from a .stream file header.
    Stores panel definitions and global parameters.
    """

    panels: list[Panel]
    photon_energy_eV: float = 0.0
    clen: float = 0.0  # camera length in meters
    res: float = 0.0  # pixels per meter
    adu_per_eV: float = 1.0
    data_path: str = ""
    mask_file: str | None = None
    mask_data: str | None = None

    @classmethod
    def from_stream(cls, path: str | Path) -> Geometry:
        """Parse geometry from first geometry section in a .stream file."""
        geom_lines = []
        in_geom = False
        with open(path, "r") as f:
            for line in f:
                if line.startswith("----- Begin geometry file -----"):
                    in_geom = True
                    continue
                if line.startswith("----- End geometry file -----"):
                    break
                if in_geom:
                    geom_lines.append(line)
        return cls._parse_geom_lines(geom_lines)

    @classmethod
    def from_geom_file(cls, path: str | Path) -> Geometry:
        """Parse a standalone .geom file."""
        with open(path, "r") as f:
            geom_lines = f.readlines()
        return cls._parse_geom_lines(geom_lines)

    @classmethod
    def _parse_geom_lines(cls, lines: list[str]) -> Geometry:
        """Parse geometry from a list of text lines."""
        global_params: dict = {}
        panel_data: dict[str, dict] = {}

        # Regex for vector components — z is optional (defaults to 0 for flat detectors)
        # Matches: "0.002402x -0.999376y +0.000000z" or "+0.000647x -0.999379y"
        _comp = r"([+-]?\d+\.?\d*(?:e[+-]?\d+)?)"
        vec_pattern = re.compile(
            _comp + r"x\s+" + _comp + r"y(?:\s+" + _comp + r"z)?"
        )

        for line in lines:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            if "=" not in line:
                continue

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()

            if "/" in key:
                # Panel property: "m0/min_fs = 0"
                panel_name, prop = key.split("/", 1)
                if panel_name not in panel_data:
                    panel_data[panel_name] = {"name": panel_name}

                if prop in ("fs", "ss"):
                    m = vec_pattern.search(value)
                    if m:
                        panel_data[panel_name][f"{prop}_vec"] = (
                            float(m.group(1)),
                            float(m.group(2)),
                            float(m.group(3)) if m.group(3) is not None else 0.0,
                        )
                elif prop in ("min_fs", "max_fs", "min_ss", "max_ss"):
                    panel_data[panel_name][prop] = int(float(value))
                elif prop in ("corner_x", "corner_y", "coffset"):
                    panel_data[panel_name][prop] = float(value)
            else:
                # Global property — values may be HDF5 paths (e.g.
                # "/LCLS/photon_energy_eV"), in which case we skip and let the
                # per-chunk value override.
                def _maybe_float(v):
                    try:
                        return float(v.split()[0])
                    except ValueError:
                        return None

                if key == "photon_energy":
                    f = _maybe_float(value)
                    if f is not None:
                        global_params["photon_energy_eV"] = f
                elif key == "clen":
                    f = _maybe_float(value)
                    if f is not None:
                        global_params["clen"] = f
                elif key == "res":
                    f = _maybe_float(value)
                    if f is not None:
                        global_params["res"] = f
                elif key == "adu_per_eV":
                    f = _maybe_float(value)
                    if f is not None:
                        global_params["adu_per_eV"] = f
                elif key == "data":
                    global_params["data_path"] = value
                elif key.startswith("mask") and key.endswith("_file"):
                    global_params["mask_file"] = value
                elif key.startswith("mask") and key.endswith("_data"):
                    global_params["mask_data"] = value

        # Build panels sorted by min_ss (natural panel order)
        panels = []
        for pname in sorted(
            panel_data.keys(),
            key=lambda n: panel_data[n].get("min_ss", 0),
        ):
            pd_entry = panel_data[pname]
            panels.append(
                Panel(
                    name=pd_entry["name"],
                    min_fs=pd_entry.get("min_fs", 0),
                    max_fs=pd_entry.get("max_fs", 0),
                    min_ss=pd_entry.get("min_ss", 0),
                    max_ss=pd_entry.get("max_ss", 0),
                    corner_x=pd_entry.get("corner_x", 0.0),
                    corner_y=pd_entry.get("corner_y", 0.0),
                    fs_vec=pd_entry.get("fs_vec", (1.0, 0.0, 0.0)),
                    ss_vec=pd_entry.get("ss_vec", (0.0, 1.0, 0.0)),
                    coffset=pd_entry.get("coffset", 0.0),
                )
            )

        return cls(
            panels=panels,
            photon_energy_eV=global_params.get("photon_energy_eV", 0.0),
            clen=global_params.get("clen", 0.0),
            res=global_params.get("res", 0.0),
            adu_per_eV=global_params.get("adu_per_eV", 1.0),
            data_path=global_params.get("data_path", ""),
            mask_file=global_params.get("mask_file"),
            mask_data=global_params.get("mask_data"),
        )

    # ── Basic properties ─────────────────────────────────────────────────

    @property
    def n_panels(self) -> int:
        return len(self.panels)

    @property
    def panel_names(self) -> list[str]:
        return [p.name for p in self.panels]

    @property
    def panel_name_to_id(self) -> dict[str, int]:
        return {p.name: i for i, p in enumerate(self.panels)}

    # ── Detector layout properties ───────────────────────────────────────

    @cached_property
    def detector_shape(self) -> tuple[int, int]:
        """Total detector shape (total_ss, total_fs)."""
        total_ss = max(p.max_ss for p in self.panels) + 1
        total_fs = max(p.max_fs for p in self.panels) + 1
        return (total_ss, total_fs)

    @cached_property
    def panel_slices(self) -> list[tuple[int, int, int, int]]:
        """Panel bounding boxes as (ss0, ss1, fs0, fs1) with exclusive end.

        Compatible with array slicing: arr[ss0:ss1, fs0:fs1].
        """
        return [
            (p.min_ss, p.max_ss + 1, p.min_fs, p.max_fs + 1)
            for p in self.panels
        ]

    # ── Precomputed panel tensors (cached, CPU) ────────────────────────

    @cached_property
    def _panel_params(self) -> dict[str, torch.Tensor]:
        """Panel parameter lookup tensors, built once on CPU."""
        n = self.n_panels
        corner_x = torch.zeros(n)
        corner_y = torch.zeros(n)
        fs_vecs = torch.zeros(n, 3)
        ss_vecs = torch.zeros(n, 3)
        min_fs = torch.zeros(n)
        min_ss = torch.zeros(n)
        coffset = torch.zeros(n)
        for i, p in enumerate(self.panels):
            corner_x[i] = p.corner_x
            corner_y[i] = p.corner_y
            fs_vecs[i] = torch.tensor(p.fs_vec)
            ss_vecs[i] = torch.tensor(p.ss_vec)
            min_fs[i] = p.min_fs
            min_ss[i] = p.min_ss
            coffset[i] = p.coffset
        return dict(
            corner_x=corner_x, corner_y=corner_y,
            fs_vecs=fs_vecs, ss_vecs=ss_vecs,
            min_fs=min_fs, min_ss=min_ss,
            coffset=coffset,
        )

    @cached_property
    def _panel_id_map(self) -> torch.Tensor:
        """2-D panel-id lookup (total_ss, total_fs) int16, CPU.

        -1 for pixels in inter-panel gaps.
        """
        total_ss, total_fs = self.detector_shape
        pid_map = torch.full((total_ss, total_fs), -1, dtype=torch.int16)
        for i, p in enumerate(self.panels):
            pid_map[p.min_ss : p.max_ss + 1, p.min_fs : p.max_fs + 1] = i
        return pid_map

    # ── Coordinate transforms ────────────────────────────────────────────

    def to_lab(
        self,
        ss: torch.Tensor,
        fs: torch.Tensor,
        panel_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convert detector (ss, fs) to lab-frame positions in meters.

        This is the primary coordinate-transform entry point.  Positions
        may span arbitrary panels — panel membership is resolved
        automatically from the 2-D panel-id map when *panel_id* is not
        supplied.

        Parameters
        ----------
        ss, fs : (N,) float tensors — detector pixel coordinates
            (fractional values are fine; panel lookup uses floor).
        panel_id : (N,) int tensor, optional
            If already known, pass it to skip the lookup.

        Returns
        -------
        xyz : (N, 3) float tensor — lab-frame position in meters
        """
        device = ss.device
        n = ss.shape[0]

        # ── resolve panel id ──
        if panel_id is None:
            pid_map = self._panel_id_map.to(device)
            ss_idx = ss.long().clamp(0, self.detector_shape[0] - 1)
            fs_idx = fs.long().clamp(0, self.detector_shape[1] - 1)
            pid = pid_map[ss_idx, fs_idx].long()
        else:
            pid = panel_id.long()

        # ── panel parameter tensors on the right device ──
        pp = self._panel_params
        corner_x = pp["corner_x"].to(device)
        corner_y = pp["corner_y"].to(device)
        fs_vecs = pp["fs_vecs"].to(device)
        ss_vecs = pp["ss_vecs"].to(device)
        min_fs_t = pp["min_fs"].to(device)
        min_ss_t = pp["min_ss"].to(device)
        coffset_t = pp["coffset"].to(device)

        # ── vectorised transform ──
        dfs = fs - min_fs_t[pid]
        dss = ss - min_ss_t[pid]

        xyz = torch.zeros(n, 3, device=device, dtype=ss.dtype)
        xyz[:, 0] = corner_x[pid] + dfs * fs_vecs[pid, 0] + dss * ss_vecs[pid, 0]
        xyz[:, 1] = corner_y[pid] + dfs * fs_vecs[pid, 1] + dss * ss_vecs[pid, 1]
        xyz[:, 2] = dfs * fs_vecs[pid, 2] + dss * ss_vecs[pid, 2]

        if self.res > 0:
            xyz[:, :2] /= self.res

        xyz[:, 2] += self.clen + coffset_t[pid]
        return xyz

    def pixel_to_lab(
        self,
        fs: torch.Tensor,
        ss: torch.Tensor,
        panel_id: torch.Tensor,
    ) -> torch.Tensor:
        """Convert pixel coordinates to lab-frame positions (meters).

        Legacy interface — delegates to :meth:`to_lab`.  Note the
        argument order is (fs, ss, panel_id) for backward compatibility.

        Parameters
        ----------
        fs, ss : (N,) float tensors — pixel coordinates
        panel_id : (N,) int tensor — panel index

        Returns
        -------
        xyz : (N, 3) float tensor — lab-frame position in meters
        """
        return self.to_lab(ss, fs, panel_id=panel_id)

    def pixel_to_mm(
        self, ss: np.ndarray, fs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert (ss, fs) detector coordinates to physical (x, y) mm.

        Thin numpy wrapper around :meth:`to_lab`.

        Parameters
        ----------
        ss, fs : (N,) int or float arrays — detector pixel coordinates

        Returns
        -------
        x_mm, y_mm : (N,) float64 arrays
        """
        ss_t = torch.from_numpy(np.asarray(ss, dtype=np.float64))
        fs_t = torch.from_numpy(np.asarray(fs, dtype=np.float64))
        xyz = self.to_lab(ss_t, fs_t)
        x_mm = xyz[:, 0].numpy() * 1000
        y_mm = xyz[:, 1].numpy() * 1000
        return x_mm, y_mm

    # ── Radial distance map ──────────────────────────────────────────────

    def radial_map(self) -> np.ndarray:
        """Build (total_ss, total_fs) radial distance map in pixel units."""
        total_ss, total_fs = self.detector_shape
        r_map = np.zeros((total_ss, total_fs), dtype=np.float32)

        for p in self.panels:
            ss0, ss1 = p.min_ss, p.max_ss
            n_ss = ss1 - ss0 + 1
            n_fs = p.max_fs - p.min_fs + 1
            cx, cy = p.corner_x, p.corner_y
            fsx, fsy, _ = p.fs_vec
            ssx, ssy, _ = p.ss_vec
            dfs = np.arange(n_fs, dtype=np.float32)[None, :]
            dss = np.arange(n_ss, dtype=np.float32)[:, None]
            x = cx + dfs * fsx + dss * ssx
            y = cy + dfs * fsy + dss * ssy
            r_map[ss0 : ss1 + 1, p.min_fs : p.max_fs + 1] = np.sqrt(x**2 + y**2)

        return r_map

    # ── Panel map ────────────────────────────────────────────────────────

    def build_panel_map(self) -> torch.Tensor:
        """Build ss → panel_id lookup.

        Returns
        -------
        panel_map : (total_ss,) int16 — panel_id for each slow-scan row.
            -1 for rows outside any panel.
        """
        total_ss = self.detector_shape[0]
        panel_map = torch.full((total_ss,), -1, dtype=torch.int16)
        for i, p in enumerate(self.panels):
            ss_start = p.min_ss
            ss_end = p.min_ss + p.n_ss
            panel_map[ss_start:ss_end] = i
        return panel_map

    # ── Corner utilities ─────────────────────────────────────────────────

    def all_corners_mm(self) -> np.ndarray:
        """All panel corners stacked for computing axis limits.

        Returns
        -------
        corners : (n_panels * 4, 2) ndarray of (x_mm, y_mm)
        """
        return np.vstack([p.corners_mm(self.res) for p in self.panels])

    # ── Serialization ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        return {
            "photon_energy_eV": self.photon_energy_eV,
            "clen": self.clen,
            "res": self.res,
            "adu_per_eV": self.adu_per_eV,
            "data_path": self.data_path,
            "mask_file": self.mask_file,
            "mask_data": self.mask_data,
            "panels": [
                {
                    "name": p.name,
                    "min_fs": p.min_fs,
                    "max_fs": p.max_fs,
                    "min_ss": p.min_ss,
                    "max_ss": p.max_ss,
                    "corner_x": p.corner_x,
                    "corner_y": p.corner_y,
                    "fs_vec": list(p.fs_vec),
                    "ss_vec": list(p.ss_vec),
                    "coffset": p.coffset,
                }
                for p in self.panels
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Geometry:
        """Reconstruct from dict."""
        panels = []
        for pd_entry in d["panels"]:
            panels.append(
                Panel(
                    name=pd_entry["name"],
                    min_fs=pd_entry["min_fs"],
                    max_fs=pd_entry["max_fs"],
                    min_ss=pd_entry["min_ss"],
                    max_ss=pd_entry["max_ss"],
                    corner_x=pd_entry["corner_x"],
                    corner_y=pd_entry["corner_y"],
                    fs_vec=tuple(pd_entry["fs_vec"]),
                    ss_vec=tuple(pd_entry["ss_vec"]),
                    coffset=pd_entry["coffset"],
                )
            )
        return cls(
            panels=panels,
            photon_energy_eV=d.get("photon_energy_eV", 0.0),
            clen=d.get("clen", 0.0),
            res=d.get("res", 0.0),
            adu_per_eV=d.get("adu_per_eV", 1.0),
            data_path=d.get("data_path", ""),
            mask_file=d.get("mask_file"),
            mask_data=d.get("mask_data"),
        )
