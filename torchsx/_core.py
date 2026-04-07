"""_SFXStreamCore: stacked-tensor dataclass for unmerged SFX data."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from torchsx.geometry import Geometry
from torchsx.io._metadata import StreamMetadata


@dataclass
class _SFXStreamCore:
    """Three-level stacked-tensor representation of unmerged SFX data.

    Levels:
    - Observations (N_obs): individual reflections
    - Crystals (N_crystals): indexed patterns
    - Images (N_images): detector frames

    Join keys: crystal_id (obs→crystal), image_id (crystal→image)
    """

    # === OBSERVATIONS (N_obs,) ===
    hkl: torch.Tensor  # (N_obs, 3) int16
    I: torch.Tensor  # (N_obs,) float32
    sigma: torch.Tensor  # (N_obs,) float32
    background: torch.Tensor  # (N_obs,) float32
    peak: torch.Tensor  # (N_obs,) float32
    fs: torch.Tensor  # (N_obs,) float32
    ss: torch.Tensor  # (N_obs,) float32
    panel_id: torch.Tensor  # (N_obs,) int16
    crystal_id: torch.Tensor  # (N_obs,) int32

    # === CRYSTALS (N_crystals,) ===
    astar: torch.Tensor  # (N_crystals, 3) float32
    bstar: torch.Tensor  # (N_crystals, 3) float32
    cstar: torch.Tensor  # (N_crystals, 3) float32
    cell: torch.Tensor  # (N_crystals, 6) float32 — a, b, c (Å), α, β, γ (deg)
    profile_rad: torch.Tensor  # (N_crystals,) float32
    resolution: torch.Tensor  # (N_crystals,) float32 — reported d_min (Å)
    n_refl: torch.Tensor  # (N_crystals,) int32
    image_id: torch.Tensor  # (N_crystals,) int32
    crystal_in_image: torch.Tensor  # (N_crystals,) int32 — 0,1,2.. for multi-crystal frames

    # === IMAGES (N_images,) ===
    beam_energy: torch.Tensor  # (N_images,) float32 — eV
    det_shift_x: torch.Tensor  # (N_images,) float32 — mm
    det_shift_y: torch.Tensor  # (N_images,) float32 — mm
    n_peaks: torch.Tensor  # (N_images,) int32
    camera_length: torch.Tensor | None = None  # (N_images,) float32 — m (from average_camera_length)

    # === METADATA + GEOMETRY ===
    geometry: Geometry | None = None
    metadata: StreamMetadata = field(default_factory=StreamMetadata)

    # === NON-TENSOR METADATA ===
    spacegroup: str = "P 1"
    source_path: str = ""

    def __post_init__(self):
        """Validate consistency of tensor dimensions."""
        n_obs = self.hkl.shape[0]
        n_crystals = self.cell.shape[0]
        n_images = self.beam_energy.shape[0]

        # Observation tensors
        for name in ("I", "sigma", "background", "peak", "fs", "ss", "panel_id", "crystal_id"):
            t = getattr(self, name)
            assert t.shape[0] == n_obs, f"{name} has {t.shape[0]} rows, expected {n_obs}"

        # Crystal tensors
        for name in ("astar", "bstar", "cstar", "profile_rad", "resolution", "n_refl", "image_id"):
            t = getattr(self, name)
            assert t.shape[0] == n_crystals, f"{name} has {t.shape[0]} rows, expected {n_crystals}"
        assert self.crystal_in_image.shape[0] == n_crystals

        # Synthesize camera_length if not provided
        if self.camera_length is None:
            object.__setattr__(self, "camera_length", torch.zeros(n_images, dtype=torch.float32))

        # Image tensors
        for name in ("det_shift_x", "det_shift_y", "n_peaks", "camera_length"):
            t = getattr(self, name)
            assert t.shape[0] == n_images, f"{name} has {t.shape[0]} rows, expected {n_images}"

        # crystal_id range
        if n_obs > 0:
            assert self.crystal_id.min() >= 0, "crystal_id has negative values"
            assert self.crystal_id.max() < n_crystals, (
                f"crystal_id max {self.crystal_id.max()} >= n_crystals {n_crystals}"
            )

        # image_id range
        if n_crystals > 0:
            assert self.image_id.min() >= 0, "image_id has negative values"
            assert self.image_id.max() < n_images, (
                f"image_id max {self.image_id.max()} >= n_images {n_images}"
            )

    @property
    def n_observations(self) -> int:
        return self.hkl.shape[0]

    @property
    def n_crystals(self) -> int:
        return self.cell.shape[0]

    @property
    def n_images(self) -> int:
        return self.beam_energy.shape[0]

    @property
    def device(self) -> torch.device:
        return self.hkl.device

    def to(self, device) -> _SFXStreamCore:
        """Move all tensors to device."""
        tensor_fields = [
            "hkl", "I", "sigma", "background", "peak", "fs", "ss",
            "panel_id", "crystal_id",
            "astar", "bstar", "cstar", "cell", "profile_rad", "resolution",
            "n_refl", "image_id", "crystal_in_image",
            "beam_energy", "det_shift_x", "det_shift_y", "n_peaks", "camera_length",
        ]
        kwargs = {}
        for fname in tensor_fields:
            kwargs[fname] = getattr(self, fname).to(device)

        kwargs["geometry"] = self.geometry
        kwargs["metadata"] = self.metadata.to(device) if self.metadata else StreamMetadata()
        kwargs["spacegroup"] = self.spacegroup
        kwargs["source_path"] = self.source_path

        return _SFXStreamCore(**kwargs)
