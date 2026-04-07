"""ROIBatch and ROIBatchLoader: GPU-ready batches from ROIStore."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from torchsx._core import _SFXStreamCore
from torchsx.io._roi_store import ROIStore
from torchsx.merge._hkl import HKLIndex
from torchsx.forward.coords import RefinableGeometry
from torchsx.forward.corrections import (
    polarization_correction,
    solid_angle_correction,
    estimate_background,
)


@dataclass
class ROIBatch:
    """A GPU-ready batch of ROI patches with pixel coordinate grids.

    Geometric quantities (s_hat_out, polarization, solid_angle) are computed
    on-the-fly by the forward model from the raw pixel grids, so that geometry
    parameter gradients can flow through.
    """

    patches: Tensor          # (B, ph, pw) float32 — observed pixel counts
    valid: Tensor             # (B, ph, pw) bool/uint8 — pixel mask
    crystal_id: Tensor        # (B,) int32
    hkl_id: Tensor            # (B,) int64
    hkl: Tensor               # (B, 3) int16
    panel_id: Tensor          # (B,) int16
    fs_center: Tensor         # (B,) float32 — global pixel center fs
    ss_center: Tensor         # (B,) float32 — global pixel center ss
    background: Tensor        # (B,) float32

    # Global observation indices (for filtering, diagnostics)
    obs_idx: Tensor | None = None     # (B,) int64 — index into core arrays

    # Raw pixel grids (for on-the-fly geometry in forward model)
    # When provided, the forward model computes geometry with gradient flow.
    fs_grid: Tensor | None = None     # (B, ph, pw) float32
    ss_grid: Tensor | None = None     # (B, ph, pw) float32
    pid_grid: Tensor | None = None    # (B, ph, pw) long

    # Precomputed geometry (optional — set by forward model or manually)
    # If s_hat_out is None and fs_grid is provided, the forward model
    # computes these on-the-fly with geometry gradient flow.
    s_hat_out: Tensor | None = None   # (B, ph, pw, 3)
    two_theta: Tensor | None = None   # (B, ph, pw)
    phi: Tensor | None = None         # (B, ph, pw)
    polarization: Tensor | None = None  # (B, ph, pw)
    solid_angle: Tensor | None = None   # (B, ph, pw)

    def to(self, device: torch.device) -> ROIBatch:
        """Move all tensors to device."""
        fields = {}
        for name in self.__dataclass_fields__:
            val = getattr(self, name)
            if isinstance(val, Tensor):
                fields[name] = val.to(device)
            else:
                fields[name] = val
        return ROIBatch(**fields)


def _build_pixel_grids(
    fs_center: Tensor,
    ss_center: Tensor,
    panel_id: Tensor,
    patch_h: int,
    patch_w: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build per-pixel (fs, ss, panel_id) grids for a batch of ROIs.

    Returns
    -------
    fs_grid : (B, ph, pw) float
    ss_grid : (B, ph, pw) float
    pid_grid : (B, ph, pw) long
    """
    B = fs_center.shape[0]
    device = fs_center.device

    # Pixel offsets from center
    half_h = patch_h // 2
    half_w = patch_w // 2
    # Row offsets (slow scan) and col offsets (fast scan)
    ss_off = torch.arange(-half_h, -half_h + patch_h, device=device, dtype=torch.float32)
    fs_off = torch.arange(-half_w, -half_w + patch_w, device=device, dtype=torch.float32)

    # (ph, pw) grids
    ss_grid_2d, fs_grid_2d = torch.meshgrid(ss_off, fs_off, indexing="ij")

    # Broadcast to (B, ph, pw)
    fs_grid = fs_center[:, None, None] + fs_grid_2d[None, :, :]
    ss_grid = ss_center[:, None, None] + ss_grid_2d[None, :, :]
    pid_grid = panel_id[:, None, None].long().expand(B, patch_h, patch_w)

    return fs_grid, ss_grid, pid_grid


def _compute_pixel_geometry(
    geometry: RefinableGeometry,
    fs_grid: Tensor,
    ss_grid: Tensor,
    pid_grid: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute per-pixel geometric quantities.

    Returns
    -------
    s_hat_out : (B, ph, pw, 3)
    two_theta : (B, ph, pw)
    phi : (B, ph, pw)
    pol : (B, ph, pw)
    sol_angle : (B, ph, pw)
    """
    shape = fs_grid.shape  # (B, ph, pw)

    # Flatten for geometry call
    fs_flat = fs_grid.reshape(-1)
    ss_flat = ss_grid.reshape(-1)
    pid_flat = pid_grid.reshape(-1)

    s_hat = geometry.pixel_to_s_hat(fs_flat, ss_flat, pid_flat)  # (N, 3)
    s_hat = s_hat.reshape(*shape, 3)  # (B, ph, pw, 3)

    # 2theta = angle between s_hat and beam (+z)
    cos_2theta = s_hat[..., 2].clamp(-1, 1)
    two_theta = torch.acos(cos_2theta)

    # phi = azimuthal angle in xy plane
    phi = torch.atan2(s_hat[..., 1], s_hat[..., 0])

    pol = polarization_correction(two_theta, phi)
    sol = solid_angle_correction(s_hat)

    return s_hat, two_theta, phi, pol, sol


class ROIBatchLoader:
    """Iterate over ROI batches grouped by crystal.

    Batches over crystals (not individual ROIs) so that per-crystal parameters
    get gradients from all their observations in each batch.

    The loader provides raw pixel grids but does NOT precompute geometry —
    that is done by the forward model so geometry gradients can flow.

    Parameters
    ----------
    roi_store : ROIStore
        Memory-mapped ROI patches.
    core : _SFXStreamCore
        Stream data (for crystal_id, hkl, etc.).
    hkl_index : HKLIndex
        HKL index mapping.
    geometry : RefinableGeometry
        Detector geometry (stored for reference, not used in iteration).
    n_crystals_per_batch : int
        Number of crystals per batch.
    device : torch.device
        Target device for batches.
    obs_mask : Tensor | None
        If the ROIStore was built with an obs_mask, pass it here.
    """

    def __init__(
        self,
        roi_store: ROIStore,
        core: _SFXStreamCore,
        hkl_index: HKLIndex,
        geometry: RefinableGeometry,
        n_crystals_per_batch: int = 500,
        device: torch.device | None = None,
        obs_mask: Tensor | None = None,
    ):
        self.roi_store = roi_store
        self.core = core
        self.hkl_index = hkl_index
        self.geometry = geometry
        self.n_crystals_per_batch = n_crystals_per_batch
        self.device = device or torch.device("cpu")

        # Map ROI store indices to observation indices
        if obs_mask is not None:
            self._obs_indices = obs_mask.nonzero(as_tuple=True)[0]
        else:
            self._obs_indices = torch.arange(core.n_observations)

        # Build crystal → observation mapping
        cid = core.crystal_id[self._obs_indices].long()
        self._crystal_ids_unique = torch.arange(core.n_crystals)
        # For each crystal, store the local (ROI store) indices
        self._crystal_to_local = []
        for c in range(core.n_crystals):
            mask = cid == c
            self._crystal_to_local.append(mask.nonzero(as_tuple=True)[0])

    def __len__(self) -> int:
        """Number of batches per epoch."""
        n = self.core.n_crystals
        return (n + self.n_crystals_per_batch - 1) // self.n_crystals_per_batch

    def __iter__(self):
        """Yield ROIBatch objects, one per crystal group."""
        n_crystals = self.core.n_crystals
        # Shuffle crystal order each epoch
        perm = torch.randperm(n_crystals)

        for start in range(0, n_crystals, self.n_crystals_per_batch):
            end = min(start + self.n_crystals_per_batch, n_crystals)
            crystal_batch = perm[start:end]

            # Gather all local ROI indices for these crystals
            local_indices = []
            for c in crystal_batch:
                local_indices.append(self._crystal_to_local[c.item()])
            if not local_indices:
                continue
            local_idx = torch.cat(local_indices)
            if local_idx.numel() == 0:
                continue

            # Load from ROI store
            patches = self.roi_store["patches"][local_idx].to(self.device)
            valid = self.roi_store["valid"][local_idx].to(self.device)
            ph, pw = patches.shape[1], patches.shape[2]

            # Observation-level data
            obs_idx = self._obs_indices[local_idx]
            crystal_id = self.core.crystal_id[obs_idx].to(self.device)
            hkl = self.core.hkl[obs_idx].to(self.device)
            hkl_id = self.hkl_index.hkl_id[obs_idx].to(self.device)
            panel_id = self.core.panel_id[obs_idx].to(self.device)
            fs_center = self.core.fs[obs_idx].float().to(self.device)
            ss_center = self.core.ss[obs_idx].float().to(self.device)

            # Background from border pixels
            bg = estimate_background(patches, valid)

            # Pixel grids (raw coordinates, geometry computed in forward model)
            fs_grid, ss_grid, pid_grid = _build_pixel_grids(
                fs_center, ss_center, panel_id, ph, pw,
            )

            yield ROIBatch(
                patches=patches,
                valid=valid,
                crystal_id=crystal_id,
                hkl_id=hkl_id,
                hkl=hkl,
                panel_id=panel_id,
                fs_center=fs_center,
                ss_center=ss_center,
                background=bg,
                obs_idx=obs_idx.to(self.device),
                fs_grid=fs_grid,
                ss_grid=ss_grid,
                pid_grid=pid_grid,
            )
