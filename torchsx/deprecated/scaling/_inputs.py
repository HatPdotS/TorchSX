"""ScalerInputs: precomputed observation-level data for scaling."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchsx._core import _SFXStreamCore
from torchsx.deprecated.merge._hkl import compute_hkl_index
from torchref.base.reciprocal.hkl import get_d_spacing
from torchref.base.alignment.normalization import compute_radial_shells, assign_to_shells


@dataclass
class ScalerInputs:
    """Precomputed observation-level indices for scaling.

    Not an nn.Module — pure data extracted from _SFXStreamCore.
    """

    hkl_id: torch.Tensor       # (N_obs,) int64 — unique reflection index
    crystal_id: torch.Tensor   # (N_obs,) int32
    I_obs: torch.Tensor        # (N_obs,) float32
    s_sq: torch.Tensor         # (N_obs,) float32 — (1/(2d))²
    shell: torch.Tensor        # (N_obs,) int64 — resolution shell
    n_unique: int
    n_crystals: int
    n_shells: int
    unique_hkl: torch.Tensor   # (N_unique, 3) int16
    I_bg: torch.Tensor | None = None  # (N_obs,) float32 — background per observation
    weight: torch.Tensor | None = None  # (N_obs,) float32 — per-obs EM weight
    hkl_obs: torch.Tensor | None = None  # (N_obs, 3) int16 — Miller indices per obs
    UB: torch.Tensor | None = None  # (N_crystals, 3, 3) float32 — orientation matrices
    wavelength: float | None = None  # beam wavelength in Å

    def to(self, device: torch.device) -> ScalerInputs:
        """Move all tensors to device."""
        return ScalerInputs(
            hkl_id=self.hkl_id.to(device),
            crystal_id=self.crystal_id.to(device),
            I_obs=self.I_obs.to(device),
            s_sq=self.s_sq.to(device),
            shell=self.shell.to(device),
            n_unique=self.n_unique,
            n_crystals=self.n_crystals,
            n_shells=self.n_shells,
            unique_hkl=self.unique_hkl.to(device),
            I_bg=self.I_bg.to(device) if self.I_bg is not None else None,
            weight=self.weight.to(device) if self.weight is not None else None,
            hkl_obs=self.hkl_obs.to(device) if self.hkl_obs is not None else None,
            UB=self.UB.to(device) if self.UB is not None else None,
            wavelength=self.wavelength,
        )


def prepare_scaler_inputs(core: _SFXStreamCore, n_shells: int = 20) -> ScalerInputs:
    """Precompute observation-level indices from a stream core.

    Parameters
    ----------
    core : _SFXStreamCore
        Unmerged SFX data.
    n_shells : int
        Number of resolution shells for the loss function.

    Returns
    -------
    ScalerInputs
    """
    # HKL indexing
    hkl_idx = compute_hkl_index(core.hkl)

    # d-spacing via median cell (avoids torchref batching issue)
    median_cell = core.cell.median(dim=0).values
    d = get_d_spacing(core.hkl.float(), median_cell)

    # s² = (1 / (2d))²
    s_sq = (1.0 / (2.0 * d.clamp(min=1e-6))) ** 2

    # Resolution shells — uniform in 1/d
    d_min = d.min().item()
    d_max = d.max().item()
    # Small margin to ensure all reflections fall within shells
    shell_edges, _ = compute_radial_shells(
        d_min * 0.99, d_max * 1.01, n_shells, device=core.hkl.device,
    )
    s_mag = 1.0 / (2.0 * d.clamp(min=1e-6))
    shell = assign_to_shells(s_mag, shell_edges)
    # Clamp out-of-range (-1) to nearest shell
    shell = shell.clamp(min=0, max=n_shells - 1).long()

    # Orientation matrices: columns = reciprocal basis vectors
    UB = torch.stack([core.astar, core.bstar, core.cstar], dim=-1)  # (N_crystals, 3, 3)

    # Wavelength from median beam energy (eV → Å)
    wavelength = 12398.4 / core.beam_energy.median().item()

    return ScalerInputs(
        hkl_id=hkl_idx.hkl_id,
        crystal_id=core.crystal_id,
        I_obs=core.I,
        s_sq=s_sq,
        shell=shell,
        n_unique=hkl_idx.n_unique,
        n_crystals=core.n_crystals,
        n_shells=n_shells,
        unique_hkl=hkl_idx.unique_hkl,
        I_bg=core.background,
        hkl_obs=core.hkl,
        UB=UB,
        wavelength=wavelength,
    )
