"""Resolution-binned completeness and multiplicity of a stream."""

from __future__ import annotations

import torch

from torchsx._core import _SFXStreamCore


def completeness(core: _SFXStreamCore, n_bins: int = 20) -> torch.Tensor:
    """Resolution-binned completeness (fraction of unique HKLs observed)."""
    from torchref.base.reciprocal.hkl import get_d_spacing, generate_possible_hkl
    from torchref.base.alignment.normalization import (
        compute_radial_shells,
        assign_to_shells,
    )

    median_cell = core.cell.median(dim=0).values
    d_min_val = core.resolution.min().item()
    d_max_val = core.resolution.max().item()

    possible_hkl = generate_possible_hkl(median_cell, d_min_val)
    d_possible = get_d_spacing(possible_hkl.float(), median_cell.unsqueeze(0))

    shell_edges, _ = compute_radial_shells(
        d_min_val, d_max_val, n_bins, device=core.hkl.device
    )

    obs_hkl_unique = core.hkl.unique(dim=0)
    obs_cell = median_cell.unsqueeze(0).expand(obs_hkl_unique.shape[0], -1)
    d_obs = get_d_spacing(obs_hkl_unique.float(), obs_cell)

    s_possible = 1.0 / d_possible
    s_obs = 1.0 / d_obs

    bins_possible = assign_to_shells(s_possible, shell_edges)
    bins_obs = assign_to_shells(s_obs, shell_edges)

    comp = torch.zeros(n_bins)
    for i in range(n_bins):
        n_poss = (bins_possible == i).sum().item()
        n_obs = (bins_obs == i).sum().item()
        comp[i] = n_obs / max(n_poss, 1)

    return comp


def multiplicity(core: _SFXStreamCore, n_bins: int = 20) -> torch.Tensor:
    """Resolution-binned mean multiplicity."""
    from torchref.base.reciprocal.hkl import get_d_spacing
    from torchref.base.alignment.normalization import (
        compute_radial_shells,
        assign_to_shells,
    )

    median_cell = core.cell.median(dim=0).values
    d_min_val = core.resolution.min().item()
    d_max_val = core.resolution.max().item()

    obs_cell = core.cell[core.crystal_id.long()]
    d_obs = get_d_spacing(core.hkl.float(), obs_cell)

    shell_edges, _ = compute_radial_shells(
        d_min_val, d_max_val, n_bins, device=core.hkl.device
    )
    s_obs = 1.0 / d_obs
    bins = assign_to_shells(s_obs, shell_edges)

    mult = torch.zeros(n_bins)
    for i in range(n_bins):
        mask = bins == i
        if mask.sum() == 0:
            continue
        n_total = mask.sum().item()
        n_unique = core.hkl[mask].unique(dim=0).shape[0]
        mult[i] = n_total / max(n_unique, 1)

    return mult
