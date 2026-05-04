"""Filtering operations on ``_SFXStreamCore``.  All return new (immutable) cores."""

from __future__ import annotations

from copy import deepcopy

import torch

from torchsx._core import _SFXStreamCore


def _subset_metadata(
    metadata,
    image_mask: torch.Tensor,
    crystal_mask: torch.Tensor,
):
    """Subset per-image / per-crystal entries in ``metadata`` to match the new core.

    Each entry under ``metadata["image"]`` / ``metadata["crystal"]`` is subset only
    when its length equals the original axis size, so axis-agnostic globals
    (e.g. ``metadata["crystal"]["panel_names"]`` — actually per-panel) are left
    untouched. Non-list / non-tensor values pass through unchanged.
    """
    if not metadata:
        return metadata

    new_meta = deepcopy(metadata)
    for axis_key, mask in (("image", image_mask), ("crystal", crystal_mask)):
        sub = new_meta.get(axis_key)
        if not isinstance(sub, dict):
            continue
        n_old = int(mask.shape[0])
        idx = mask.nonzero(as_tuple=True)[0].tolist()
        for k, v in list(sub.items()):
            if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == n_old:
                sub[k] = v[mask]
            elif isinstance(v, list) and len(v) == n_old:
                sub[k] = [v[i] for i in idx]
    return new_meta


def _reindex(
    core: _SFXStreamCore,
    obs_mask: torch.Tensor,
    crystal_mask: torch.Tensor,
    image_mask: torch.Tensor,
) -> _SFXStreamCore:
    """Apply boolean masks to all three levels and rebuild contiguous IDs."""
    crystal_remap = torch.full((core.n_crystals,), -1, dtype=torch.int32)
    kept_crystal_ids = crystal_mask.nonzero(as_tuple=True)[0]
    crystal_remap[kept_crystal_ids] = torch.arange(
        len(kept_crystal_ids), dtype=torch.int32
    )

    image_remap = torch.full((core.n_images,), -1, dtype=torch.int32)
    kept_image_ids = image_mask.nonzero(as_tuple=True)[0]
    image_remap[kept_image_ids] = torch.arange(
        len(kept_image_ids), dtype=torch.int32
    )

    new_crystal_id = crystal_remap[core.crystal_id[obs_mask].long()]
    new_image_id = image_remap[core.image_id[crystal_mask].long()]

    return _SFXStreamCore(
        hkl=core.hkl[obs_mask],
        I=core.I[obs_mask],
        sigma=core.sigma[obs_mask],
        background=core.background[obs_mask],
        peak=core.peak[obs_mask],
        fs=core.fs[obs_mask],
        ss=core.ss[obs_mask],
        panel_id=core.panel_id[obs_mask],
        crystal_id=new_crystal_id,
        astar=core.astar[crystal_mask],
        bstar=core.bstar[crystal_mask],
        cstar=core.cstar[crystal_mask],
        cell=core.cell[crystal_mask],
        profile_rad=core.profile_rad[crystal_mask],
        resolution=core.resolution[crystal_mask],
        n_refl=core.n_refl[crystal_mask],
        image_id=new_image_id,
        crystal_in_image=core.crystal_in_image[crystal_mask],
        beam_energy=core.beam_energy[image_mask],
        det_shift_x=core.det_shift_x[image_mask],
        det_shift_y=core.det_shift_y[image_mask],
        n_peaks=core.n_peaks[image_mask],
        geometry=core.geometry,
        metadata=_subset_metadata(core.metadata, image_mask, crystal_mask),
        spacegroup=core.spacegroup,
        source_path=core.source_path,
    )


def filter_resolution(
    core: _SFXStreamCore, d_min: float = 0.0, d_max: float = 999.0
) -> _SFXStreamCore:
    """Keep observations within the resolution range [d_min, d_max] in Å."""
    from torchref.base.reciprocal.hkl import get_d_spacing

    obs_cell = core.cell[core.crystal_id.long()]
    d = get_d_spacing(core.hkl.float(), obs_cell)

    obs_mask = (d >= d_min) & (d <= d_max)

    kept_crystal_ids = core.crystal_id[obs_mask].unique()
    crystal_mask = torch.zeros(core.n_crystals, dtype=torch.bool)
    crystal_mask[kept_crystal_ids.long()] = True

    kept_image_ids = core.image_id[crystal_mask].unique()
    image_mask = torch.zeros(core.n_images, dtype=torch.bool)
    image_mask[kept_image_ids.long()] = True

    return _reindex(core, obs_mask, crystal_mask, image_mask)


def filter_crystals(core: _SFXStreamCore, mask: torch.Tensor) -> _SFXStreamCore:
    """Keep crystals where mask is True.  Boolean (N_crystals,) tensor."""
    assert mask.shape[0] == core.n_crystals

    obs_mask = mask[core.crystal_id.long()]

    kept_image_ids = core.image_id[mask].unique()
    image_mask = torch.zeros(core.n_images, dtype=torch.bool)
    image_mask[kept_image_ids.long()] = True

    return _reindex(core, obs_mask, mask, image_mask)


def filter_images(core: _SFXStreamCore, mask: torch.Tensor) -> _SFXStreamCore:
    """Keep images where mask is True.  Boolean (N_images,) tensor."""
    assert mask.shape[0] == core.n_images

    crystal_mask = mask[core.image_id.long()]
    obs_mask = crystal_mask[core.crystal_id.long()]

    return _reindex(core, obs_mask, crystal_mask, mask)


def filter_outliers(core: _SFXStreamCore, n_sigma: float = 4.0) -> _SFXStreamCore:
    """Remove per-HKL outlier observations beyond n_sigma."""
    hkl = core.hkl.long()
    h_off = hkl[:, 0] - hkl[:, 0].min()
    k_off = hkl[:, 1] - hkl[:, 1].min()
    l_off = hkl[:, 2] - hkl[:, 2].min()
    k_range = k_off.max() + 1
    l_range = l_off.max() + 1
    hkl_linear = h_off * k_range * l_range + k_off * l_range + l_off

    unique_ids, inverse = hkl_linear.unique(return_inverse=True)
    n_groups = unique_ids.shape[0]

    group_sum = torch.zeros(n_groups, dtype=torch.float64)
    group_sum.scatter_add_(0, inverse, core.I.double())
    group_count = torch.zeros(n_groups, dtype=torch.float64)
    group_count.scatter_add_(0, inverse, torch.ones_like(core.I, dtype=torch.float64))
    group_mean = group_sum / group_count.clamp(min=1)

    diff = core.I.double() - group_mean[inverse]
    group_var = torch.zeros(n_groups, dtype=torch.float64)
    group_var.scatter_add_(0, inverse, diff ** 2)
    group_std = (group_var / group_count.clamp(min=1)).sqrt()

    obs_mask = (diff.abs() <= n_sigma * group_std[inverse]) | (group_count[inverse] <= 2)

    kept_crystal_ids = core.crystal_id[obs_mask].unique()
    crystal_mask = torch.zeros(core.n_crystals, dtype=torch.bool)
    crystal_mask[kept_crystal_ids.long()] = True

    kept_image_ids = core.image_id[crystal_mask].unique()
    image_mask = torch.zeros(core.n_images, dtype=torch.bool)
    image_mask[kept_image_ids.long()] = True

    return _reindex(core, obs_mask, crystal_mask, image_mask)


def filter_cell(core: _SFXStreamCore, tolerance: float = 0.1) -> _SFXStreamCore:
    """Remove crystals whose cell parameters deviate by > tolerance fractionally."""
    median_cell = core.cell.median(dim=0).values
    frac_dev = ((core.cell - median_cell) / median_cell).abs()
    crystal_mask = (frac_dev <= tolerance).all(dim=1)

    obs_mask = crystal_mask[core.crystal_id.long()]

    kept_image_ids = core.image_id[crystal_mask].unique()
    image_mask = torch.zeros(core.n_images, dtype=torch.bool)
    image_mask[kept_image_ids.long()] = True

    return _reindex(core, obs_mask, crystal_mask, image_mask)
