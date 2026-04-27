"""Filtering operations for SFX data. All return new _SFXStreamCore (immutable)."""

from __future__ import annotations

import torch

from torchsx._core import _SFXStreamCore


def _reindex(
    core: _SFXStreamCore,
    obs_mask: torch.Tensor,
    crystal_mask: torch.Tensor,
    image_mask: torch.Tensor,
) -> _SFXStreamCore:
    """Apply boolean masks to all three levels and rebuild contiguous IDs.

    Parameters
    ----------
    obs_mask : (N_obs,) bool — which observations to keep
    crystal_mask : (N_crystals,) bool — which crystals to keep
    image_mask : (N_images,) bool — which images to keep
    """
    # Build remapping tables
    # crystal: old_id → new_id (-1 if removed)
    crystal_remap = torch.full((core.n_crystals,), -1, dtype=torch.int32)
    kept_crystal_ids = crystal_mask.nonzero(as_tuple=True)[0]
    crystal_remap[kept_crystal_ids] = torch.arange(
        len(kept_crystal_ids), dtype=torch.int32
    )

    # image: old_id → new_id
    image_remap = torch.full((core.n_images,), -1, dtype=torch.int32)
    kept_image_ids = image_mask.nonzero(as_tuple=True)[0]
    image_remap[kept_image_ids] = torch.arange(
        len(kept_image_ids), dtype=torch.int32
    )

    # Filter observations
    new_crystal_id = crystal_remap[core.crystal_id[obs_mask].long()]

    # Filter and remap crystal-level
    new_image_id = image_remap[core.image_id[crystal_mask].long()]

    return _SFXStreamCore(
        # Observations
        hkl=core.hkl[obs_mask],
        I=core.I[obs_mask],
        sigma=core.sigma[obs_mask],
        background=core.background[obs_mask],
        peak=core.peak[obs_mask],
        fs=core.fs[obs_mask],
        ss=core.ss[obs_mask],
        panel_id=core.panel_id[obs_mask],
        crystal_id=new_crystal_id,
        # Crystals
        astar=core.astar[crystal_mask],
        bstar=core.bstar[crystal_mask],
        cstar=core.cstar[crystal_mask],
        cell=core.cell[crystal_mask],
        profile_rad=core.profile_rad[crystal_mask],
        resolution=core.resolution[crystal_mask],
        n_refl=core.n_refl[crystal_mask],
        image_id=new_image_id,
        crystal_in_image=core.crystal_in_image[crystal_mask],
        # Images
        beam_energy=core.beam_energy[image_mask],
        det_shift_x=core.det_shift_x[image_mask],
        det_shift_y=core.det_shift_y[image_mask],
        n_peaks=core.n_peaks[image_mask],
        # Metadata
        geometry=core.geometry,
        metadata=core.metadata,  # TODO: filter metadata lists
        spacegroup=core.spacegroup,
        source_path=core.source_path,
    )


def filter_resolution(
    core: _SFXStreamCore, d_min: float = 0.0, d_max: float = 999.0
) -> _SFXStreamCore:
    """Keep only observations within resolution range [d_min, d_max] in Å.

    Uses torchref.base.reciprocal.hkl.get_d_spacing() to compute d-spacing
    from per-crystal cell parameters and Miller indices.
    """
    from torchref.base.reciprocal.hkl import get_d_spacing

    # Compute d-spacing for each observation using its crystal's cell
    # get_d_spacing expects hkl (N, 3) and cell (N, 6) or broadcast
    obs_cell = core.cell[core.crystal_id.long()]  # (N_obs, 6)
    d = get_d_spacing(core.hkl.float(), obs_cell)  # (N_obs,)

    obs_mask = (d >= d_min) & (d <= d_max)

    # Determine which crystals still have observations
    kept_crystal_ids = core.crystal_id[obs_mask].unique()
    crystal_mask = torch.zeros(core.n_crystals, dtype=torch.bool)
    crystal_mask[kept_crystal_ids.long()] = True

    # Determine which images still have crystals
    kept_image_ids = core.image_id[crystal_mask].unique()
    image_mask = torch.zeros(core.n_images, dtype=torch.bool)
    image_mask[kept_image_ids.long()] = True

    return _reindex(core, obs_mask, crystal_mask, image_mask)


def filter_crystals(
    core: _SFXStreamCore, mask: torch.Tensor
) -> _SFXStreamCore:
    """Keep crystals where mask is True. Boolean (N_crystals,) tensor."""
    assert mask.shape[0] == core.n_crystals

    # Observations belonging to kept crystals
    obs_mask = mask[core.crystal_id.long()]

    # Images that still have crystals
    kept_image_ids = core.image_id[mask].unique()
    image_mask = torch.zeros(core.n_images, dtype=torch.bool)
    image_mask[kept_image_ids.long()] = True

    return _reindex(core, obs_mask, mask, image_mask)


def filter_images(
    core: _SFXStreamCore, mask: torch.Tensor
) -> _SFXStreamCore:
    """Keep images where mask is True. Boolean (N_images,) tensor."""
    assert mask.shape[0] == core.n_images

    # Crystals on kept images
    crystal_mask = mask[core.image_id.long()]

    # Observations belonging to kept crystals
    obs_mask = crystal_mask[core.crystal_id.long()]

    return _reindex(core, obs_mask, crystal_mask, mask)


def filter_outliers(
    core: _SFXStreamCore, n_sigma: float = 4.0
) -> _SFXStreamCore:
    """Remove per-hkl outlier observations.

    Groups by HKL, computes mean/std per group, removes observations
    where |I - mean| > n_sigma * std.
    """
    # Linearize HKL for grouping
    hkl = core.hkl.long()
    # Shift to positive range for hashing
    h_off = hkl[:, 0] - hkl[:, 0].min()
    k_off = hkl[:, 1] - hkl[:, 1].min()
    l_off = hkl[:, 2] - hkl[:, 2].min()
    k_range = k_off.max() + 1
    l_range = l_off.max() + 1
    hkl_linear = h_off * k_range * l_range + k_off * l_range + l_off

    # Compute per-group mean and std
    unique_ids, inverse = hkl_linear.unique(return_inverse=True)
    n_groups = unique_ids.shape[0]

    # scatter mean
    group_sum = torch.zeros(n_groups, dtype=torch.float64)
    group_sum.scatter_add_(0, inverse, core.I.double())
    group_count = torch.zeros(n_groups, dtype=torch.float64)
    group_count.scatter_add_(0, inverse, torch.ones_like(core.I, dtype=torch.float64))
    group_mean = group_sum / group_count.clamp(min=1)

    # scatter variance
    diff = core.I.double() - group_mean[inverse]
    group_var = torch.zeros(n_groups, dtype=torch.float64)
    group_var.scatter_add_(0, inverse, diff ** 2)
    group_std = (group_var / group_count.clamp(min=1)).sqrt()

    # Mask: keep if |I - mean| <= n_sigma * std, or if group has <=2 members
    obs_mask = (diff.abs() <= n_sigma * group_std[inverse]) | (group_count[inverse] <= 2)

    # Determine which crystals/images remain
    kept_crystal_ids = core.crystal_id[obs_mask].unique()
    crystal_mask = torch.zeros(core.n_crystals, dtype=torch.bool)
    crystal_mask[kept_crystal_ids.long()] = True

    kept_image_ids = core.image_id[crystal_mask].unique()
    image_mask = torch.zeros(core.n_images, dtype=torch.bool)
    image_mask[kept_image_ids.long()] = True

    return _reindex(core, obs_mask, crystal_mask, image_mask)


def filter_cell(
    core: _SFXStreamCore, tolerance: float = 0.1
) -> _SFXStreamCore:
    """Remove crystals with unit cell deviating from median by > tolerance (fractional).

    Checks all 6 cell parameters (a, b, c, alpha, beta, gamma).
    """
    median_cell = core.cell.median(dim=0).values  # (6,)
    frac_dev = ((core.cell - median_cell) / median_cell).abs()
    # Crystal passes if ALL 6 parameters are within tolerance
    crystal_mask = (frac_dev <= tolerance).all(dim=1)

    obs_mask = crystal_mask[core.crystal_id.long()]

    kept_image_ids = core.image_id[crystal_mask].unique()
    image_mask = torch.zeros(core.n_images, dtype=torch.bool)
    image_mask[kept_image_ids.long()] = True

    return _reindex(core, obs_mask, crystal_mask, image_mask)
