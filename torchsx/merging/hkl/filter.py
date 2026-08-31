"""Filtering operations on ``_SFXStreamCore``.  All return new (immutable) cores."""

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

    Thin wrapper over the generic columnar cascade
    (:meth:`BaseStream.select_cascade`), which subsets every column in one loop
    and remaps each join key onto the surviving parent rows — so any new column
    (e.g. a shoebox) rides along automatically.
    """
    return core.select_cascade(
        {"obs": obs_mask, "crystal": crystal_mask, "image": image_mask}
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


def filter_ewald(
    core: _SFXStreamCore,
    max_offset: float | None = None,
    invert: bool = False,
    *,
    max_offset_rel: float | None = None,
) -> _SFXStreamCore:
    """Keep observations close to the Ewald sphere.

    Exactly one threshold mode must be supplied:

    * ``max_offset`` (Å⁻¹) — uniform absolute cutoff, ``|r| ≤ max_offset``.
    * ``max_offset_rel`` (dimensionless) — per-crystal cutoff scaled by the
      crystal's CrystFEL integration radius, ``|r| ≤ max_offset_rel ·
      profile_rad[crystal]``.  Adapts to per-crystal mosaicity / divergence.

    ``invert=True`` flips the predicate (keep the off-Bragg tail), useful for
    building a background-only sample for σ²(bg) calibration.
    """
    from torchsx.merging.observations import _ewald_offset

    if (max_offset is None) == (max_offset_rel is None):
        raise ValueError(
            "filter_ewald: supply exactly one of max_offset or max_offset_rel"
        )

    r = _ewald_offset(core).abs()
    if max_offset_rel is not None:
        thr_per_obs = max_offset_rel * core.profile_rad[core.crystal_id.long()]
        obs_mask = (r > thr_per_obs) if invert else (r <= thr_per_obs)
    else:
        obs_mask = (r > max_offset) if invert else (r <= max_offset)

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
