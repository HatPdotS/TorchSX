"""Tests for filtering operations.

Exercises the live ``torchsx.merging.hkl.filter`` implementation (the
deprecated ``merge._filter`` copy is retained for frozen ``deprecated``
subpackages; see ``torchsx.deprecated``).
"""

import torch

from torchsx.merging.hkl.filter import (
    filter_cell,
    filter_crystals,
    filter_images,
    filter_outliers,
)


def test_filter_crystals(sample_core):
    """Filter to keep only crystal 0."""
    mask = torch.tensor([True, False])
    result = filter_crystals(sample_core, mask)

    assert result.n_crystals == 1
    assert result.n_observations == 3  # crystal 0 has 3 obs
    assert result.crystal_id.max().item() == 0


def test_filter_images(sample_core):
    """Filter to keep only image 0."""
    mask = torch.tensor([True, False])
    result = filter_images(sample_core, mask)

    assert result.n_images == 1
    assert result.n_crystals == 1  # crystal 0 on image 0
    assert result.n_observations == 3


def test_filter_outliers(sample_core):
    """Outlier filter should not remove anything from small uniform data."""
    result = filter_outliers(sample_core, n_sigma=4.0)
    # With only 2 crystals and unique HKLs, nothing should be removed
    assert result.n_observations == sample_core.n_observations


def test_filter_cell(sample_core):
    """Cell filter with tight tolerance on identical cells should keep all."""
    result = filter_cell(sample_core, tolerance=0.1)
    assert result.n_crystals == sample_core.n_crystals


def test_filter_cell_removes_outlier():
    """Cell filter should remove crystals with deviant cells."""
    from torchsx._core import _SFXStreamCore

    core = _SFXStreamCore(
        hkl=torch.tensor([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=torch.int16),
        I=torch.tensor([10.0, 20.0, 30.0]),
        sigma=torch.tensor([1.0, 2.0, 3.0]),
        background=torch.zeros(3),
        peak=torch.ones(3),
        fs=torch.zeros(3),
        ss=torch.zeros(3),
        panel_id=torch.zeros(3, dtype=torch.int16),
        crystal_id=torch.tensor([0, 1, 2], dtype=torch.int32),
        astar=torch.zeros(3, 3),
        bstar=torch.zeros(3, 3),
        cstar=torch.zeros(3, 3),
        cell=torch.tensor([
            [14.97, 18.85, 18.89, 89.37, 84.94, 67.84],
            [14.97, 18.85, 18.89, 89.37, 84.94, 67.84],
            [20.00, 25.00, 25.00, 90.00, 90.00, 90.00],  # outlier
        ], dtype=torch.float32),
        profile_rad=torch.zeros(3),
        resolution=torch.ones(3) * 2.57,
        n_refl=torch.tensor([1, 1, 1], dtype=torch.int32),
        image_id=torch.tensor([0, 1, 2], dtype=torch.int32),
        crystal_in_image=torch.tensor([0, 0, 0], dtype=torch.int32),
        beam_energy=torch.ones(3) * 12960.0,
        det_shift_x=torch.zeros(3),
        det_shift_y=torch.zeros(3),
        n_peaks=torch.ones(3, dtype=torch.int32),
    )

    result = filter_cell(core, tolerance=0.05)
    assert result.n_crystals == 2
    assert result.n_observations == 2
