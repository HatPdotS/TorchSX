"""Tests for _SFXStreamCore."""

import pytest
import torch

from torchsx._core import _SFXStreamCore


def test_core_dimensions(sample_core):
    assert sample_core.n_observations == 5
    assert sample_core.n_crystals == 2
    assert sample_core.n_images == 2


def test_core_validation_crystal_id_range():
    """crystal_id out of range should raise."""
    with pytest.raises(AssertionError, match="crystal_id max"):
        _SFXStreamCore(
            hkl=torch.tensor([[1, 0, 0]], dtype=torch.int16),
            I=torch.tensor([1.0]),
            sigma=torch.tensor([1.0]),
            background=torch.tensor([0.1]),
            peak=torch.tensor([1.0]),
            fs=torch.tensor([100.0]),
            ss=torch.tensor([100.0]),
            panel_id=torch.tensor([0], dtype=torch.int16),
            crystal_id=torch.tensor([5], dtype=torch.int32),  # out of range
            astar=torch.zeros(1, 3),
            bstar=torch.zeros(1, 3),
            cstar=torch.zeros(1, 3),
            cell=torch.zeros(1, 6),
            profile_rad=torch.zeros(1),
            resolution=torch.zeros(1),
            n_refl=torch.tensor([1], dtype=torch.int32),
            image_id=torch.tensor([0], dtype=torch.int32),
            crystal_in_image=torch.tensor([0], dtype=torch.int32),
            beam_energy=torch.tensor([12960.0]),
            det_shift_x=torch.tensor([0.0]),
            det_shift_y=torch.tensor([0.0]),
            n_peaks=torch.tensor([1], dtype=torch.int32),
        )


def test_core_to_device(sample_core):
    """Test device transfer."""
    core2 = sample_core.to("cpu")
    assert core2.hkl.device == torch.device("cpu")
    assert core2.n_observations == sample_core.n_observations


def test_core_obs_length_mismatch():
    """Mismatched observation tensor lengths should raise."""
    with pytest.raises(AssertionError):
        _SFXStreamCore(
            hkl=torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.int16),
            I=torch.tensor([1.0]),  # wrong length
            sigma=torch.tensor([1.0, 2.0]),
            background=torch.tensor([0.1, 0.2]),
            peak=torch.tensor([1.0, 2.0]),
            fs=torch.tensor([100.0, 200.0]),
            ss=torch.tensor([100.0, 200.0]),
            panel_id=torch.tensor([0, 0], dtype=torch.int16),
            crystal_id=torch.tensor([0, 0], dtype=torch.int32),
            astar=torch.zeros(1, 3),
            bstar=torch.zeros(1, 3),
            cstar=torch.zeros(1, 3),
            cell=torch.zeros(1, 6),
            profile_rad=torch.zeros(1),
            resolution=torch.zeros(1),
            n_refl=torch.tensor([2], dtype=torch.int32),
            image_id=torch.tensor([0], dtype=torch.int32),
            crystal_in_image=torch.tensor([0], dtype=torch.int32),
            beam_energy=torch.tensor([12960.0]),
            det_shift_x=torch.tensor([0.0]),
            det_shift_y=torch.tensor([0.0]),
            n_peaks=torch.tensor([1], dtype=torch.int32),
        )
