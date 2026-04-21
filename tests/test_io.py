"""Tests for stream parsing and sfx I/O."""

import pytest
import torch

from torchsx.io._safetensors import load_sfx, save_sfx


def test_sfx_round_trip(sample_core, tmp_path):
    """Save and load .sfx should produce identical tensors."""
    sfx_path = tmp_path / "test.sfx"
    save_sfx(sample_core, sfx_path)
    loaded = load_sfx(sfx_path)

    assert loaded.n_observations == sample_core.n_observations
    assert loaded.n_crystals == sample_core.n_crystals
    assert loaded.n_images == sample_core.n_images

    assert torch.equal(loaded.hkl, sample_core.hkl)
    assert torch.allclose(loaded.I, sample_core.I)
    assert torch.allclose(loaded.sigma, sample_core.sigma)
    assert torch.allclose(loaded.cell, sample_core.cell)
    assert torch.equal(loaded.crystal_id, sample_core.crystal_id)
    assert torch.equal(loaded.image_id, sample_core.image_id)
    assert loaded.spacegroup == sample_core.spacegroup


def test_sfx_geometry_round_trip(sample_core, tmp_path):
    """Geometry should survive sfx round-trip."""
    sfx_path = tmp_path / "test.sfx"
    save_sfx(sample_core, sfx_path)
    loaded = load_sfx(sfx_path)

    assert loaded.geometry is not None
    assert loaded.geometry.n_panels == sample_core.geometry.n_panels
    assert loaded.geometry.photon_energy_eV == sample_core.geometry.photon_energy_eV


@pytest.mark.slow
def test_parse_real_stream():
    """Parse the real merged_368.stream file.

    Run with: pytest -m slow
    Expected: ~81k crystals, ~195k images, 32 panels
    """
    from tests.conftest import STREAM_PATH
    from torchsx.io._parser import parse_stream

    core = parse_stream(STREAM_PATH, spacegroup="P 1", n_workers=1)

    # Known properties of merged_368.stream
    assert core.n_crystals > 80000, f"Expected >80k crystals, got {core.n_crystals}"
    assert core.n_images > 190000, f"Expected >190k images, got {core.n_images}"
    assert core.n_observations > 0

    # Cell parameters should be in Å range
    mean_cell = core.cell.mean(dim=0)
    assert 14.0 < mean_cell[0].item() < 16.0, f"a={mean_cell[0]:.2f}, expected ~14.97"
    assert 18.0 < mean_cell[1].item() < 20.0, f"b={mean_cell[1]:.2f}, expected ~18.85"
    assert 18.0 < mean_cell[2].item() < 20.0, f"c={mean_cell[2]:.2f}, expected ~18.89"

    # Geometry
    assert core.geometry is not None
    assert core.geometry.n_panels == 32
