"""Tests for pixel patch extraction from HDF5 images."""

import numpy as np
import pytest
import torch

from torchsx._core import _SFXStreamCore
from torchsx.io._excise import (
    ExcisedPatches,
    _clip_patch,
    _extract_frame_patches,
    _parse_event_id,
    excise_patches,
)
from torchsx.io._metadata import StreamMetadata


# --- Unit tests ---


@pytest.mark.parametrize(
    "event_str, expected",
    [
        ("//948", 948),
        ("/948", 948),
        ("948", 948),
        ("//0", 0),
        ("  //42  ", 42),
    ],
)
def test_parse_event_id(event_str, expected):
    assert _parse_event_id(event_str) == expected


def test_clip_patch_interior():
    src, dst = _clip_patch(center=50, half=10, array_dim=100)
    assert src == slice(40, 61)
    assert dst == slice(0, 21)


def test_clip_patch_left_boundary():
    src, dst = _clip_patch(center=3, half=10, array_dim=100)
    assert src == slice(0, 14)
    assert dst == slice(7, 21)


def test_clip_patch_right_boundary():
    src, dst = _clip_patch(center=95, half=10, array_dim=100)
    assert src == slice(85, 100)
    assert dst == slice(0, 15)


def test_clip_patch_both_boundaries():
    # Patch bigger than array
    src, dst = _clip_patch(center=5, half=10, array_dim=12)
    assert src == slice(0, 12)
    assert dst == slice(5, 17)


def test_extract_frame_patches_interior():
    frame = np.arange(400, dtype=np.float32).reshape(20, 20)
    centers_ss = np.array([10])
    centers_fs = np.array([10])
    patches, valid = _extract_frame_patches(frame, centers_ss, centers_fs, 2, 2, 0.0)
    assert patches.shape == (1, 5, 5)
    assert valid.all()
    assert patches[0, 2, 2] == frame[10, 10]


def test_extract_frame_patches_boundary():
    frame = np.ones((20, 20), dtype=np.float32) * 7.0
    centers_ss = np.array([0])
    centers_fs = np.array([0])
    patches, valid = _extract_frame_patches(frame, centers_ss, centers_fs, 2, 2, -1.0)
    assert patches.shape == (1, 5, 5)
    # Top-left corner: only bottom-right quadrant should be valid
    assert valid[0, 2, 2]  # center pixel
    assert not valid[0, 0, 0]  # out of bounds
    assert patches[0, 0, 0] == -1.0  # fill value


# --- Integration tests with synthetic HDF5 ---


def _make_core_with_hdf5(tmp_path, n_obs=4, n_events=2, patch_val=42.0):
    """Create a synthetic HDF5 file and matching _SFXStreamCore."""
    h5py = pytest.importorskip("h5py")

    h5_path = str(tmp_path / "test.h5")
    frame_h, frame_w = 100, 100

    # Write HDF5 with known pixel values
    with h5py.File(h5_path, "w") as f:
        data = np.zeros((n_events, frame_h, frame_w), dtype=np.float32)
        for ev in range(n_events):
            # Each pixel = event_idx * 1000 + ss * 100 + fs (for unique identification)
            for ss in range(frame_h):
                for fs in range(frame_w):
                    data[ev, ss, fs] = ev * 10000 + ss * 100 + fs
        f.create_dataset("/data/data", data=data)

    # Build metadata: 1 image per event
    meta = StreamMetadata()
    meta["image"] = {
        "filename": [h5_path] * n_events,
        "event": [f"//{i}" for i in range(n_events)],
    }

    # Observations spread across events
    crystal_ids = []
    image_ids_per_crystal = []
    obs_fs = []
    obs_ss = []
    for ev in range(n_events):
        n_per_ev = n_obs // n_events
        crystal_ids.extend([ev] * n_per_ev)
        image_ids_per_crystal.append(ev)
        for j in range(n_per_ev):
            obs_fs.append(50.0 + j * 5)
            obs_ss.append(50.0 + j * 5)

    n_total = len(crystal_ids)
    n_crys = n_events

    core = _SFXStreamCore(
        hkl=torch.zeros(n_total, 3, dtype=torch.int16),
        I=torch.ones(n_total, dtype=torch.float32),
        sigma=torch.ones(n_total, dtype=torch.float32),
        background=torch.zeros(n_total, dtype=torch.float32),
        peak=torch.ones(n_total, dtype=torch.float32),
        fs=torch.tensor(obs_fs, dtype=torch.float32),
        ss=torch.tensor(obs_ss, dtype=torch.float32),
        panel_id=torch.zeros(n_total, dtype=torch.int16),
        crystal_id=torch.tensor(crystal_ids, dtype=torch.int32),
        astar=torch.zeros(n_crys, 3, dtype=torch.float32),
        bstar=torch.zeros(n_crys, 3, dtype=torch.float32),
        cstar=torch.zeros(n_crys, 3, dtype=torch.float32),
        cell=torch.zeros(n_crys, 6, dtype=torch.float32),
        profile_rad=torch.zeros(n_crys, dtype=torch.float32),
        resolution=torch.ones(n_crys, dtype=torch.float32),
        n_refl=torch.full((n_crys,), n_obs // n_events, dtype=torch.int32),
        image_id=torch.tensor(image_ids_per_crystal, dtype=torch.int32),
        crystal_in_image=torch.zeros(n_crys, dtype=torch.int32),
        beam_energy=torch.full((n_events,), 12960.0),
        det_shift_x=torch.zeros(n_events),
        det_shift_y=torch.zeros(n_events),
        n_peaks=torch.full((n_events,), 10, dtype=torch.int32),
        metadata=meta,
    )
    return core, h5_path


def test_excise_synthetic(tmp_path):
    """End-to-end: write HDF5, extract patches, verify pixel values."""
    core, h5_path = _make_core_with_hdf5(tmp_path, n_obs=4, n_events=2)
    result = excise_patches(core, patch_size=5, data_path="/data/data")

    assert isinstance(result, ExcisedPatches)
    assert result.patches.shape == (4, 5, 5)
    assert result.valid.shape == (4, 5, 5)
    assert result.patch_size == (5, 5)

    # Verify center pixel values
    for i in range(4):
        fs_center = round(core.fs[i].item())
        ss_center = round(core.ss[i].item())
        img_id = core.image_id[core.crystal_id[i]].item()
        expected_val = img_id * 10000 + ss_center * 100 + fs_center
        assert result.patches[i, 2, 2].item() == pytest.approx(expected_val)

    # All interior patches should be fully valid
    assert result.valid.all()


def test_mask_application(tmp_path):
    """Masked pixels should have valid=False."""
    h5py = pytest.importorskip("h5py")
    core, h5_path = _make_core_with_hdf5(tmp_path, n_obs=4, n_events=2)

    # Create a mask file where some pixels are bad (True = bad)
    mask_path = str(tmp_path / "mask.h5")
    mask = np.zeros((100, 100), dtype=bool)
    mask[50, 50] = True  # Mark center pixel as bad
    with h5py.File(mask_path, "w") as f:
        f.create_dataset("/data/mask", data=mask)

    result = excise_patches(
        core, patch_size=5, data_path="/data/data",
        mask_path=mask_path, mask_dataset="/data/mask",
    )

    # Observation at (50, 50) should have center pixel invalid
    for i in range(result.patches.shape[0]):
        fs_center = round(core.fs[i].item())
        ss_center = round(core.ss[i].item())
        if fs_center == 50 and ss_center == 50:
            assert not result.valid[i, 2, 2]


def test_obs_mask_subset(tmp_path):
    """Extract only a subset via obs_mask."""
    core, h5_path = _make_core_with_hdf5(tmp_path, n_obs=4, n_events=2)

    mask = torch.zeros(4, dtype=torch.bool)
    mask[0] = True
    mask[2] = True

    result = excise_patches(core, patch_size=5, data_path="/data/data", obs_mask=mask)

    assert result.patches.shape[0] == 2
    assert result.obs_index.tolist() == [0, 2]


def test_subpixel_center(tmp_path):
    """Verify center_fs/center_ss encode sub-pixel offsets."""
    core, h5_path = _make_core_with_hdf5(tmp_path, n_obs=4, n_events=2)

    # Set a sub-pixel offset
    core.fs[0] = 50.3
    core.ss[0] = 50.7

    result = excise_patches(core, patch_size=5, data_path="/data/data")

    # center should be half-width + fractional offset from rounded center
    # rounded center: fs=50, ss=51 → frac_fs = 0.3, frac_ss = -0.3
    # But rounding 50.7 → 51, so frac = 50.7 - 51 = -0.3
    expected_fs = 2 + 0.3  # hw + frac
    expected_ss = 2 + (50.7 - 51)  # hh + frac
    assert result.center_fs[0].item() == pytest.approx(expected_fs, abs=1e-5)
    assert result.center_ss[0].item() == pytest.approx(expected_ss, abs=1e-5)


def test_missing_file_warns(tmp_path):
    """Missing HDF5 file should warn, not crash."""
    pytest.importorskip("h5py")
    meta = StreamMetadata()
    meta["image"] = {
        "filename": ["/nonexistent/file.h5"],
        "event": ["//0"],
    }

    core = _SFXStreamCore(
        hkl=torch.zeros(1, 3, dtype=torch.int16),
        I=torch.ones(1), sigma=torch.ones(1),
        background=torch.zeros(1), peak=torch.ones(1),
        fs=torch.tensor([10.0]), ss=torch.tensor([10.0]),
        panel_id=torch.zeros(1, dtype=torch.int16),
        crystal_id=torch.zeros(1, dtype=torch.int32),
        astar=torch.zeros(1, 3), bstar=torch.zeros(1, 3), cstar=torch.zeros(1, 3),
        cell=torch.zeros(1, 6), profile_rad=torch.zeros(1),
        resolution=torch.ones(1), n_refl=torch.ones(1, dtype=torch.int32),
        image_id=torch.zeros(1, dtype=torch.int32),
        crystal_in_image=torch.zeros(1, dtype=torch.int32),
        beam_energy=torch.ones(1), det_shift_x=torch.zeros(1),
        det_shift_y=torch.zeros(1), n_peaks=torch.ones(1, dtype=torch.int32),
        metadata=meta,
    )

    with pytest.warns(UserWarning, match="Cannot open HDF5"):
        result = excise_patches(core, patch_size=5)

    assert not result.valid.any()
    assert (result.patches == 0.0).all()
