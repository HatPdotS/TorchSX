"""Tests for ROIStore — memory-mapped ROI patch storage."""

import numpy as np
import pytest
import torch

from torchsx._core import _SFXStreamCore
from torchsx.io._metadata import StreamMetadata
from torchsx.io._roi_store import ROIStore


def _make_core_with_hdf5(tmp_path, n_obs=4, n_events=2):
    """Create a synthetic HDF5 file and matching _SFXStreamCore."""
    h5py = pytest.importorskip("h5py")

    h5_path = str(tmp_path / "test.h5")
    frame_h, frame_w = 100, 100

    with h5py.File(h5_path, "w") as f:
        data = np.zeros((n_events, frame_h, frame_w), dtype=np.float32)
        for ev in range(n_events):
            for ss in range(frame_h):
                for fs in range(frame_w):
                    data[ev, ss, fs] = ev * 10000 + ss * 100 + fs
        f.create_dataset("/data/data", data=data)

    meta = StreamMetadata()
    meta["image"] = {
        "filename": [h5_path] * n_events,
        "event": [f"//{i}" for i in range(n_events)],
    }

    crystal_ids = []
    obs_fs = []
    obs_ss = []
    for ev in range(n_events):
        n_per_ev = n_obs // n_events
        crystal_ids.extend([ev] * n_per_ev)
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
        astar=torch.zeros(n_crys, 3), bstar=torch.zeros(n_crys, 3),
        cstar=torch.zeros(n_crys, 3), cell=torch.zeros(n_crys, 6),
        profile_rad=torch.zeros(n_crys), resolution=torch.ones(n_crys),
        n_refl=torch.full((n_crys,), n_obs // n_events, dtype=torch.int32),
        image_id=torch.arange(n_crys, dtype=torch.int32),
        crystal_in_image=torch.zeros(n_crys, dtype=torch.int32),
        beam_energy=torch.full((n_events,), 12960.0),
        det_shift_x=torch.zeros(n_events),
        det_shift_y=torch.zeros(n_events),
        n_peaks=torch.full((n_events,), 10, dtype=torch.int32),
        metadata=meta,
    )
    return core


def test_excise_and_load(tmp_path):
    """Excise → save → memory-mapped access."""
    core = _make_core_with_hdf5(tmp_path)
    roi_path = tmp_path / "rois.safetensors"

    rois = ROIStore.excise(core, roi_path, patch_size=5, data_path="/data/data")

    assert roi_path.exists()
    assert len(rois) == 4
    assert rois.patch_size == (5, 5)
    assert rois["patches"].shape == torch.Size([4, 5, 5])
    assert rois["valid"].shape == torch.Size([4, 5, 5])
    assert rois["fs_ref"].shape == torch.Size([4])
    assert rois["ss_ref"].shape == torch.Size([4])
    assert "patches" in rois
    assert "nonexistent" not in rois

    # Verify pixel values
    for i in range(4):
        fs_center = round(core.fs[i].item())
        ss_center = round(core.ss[i].item())
        img_id = core.image_id[core.crystal_id[i]].item()
        expected = img_id * 10000 + ss_center * 100 + fs_center
        assert rois["patches"][i, 2, 2].item() == pytest.approx(expected)


def test_reload_from_disk(tmp_path):
    """ROIStore can be reopened from the saved file."""
    core = _make_core_with_hdf5(tmp_path)
    roi_path = tmp_path / "rois.safetensors"

    rois = ROIStore.excise(core, roi_path, patch_size=5, data_path="/data/data")
    rois.close()

    rois2 = ROIStore(roi_path)
    assert len(rois2) == 4
    assert rois2["patches"].shape == torch.Size([4, 5, 5])


def test_update_no_change(tmp_path):
    """Update with no position change returns same store."""
    core = _make_core_with_hdf5(tmp_path)
    roi_path = tmp_path / "rois.safetensors"

    rois = ROIStore.excise(core, roi_path, patch_size=5, data_path="/data/data")
    rois2 = rois.update(core, threshold=0.5, data_path="/data/data")

    # Same object (no re-excision needed)
    assert rois2 is rois


def test_update_with_movement(tmp_path):
    """Update after moving positions re-excises only moved observations."""
    core = _make_core_with_hdf5(tmp_path)
    roi_path = tmp_path / "rois.safetensors"

    rois = ROIStore.excise(core, roi_path, patch_size=5, data_path="/data/data")
    old_patch_0 = rois["patches"][0].clone()
    old_patch_1 = rois["patches"][1].clone()

    # Move observation 0 by 2 pixels in fs
    core.fs[0] = core.fs[0] + 2.0

    rois2 = rois.update(core, threshold=0.5, data_path="/data/data")

    # Observation 0 should have changed
    new_patch_0 = rois2["patches"][0]
    assert not torch.equal(new_patch_0, old_patch_0)

    # Observation 1 should be unchanged
    new_patch_1 = rois2["patches"][1]
    assert torch.equal(new_patch_1, old_patch_1)

    # Reference positions should be updated
    assert rois2["fs_ref"][0].item() == pytest.approx(core.fs[0].item())


def test_update_with_obs_mask(tmp_path):
    """Update works correctly with obs_mask subset."""
    core = _make_core_with_hdf5(tmp_path)
    roi_path = tmp_path / "rois.safetensors"

    mask = torch.tensor([True, False, True, False], dtype=torch.bool)
    rois = ROIStore.excise(
        core, roi_path, patch_size=5, data_path="/data/data", obs_mask=mask,
    )
    assert len(rois) == 2

    # Move obs 0 (first in mask)
    core.fs[0] = core.fs[0] + 3.0

    rois2 = rois.update(core, threshold=0.5, data_path="/data/data", obs_mask=mask)
    assert len(rois2) == 2
    assert rois2["fs_ref"][0].item() == pytest.approx(core.fs[0].item())


def test_repr(tmp_path):
    """Repr shows useful info."""
    core = _make_core_with_hdf5(tmp_path)
    roi_path = tmp_path / "rois.safetensors"

    rois = ROIStore.excise(core, roi_path, patch_size=5, data_path="/data/data")
    r = repr(rois)
    assert "ROIStore" in r
    assert "n=4" in r
    assert "5x5" in r
