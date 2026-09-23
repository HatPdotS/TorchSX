"""Test fixtures for torchsx."""

import os
from pathlib import Path

import pytest
import torch

from torchsx._core import _SFXStreamCore
from torchsx.geometry import Geometry, Panel
from torchsx.io._metadata import StreamMetadata

# Path to the real stream file for integration tests
STREAM_PATH = "/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/data/merged_368.stream"

# Real streams for the native parser/writer parity tests; the first that exists
# is used. TORCHSX_TEST_STREAM overrides.
REAL_STREAMS = (
    os.environ.get("TORCHSX_TEST_STREAM", ""),
    STREAM_PATH,
    "/data/project/cls/standfuss/torchsx_new/gym/data/sm20k/A.stream",
)


@pytest.fixture
def merged_368_stream() -> str:
    """The reference stream the geometry/parse tests were written against."""
    if not Path(STREAM_PATH).exists():
        pytest.skip(f"{STREAM_PATH} not available")
    return STREAM_PATH


@pytest.fixture(scope="session")
def real_stream() -> Path:
    for p in REAL_STREAMS:
        if p and Path(p).exists():
            return Path(p)
    pytest.skip("no real .stream file available (set TORCHSX_TEST_STREAM)")


@pytest.fixture
def sample_geometry():
    """Minimal 2-panel geometry."""
    panels = [
        Panel(
            name="m0",
            min_fs=0, max_fs=1029, min_ss=0, max_ss=513,
            corner_x=2138.35, corner_y=1139.55,
            fs_vec=(0.002, -0.999, 0.0),
            ss_vec=(-0.997, -0.002, 0.0),
            coffset=0.0,
        ),
        Panel(
            name="m1",
            min_fs=0, max_fs=1029, min_ss=514, max_ss=1027,
            corner_x=2208.49, corner_y=112.40,
            fs_vec=(-0.0001, -0.9995, 0.0),
            ss_vec=(-0.9977, -0.0001, 0.0),
            coffset=0.0,
        ),
    ]
    return Geometry(
        panels=panels,
        photon_energy_eV=12960.0,
        clen=0.0939,
        res=13333.3,
        adu_per_eV=1.0,
        data_path="/data/test",
    )


@pytest.fixture
def sample_core(sample_geometry):
    """Small synthetic _SFXStreamCore with 2 crystals, 5 observations, 2 images."""
    n_obs = 5
    n_crys = 2
    n_imgs = 2

    return _SFXStreamCore(
        # Observations
        hkl=torch.tensor([[1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 0], [0, 2, 0]], dtype=torch.int16),
        I=torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0], dtype=torch.float32),
        sigma=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32),
        background=torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32),
        peak=torch.tensor([5.0, 6.0, 7.0, 8.0, 9.0], dtype=torch.float32),
        fs=torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0], dtype=torch.float32),
        ss=torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0], dtype=torch.float32),
        panel_id=torch.tensor([0, 0, 1, 0, 1], dtype=torch.int16),
        crystal_id=torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32),
        # Crystals
        astar=torch.tensor([[0.067, 0.0, 0.0], [0.067, 0.0, 0.0]], dtype=torch.float32),
        bstar=torch.tensor([[0.0, 0.053, 0.0], [0.0, 0.053, 0.0]], dtype=torch.float32),
        cstar=torch.tensor([[0.0, 0.0, 0.053], [0.0, 0.0, 0.053]], dtype=torch.float32),
        cell=torch.tensor(
            [[14.97, 18.85, 18.89, 89.37, 84.94, 67.84],
             [14.97, 18.85, 18.89, 89.37, 84.94, 67.84]],
            dtype=torch.float32,
        ),
        profile_rad=torch.tensor([0.004, 0.004], dtype=torch.float32),
        resolution=torch.tensor([2.57, 2.57], dtype=torch.float32),
        n_refl=torch.tensor([3, 2], dtype=torch.int32),
        image_id=torch.tensor([0, 1], dtype=torch.int32),
        crystal_in_image=torch.tensor([0, 0], dtype=torch.int32),
        # Images
        beam_energy=torch.tensor([12960.0, 12960.0], dtype=torch.float32),
        det_shift_x=torch.tensor([0.0, 0.0], dtype=torch.float32),
        det_shift_y=torch.tensor([0.0, 0.0], dtype=torch.float32),
        n_peaks=torch.tensor([10, 8], dtype=torch.int32),
        # Meta
        geometry=sample_geometry,
        metadata=StreamMetadata(),
        spacegroup="P 1",
        source_path="test.stream",
    )


@pytest.fixture
def sample_metadata():
    """StreamMetadata with diverse field types."""
    meta = StreamMetadata()
    meta["crystal"] = {
        "cell": [[14.97, 18.85, 18.89, 89.37, 84.94, 67.84],
                 [15.00, 18.90, 18.92, 89.40, 85.00, 67.90]],
        "scale": [1.0, 1.1],
    }
    meta["image"] = {
        "filename": ["a.h5", "b.h5", "a.h5"],
        "energy": [12960.0, 12960.0, 12960.0],
        "event_id": [1, 2, 3],
    }
    return meta
