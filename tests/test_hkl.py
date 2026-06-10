"""Tests for HKL indexing and merging.

Exercises the live ``torchsx.merging.indexing.hkl`` implementation (the
deprecated ``merge._hkl`` copy it was lifted from is retained only because
frozen ``deprecated`` subpackages still import it; see ``torchsx.deprecated``).
"""

import torch

from torchsx.merging.indexing.hkl import compute_hkl_index, merge_intensities


def test_compute_hkl_index_basic():
    hkl = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=torch.int16)
    idx = compute_hkl_index(hkl)

    assert idx.n_unique == 3
    assert idx.hkl_id.shape == (5,)
    # Same HKL should get same index
    assert idx.hkl_id[0] == idx.hkl_id[2]  # both (1,0,0)
    assert idx.hkl_id[1] == idx.hkl_id[4]  # both (0,1,0)
    assert idx.hkl_id[0] != idx.hkl_id[1]  # (1,0,0) != (0,1,0)


def test_compute_hkl_index_negative():
    """Negative Miller indices should work."""
    hkl = torch.tensor([[-1, 2, 3], [1, -2, 3], [-1, 2, 3]], dtype=torch.int16)
    idx = compute_hkl_index(hkl)
    assert idx.n_unique == 2
    assert idx.hkl_id[0] == idx.hkl_id[2]


def test_compute_hkl_index_all_unique():
    hkl = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.int16)
    idx = compute_hkl_index(hkl)
    assert idx.n_unique == 3


def test_compute_hkl_index_all_same():
    hkl = torch.tensor([[1, 2, 3]] * 100, dtype=torch.int16)
    idx = compute_hkl_index(hkl)
    assert idx.n_unique == 1
    assert (idx.hkl_id == 0).all()


def test_merge_intensities_simple():
    """Merge two observations of the same HKL."""
    hkl_id = torch.tensor([0, 0, 1], dtype=torch.int64)
    I = torch.tensor([10.0, 20.0, 30.0])
    sigma = torch.tensor([1.0, 1.0, 2.0])

    I_m, sig_m, mult = merge_intensities(hkl_id, I, sigma, n_unique=2)

    assert I_m.shape == (2,)
    assert sig_m.shape == (2,)
    assert mult[0] == 2
    assert mult[1] == 1
    # Equal weights → simple mean for hkl_id=0
    assert abs(I_m[0].item() - 15.0) < 0.01
    # Single obs for hkl_id=1
    assert abs(I_m[1].item() - 30.0) < 0.01


def test_merge_intensities_weighted():
    """Observations with different sigma get different weights."""
    hkl_id = torch.tensor([0, 0], dtype=torch.int64)
    I = torch.tensor([10.0, 20.0])
    sigma = torch.tensor([1.0, 2.0])  # first has 4x weight

    I_m, _, _ = merge_intensities(hkl_id, I, sigma, n_unique=1)

    # w1 = 1/1 = 1, w2 = 1/4 = 0.25
    # I_merged = (10*1 + 20*0.25) / (1 + 0.25) = 15/1.25 = 12.0
    assert abs(I_m[0].item() - 12.0) < 0.01


def test_sfxstream_compute_hkl_index(sample_core):
    """HKL index via SFXStream API."""
    from torchsx.stream import SFXStream
    stream = SFXStream(sample_core)
    idx = stream.compute_hkl_index()
    assert idx.n_unique <= stream.n_observations
    assert idx.hkl_id.shape == (stream.n_observations,)


def test_sfxstream_merge(sample_core):
    """Merge via SFXStream API."""
    from torchsx.stream import SFXStream
    stream = SFXStream(sample_core)
    merged = stream.merge()
    assert "hkl" in merged
    assert "I" in merged
    assert "sigma" in merged
    assert "multiplicity" in merged
    assert merged["hkl"].shape[0] == merged["I"].shape[0]
