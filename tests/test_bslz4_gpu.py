"""Tests for the GPU bslz4 decoder (torchsx.datareduction._bslz4_gpu).

The framing parser and filter detection run anywhere (h5py + hdf5plugin). The
bit-exact decode test is guarded on CUDA + nvcomp + Triton and skips otherwise.
A self-contained fixture writes a real bitshuffle+LZ4 dataset (multiple full
blocks + a partial last block) so no external data is required.
"""
from __future__ import annotations

import numpy as np
import pytest

from torchsx.datareduction._bslz4_gpu import (
    dataset_is_bslz4,
    parse_bslz4_chunk,
)

h5py = pytest.importorskip("h5py")
hdf5plugin = pytest.importorskip("hdf5plugin")

# 10000 int32 / frame, 1024-elem blocks -> 9 full blocks + a 784-elem partial.
_NFRAMES, _H, _W, _NELEMS = 3, 100, 100, 1024


@pytest.fixture
def bslz4_file(tmp_path):
    path = str(tmp_path / "bslz4.h5")
    rng = np.random.default_rng(0)
    data = rng.integers(0, 2000, size=(_NFRAMES, _H, _W), dtype=np.int32)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data, chunks=(1, _H, _W),
                         **hdf5plugin.Bitshuffle(nelems=_NELEMS, cname="lz4"))
    return path, data


def test_parse_bslz4_chunk(bslz4_file):
    path, data = bslz4_file
    with h5py.File(path, "r") as f:
        ds = f["data"]
        _mask, raw = ds.id.read_direct_chunk((0, 0, 0))
        total_nbytes, block_nbytes, blocks = parse_bslz4_chunk(bytes(raw))
    assert total_nbytes == _H * _W * 4
    assert block_nbytes == _NELEMS * 4
    assert sum(u for _o, _c, u in blocks) == total_nbytes      # blocks (bytes) tile the frame
    assert len(blocks) == -(-total_nbytes // block_nbytes)     # ceil division
    # partial last block, in bytes
    assert blocks[-1][2] == total_nbytes - (total_nbytes // block_nbytes) * block_nbytes


def test_dataset_is_bslz4(bslz4_file, tmp_path):
    path, _ = bslz4_file
    with h5py.File(path, "r") as f:
        assert dataset_is_bslz4(f["data"]) is True
    plain = str(tmp_path / "plain.h5")
    with h5py.File(plain, "w") as f:
        f.create_dataset("data", data=np.zeros((2, 8, 8), np.int32), chunks=(1, 8, 8))
        assert dataset_is_bslz4(f["data"]) is False


def _gpu_available():
    try:
        import torch
        from nvidia import nvcomp  # noqa: F401
        import triton  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_available(), reason="needs CUDA + nvcomp + triton")
def test_decoder_bitexact(bslz4_file):
    import torch
    from torchsx.datareduction._bslz4_gpu import Bslz4GpuDecoder

    path, data = bslz4_file
    dev = torch.device("cuda")
    with h5py.File(path, "r") as f:
        ds = f["data"]
        _m, raw0 = ds.id.read_direct_chunk((0, 0, 0))
        _t, block_nbytes, _b = parse_bslz4_chunk(bytes(raw0))
        decoder = Bslz4GpuDecoder(_H * _W, block_nbytes // 4, dev)
        for i in range(_NFRAMES):
            _m, raw = ds.id.read_direct_chunk((i, 0, 0))
            got = decoder.decode(bytes(raw)).cpu().numpy().reshape(_H, _W)
            assert np.array_equal(got, data[i]), f"frame {i} mismatch"
