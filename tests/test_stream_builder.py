"""Tests for the async, chunked StreamBuilder."""

import threading

import torch

from torchsx.io._stream_builder import StreamBuilder, StreamChunk


def _generic_chunk(n_img, crystals_per_img, obs_per_crystal, key=None, base_val=0.0):
    """Build a chunk with chunk-local join keys (generic, non-SX, columns)."""
    image = {"ival": base_val + torch.arange(n_img, dtype=torch.float32)}
    n_cryst = n_img * crystals_per_img
    image_id = torch.arange(n_img, dtype=torch.int32).repeat_interleave(crystals_per_img)
    crystal = {"image_id": image_id, "cval": torch.arange(n_cryst, dtype=torch.float32)}
    n_obs = n_cryst * obs_per_crystal
    crystal_id = torch.arange(n_cryst, dtype=torch.int32).repeat_interleave(obs_per_crystal)
    obs = {"crystal_id": crystal_id, "val": base_val + torch.arange(n_obs, dtype=torch.float32)}
    return StreamChunk(levels={"obs": obs, "crystal": crystal, "image": image}, key=key)


def test_single_chunk_finalizes():
    b = StreamBuilder(async_writer=False)
    b.add(_generic_chunk(1, 2, 3))
    core = b.finalize()
    assert core.n_images == 1
    assert core.n_crystals == 2
    assert core.n_observations == 6


def test_join_keys_offset_across_chunks():
    b = StreamBuilder(async_writer=False)
    b.add(_generic_chunk(1, 2, 2))   # img0, crystals 0-1, obs ref crystal 0/1
    b.add(_generic_chunk(1, 2, 2))   # img1 -> global image_id 1; crystals 2-3
    core = b.finalize()
    assert core.n_images == 2
    assert core.n_crystals == 4
    # second chunk's crystals point at global image 1
    assert torch.equal(core.image_id, torch.tensor([0, 0, 1, 1], dtype=torch.int32))
    # second chunk's obs point at global crystals 2,3
    assert core.crystal_id.max().item() == 3
    assert torch.equal(
        core.crystal_id, torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int32)
    )


def test_key_row_range_index():
    b = StreamBuilder(async_writer=False)
    b.add(_generic_chunk(1, 1, 3, key=("a.h5", 0)))
    b.add(_generic_chunk(1, 1, 2, key=("a.h5", 1)))
    b.finalize()
    assert b.index[("a.h5", 0)]["obs"] == (0, 3)
    assert b.index[("a.h5", 1)]["obs"] == (3, 5)


def test_async_handoff_and_finalize():
    b = StreamBuilder(async_writer=True)
    for _ in range(5):
        b.add(_generic_chunk(1, 2, 2))
    core = b.finalize()
    assert core.n_images == 5
    assert core.n_crystals == 10
    assert core.n_observations == 20
    # join integrity validated inside finalize (_make._validate); also check ranges
    assert core.crystal_id.max().item() == 9
    assert core.image_id.max().item() == 4


def test_concurrent_producers_preserve_integrity():
    b = StreamBuilder(async_writer=True)

    def produce():
        for _ in range(10):
            b.add(_generic_chunk(1, 2, 2))

    threads = [threading.Thread(target=produce) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    core = b.finalize()
    # 4 producers x 10 chunks x (1 img, 2 cryst, 4 obs)
    assert core.n_images == 40
    assert core.n_crystals == 80
    assert core.n_observations == 160
    # finalize ran _validate() without raising -> all join keys in range
    assert core.crystal_id.min().item() >= 0
    assert core.crystal_id.max().item() < core.n_crystals
    assert core.image_id.max().item() < core.n_images


def test_equivalence_to_original_sx_core(sample_core):
    """Splitting an SX core into per-image chunks and rebuilding reproduces it."""
    from torchsx._core import _OBS_FIELDS, _CRYSTAL_FIELDS, _IMAGE_FIELDS

    b = StreamBuilder(
        geometry=sample_core.geometry, spacegroup=sample_core.spacegroup
    )
    for img in range(sample_core.n_images):
        cryst_mask = sample_core.image_id == img
        cryst_idx = cryst_mask.nonzero(as_tuple=True)[0]
        # local remap of crystals within this chunk
        local_of_global = {int(g): i for i, g in enumerate(cryst_idx.tolist())}
        obs_mask = torch.tensor(
            [int(c) in local_of_global for c in sample_core.crystal_id.tolist()]
        )
        obs_cols = {f: getattr(sample_core, f)[obs_mask].clone() for f in _OBS_FIELDS}
        obs_cols["crystal_id"] = torch.tensor(
            [local_of_global[int(c)] for c in sample_core.crystal_id[obs_mask].tolist()],
            dtype=torch.int32,
        )
        crystal_cols = {f: getattr(sample_core, f)[cryst_mask].clone() for f in _CRYSTAL_FIELDS}
        crystal_cols["image_id"] = torch.zeros(int(cryst_mask.sum()), dtype=torch.int32)
        image_cols = {
            f: getattr(sample_core, f)[img:img + 1].clone() for f in _IMAGE_FIELDS
        }
        b.add(StreamChunk(levels={"obs": obs_cols, "crystal": crystal_cols, "image": image_cols}))

    rebuilt = b.finalize()
    assert rebuilt.n_observations == sample_core.n_observations
    assert rebuilt.n_crystals == sample_core.n_crystals
    assert rebuilt.n_images == sample_core.n_images
    assert torch.equal(rebuilt.hkl, sample_core.hkl)
    assert torch.allclose(rebuilt.I, sample_core.I)
    assert torch.equal(rebuilt.crystal_id, sample_core.crystal_id)
    assert torch.equal(rebuilt.image_id, sample_core.image_id)
    assert type(rebuilt).__name__ == "SFXStreamCore"
