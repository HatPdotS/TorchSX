"""Tests for StreamBatcher / GroupedBatch."""

import torch

from torchsx.nn.batch import StreamBatcher
from torchsx.stream import SFXStream


def test_batch_by_crystal_shapes(sample_core):
    # sample_core: 2 crystals, obs crystal_id = [0,0,0,1,1]
    batcher = StreamBatcher(sample_core, group_by="crystal", obs_columns=["I"],
                            groups_per_batch=8)
    batches = list(batcher)
    assert len(batches) == 1
    b = batches[0]
    # 2 groups, L_max = 3 (crystal 0 has 3 obs)
    assert b.columns["I"].shape == (2, 3)
    assert b.padding_mask.shape == (2, 3)
    assert torch.equal(b.n_per_group, torch.tensor([3, 2]))
    # crystal 1's third slot is padded
    assert b.padding_mask[1, 2].item() is True
    assert b.padding_mask[0, 2].item() is False


def test_unpack_recovers_values(sample_core):
    batcher = StreamBatcher(sample_core, group_by="crystal", obs_columns=["I"],
                            groups_per_batch=8)
    b = next(iter(batcher))
    flat = b.unpack("I")
    # obs_idx maps back to original rows; I should match
    assert torch.allclose(flat, sample_core.I[b.obs_idx])


def test_batch_by_image(sample_core):
    # crystal 0 -> image 0, crystal 1 -> image 1; obs per image = [3, 2]
    batcher = StreamBatcher(sample_core, group_by="image", obs_columns=["I"],
                            groups_per_batch=8)
    b = next(iter(batcher))
    assert torch.equal(b.n_per_group, torch.tensor([3, 2]))


def test_group_columns_gathered(sample_core):
    batcher = StreamBatcher(sample_core, group_by="crystal",
                            obs_columns=["I"], group_columns=("cell",),
                            groups_per_batch=8)
    b = next(iter(batcher))
    assert b.group_columns["cell"].shape == (2, 6)
    # both crystals share the same cell in the fixture
    assert torch.allclose(b.group_columns["cell"][0], sample_core.cell[0])


def test_covers_all_observations(sample_core):
    batcher = StreamBatcher(sample_core, group_by="crystal", obs_columns=["I"],
                            groups_per_batch=1)  # one group per batch
    seen = torch.cat([b.obs_idx for b in batcher])
    assert torch.equal(seen.sort().values, torch.arange(sample_core.n_observations))


def test_shuffle_is_deterministic_with_seed(sample_core):
    b1 = list(StreamBatcher(sample_core, obs_columns=["I"], groups_per_batch=1,
                            shuffle=True, seed=0))
    b2 = list(StreamBatcher(sample_core, obs_columns=["I"], groups_per_batch=1,
                            shuffle=True, seed=0))
    ids1 = [int(b.group_id[0]) for b in b1]
    ids2 = [int(b.group_id[0]) for b in b2]
    assert ids1 == ids2
