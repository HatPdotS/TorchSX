"""Tests for the ragged↔padded packing utility."""

import pytest
import torch

from torchsx.nn.pad import group_offsets_from_counts, pack, unpack


def test_group_offsets_from_counts():
    counts = torch.tensor([3, 1, 2])
    off = group_offsets_from_counts(counts)
    assert torch.equal(off, torch.tensor([0, 3, 4, 6]))


@pytest.mark.parametrize("trailing", [(), (3,), (8,), (21, 21)])
def test_pack_unpack_roundtrip(trailing):
    counts = torch.tensor([3, 1, 4, 2])
    offsets = group_offsets_from_counts(counts)
    n = int(offsets[-1])
    flat = torch.randn(n, *trailing)
    padded, mask = pack(flat, offsets)
    assert padded.shape == (4, 4, *trailing)   # L_max = 4
    assert mask.shape == (4, 4)
    # mask: True only at pad positions
    assert mask.sum().item() == (4 * 4) - n
    recovered = unpack(padded, offsets)
    assert torch.allclose(recovered, flat)


def test_padding_mask_matches_counts():
    counts = torch.tensor([2, 0, 3])
    offsets = group_offsets_from_counts(counts)
    flat = torch.arange(5, dtype=torch.float32).reshape(5, 1)
    padded, mask = pack(flat, offsets)
    # row counts == number of unmasked entries per group
    unmasked = (~mask).sum(dim=1)
    assert torch.equal(unmasked, counts)


def test_max_len_cap_drops_overflow():
    counts = torch.tensor([5, 1])
    offsets = group_offsets_from_counts(counts)
    flat = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    padded, mask = pack(flat, offsets, max_len=3)
    assert padded.shape == (2, 3, 1)
    # group 0 capped to 3 valid, group 1 has 1 valid
    assert torch.equal((~mask).sum(dim=1), torch.tensor([3, 1]))


def test_values_placed_correctly():
    counts = torch.tensor([2, 1])
    offsets = group_offsets_from_counts(counts)
    flat = torch.tensor([[10.0], [11.0], [20.0]])
    padded, mask = pack(flat, offsets)
    assert padded[0, 0, 0] == 10.0
    assert padded[0, 1, 0] == 11.0
    assert padded[1, 0, 0] == 20.0
    assert padded[1, 1, 0] == 0.0  # pad
    assert mask[1, 1].item() is True
