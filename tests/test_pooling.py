"""Group reductions and the fold assignment."""
from __future__ import annotations

import torch

from torchsx.nn.pooling import fold_ids, group_count, group_sum


def test_group_sum_and_count_handle_empty_groups():
    gid, n_groups = torch.tensor([0, 0, 0, 1, 2, 2, 4, 4, 4, 4]), 5   # group 3 empty, 1 singleton
    x = torch.ones(gid.shape[0])
    assert group_sum(x, gid, n_groups).tolist() == [3.0, 1.0, 2.0, 0.0, 4.0]
    assert group_count(gid, n_groups).tolist() == [3, 1, 2, 0, 4]


def test_fold_ids_depend_only_on_within_group_order():
    """Grouped vs interleaved presentation must give identical folds per row."""
    torch.manual_seed(3)
    gid = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2])
    perm = torch.randperm(gid.numel())
    # rows of one group keep their relative order under a stable sort of perm
    inv = torch.argsort(gid[perm], stable=True)
    f_grouped = fold_ids(gid, 3, 2)
    f_shuffled = fold_ids(gid[perm], 3, 2)
    # map shuffled labels back to grouped order
    assert torch.equal(f_shuffled[inv], f_grouped)
    # balanced to within one per group
    for g in range(3):
        c = torch.bincount(f_grouped[gid == g], minlength=2)
        assert int(c.max() - c.min()) <= 1
