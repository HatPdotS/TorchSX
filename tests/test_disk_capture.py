"""Disk capture: a known log-scale offset per observation from its disk."""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from tests.test_amortized_scale import _toy_prep
from torchsx.merging.scale import AmortizedScale, DiskCapture


def _capture() -> DiskCapture:
    return DiskCapture(q=[0.2, 0.4], n_pixels=[21, 37],
                       log_ratio=[[math.log(1.2), math.log(1.6)], [math.log(1.5), math.log(2.5)]],
                       base_pixels=9)


def test_log_offset_follows_disk_and_resolution():
    cap = _capture()
    q = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.9])
    s_sq = (q / 2.0) ** 2
    base = cap.log_offset(s_sq, torch.full((5,), 9.0))
    assert base.tolist() == [0.0] * 5
    mid = cap.log_offset(s_sq, torch.full((5,), 21.0)).exp()
    # constant beyond the knots, linear in the log between them
    assert mid.tolist() == pytest.approx([1.2, 1.2, math.sqrt(1.2 * 1.6), 1.6, 1.6], rel=1e-5)
    # a disk that lost pixels belongs to the smallest disk that holds its count
    n_pix = torch.tensor([5.0, 10.0, 21.0, 22.0, 80.0])
    got = cap.log_offset(torch.full((5,), 0.01), n_pix).exp()
    assert got.tolist() == pytest.approx([1.0, 1.2, 1.2, 1.5, 1.5], rel=1e-5)


def test_every_disk_can_carry_a_row():
    # reference = the largest disk: base_pixels 0, values at or below zero
    cap = DiskCapture(q=[0.2, 0.4], n_pixels=[9, 21, 37],
                      log_ratio=[[math.log(0.6), math.log(0.3)], [math.log(0.8), math.log(0.6)], [0.0, 0.0]],
                      base_pixels=0)
    s_sq = torch.full((5,), (0.3 / 2.0) ** 2)
    n_pix = torch.tensor([0.0, 7.0, 9.0, 21.0, 37.0])
    got = cap.log_offset(s_sq, n_pix).exp()
    # an observation without a recorded disk gets no offset
    assert got.tolist() == pytest.approx([1.0, math.sqrt(0.18), math.sqrt(0.18), math.sqrt(0.48), 1.0], rel=1e-5)


def test_definition_round_trip(tmp_path):
    cap = _capture()
    again = DiskCapture.from_definition(cap.definition())
    s_sq, n_pix = torch.tensor([0.02, 0.03]), torch.tensor([21.0, 37.0])
    torch.testing.assert_close(again.log_offset(s_sq, n_pix), cap.log_offset(s_sq, n_pix))
    with pytest.raises(ValueError):
        DiskCapture(q=[0.2, 0.4], n_pixels=[9], log_ratio=[[0.0, 0.0]], base_pixels=9)


def test_scale_adds_the_offset_after_the_anchor():
    prep = _toy_prep()
    n = prep.obs_info.hkl_idx.numel()
    n_pix = torch.tensor([9.0, 21.0, 37.0])[torch.arange(n) % 3]
    prep = dataclasses.replace(prep, obs_info=dataclasses.replace(prep.obs_info, n_pix=n_pix))
    torch.manual_seed(1)
    scale = AmortizedScale.from_prep(prep, d_model=16, context_folds=2, n_heads=2).eval()
    meta = prep.obs_info
    with torch.no_grad():
        plain = scale.predict(meta)
        scale.capture = _capture()
        pred = scale.predict(meta)
    offset = scale.capture.log_offset(meta.s_sq, meta.n_pix)
    assert plain.log_offset is None
    torch.testing.assert_close(pred.log_offset, offset)
    # the network's own scale is unchanged, and it is what the anchor pinned
    torch.testing.assert_close(pred.log_K_net, plain.log_K, atol=1e-6, rtol=0)
    torch.testing.assert_close(pred.log_K, plain.log_K + offset, atol=1e-6, rtol=0)
    assert float(offset[n_pix == 9.0].abs().max()) == 0.0
    assert float(offset[n_pix == 37.0].min()) > 0.0
