"""Unit tests for the scale-model subpackage."""

from __future__ import annotations

import math

import torch
from torchsx.merging.scatter_shape import FreePerHKLShape

from torchsx.merging.observations import ObsInfo
from torchsx.merging.parameters import Parameters
from torchsx.merging.scale import (
    ScaleBPerCrystal,
    PanelScale,
    FixedScale,
    CompositeScale,
)


def _params(n_hkl: int = 4) -> Parameters:
    return Parameters(
        log_I0_init=torch.zeros(n_hkl),
        scatter_shape=FreePerHKLShape(torch.zeros(n_hkl), torch.zeros(n_hkl)),
    )


def _meta(
    n_obs: int, n_crystals: int, *, with_panels: bool = False
) -> ObsInfo:
    cid = torch.arange(n_obs) % n_crystals
    return ObsInfo(
        crystal_id=cid.long(),
        s_sq=torch.linspace(0.0, 0.2, n_obs),
        hkl_idx=torch.zeros(n_obs, dtype=torch.long),
        n_crystals=n_crystals,
        panel_id=torch.zeros(n_obs, dtype=torch.long) if with_panels else None,
        fs=torch.zeros(n_obs) if with_panels else None,
        ss=torch.zeros(n_obs) if with_panels else None,
    )


# ─── ScaleBPerCrystal ───────────────────────────────────────────────────


def test_scale_b_centring_preserves_zero_mean() -> None:
    """The log_G view must always have mean zero, even after params change."""
    sc = ScaleBPerCrystal(n_crystals=4)
    with torch.no_grad():
        sc.log_G_raw.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    lg = sc._log_G_centred()
    assert abs(lg.mean().item()) < 1e-6


def test_scale_b_log_k_matches_manual() -> None:
    n_obs, n_c = 6, 3
    sc = ScaleBPerCrystal(n_crystals=n_c)
    with torch.no_grad():
        sc.log_G_raw.copy_(torch.tensor([0.5, -0.2, 0.1]))
        sc.B.copy_(torch.tensor([0.0, 10.0, -5.0]))
    meta = _meta(n_obs, n_c)
    params = _params()
    got = sc.log_k(params, meta)
    # Manual
    lg = sc._log_G_centred()
    exp = lg[meta.crystal_id] - 2.0 * sc.B[meta.crystal_id] * meta.s_sq
    assert torch.allclose(got, exp, atol=1e-6)


def test_scale_b_regularizer_matches_manual() -> None:
    sc = ScaleBPerCrystal(n_crystals=3, scale_prior=2.0, bfactor_prior=0.5)
    with torch.no_grad():
        sc.log_G_raw.copy_(torch.tensor([1.0, 2.0, 3.0]))
        sc.B.copy_(torch.tensor([0.0, 1.0, 2.0]))
    params = _params()
    got = sc.regularizer(params)
    lg = sc._log_G_centred()
    exp = 2.0 * (lg * lg).sum() + 0.5 * (sc.B * sc.B).sum()
    assert torch.isclose(got, exp, atol=1e-6)


def test_scale_b_gradient_flows() -> None:
    sc = ScaleBPerCrystal(n_crystals=3)
    meta = _meta(9, 3)
    params = _params()
    loss = sc.log_k(params, meta).sum() + sc.regularizer(params)
    loss.backward()
    assert sc.log_G_raw.grad is not None
    assert sc.B.grad is not None
    assert torch.isfinite(sc.log_G_raw.grad).all()
    assert torch.isfinite(sc.B.grad).all()


# ─── PanelScale ─────────────────────────────────────────────────────────


def test_panel_scale_basic() -> None:
    pan = PanelScale(n_panels=2)
    with torch.no_grad():
        pan.gain.copy_(torch.tensor([0.3, -0.7]))
    meta = _meta(4, 2, with_panels=True)
    with torch.no_grad():
        # Alternate panel ids to check indexing.
        meta.panel_id.copy_(torch.tensor([0, 1, 0, 1], dtype=torch.long))
    params = _params()
    got = pan.log_k(params, meta)
    assert torch.allclose(got, torch.tensor([0.3, -0.7, 0.3, -0.7]), atol=1e-6)


def test_panel_scale_with_gradient() -> None:
    pan = PanelScale(n_panels=1, fit_gradient=True)
    with torch.no_grad():
        pan.gain.copy_(torch.tensor([0.0]))
        pan.grad_fs.copy_(torch.tensor([2.0]))
        pan.grad_ss.copy_(torch.tensor([-1.0]))
    n_obs = 3
    meta = ObsInfo(
        crystal_id=torch.zeros(n_obs, dtype=torch.long),
        s_sq=torch.zeros(n_obs),
        hkl_idx=torch.zeros(n_obs, dtype=torch.long),
        n_crystals=1,
        panel_id=torch.zeros(n_obs, dtype=torch.long),
        fs=torch.tensor([1.0, 2.0, 3.0]),
        ss=torch.tensor([0.0, 1.0, 2.0]),
    )
    got = pan.log_k(_params(), meta)
    # 2·fs − ss
    assert torch.allclose(got, torch.tensor([2.0, 3.0, 4.0]), atol=1e-6)


# ─── FixedScale ─────────────────────────────────────────────────────────


def test_fixed_scale_returns_slice() -> None:
    log_c = torch.linspace(0.0, 1.0, 10)
    fx = FixedScale(log_c)
    meta = _meta(10, 1)
    # Full range
    assert torch.equal(fx.log_k(_params(), meta), log_c)
    # Subrange via obs_range
    got = fx.log_k(_params(), meta.slice(2, 6), obs_range=(2, 6))
    assert torch.equal(got, log_c[2:6])


# ─── CompositeScale ─────────────────────────────────────────────────────


def test_composite_scale_sums_submodules() -> None:
    sc = ScaleBPerCrystal(n_crystals=2)
    with torch.no_grad():
        sc.log_G_raw.copy_(torch.tensor([0.5, -0.5]))
    log_c = torch.tensor([0.1, 0.2, 0.3, 0.4])
    fx = FixedScale(log_c)
    comp = CompositeScale([sc, fx])
    meta = _meta(4, 2)
    params = _params()

    got = comp.log_k(params, meta)
    exp = sc.log_k(params, meta) + fx.log_k(params, meta)
    assert torch.allclose(got, exp, atol=1e-6)

    reg_got = comp.regularizer(params)
    reg_exp = sc.regularizer(params) + fx.regularizer(params)
    assert torch.isclose(reg_got, reg_exp, atol=1e-6)


def test_composite_scale_empty() -> None:
    comp = CompositeScale()
    meta = _meta(5, 1)
    got = comp.log_k(_params(), meta)
    assert torch.equal(got, torch.zeros(5))
    assert torch.isclose(comp.regularizer(_params()), torch.tensor(0.0))
