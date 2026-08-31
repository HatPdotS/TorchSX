"""Tests for the SelfConsistentTarget."""

import math

import torch
import pytest

from torchsx.deprecated.scaling import AbsoluteScale, BScale, ScalerInputs
from torchsx.deprecated.target import SelfConsistentTarget


# ---------------------------------------------------------------------------
# Synthetic data helpers (reused from test_kl_target)
# ---------------------------------------------------------------------------

_SIGMA_LN = 1.30
_SIGMA_NORMAL = 50.0


def _make_synthetic_inputs(
    n_crystals: int = 80,
    n_unique: int = 300,
    obs_per_crystal: int = 120,
    n_shells: int = 10,
    sigma_LN: float = _SIGMA_LN,
    sigma_normal: float = _SIGMA_NORMAL,
    include_B: bool = False,
    allow_negative: bool = False,
    seed: int = 123,
):
    gen = torch.Generator().manual_seed(seed)
    n_obs = n_crystals * obs_per_crystal

    true_log_G = torch.randn(n_crystals, generator=gen) * 0.3
    true_log_G -= true_log_G.mean()
    true_log_F = torch.randn(n_unique, generator=gen) * 0.8 + 4.0

    true_B = None
    if include_B:
        true_B = torch.randn(n_crystals, generator=gen).abs() * 10.0

    crystal_id = torch.arange(n_crystals).repeat_interleave(obs_per_crystal).int()
    hkl_id = torch.randint(0, n_unique, (n_obs,), generator=gen)
    shell = torch.randint(0, n_shells, (n_obs,), generator=gen)
    s_sq = (shell.float() + 1) * 0.002

    log_I_true = 2.0 * true_log_F[hkl_id] + true_log_G[crystal_id.long()]
    if include_B:
        log_I_true = log_I_true - 2.0 * true_B[crystal_id.long()] * s_sq

    I_obs = torch.exp(log_I_true) * torch.exp(
        torch.randn(n_obs, generator=gen) * sigma_LN
    )
    I_obs = I_obs + torch.randn(n_obs, generator=gen) * sigma_normal
    if not allow_negative:
        I_obs = I_obs.clamp(min=1.0)

    # Simulate background ~ Poisson-like
    I_bg = torch.abs(torch.randn(n_obs, generator=gen) * 10.0 + 20.0)

    unique_hkl = torch.zeros(n_unique, 3, dtype=torch.int16)

    inputs = ScalerInputs(
        hkl_id=hkl_id,
        crystal_id=crystal_id,
        I_obs=I_obs,
        s_sq=s_sq,
        shell=shell,
        n_unique=n_unique,
        n_crystals=n_crystals,
        n_shells=n_shells,
        unique_hkl=unique_hkl,
        I_bg=I_bg,
    )
    if include_B:
        return inputs, true_log_G, true_log_F, true_B
    return inputs, true_log_G, true_log_F


def _corrcoef(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float()
    b = b.detach().float()
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm())).item()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_finite_positive():
    """Scalar, finite, has grad for I_true."""
    inp, _, _ = _make_synthetic_inputs(n_crystals=20, n_unique=50, obs_per_crystal=30, seed=1)
    target = SelfConsistentTarget(inp.n_unique, inp.n_shells)
    total_scale = torch.ones(inp.I_obs.shape[0])

    loss = target(inp.I_obs, total_scale, inp.hkl_id, inp.I_bg)

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert target.log_I_true.grad is not None


def test_self_regularizing():
    """Loss at wrong I_true > loss at correct (calibrated) I_true."""
    inp, _, true_log_F = _make_synthetic_inputs(
        n_crystals=20, n_unique=50, obs_per_crystal=30, seed=2,
    )
    total_scale = torch.ones(inp.I_obs.shape[0])

    target = SelfConsistentTarget(inp.n_unique, inp.n_shells)
    target.calibrate(inp.I_obs, total_scale, inp.hkl_id, inp.shell, inp.crystal_id)
    loss_calibrated = target(inp.I_obs, total_scale, inp.hkl_id, inp.I_bg).item()

    with torch.no_grad():
        target.log_I_true.fill_(0.0)
    loss_wrong = target(inp.I_obs, total_scale, inp.hkl_id, inp.I_bg).item()

    assert loss_wrong > loss_calibrated


def test_calibrate_initializes():
    """After calibrate, sigma_LN_prior is set and shell_per_hkl assigned."""
    inp, _, _ = _make_synthetic_inputs(n_crystals=40, n_unique=100, obs_per_crystal=60, seed=3)
    target = SelfConsistentTarget(inp.n_unique, inp.n_shells)
    total_scale = torch.ones(inp.I_obs.shape[0])

    target.calibrate(inp.I_obs, total_scale, inp.hkl_id, inp.shell, inp.crystal_id)

    # Priors should be set to non-default values
    assert not torch.allclose(target.sigma_LN_prior, torch.full_like(target.sigma_LN_prior, 0.1))
    # shell_per_hkl should have a range of shells
    assert target.shell_per_hkl.max() > 0


def test_sigma_not_parameter():
    """list(target.parameters()) has only I_true — no sigma params."""
    target = SelfConsistentTarget(100, 10)
    param_names = [name for name, _ in target.named_parameters()]
    assert param_names == ["log_I_true"], f"Unexpected params: {param_names}"
    # sigma_LN_prior should be a buffer
    buf_names = [name for name, _ in target.named_buffers()]
    assert "sigma_LN_prior" in buf_names
    assert "shell_per_hkl" in buf_names


def test_recover_scale_factors():
    """Optimizer recovers known per-crystal G and |F|^2."""
    inp, true_log_G, true_log_F = _make_synthetic_inputs(seed=10)

    abs_scale = AbsoluteScale(inp.n_crystals)
    target = SelfConsistentTarget(inp.n_unique, inp.n_shells)
    total_scale = torch.ones(inp.I_obs.shape[0])
    target.calibrate(inp.I_obs, total_scale, inp.hkl_id, inp.shell, inp.crystal_id)

    params = list(abs_scale.parameters()) + list(target.parameters())
    optimizer = torch.optim.Adam(params, lr=0.005)

    for step in range(600):
        optimizer.zero_grad()
        log_scale = abs_scale.log_scale_per_obs(inp.crystal_id)
        ts = torch.exp(log_scale)
        loss = target(inp.I_obs, ts, inp.hkl_id, inp.I_bg)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            abs_scale.log_G.data -= abs_scale.log_G.data.mean()

    cc_G = _corrcoef(abs_scale.log_G.data, true_log_G)
    assert cc_G > 0.30, f"log_G correlation too low: {cc_G:.3f}"

    cc_F = _corrcoef(target.log_I_true.data / 2.0, true_log_F)
    assert cc_F > 0.95, f"log_F correlation too low: {cc_F:.3f}"


def test_recover_with_bscale():
    """Joint G + B recovery."""
    inp, true_log_G, true_log_F, true_B = _make_synthetic_inputs(
        include_B=True, seed=456,
    )

    abs_scale = AbsoluteScale(inp.n_crystals)
    b_scale = BScale(inp.n_crystals)
    target = SelfConsistentTarget(inp.n_unique, inp.n_shells)
    total_scale = torch.ones(inp.I_obs.shape[0])
    target.calibrate(inp.I_obs, total_scale, inp.hkl_id, inp.shell, inp.crystal_id)

    params = (
        list(abs_scale.parameters())
        + list(b_scale.parameters())
        + list(target.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=0.005)

    for step in range(800):
        optimizer.zero_grad()
        log_scale = (
            abs_scale.log_scale_per_obs(inp.crystal_id)
            + b_scale.log_scale_per_obs(inp.crystal_id, inp.s_sq)
        )
        ts = torch.exp(log_scale)
        loss = target(inp.I_obs, ts, inp.hkl_id, inp.I_bg)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            abs_scale.log_G.data -= abs_scale.log_G.data.mean()

    # Total scale at median s^2
    s_sq_median = inp.s_sq.median()
    true_total = true_log_G - 2.0 * true_B * s_sq_median
    recovered_total = abs_scale.log_G.data - 2.0 * b_scale.B.data * s_sq_median
    cc_total = _corrcoef(recovered_total, true_total)
    assert cc_total > 0.35, f"Total scale correlation too low: {cc_total:.3f}"

    cc_F = _corrcoef(target.log_I_true.data / 2.0, true_log_F)
    assert cc_F > 0.95, f"log_F correlation too low: {cc_F:.3f}"


def test_negative_obs_finite():
    """Loss is finite and differentiable with negative observations."""
    n_unique = 20
    gen = torch.Generator().manual_seed(42)
    hkl_id = torch.randint(0, n_unique, (100,), generator=gen)
    I_obs = torch.randn(100, generator=gen) * 50.0
    I_bg = torch.ones(100) * 10.0
    total_scale = torch.ones(100, requires_grad=True)

    target = SelfConsistentTarget(n_unique, n_shells=3)
    loss = target(I_obs, total_scale, hkl_id, I_bg)

    assert loss.dim() == 0
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    loss.backward()
    assert target.log_I_true.grad is not None
    assert torch.isfinite(target.log_I_true.grad).all()


def test_weighted_loss():
    """Loss is finite with weight and uniform weight matches unweighted."""
    inp, _, _ = _make_synthetic_inputs(n_crystals=20, n_unique=50, obs_per_crystal=30, seed=80)
    target = SelfConsistentTarget(inp.n_unique, inp.n_shells)
    total_scale = torch.ones(inp.I_obs.shape[0])
    n_obs = inp.I_obs.shape[0]

    # Weighted loss should be finite
    weight = torch.ones(n_obs)
    loss_w = target(inp.I_obs, total_scale, inp.hkl_id, inp.I_bg, weight=weight)
    assert loss_w.dim() == 0
    assert torch.isfinite(loss_w)

    # Uniform weight should match unweighted
    loss_none = target(inp.I_obs, total_scale, inp.hkl_id, inp.I_bg, weight=None).item()
    loss_uniform = target(inp.I_obs, total_scale, inp.hkl_id, inp.I_bg, weight=weight).item()
    assert abs(loss_none - loss_uniform) < 1e-5 * abs(loss_none), (
        f"Uniform weight should match unweighted: {loss_none:.6f} vs {loss_uniform:.6f}"
    )


def test_to_dict():
    """to_dict returns expected keys with correct shapes."""
    n_unique = 50
    n_shells = 5
    target = SelfConsistentTarget(n_unique, n_shells)

    d = target.to_dict()
    assert "I" in d
    assert "F" in d
    assert "sigma_LN" in d
    assert d["I"].shape == (n_unique,)
    assert d["F"].shape == (n_unique,)
    assert d["sigma_LN"].shape == (n_unique,)
