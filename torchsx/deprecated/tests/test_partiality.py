"""Tests for EwaldPartiality module."""

import math

import pytest
import torch

from torchsx.deprecated.scaling._partiality import EwaldPartiality


@pytest.fixture
def simple_setup():
    """Simple cubic crystal, λ=1Å, UB=identity."""
    n_crystals = 2
    UB = torch.eye(3).unsqueeze(0).expand(n_crystals, -1, -1).clone()
    wavelength = 1.0
    return n_crystals, UB, wavelength


def test_rodrigues_identity(simple_setup):
    """δω=0 → RUB=UB (no rotation, identity UB)."""
    n_crystals, UB, wl = simple_setup
    model = EwaldPartiality(n_crystals, UB, wl)
    RUB = model._all_RUB()
    assert RUB.shape == (2, 3, 3)
    # UB=I, delta_omega=0 → R=I → RUB=I
    torch.testing.assert_close(RUB, torch.eye(3).expand(2, -1, -1), atol=1e-6, rtol=0)


def test_rodrigues_small_angle(simple_setup):
    """Small δω ≈ I + [δω]×, so RUB ≈ (I + [δω]×) @ UB."""
    n_crystals, UB, wl = simple_setup
    model = EwaldPartiality(n_crystals, UB, wl)
    eps = 1e-4
    with torch.no_grad():
        model.delta_omega[0] = torch.tensor([eps, 0.0, 0.0])
    RUB = model._all_RUB()
    # UB=I, so RUB[0] ≈ I + [eps,0,0]× = [[0,0,0],[0,0,-eps],[0,eps,0]] + I
    expected = torch.eye(3)
    expected[1, 2] = -eps
    expected[2, 1] = eps
    torch.testing.assert_close(RUB[0], expected, atol=1e-6, rtol=0)


def test_rodrigues_finite_angle(simple_setup):
    """~9.7° rotation about z-axis (within ±0.17 rad clamp)."""
    n_crystals, UB, wl = simple_setup
    model = EwaldPartiality(n_crystals, UB, wl)
    angle = 0.17  # ~9.7°
    with torch.no_grad():
        model.delta_omega[0] = torch.tensor([0.0, 0.0, angle])
    RUB = model._all_RUB()
    # UB=I, so RUB[0] = R_z(angle)
    c, s = math.cos(angle), math.sin(angle)
    expected = torch.tensor([[c, -s, 0.0],
                             [s, c, 0.0],
                             [0.0, 0.0, 1.0]])
    torch.testing.assert_close(RUB[0], expected, atol=1e-5, rtol=0)


def test_excitation_error_on_ewald(simple_setup):
    """Reflection exactly on Ewald sphere has r≈0."""
    n_crystals, UB, wl = simple_setup
    model = EwaldPartiality(n_crystals, UB, wl)
    # For λ=1Å, s0=[0,0,1]. A reflection at hkl=[1,0,0] gives
    # q = UB @ h = [1,0,0], q+s0 = [1,0,1], |q+s0| = sqrt(2) ≠ 1/λ=1
    # So [1,0,0] is NOT on the sphere. Need to find one that is.
    # On sphere: |q + s0|² = (1/λ)² → |q|² + 2 q·s0 + |s0|² = 1/λ²
    # → |q|² + 2 qz/λ = 0 → qz = -|q|²λ/2
    # For q = [qx, 0, qz]: qx² + qz² + 2qz = 0
    # Let qx = 0.5: 0.25 + qz² + 2qz = 0 → qz = -1 ± sqrt(1-0.25) = -1 + 0.866 = -0.134
    qx = 0.5
    qz = -1.0 + math.sqrt(1.0 - qx**2)
    hkl_obs = torch.tensor([[qx, 0.0, qz]])
    crystal_id = torch.tensor([0])
    r = model.excitation_error(hkl_obs, crystal_id)
    assert r.abs().item() < 1e-5


def test_excitation_error_gradient(simple_setup):
    """Gradients flow through δω."""
    n_crystals, UB, wl = simple_setup
    model = EwaldPartiality(n_crystals, UB, wl)
    hkl_obs = torch.tensor([[1.0, 0.0, 0.0]])
    crystal_id = torch.tensor([0])
    r = model.excitation_error(hkl_obs, crystal_id)
    r.sum().backward()
    assert model.delta_omega.grad is not None
    assert model.delta_omega.grad.abs().sum() > 0


def test_log_partiality_range(simple_setup):
    """log_p ∈ [log(p_min), 0]."""
    n_crystals, UB, wl = simple_setup
    p_min = 0.01
    model = EwaldPartiality(n_crystals, UB, wl, p_min=p_min)
    hkl_obs = torch.tensor([[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.1]])
    crystal_id = torch.tensor([0, 0, 0])
    log_p = model.log_partiality(hkl_obs, crystal_id)
    assert (log_p <= 0.0 + 1e-6).all()
    assert (log_p >= math.log(p_min) - 1e-6).all()


def test_log_partiality_gaussian_shape(simple_setup):
    """log_p is quadratic in excitation error (Gaussian shape)."""
    n_crystals, UB, wl = simple_setup
    model = EwaldPartiality(n_crystals, UB, wl)
    # Two reflections with different excitation errors
    # Use fractional hkl to control q precisely
    h1 = torch.tensor([[0.1, 0.0, 0.0]])
    h2 = torch.tensor([[0.2, 0.0, 0.0]])
    cid = torch.tensor([0])
    r1 = model.excitation_error(h1, cid).item()
    r2 = model.excitation_error(h2, cid).item()
    lp1 = model.log_partiality(h1, cid).item()
    lp2 = model.log_partiality(h2, cid).item()
    # log_p = -r²/(2σ²), so log_p1/log_p2 ≈ r1²/r2² (if not clamped)
    if lp1 > model.log_p_min and lp2 > model.log_p_min:
        ratio_r = r1**2 / r2**2
        ratio_lp = lp1 / lp2
        assert abs(ratio_r - ratio_lp) < 0.01


def test_ub_construction():
    """UB from astar/bstar/cstar gives correct q = UB @ h."""
    astar = torch.tensor([[1.0, 0.0, 0.0]])
    bstar = torch.tensor([[0.0, 2.0, 0.0]])
    cstar = torch.tensor([[0.0, 0.0, 3.0]])
    UB = torch.stack([astar, bstar, cstar], dim=-1)  # (1, 3, 3)
    h = torch.tensor([1.0, 1.0, 1.0])
    q = (UB[0] @ h)
    torch.testing.assert_close(q, torch.tensor([1.0, 2.0, 3.0]))


def test_log_scale_per_obs_matches_log_partiality(simple_setup):
    """log_scale_per_obs returns same as log_partiality."""
    n_crystals, UB, wl = simple_setup
    model = EwaldPartiality(n_crystals, UB, wl)
    hkl_obs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    crystal_id = torch.tensor([0, 1])
    lp = model.log_partiality(hkl_obs, crystal_id)
    ls = model.log_scale_per_obs(hkl_obs, crystal_id)
    torch.testing.assert_close(lp, ls)


def test_resolution_dependent_width(simple_setup):
    """Higher s_sq gives broader relp → higher partiality for same r."""
    n_crystals, UB, wl = simple_setup
    model = EwaldPartiality(n_crystals, UB, wl, sigma_0=0.01, B_relp=1.0)
    hkl_obs = torch.tensor([[1.0, 0.0, 0.0]])
    crystal_id = torch.tensor([0])
    # Same reflection, different resolution contexts
    s_sq_low = torch.tensor([0.01])  # low resolution
    s_sq_high = torch.tensor([1.0])  # high resolution
    lp_low = model.log_partiality(hkl_obs, crystal_id, s_sq=s_sq_low)
    lp_high = model.log_partiality(hkl_obs, crystal_id, s_sq=s_sq_high)
    # Higher s_sq → broader σ_eff → less negative log_p (higher partiality)
    assert lp_high > lp_low or (lp_high == lp_low == model.log_p_min)
