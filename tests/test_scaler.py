"""Tests for the scaling/merge pipeline."""

import math

import torch
import pytest

from torchsx.scaling import (
    AbsoluteScale,
    BScale,
    CompositeScalingLoss,
    ScalerInputs,
    prepare_scaler_inputs,
)
from torchsx.merge import MergedStructureFactor


# ---------------------------------------------------------------------------
# ScalerInputs
# ---------------------------------------------------------------------------

def test_prepare_scaler_inputs(sample_core):
    """shapes, s_sq > 0, shell in range"""
    inp = prepare_scaler_inputs(sample_core, n_shells=5)

    assert inp.hkl_id.shape == (sample_core.n_observations,)
    assert inp.crystal_id.shape == (sample_core.n_observations,)
    assert inp.I_obs.shape == (sample_core.n_observations,)
    assert inp.s_sq.shape == (sample_core.n_observations,)
    assert inp.shell.shape == (sample_core.n_observations,)
    assert (inp.s_sq > 0).all()
    assert (inp.shell >= 0).all()
    assert (inp.shell < inp.n_shells).all()
    assert inp.n_unique > 0
    assert inp.n_crystals == sample_core.n_crystals
    assert inp.n_shells == 5


# ---------------------------------------------------------------------------
# AbsoluteScale
# ---------------------------------------------------------------------------

def test_absolute_scale_init(sample_core):
    """log_G shape, default = 0"""
    scaler = AbsoluteScale.from_core(sample_core)
    assert scaler.log_G.shape == (sample_core.n_crystals,)
    assert (scaler.log_G == 0).all()


def test_absolute_scale_forward(sample_core):
    """I unchanged when log_G = 0"""
    scaler = AbsoluteScale.from_core(sample_core)
    scaled = scaler(sample_core)
    assert torch.allclose(scaled.I, sample_core.I)


def test_absolute_scale_applies(sample_core):
    """set log_G=1, verify I_scaled = I/e"""
    scaler = AbsoluteScale.from_core(sample_core)
    with torch.no_grad():
        scaler.log_G.fill_(1.0)
    scaled = scaler(sample_core)
    expected = sample_core.I / math.e
    assert torch.allclose(scaled.I, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# BScale
# ---------------------------------------------------------------------------

def test_bscale_forward(sample_core):
    """I unchanged when B = 0"""
    scaler = BScale.from_core(sample_core)
    inp = prepare_scaler_inputs(sample_core, n_shells=5)
    scaled = scaler(sample_core, inp.s_sq)
    assert torch.allclose(scaled.I, sample_core.I)


# ---------------------------------------------------------------------------
# MergedStructureFactor
# ---------------------------------------------------------------------------

def test_merged_sf_from_merge(sample_core):
    """log_F initialized, predict_I > 0"""
    sf = MergedStructureFactor.from_merge(sample_core)
    assert sf.log_F.shape[0] > 0
    inp = prepare_scaler_inputs(sample_core, n_shells=5)
    I_pred = sf.predict_I(inp.hkl_id)
    assert (I_pred > 0).all()


def test_merged_sf_predict(sample_core):
    """I = exp(2 * log_F)"""
    sf = MergedStructureFactor.from_merge(sample_core)
    inp = prepare_scaler_inputs(sample_core, n_shells=5)
    log_I = sf.predict_log_I(inp.hkl_id)
    I_pred = sf.predict_I(inp.hkl_id)
    assert torch.allclose(I_pred, torch.exp(log_I), rtol=1e-5)


# ---------------------------------------------------------------------------
# CompositeScalingLoss
# ---------------------------------------------------------------------------

def test_loss_finite_positive(sample_core):
    """scalar, finite, has grad"""
    inp = prepare_scaler_inputs(sample_core, n_shells=5)
    sf = MergedStructureFactor.from_merge(sample_core)
    loss_fn = CompositeScalingLoss(n_shells=5)

    log_I_pred = sf.predict_log_I(inp.hkl_id)
    I_pred = torch.exp(log_I_pred)
    loss = loss_fn(inp.I_obs, I_pred, log_I_pred, inp.shell)

    assert loss.dim() == 0  # scalar
    assert torch.isfinite(loss)
    loss.backward()
    assert sf.log_F.grad is not None


def test_loss_decreases(sample_core):
    """50 Adam steps -> loss drops"""
    inp = prepare_scaler_inputs(sample_core, n_shells=5)
    abs_scale = AbsoluteScale.from_core(sample_core)
    sf = MergedStructureFactor.from_merge(sample_core)
    loss_fn = CompositeScalingLoss(n_shells=5)

    # Perturb log_G so initial loss is non-trivial
    with torch.no_grad():
        abs_scale.log_G.copy_(torch.randn_like(abs_scale.log_G) * 0.5)

    params = list(abs_scale.parameters()) + list(sf.parameters())
    optimizer = torch.optim.Adam(params, lr=0.01)

    initial_loss = None
    for step in range(50):
        optimizer.zero_grad()
        log_scale = abs_scale.log_scale_per_obs(inp.crystal_id)
        log_I_pred = sf.predict_log_I(inp.hkl_id) + log_scale
        I_pred = torch.exp(log_I_pred)
        loss = loss_fn(inp.I_obs, I_pred, log_I_pred, inp.shell)
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()

    assert loss.item() < initial_loss


def test_gaussian_suppressed_at_large_sigma(sample_core):
    """Large sigma_LN suppresses Gaussian via Poisson-fraction weight."""
    inp = prepare_scaler_inputs(sample_core, n_shells=5)
    sf = MergedStructureFactor.from_merge(sample_core)
    loss_fn = CompositeScalingLoss(n_shells=5)

    # Perturb log_F so there's a non-trivial residual
    with torch.no_grad():
        sf.log_F.add_(0.5)

    log_I_pred = sf.predict_log_I(inp.hkl_id)
    I_pred = torch.exp(log_I_pred)

    # Small sigma_LN -> Gaussian dominates (w ≈ 1/I), no strong obs
    loss_fn.sigma_LN.fill_(1e-3)
    loss_small_sigma = loss_fn(inp.I_obs, I_pred, log_I_pred, inp.shell)

    # Large sigma_LN -> Gaussian suppressed (w ≈ 1/(I²σ²)), all strong
    loss_fn.sigma_LN.fill_(100.0)
    loss_large_sigma = loss_fn(inp.I_obs, I_pred, log_I_pred, inp.shell)

    # Gaussian term should be much larger when σ_LN is small
    assert loss_small_sigma.item() > loss_large_sigma.item()


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def test_calibrate_pure_shot():
    """Poisson data -> sigma_LN ~ 0"""
    torch.manual_seed(42)
    n = 10000
    n_shells = 5
    # Poisson-like: I_obs ~ exp(N(mu, 1/I))  i.e. log I ~ N(mu, sigma_shot)
    I_true = torch.full((n,), 1000.0)
    # Pure shot noise: variance in log space = 1/I
    log_I = torch.log(I_true) + torch.randn(n) * (1.0 / I_true).sqrt()
    I_obs = torch.exp(log_I)
    log_scale = torch.zeros(n)
    shells = torch.randint(0, n_shells, (n,))

    loss_fn = CompositeScalingLoss(n_shells=n_shells)
    loss_fn.calibrate(I_obs, log_scale, shells, I_min_abs=1.0)

    # sigma_LN should be small (close to 0)
    assert (loss_fn.sigma_LN < 0.05).all(), f"sigma_LN too large: {loss_fn.sigma_LN}"


def test_calibrate_with_systematic():
    """Injected sigma=0.1 -> sigma_LN ~ 0.1"""
    torch.manual_seed(42)
    n = 50000
    n_shells = 5
    sigma_inject = 0.1
    I_true = torch.full((n,), 1000.0)
    # Shot noise + systematic
    sigma_shot = (1.0 / I_true).sqrt()
    sigma_total = (sigma_shot ** 2 + sigma_inject ** 2).sqrt()
    log_I = torch.log(I_true) + torch.randn(n) * sigma_total
    I_obs = torch.exp(log_I)
    log_scale = torch.zeros(n)
    shells = torch.randint(0, n_shells, (n,))

    loss_fn = CompositeScalingLoss(n_shells=n_shells)
    loss_fn.calibrate(I_obs, log_scale, shells, I_min_abs=1.0)

    # sigma_LN should be close to 0.1
    assert (loss_fn.sigma_LN > 0.05).all(), f"sigma_LN too small: {loss_fn.sigma_LN}"
    assert (loss_fn.sigma_LN < 0.20).all(), f"sigma_LN too large: {loss_fn.sigma_LN}"


# ---------------------------------------------------------------------------
# Functional recovery test — known ground truth
# ---------------------------------------------------------------------------

# Noise model: constant σ_LN (multiplicative/lognormal) and constant
# σ_normal (additive/Gaussian readout).  The dynamic range comes from
# varying |F|².  σ_LN = 1.30 gives a realistic single-crystal CC ≈ 0.55.
_SIGMA_LN = 1.30      # constant lognormal width (partiality + systematics)
_SIGMA_NORMAL = 50.0   # constant additive Gaussian (readout noise)


def _make_synthetic_inputs(
    n_crystals: int = 80,
    n_unique: int = 300,
    obs_per_crystal: int = 120,
    n_shells: int = 10,
    sigma_LN: float = _SIGMA_LN,
    sigma_normal: float = _SIGMA_NORMAL,
    include_B: bool = False,
    seed: int = 123,
):
    """Generate synthetic ScalerInputs with known ground truth.

    Noise model (constant sigmas, varying intensity):
        I_obs = I_true · exp(ε_LN) + ε_normal
    where ε_LN ~ N(0, σ_LN²) and ε_normal ~ N(0, σ_normal²) are
    constant across observations.  The intensity variation comes from
    |F(h)|² spanning ~2 orders of magnitude.

    Returns (inputs, true_log_G, true_log_F [, true_B]).
    """
    gen = torch.Generator().manual_seed(seed)

    n_obs = n_crystals * obs_per_crystal

    # Ground truth
    true_log_G = torch.randn(n_crystals, generator=gen) * 0.3
    true_log_G -= true_log_G.mean()
    true_log_F = torch.randn(n_unique, generator=gen) * 0.8 + 4.0

    true_B = None
    if include_B:
        true_B = torch.randn(n_crystals, generator=gen).abs() * 10.0

    # Assignments
    crystal_id = torch.arange(n_crystals).repeat_interleave(obs_per_crystal).int()
    hkl_id = torch.randint(0, n_unique, (n_obs,), generator=gen)
    shell = torch.randint(0, n_shells, (n_obs,), generator=gen)
    s_sq = (shell.float() + 1) * 0.002

    # Noiseless intensity
    log_I_true = 2.0 * true_log_F[hkl_id] + true_log_G[crystal_id.long()]
    if include_B:
        log_I_true = log_I_true - 2.0 * true_B[crystal_id.long()] * s_sq

    # Constant noise: multiplicative lognormal + additive Gaussian
    I_obs = torch.exp(log_I_true) * torch.exp(
        torch.randn(n_obs, generator=gen) * sigma_LN
    )
    I_obs = I_obs + torch.randn(n_obs, generator=gen) * sigma_normal
    I_obs = I_obs.clamp(min=1.0)

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
    )
    if include_B:
        return inputs, true_log_G, true_log_F, true_B
    return inputs, true_log_G, true_log_F


def _corrcoef(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation between two 1D tensors."""
    a = a.detach().float()
    b = b.detach().float()
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm())).item()


def _init_log_F(hkl_id, I_obs, n_unique):
    """Initialize log_F from per-hkl mean of log(I_obs)."""
    log_I = torch.log(I_obs.clamp(min=1.0))
    sum_log = torch.zeros(n_unique)
    count = torch.zeros(n_unique)
    sum_log.scatter_add_(0, hkl_id, log_I)
    count.scatter_add_(0, hkl_id, torch.ones_like(log_I))
    return sum_log / count.clamp(min=1) / 2.0


def test_single_crystal_cc():
    """Verify synthetic noise gives single-crystal CC ≈ 0.55."""
    inp, true_log_G, true_log_F = _make_synthetic_inputs()
    I_true_per_obs = torch.exp(2.0 * true_log_F[inp.hkl_id])

    ccs = []
    for c in range(inp.n_crystals):
        mask = inp.crystal_id == c
        a = inp.I_obs[mask] - inp.I_obs[mask].mean()
        b = I_true_per_obs[mask] - I_true_per_obs[mask].mean()
        ccs.append((a @ b / (a.norm() * b.norm() + 1e-30)).item())

    mean_cc = sum(ccs) / len(ccs)
    assert 0.45 < mean_cc < 0.65, f"Single-crystal CC = {mean_cc:.3f}, expected ~0.55"


def test_recover_scale_factors():
    """Optimizer recovers known per-crystal G and |F|² at realistic noise."""
    inp, true_log_G, true_log_F = _make_synthetic_inputs()

    abs_scale = AbsoluteScale(inp.n_crystals)
    sf = MergedStructureFactor(inp.n_unique, inp.unique_hkl)

    with torch.no_grad():
        sf.log_F.copy_(_init_log_F(inp.hkl_id, inp.I_obs, inp.n_unique))

    loss_fn = CompositeScalingLoss(n_shells=inp.n_shells)
    loss_fn.calibrate(inp.I_obs, torch.zeros_like(inp.I_obs), inp.shell, I_min_abs=10.0)

    params = list(abs_scale.parameters()) + list(sf.parameters())
    optimizer = torch.optim.Adam(params, lr=0.005)

    for step in range(600):
        optimizer.zero_grad()
        log_scale = abs_scale.log_scale_per_obs(inp.crystal_id)
        log_I_pred = sf.predict_log_I(inp.hkl_id) + log_scale
        I_pred = torch.exp(log_I_pred)
        loss = loss_fn(inp.I_obs, I_pred, log_I_pred, inp.shell)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            abs_scale.log_G.data -= abs_scale.log_G.data.mean()

    cc_G = _corrcoef(abs_scale.log_G.data, true_log_G)
    assert cc_G > 0.30, f"log_G correlation too low: {cc_G:.3f}"

    cc_F = _corrcoef(sf.log_F.data, true_log_F)
    assert cc_F > 0.95, f"log_F correlation too low: {cc_F:.3f}"


def test_recover_with_bscale():
    """Joint G + B recovery at realistic noise."""
    inp, true_log_G, true_log_F, true_B = _make_synthetic_inputs(
        include_B=True, seed=456,
    )

    abs_scale = AbsoluteScale(inp.n_crystals)
    b_scale = BScale(inp.n_crystals)
    sf = MergedStructureFactor(inp.n_unique, inp.unique_hkl)

    with torch.no_grad():
        sf.log_F.copy_(_init_log_F(inp.hkl_id, inp.I_obs, inp.n_unique))

    loss_fn = CompositeScalingLoss(n_shells=inp.n_shells)
    loss_fn.calibrate(inp.I_obs, torch.zeros_like(inp.I_obs), inp.shell, I_min_abs=10.0)

    params = (
        list(abs_scale.parameters())
        + list(b_scale.parameters())
        + list(sf.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=0.005)

    for step in range(800):
        optimizer.zero_grad()
        log_scale = (
            abs_scale.log_scale_per_obs(inp.crystal_id)
            + b_scale.log_scale_per_obs(inp.crystal_id, inp.s_sq)
        )
        log_I_pred = sf.predict_log_I(inp.hkl_id) + log_scale
        I_pred = torch.exp(log_I_pred)
        loss = loss_fn(inp.I_obs, I_pred, log_I_pred, inp.shell)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            abs_scale.log_G.data -= abs_scale.log_G.data.mean()

    # Total scale at median s² (G and B have a gauge ambiguity at s²=0)
    s_sq_median = inp.s_sq.median()
    true_total = true_log_G - 2.0 * true_B * s_sq_median
    recovered_total = (
        abs_scale.log_G.data - 2.0 * b_scale.B.data * s_sq_median
    )
    cc_total = _corrcoef(recovered_total, true_total)
    assert cc_total > 0.35, f"Total scale correlation too low: {cc_total:.3f}"

    cc_F = _corrcoef(sf.log_F.data, true_log_F)
    assert cc_F > 0.95, f"log_F correlation too low: {cc_F:.3f}"


def test_lognormal_removes_bias():
    """Lognormal term corrects the upward bias from linear-space merging.

    At moderate σ_LN (0.25), the Gaussian-only estimator is biased by
    exp(σ²/2).  The composite loss should produce less-biased |F|.
    Uses moderate noise (not realistic CC) to isolate the bias effect.
    """
    inp, true_log_G, true_log_F = _make_synthetic_inputs(
        sigma_LN=0.25, sigma_normal=10.0, seed=789,
    )

    def _run(use_lognormal: bool, steps: int = 400) -> torch.Tensor:
        sf = MergedStructureFactor(inp.n_unique, inp.unique_hkl)
        scaler = AbsoluteScale(inp.n_crystals)
        loss_fn = CompositeScalingLoss(n_shells=inp.n_shells)
        loss_fn.calibrate(inp.I_obs, torch.zeros_like(inp.I_obs), inp.shell, I_min_abs=10.0)
        if not use_lognormal:
            loss_fn.alpha.fill_(0.0)  # Gaussian-only

        with torch.no_grad():
            sf.log_F.copy_(_init_log_F(inp.hkl_id, inp.I_obs, inp.n_unique))

        params = list(scaler.parameters()) + list(sf.parameters())
        opt = torch.optim.Adam(params, lr=0.005)
        for _ in range(steps):
            opt.zero_grad()
            ls = scaler.log_scale_per_obs(inp.crystal_id)
            log_I_pred = sf.predict_log_I(inp.hkl_id) + ls
            loss = loss_fn(inp.I_obs, torch.exp(log_I_pred), log_I_pred, inp.shell)
            loss.backward()
            opt.step()
            with torch.no_grad():
                scaler.log_G.data -= scaler.log_G.data.mean()
        return sf.log_F.data.detach()

    log_F_gauss_only = _run(use_lognormal=False)
    log_F_composite = _run(use_lognormal=True)

    bias_gauss = (log_F_gauss_only - true_log_F).mean().abs().item()
    bias_composite = (log_F_composite - true_log_F).mean().abs().item()

    assert bias_composite < bias_gauss, (
        f"Composite bias ({bias_composite:.4f}) should be less than "
        f"Gaussian-only bias ({bias_gauss:.4f})"
    )
