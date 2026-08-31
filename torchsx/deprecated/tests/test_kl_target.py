"""Tests for the KLScalingTarget unified loss."""

import math

import torch
import pytest

from torchsx.deprecated.scaling import AbsoluteScale, BScale, ScalerInputs
from torchsx.deprecated.target import KLScalingTarget


# ---------------------------------------------------------------------------
# Reuse synthetic data helpers from test_scaler
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
    """Generate synthetic ScalerInputs with known ground truth.

    Parameters
    ----------
    allow_negative : bool
        If False (default), clamp I_obs >= 1.0 (backwards compatible).
        If True, keep negative observations (for truncation bias tests).
    """
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
    a = a.detach().float()
    b = b.detach().float()
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm())).item()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_kl_finite_positive():
    """Scalar, finite, has grad for all parameters."""
    inp, _, _ = _make_synthetic_inputs(n_crystals=20, n_unique=50, obs_per_crystal=30, seed=1)
    target = KLScalingTarget(inp.n_unique, inp.n_shells)
    total_scale = torch.ones(inp.I_obs.shape[0])

    loss = target(inp.I_obs, total_scale, inp.hkl_id)

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert target.log_I_true.grad is not None
    assert target.log_sigma_LN.grad is not None
    assert target.log_sigma_gauss.grad is not None


def test_kl_self_regularizing():
    """KL at wrong I_true > KL at correct I_true."""
    inp, _, true_log_F = _make_synthetic_inputs(
        n_crystals=20, n_unique=50, obs_per_crystal=30, seed=2,
    )
    total_scale = torch.ones(inp.I_obs.shape[0])

    target = KLScalingTarget(inp.n_unique, inp.n_shells, lam=0.0)
    target.calibrate(inp.I_obs, total_scale, inp.hkl_id, inp.shell, inp.crystal_id)
    loss_calibrated = target(inp.I_obs, total_scale, inp.hkl_id).item()

    # Set I_true to wrong values
    with torch.no_grad():
        target.log_I_true.fill_(0.0)
    loss_wrong = target(inp.I_obs, total_scale, inp.hkl_id).item()

    assert loss_wrong > loss_calibrated


def test_calibrate_initializes():
    """After calibrate, sigma_LN and sigma_gauss are reasonable (not default)."""
    inp, _, _ = _make_synthetic_inputs(n_crystals=40, n_unique=100, obs_per_crystal=60, seed=3)
    target = KLScalingTarget(inp.n_unique, inp.n_shells)
    total_scale = torch.ones(inp.I_obs.shape[0])

    target.calibrate(inp.I_obs, total_scale, inp.hkl_id, inp.shell, inp.crystal_id)

    # Priors should be set to non-default values
    assert not torch.allclose(target.sigma_LN_prior, torch.full_like(target.sigma_LN_prior, 0.05))
    assert not torch.allclose(target.sigma_gauss_prior, torch.full_like(target.sigma_gauss_prior, 10.0))

    # Per-shell params should match priors after calibration
    assert torch.allclose(target.sigma_LN, target.sigma_LN_prior, rtol=1e-5)


def test_singleton_masked():
    """Reflections with count < 2 are excluded from fit/mean terms."""
    n_unique = 10
    # 5 observations: hkl 0 has 3 obs, hkl 1 has 1 obs (singleton), hkl 2 has 1 obs
    hkl_id = torch.tensor([0, 0, 0, 1, 2])
    I_obs = torch.tensor([100.0, 110.0, 90.0, 50.0, 200.0])
    shell = torch.zeros(5, dtype=torch.long)
    total_scale = torch.ones(5)

    target = KLScalingTarget(n_unique, n_shells=1)
    loss = target(I_obs, total_scale, hkl_id)

    assert torch.isfinite(loss)
    loss.backward()


def test_equilibrium_var_match():
    """At convergence, log-space var_model approx log-space var_emp."""
    inp, _, _ = _make_synthetic_inputs(
        n_crystals=40, n_unique=50, obs_per_crystal=80, n_shells=5, seed=4,
    )
    target = KLScalingTarget(inp.n_unique, inp.n_shells, lam=0.01)
    total_scale = torch.ones(inp.I_obs.shape[0])
    target.calibrate(inp.I_obs, total_scale, inp.hkl_id, inp.shell, inp.crystal_id)

    # Only optimize target params (no scaling)
    optimizer = torch.optim.Adam(target.parameters(), lr=0.01)
    for _ in range(500):
        optimizer.zero_grad()
        loss = target(inp.I_obs, total_scale, inp.hkl_id)
        loss.backward()
        optimizer.step()

    # Compute log-space var_emp and var_log_model
    with torch.no_grad():
        log_r = torch.log(inp.I_obs.clamp(min=1e-30))
        ones = torch.ones_like(log_r)
        count = torch.zeros(inp.n_unique).scatter_add_(0, inp.hkl_id, ones)
        sum_lr = torch.zeros(inp.n_unique).scatter_add_(0, inp.hkl_id, log_r)
        sum_lr2 = torch.zeros(inp.n_unique).scatter_add_(0, inp.hkl_id, log_r * log_r)
        valid = count >= 5
        n = count.clamp(min=2)
        var_log_emp = ((sum_lr2 - sum_lr * sum_lr / n) / (n - 1).clamp(min=1)).clamp(min=1e-10)

        I_true = target.I_true
        sh = target.shell_per_hkl
        sLN = target.sigma_LN[sh]
        sg = target.sigma_gauss[sh]
        var_log_model = (sLN * sLN + sg * sg / I_true.clamp(min=1.0) ** 2).clamp(min=1e-10)

        # Check ratio for valid reflections
        ratio = var_log_model[valid] / var_log_emp[valid]
        median_ratio = ratio.median().item()
        assert 0.3 < median_ratio < 3.0, f"Median var_log_model/var_log_emp = {median_ratio:.2f}"


# ---------------------------------------------------------------------------
# Functional recovery tests
# ---------------------------------------------------------------------------


def test_recover_scale_factors():
    """Optimizer recovers known per-crystal G and |F|²."""
    inp, true_log_G, true_log_F = _make_synthetic_inputs(seed=10)

    abs_scale = AbsoluteScale(inp.n_crystals)
    target = KLScalingTarget(inp.n_unique, inp.n_shells, lam=1.0)
    total_scale = torch.ones(inp.I_obs.shape[0])
    target.calibrate(inp.I_obs, total_scale, inp.hkl_id, inp.shell, inp.crystal_id)

    params = list(abs_scale.parameters()) + list(target.parameters())
    optimizer = torch.optim.Adam(params, lr=0.005)

    for step in range(600):
        optimizer.zero_grad()
        log_scale = abs_scale.log_scale_per_obs(inp.crystal_id)
        ts = torch.exp(log_scale)
        loss = target(inp.I_obs, ts, inp.hkl_id)
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
    target = KLScalingTarget(inp.n_unique, inp.n_shells, lam=1.0)
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
        loss = target(inp.I_obs, ts, inp.hkl_id)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            abs_scale.log_G.data -= abs_scale.log_G.data.mean()

    # Total scale at median s²
    s_sq_median = inp.s_sq.median()
    true_total = true_log_G - 2.0 * true_B * s_sq_median
    recovered_total = abs_scale.log_G.data - 2.0 * b_scale.B.data * s_sq_median
    cc_total = _corrcoef(recovered_total, true_total)
    assert cc_total > 0.35, f"Total scale correlation too low: {cc_total:.3f}"

    cc_F = _corrcoef(target.log_I_true.data / 2.0, true_log_F)
    assert cc_F > 0.95, f"log_F correlation too low: {cc_F:.3f}"


def test_sigma_refines_toward_prior():
    """Starting from wrong sigma, optimizer drives sigma toward calibrated prior."""
    inp, _, _ = _make_synthetic_inputs(
        n_crystals=40, n_unique=50, obs_per_crystal=80, n_shells=5, seed=5,
    )
    target = KLScalingTarget(inp.n_unique, inp.n_shells, lam=5.0)
    total_scale = torch.ones(inp.I_obs.shape[0])
    target.calibrate(inp.I_obs, total_scale, inp.hkl_id, inp.shell, inp.crystal_id)

    # Save calibrated prior values
    prior_LN = target.sigma_LN_prior.clone()
    prior_gauss = target.sigma_gauss_prior.clone()

    # Perturb sigma away from prior
    with torch.no_grad():
        target.log_sigma_LN.add_(1.0)
        target.log_sigma_gauss.add_(1.0)

    initial_dist_LN = (target.sigma_LN - prior_LN).abs().mean().item()

    optimizer = torch.optim.Adam(target.parameters(), lr=0.01)
    for _ in range(300):
        optimizer.zero_grad()
        loss = target(inp.I_obs, total_scale, inp.hkl_id)
        loss.backward()
        optimizer.step()

    final_dist_LN = (target.sigma_LN - prior_LN).abs().mean().item()
    assert final_dist_LN < initial_dist_LN, (
        f"sigma_LN did not converge toward prior: {initial_dist_LN:.4f} -> {final_dist_LN:.4f}"
    )


def test_floor_penalty_drives_scaling():
    """With floor penalty, var_emp should decrease toward var_model over optimization."""
    inp, _, _ = _make_synthetic_inputs(
        n_crystals=40, n_unique=80, obs_per_crystal=60, n_shells=5, seed=6,
    )

    abs_scale = AbsoluteScale(inp.n_crystals)
    # Perturb scales to create excess variance
    with torch.no_grad():
        abs_scale.log_G.copy_(torch.randn(inp.n_crystals) * 0.5)

    target = KLScalingTarget(inp.n_unique, inp.n_shells, lam=5.0)
    total_scale = torch.exp(abs_scale.log_scale_per_obs(inp.crystal_id))
    target.calibrate(inp.I_obs, total_scale.detach(), inp.hkl_id, inp.shell, inp.crystal_id)

    def _compute_mean_var_emp():
        with torch.no_grad():
            ts = torch.exp(abs_scale.log_scale_per_obs(inp.crystal_id))
            r = inp.I_obs / ts.clamp(min=1e-30)
            ones = torch.ones_like(r)
            count = torch.zeros(inp.n_unique).scatter_add_(0, inp.hkl_id, ones)
            sum_r = torch.zeros(inp.n_unique).scatter_add_(0, inp.hkl_id, r)
            sum_r2 = torch.zeros(inp.n_unique).scatter_add_(0, inp.hkl_id, r * r)
            n = count.clamp(min=2)
            var_emp = ((sum_r2 - sum_r * sum_r / n) / (n - 1).clamp(min=1)).clamp(min=1e-10)
            valid = count >= 2
            return var_emp[valid].mean().item()

    var_emp_initial = _compute_mean_var_emp()

    params = list(abs_scale.parameters()) + list(target.parameters())
    optimizer = torch.optim.Adam(params, lr=0.005)

    for _ in range(400):
        optimizer.zero_grad()
        ts = torch.exp(abs_scale.log_scale_per_obs(inp.crystal_id))
        loss = target(inp.I_obs, ts, inp.hkl_id)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            abs_scale.log_G.data -= abs_scale.log_G.data.mean()

    var_emp_final = _compute_mean_var_emp()
    assert var_emp_final < var_emp_initial, (
        f"var_emp did not decrease: {var_emp_initial:.1f} -> {var_emp_final:.1f}"
    )


def test_lambda_tradeoff():
    """Larger lambda -> sigma closer to prior, more aggressive scaling."""
    inp, _, _ = _make_synthetic_inputs(
        n_crystals=40, n_unique=80, obs_per_crystal=60, n_shells=5, seed=7,
    )

    results = {}
    for lam in [0.1, 10.0]:
        abs_scale = AbsoluteScale(inp.n_crystals)
        with torch.no_grad():
            abs_scale.log_G.copy_(torch.randn(inp.n_crystals) * 0.5)

        target = KLScalingTarget(inp.n_unique, inp.n_shells, lam=lam)
        ts = torch.ones(inp.I_obs.shape[0])
        target.calibrate(inp.I_obs, ts, inp.hkl_id, inp.shell)

        prior_LN = target.sigma_LN_prior.clone()

        params = list(abs_scale.parameters()) + list(target.parameters())
        optimizer = torch.optim.Adam(params, lr=0.005)

        for _ in range(400):
            optimizer.zero_grad()
            ts = torch.exp(abs_scale.log_scale_per_obs(inp.crystal_id))
            loss = target(inp.I_obs, ts, inp.hkl_id)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                abs_scale.log_G.data -= abs_scale.log_G.data.mean()

        dist_to_prior = (target.sigma_LN - prior_LN).abs().mean().item()
        results[lam] = dist_to_prior

    assert results[10.0] < results[0.1], (
        f"Higher lambda should give sigma closer to prior: "
        f"lam=0.1 dist={results[0.1]:.4f}, lam=10 dist={results[10.0]:.4f}"
    )


# ---------------------------------------------------------------------------
# Truncation bias correction tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Crystal weight tests
# ---------------------------------------------------------------------------


def test_weighted_loss_finite():
    """Loss is finite and has gradients when weight is provided."""
    inp, _, _ = _make_synthetic_inputs(n_crystals=20, n_unique=50, obs_per_crystal=30, seed=80)
    target = KLScalingTarget(inp.n_unique, inp.n_shells)
    total_scale = torch.ones(inp.I_obs.shape[0])
    n_obs = inp.I_obs.shape[0]
    weight = torch.ones(n_obs)

    loss = target(inp.I_obs, total_scale, inp.hkl_id, weight=weight)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert target.log_I_true.grad is not None


def test_uniform_weight_matches_unweighted():
    """Uniform weights (summing to N_obs) give the same loss as weight=None."""
    inp, _, _ = _make_synthetic_inputs(n_crystals=20, n_unique=50, obs_per_crystal=30, seed=81)
    target = KLScalingTarget(inp.n_unique, inp.n_shells)
    total_scale = torch.ones(inp.I_obs.shape[0])
    n_obs = inp.I_obs.shape[0]

    loss_none = target(inp.I_obs, total_scale, inp.hkl_id, weight=None).item()
    weight = torch.ones(n_obs)
    # uniform weight summing to N_obs → same denominator
    loss_uniform = target(inp.I_obs, total_scale, inp.hkl_id, weight=weight).item()

    assert abs(loss_none - loss_uniform) < 1e-5 * abs(loss_none), (
        f"Uniform weight should match unweighted: {loss_none:.6f} vs {loss_uniform:.6f}"
    )


def test_compute_crystal_weights_shape():
    """compute_crystal_weights returns correct shape, sum, and floor."""
    inp, _, _ = _make_synthetic_inputs(n_crystals=40, n_unique=100, obs_per_crystal=60, seed=82)
    target = KLScalingTarget(inp.n_unique, inp.n_shells)
    total_scale = torch.ones(inp.I_obs.shape[0])
    target.calibrate(inp.I_obs, total_scale, inp.hkl_id, inp.shell, inp.crystal_id)

    w = target.compute_crystal_weights(
        inp.I_obs, total_scale, inp.hkl_id, inp.crystal_id, w_min=0.1,
    )

    n_obs = inp.I_obs.shape[0]
    assert w.shape == (n_obs,)
    assert torch.isfinite(w).all()
    assert abs(w.sum().item() - n_obs) < 1e-3, f"Sum should be N_obs={n_obs}, got {w.sum().item():.1f}"
    assert w.min().item() >= 0.1 - 1e-5, f"Min weight below w_min: {w.min().item():.4f}"


def test_reweighting_improves_recovery():
    """EM reweighting improves scale factor recovery with noisy crystals."""
    torch.manual_seed(83)
    n_crystals = 60
    n_unique = 200
    obs_per_crystal = 100
    n_noisy = 15  # 25% bad crystals

    gen = torch.Generator().manual_seed(83)

    true_log_G = torch.randn(n_crystals, generator=gen) * 0.3
    true_log_G -= true_log_G.mean()
    true_log_F = torch.randn(n_unique, generator=gen) * 0.8 + 4.0

    crystal_id = torch.arange(n_crystals).repeat_interleave(obs_per_crystal).int()
    n_obs = n_crystals * obs_per_crystal
    hkl_id = torch.randint(0, n_unique, (n_obs,), generator=gen)
    shell = torch.randint(0, 5, (n_obs,), generator=gen)
    s_sq = (shell.float() + 1) * 0.002

    log_I_true = 2.0 * true_log_F[hkl_id] + true_log_G[crystal_id.long()]
    I_obs = torch.exp(log_I_true) * torch.exp(torch.randn(n_obs, generator=gen) * 0.3)

    # Make some crystals very noisy (10x noise)
    noisy_mask = crystal_id < n_noisy
    I_obs[noisy_mask] *= torch.exp(torch.randn(noisy_mask.sum(), generator=gen) * 3.0)
    I_obs = I_obs.clamp(min=1.0)

    unique_hkl = torch.zeros(n_unique, 3, dtype=torch.int16)
    inp = ScalerInputs(
        hkl_id=hkl_id, crystal_id=crystal_id, I_obs=I_obs,
        s_sq=s_sq, shell=shell, n_unique=n_unique,
        n_crystals=n_crystals, n_shells=5, unique_hkl=unique_hkl,
    )

    def _run(use_reweight: bool) -> float:
        abs_scale = AbsoluteScale(n_crystals)
        target = KLScalingTarget(n_unique, 5, lam=1.0)
        ts_init = torch.ones(n_obs)
        target.calibrate(I_obs, ts_init, hkl_id, shell, crystal_id)
        params = list(abs_scale.parameters()) + list(target.parameters())
        weight = None

        n_cycles = 3 if use_reweight else 1
        steps_per = 200

        for cycle in range(n_cycles):
            if use_reweight and cycle > 0:
                with torch.no_grad():
                    ts = torch.exp(abs_scale.log_scale_per_obs(crystal_id))
                weight = target.compute_crystal_weights(
                    I_obs, ts, hkl_id, crystal_id,
                )

            optimizer = torch.optim.Adam(params, lr=0.005)
            for _ in range(steps_per):
                optimizer.zero_grad()
                ts = torch.exp(abs_scale.log_scale_per_obs(crystal_id))
                loss = target(I_obs, ts, hkl_id, weight=weight)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    abs_scale.log_G.data -= abs_scale.log_G.data.mean()

        return _corrcoef(target.log_I_true.data / 2.0, true_log_F)

    cc_no_reweight = _run(use_reweight=False)
    cc_reweight = _run(use_reweight=True)

    assert cc_reweight > cc_no_reweight, (
        f"Reweighting should improve F recovery: cc_no_reweight={cc_no_reweight:.3f}, "
        f"cc_reweight={cc_reweight:.3f}"
    )


def test_negative_obs_finite():
    """Loss is finite and differentiable with negative observations."""
    n_unique = 20
    hkl_id = torch.randint(0, n_unique, (100,))
    # Mix of positive and negative observations
    I_obs = torch.randn(100) * 50.0  # roughly half negative
    total_scale = torch.ones(100, requires_grad=True)

    target = KLScalingTarget(n_unique, n_shells=3)
    loss = target(I_obs, total_scale, hkl_id)

    assert loss.dim() == 0
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    loss.backward()
    assert target.log_I_true.grad is not None
    assert torch.isfinite(target.log_I_true.grad).all()


def test_negative_obs_closer_to_truth():
    """Including negative observations gives I_true closer to ground truth."""
    torch.manual_seed(42)
    n_unique = 5
    n_obs = 500

    # Weak reflections: I_true = 10 with σ_gauss = 50 → ~30% negative
    hkl_id = torch.zeros(n_obs, dtype=torch.long)
    I_true_real = 10.0
    I_obs_full = I_true_real + torch.randn(n_obs) * 50.0

    total_scale = torch.ones(n_obs)

    def _fit_I_true(I_obs_in):
        target = KLScalingTarget(n_unique, n_shells=1, lam=0.0)
        with torch.no_grad():
            target.log_I_true.fill_(math.log(50.0))  # start high
            target.log_sigma_gauss.fill_(math.log(50.0))
            target.log_sigma_LN.fill_(math.log(0.01))
        optimizer = torch.optim.Adam([target.log_I_true], lr=0.05)
        for _ in range(300):
            optimizer.zero_grad()
            loss = target(I_obs_in, total_scale[:I_obs_in.shape[0]], hkl_id[:I_obs_in.shape[0]])
            loss.backward()
            optimizer.step()
        return target.I_true[0].item()

    # Fit with positive-only (clamped)
    I_pos_only = I_obs_full.clamp(min=0.01)
    I_est_pos = _fit_I_true(I_pos_only)

    # Fit with all observations (including negatives)
    I_est_all = _fit_I_true(I_obs_full)

    # Including negatives should give I_true closer to ground truth
    err_pos = abs(I_est_pos - I_true_real)
    err_all = abs(I_est_all - I_true_real)
    assert err_all < err_pos, (
        f"Negative obs should reduce error: pos_only={I_est_pos:.1f} (err={err_pos:.1f}), "
        f"with_neg={I_est_all:.1f} (err={err_all:.1f}), true={I_true_real:.1f}"
    )


def test_truncation_correction_reduces_bias():
    """For weak reflections near noise floor, truncation correction reduces
    the upward bias compared to naive positive-only log-space NLL."""
    torch.manual_seed(99)
    n_unique = 30
    n_per_hkl = 200
    n_obs = n_unique * n_per_hkl
    n_shells = 1

    # True intensities spanning the noise floor (σ_gauss = 100)
    sigma_gauss_true = 100.0
    # Some reflections well below noise floor, some above
    I_true_per_hkl = torch.linspace(5.0, 500.0, n_unique)

    hkl_id = torch.arange(n_unique).repeat_interleave(n_per_hkl)
    shell = torch.zeros(n_obs, dtype=torch.long)
    crystal_id = torch.arange(n_per_hkl).repeat(n_unique).int()

    # Generate with additive Gaussian noise (no lognormal for clarity)
    I_obs = I_true_per_hkl[hkl_id] + torch.randn(n_obs) * sigma_gauss_true

    total_scale = torch.ones(n_obs)

    def _fit(I_obs_in):
        target = KLScalingTarget(n_unique, n_shells, lam=0.0)
        with torch.no_grad():
            target.log_I_true.copy_(torch.log(I_true_per_hkl.clamp(min=1.0)))
            target.log_sigma_gauss.fill_(math.log(sigma_gauss_true))
            target.log_sigma_LN.fill_(math.log(0.01))
        optimizer = torch.optim.Adam([target.log_I_true], lr=0.02)
        for _ in range(400):
            optimizer.zero_grad()
            loss = target(I_obs_in, total_scale[:I_obs_in.shape[0]], hkl_id[:I_obs_in.shape[0]])
            loss.backward()
            optimizer.step()
        return target.I_true.detach()

    # Fit with positive-only
    I_est_pos = _fit(I_obs.clamp(min=0.01))
    # Fit with all (including negatives)
    I_est_all = _fit(I_obs)

    # Weak reflections (I_true < σ_gauss): positive-only should overestimate
    weak = I_true_per_hkl < sigma_gauss_true * 0.5
    bias_pos = (I_est_pos[weak] - I_true_per_hkl[weak]).mean().item()
    bias_all = (I_est_all[weak] - I_true_per_hkl[weak]).mean().item()

    assert abs(bias_all) < abs(bias_pos), (
        f"Truncation correction should reduce bias for weak reflections: "
        f"pos_only_bias={bias_pos:.1f}, with_neg_bias={bias_all:.1f}"
    )
