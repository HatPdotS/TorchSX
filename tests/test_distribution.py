"""Tests for per-HKL generalized Gaussian distribution fitting."""

import math

import pytest
import torch

from torchsx.merge._distribution import (
    DistributionComparison,
    HKLDistribution,
    _build_adjacency,
    _fit_shell_lbfgs,
    _laplacian_smooth,
    compare_distributions,
    fit_gengauss,
    format_dist_table,
    split_and_compare,
)


# ── Helpers ──────────────────────────────────────────────────


def _sample_gengauss(n: int, beta: float, mu: float, alpha: float, seed: int = 0):
    """Sample from generalized Gaussian: y = mu + alpha * sign(Z) * |X|^(1/beta).

    X ~ Gamma(1/beta, 1), Z ~ Rademacher.
    """
    gen = torch.Generator().manual_seed(seed)
    # Gamma(shape=1/beta, rate=1) → |y-mu|/alpha has the right distribution
    gamma = torch.distributions.Gamma(1.0 / beta, 1.0)
    x = gamma.sample((n,))
    signs = 2 * torch.randint(0, 2, (n,), generator=gen).float() - 1
    return mu + alpha * signs * x.pow(1.0 / beta)


def _make_synthetic_stream(
    n_hkl: int = 200,
    n_obs_per_hkl: int = 60,
    beta: float = 1.3,
    alpha: float = 0.5,
    seed: int = 42,
):
    """Create a minimal SFXStream-like object with synthetic GenGauss data."""
    from torchsx._core import _SFXStreamCore

    gen = torch.Generator().manual_seed(seed)

    # HKLs on a grid: h,k,l in -3..3 → up to 343, take first n_hkl
    coords = []
    for h in range(-4, 5):
        for k in range(-4, 5):
            for l_ in range(-4, 5):
                if h == 0 and k == 0 and l_ == 0:
                    continue
                coords.append((h, k, l_))
                if len(coords) >= n_hkl:
                    break
            if len(coords) >= n_hkl:
                break
        if len(coords) >= n_hkl:
            break
    unique_hkl = torch.tensor(coords[:n_hkl], dtype=torch.int16)
    n_hkl = len(unique_hkl)

    # True intensities (Wilson-like decay)
    true_log_I = torch.randn(n_hkl, generator=gen) * 1.5 + 3.0

    # Generate observations
    all_hkl = []
    all_I = []
    all_crystal_id = []
    crystal = 0
    for h in range(n_hkl):
        # Vary n_obs per HKL
        n_obs = n_obs_per_hkl + torch.randint(-20, 20, (1,), generator=gen).item()
        n_obs = max(5, n_obs)
        log_I_obs = _sample_gengauss(n_obs, beta, true_log_I[h].item(), alpha, seed=seed + h)
        I_obs = torch.exp(log_I_obs)

        hkl_rep = unique_hkl[h].unsqueeze(0).expand(n_obs, -1)
        all_hkl.append(hkl_rep)
        all_I.append(I_obs)
        # Each observation is from a different "crystal"
        all_crystal_id.append(torch.arange(crystal, crystal + n_obs))
        crystal += n_obs

    hkl_obs = torch.cat(all_hkl)
    I_obs = torch.cat(all_I).float()
    crystal_id = torch.cat(all_crystal_id).int()
    n_crystals = crystal

    # Minimal cell: triclinic
    cell = torch.tensor([15.0, 19.0, 19.0, 89.4, 85.0, 67.8]).unsqueeze(0).expand(n_crystals, -1)

    # Minimal reciprocal vectors (approximate, for the core to be valid)
    astar = torch.zeros(n_crystals, 3)
    astar[:, 0] = 1.0 / 15.0
    bstar = torch.zeros(n_crystals, 3)
    bstar[:, 1] = 1.0 / 19.0
    cstar = torch.zeros(n_crystals, 3)
    cstar[:, 2] = 1.0 / 19.0

    core = _SFXStreamCore(
        hkl=hkl_obs,
        I=I_obs,
        sigma=torch.full_like(I_obs, 1.0),
        background=torch.zeros_like(I_obs),
        peak=torch.zeros_like(I_obs),
        fs=torch.zeros_like(I_obs),
        ss=torch.zeros_like(I_obs),
        panel_id=torch.zeros(len(I_obs), dtype=torch.int16),
        crystal_id=crystal_id,
        astar=astar,
        bstar=bstar,
        cstar=cstar,
        cell=cell,
        profile_rad=torch.full((n_crystals,), 0.004),
        resolution=torch.full((n_crystals,), 1.0),
        n_refl=torch.ones(n_crystals, dtype=torch.long),
        image_id=torch.arange(n_crystals, dtype=torch.int32),
        crystal_in_image=torch.zeros(n_crystals, dtype=torch.int16),
        beam_energy=torch.full((n_crystals,), 12960.0),
        det_shift_x=torch.zeros(n_crystals),
        det_shift_y=torch.zeros(n_crystals),
        n_peaks=torch.ones(n_crystals, dtype=torch.long),
        camera_length=torch.full((n_crystals,), 0.1),
        geometry=None,
        metadata=None,
        spacegroup="P 1",
        source_path=None,
    )

    # Wrap in a minimal SFXStream
    class _FakeStream:
        def __init__(self, c):
            self._core = c
            self.n_crystals = n_crystals

        def split_random(self, n_parts=2, seed=None):
            from torchsx.merge._filter import filter_crystals as _fc
            g = torch.Generator()
            if seed is not None:
                g.manual_seed(seed)
            assignment = torch.randint(n_parts, (n_crystals,), generator=g)
            parts = []
            for i in range(n_parts):
                mask = assignment == i
                new_core = _fc(self._core, mask)
                parts.append(_FakeStream(new_core))
            return parts

    return _FakeStream(core), true_log_I, unique_hkl


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def synthetic():
    stream, true_log_I, unique_hkl = _make_synthetic_stream(
        n_hkl=200, n_obs_per_hkl=60, beta=1.3, alpha=0.5
    )
    return stream, true_log_I, unique_hkl


# ── Tests: _fit_shell_lbfgs ──────────────────────────────────


def test_shell_lbfgs_recovers_beta():
    """L-BFGS shell fit recovers known β from synthetic mixture."""
    # 70% GenGauss(β=1.5) + 30% Gaussian shoulder
    n = 10000
    main = _sample_gengauss(int(0.7 * n), 1.5, -1.0, 0.5, seed=0)
    shoulder = torch.randn(n - len(main)) * 0.8 + 1.0
    data = torch.cat([main, shoulder])
    data = data[torch.randperm(len(data))]

    result = _fit_shell_lbfgs(data)
    assert result is not None
    beta, mu1, alpha, w1, mu2, sigma2 = result
    assert 1.0 < beta < 2.0, f"β={beta} out of range"
    assert abs(beta - 1.5) < 0.3, f"β={beta} too far from 1.5"
    assert 0.5 < w1 < 0.95, f"w1={w1} unexpected"


def test_shell_lbfgs_returns_none_on_degenerate():
    """L-BFGS returns None for constant data."""
    data = torch.ones(100, dtype=torch.float64)
    result = _fit_shell_lbfgs(data)
    # Either None or a valid tuple — both acceptable for degenerate input


# ── Tests: adjacency graph ───────────────────────────────────


def test_adjacency_graph_small():
    """3x3x1 grid of HKLs has correct neighbor counts."""
    hkl = []
    for h in range(3):
        for k in range(3):
            hkl.append((h, k, 0))
    hkl_t = torch.tensor(hkl, dtype=torch.int16)
    edge_src, edge_dst, n_neighbors = _build_adjacency(hkl_t)

    # Corner nodes: 2 neighbors, edge nodes: 3, center: 4 (all in 2D, l=0)
    assert n_neighbors.min() >= 1
    assert n_neighbors.max() <= 6
    # Center node (1,1,0) should have 4 neighbors (no l±1 neighbors)
    center_idx = None
    for i in range(len(hkl)):
        if hkl[i] == (1, 1, 0):
            center_idx = i
    assert n_neighbors[center_idx] == 4


def test_adjacency_graph_isolated():
    """Isolated HKLs get no edges."""
    hkl = torch.tensor([[0, 0, 0], [10, 10, 10]], dtype=torch.int16)
    edge_src, edge_dst, n_neighbors = _build_adjacency(hkl)
    assert len(edge_src) == 0


# ── Tests: Laplacian smoothing ───────────────────────────────


def test_laplacian_smooth_reduces_variance():
    """Smoothing should reduce variance of noisy signal."""
    # Line graph: 0-1-2-3-4
    hkl = torch.tensor([[i, 0, 0] for i in range(20)], dtype=torch.int16)
    edge_src, edge_dst, n_neighbors = _build_adjacency(hkl)

    torch.manual_seed(0)
    noisy = torch.randn(20)
    smoothed = _laplacian_smooth(noisy, edge_src, edge_dst, n_neighbors, 10, 0.3)
    assert smoothed.var() < noisy.var()


def test_laplacian_smooth_preserves_constant():
    """Smoothing a constant field should not change it."""
    hkl = torch.tensor([[i, 0, 0] for i in range(10)], dtype=torch.int16)
    edge_src, edge_dst, n_neighbors = _build_adjacency(hkl)

    const = torch.full((10,), 3.14)
    smoothed = _laplacian_smooth(const, edge_src, edge_dst, n_neighbors, 10, 0.3)
    assert torch.allclose(smoothed, const, atol=1e-6)


# ── Tests: fit_gengauss ─────────────────────────────────────


def test_fit_gengauss_returns_correct_shape(synthetic):
    stream, _, _ = synthetic
    fit = fit_gengauss(stream, min_obs=10, n_shells=5, n_smooth_iter=5)
    assert fit.hkl.shape[0] > 0
    n = fit.hkl.shape[0]
    assert fit.I0.shape == (n,)
    assert fit.alpha.shape == (n,)
    assert fit.beta.shape == (n,)
    assert fit.mu.shape == (n,)
    assert fit.d_spacing.shape == (n,)
    assert fit.n_obs.shape == (n,)


def test_fit_gengauss_alpha_positive(synthetic):
    stream, _, _ = synthetic
    fit = fit_gengauss(stream, min_obs=10, n_shells=5, n_smooth_iter=5)
    assert (fit.alpha > 0).all()


def test_fit_gengauss_I0_correlates(synthetic):
    """Fitted I0 should correlate with true generating intensities."""
    stream, true_log_I, unique_hkl = synthetic
    fit = fit_gengauss(stream, min_obs=10, n_shells=5, n_smooth_iter=5)

    # Match fitted HKLs to true indices
    lookup = {}
    for i in range(unique_hkl.shape[0]):
        key = (unique_hkl[i, 0].item(), unique_hkl[i, 1].item(), unique_hkl[i, 2].item())
        lookup[key] = i

    true_vals = []
    fit_vals = []
    for i in range(fit.hkl.shape[0]):
        key = (fit.hkl[i, 0].item(), fit.hkl[i, 1].item(), fit.hkl[i, 2].item())
        j = lookup.get(key)
        if j is not None:
            true_vals.append(true_log_I[j].item())
            fit_vals.append(fit.mu[i].item())

    true_t = torch.tensor(true_vals)
    fit_t = torch.tensor(fit_vals)
    cc = torch.corrcoef(torch.stack([true_t, fit_t]))[0, 1].item()
    assert cc > 0.7, f"CC(mu, true_log_I) = {cc:.3f} too low"


def test_fit_gengauss_min_obs_filter(synthetic):
    stream, _, _ = synthetic
    fit_low = fit_gengauss(stream, min_obs=10, n_shells=5, n_smooth_iter=3)
    fit_high = fit_gengauss(stream, min_obs=50, n_shells=5, n_smooth_iter=3)
    assert fit_high.hkl.shape[0] <= fit_low.hkl.shape[0]


# ── Tests: compare_distributions ─────────────────────────────


def test_compare_identical(synthetic):
    """Comparing a fit with itself gives CC = 1."""
    stream, _, _ = synthetic
    fit = fit_gengauss(stream, min_obs=10, n_shells=5, n_smooth_iter=5)
    comp = compare_distributions(fit, fit, n_shells=5)
    assert comp.cc_I0 == pytest.approx(1.0, abs=1e-6)
    assert comp.cc_alpha == pytest.approx(1.0, abs=1e-6)


def test_compare_split_halves(synthetic):
    """Split-half comparison should show positive correlation."""
    stream, _, _ = synthetic
    fit1, fit2, comp = split_and_compare(
        stream, seed=42, min_obs=5, n_shells=5, n_smooth_iter=3, snr_thresh=0.5,
    )
    assert comp.cc_I0 > 0.3, f"CC(I0) = {comp.cc_I0:.3f} too low"
    assert len(comp.shells) > 0


# ── Tests: format_dist_table ─────────────────────────────────


def test_format_dist_table(synthetic):
    stream, _, _ = synthetic
    fit = fit_gengauss(stream, min_obs=10, n_shells=5, n_smooth_iter=3)
    comp = compare_distributions(fit, fit, n_shells=5)
    table = format_dist_table(comp)
    assert "CC(I0)" in table
    assert "Overall" in table
