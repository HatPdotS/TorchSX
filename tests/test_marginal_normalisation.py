"""The marginal likelihood must be a normalised density.

The quadrature integrates the scatter over a finite grid, so mass that falls
off the ends is silently discarded and ``p`` stops being a density.  Measured
on the fixed grid (log S in [-6.9, 13.8]) before the correction: at w = 6.75,
beta = 3.49 only 81.4 % of the GenGauss mass is inside the grid, and at w = 12
only 54.7 % -- the scatter is then wider than the entire grid.

That matters because the deficit is a *function of the width*, so it silently
reweights exactly the parameter the width head is optimising.
"""

from __future__ import annotations

import math

import pytest
import torch

from torchsx.merging.scatter.gengauss import (
    gengauss_log_prob_from_obs,
    log_alpha_from_log_width,
)
from torchsx.nn.objective import gengauss_grid_log_mass

LO, HI = math.log(1e-3), math.log(1e6)


def _numeric_mass(log_mean, log_alpha, beta, n=200001):
    grid = torch.linspace(LO, HI, n)
    dx = (HI - LO) / (n - 1)
    eps = grid[None, :] - torch.tensor([log_mean])[:, None]
    lp = gengauss_log_prob_from_obs(
        eps, torch.tensor([log_alpha]), torch.tensor([float(beta)]).log()
    )
    return float((lp.exp().sum() * dx).item())


@pytest.mark.parametrize("w", [1.0, 1.65, 6.75, 12.0])
@pytest.mark.parametrize("beta", [1.0, 2.0, 3.489])
@pytest.mark.parametrize("log_mean", [0.0, 3.4])
def test_analytic_grid_mass_matches_numerical_integration(w, beta, log_mean):
    la = log_alpha_from_log_width(
        torch.tensor(math.log(w)), torch.tensor(float(beta))
    )
    got = float(
        gengauss_grid_log_mass(
            torch.tensor([log_mean]), la.reshape(1), torch.tensor([float(beta)]),
            LO, HI,
        ).exp().item()
    )
    assert got == pytest.approx(_numeric_mass(log_mean, float(la), beta), rel=1e-4)


def test_grid_mass_is_one_when_the_scatter_fits():
    """No correction where none is needed — otherwise it is pure noise."""
    la = log_alpha_from_log_width(torch.tensor(math.log(0.5)), torch.tensor(2.0))
    m = gengauss_grid_log_mass(
        torch.tensor([3.4]), la.reshape(1), torch.tensor([2.0]), LO, HI
    ).exp()
    assert m.item() == pytest.approx(1.0, abs=1e-6)


def test_grid_mass_falls_as_the_scatter_outgrows_the_grid():
    ms = []
    for w in (1.0, 3.0, 6.75, 12.0, 25.0):
        la = log_alpha_from_log_width(torch.tensor(math.log(w)), torch.tensor(2.0))
        ms.append(
            gengauss_grid_log_mass(
                torch.tensor([3.4]), la.reshape(1), torch.tensor([2.0]), LO, HI
            ).exp().item()
        )
    assert all(a > b for a, b in zip(ms, ms[1:])), ms
    assert ms[0] == pytest.approx(1.0, abs=1e-4) and ms[-1] < 0.5


@pytest.mark.parametrize("log_w", [-6.0, -3.0, -1.0, 0.0, 1.0, 2.0, 3.0])
@pytest.mark.parametrize("beta", [0.3, 1.0, 2.0, 4.0])
@pytest.mark.parametrize("log_mean", [-6.0, 0.0, 13.0])
def test_mass_and_gradient_stay_finite_at_extremes(log_w, beta, log_mean):
    """Regression: |t|**beta overflowed float32 for small alpha.

    t = (grid_bound - log_mean)/alpha reaches ~1e9 when alpha is small, and
    beta up to 4 takes |t|**beta past 3.4e38 -> inf -> NaN gradients.  That
    NaN-ed the whole network on the first real batch, and it struck precisely
    where the scatter sits well inside the grid and the correction is a no-op.
    """
    la = torch.tensor([float(log_w)], requires_grad=True)
    b = torch.tensor([float(beta)], requires_grad=True)
    m = gengauss_grid_log_mass(torch.tensor([float(log_mean)]), la, b, LO, HI)
    m.backward()
    assert torch.isfinite(m).all()
    assert torch.isfinite(la.grad).all()
    assert torch.isfinite(b.grad).all()


def test_wider_scatter_loses_mass_gradient_sign():
    """d log_mass / d log_alpha < 0 once truncation bites."""
    la = torch.tensor([2.5], requires_grad=True)
    m = gengauss_grid_log_mass(torch.tensor([0.0]), la, torch.tensor([2.0]), LO, HI)
    m.backward()
    assert la.grad.item() < 0


def test_hkl_permutation_reads_zero_for_a_constant_scale():
    """A constant log_K has no hkl dependence — the guard must say 0, not NaN.

    NaN here would make the degeneracy monitor silently un-assertable in
    exactly the case it should pass (untrained or collapsed model).
    """
    from torchsx.nn.metrics import hkl_permutation_sensitivity

    class _Const:
        def predict(self, meta):
            class P:
                log_K = torch.zeros(meta.hkl_idx.numel())
            return P()

    from torchsx.merging.observations import ObsInfo

    meta = ObsInfo(
        crystal_id=torch.zeros(20, dtype=torch.long), s_sq=torch.rand(20),
        hkl_idx=torch.arange(20) % 5, n_crystals=1, obs_id=torch.arange(20),
    )
    assert hkl_permutation_sensitivity(_Const(), meta) == 0.0
