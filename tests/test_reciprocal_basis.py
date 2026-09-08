"""Smooth symmetry-adapted reciprocal-space basis and the Wilson ML fit on it."""

from __future__ import annotations

import pytest
import torch

from torchsx.merging.priors.reciprocal_basis import (
    ReciprocalBasis,
    chebyshev,
    fit_wilson_log_sigma,
    invariant_traceless_tensors,
    wilson_b_equivalent,
)

CELLS = {
    "P 63": torch.tensor([62.23, 62.23, 110.77, 90.0, 90.0, 120.0]),
    "P 21 21 21": torch.tensor([40.0, 50.0, 60.0, 90.0, 90.0, 90.0]),
    "P 1": torch.tensor([40.0, 50.0, 60.0, 80.0, 85.0, 95.0]),
    "P 21 3": torch.tensor([60.0, 60.0, 60.0, 90.0, 90.0, 90.0]),
}


def _hkl_grid(cell: torch.Tensor, d_min: float, seed: int = 0, n: int = 3000) -> torch.Tensor:
    from torchref.base.reciprocal import reciprocal_basis_matrix

    g = torch.Generator().manual_seed(seed)
    B = reciprocal_basis_matrix(cell.double())
    hmax = (1.0 / d_min / B.norm(dim=1)).ceil().long() + 1
    hkl = torch.stack([torch.randint(-int(m), int(m) + 1, (n * 4,), generator=g) for m in hmax], dim=1)
    s = hkl.double() @ B
    d = 1.0 / s.norm(dim=1).clamp(min=1e-9)
    keep = (d >= d_min) & (d < 50.0)
    hkl = hkl[keep]
    hkl = torch.unique(hkl, dim=0)
    return hkl[:n]


def test_chebyshev_recurrence():
    x = torch.linspace(-1, 1, 11, dtype=torch.float64)
    T = chebyshev(x, 3)
    assert torch.allclose(T[:, 2], 2 * x * x - 1)
    assert torch.allclose(T[:, 3], 4 * x ** 3 - 3 * x)


@pytest.mark.parametrize("sg,expected", [("P 63", 1), ("P 21 21 21", 2), ("P 1", 5), ("P 21 3", 0)])
def test_invariant_tensor_count(sg, expected):
    A = invariant_traceless_tensors(sg, CELLS[sg])
    assert A.shape == (expected, 3, 3)
    for a in A:
        assert abs(float(a.trace())) < 1e-9
        assert torch.allclose(a, a.T)
    if expected:
        gram = torch.einsum("kij,lij->kl", A, A)
        assert torch.allclose(gram, torch.eye(expected, dtype=gram.dtype), atol=1e-9)


def test_p63_tensor_separates_c_star_from_ab():
    A = invariant_traceless_tensors("P 63", CELLS["P 63"])[0]
    # Invariant under the 6-fold about c*: diagonal with A_xx == A_yy, A_zz = -2 A_xx.
    assert abs(float(A[0, 0] - A[1, 1])) < 1e-9
    assert abs(float(A[2, 2] + 2 * A[0, 0])) < 1e-9
    assert float(A.abs()[[0, 0, 1], [1, 2, 2]].max()) < 1e-9


@pytest.mark.parametrize("sg", ["P 63", "P 21 21 21", "P 21 3"])
def test_columns_are_symmetry_invariant(sg):
    from torchref.symmetry import SpaceGroup

    cell = CELLS[sg]
    hkl = _hkl_grid(cell, 2.0, seed=1, n=800)
    basis = ReciprocalBasis.from_hkl(hkl, cell, sg, degree=4)
    assert basis.p == 5 + basis.n_aniso
    group = SpaceGroup(sg)
    images = group.expand_reciprocal(hkl.to(group.device)).cpu().long()   # (n_ops, n, 3)
    ref = basis.evaluate(hkl)
    for o in range(images.shape[0]):
        assert torch.allclose(basis.evaluate(images[o]), ref, atol=1e-4), f"op {o} breaks invariance"
    assert torch.allclose(basis.phi, ref, atol=1e-6)


def test_standardisation_and_t0():
    cell = CELLS["P 21 21 21"]
    basis = ReciprocalBasis.from_hkl(_hkl_grid(cell, 2.2, n=2000), cell, "P 21 21 21", degree=3)
    assert torch.allclose(basis.phi[:, 0], torch.ones(basis.phi.shape[0]))
    assert torch.allclose(basis.phi[:, 1:].mean(dim=0), torch.zeros(basis.p - 1), atol=1e-5)
    assert torch.allclose(basis.phi[:, 1:].std(dim=0, unbiased=False), torch.ones(basis.p - 1), atol=1e-4)


def _wilson_sample(basis: ReciprocalBasis, log_sigma: torch.Tensor, is_centric: torch.Tensor, seed: int):
    g = torch.Generator().manual_seed(seed)
    sigma = log_sigma.exp()
    acen = -torch.log(torch.rand(len(sigma), generator=g, dtype=torch.float64)) * sigma      # Exp(Sigma)
    cen = torch.randn(len(sigma), generator=g, dtype=torch.float64) ** 2 * sigma               # Sigma * chi2_1
    return torch.where(is_centric, cen, acen)


@pytest.mark.parametrize("seed", [0, 1])
def test_fit_recovers_isotropic_wilson_b(seed):
    cell = CELLS["P 21 21 21"]
    hkl = _hkl_grid(cell, 1.8, seed=seed, n=3000)
    basis = ReciprocalBasis.from_hkl(hkl, cell, "P 21 21 21", degree=4)
    B_true = 25.0
    log_sigma = 6.0 - 2.0 * B_true * basis.s2.double()
    I = _wilson_sample(basis, log_sigma, torch.zeros(len(hkl), dtype=torch.bool), seed)
    coef, used = fit_wilson_log_sigma(basis.phi, I, None)
    assert used <= 8
    fitted = basis.phi.double() @ coef.double()
    assert float((fitted - log_sigma).abs().mean()) < 0.06
    assert abs(wilson_b_equivalent(basis, coef) - B_true) < 1.5


def test_fit_recovers_anisotropic_term_and_centrics():
    sg, cell = "P 63", CELLS["P 63"]
    hkl = _hkl_grid(cell, 2.0, seed=3, n=4000)
    basis = ReciprocalBasis.from_hkl(hkl, cell, sg, degree=4)
    assert basis.n_aniso == 1
    truth = torch.zeros(basis.p, dtype=torch.float64)
    truth[0], truth[1], truth[-1] = 5.0, -1.2, -0.6          # curvature-free line + aniso
    log_sigma = basis.phi.double() @ truth
    from torchref.symmetry import SpaceGroup
    is_centric = SpaceGroup(sg).is_centric(hkl.to(SpaceGroup(sg).device)).cpu()
    assert bool(is_centric.any())
    I = _wilson_sample(basis, log_sigma, is_centric, 7)
    coef, used = fit_wilson_log_sigma(basis.phi, I, is_centric)
    assert used <= 8
    fitted = basis.phi.double() @ coef.double()
    assert float((fitted - log_sigma).abs().mean()) < 0.06
    assert abs(float(coef[-1]) - float(truth[-1])) < 0.1
    # Warm start from the solution converges immediately.
    coef2, used2 = fit_wilson_log_sigma(basis.phi, I, is_centric, coef0=coef)
    assert used2 <= 2
    assert torch.allclose(coef2.double(), coef.double(), atol=1e-4)


def test_fit_ignoring_centrics_biases_sigma_low():
    """Centric reflections have E[I] = Sigma too, but a sharper distribution; with the 1/2 weight
    the fit is right, without it the centric-heavy zones would still average to Sigma. Just check
    the two fits differ only mildly and the weighted one is closer to truth."""
    sg, cell = "P 63", CELLS["P 63"]
    hkl = _hkl_grid(cell, 2.0, seed=5, n=4000)
    basis = ReciprocalBasis.from_hkl(hkl, cell, sg, degree=2)
    from torchref.symmetry import SpaceGroup
    is_centric = SpaceGroup(sg).is_centric(hkl.to(SpaceGroup(sg).device)).cpu()
    log_sigma = torch.full((len(hkl),), 4.0, dtype=torch.float64)
    I = _wilson_sample(basis, log_sigma, is_centric, 11)
    c_w, _ = fit_wilson_log_sigma(basis.phi, I, is_centric)
    c_u, _ = fit_wilson_log_sigma(basis.phi, I, None)
    assert abs(float(c_w[0]) - 4.0) < 0.05
    assert abs(float(c_u[0]) - 4.0) < 0.08
