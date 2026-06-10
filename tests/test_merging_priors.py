"""Unit tests for the modular priors in ``torchsx.merging.priors``.

Mirrors ``tests/test_merging_clean_fw.py`` but against the refactored
module layout.  The old test file stays in place until merging_clean is
deleted in step 7 of the refactor.
"""

from __future__ import annotations

import math

import pytest
import torch

from torchsx.merging.priors import WilsonPrior, WilsonMatchPrior, FlatPrior


# ─── WilsonPrior ────────────────────────────────────────────────────────


def test_acentric_single_shell_matches_analytic() -> None:
    """Prior on φ = ½·log I, acentric: nlp = I/Σ + log Σ − log I."""
    log_I0 = torch.tensor(
        [math.log(1.0), math.log(2.0), math.log(3.0), math.log(4.0)]
    )
    n = log_I0.numel()
    shell_id = torch.zeros(n, dtype=torch.long)
    is_centric = torch.zeros(n, dtype=torch.bool)
    prior = WilsonPrior(shell_id, is_centric, n_shells=1, eps=1e-12)
    nlp = prior(log_I0)

    I0 = torch.exp(log_I0)
    Sigma = I0.mean()
    expected = I0 / Sigma + torch.log(Sigma) - log_I0
    assert torch.allclose(nlp, expected, atol=1e-6)


def test_centric_single_shell_matches_analytic() -> None:
    """Prior on φ, centric: nlp = I/(2Σ) + ½·log Σ − ½·log I."""
    log_I0 = torch.tensor(
        [math.log(0.5), math.log(1.5), math.log(2.5), math.log(5.0)]
    )
    n = log_I0.numel()
    shell_id = torch.zeros(n, dtype=torch.long)
    is_centric = torch.ones(n, dtype=torch.bool)
    prior = WilsonPrior(shell_id, is_centric, n_shells=1, eps=1e-12)
    nlp = prior(log_I0)

    I0 = torch.exp(log_I0)
    Sigma = I0.mean()
    expected = 0.5 * (I0 / Sigma) + 0.5 * torch.log(Sigma) - 0.5 * log_I0
    assert torch.allclose(nlp, expected, atol=1e-6)


def test_mixed_two_shells() -> None:
    """Two shells with mixed centric/acentric; per-shell Σ and per-HKL branch."""
    log_I0 = torch.tensor(
        [math.log(1.0), math.log(3.0), math.log(10.0), math.log(20.0)]
    )
    shell_id = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    is_centric = torch.tensor([True, False, False, True], dtype=torch.bool)
    prior = WilsonPrior(shell_id, is_centric, n_shells=2, eps=1e-12)
    nlp = prior(log_I0)

    I0 = torch.exp(log_I0)
    Sigma0 = I0[:2].mean()
    Sigma1 = I0[2:].mean()
    Sigma = torch.tensor([Sigma0, Sigma0, Sigma1, Sigma1])
    J = I0 / Sigma
    log_Sigma = torch.log(Sigma)
    nlp_acen = J + log_Sigma - log_I0
    nlp_cent = 0.5 * J + 0.5 * log_Sigma - 0.5 * log_I0
    expected = torch.where(is_centric, nlp_cent, nlp_acen)
    assert torch.allclose(nlp, expected, atol=1e-6)


def test_shell_mean_gradient_flows() -> None:
    """Gradient through Σ must flow (prior behaves as EM only in the limit)."""
    log_I0 = torch.tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
    shell_id = torch.zeros(4, dtype=torch.long)
    is_centric = torch.zeros(4, dtype=torch.bool)
    prior = WilsonPrior(shell_id, is_centric, n_shells=1, eps=1e-12)
    nlp = prior(log_I0).sum()
    nlp.backward()
    assert log_I0.grad is not None
    assert torch.isfinite(log_I0.grad).all()


def test_gradcheck_small() -> None:
    """Tight float64 gradcheck on a small setup."""
    torch.manual_seed(0)
    n, K = 6, 2
    log_I0 = (torch.rand(n, dtype=torch.float64) - 0.5).requires_grad_(True)
    shell_id = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    is_centric = torch.tensor([False, True, False, True, False, True])
    prior = WilsonPrior(shell_id, is_centric, n_shells=K, eps=1e-12)
    prior.double()

    def fn(x):
        return prior(x).sum()

    assert torch.autograd.gradcheck(fn, (log_I0,), eps=1e-6, atol=1e-4)


# ─── FlatPrior ──────────────────────────────────────────────────────────


def test_flat_prior_is_zero() -> None:
    log_I0 = torch.randn(10)
    prior = FlatPrior()
    assert torch.equal(prior(log_I0), torch.zeros_like(log_I0))


# ─── WilsonMatchPrior ───────────────────────────────────────────────────


def _draw_wilson_like(n: int, centric: bool, seed: int = 0) -> torch.Tensor:
    """Draw I from Wilson and return log(I/Σ) with Σ=1."""
    gen = torch.Generator().manual_seed(seed)
    if not centric:
        u = torch.rand(n, generator=gen, dtype=torch.float64)
        J = -torch.log1p(-u.clamp(max=1 - 1e-12))
    else:
        z = torch.randn(n, generator=gen, dtype=torch.float64)
        J = z * z
    return torch.log(J.clamp(min=1e-12))


def test_wilson_match_zero_when_wilson_sampled() -> None:
    """KL rate → ~0 as n → ∞ on a true Wilson sample."""
    n = 40000
    log_J_acen = _draw_wilson_like(n, centric=False, seed=1)
    log_I0 = log_J_acen.float()
    shell_id = torch.zeros(n, dtype=torch.long)
    is_centric = torch.zeros(n, dtype=torch.bool)
    prior = WilsonMatchPrior(
        shell_id, is_centric, n_shells=1, eps=1e-10, min_count=10
    )
    kl = prior(log_I0).item() / n
    assert kl < 1e-3, f"KL rate={kl:.5f} too large for Wilson sample"


def test_wilson_match_large_when_degenerate() -> None:
    """All-equal intensities ⇒ huge KL (zero empirical variance)."""
    n = 1000
    log_I0 = torch.zeros(n)
    shell_id = torch.zeros(n, dtype=torch.long)
    is_centric = torch.zeros(n, dtype=torch.bool)
    prior = WilsonMatchPrior(
        shell_id, is_centric, n_shells=1, eps=1e-6, min_count=10
    )
    kl_degen = prior(log_I0)
    log_I_wilson = _draw_wilson_like(n, centric=False, seed=2).float()
    kl_wilson = prior(log_I_wilson)
    assert kl_degen > kl_wilson * 100, (
        f"degenerate KL={kl_degen.item():.2f} not >> "
        f"Wilson KL={kl_wilson.item():.2f}"
    )


def test_wilson_match_separate_centric_acentric_shells() -> None:
    """Each shell×centricity group is KL'd against the right target."""
    torch.manual_seed(0)
    n = 2000
    shell_id = torch.tensor([0] * (n // 2) + [1] * (n // 2), dtype=torch.long)
    is_centric = torch.tensor([i % 2 == 0 for i in range(n)], dtype=torch.bool)
    log_I0 = torch.randn(n) * 1.5
    prior = WilsonMatchPrior(
        shell_id, is_centric, n_shells=2, eps=1e-6, min_count=10
    )
    kl = prior(log_I0)
    assert torch.isfinite(kl).item() and kl.item() > 0


def test_wilson_match_gradient_flows() -> None:
    torch.manual_seed(1)
    n = 200
    log_I0 = torch.randn(n, requires_grad=True)
    shell_id = torch.zeros(n, dtype=torch.long)
    is_centric = torch.zeros(n, dtype=torch.bool)
    prior = WilsonMatchPrior(
        shell_id, is_centric, n_shells=1, eps=1e-6, min_count=10
    )
    kl = prior(log_I0)
    kl.backward()
    assert log_I0.grad is not None
    assert torch.isfinite(log_I0.grad).all()
    assert (log_I0.grad.abs().sum() > 0).item()


# ─── Regularizers ──────────────────────────────────────────────────────


def test_miller_graph_smoothness_equals_dirichlet() -> None:
    """MillerGraphSmoothness equals the old hand-rolled Dirichlet block."""
    from torchsx.merging.indexing import build_adjacency, dirichlet_energy
    from torchsx.merging.regularizers import MillerGraphSmoothness
    from torchsx.merging.parameters import Parameters

    # Small 3-HKL path: (0,0,0)-(1,0,0)-(2,0,0)
    hkl = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=torch.int16)
    adj = build_adjacency(hkl)
    params = Parameters(
        log_I0_init=torch.tensor([0.0, 2.0, 5.0]),
        log_alpha_init=torch.tensor([-1.0, -1.0, -1.0]),
        log_beta_init=torch.tensor([0.0, 0.0, 0.0]),
    )
    reg = MillerGraphSmoothness(adj, target="log_I0")
    expected = dirichlet_energy(params.log_I0, adj)
    assert torch.isclose(reg(params), expected)


def test_gaussian_anchor_equals_sum_of_squares() -> None:
    from torchsx.merging.regularizers import GaussianAnchor
    from torchsx.merging.parameters import Parameters

    params = Parameters(
        log_I0_init=torch.tensor([1.0, 2.0, 3.0]),
        log_alpha_init=torch.zeros(3),
        log_beta_init=torch.zeros(3),
    )
    reg = GaussianAnchor(target="log_I0", anchor=0.0)
    assert torch.isclose(reg(params), torch.tensor(1.0 + 4.0 + 9.0))

    reg2 = GaussianAnchor(target="log_I0", anchor=torch.tensor([1.0, 2.0, 3.0]))
    assert torch.isclose(reg2(params), torch.tensor(0.0))


def test_sigma_cauchy_matches_manual() -> None:
    from torchsx.merging.regularizers import SigmaCauchy
    from torchsx.merging.parameters import Parameters

    log_alpha = torch.tensor([0.0, 0.5, 1.0])
    log_beta = torch.log(torch.tensor([1.5, 1.8, 1.2]))
    params = Parameters(
        log_I0_init=torch.zeros(3),
        log_alpha_init=log_alpha,
        log_beta_init=log_beta,
    )
    reg = SigmaCauchy(tau=1.0)
    # Manual recomputation
    beta = log_beta.exp()
    log_c_beta = 0.5 * (torch.lgamma(3.0 / beta) - torch.lgamma(1.0 / beta))
    log_sigma = log_alpha + log_c_beta
    sigma2 = torch.exp(2.0 * log_sigma)
    expected = torch.log(1.0 + sigma2).sum()
    assert torch.isclose(reg(params), expected, atol=1e-6)
