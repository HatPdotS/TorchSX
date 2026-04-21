"""Unit tests for the unified LBFGS merge (torchsx.merging_clean)."""

from __future__ import annotations

import math

import pytest
import torch

from torchsx.merging_clean.adjacency import (
    Adjacency,
    build_adjacency,
    dirichlet_energy,
)
from torchsx.merging_clean.corrections import (
    FixedCorrection,
    ObsInfo,
    PanelCorrection,
    ScaleBCorrection,
)
from torchsx.merging_clean.quadrature import build_softplus_grid


# ─────────────────────────────────────────────────────────────
# adjacency
# ─────────────────────────────────────────────────────────────


def test_build_adjacency_shape():
    # 3 adjacent HKLs in a line: (0,0,0) — (1,0,0) — (2,0,0)
    hkl = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=torch.int16)
    adj = build_adjacency(hkl)
    # each interior node has 2 neighbours, the endpoints have 1
    # the graph is directed (each undirected edge counted twice)
    assert adj.edge_src.numel() == 4
    assert adj.n_neighbors.tolist() == [1.0, 2.0, 1.0]


def test_dirichlet_energy_analytical():
    hkl = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=torch.int16)
    adj = build_adjacency(hkl)
    vals = torch.tensor([0.0, 1.0, 3.0])
    # edges: (0-1) twice, (1-2) twice → 2·(1²) + 2·(2²) = 2 + 8 = 10
    energy = dirichlet_energy(vals, adj)
    assert torch.allclose(energy, torch.tensor(10.0))


def test_dirichlet_energy_gradient_finite_diff():
    hkl = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=torch.int16)
    adj = build_adjacency(hkl)
    x = torch.randn(4, requires_grad=True, dtype=torch.float64)
    loss = dirichlet_energy(x, adj)
    loss.backward()
    ag = x.grad.clone()

    eps = 1e-5
    fd = torch.zeros_like(x)
    with torch.no_grad():
        for i in range(x.numel()):
            xp = x.clone().detach()
            xm = x.clone().detach()
            xp[i] += eps
            xm[i] -= eps
            fd[i] = (dirichlet_energy(xp, adj) - dirichlet_energy(xm, adj)) / (2 * eps)
    assert torch.allclose(ag, fd, atol=1e-7), (ag, fd)


# ─────────────────────────────────────────────────────────────
# quadrature grid
# ─────────────────────────────────────────────────────────────


def test_softplus_grid_shape_and_finite():
    n_ph = torch.tensor([5.0, 50.0, 500.0])
    bg = torch.tensor([1.0, 5.0, 20.0])
    grid = build_softplus_grid(n_ph, bg, n_quad=21)
    assert grid.signal.shape == (3, 21)
    assert grid.log_poi_plus_jac.shape == (3, 21)
    assert torch.isfinite(grid.signal).all()
    assert torch.isfinite(grid.log_poi_plus_jac).all()


# ─────────────────────────────────────────────────────────────
# ScaleBCorrection (zero-mean re-param, ridge)
# ─────────────────────────────────────────────────────────────


def test_scaleb_zero_mean_invariant():
    c = ScaleBCorrection(n_crystals=4)
    with torch.no_grad():
        c.log_G_raw.add_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    obs_info = ObsInfo(
        crystal_id=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        s_sq=torch.zeros(4),
        hkl_idx=torch.zeros(4, dtype=torch.long),
        n_crystals=4,
    )
    out = c(obs_info)
    # centred mean should be exactly zero
    assert torch.allclose(out.mean(), torch.tensor(0.0), atol=1e-6)


def test_scaleb_regularizer_positive():
    c = ScaleBCorrection(n_crystals=3, scale_prior=10.0, bfactor_prior=1.0)
    with torch.no_grad():
        c.log_G_raw.add_(torch.tensor([0.0, 1.0, -1.0]))
        c.B.add_(torch.tensor([0.5, 0.0, 0.0]))
    reg = c.regularizer()
    # log_G centred = [0, 1, -1] (already zero-mean) → ‖·‖² = 2.  B → 0.25.
    # reg = 10·2 + 1·0.25 = 20.25
    assert torch.allclose(reg, torch.tensor(20.25), atol=1e-5)


# ─────────────────────────────────────────────────────────────
# End-to-end: synthetic data recovery
# ─────────────────────────────────────────────────────────────


def _make_synthetic(n_hkl=30, n_crystals=20, obs_per_hkl=40, seed=0):
    """Generate synthetic observations from the GenGauss + Poisson model."""
    torch.manual_seed(seed)

    # True per-HKL parameters
    log_I0_true = torch.linspace(math.log(5.0), math.log(500.0), n_hkl)
    alpha_true = torch.full((n_hkl,), 0.3)
    beta_true = torch.full((n_hkl,), 1.5)

    # Per-crystal scale (G only, no B-factor in the simplest test)
    log_G_true = torch.randn(n_crystals) * 0.2
    log_G_true = log_G_true - log_G_true.mean()  # zero-mean so Merger can match

    total_obs = n_hkl * obs_per_hkl
    hkl_idx = torch.arange(n_hkl).repeat_interleave(obs_per_hkl)
    crystal_id = torch.randint(0, n_crystals, (total_obs,))

    # Draw ε from a Gaussian for simplicity (β=2 case); but we set β=1.5 in fit
    # so the objective is not perfectly matched — this still tests recovery.
    eps = torch.randn(total_obs) * 0.3

    # Generative: I_obs = Poisson(I0 · G · exp(ε) + bg) − bg
    scale = torch.exp(log_G_true[crystal_id])
    I0 = torch.exp(log_I0_true[hkl_idx])
    bg = torch.full((total_obs,), 2.0)
    lam = I0 * scale * torch.exp(eps) + bg
    n_peak = torch.poisson(lam)
    I_obs = n_peak - bg

    return dict(
        hkl_idx=hkl_idx,
        crystal_id=crystal_id,
        I_obs=I_obs,
        bg=bg,
        log_I0_true=log_I0_true,
        log_G_true=log_G_true,
        alpha_true=alpha_true,
        beta_true=beta_true,
        n_hkl=n_hkl,
        n_crystals=n_crystals,
    )


def _run_loss_on_synth(synth, *, chunk_size=0, n_lbfgs_steps=40):
    """Build a UnifiedMergeLoss directly on synthetic tensors (no stream)."""
    from torchsx.merging_clean.config import MergeConfig
    from torchsx.merging_clean.corrections import ScaleBCorrection
    from torchsx.merging_clean.loss import UnifiedMergeLoss
    from torchsx.merging_clean.observations import PreparedObs
    from torchsx.merging_clean.adjacency import build_adjacency

    n_hkl = synth["n_hkl"]
    n_obs = synth["I_obs"].numel()

    # Sort by HKL (required by chunking code path)
    order = synth["hkl_idx"].argsort()
    I_obs = synth["I_obs"][order]
    hkl_id = synth["hkl_idx"][order]
    bg = synth["bg"][order]
    cid = synth["crystal_id"][order].long()
    obs_counts = torch.bincount(hkl_id, minlength=n_hkl)

    # Fake "HKL" grid just for adjacency (use 1D line so some edges exist)
    hkl_valid = torch.stack([torch.arange(n_hkl, dtype=torch.int16),
                              torch.zeros(n_hkl, dtype=torch.int16),
                              torch.zeros(n_hkl, dtype=torch.int16)], dim=1)
    adj = build_adjacency(hkl_valid)

    # Quadrature grid
    n_ph = (I_obs + bg).clamp(min=0)
    grid = build_softplus_grid(n_ph, bg, n_quad=51)

    obs_info = ObsInfo(
        crystal_id=cid,
        s_sq=torch.zeros(n_obs),
        hkl_idx=hkl_id,
        n_crystals=synth["n_crystals"],
    )

    prep = PreparedObs(
        hkl_valid=hkl_valid,
        d_spacing=torch.linspace(2.0, 10.0, n_hkl),
        n_valid=n_hkl,
        I_obs=I_obs,
        hkl_id=hkl_id,
        bg=bg,
        obs_counts=obs_counts,
        n_obs_total=n_obs,
        mu_init=torch.full((n_hkl,), math.log(50.0)),
        alpha_init=torch.full((n_hkl,), 0.5),
        beta_perhkl=torch.full((n_hkl,), 1.5),
        grid=grid,
        adjacency=adj,
        obs_info=obs_info,
        partiality=None,
    )

    cfg = MergeConfig(
        lam_alpha_smooth=0.1,
        lam_beta_smooth=100.0,  # heavy → β stays near init
        lam_I0_smooth=0.0,
        obs_chunk_size=chunk_size,
        max_iter=n_lbfgs_steps,
        history_size=10,
    )
    sb = ScaleBCorrection(n_crystals=synth["n_crystals"],
                          scale_prior=1.0, bfactor_prior=1000.0)  # B heavily penalised
    loss = UnifiedMergeLoss(prep, [sb], cfg)
    return loss, prep, cfg


def test_chunked_equals_unchunked_loss_and_grad():
    synth = _make_synthetic(n_hkl=12, n_crystals=8, obs_per_hkl=50, seed=42)
    loss_u, _, _ = _run_loss_on_synth(synth, chunk_size=0)
    loss_c, _, _ = _run_loss_on_synth(synth, chunk_size=37)

    # Match params: need to copy across (new modules have same init so already equal)
    loss_c.load_state_dict(loss_u.state_dict())

    # ── unchunked ──
    for p in loss_u.parameters():
        if p.grad is not None:
            p.grad.zero_()
    total_u = loss_u.data_nll(0, loss_u.n_obs_total)
    total_u.backward()
    grads_u = {n: p.grad.clone() for n, p in loss_u.named_parameters() if p.grad is not None}

    # ── chunked ──
    for p in loss_c.parameters():
        if p.grad is not None:
            p.grad.zero_()
    total_c = torch.zeros(())
    for s, e in loss_c.obs_chunks(37):
        nll = loss_c.data_nll(s, e)
        nll.backward()
        total_c = total_c + nll.detach()
    grads_c = {n: p.grad.clone() for n, p in loss_c.named_parameters() if p.grad is not None}

    assert torch.allclose(total_u.detach(), total_c, rtol=1e-5, atol=1e-4), \
        (total_u.item(), total_c.item())
    for k in grads_u:
        assert torch.allclose(grads_u[k], grads_c[k], rtol=1e-4, atol=1e-4), k


def test_recovery_on_synthetic_minimal():
    """Optimize on synthetic data and check I0 recovery."""
    synth = _make_synthetic(n_hkl=20, n_crystals=15, obs_per_hkl=80, seed=7)
    loss, prep, cfg = _run_loss_on_synth(synth, chunk_size=0, n_lbfgs_steps=80)

    opt = torch.optim.LBFGS(
        list(loss.parameters()),
        lr=1.0, max_iter=cfg.max_iter, history_size=10,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7,
    )

    def closure():
        opt.zero_grad()
        l = loss.data_nll(0, loss.n_obs_total) + loss.regularizers_only()
        l.backward()
        return l

    opt.step(closure)

    I0_fit = torch.exp(loss.log_I0.detach())
    I0_true = torch.exp(synth["log_I0_true"])
    # Correlation should be high; absolute scale may be slightly biased
    cc = torch.corrcoef(torch.stack([I0_fit, I0_true]))[0, 1].item()
    assert cc > 0.98, f"I0 CC too low: {cc:.4f}"
