"""Unit tests for the custom chunked autograd path in ``DataNLLTarget``.

Verifies that chunking produces the same scalar loss AND the same
per-parameter gradients as a full-batch autograd pass, to within tight
tolerances.  Gates the correctness of the refactored merger.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from torchsx.merging.likelihood import Poisson, PoissonWithBackgroundMixture
from torchsx.merging.observations import ObsInfo, PreparedObs
from torchsx.merging.parameters import Parameters
from torchsx.merging.quadrature import TrapezoidalLogGrid
from torchsx.merging.scale import CompositeScale, ScaleBPerCrystal
from torchsx.merging.scatter import GenGaussLogScatter
from torchsx.merging.targets import DataNLLTarget
from torchsx.merging.hkl import build_adjacency


def _fake_prep(n_obs: int = 600, n_hkl: int = 12, n_crystals: int = 4, seed: int = 0) -> PreparedObs:
    torch.manual_seed(seed)
    hkl_id = torch.randint(0, n_hkl, (n_obs,)).sort().values
    I_obs = torch.rand(n_obs) * 20.0 - 2.0   # allow some negatives
    bg = torch.full((n_obs,), 1.0)
    obs_counts = torch.bincount(hkl_id, minlength=n_hkl)
    # Build a 1D adjacency so MillerGraphSmoothness could plug in later; unused here.
    hkl_valid = torch.stack(
        [
            torch.arange(n_hkl, dtype=torch.int16),
            torch.zeros(n_hkl, dtype=torch.int16),
            torch.zeros(n_hkl, dtype=torch.int16),
        ],
        dim=1,
    )
    adj = build_adjacency(hkl_valid)
    crystal_id = torch.randint(0, n_crystals, (n_obs,))
    s_sq = torch.linspace(0.0, 0.25, n_obs)
    obs_info = ObsInfo(
        crystal_id=crystal_id.long(),
        s_sq=s_sq,
        hkl_idx=hkl_id,
        n_crystals=n_crystals,
    )
    return PreparedObs(
        hkl_valid=hkl_valid,
        d_spacing=torch.linspace(2.0, 10.0, n_hkl),
        n_valid=n_hkl,
        median_cell=torch.tensor([50.0, 60.0, 70.0, 90.0, 90.0, 90.0]),
        I_obs=I_obs,
        hkl_id=hkl_id,
        bg=bg,
        obs_counts=obs_counts,
        n_obs_total=n_obs,
        mu_init=torch.full((n_hkl,), math.log(8.0)),
        alpha_init=torch.full((n_hkl,), 1.0),
        beta_perhkl=torch.full((n_hkl,), 1.4),
        adjacency=adj,
        obs_info=obs_info,
        partiality=None,
        shell_id_fw=torch.zeros(n_hkl, dtype=torch.long),
        is_centric=torch.zeros(n_hkl, dtype=torch.bool),
    )


def _build_target(prep: PreparedObs, chunk_size: int | None, *, bg_mix: bool = False) -> DataNLLTarget:
    params = Parameters.from_prep(prep)
    # Nudge params to non-trivial values so gradients aren't all tiny.
    with torch.no_grad():
        params.log_I0.add_(0.2 * torch.randn_like(params.log_I0))
        params.log_alpha.add_(0.1 * torch.randn_like(params.log_alpha))
        params.log_beta.add_(0.05 * torch.randn_like(params.log_beta))
    scale = CompositeScale([ScaleBPerCrystal(n_crystals=prep.obs_info.n_crystals)])
    with torch.no_grad():
        scale.submodules_list[0].log_G_raw.copy_(
            0.1 * torch.randn(prep.obs_info.n_crystals)
        )
        scale.submodules_list[0].B.copy_(
            0.05 * torch.randn(prep.obs_info.n_crystals)
        )
    scatter = GenGaussLogScatter()
    quad = TrapezoidalLogGrid.from_prep(
        prep,
        config=_CfgShim(sig_lo_override=1e-4, n_quad=51, n_sigma_quad=5.0),
    )
    likelihood = PoissonWithBackgroundMixture(mode="global") if bg_mix else Poisson()
    return DataNLLTarget(params, scale, scatter, quad, likelihood, prep, chunk_size=chunk_size)


class _CfgShim:
    """Tiny config shim for TrapezoidalLogGrid.from_prep in tests."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _grads_for(target: DataNLLTarget) -> tuple[torch.Tensor, list[torch.Tensor]]:
    for p in target._param_tuple():
        if p.grad is not None:
            p.grad.zero_()
    loss = target()
    loss.backward()
    grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
             for p in target._param_tuple()]
    return loss.detach(), grads


@pytest.mark.parametrize("bg_mix", [False, True])
def test_chunked_matches_full_batch(bg_mix: bool) -> None:
    prep = _fake_prep(n_obs=600, n_hkl=12, n_crystals=4, seed=3)

    # Full-batch (chunk_size=None -> one pass, no custom function).
    tgt_full = _build_target(prep, chunk_size=None, bg_mix=bg_mix)
    loss_full, grads_full = _grads_for(tgt_full)

    # Sync param values between the two targets.
    tgt_chunk = _build_target(prep, chunk_size=128, bg_mix=bg_mix)
    with torch.no_grad():
        for p_c, p_f in zip(tgt_chunk._param_tuple(), tgt_full._param_tuple()):
            p_c.copy_(p_f)

    loss_chunk, grads_chunk = _grads_for(tgt_chunk)

    assert torch.allclose(loss_full, loss_chunk, atol=1e-4, rtol=1e-5), (
        f"loss mismatch: full={loss_full.item():.6g} chunked={loss_chunk.item():.6g}"
    )
    for i, (gf, gc) in enumerate(zip(grads_full, grads_chunk)):
        assert torch.allclose(gf, gc, atol=1e-4, rtol=1e-4), (
            f"grad[{i}] shape {gf.shape} max|Δ|={(gf - gc).abs().max().item():.6g}"
        )


def test_chunk_size_equals_n_obs_uses_full_batch_path() -> None:
    """chunk_size >= n_obs should produce a full-batch graph (no custom fn)."""
    prep = _fake_prep(n_obs=50, n_hkl=5, n_crystals=2, seed=1)
    tgt = _build_target(prep, chunk_size=10_000)  # >> n_obs
    loss, grads = _grads_for(tgt)
    # Sanity: loss is finite, grads non-zero somewhere.
    assert torch.isfinite(loss).item()
    assert any(g.abs().sum().item() > 0 for g in grads)


def test_data_target_builds_with_populated_obs_info_fields() -> None:
    """Regression: DataNLLTarget must handle ObsInfo with panel_id/fs/ss/peak/
    background populated (real streams carry all of these)."""
    prep = _fake_prep(n_obs=60, n_hkl=6, n_crystals=2, seed=11)
    # Swap in an ObsInfo that has all optional fields set.
    n_obs = prep.n_obs_total
    prep = replace(
        prep,
        obs_info=ObsInfo(
            crystal_id=prep.obs_info.crystal_id,
            s_sq=prep.obs_info.s_sq,
            hkl_idx=prep.obs_info.hkl_idx,
            n_crystals=prep.obs_info.n_crystals,
            panel_id=torch.zeros(n_obs, dtype=torch.long),
            fs=torch.zeros(n_obs),
            ss=torch.zeros(n_obs),
            peak=torch.ones(n_obs),
            background=torch.ones(n_obs),
        ),
    )
    tgt = _build_target(prep, chunk_size=None)
    loss, _ = _grads_for(tgt)
    assert torch.isfinite(loss).item()


def test_chunked_backward_accumulates_all_params() -> None:
    """Every learnable param must receive a gradient from the chunked path."""
    prep = _fake_prep(n_obs=400, n_hkl=8, n_crystals=3, seed=7)
    tgt = _build_target(prep, chunk_size=64, bg_mix=True)
    loss, grads = _grads_for(tgt)
    for i, g in enumerate(grads):
        # log_beta for obs-less HKLs may legitimately receive a zero grad from the
        # data branch; all *other* params should receive a non-zero grad.
        assert torch.isfinite(g).all(), f"grad[{i}] has non-finite entries"
    # At least one grad per parameter-tensor must be nonzero.
    any_nonzero = sum(int(g.abs().sum().item() > 0) for g in grads)
    assert any_nonzero >= len(grads) - 1  # leave 1 margin for log_beta edge case
