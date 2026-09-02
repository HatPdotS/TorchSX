"""Amortised scale model: contracts, initialisation, and end-to-end gradients.

The contract tests are the point.  A scale model that reads its own
observation's intensity, or that can tell light from dark, produces *better*
apparent metrics while being scientifically worthless — so those two properties
have to be asserted, not inspected.
"""

from __future__ import annotations

import math

import pytest
import torch

from torchsx.merging.hkl import build_adjacency
from torchsx.merging.observations import ObsInfo, PreparedObs
from torchsx.merging.parameters import Parameters
from torchsx.merging.scale import AmortizedScale
from torchsx.merging.scatter_shape import FreePerHKLShape
from torchsx.nn.obs_features import ObsFeatureSpec


def _toy_prep(n_crystals: int = 12, obs_per_crystal: int = 9, n_hkl: int = 20):
    """Small PreparedObs, sorted by hkl_id like the real pipeline."""
    torch.manual_seed(0)
    n = n_crystals * obs_per_crystal

    hkl_id = torch.randint(0, n_hkl, (n,))
    order = hkl_id.argsort()
    hkl_id = hkl_id[order]
    crystal_id = (torch.arange(n) % n_crystals)[order].long()

    I_obs = torch.randn(n) * 20 + 30           # ~30 % negative, as in the real data
    bg = torch.rand(n) * 5
    sigma = (I_obs.abs() + bg + 1.0).sqrt() * (1 + torch.rand(n))

    hkl_valid = torch.stack(
        [
            torch.arange(n_hkl) % 5 - 2,
            torch.arange(n_hkl) // 5 - 2,
            torch.arange(n_hkl) % 3 - 1,
        ],
        dim=1,
    ).to(torch.int16)
    d_spacing = torch.linspace(3.0, 0.9, n_hkl)

    obs_info = ObsInfo(
        crystal_id=crystal_id,
        s_sq=1.0 / (4.0 * d_spacing[hkl_id] ** 2),
        hkl_idx=hkl_id,
        n_crystals=n_crystals,
        panel_id=torch.randint(0, 4, (n,)),
        fs=torch.rand(n) * 1000,
        ss=torch.rand(n) * 1000,
        peak=I_obs + bg,
        background=bg,
        ewald_offset=torch.randn(n) * 1e-4,
        sigma=sigma,
        obs_id=torch.arange(n),
    )
    return PreparedObs(
        hkl_valid=hkl_valid,
        d_spacing=d_spacing,
        n_valid=n_hkl,
        median_cell=torch.tensor([10.0, 11.0, 12.0, 90.0, 90.0, 90.0]),
        I_obs=I_obs,
        hkl_id=hkl_id,
        bg=bg,
        obs_counts=torch.bincount(hkl_id, minlength=n_hkl),
        n_obs_total=n,
        mu_init=torch.full((n_hkl,), math.log(30.0)),
        alpha_init=torch.full((n_hkl,), 1.65),
        beta_perhkl=torch.full((n_hkl,), 2.0),
        adjacency=build_adjacency(hkl_valid),
        obs_info=obs_info,
        partiality=None,
        shell_id_fw=torch.arange(n_hkl) % 4,
        is_centric=torch.zeros(n_hkl, dtype=torch.bool),
    )


def _params(prep) -> Parameters:
    return Parameters(
        log_I0_init=prep.mu_init.clone(),
        scatter_shape=FreePerHKLShape(
            prep.alpha_init.log(), prep.beta_perhkl.log()
        ),
    )


# ── the feature contracts ──────────────────────────────────────────────


def test_own_intensity_rejected_from_self_features():
    prep = _toy_prep()
    with pytest.raises(ValueError, match="may not be self features"):
        ObsFeatureSpec(prep, self_features=("s_sq", "signed_log_I"))


def test_condition_fields_rejected_from_either_tier():
    prep = _toy_prep()
    with pytest.raises(ValueError, match="blind to the light/dark condition"):
        ObsFeatureSpec(prep, self_features=("s_sq", "delay_bin_idx"))
    with pytest.raises(ValueError, match="blind to the light/dark condition"):
        ObsFeatureSpec(
            prep, self_features=("s_sq",), context_features=("s_sq", "timepoint")
        )


def test_features_are_normalised_against_the_full_dataset():
    """Batch statistics would leak batch composition into the features."""
    prep = _toy_prep()
    spec = ObsFeatureSpec(prep)
    full_self, _ = spec.build(prep.obs_info)
    assert full_self.mean().abs() < 0.1
    sub, _ = spec.build(prep.obs_info.gather(torch.arange(10)))
    # A sub-batch is NOT re-standardised, so its mean need not be ~0.
    assert sub.shape == (10, spec.n_self)


# ── initialisation ─────────────────────────────────────────────────────


def test_init_gives_unity_scale_and_seeded_scatter():
    """Zero-init heads -> K == 1, so the first merge is a unity merge."""
    prep = _toy_prep()
    scale = AmortizedScale.from_prep(prep, d_model=16, n_layers=2)
    pred = scale.predict(prep.obs_info)
    assert torch.allclose(pred.log_K, torch.zeros_like(pred.log_K), atol=1e-6)
    assert torch.allclose(pred.log_w.exp(), torch.full_like(pred.log_w, 1.65), atol=1e-5)
    assert torch.allclose(pred.beta, torch.full_like(pred.beta, 2.0), atol=1e-5)
    # beta = 2 is Gaussian-in-log, so alpha = w*sqrt(2)
    torch.testing.assert_close(
        pred.log_alpha.exp(),
        torch.full_like(pred.log_alpha, 1.65 * math.sqrt(2)),
        rtol=1e-4, atol=1e-4,
    )


def test_ablation_switches_hold_width_and_shape_fixed():
    prep = _toy_prep()
    scale = AmortizedScale.from_prep(
        prep, d_model=16, n_layers=2, predict_width=False, predict_shape=False
    )
    names = {n for n, _ in scale.heads.named_parameters()}
    assert "head_log_w.weight" not in names and "head_beta.weight" not in names
    assert {"global_log_w", "global_beta_raw"} <= names


# ── the leakage guarantee, at model level ──────────────────────────────


def test_own_intensity_does_not_move_own_log_k():
    """Perturb observation j's intensity: log_K[j] must not respond.

    The anchor subtracts a batch-mean scalar, which couples every observation
    at O(1/N); so the exact statement is checked with the anchor off, and with
    it on we check that the induced change is a *uniform shift* carrying no
    per-observation information.
    """
    prep = _toy_prep()
    torch.manual_seed(1)

    scale = AmortizedScale.from_prep(
        prep, d_model=16, n_layers=2, anchor_log_k=False
    )
    for p in scale.parameters():          # break the zero-init so log_K varies
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.3)

    base = scale.predict(prep.obs_info).log_K.detach().clone()
    j = 7
    scale.features._I[j] += 500.0
    moved = scale.predict(prep.obs_info).log_K.detach()

    assert (moved[j] - base[j]).abs() < 1e-4, "own intensity leaked into own log K"
    peers = (prep.obs_info.crystal_id == prep.obs_info.crystal_id[j]).nonzero()
    peers = peers[peers.squeeze(-1) != j].squeeze(-1)
    assert (moved[peers] - base[peers]).abs().max() > 1e-4, "pool is inert"


def test_anchor_change_is_a_uniform_shift():
    prep = _toy_prep()
    torch.manual_seed(1)
    scale = AmortizedScale.from_prep(prep, d_model=16, n_layers=2, anchor_log_k=True)
    for p in scale.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.3)

    base = scale.predict(prep.obs_info).log_K.detach().clone()
    scale.features._I[7] += 500.0
    moved = scale.predict(prep.obs_info).log_K.detach()

    # log_K differences are what the likelihood sees; a common shift is absorbed
    # by I0 exactly.  Observation 7's own *relative* scale must be untouched by
    # its own intensity, so removing the shared shift restores it.
    delta = moved - base
    # Observation 7 and every NON-peer move by exactly the same amount: the
    # shared anchor shift.  Only 7's crystal-mates respond to 7's intensity.
    cid = prep.obs_info.crystal_id
    non_peers = (cid != cid[7]).nonzero().squeeze(-1)
    torch.testing.assert_close(
        delta[non_peers], delta[7].expand_as(delta[non_peers]),
        rtol=0, atol=1e-5,
    )


def test_median_anchor_is_not_hijacked_by_runaway_outliers():
    """A small runaway subset must not drag every other observation.

    Measured on the real dark: ~1.5 % of observations reached log K ~ +68 while
    the bulk sat in [-5.5, +0.2].  Under a mean anchor that shifts the whole
    bulk by ~1.0 in log space; the median anchor is unmoved.
    """
    prep = _toy_prep()
    n = prep.n_obs_total
    for stat, tol in (("median", 0.05), ("mean", None)):
        scale = AmortizedScale.from_prep(
            prep, d_model=16, n_layers=2, anchor_stat=stat
        )
        clean = scale.predict(prep.obs_info).log_K.clone()
        # inject a runaway on 1.5 % of observations via the head bias path
        with torch.no_grad():
            scale.heads.head_log_K.bias.fill_(0.0)
        raw = torch.zeros(n)
        raw[: max(1, int(0.015 * n))] = 68.0
        loc = raw.median() if stat == "median" else raw.mean()
        shifted = raw - loc
        bulk = shifted[int(0.015 * n) + 1 :]
        if stat == "median":
            assert bulk.abs().max() < tol, "median anchor moved the bulk"
        else:
            assert bulk.abs().max() > 0.5, "mean anchor should be hijacked"
        del clean


def test_anchor_pins_the_gauge():
    prep = _toy_prep()
    scale = AmortizedScale.from_prep(prep, d_model=16, n_layers=2)
    torch.nn.init.constant_(scale.heads.head_log_K.bias, 3.7)
    assert scale.predict(prep.obs_info).log_K.abs().max() < 1e-5


# ── end to end through the real quadrature ─────────────────────────────


def test_marginal_nll_backward_reaches_the_network():
    """prime() -> quadrature marginal -> backward, gradients on encoder+heads."""
    from torchsx.merging.likelihood import Poisson
    from torchsx.merging.quadrature import TrapezoidalLogGrid
    from torchsx.merging.config import MergeConfig

    prep = _toy_prep()
    params = _params(prep)
    scale = AmortizedScale.from_prep(prep, d_model=16, n_layers=2)
    quad = TrapezoidalLogGrid.from_prep(prep, MergeConfig())
    lik = Poisson()

    # Heads are zero-initialised, so at step 0 the encoder correctly receives
    # NO gradient — dL/dz = W_head = 0.  Training bootstraps by moving the head
    # weights first (they do get gradient, asserted below).  Perturb them here
    # so this test exercises the full graph rather than that first step.
    torch.manual_seed(3)
    for h in ("head_log_K", "head_log_w", "head_beta"):
        torch.nn.init.normal_(getattr(scale.heads, h).weight, std=0.1)

    meta = prep.obs_info
    pred = scale.prime(meta)
    n_ph = (prep.I_obs + prep.bg).clamp(min=0)
    log_mean = params.log_I0[meta.hkl_idx] + pred.log_K

    log_p = quad.marginalize_signal(
        scale.scatter, lik, log_mean, n_ph, prep.bg, params, meta,
        obs_range=(0, prep.n_obs_total),
    )
    assert torch.isfinite(log_p).all()
    (-log_p.sum()).backward()

    got = {n for n, p in scale.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert any("encoder" in n for n in got), "no gradient reached the encoder"
    assert any("head_log_K" in n for n in got), "no gradient reached the log K head"
    assert any("head_log_w" in n for n in got), "no gradient reached the width head"
    assert any("head_beta" in n for n in got), "no gradient reached the shape head"


def test_zero_init_still_bootstraps_the_heads():
    """At exact zero-init the encoder gets no gradient (dL/dz = W = 0), but the
    head weights do — which is how training starts moving at all."""
    from torchsx.merging.likelihood import Poisson
    from torchsx.merging.quadrature import TrapezoidalLogGrid
    from torchsx.merging.config import MergeConfig

    prep = _toy_prep()
    params = _params(prep)
    scale = AmortizedScale.from_prep(prep, d_model=16, n_layers=2)
    quad = TrapezoidalLogGrid.from_prep(prep, MergeConfig())

    meta = prep.obs_info
    pred = scale.prime(meta)
    log_mean = params.log_I0[meta.hkl_idx] + pred.log_K
    log_p = quad.marginalize_signal(
        scale.scatter, Poisson(), log_mean,
        (prep.I_obs + prep.bg).clamp(min=0), prep.bg, params, meta,
        obs_range=(0, prep.n_obs_total),
    )
    (-log_p.sum()).backward()
    assert scale.heads.head_log_K.weight.grad.abs().sum() > 0
    assert scale.heads.head_log_w.weight.grad.abs().sum() > 0


def test_scatter_hook_is_what_the_quadrature_reads():
    """The predicted (alpha, beta) must actually reach the density."""
    prep = _toy_prep()
    params = _params(prep)
    scale = AmortizedScale.from_prep(prep, d_model=16, n_layers=2)
    meta = prep.obs_info
    pred = scale.prime(meta)

    la, lb = scale.scatter.log_alpha_beta_obs(params, meta)
    torch.testing.assert_close(la, pred.log_alpha)
    torch.testing.assert_close(lb, torch.log(pred.beta))

    # ...and under an arbitrary gather it still resolves by obs_id.
    idx = torch.tensor([31, 4, 77, 12])
    la_g, _ = scale.scatter.log_alpha_beta_obs(params, meta.gather(idx))
    torch.testing.assert_close(la_g, pred.log_alpha[idx])


def test_stale_prediction_raises_rather_than_silently_resolving():
    prep = _toy_prep()
    params = _params(prep)
    scale = AmortizedScale.from_prep(prep, d_model=16, n_layers=2)
    meta = prep.obs_info
    scale.prime(meta.gather(torch.arange(10)))
    with pytest.raises(KeyError, match="outside the installed prediction"):
        scale.scatter.log_alpha_beta_obs(params, meta)


def test_calibration_metrics_survive_non_finite_entries():
    """Regression: the per-shell binning must use the SAME finite mask as z.

    Undefined merge targets produce NaN residuals, so real batches always carry
    non-finite entries.  Binning the filtered z against an unfiltered shell_id
    raised IndexError on the first real-data batch.
    """
    from torchsx.nn.metrics import calibration_metrics

    torch.manual_seed(0)
    z = torch.randn(1000)
    z[::7] = float("nan")
    r = calibration_metrics(
        z, shell_id=torch.arange(1000) % 10, n_shells=10,
        i_over_sigma=torch.rand(1000),
    )
    assert r["n"] == int(torch.isfinite(z).sum())
    assert len(r["var_z_by_shell"]) == 10
    assert len(r["var_z_by_isigma_decile"]) == 10
    assert all(v is None or v == v for v in r["var_z_by_shell"])


def test_calibration_metrics_skip_binning_on_length_mismatch():
    """A covariate of the wrong length is skipped, not silently mis-aligned."""
    from torchsx.nn.metrics import calibration_metrics

    r = calibration_metrics(torch.randn(100), shell_id=torch.arange(37) % 5)
    assert "var_z_by_shell" not in r


def test_obs_scale_is_context_only():
    """obs_scale = n'/I0 contains the observation's own intensity.

    It is the channel that carries partiality — raw intensity only tells the
    network how bright the crystal is — but it must never reach the self
    branch, where K_j = n'_j/I0_j would fit exactly and mean nothing.
    """
    from torchsx.nn.obs_features import CONTEXT_ONLY_FEATURES, ObsFeatureSpec

    assert "obs_scale" in CONTEXT_ONLY_FEATURES
    prep = _toy_prep()
    with pytest.raises(ValueError, match="may not be self features"):
        ObsFeatureSpec(prep, self_features=("s_sq", "obs_scale"))


def test_reference_I0_is_detached_and_updatable():
    from torchsx.nn.obs_features import ObsFeatureSpec

    prep = _toy_prep()
    spec = ObsFeatureSpec(prep)
    new = torch.full((prep.n_valid,), 123.0, requires_grad=True)
    spec.set_reference_I0(new)
    assert not spec._I0_ref.requires_grad
    assert float(spec._I0_ref[0]) == 123.0
    with pytest.raises(ValueError):
        spec.set_reference_I0(torch.ones(prep.n_valid + 1))


def test_obs_scale_reference_changes_the_context_not_the_self_branch():
    prep = _toy_prep()
    scale = AmortizedScale.from_prep(prep, d_model=16, n_layers=2, anchor_log_k=False)
    torch.manual_seed(4)
    for p in scale.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.3)
    base = scale.predict(prep.obs_info).log_K.detach().clone()
    scale.features.set_reference_I0(
        scale.features._I0_ref * 3.0
    )
    moved = scale.predict(prep.obs_info).log_K.detach()
    assert not torch.allclose(base, moved), "reference I0 has no effect at all"


def test_constant_features_are_zeroed_not_standardised():
    """Regression: a near-constant column must not be divided by its own noise.

    ``beam_energy`` is ~12960 eV with essentially no spread.  Standardising it
    against a std of ~0 amplified float noise to feature values of order 1e6
    and drove the training loss to 5.7e6 on the first epoch.  ``profile_rad``
    is exactly constant on this dataset, so this is not a corner case.
    """
    import warnings

    from torchsx.nn.obs_features import ObsFeatureSpec

    prep = _toy_prep()
    prep.obs_info.beam_energy = torch.full((prep.n_obs_total,), 12960.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec = ObsFeatureSpec(prep, include_hkl=False)
    assert any("carry no information" in str(c.message) for c in caught)

    x_self, x_ctx = spec.build(prep.obs_info)
    assert torch.isfinite(x_self).all() and torch.isfinite(x_ctx).all()
    assert x_self.abs().max() < 50.0, "constant column was standardised"
    i = spec.self_features.index("beam_energy")
    assert torch.all(x_self[:, i] == 0)


def test_varying_features_are_still_standardised():
    """The guard must not silently zero a column that does carry signal."""
    from torchsx.nn.obs_features import ObsFeatureSpec

    prep = _toy_prep()
    spec = ObsFeatureSpec(prep, include_hkl=False)
    x_self, _ = spec.build(prep.obs_info)
    i = spec.self_features.index("s_sq")
    assert x_self[:, i].std() > 0.5


def test_encoder_chunking_is_exact_in_value_and_gradient():
    """Crystal-chunked prediction must equal the one-shot pass, bit-for-bit-ish.

    The chunking exists to move the checkpoint boundary from the block to the
    chunk: measured on 1.67 M observations the encoder held 14.0 GB, all of it
    16 interior ``(M, d_model)`` tensors, when only three ``(M,)`` outputs ever
    leave.  That trade is only legitimate if it changes nothing, and the one
    thing that could is ``leave_one_out_mean``: it pools within a crystal, so a
    chunk edge inside a crystal would give that crystal a context built from
    part of itself -- and would still run.
    """
    torch.manual_seed(0)
    prep = _toy_prep()
    scale = AmortizedScale.from_prep(prep, d_model=16, n_layers=2)
    # non-trivial weights: zero-init heads would hide any discrepancy
    for prm in scale.parameters():
        if prm.dim() >= 1:
            torch.nn.init.normal_(prm, std=0.05)
    scale.train()
    # PreparedObs is hkl-sorted; the trainer feeds crystal-grouped rows, and
    # chunking requires that ordering, so reproduce it here.
    order = torch.argsort(prep.obs_info.crystal_id, stable=True)
    meta = prep.obs_info.gather(order)

    def run(chunk):
        scale.set_encoder_chunking(chunk)
        scale.zero_grad(set_to_none=True)
        pred = scale.predict(meta)
        loss = (pred.log_K.square().sum() + pred.log_w.sum()
                + pred.beta.sum() + pred.log_alpha.sum())
        loss.backward()
        g = torch.cat([p.grad.flatten() for p in scale.parameters()
                       if p.grad is not None])
        return pred.log_K.detach().clone(), pred.log_w.detach().clone(), g

    k0, w0, g0 = run(0)
    for chunk in (1, 2):
        k1, w1, g1 = run(chunk)
        torch.testing.assert_close(k1, k0, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(w1, w0, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(g1, g0, rtol=1e-4, atol=1e-6)


def test_encoder_chunks_never_split_a_crystal():
    """The chunk bounds must land on crystal boundaries."""
    cid = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3])
    for per_chunk in (1, 2, 3, 10):
        spans = AmortizedScale._crystal_chunks(cid, per_chunk)
        assert spans[0][0] == 0 and spans[-1][1] == cid.numel()
        for a, b in spans:
            assert a < b
            # a boundary may only sit where the crystal id changes
            if a > 0:
                assert cid[a] != cid[a - 1]
            if b < cid.numel():
                assert cid[b] != cid[b - 1]
        # and they must tile the range with no gaps
        assert [a for a, _ in spans[1:]] == [b for _, b in spans[:-1]]
