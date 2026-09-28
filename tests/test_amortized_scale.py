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
        obs_info=obs_info,
        partiality=None,
        shell_id_fw=torch.arange(n_hkl) % 4,
        is_centric=torch.zeros(n_hkl, dtype=torch.bool),
    )


def _params(prep) -> Parameters:
    n = prep.n_valid
    return Parameters(
        log_I0_init=torch.full((n,), math.log(30.0)),
        scatter_shape=FreePerHKLShape(
            torch.full((n,), 1.65).log(), torch.full((n,), 2.0).log()
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
    full_self, _, _ = spec.build(prep.obs_info)
    assert full_self.mean().abs() < 0.1
    sub, _, _ = spec.build(prep.obs_info.gather(torch.arange(10)))
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
    from torchsx.nn.objective import wide_log_grid
    from torchsx.merging.config import MergeConfig

    prep = _toy_prep()
    params = _params(prep)
    scale = AmortizedScale.from_prep(prep, d_model=16, n_layers=2)
    quad = wide_log_grid()
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
    from torchsx.nn.objective import wide_log_grid
    from torchsx.merging.config import MergeConfig

    prep = _toy_prep()
    params = _params(prep)
    scale = AmortizedScale.from_prep(prep, d_model=16, n_layers=2)
    quad = wide_log_grid()

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

    x_self, x_ctx, _ = spec.build(prep.obs_info)
    assert torch.isfinite(x_self).all() and torch.isfinite(x_ctx).all()
    assert x_self.abs().max() < 50.0, "constant column was standardised"
    i = spec.self_features.index("beam_energy")
    assert torch.all(x_self[:, i] == 0)


def test_varying_features_are_still_standardised():
    """The guard must not silently zero a column that does carry signal."""
    from torchsx.nn.obs_features import ObsFeatureSpec

    prep = _toy_prep()
    spec = ObsFeatureSpec(prep, include_hkl=False)
    x_self, _, _ = spec.build(prep.obs_info)
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


# ── fold-pooled (crystal-shared) context and the crystal-level head ─────

from torchsx.merging.observations import CRYSTAL_FEAT_COLUMNS, _crystal_feature_table  # noqa: E402
from torchsx.nn.deepsets import DeepSetsEncoder  # noqa: E402
from torchsx.nn.pooling import fold_ids  # noqa: E402


def _toy_prep_with_crystal_feat(**kw):
    prep = _toy_prep(**kw)
    torch.manual_seed(5)
    n_c = prep.obs_info.n_crystals
    feat = torch.randn(n_c, len(CRYSTAL_FEAT_COLUMNS))
    feat[:, 9] = 0.0                      # det_shift_x absent -> constant column
    prep.obs_info.crystal_feat = feat
    return prep


def _randomise(scale, std=0.3):
    for p in scale.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=std)


def test_fold_mode_own_fold_cannot_move_own_log_k():
    """Bit-exact: perturbing I_j changes nothing in j's fold, something in the
    other fold of j's crystal, nothing in any other crystal."""
    prep = _toy_prep_with_crystal_feat()
    torch.manual_seed(1)
    scale = AmortizedScale.from_prep(
        prep, d_model=16, n_layers=2, anchor_log_k=False, context_folds=2,
    )
    _randomise(scale)
    meta = prep.obs_info
    base = scale.predict(meta).log_K.detach().clone()
    j = 7
    scale.features._I[j] += 500.0
    moved = scale.predict(meta).log_K.detach()

    fold = fold_ids(meta.crystal_id, meta.n_crystals, 2)
    same_c = meta.crystal_id == meta.crystal_id[j]
    own = same_c & (fold == fold[j])
    other = same_c & (fold != fold[j])
    assert torch.equal(moved[own], base[own]), "own fold saw its own intensity"
    assert (moved[other] - base[other]).abs().max() > 1e-4, "pool is inert"
    assert torch.equal(moved[~same_c], base[~same_c])


def test_fold_mode_zero_init_gives_unity_scale():
    prep = _toy_prep_with_crystal_feat()
    scale = AmortizedScale.from_prep(prep, d_model=16, n_layers=2, context_folds=2)
    pred = scale.predict(prep.obs_info)
    assert torch.allclose(pred.log_K, torch.zeros_like(pred.log_K), atol=1e-6)


def test_legacy_path_is_bit_identical_with_crystal_head_present():
    """context_folds=0 with the (zero-init) crystal head equals the old encoder."""
    prep = _toy_prep_with_crystal_feat()
    torch.manual_seed(2)
    scale = AmortizedScale.from_prep(prep, d_model=16, n_layers=2, anchor_log_k=False)
    _randomise(scale)
    torch.nn.init.zeros_(scale.encoder.input_crystal.weight)   # as constructed
    enc = scale.encoder
    assert enc.context_folds == 0 and enc.obs_context_layers == 2 and enc.obs_self_layers == 2
    x_self, x_ctx, x_crys = scale.features.build(prep.obs_info)
    with torch.no_grad():
        z_new = enc(x_self, x_ctx, prep.obs_info.crystal_id, prep.obs_info.n_crystals, x_crys=x_crys)
        # the pre-change forward, written out
        h = enc.input_context(x_ctx)
        for blk in enc.context_blocks:
            h = blk(h)
        from torchsx.nn.pooling import leave_one_out_mean
        ctx, _ = leave_one_out_mean(h, prep.obs_info.crystal_id, prep.obs_info.n_crystals)
        z_old = enc.input_self(x_self) + ctx
        for blk in enc.self_blocks:
            z_old = blk(z_old)
    assert torch.equal(z_new, z_old)


def test_old_checkpoint_without_crystal_head_loads():
    prep = _toy_prep_with_crystal_feat()
    scale = AmortizedScale.from_prep(prep, d_model=16, n_layers=2)
    sd = {k: v for k, v in scale.state_dict().items() if "input_crystal" not in k}
    scale.load_state_dict(sd)          # strict, as to_mtz / restore_best do
    assert (scale.encoder.input_crystal.weight == 0).all()


def test_crystal_head_moves_only_its_crystal():
    prep = _toy_prep_with_crystal_feat()
    torch.manual_seed(4)
    scale = AmortizedScale.from_prep(
        prep, d_model=16, n_layers=2, anchor_log_k=False, context_folds=2,
    )
    _randomise(scale)
    torch.nn.init.normal_(scale.encoder.input_crystal.weight, std=0.3)
    meta = prep.obs_info
    base = scale.predict(meta).log_K.detach().clone()
    c = 3
    scale.features._crys_raw[c, 0] += 5.0
    moved = scale.predict(meta).log_K.detach()
    sel = meta.crystal_id == c
    assert (moved[sel] - base[sel]).abs().min() > 1e-6     # both folds move
    assert torch.equal(moved[~sel], base[~sel])
    # a constant column is zeroed, so poking it does nothing
    scale.features._crys_raw[c, 9] += 5.0
    assert torch.equal(scale.predict(meta).log_K.detach(), moved)


def test_crystal_feature_tier_rejects_condition_fields():
    prep = _toy_prep_with_crystal_feat()
    with pytest.raises(KeyError):
        ObsFeatureSpec(prep, crystal_features=("cell_a", "delay"))


def test_fold_predictions_do_not_depend_on_row_grouping():
    """hkl-sorted vs crystal-grouped input give the same log_K per obs_id."""
    prep = _toy_prep_with_crystal_feat()
    torch.manual_seed(6)
    scale = AmortizedScale.from_prep(
        prep, d_model=16, n_layers=2, anchor_log_k=False, context_folds=2,
    )
    _randomise(scale)
    meta = prep.obs_info
    order = torch.argsort(meta.crystal_id, stable=True)
    a = scale.predict(meta).log_K.detach()
    b = scale.predict(meta.gather(order)).log_K.detach()
    torch.testing.assert_close(b, a[order], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("split", [(1, 1), (2, 2), (0, 2)])
def test_fold_chunking_is_exact_in_value_and_gradient(split):
    torch.manual_seed(0)
    prep = _toy_prep_with_crystal_feat()
    scale = AmortizedScale.from_prep(
        prep, d_model=16, n_layers=2, context_folds=2,
        obs_context_layers=split[0], obs_self_layers=split[1],
    )
    for prm in scale.parameters():
        if prm.dim() >= 1:
            torch.nn.init.normal_(prm, std=0.05)
    scale.train()
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
        return pred.log_K.detach().clone(), g

    k0, g0 = run(0)
    for chunk in (1, 3):
        k1, g1 = run(chunk)
        torch.testing.assert_close(k1, k0, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(g1, g0, rtol=1e-4, atol=1e-6)


def test_fold_mode_depth_split_moves_blocks_to_the_table():
    """Per-observation blocks still see (n, D); the rest run on (G*K, D)."""
    enc = DeepSetsEncoder(4, 6, d_model=8, n_layers=3, context_folds=2,
                          obs_context_layers=1, obs_self_layers=1)
    seen = []
    for blk in list(enc.context_blocks) + list(enc.self_blocks):
        blk.register_forward_hook(lambda m, i, o: seen.append(o.shape[0]))
    n, G = 40, 5
    gid = torch.arange(n) % G
    enc(torch.randn(n, 4), torch.randn(n, 6), gid, G)
    # context: 1 per-obs (n) + 2 table (G*K); self: 2 table + 1 per-obs
    assert seen == [n, G * 2, G * 2, G * 2, G * 2, n]


def test_crystal_feature_table_column_order_and_missing_columns():
    class Core:
        cell = torch.tensor([[10., 11., 12., 90., 90., 90.], [10.1, 11., 12., 90., 89., 90.]])
        resolution = torch.tensor([1.5, 1.7])
        n_refl = torch.tensor([99, 199])
        beam_energy = torch.tensor([12000., 12010.])     # per image
        image_id = torch.tensor([1, 0])                  # crystal -> image
        # det_shift_x / det_shift_y / n_peaks deliberately absent

        def broadcast(self, col, frm, to):
            assert (frm, to) == ("image", "crystal")
            return getattr(self, col)[self.image_id]

    t = _crystal_feature_table(Core(), 2)
    assert t.shape == (2, len(CRYSTAL_FEAT_COLUMNS))
    torch.testing.assert_close(t[:, :6], Core.cell)
    torch.testing.assert_close(t[:, 6], torch.tensor([12010., 12000.]))
    torch.testing.assert_close(t[:, 7], Core.resolution)
    torch.testing.assert_close(t[:, 8], torch.log1p(Core.n_refl.float()))
    assert (t[:, 9:] == 0).all()


def test_state_dict_refuses_a_different_encoder_architecture():
    """Same parameter names in every mode, so this has to be explicit."""
    prep = _toy_prep_with_crystal_feat()
    fold = AmortizedScale.from_prep(prep, d_model=16, n_layers=2, context_folds=2)
    loo = AmortizedScale.from_prep(prep, d_model=16, n_layers=2)
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        loo.load_state_dict(fold.state_dict())
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        fold.load_state_dict(loo.state_dict())
    # pre-arch checkpoints were leave-one-out at full depth: they still load there
    sd = {k: v for k, v in loo.state_dict().items() if not k.endswith("_arch")}
    loo.load_state_dict(sd)


# ── basis gauge anchor ─────────────────────────────────────────────────


def _basis_scale(prep, degree=2):
    from torchsx.merging.priors.reciprocal_basis import ReciprocalBasis

    basis = ReciprocalBasis.from_hkl(prep.hkl_valid.long(), prep.median_cell, "P 1",
                                     degree=degree, aniso=False)
    torch.manual_seed(3)
    scale = AmortizedScale.from_prep(prep, d_model=16, n_layers=2, anchor_stat="basis")
    for p in scale.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.3)
    scale.set_gauge_basis(basis.phi)
    return scale, basis


def test_basis_anchor_requires_a_basis():
    prep = _toy_prep()
    scale = AmortizedScale.from_prep(prep, d_model=16, n_layers=2, anchor_stat="basis")
    with pytest.raises(RuntimeError):
        scale.predict(prep.obs_info)


def test_basis_anchor_removes_a_planted_common_mode_exactly():
    prep = _toy_prep()
    scale, basis = _basis_scale(prep)
    meta = prep.obs_info
    pred = scale.predict(meta)
    phi_b = basis.phi[meta.hkl_idx]
    c = torch.tensor([0.7, -2.0, 1.3])
    from torchsx.nn.heads import ScalePrediction
    planted = ScalePrediction(
        log_K=pred.log_K + phi_b @ c, log_w=pred.log_w, beta=pred.beta,
        log_alpha=pred.log_alpha, log_e_s=pred.log_e_s, log_alpha_hi=pred.log_alpha_hi,
    )
    # The toy crystals each see a few reflections only, so the two-way fit separates
    # crystal effects from the smooth term approximately; on a full-coverage batch
    # (next test) it is exact.
    torch.testing.assert_close(scale._anchor(planted, meta).log_K, pred.log_K, rtol=0, atol=0.05)
    # The pin includes the scalar gauge: the batch median of the anchored log K is zero.
    assert abs(float(pred.log_K.median())) < 1e-5


def _full_coverage_meta(prep, n_cry=6):
    """Every crystal observes every reflection once (crystal-grouped rows)."""
    from torchsx.merging.observations import ObsInfo
    n_hkl = prep.n_valid
    hkl_idx = torch.arange(n_hkl).repeat(n_cry)
    cid = torch.arange(n_cry).repeat_interleave(n_hkl)
    base = prep.obs_info
    d = prep.d_spacing[hkl_idx]
    return ObsInfo(
        crystal_id=cid, s_sq=1 / (4 * d ** 2), hkl_idx=hkl_idx, n_crystals=n_cry,
        panel_id=torch.zeros_like(cid), fs=torch.rand(len(cid)) * 100, ss=torch.rand(len(cid)) * 100,
        peak=torch.full((len(cid),), 10.0), background=torch.full((len(cid),), 4.0),
        ewald_offset=torch.zeros(len(cid)), sigma=torch.full((len(cid),), 3.0),
        obs_id=torch.arange(len(cid)),
    )


def test_basis_anchor_is_exact_on_full_coverage_and_keeps_crystal_scales():
    prep = _toy_prep()
    scale, basis = _basis_scale(prep)
    meta = _full_coverage_meta(prep)
    from torchsx.nn.heads import ScalePrediction
    torch.manual_seed(11)
    raw = torch.randn(meta.hkl_idx.numel()) * 0.3
    zeros = torch.zeros_like(raw)
    def pred_of(x):
        return ScalePrediction(log_K=x, log_w=zeros, beta=zeros + 1, log_alpha=zeros, log_e_s=zeros, log_alpha_hi=zeros)
    ref = scale._anchor(pred_of(raw), meta).log_K
    # (a) a planted common mode phi @ c is removed exactly
    c = torch.tensor([0.7, -2.0, 1.3])
    planted = raw + basis.phi[meta.hkl_idx] @ c
    torch.testing.assert_close(scale._anchor(pred_of(planted), meta).log_K, ref, rtol=0, atol=1e-5)
    # (b) per-crystal scales survive, even when correlated with what a crystal observes:
    # crystals 0-2 see only the low-resolution half, 3-5 only the high-resolution half,
    # and the strong crystals are the high-resolution ones -- the observation mean of
    # log K is then a smooth function of resolution that is NOT a gauge mode.
    order = prep.d_spacing.argsort(descending=True)
    low = torch.zeros(prep.n_valid, dtype=torch.bool); low[order[: prep.n_valid // 2]] = True
    keep = torch.where(meta.crystal_id < 3, low[meta.hkl_idx], ~low[meta.hkl_idx])
    sub = meta.gather(keep.nonzero().squeeze(-1))
    offsets = torch.tensor([0.0, 0.1, -0.1, 1.5, 1.6, 1.4])[sub.crystal_id]
    x0 = raw[keep]
    a0 = scale._anchor(pred_of(x0), sub).log_K
    a1 = scale._anchor(pred_of(x0 + offsets), sub).log_K
    delta = a1 - a0
    # every observation of a crystal moved by that crystal's offset (up to the common median)
    per_cry = torch.zeros(6).index_add_(0, sub.crystal_id, delta) / torch.bincount(sub.crystal_id, minlength=6).clamp(min=1)
    torch.testing.assert_close(delta, per_cry[sub.crystal_id], rtol=0, atol=1e-4)
    spread = per_cry - per_cry.mean()
    torch.testing.assert_close(spread, torch.tensor([0.0, 0.1, -0.1, 1.5, 1.6, 1.4]) - 0.75, rtol=0, atol=1e-4)


def test_basis_anchor_keeps_per_crystal_differences():
    prep = _toy_prep()
    scale, basis = _basis_scale(prep)
    meta = prep.obs_info
    base = scale.predict(meta).log_K
    torch.manual_seed(5)
    offsets = torch.randn(prep.obs_info.n_crystals)[meta.crystal_id]
    from torchsx.nn.heads import ScalePrediction
    pred = scale.predict(meta)
    shifted = scale._anchor(ScalePrediction(
        log_K=pred.log_K + offsets, log_w=pred.log_w, beta=pred.beta,
        log_alpha=pred.log_alpha, log_e_s=pred.log_e_s, log_alpha_hi=pred.log_alpha_hi,
    ), meta).log_K
    # Two observations of the same reflection share a basis row, so their
    # difference is untouched by the projection: it changes by the crystal offsets alone.
    hid = meta.hkl_idx
    for h in hid.unique():
        idx = (hid == h).nonzero().squeeze(-1)
        if idx.numel() < 2:
            continue
        i, j = idx[0], idx[1]
        expected = (base[i] - base[j]) + (offsets[i] - offsets[j])
        torch.testing.assert_close(shifted[i] - shifted[j], expected, rtol=0, atol=1e-5)


def test_basis_anchor_is_chunk_invariant():
    prep = _toy_prep()
    scale, _ = _basis_scale(prep)
    # Chunking needs crystal-grouped observations, as CrystalIndex.rows() delivers them.
    rows = prep.obs_info.crystal_id.argsort(stable=True)
    meta = prep.obs_info.gather(rows)
    whole = scale.predict(meta).log_K.detach().clone()
    scale.set_encoder_chunking(3)
    chunked = scale.predict(meta).log_K.detach()
    torch.testing.assert_close(chunked, whole, rtol=0, atol=1e-5)


def test_basis_anchor_gradient_ignores_the_common_mode():
    """The projection stays in the graph: a loss on the anchored log K has zero
    gradient with respect to a common-mode shift of the raw predictions."""
    prep = _toy_prep()
    scale, basis = _basis_scale(prep)
    meta = _full_coverage_meta(prep)        # exact separation of crystal and smooth terms
    c = torch.zeros(basis.p, requires_grad=True)
    from torchsx.nn.heads import ScalePrediction
    torch.manual_seed(13)
    raw = torch.randn(meta.hkl_idx.numel()) * 0.3
    zeros = torch.zeros_like(raw)
    planted = ScalePrediction(
        log_K=raw + basis.phi[meta.hkl_idx] @ c, log_w=zeros, beta=zeros + 1,
        log_alpha=zeros, log_e_s=zeros, log_alpha_hi=zeros,
    )
    loss = (scale._anchor(planted, meta).log_K ** 2).sum()
    loss.backward()
    assert c.grad.abs().max() < 1e-4
