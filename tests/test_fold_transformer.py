"""Fold transformer: the exclusion, the layout, the identity at step 0, chunking and packing."""
from __future__ import annotations

import pytest
import torch

from torchsx.merging.scale import AmortizedScale
from torchsx.nn.fold_transformer import FoldTransformerEncoder, fold_layout
from torchsx.nn.pooling import fold_ids

from tests.test_amortized_scale import _randomise, _toy_prep_with_crystal_feat


def _scale(prep, *, folds=3, ctx=2, query=1, peers=False, seed=3, heads=4):
    torch.manual_seed(seed)
    return AmortizedScale.from_prep(prep, d_model=16, anchor_log_k=False, encoder="fold",
                                    context_folds=folds, ctx_layers=ctx, query_layers=query,
                                    query_peers=peers, peer_heads=heads)


def _encoder(folds=3, ctx=2, query=1, n_self=4, n_ctx=7, meas=(4, 5, 6), **kw):
    torch.manual_seed(0)
    enc = FoldTransformerEncoder(n_self, n_ctx, list(meas), d_model=16, n_heads=4, folds=folds,
                                 ctx_layers=ctx, query_layers=query, compile=False, **kw)
    for prm in enc.parameters():
        if prm.dim() > 1:
            torch.nn.init.normal_(prm, std=0.3)
    torch.nn.init.normal_(enc.mask_embedding, std=0.3)
    if enc.query_peers:
        torch.nn.init.normal_(enc.peer_embedding, std=0.3)
    return enc


@pytest.mark.parametrize("query,peers", [(0, False), (1, False), (2, True)])
def test_own_fold_cannot_see_own_intensity(query, peers):
    """Bit-exact: perturbing I_j changes nothing in j's own output (nor in j's
    fold, unless the peers are read), something in the crystal's other folds,
    nothing in any other crystal."""
    prep = _toy_prep_with_crystal_feat()
    scale = _scale(prep, query=query, peers=peers)
    _randomise(scale)
    meta = prep.obs_info
    base = scale.predict(meta).log_K.detach().clone()
    j = 7
    scale.features._I[j] += 500.0
    moved = scale.predict(meta).log_K.detach()
    fold = fold_ids(meta.crystal_id, meta.n_crystals, 3)
    same_c = meta.crystal_id == meta.crystal_id[j]
    own = same_c & (fold == fold[j])
    assert torch.equal(moved[j], base[j]), "j saw its own intensity"
    mates = own.clone(); mates[j] = False
    if peers:
        # the fold mates read j's own token as a peer
        assert (moved[mates] - base[mates]).abs().max() > 1e-4, "fold mates do not read j"
    else:
        assert torch.equal(moved[own], base[own]), "j's fold saw j's intensity"
    assert (moved[same_c & ~own] - base[same_c & ~own]).abs().max() > 1e-4, "other folds do not see j"
    assert torch.equal(moved[~same_c], base[~same_c])


@pytest.mark.parametrize("peers", [False, True])
def test_gradient_of_own_fold_to_its_measurement_is_exactly_zero(peers):
    enc = _encoder(query_peers=peers)
    sizes = torch.tensor([1, 4, 9, 13, 6])
    gid = torch.repeat_interleave(torch.arange(sizes.numel()), sizes)
    gid = gid[torch.randperm(gid.numel(), generator=torch.Generator().manual_seed(1))]
    xs = torch.randn(gid.numel(), 4)
    xc = torch.randn(gid.numel(), 7, requires_grad=True)
    z = enc(xs, xc, gid, sizes.numel())
    fold = fold_ids(gid, sizes.numel(), 3)
    j = int((gid == 3).nonzero()[4])
    z[j].square().sum().backward()
    own = (gid == gid[j]) & (fold == fold[j])
    others = (gid == gid[j]) & ~own
    assert torch.equal(xc.grad[j, 4:], torch.zeros_like(xc.grad[j, 4:]))
    mates = own.clone(); mates[j] = False
    if peers:
        assert xc.grad[mates][:, 4:].abs().max() > 0, "the peer read does not reach the fold mates"
    else:
        assert torch.equal(xc.grad[own][:, 4:], torch.zeros_like(xc.grad[own][:, 4:]))
    assert xc.grad[others][:, 4:].abs().max() > 0
    # the geometry columns of x_ctx are not read at all: geometry comes from x_self
    assert torch.equal(xc.grad[:, :4], torch.zeros_like(xc.grad[:, :4]))
    assert torch.equal(xc.grad[gid != gid[j]], torch.zeros_like(xc.grad[gid != gid[j]]))


def test_layout_folds_are_fold_ids():
    sizes = torch.tensor([1, 3, 9, 17, 2, 40, 8, 8, 1, 25])
    gid = torch.repeat_interleave(torch.arange(sizes.numel()), sizes)
    gid = gid[torch.randperm(gid.numel(), generator=torch.Generator().manual_seed(0))]
    for K in (2, 3, 5):
        fold = fold_ids(gid, sizes.numel(), K)
        for budget in (64, 4096, 1 << 24):
            blocks = fold_layout(gid, K, budget)
            seen = torch.cat([idx[valid] for idx, valid in blocks])
            assert torch.equal(seen.sort().values, torch.arange(gid.numel()))
            for idx, valid in blocks:
                L = idx.shape[1]
                assert L % K == 0
                slot = torch.arange(L).expand_as(idx)
                assert torch.equal(fold[idx[valid]], slot[valid] % K)
                assert (gid[idx] == gid[idx[:, :1]]).all()


@pytest.mark.parametrize("peers", [False, True])
def test_step_zero_embedding_is_the_masked_token(peers):
    enc = FoldTransformerEncoder(4, 7, [4, 5, 6], d_model=16, n_heads=4, folds=3, compile=False,
                                 query_peers=peers)
    gid = torch.repeat_interleave(torch.arange(3), torch.tensor([5, 1, 8]))
    xs, xc = torch.randn(gid.numel(), 4), torch.randn(gid.numel(), 7)
    z = enc(xs, xc, gid, 3)
    torch.testing.assert_close(z, enc.input_geometry(xs) + enc.mask_embedding, rtol=0, atol=1e-6)


def test_peers_need_a_query_layer():
    with pytest.raises(ValueError, match="query_layers >= 1"):
        FoldTransformerEncoder(4, 7, [4, 5, 6], d_model=16, n_heads=4, query_layers=0, query_peers=True)


@pytest.mark.parametrize("peers", [False, True])
def test_singleton_crystal_stays_finite(peers):
    enc = _encoder(query_peers=peers)
    gid = torch.tensor([0, 0, 0, 1, 2, 2])
    xs, xc = torch.randn(6, 4), torch.randn(6, 7)
    z = enc(xs, xc, gid, 3)
    assert torch.isfinite(z).all()
    xc2 = xc.clone()
    xc2[3] += 10.0
    assert torch.equal(enc(xs, xc2, gid, 3)[3], z[3])


@pytest.mark.parametrize("peers", [False, True])
def test_packing_does_not_change_the_prediction(peers):
    enc = _encoder(query_peers=peers)
    sizes = torch.tensor([3, 11, 7, 20, 1, 6])
    gid = torch.repeat_interleave(torch.arange(sizes.numel()), sizes)
    xs, xc = torch.randn(gid.numel(), 4), torch.randn(gid.numel(), 7)
    enc.budget = 1 << 24
    a = enc(xs, xc, gid, sizes.numel())
    enc.budget = 64                                     # one crystal per call
    b = enc(xs, xc, gid, sizes.numel())
    torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-6)


def test_predictions_do_not_depend_on_row_grouping():
    prep = _toy_prep_with_crystal_feat()
    scale = _scale(prep)
    _randomise(scale)
    meta = prep.obs_info
    order = torch.argsort(meta.crystal_id, stable=True)
    a = scale.predict(meta).log_K.detach()
    b = scale.predict(meta.gather(order)).log_K.detach()
    torch.testing.assert_close(b, a[order], rtol=1e-5, atol=1e-6)


def test_chunking_is_exact_in_value_and_gradient():
    torch.manual_seed(0)
    prep = _toy_prep_with_crystal_feat()
    scale = _scale(prep)
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
        (pred.log_K.square().sum() + pred.log_w.sum()).backward()
        g = torch.cat([p.grad.flatten() for p in scale.parameters() if p.grad is not None])
        return pred.log_K.detach().clone(), g

    k0, g0 = run(0)
    for chunk in (3, 9):                  # 1 and 3 crystals per chunk at K = 3
        k1, g1 = run(chunk)
        assert scale.chunk_crystals == chunk // 3
        torch.testing.assert_close(k1, k0, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(g1, g0, rtol=1e-4, atol=1e-6)


def test_measurement_columns_are_the_context_only_features():
    prep = _toy_prep_with_crystal_feat()
    scale = _scale(prep)
    feats = scale.features
    meas = [feats.context_features[i] for i in scale.encoder.meas_idx.tolist()]
    assert set(meas) == set(feats.context_features) - set(feats.self_features)
    assert "signed_log_I" in meas and "log_I_mean" in meas
    assert not set(meas) & set(feats.self_features)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the compiled stack runs on CUDA only")
@pytest.mark.parametrize("peers", [False, True])
def test_compiled_bf16_is_close_and_keeps_the_exclusion(peers):
    torch.manual_seed(0)
    kw = dict(d_model=32, n_heads=4, folds=5, ctx_layers=2, query_layers=1, query_peers=peers)
    ref = FoldTransformerEncoder(6, 9, [6, 7, 8], compile=False, bf16=False, **kw).cuda()
    for prm in ref.parameters():
        if prm.dim() > 1:
            torch.nn.init.normal_(prm, std=0.2)
    fast = FoldTransformerEncoder(6, 9, [6, 7, 8], compile=True, bf16=True, **kw).cuda()
    fast.load_state_dict(ref.state_dict())
    sizes = torch.tensor([1, 5, 12, 30, 7, 64, 3, 19, 120])
    gid = torch.repeat_interleave(torch.arange(sizes.numel()), sizes).cuda()
    xs, xc = torch.randn(gid.numel(), 6, device="cuda"), torch.randn(gid.numel(), 9, device="cuda")
    for enc in (ref, fast):
        enc.budget = 1 << 16                          # several buckets
        enc.train()
    outs, grads = [], []
    for enc in (ref, fast):
        enc.zero_grad(set_to_none=True)
        z = enc(xs, xc, gid, sizes.numel())
        z.square().sum().backward()
        outs.append(z.detach())
        grads.append(torch.cat([p.grad.flatten() for p in enc.parameters() if p.grad is not None]))
    assert (outs[1] - outs[0]).norm() / outs[0].norm() < 2e-2
    assert (grads[1] - grads[0]).norm() / grads[0].norm() < 5e-2
    fast.eval()
    j = int(sizes[:5].sum()) + 2                      # a row of the 64-row crystal
    with torch.no_grad():
        base = fast(xs, xc, gid, sizes.numel())
        xc2 = xc.clone()
        xc2[j, 6:] -= 30.0
        moved = fast(xs, xc2, gid, sizes.numel())
    fold = fold_ids(gid, sizes.numel(), 5)
    same = gid == gid[j]
    own = same & (fold == fold[j])
    assert torch.equal(moved[j], base[j])
    if not peers:
        assert torch.equal(moved[own], base[own])
    assert (moved[same & ~own] - base[same & ~own]).abs().max() > 1e-2


# ── geometry correction ────────────────────────────────────────────────

GEO_MEAN = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.1, 9000.0])
GEO_STD = torch.tensor([1e-3, 0.5, 0.5, 0.5, 0.05, 10.0])


def _geo_encoder(refine=True, **kw):
    """Self features = the six geometry columns; context = them plus three measurement columns."""
    torch.manual_seed(0)
    enc = FoldTransformerEncoder(6, 9, [6, 7, 8], d_model=16, n_heads=4, folds=3, ctx_layers=2,
                                 query_layers=1, compile=False, refine_geometry=refine,
                                 geometry={"cols": list(range(6)), "mean": GEO_MEAN, "std": GEO_STD}
                                 if refine else None, **kw)
    for prm in enc.parameters():
        if prm.dim() > 1:
            torch.nn.init.normal_(prm, std=0.3)
    torch.nn.init.normal_(enc.mask_embedding, std=0.3)
    return enc


def _geo_batch(sizes, seed=1):
    g = torch.Generator().manual_seed(seed)
    gid = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
    xs = torch.randn(gid.numel(), 6, generator=g)
    xc = torch.cat([xs, torch.randn(gid.numel(), 3, generator=g)], dim=1)
    return gid, xs, xc


def test_geometry_change_matches_the_exact_rotation_and_energy_shift():
    enc = _geo_encoder().double()
    g = torch.Generator().manual_seed(4)
    direction = torch.nn.functional.normalize(torch.randn(2, 7, 3, generator=g, dtype=torch.float64), dim=-1)
    s_sq = 0.02 + 0.2 * torch.rand(2, 7, 1, generator=g, dtype=torch.float64)
    q0 = 2 * s_sq.sqrt() * direction
    k0 = torch.full((2, 7), 9000.0 / enc.HC, dtype=torch.float64)
    K = 1
    corr = torch.tensor([[0.8, -0.5, 0.3, 1.5], [-0.2, 0.9, -0.7, -2.0]], dtype=torch.float64)
    got = enc._geometry_change(q0, k0, corr, K) * enc.geo_std[:4].double()
    delta = corr[:, :3] * enc.ROT_SCALE
    skew = torch.zeros(2, 3, 3, dtype=torch.float64)
    skew[:, 0, 1], skew[:, 0, 2], skew[:, 1, 2] = -delta[:, 2], delta[:, 1], -delta[:, 0]
    skew = skew - skew.transpose(1, 2)
    q1 = torch.einsum("bij,blj->bli", torch.linalg.matrix_exp(skew), q0)
    k1 = k0 * (1 + corr[:, 3:] * enc.ENERGY_SCALE)
    ewald = lambda v, k: torch.sqrt(v[..., 0] ** 2 + v[..., 1] ** 2 + (v[..., 2] + k) ** 2) - k  # noqa: E731
    torch.testing.assert_close(got[..., 0], ewald(q1, k1) - ewald(q0, k0), rtol=0, atol=2e-6)
    unit = lambda v: v / v.norm(dim=-1, keepdim=True)  # noqa: E731
    torch.testing.assert_close(got[..., 1:], unit(q1) - unit(q0), rtol=0, atol=2e-6)
    # no correction, no change
    assert torch.equal(enc._geometry_change(q0, k0, torch.zeros_like(corr), K), torch.zeros(2, 7, 4, dtype=torch.float64))


def test_refined_geometry_keeps_the_exclusion():
    enc = _geo_encoder()
    for h in enc.geometry_heads:                    # a correction that actually moves the geometry
        torch.nn.init.normal_(h.weight, std=0.5)
    gid, xs, xc = _geo_batch([5, 13, 9])
    xc = xc.clone().requires_grad_(True)
    z = enc(xs, xc, gid, 3)
    assert enc.last_correction.shape == (3 * 3, 4) and enc.last_correction[:, :3].abs().max() > 0
    fold = fold_ids(gid, 3, 3)
    j = int((gid == 1).nonzero()[6])
    z[j].square().sum().backward()
    own = (gid == gid[j]) & (fold == fold[j])
    assert torch.equal(xc.grad[own][:, 6:], torch.zeros_like(xc.grad[own][:, 6:]))
    assert xc.grad[(gid == gid[j]) & ~own][:, 6:].abs().max() > 0
    with torch.no_grad():
        moved_in = xc.detach().clone()
        moved_in[j, 6:] += 5.0
        moved = enc(xs, moved_in, gid, 3)
        base = enc(xs, xc.detach(), gid, 3)
    assert torch.equal(moved[own], base[own])


def test_zero_geometry_heads_change_nothing():
    plain = _geo_encoder(refine=False)
    refined = _geo_encoder(refine=True)
    sd = refined.state_dict()
    sd.update(plain.state_dict())
    refined.load_state_dict(sd)
    for h in refined.geometry_heads:
        torch.nn.init.zeros_(h.weight); torch.nn.init.zeros_(h.bias)
    gid, xs, xc = _geo_batch([4, 11, 7])
    torch.testing.assert_close(refined(xs, xc, gid, 3), plain(xs, xc, gid, 3), rtol=0, atol=0)


def test_geometry_heads_receive_gradient():
    enc = _geo_encoder()
    gid, xs, xc = _geo_batch([6, 12])
    enc(xs, xc, gid, 2).square().sum().backward()
    assert any(h.weight.grad is not None and h.weight.grad.abs().sum() > 0 for h in enc.geometry_heads)


def test_refine_needs_context_and_geometry():
    with pytest.raises(ValueError, match="ctx_layers >= 1"):
        FoldTransformerEncoder(6, 9, [6, 7, 8], d_model=16, n_heads=4, ctx_layers=0, refine_geometry=True,
                               geometry={"cols": list(range(6)), "mean": GEO_MEAN, "std": GEO_STD})
