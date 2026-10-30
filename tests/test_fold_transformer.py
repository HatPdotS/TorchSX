"""Fold transformer: the exclusion, the layout, the identity at step 0, packing and compiling."""
from __future__ import annotations

import itertools

import pytest
import torch

from torchsx.merging.scale import AmortizedScale
from torchsx.nn.fold_transformer import FoldTransformerEncoder, fold_layout, key_prefix_mask
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
        for tokens in (1, 64, 4096, 1 << 19):
            blocks = fold_layout(gid, K, tokens)
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
    enc.tokens = 1 << 19
    a = enc(xs, xc, gid, sizes.numel())
    enc.tokens = 1                                      # one crystal per block
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


def test_packing_is_exact_in_value_and_gradient():
    """The scale's fold path (checkpointed blocks, heads per block, one gather back to
    row order) against the whole batch in one block, rows in arrival order."""
    torch.manual_seed(0)
    prep = _toy_prep_with_crystal_feat()
    scale = _scale(prep)
    for prm in scale.parameters():
        if prm.dim() >= 1:
            torch.nn.init.normal_(prm, std=0.05)
    scale.train()
    meta = prep.obs_info

    def run(tokens):
        scale.encoder.tokens = tokens
        scale.zero_grad(set_to_none=True)
        pred = scale.predict(meta)
        (pred.log_K.square().sum() + pred.log_w.sum()).backward()
        g = torch.cat([p.grad.flatten() for p in scale.parameters() if p.grad is not None])
        return pred.log_K.detach().clone(), g

    k0, g0 = run(1 << 19)
    assert len(scale.encoder.layout(meta.crystal_id)) == 1
    for tokens in (1, 64):                # one crystal per block, a few per block
        k1, g1 = run(tokens)
        torch.testing.assert_close(k1, k0, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(g1, g0, rtol=1e-4, atol=1e-6)
    # the encoder alone, then the heads on every row
    x_self, x_ctx, x_crys = scale.features.build(meta)
    z = scale.encoder(x_self, x_ctx, meta.crystal_id, meta.n_crystals, x_crys=x_crys)
    torch.testing.assert_close(scale.predict(meta).log_K, scale._anchor(scale.heads(z), meta).log_K,
                               rtol=1e-5, atol=1e-6)


def test_packed_block_compiles_once_for_every_length():
    """Blocks of different B and L run one graph: no size is specialised."""
    import torch._dynamo
    torch._dynamo.reset()
    enc = _geo_encoder(query_peers=True).eval()
    fn = torch.compile(enc.encode, dynamic=True, backend="aot_eager")
    with torch.no_grad():
        for i, sizes in enumerate(([20, 24, 17], [40, 33], [9, 12, 7, 11], [60, 58, 61])):
            gid, xs, xc = _geo_batch(sizes, seed=i)
            (idx, valid), = enc.layout(gid)
            with torch._dynamo.config.patch(error_on_recompile=i > 0):
                got = fn(xs, xc, None, gid, idx, valid)[0]
            torch.testing.assert_close(got, enc.encode(xs, xc, None, gid, idx, valid)[0])


def test_measurement_columns_are_the_context_only_features():
    prep = _toy_prep_with_crystal_feat()
    scale = _scale(prep)
    feats = scale.features
    meas = [feats.context_features[i] for i in scale.encoder.meas_idx.tolist()]
    assert set(meas) == set(feats.context_features) - set(feats.self_features)
    assert "signed_log_I" in meas and "log_I_mean" in meas
    assert not set(meas) & set(feats.self_features)


@pytest.mark.parametrize("q_len,kv_len", [(40, 300), (200, 385), (256, 384), (300, 300), (128, 129)])
def test_key_prefix_mask_is_create_block_mask(q_len, kv_len):
    """Block for block (the visited sets; order past each count is free) against
    flex attention's own builder on the dense rule."""
    from torch.nn.attention.flex_attention import create_block_mask
    n_keys = torch.tensor([1, 40, 127, 128, 129, 256, 257, kv_len]).clamp(max=kv_len)
    ours = key_prefix_mask(n_keys, q_len, kv_len)
    ref = create_block_mask(lambda b, h, q, kv: kv < n_keys[b], n_keys.numel(), None, q_len, kv_len,
                            device="cpu")
    assert ours.seq_lengths == ref.seq_lengths
    for num, idx in (("kv_num_blocks", "kv_indices"), ("full_kv_num_blocks", "full_kv_indices"),
                     ("q_num_blocks", "q_indices"), ("full_q_num_blocks", "full_q_indices")):
        a_num, a_idx = getattr(ours, num), getattr(ours, idx)
        b_num, b_idx = getattr(ref, num), getattr(ref, idx)
        assert a_num.shape == b_num.shape and a_idx.shape == b_idx.shape
        assert torch.equal(a_num, b_num), num
        for row in itertools.product(*map(range, a_num.shape)):
            k = int(a_num[row])
            assert sorted(a_idx[row][:k].tolist()) == sorted(b_idx[row][:k].tolist()), (idx, row)


def test_flex_attention_on_the_prefix_mask_is_masked_attention():
    from torch.nn.attention.flex_attention import flex_attention
    g = torch.Generator().manual_seed(0)
    n_keys = torch.tensor([1, 40, 128, 200, 257])
    q, k, v = (torch.randn(5, 2, 260, 16, generator=g) for _ in range(3))
    keep = (torch.arange(260)[None, :] < n_keys[:, None])[:, None, None, :]
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=keep)
    got = flex_attention(q, k, v, block_mask=key_prefix_mask(n_keys, 260, 260))
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


def test_flex_needs_heads_of_at_least_16():
    kw = dict(d_model=32, compile=False, flex=True)
    assert FoldTransformerEncoder(4, 7, [4, 5, 6], n_heads=2, **kw).flex
    assert not FoldTransformerEncoder(4, 7, [4, 5, 6], n_heads=4, **kw).flex


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the compiled stack runs on CUDA only")
@pytest.mark.parametrize("peers,flex", [(False, False), (False, True), (True, True)])
def test_compiled_bf16_is_close_and_keeps_the_exclusion(peers, flex):
    """The compiled bf16 block (flex attention or the masked kernel) against the eager
    float32 reference, and the exclusion bit for bit."""
    import torch._dynamo
    torch._dynamo.reset()
    torch.manual_seed(0)
    kw = dict(d_model=64, n_heads=4, folds=5, ctx_layers=2, query_layers=1, query_peers=peers)
    ref = FoldTransformerEncoder(6, 9, [6, 7, 8], compile=False, bf16=False, **kw).cuda()
    for prm in ref.parameters():
        if prm.dim() > 1:
            torch.nn.init.normal_(prm, std=0.15)
    fast = FoldTransformerEncoder(6, 9, [6, 7, 8], compile=True, bf16=True, flex=flex, **kw).cuda()
    assert fast.flex == flex
    fast.load_state_dict(ref.state_dict())
    sizes = torch.tensor([1, 5, 12, 30, 7, 64, 3, 19, 120])
    gid = torch.repeat_interleave(torch.arange(sizes.numel()), sizes).cuda()
    xs, xc = torch.randn(gid.numel(), 6, device="cuda"), torch.randn(gid.numel(), 9, device="cuda")
    for enc in (ref, fast):
        enc.tokens = 1 << 9                           # several blocks
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
    xc2 = xc.clone()
    xc2[j, 6:] -= 30.0
    fold = fold_ids(gid, sizes.numel(), 5)
    same = gid == gid[j]
    own = same & (fold == fold[j])
    for grad in (True, False):                        # flex attention with autograd, masked without
        with torch.set_grad_enabled(grad):
            base = fast(xs, xc, gid, sizes.numel()).detach()
            moved = fast(xs, xc2, gid, sizes.numel()).detach()
        assert torch.equal(moved[j], base[j])
        if not peers:
            assert torch.equal(moved[own], base[own])
        assert (moved[same & ~own] - base[same & ~own]).abs().max() > 1e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the compiled stack runs on CUDA only")
@pytest.mark.parametrize("q_len", [40, 300])
def test_compiled_flex_on_the_prefix_mask_is_masked_attention(q_len):
    """The inductor kernels on the block mask (a short query block, a ragged one), in
    float32 against masked SDPA: value and gradient."""
    import torch._dynamo
    from torch.nn.attention.flex_attention import flex_attention
    torch._dynamo.reset()
    g = torch.Generator(device="cuda").manual_seed(0)
    n_keys = torch.tensor([1, 40, 128, 200, 256, 257, 300], device="cuda")
    q = torch.randn(7, 2, q_len, 16, device="cuda", generator=g)
    k, v = (torch.randn(7, 2, 300, 16, device="cuda", generator=g) for _ in range(2))
    keep = (torch.arange(300, device="cuda")[None, :] < n_keys[:, None])[:, None, None, :]
    fn = torch.compile(lambda q, k, v, n: flex_attention(q, k, v, block_mask=key_prefix_mask(
        n, q.shape[-2], k.shape[-2])), dynamic=True, fullgraph=True)
    w = torch.randn(7, 2, q_len, 16, device="cuda", generator=g)
    outs = []
    for attend in (lambda *a: fn(*a, n_keys),
                   lambda *a: torch.nn.functional.scaled_dot_product_attention(*a, attn_mask=keep)):
        qkv = [t.clone().requires_grad_() for t in (q, k, v)]
        out = attend(*qkv)
        (out * w).sum().backward()
        outs.append((out.detach(), *(t.grad for t in qkv)))
    for got, ref in zip(*outs):
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the compiled stack runs on CUDA only")
def test_flex_block_is_one_graph_for_every_key_block_count():
    """No graph break, and no size specialised by the number of key blocks (autograd on,
    where flex attention runs; blocks of two or more crystals: one crystal alone is a
    shape of its own)."""
    import torch._dynamo
    torch._dynamo.reset()
    torch.manual_seed(0)
    enc = FoldTransformerEncoder(6, 9, [6, 7, 8], d_model=64, n_heads=4, folds=5, ctx_layers=2,
                                 query_layers=1, bf16=True, flex=True).cuda().eval()
    for prm in enc.parameters():
        if prm.dim() > 1:
            torch.nn.init.normal_(prm, std=0.1)
    fn = torch.compile(enc.encode, dynamic=True, fullgraph=True)
    # every run's queries number at least 128, the key blocks 6 to 11
    for i, sizes in enumerate(([650, 700], [900, 820, 777], [1300, 1250], [640, 1000, 1001])):
        gid, xs, xc = (t.cuda() for t in _geo_batch(sizes, seed=i))
        (idx, valid), = enc.layout(gid)
        with torch._dynamo.config.patch(error_on_recompile=i > 0):
            got = fn(xs, xc, None, gid, idx, valid)[0].detach()
        ref = enc.encode(xs, xc, None, gid, idx, valid)[0].detach()
        assert (got - ref)[valid].norm() / ref[valid].norm() < 1e-3


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
