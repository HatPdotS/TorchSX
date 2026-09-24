"""Peer cross-attention in the Deep Sets encoder: the exclusion, the packing, the gauge of step 0."""
from __future__ import annotations

import pytest
import torch

from torchsx.merging.scale import AmortizedScale
from torchsx.nn.deepsets import DeepSetsEncoder, peer_layout
from torchsx.nn.pooling import fold_ids

from tests.test_amortized_scale import _randomise, _toy_prep_with_crystal_feat


def _scale(prep, *, folds=2, layers=2, budget=None, seed=3, dim=None, heads=4):
    torch.manual_seed(seed)
    s = AmortizedScale.from_prep(prep, d_model=16, n_layers=2, anchor_log_k=False,
                                 context_folds=folds, peer_attention_layers=layers, peer_heads=heads,
                                 peer_dim=dim)
    if budget is not None:
        s.encoder.peer_budget = budget
    return s


@pytest.mark.parametrize("folds", [2, 0])
def test_own_intensity_cannot_move_own_log_k(folds):
    """With every weight random (the zero-init output projections included)."""
    prep = _toy_prep_with_crystal_feat()
    scale = _scale(prep, folds=folds)
    _randomise(scale)
    meta = prep.obs_info
    base = scale.predict(meta).log_K.detach().clone()
    j = 7
    scale.features._I[j] += 500.0
    moved = scale.predict(meta).log_K.detach()
    same_c = meta.crystal_id == meta.crystal_id[j]
    others = same_c.clone()
    others[j] = False
    if folds:
        # the fold pool is exact, so this is the attention's own exclusion, bit for bit
        assert torch.equal(moved[j], base[j]), "j's intensity reached its own scale"
    else:
        # the leave-one-out mean is sum minus self, whose rounding carries j at float
        # precision (test_amortized_scale.test_own_intensity_does_not_move_own_log_k)
        assert (moved[j] - base[j]).abs() < 1e-4, "j's intensity reached its own scale"
    assert (moved[others] - base[others]).abs().max() > 1e-4, "peers do not see j"
    assert torch.equal(moved[~same_c], base[~same_c])
    if folds:
        # unlike the fold pool alone, attention lets j's own fold-mates read j
        fold = fold_ids(meta.crystal_id, meta.n_crystals, folds)
        mates = same_c & (fold == fold[j])
        mates[j] = False
        assert (moved[mates] - base[mates]).abs().max() > 1e-4


def test_zero_init_attention_is_the_encoder_without_it():
    prep = _toy_prep_with_crystal_feat()
    plain = _scale(prep, layers=0)
    _randomise(plain)
    att = _scale(prep, layers=2)
    sd = plain.state_dict()
    for k, v in att.state_dict().items():
        if k not in sd or k.endswith("_arch"):
            sd[k] = v
    att.load_state_dict(sd)
    meta = prep.obs_info
    torch.testing.assert_close(att.predict(meta).log_K, plain.predict(meta).log_K, rtol=0, atol=1e-6)


def test_layout_covers_every_row_once_within_its_crystal():
    torch.manual_seed(0)
    sizes = torch.tensor([1, 3, 9, 17, 2, 40, 8, 8, 1, 25])
    gid = torch.repeat_interleave(torch.arange(sizes.numel()), sizes)
    gid = gid[torch.randperm(gid.numel())]                  # rows not grouped by crystal
    for budget in (64, 4096, 1 << 24):
        blocks = peer_layout(gid, budget)
        seen = torch.cat([idx[valid] for idx, valid in blocks])
        assert torch.equal(seen.sort().values, torch.arange(gid.numel()))
        for idx, valid in blocks:
            L = idx.shape[1]
            assert L % 8 == 0
            assert idx.shape[0] == 1 or idx.shape[0] * L * L <= budget
            assert (gid[idx] == gid[idx[:, :1]]).all(), "a padded row mixes crystals"
            assert torch.equal(valid.sum(1), torch.bincount(gid)[gid[idx[:, 0]]])


def test_bucketing_does_not_change_the_prediction():
    prep = _toy_prep_with_crystal_feat()
    meta = prep.obs_info
    a = _scale(prep, budget=1 << 24)
    _randomise(a)
    b = _scale(prep, budget=64)            # one crystal per call
    b.load_state_dict(a.state_dict())
    b.encoder.peer_budget = 64
    torch.testing.assert_close(b.predict(meta).log_K, a.predict(meta).log_K, rtol=1e-5, atol=1e-6)


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
    for chunk in (1, 3):
        k1, g1 = run(chunk)
        torch.testing.assert_close(k1, k0, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(g1, g0, rtol=1e-4, atol=1e-6)


def test_singleton_crystal_stays_finite_and_gets_no_attention():
    torch.manual_seed(0)
    enc = DeepSetsEncoder(4, 6, d_model=16, n_layers=2, context_folds=2, peer_attention_layers=1, peer_heads=4)
    for prm in enc.parameters():
        if prm.dim() > 1:
            torch.nn.init.normal_(prm, std=0.3)
    gid = torch.tensor([0, 0, 0, 1, 2, 2])                 # crystal 1 has one row
    xs, xc = torch.randn(6, 4), torch.randn(6, 6)
    z = enc(xs, xc, gid, 3)
    assert torch.isfinite(z).all()
    xc2 = xc.clone()
    xc2[3] += 10.0
    assert torch.equal(enc(xs, xc2, gid, 3)[3], z[3])


def test_state_dict_refuses_a_different_attention_depth():
    prep = _toy_prep_with_crystal_feat()
    att, plain = _scale(prep, layers=2), _scale(prep, layers=0)
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        plain.load_state_dict(att.state_dict(), strict=False)
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        att.load_state_dict(plain.state_dict(), strict=False)
    # a checkpoint written before the attention record (3-entry arch) is an encoder without attention
    sd = plain.state_dict()
    sd["encoder._arch"] = sd["encoder._arch"][:3].clone()
    plain.load_state_dict(sd)
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        att.load_state_dict(sd, strict=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the compiled stack runs on CUDA only")
def test_compiled_stack_matches_eager_and_keeps_the_exclusion():
    torch.manual_seed(0)
    kw = dict(d_model=32, n_layers=2, context_folds=2, peer_attention_layers=2, peer_heads=4, peer_bf16=False)
    eager = DeepSetsEncoder(6, 9, peer_compile=False, **kw).cuda()
    for prm in eager.parameters():
        if prm.dim() > 1:
            torch.nn.init.normal_(prm, std=0.2)
    comp = DeepSetsEncoder(6, 9, peer_compile=True, **kw).cuda()
    comp.load_state_dict(eager.state_dict())
    sizes = torch.tensor([1, 5, 12, 30, 7, 64, 3, 19])
    gid = torch.repeat_interleave(torch.arange(sizes.numel()), sizes).cuda()
    xs, xc = torch.randn(gid.numel(), 6, device="cuda"), torch.randn(gid.numel(), 9, device="cuda")
    for enc in (eager, comp):
        enc.peer_budget = 2048                       # several buckets
        enc.train()
    outs, grads = [], []
    for enc in (eager, comp):
        enc.zero_grad(set_to_none=True)
        z = enc(xs, xc, gid, sizes.numel())
        z.square().sum().backward()
        outs.append(z.detach())
        grads.append(torch.cat([p.grad.flatten() for p in enc.parameters() if p.grad is not None]))
    torch.testing.assert_close(outs[1], outs[0], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(grads[1], grads[0], rtol=1e-3, atol=5e-5)
    comp.eval()
    j = int(sizes[:5].sum()) + 2                     # a row of the 64-row crystal
    own, peers = _exclusion_on_cuda(comp, xs, xc, gid, sizes.numel(), j)
    assert own < 1e-5 and peers > 1e-2, (own, peers)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bfloat16 attention runs on CUDA only")
def test_bf16_attention_is_close_and_keeps_the_exclusion():
    torch.manual_seed(1)
    kw = dict(d_model=32, n_layers=2, context_folds=2, peer_attention_layers=2, peer_heads=4, peer_compile=True)
    ref = DeepSetsEncoder(6, 9, peer_bf16=False, **kw).cuda().eval()
    for prm in ref.parameters():
        if prm.dim() > 1:
            torch.nn.init.normal_(prm, std=0.2)
    low = DeepSetsEncoder(6, 9, peer_bf16=True, **kw).cuda().eval()
    low.load_state_dict(ref.state_dict())
    sizes = torch.tensor([2, 40, 9, 120, 1, 33])
    gid = torch.repeat_interleave(torch.arange(sizes.numel()), sizes).cuda()
    xs, xc = torch.randn(gid.numel(), 6, device="cuda"), torch.randn(gid.numel(), 9, device="cuda")
    with torch.no_grad():
        a, b = ref(xs, xc, gid, sizes.numel()), low(xs, xc, gid, sizes.numel())
    rel = (b - a).norm() / a.norm()
    assert rel < 2e-2, f"bf16 attention moved the embedding by {rel:.3g}"
    j = int(sizes[:3].sum()) + 5                     # a row of the 120-row crystal
    own, peers = _exclusion_on_cuda(low, xs, xc, gid, sizes.numel(), j)
    assert own < 1e-5 and peers > 1e-2, (own, peers)


def _exclusion_on_cuda(enc, xs, xc, gid, n_groups, j):
    """``(own, peers)``: how far row ``j`` and its crystal's other rows move when ``x_ctx[j]`` moves.

    Bit-exactness is a CPU statement (the tests above).  On CUDA the pools sum
    by atomics, so every call rounds differently: an unrelated crystal's input
    moves row ``j`` by ~1e-6 in every configuration, eager float32 included.
    A real leak of a 30-unit input would move it by orders of magnitude more.
    """
    with torch.no_grad():
        base = enc(xs, xc, gid, n_groups)
        xc2 = xc.clone()
        xc2[j] -= 30.0
        moved = enc(xs, xc2, gid, n_groups)
    same = gid == gid[j]
    same[j] = False
    return (moved[j] - base[j]).abs().max().item(), (moved[same] - base[same]).abs().max().item()


def test_narrow_deep_attention_keeps_the_exclusion_and_starts_as_the_identity():
    prep = _toy_prep_with_crystal_feat()
    narrow = _scale(prep, layers=4, dim=8, heads=2)
    blk = narrow.encoder.peer_blocks[0]
    assert (blk.w_q.out_features, blk.w_o.in_features, blk.ff.fc1.out_features) == (8, 8, 16)
    per_layer = sum(p.numel() for p in blk.parameters())
    assert per_layer < 0.6 * sum(p.numel() for p in _scale(prep, layers=1).encoder.peer_blocks[0].parameters())
    # step 0: the encoder without attention
    plain = _scale(prep, layers=0)
    _randomise(plain)
    sd = plain.state_dict()
    for k, v in narrow.state_dict().items():
        if k not in sd or k.endswith("_arch"):
            sd[k] = v
    narrow.load_state_dict(sd)
    meta = prep.obs_info
    torch.testing.assert_close(narrow.predict(meta).log_K, plain.predict(meta).log_K, rtol=0, atol=1e-6)
    # the exclusion, bit for bit under the fold pool, with every weight random
    _randomise(narrow)
    base = narrow.predict(meta).log_K.detach().clone()
    j = 7
    narrow.features._I[j] += 500.0
    moved = narrow.predict(meta).log_K.detach()
    same = meta.crystal_id == meta.crystal_id[j]
    assert torch.equal(moved[j], base[j])
    same[j] = False
    assert (moved[same] - base[same]).abs().max() > 1e-4


def test_state_dict_refuses_a_different_attention_width():
    prep = _toy_prep_with_crystal_feat()
    wide, narrow = _scale(prep, layers=2), _scale(prep, layers=2, dim=8, heads=4)
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        narrow.load_state_dict(wide.state_dict(), strict=False)
    # a 5-entry record (before the width existed) is full width
    sd = wide.state_dict()
    sd["encoder._arch"] = sd["encoder._arch"][:5].clone()
    wide.load_state_dict(sd)


def test_tapered_stack_wide_then_narrow():
    prep = _toy_prep_with_crystal_feat()
    taper = _scale(prep, layers=3, dim=[16, 8, 8], heads=4)
    enc = taper.encoder
    assert [b.d_attn for b in enc.peer_blocks] == [16, 8, 8]
    assert [b.n_heads for b in enc.peer_blocks] == [4, 1, 1]
    assert enc.peer_dim == [16, 8, 8] and enc._arch.tolist()[5:] == [16, 8, 8]
    _randomise(taper)
    meta = prep.obs_info
    base = taper.predict(meta).log_K.detach().clone()
    j = 7
    taper.features._I[j] += 500.0
    moved = taper.predict(meta).log_K.detach()
    assert torch.equal(moved[j], base[j])
    uniform = _scale(prep, layers=3, dim=8, heads=4)
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        uniform.load_state_dict(taper.state_dict(), strict=False)
    with pytest.raises(ValueError, match="attention widths"):
        _scale(prep, layers=2, dim=[16, 8, 8])


def test_peer_dim_flag_parses_a_taper():
    from types import SimpleNamespace
    from torchsx.cli._pipeline import _peer_dims, _peer_layers
    assert _peer_dims(SimpleNamespace(peer_dim="128,128,32,32")) == [128, 128, 32, 32]
    assert _peer_layers(SimpleNamespace(peer_dim="128,128,32,32", peer_attention=None)) == 4
    assert _peer_dims(SimpleNamespace(peer_dim="64")) == 64
    assert _peer_layers(SimpleNamespace(peer_dim="64", peer_attention=4)) == 4
    with pytest.raises(SystemExit):
        _peer_layers(SimpleNamespace(peer_dim="128,32", peer_attention=3))
