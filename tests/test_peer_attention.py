"""Peer-attention encoder: the leave-one-out contract under attention.

Zeroing the query's intensity is NOT sufficient on its own — if token j attends
to itself its intensity returns through the *value* and the objective collapses
to the tautology K_j = n'_j / I0_j.  The diagonal mask is what makes it sound,
and that is what these tests pin down.
"""

from __future__ import annotations

import pytest
import torch

from torchsx.nn.attention import PeerAttentionEncoder
from torchsx.nn.deepsets import ScaleEncoder


def _enc(**kw):
    torch.manual_seed(0)
    e = PeerAttentionEncoder(6, 6, d_model=32, n_layers=2, n_heads=4, **kw)
    for p in e.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.2)
    e.eval()
    return e


def test_satisfies_the_encoder_protocol():
    assert isinstance(_enc(), ScaleEncoder)


def test_shares_the_embedding_when_widths_match():
    assert _enc().shared_embedding
    e = PeerAttentionEncoder(4, 9, d_model=16, n_layers=1, n_heads=2)
    assert not e.shared_embedding


def test_own_token_cannot_influence_its_own_output():
    """THE test.  Perturb token j's key/value embedding; output j must not move.

    Its crystal-mates must move, otherwise the attention is inert and the
    invariance is vacuous.
    """
    enc = _enc()
    n = 48
    gid = torch.arange(n) % 6
    x_self = torch.randn(n, 6)
    x_ctx = torch.randn(n, 6)
    base = enc(x_self, x_ctx, gid, 6)

    for j in (0, 7, 23, 41):
        pert = x_ctx.clone()
        pert[j] += 50.0
        out = enc(x_self, pert, gid, 6)
        assert torch.allclose(out[j], base[j], atol=1e-5), f"self-leak at j={j}"
        peers = (gid == gid[j]).nonzero().squeeze(-1)
        peers = peers[peers != j]
        assert not torch.allclose(out[peers], base[peers], atol=1e-4), "attention inert"


def test_output_order_matches_input_order():
    """The encoder sorts by crystal internally; it must invert that exactly."""
    enc = _enc()
    n = 40
    gid = torch.randint(0, 5, (n,))
    x_self, x_ctx = torch.randn(n, 6), torch.randn(n, 6)
    out = enc(x_self, x_ctx, gid, 5)

    perm = torch.randperm(n)
    out_p = enc(x_self[perm], x_ctx[perm], gid[perm], 5)
    torch.testing.assert_close(out_p, out[perm], rtol=1e-4, atol=1e-4)


def test_singleton_crystal_has_no_peers_and_stays_finite():
    enc = _enc()
    gid = torch.tensor([0, 1, 1, 2, 2, 2])
    out = enc(torch.randn(6, 6), torch.randn(6, 6), gid, 3)
    assert torch.isfinite(out).all(), "singleton crystal produced non-finite output"


def test_token_mode_zeroes_the_query_measurement_channels():
    from torchsx.nn.obs_features import CONTEXT_ONLY_FEATURES

    import tests.test_amortized_scale as t
    prep = t._toy_prep()
    from torchsx.nn.obs_features import ObsFeatureSpec

    spec = ObsFeatureSpec(prep, token_mode=True, include_hkl=False)
    x_self, x_ctx = spec.build(prep.obs_info)
    assert x_self.shape == x_ctx.shape, "token mode must give one shared width"
    mask = spec.measurement_mask
    assert mask.any(), "no measurement channels identified"
    assert torch.all(x_self[:, mask] == 0), "query keeps a measurement channel"
    assert torch.equal(x_self[:, ~mask], x_ctx[:, ~mask])
