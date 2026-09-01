"""Leave-one-out pooling: exactness, singletons, and the leakage guarantee.

The structural-leakage test is the important one.  The entire amortised-scale
design rests on ``ctx[j]`` being independent of ``x[j]``; if a refactor ever
breaks that, every downstream metric looks *better*, not worse, so nothing else
in the pipeline would catch it.
"""

import pytest
import torch

from torchsx.nn.pooling import (
    group_count,
    group_sum,
    leave_one_out_mean,
    loo_weighted_mean,
)


def _brute_loo_mean(x, gid, n_groups, fill=0.0):
    """Reference: explicit per-member loop over the group's other members."""
    out = torch.full_like(x, fill)
    for j in range(x.shape[0]):
        peers = (gid == gid[j]).nonzero(as_tuple=True)[0]
        peers = peers[peers != j]
        if peers.numel() > 0:
            out[j] = x[peers].mean(dim=0)
    return out


def _brute_loo_wmean(y, w, gid, n_groups, fill=0.0):
    out = torch.full_like(y, fill)
    for j in range(y.shape[0]):
        peers = (gid == gid[j]).nonzero(as_tuple=True)[0]
        peers = peers[peers != j]
        den = w[peers].sum()
        if den > 1e-12:
            out[j] = (w[peers] * y[peers]).sum() / den
    return out


@pytest.fixture
def ragged():
    """Ragged groups incl. a singleton and an empty slot in the accumulator."""
    torch.manual_seed(0)
    gid = torch.tensor([0, 0, 0, 1, 2, 2, 4, 4, 4, 4])  # group 3 empty, 1 singleton
    return gid, 5


# ── exactness ──────────────────────────────────────────────────────────


def test_loo_mean_matches_bruteforce_1d(ragged):
    gid, n_groups = ragged
    x = torch.randn(gid.shape[0], dtype=torch.float64)
    out, valid = leave_one_out_mean(x, gid, n_groups)
    torch.testing.assert_close(out, _brute_loo_mean(x, gid, n_groups))
    assert valid.tolist() == [True] * 3 + [False] + [True] * 6


def test_loo_mean_matches_bruteforce_2d(ragged):
    gid, n_groups = ragged
    x = torch.randn(gid.shape[0], 7, dtype=torch.float64)
    out, _ = leave_one_out_mean(x, gid, n_groups)
    torch.testing.assert_close(out, _brute_loo_mean(x, gid, n_groups))


def test_loo_mean_singleton_gets_fill(ragged):
    gid, n_groups = ragged
    x = torch.randn(gid.shape[0], 3)
    out, valid = leave_one_out_mean(x, gid, n_groups, fill=-7.0)
    assert not valid[3]
    assert torch.all(out[3] == -7.0)
    assert torch.isfinite(out).all()


def test_loo_wmean_matches_bruteforce(ragged):
    gid, n_groups = ragged
    y = torch.randn(gid.shape[0], dtype=torch.float64)
    w = torch.rand(gid.shape[0], dtype=torch.float64) + 0.1
    out, valid = loo_weighted_mean(y, w, gid, n_groups)
    torch.testing.assert_close(out, _brute_loo_wmean(y, w, gid, n_groups))
    assert not valid[3]


def test_loo_wmean_fill_accepts_per_obs_tensor(ragged):
    """Undefined entries take a per-observation prior, not a global constant."""
    gid, n_groups = ragged
    y = torch.randn(gid.shape[0])
    w = torch.rand(gid.shape[0]) + 0.1
    prior = torch.arange(gid.shape[0], dtype=y.dtype) * 10.0
    out, valid = loo_weighted_mean(y, w, gid, n_groups, fill=prior)
    assert out[3] == prior[3]
    assert torch.equal(out[valid], loo_weighted_mean(y, w, gid, n_groups)[0][valid])


def test_loo_wmean_zero_weights_are_undefined():
    """A group whose peers all carry zero weight has no usable target."""
    gid = torch.tensor([0, 0, 1, 1])
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    w = torch.tensor([1.0, 0.0, 1.0, 1.0])
    out, valid = loo_weighted_mean(y, w, gid, 2, fill=-1.0)
    assert not valid[0]           # its only peer has zero weight
    assert out[0] == -1.0
    assert valid[1] and out[1] == pytest.approx(1.0)


# ── the leakage guarantee ──────────────────────────────────────────────


def test_context_leak_is_purely_rounding():
    """Diagnostic: with a wider accumulator the forward leak vanishes entirely.

    Production runs float32; this test exists only to prove *what* the float32
    residue is.  If it were a real information channel it would survive a
    change of accumulator width.  It does not — so it is rounding, and rounding
    carries no gradient (see ``test_loo_gradient_excludes_self``).
    """
    torch.manual_seed(0)
    gid = torch.randint(0, 12, (120,))
    x = torch.randn(120, 5, dtype=torch.float32)
    base, _ = leave_one_out_mean(x, gid, 12, accum_dtype=torch.float64)

    for j in range(x.shape[0]):
        pert = x.clone()
        pert[j] += 1e3                       # gross perturbation, not epsilon
        out, _ = leave_one_out_mean(pert, gid, 12, accum_dtype=torch.float64)
        assert torch.equal(out[j], base[j]), f"leak at j={j}"
        # ...and its peers *must* move (otherwise the pool is doing nothing)
        peers = (gid == gid[j]).nonzero(as_tuple=True)[0]
        peers = peers[peers != j]
        if peers.numel():
            assert not torch.equal(out[peers], base[peers])


def test_own_value_leak_is_rounding_scale_in_native_dtype():
    """Dtype-preserving accumulation leaks only at rounding scale.

    Sum-minus-self is exact in real arithmetic but not in floating point: the
    sum includes x[j], so perturbing x[j] changes how the sum rounds.  The
    residue is deterministic rounding noise carrying no gradient (see
    ``test_loo_gradient_excludes_self``), but it is not zero, and the module
    docstring must not claim that it is.
    """
    torch.manual_seed(0)
    gid = torch.randint(0, 12, (120,))
    x = torch.randn(120, 5, dtype=torch.float32)
    base, _ = leave_one_out_mean(x, gid, 12)

    worst = 0.0
    for j in range(x.shape[0]):
        pert = x.clone()
        pert[j] += 1e3
        out, _ = leave_one_out_mean(pert, gid, 12)
        worst = max(worst, (out[j] - base[j]).abs().max().item())
    assert 0.0 < worst < 1e-3, worst          # present, but rounding-scale only


def test_merge_target_leak_is_rounding_scale_in_float32(ragged):
    """float32 merge target: own-value influence is rounding only, not signal."""
    gid, n_groups = ragged
    y = torch.randn(gid.shape[0], dtype=torch.float32)
    w = torch.rand(gid.shape[0], dtype=torch.float32) + 0.1
    base, _ = loo_weighted_mean(y, w, gid, n_groups)
    worst = 0.0
    for j in range(y.shape[0]):
        pert = y.clone()
        pert[j] += 1e3
        out, _ = loo_weighted_mean(pert, w, gid, n_groups)
        worst = max(worst, (out[j] - base[j]).abs().item())
    assert worst < 1e-3, worst


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_loo_gradient_excludes_self(dtype):
    """d ctx[j] / d x[j] == 0 EXACTLY; d ctx[peer] / d x[j] == 1/(c-1).

    This is the load-bearing guarantee.  Forward independence is only exact up
    to rounding, but an optimiser moves along gradients, so exact gradient
    independence is what makes the leave-one-out design sound.
    """
    gid = torch.tensor([0, 0, 0, 1, 1])
    x = torch.randn(5, dtype=dtype, requires_grad=True)
    out, _ = leave_one_out_mean(x, gid, 2)
    g_self = torch.autograd.grad(out[0], x, retain_graph=True)[0]
    assert g_self[0].item() == 0.0                       # exact, not approx
    assert torch.allclose(g_self[1:3], torch.full((2,), 0.5, dtype=dtype))
    assert torch.all(g_self[3:] == 0.0)


def test_loo_gradient_excludes_self_2d():
    gid = torch.tensor([0, 0, 0, 1, 1])
    x = torch.randn(5, 4, dtype=torch.float32, requires_grad=True)
    out, _ = leave_one_out_mean(x, gid, 2)
    g = torch.autograd.grad(out[2].sum(), x)[0]
    assert torch.all(g[2] == 0.0)
    assert torch.allclose(g[0:2], torch.full((2, 4), 0.5))


def test_loo_wmean_gradient_excludes_self():
    gid = torch.tensor([0, 0, 0])
    y = torch.randn(3, dtype=torch.float64, requires_grad=True)
    w = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    out, _ = loo_weighted_mean(y, w, gid, 1)
    g = torch.autograd.grad(out[0], y)[0]
    assert g[0].item() == 0.0
    torch.testing.assert_close(g[1:], torch.tensor([0.4, 0.6], dtype=torch.float64))


# ── numerics ───────────────────────────────────────────────────────────


def test_dominant_member_is_flagged_not_silently_wrong():
    """A member holding ~all the weight is invalidated, not silently corrupted.

    In float32 the cancellation ``Σw·y − w_j·y_j`` destroys the dominant
    member's own LOO target (true value 1.0 comes back as 0.0).  The guard must
    catch it.  This is also the statistically correct call: peers holding
    ~0 % of the weight carry no independent information about that observation.
    """
    gid = torch.zeros(6, dtype=torch.long)
    y = torch.tensor([1.0, 1.1, 0.9, 1.05, 0.95, 1e6])
    w = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1e9])
    out, valid = loo_weighted_mean(y, w, gid, 1, fill=-99.0)

    assert not valid[5], "cancellation guard failed to fire"
    assert out[5] == -99.0                      # fill, not 0.0
    # The other five are dragged to the dominant value — inverse-variance says so.
    assert valid[:5].all()
    assert out[0] == pytest.approx(1e6, rel=1e-3)


def test_guard_is_what_catches_the_silent_corruption():
    """Without the guard float32 returns 0.0 for a true target of 1.0 — and
    calls it valid.  With the guard the observation is flagged.

    The weights here are tuned so the *denominator* survives (Sum w - w_j = 5.0)
    and only the numerator cancels; that isolates the numerator guard from the
    ``min_weight`` check, which happens to catch the more extreme 1e9 case.
    """
    gid = torch.zeros(6, dtype=torch.long)
    y = torch.tensor([1.0, 1.1, 0.9, 1.05, 0.95, 1e6])
    w = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1e6])

    off, valid_off = loo_weighted_mean(
        y, w, gid, 1, fill=-99.0, max_rel_error=float("inf")
    )
    assert valid_off[5] and off[5] == 0.0        # silently wrong: truth is 1.0

    on, valid_on = loo_weighted_mean(y, w, gid, 1, fill=-99.0)
    assert not valid_on[5] and on[5] == -99.0


def test_float32_matches_float64_on_realistic_groups():
    """float32 is accurate where it matters: 77 obs/reflection, heavy-tailed w.

    Guards the decision to stay in float32 rather than widen the accumulator.
    """
    torch.manual_seed(0)
    n_h, mult = 200, 77
    gid = torch.arange(n_h).repeat_interleave(mult)
    I0 = torch.rand(n_h) * 990 + 10
    y = (I0[gid] * (1 + 0.3 * torch.randn(n_h * mult))).float()
    sigma = (y.abs().sqrt() + 1.0) * torch.exp(0.8 * torch.randn(n_h * mult))
    w = (1.0 / sigma**2).float()

    ref, _ = loo_weighted_mean(
        y.double(), w.double(), gid, n_h, accum_dtype=torch.float64
    )
    got, valid = loo_weighted_mean(y, w, gid, n_h)

    assert valid.all(), "guard should not fire on realistic data"
    rel = ((got.double() - ref).abs() / ref.abs().clamp(min=1e-6))
    assert rel.median() < 1e-6
    assert rel.max() < 1e-4


def test_group_sum_and_count_handle_empty_groups(ragged):
    gid, n_groups = ragged
    x = torch.ones(gid.shape[0])
    assert group_sum(x, gid, n_groups).tolist() == [3.0, 1.0, 2.0, 0.0, 4.0]
    assert group_count(gid, n_groups).tolist() == [3, 1, 2, 0, 4]


def test_shape_and_dtype_are_preserved(ragged):
    gid, n_groups = ragged
    x = torch.randn(gid.shape[0], 4, dtype=torch.float32)
    out, valid = leave_one_out_mean(x, gid, n_groups, accum_dtype=torch.float64)
    assert out.shape == x.shape and out.dtype == torch.float32
    assert valid.shape == (x.shape[0],) and valid.dtype == torch.bool


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        leave_one_out_mean(torch.randn(5), torch.zeros(4, dtype=torch.long), 1)
    with pytest.raises(ValueError):
        loo_weighted_mean(
            torch.randn(5), torch.randn(4), torch.zeros(5, dtype=torch.long), 1
        )


# ── EMA blend via the prior pseudo-observation ─────────────────────────


def test_prior_pseudo_observation_blends_exactly():
    """prior_num/prior_den must reproduce the blend formula term for term."""
    gid = torch.tensor([0, 0, 0, 1, 1])
    y = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0])
    w = torch.tensor([1.0, 1.0, 2.0, 1.0, 3.0])
    lam, den_ema, I0_ema = 2.0, torch.tensor([4.0, 5.0]), torch.tensor([7.0, 9.0])

    out, _ = loo_weighted_mean(
        y, w, gid, 2,
        prior_num=lam * den_ema * I0_ema,
        prior_den=lam * den_ema,
    )
    # observation 0: peers are 1 and 2, plus the group-0 pseudo-observation
    num = (w[1] * y[1] + w[2] * y[2]) + lam * den_ema[0] * I0_ema[0]
    den = (w[1] + w[2]) + lam * den_ema[0]
    assert out[0] == pytest.approx((num / den).item(), rel=1e-6)


def test_prior_does_not_break_leave_one_out():
    gid = torch.tensor([0, 0, 0, 0])
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    w = torch.ones(4)
    kw = dict(prior_num=torch.tensor([50.0]), prior_den=torch.tensor([5.0]))
    base, _ = loo_weighted_mean(y, w, gid, 1, **kw)
    pert = y.clone()
    pert[2] += 100.0
    moved, _ = loo_weighted_mean(pert, w, gid, 1, **kw)
    assert abs((moved[2] - base[2]).item()) < 1e-4


def test_large_lambda_recovers_the_fixed_target():
    """lambda -> inf is the EM fallback: the target stops depending on the batch."""
    gid = torch.zeros(4, dtype=torch.long)
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    w = torch.ones(4)
    I0_ema = torch.tensor([9.0])
    out, _ = loo_weighted_mean(
        y, w, gid, 1,
        prior_num=1e8 * I0_ema, prior_den=torch.tensor([1e8]),
    )
    torch.testing.assert_close(out, I0_ema.expand(4), rtol=1e-5, atol=1e-4)
