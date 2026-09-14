"""``Poisson.bg_mean_offset``: the annulus estimate's background-MEAN bias.

``bg`` is ``c·A`` for ``A`` annulus counts and ``c = n_peak/n_bg``, so the true
peak background has predictive mean ``E[B|A] = c(A+1) = bg + c``.  The offset
puts that ``+c`` into the detection mean while the estimate's variance stays
``c·bg`` -- and only into the detection mean: the pooled estimator, the encoder
features and the calibration keep the raw value.

The two things these tests exist to catch:

* the offset silently going missing in one of the paths that reimplement the
  Poisson integrand instead of calling ``log_prob_given_signal`` (the eager
  window, both Triton kernels, the analytic tail, the miss term), and
* the mirror-image failure -- applying it twice, once in the likelihood and
  again in a quadrature that only meant to place its grid.
"""
import math

import pytest
import torch

from torchsx.cli._pipeline import bg_mean_offset_from_ckpt
from torchsx.merging.likelihood import Poisson
from torchsx.merging.quadrature.fixed_log_grid import FixedLogGrid, eager_only
from torchsx.merging.quadrature.split_log_grid import SplitLogGrid, HAVE_TRITON
from torchsx.nn.objective import MarginalNLL
from tests.test_split_log_grid import _batch, _reference, _scatter

CUDA = torch.cuda.is_available()
C = 149 / 128          # the small-molecule geometry this was measured on


# ── the definition itself ─────────────────────────────────────────────

def test_offset_moves_the_mean_and_leaves_the_variance_alone():
    """``c·bg``, not ``c·(bg + c)`` -- the plausible wrong pairing.

    The self-consistent predictive variance really is ``Var[B|A] = c·(bg+c)``;
    ``c·bg`` is what the qualified experiment used and what this reproduces.
    """
    _, bg, _, _, _ = _batch("cpu", M=256)
    bg_mean, sig_b2 = Poisson(bg_var_factor=C, bg_mean_offset=C).detection_params(bg)
    assert torch.equal(bg_mean, bg + C)
    assert torch.equal(sig_b2, C * bg.clamp(min=0.0))
    assert not torch.equal(sig_b2, C * (bg + C).clamp(min=0.0))


def test_offset_zero_is_exactly_the_legacy_likelihood():
    n_ph, bg, _, _, _ = _batch("cpu", M=512)
    S = torch.linspace(-6.0, 10.0, 64).exp()[None, :].expand(n_ph.numel(), 64)
    for bvf in (0.0, C):
        a = Poisson(bg_var_factor=bvf).log_prob_given_signal(n_ph, S, bg)
        b = Poisson(bg_var_factor=bvf,
                    bg_mean_offset=0.0).log_prob_given_signal(n_ph, S, bg)
        assert torch.equal(a, b)


@pytest.mark.parametrize("offset", [0.0, C])
def test_split_eager_matches_dense_reference_under_offset(offset):
    """The window+tail integral against the float64 brute-force grid.

    ``_reference`` integrates via ``log_prob_given_signal``, i.e. the likelihood
    applies the offset itself; ``SplitLogGrid`` instead shifts the background it
    builds its geometry and its inline integrand from.  Agreement is what says
    the two arrived at the same model.
    """
    n_ph, bg, la, lb, mu = _batch("cpu")
    lik = Poisson(bg_var_factor=C, bg_mean_offset=offset)
    ref = _reference(n_ph, bg, la, lb, mu, lik)
    got = SplitLogGrid(n_points=192).marginalize_signal(
        _scatter(la, lb), lik, mu, n_ph, bg, None, None)
    d = (got - ref).abs()
    assert float(d.median()) < 1e-4, float(d.median())
    assert float(d.quantile(0.99)) < 2e-3, float(d.quantile(0.99))
    assert float(d.max()) < 1e-2, (float(d.max()), int(d.argmax()))


def test_offset_changes_the_marginal():
    """Guard against the whole thing quietly becoming a no-op."""
    n_ph, bg, la, lb, mu = _batch("cpu", M=512)
    q = SplitLogGrid(n_points=192)
    off = q.marginalize_signal(_scatter(la, lb), Poisson(C, bg_mean_offset=C),
                               mu, n_ph, bg, None, None)
    on = q.marginalize_signal(_scatter(la, lb), Poisson(C), mu, n_ph, bg, None, None)
    assert float((off - on).abs().max()) > 1e-3


# ── the miss term ─────────────────────────────────────────────────────

@pytest.mark.parametrize("offset", [0.0, C])
def test_log_miss_is_the_grids_own_analytic_tail(offset):
    """``_log_miss`` must be the ``S = 0`` limit of the marginal it is mixed
    with, on the same background -- so it is taken from the cached ``log_L0``."""
    n_ph, bg, _, _, _ = _batch("cpu", M=512)
    lik = Poisson(bg_var_factor=C, bg_mean_offset=offset)
    q = SplitLogGrid(n_points=192)
    obj = MarginalNLL(q, lik, miss_frac=0.03)
    p = q.grid_params(n_ph, bg, q._sig_b2(lik, bg), offset)
    assert torch.equal(obj._log_miss(n_ph, bg), p["log_L0"])
    # and the generic (non-SplitLogGrid) path agrees with it
    assert torch.allclose(obj._log_miss(n_ph, bg),
                          super(SplitLogGrid, q).log_miss(lik, n_ph, bg), atol=0)


# ── caching ───────────────────────────────────────────────────────────

def test_grid_cache_keys_on_the_offset():
    """Same ``(n_ph, bg)`` tensors, different offset: must not be served stale."""
    n_ph, bg, _, _, _ = _batch("cpu", M=512)
    q = SplitLogGrid(n_points=192)
    s2 = C * bg.clamp(min=0)
    q._last_bvf = C
    p0 = q.grid_params(n_ph, bg, s2, 0.0)
    p0_again = q.grid_params(n_ph, bg, s2, 0.0)
    assert p0 is p0_again
    pc = q.grid_params(n_ph, bg, s2, C)
    assert pc is not p0
    assert torch.equal(pc["bg_model"], bg + C)
    assert torch.equal(p0["bg_model"], bg)
    assert float((pc["log_L0"] - p0["log_L0"]).abs().max()) > 1e-3


# ── refusals: never silently drop it ──────────────────────────────────

def test_offset_requires_the_matching_variance_term():
    with pytest.raises(ValueError):
        Poisson(bg_var_factor=0.0, bg_mean_offset=C)
    with pytest.raises(ValueError):
        Poisson(bg_var_factor=C, bg_mean_offset=-1.0)
    with pytest.raises(ValueError):
        Poisson(bg_var_factor=C, bg_mean_offset=float("inf"))


def test_learned_background_is_refused():
    """``grid_params`` is ``no_grad``, so a shifted bg derived inside it would
    be silently detached; and the kernels' d log p/d bg is the plain-Poisson
    statistic, which the shift invalidates."""
    _, bg, _, _, _ = _batch("cpu", M=64)
    lik = Poisson(bg_var_factor=C, bg_mean_offset=C)
    with pytest.raises(RuntimeError):
        lik.detection_params(bg.clone().requires_grad_(True))


def test_quadratures_that_cannot_model_it_raise():
    """A path that hardcodes the plain Poisson says so rather than compute
    the uncorrected quantity."""
    n_ph, bg, la, lb, mu = _batch("cpu", M=64)
    lik = Poisson(bg_var_factor=C, bg_mean_offset=C)
    with pytest.raises(NotImplementedError):
        FixedLogGrid(S_min=1e-6, S_max=1e6, n_points=64) \
            .marginalize_signal_with_reml_aux(
                _scatter(la, lb), lik, mu, n_ph, bg, None, None)


# ── checkpoint round-trip ─────────────────────────────────────────────

def test_checkpoint_offset_round_trip():
    bvf = 1.1640625
    assert bg_mean_offset_from_ckpt({}, bvf) == 0.0
    assert bg_mean_offset_from_ckpt(
        {"quadrature": {"kind": "SplitLogGrid"},
         "likelihood": {"kind": "Poisson"}}, bvf) == 0.0
    assert bg_mean_offset_from_ckpt(
        {"quadrature": {"kind": "SplitLogGrid+bgmean"},
         "likelihood": {"kind": "Poisson+bgmean", "bg_mean_offset": 1.25}}, bvf) == 1.25
    # the out-of-tree experiment wrapper's markers, which record no offset
    assert bg_mean_offset_from_ckpt(
        {"quadrature": {"kind": "AnnulusMeanOnlySplitLogGridV1"},
         "likelihood": {"kind": "AnnulusMeanOnlyPoissonV1"}}, bvf) == bvf
    with pytest.raises(SystemExit):
        bg_mean_offset_from_ckpt(
            {"quadrature": {"kind": "SplitLogGrid"},
             "likelihood": {"kind": "AnnulusMeanOnlyPoissonV1"}}, bvf)


# ── kernels ───────────────────────────────────────────────────────────

def _run_split(la, lb, mu, n_ph, bg, lik, eager):
    import torchsx.merging.quadrature.split_log_grid as slg
    slg._DISABLE_TRITON = eager
    try:
        m, a, b = (t.clone().requires_grad_(True) for t in (mu, la, lb))
        lp = SplitLogGrid(n_points=192).marginalize_signal(
            _scatter(a, b), lik, m, n_ph, bg, None, None)
        g = torch.autograd.grad(lp.sum(), (m, a, b))
        return lp.detach(), g
    finally:
        slg._DISABLE_TRITON = False


@pytest.mark.skipif(not (CUDA and HAVE_TRITON), reason="needs CUDA + triton")
@pytest.mark.parametrize("offset", [0.0, C])
def test_split_triton_matches_eager_under_offset(offset):
    """Kernel parity, at the bars `test_split_triton_matches_eager` calibrated.

    Those are loose (max 3e-2 nats) because the kernel's `lg2.approx` enters as
    `n log(lam/n)`, so a 1e-7 relative log error is ~0.02 nats at n ~ 2e5.  What
    this checks is that the shifted background reaches the kernel at all -- the
    *exactness* of the offset is pinned on CPU by
    `test_split_eager_matches_dense_reference_under_offset`, where the reference
    reaches the offset through `log_prob_given_signal` instead.
    """
    n_ph, bg, la, lb, mu = _batch("cuda", seed=2)
    lik = Poisson(bg_var_factor=C, bg_mean_offset=offset)
    lp_e, g_e = _run_split(la, lb, mu, n_ph, bg, lik, eager=True)
    lp_t, g_t = _run_split(la, lb, mu, n_ph, bg, lik, eager=False)
    d = (lp_e - lp_t).abs()
    assert float(d.median()) < 2e-5, float(d.median())
    assert float(d.quantile(0.99)) < 1e-2, float(d.quantile(0.99))
    assert float(d.max()) < 3e-2, float(d.max())
    for name, ge, gt in zip(("mu", "la", "lb"), g_e, g_t):
        dg = (ge - gt).abs().max().item()
        assert dg < 2e-2 * float(ge.abs().max().clamp(min=1e-6)) + 1e-3, (name, dg)


@pytest.mark.skipif(not (CUDA and HAVE_TRITON), reason="needs CUDA + triton")
@pytest.mark.parametrize("offset", [0.0, C])
def test_fixed_eager_matches_triton_under_offset(offset):
    """The double-apply trap.

    ``FixedLogGrid``'s eager branch delegates to ``log_prob_given_signal``, so
    it must hand over the RAW bg and let the likelihood shift it; its Triton
    branch reimplements the integrand, so it must hand the kernel the shifted
    one.  Getting that backwards applies the offset twice on one path and not at
    all on the other, and this is what notices.
    """
    n_ph, bg, la, lb, mu = _batch("cuda", seed=2, M=1024)
    lik = Poisson(bg_var_factor=C, bg_mean_offset=offset)
    q = FixedLogGrid(S_min=math.exp(-20), S_max=math.exp(20), n_points=1024).to("cuda")
    with eager_only():
        ref = q.marginalize_signal(_scatter(la, lb), lik, mu, n_ph, bg, None, None)
    got = q.marginalize_signal(_scatter(la, lb), lik, mu, n_ph, bg, None, None)
    d = (got - ref).abs()
    # same float32 kernel bars as the split grid; a double application would
    # shift lam by another c and move this by O(1) nat, not 1e-2
    assert float(d.median()) < 2e-5, float(d.median())
    assert float(d.quantile(0.99)) < 1e-2, float(d.quantile(0.99))
    assert float(d.max()) < 3e-2, float(d.max())


# ── end to end through build_stack ────────────────────────────────────

def test_build_stack_and_checkpoint_carry_the_offset(tmp_path):
    """Flag -> likelihood -> checkpoint -> merge, and the refusal in between.

    The ``+bgmean`` kind marker is the load-bearing part: it is what makes a
    build without this parameter REFUSE the checkpoint rather than merge the
    model at the uncorrected background convention.
    """
    import torch as _torch
    from torchsx.cli import _pipeline as P
    from torchsx.cli.merge_nn import _build_parser
    from tests.test_amortized_training import _synthetic
    from tests.test_cli_pipeline import _stack

    prep, params, *_ = _synthetic(n_hkl=20, n_xtal=12, per=10)
    bundle = P.PrepBundle(prep, {"d_min": 0.9, "spacegroup": "P 1", "gain": 1.0,
                                 "radii": [7.0, 9.0, 11.0]}, [], C)
    base = ["ck.pt", "x.sfx", "-o", str(tmp_path / "o.mtz"),
            "--device", "cpu", "--inner-points", "64"]

    # off by default
    args = _build_parser().parse_args(base)
    *_, obj, settings = P.build_stack(bundle, args, verbose=False)
    assert obj.likelihood.bg_mean_offset == 0.0
    assert settings["bg_mean_offset"] == 0.0

    # 'auto' resolves to n_peak/n_bg
    args = _build_parser().parse_args(base + ["--bg-mean-offset", "auto"])
    *_, obj, settings = P.build_stack(bundle, args, verbose=False)
    assert obj.likelihood.bg_mean_offset == pytest.approx(C)
    assert settings["bg_mean_offset"] == pytest.approx(C)

    # ... and survives the checkpoint, markers included
    _scale, _o, tr = _stack(prep, params, tmp_path)
    tr.objective.likelihood.bg_mean_offset = C
    tr.data_recipe = dict(bundle.recipe)
    ck = _torch.load(tr.save_best(tmp_path / "g.pt"), weights_only=False)
    assert ck["likelihood"]["bg_mean_offset"] == pytest.approx(C)
    assert ck["likelihood"]["kind"].endswith("+bgmean")
    assert ck["quadrature"]["kind"].endswith("+bgmean")
    assert P.bg_mean_offset_from_ckpt(ck, C) == pytest.approx(C)

    # a merge inherits it without being told
    args = _build_parser().parse_args(base)
    *_, obj, _s = P.build_stack(bundle, args, ckpt=ck, verbose=False)
    assert obj.likelihood.bg_mean_offset == pytest.approx(C)
