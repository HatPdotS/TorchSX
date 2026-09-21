"""``Poisson.bg_var``: a per-observation variance of the background estimate.

Where the annulus model derives ``sig_b2 = c * bg`` from the stored background,
a producer that measured its estimate's error (probixi's noise-model background
under the disk) hands the variance over as data. The likelihood gathers it by
``meta.obs_id``, so every path that reimplements the Poisson integrand has to
carry the batch's metadata down to the detection model. These tests pin that
plumbing against the scalar-factor model it must reproduce exactly when given
the same numbers, and the checkpoint markers that keep an old build from
merging such a model at the annulus convention.
"""
from types import SimpleNamespace

import math

import pytest
import torch

from torchsx.merging.likelihood import Poisson
from torchsx.merging.quadrature.fixed_log_grid import FixedLogGrid, eager_only
from torchsx.merging.quadrature.split_log_grid import SplitLogGrid, HAVE_TRITON
from torchsx.nn.objective import MarginalNLL
from tests.test_split_log_grid import _batch, _scatter

CUDA = torch.cuda.is_available()
C = 149 / 128


def _meta(n: int, device="cpu", offset: int = 0):
    """A minimal ObsInfo: only ``obs_id`` matters here, but slicing must work."""
    from torchsx.merging.observations import ObsInfo
    zeros = torch.zeros(n, dtype=torch.long, device=device)
    return ObsInfo(crystal_id=zeros, s_sq=zeros.float(), hkl_idx=zeros, n_crystals=1,
                   obs_id=torch.arange(offset, offset + n, device=device))


def _pair(device="cpu", M=4096, seed=0):
    """The factor model and a per-observation model holding the same variances."""
    n_ph, bg, la, lb, mu = _batch(device, seed=seed, M=M)
    factor = Poisson(bg_var_factor=C)
    per_obs = Poisson(bg_var=C * bg.clamp(min=0.0)).to(device)
    return n_ph, bg, la, lb, mu, factor, per_obs


# ── the definition ────────────────────────────────────────────────────

def test_per_observation_variance_is_gathered_by_obs_id():
    n_ph, bg, *_ , factor, per_obs = _pair(M=256)
    assert per_obs.has_bg_var and not factor.has_bg_var
    mean_f, s2_f = factor.detection_params(bg)
    mean_v, s2_v = per_obs.detection_params(bg, _meta(256))
    assert torch.equal(mean_v, bg)            # no +c: the estimate is not an annulus count
    assert torch.allclose(s2_v, s2_f)
    # a permuted batch reads the permuted rows
    perm = torch.randperm(256)
    _, s2_p = per_obs.detection_params(bg[perm], SimpleNamespace(obs_id=perm))
    assert torch.allclose(s2_p, s2_f[perm])


def test_per_observation_variance_needs_the_batch_metadata():
    n_ph, bg, *_, per_obs = _pair(M=64)
    with pytest.raises(ValueError, match="obs_id"):
        per_obs.detection_params(bg)
    with pytest.raises(ValueError, match="rows"):
        per_obs.detection_params(bg, _meta(32))


def test_per_observation_variance_excludes_the_annulus_terms():
    bg_var = torch.ones(8)
    with pytest.raises(ValueError, match="annulus"):
        Poisson(bg_var_factor=C, bg_var=bg_var)
    with pytest.raises(ValueError, match="annulus"):
        Poisson(bg_var_factor=C, bg_mean_offset=C, bg_var=bg_var)
    with pytest.raises(ValueError, match="finite"):
        Poisson(bg_var=torch.tensor([1.0, float("nan")]))
    with pytest.raises(ValueError, match="finite"):
        Poisson(bg_var=torch.tensor([1.0, -1.0]))


# ── every integrand path ──────────────────────────────────────────────

def test_split_eager_matches_the_factor_model():
    n_ph, bg, la, lb, mu, factor, per_obs = _pair()
    meta = _meta(n_ph.numel())
    ref = SplitLogGrid(n_points=192).marginalize_signal(_scatter(la, lb), factor, mu, n_ph, bg, None, meta)
    got = SplitLogGrid(n_points=192).marginalize_signal(_scatter(la, lb), per_obs, mu, n_ph, bg, None, meta)
    assert torch.allclose(got, ref, atol=1e-6, rtol=0)


def test_fixed_eager_matches_the_factor_model():
    n_ph, bg, la, lb, mu, factor, per_obs = _pair(M=512)
    meta = _meta(n_ph.numel())
    q = FixedLogGrid(S_min=math.exp(-20), S_max=math.exp(20), n_points=512)
    with eager_only():
        ref = q.marginalize_signal(_scatter(la, lb), factor, mu, n_ph, bg, None, meta)
        got = q.marginalize_signal(_scatter(la, lb), per_obs, mu, n_ph, bg, None, meta)
    assert torch.allclose(got, ref, atol=1e-6, rtol=0)


def test_log_prob_given_signal_takes_the_metadata():
    n_ph, bg, *_, factor, per_obs = _pair(M=64)
    S = torch.linspace(0.1, 50.0, 7)[None, :].expand(64, 7)
    ref = factor.log_prob_given_signal(n_ph, S, bg)
    got = per_obs.log_prob_given_signal(n_ph, S, bg, _meta(64))
    assert torch.allclose(got, ref, atol=1e-6, rtol=0)


def test_miss_term_and_mixture_carry_the_metadata():
    n_ph, bg, la, lb, mu, factor, per_obs = _pair(M=512)
    meta = _meta(512)
    q_f, q_v = SplitLogGrid(n_points=192), SplitLogGrid(n_points=192)
    obj_f = MarginalNLL(q_f, factor, miss_frac=0.03)
    obj_v = MarginalNLL(q_v, per_obs, miss_frac=0.03)
    assert torch.allclose(obj_v._log_miss(n_ph, bg, meta), obj_f._log_miss(n_ph, bg, meta), atol=1e-6)
    # the generic path agrees with the grid's own tail
    assert torch.allclose(obj_v._log_miss(n_ph, bg, meta),
                          super(SplitLogGrid, q_v).log_miss(per_obs, n_ph, bg, meta), atol=1e-5)
    lp_f = obj_f.log_prob(_scatter(la, lb), None, mu, n_ph, bg, meta)
    lp_v = obj_v.log_prob(_scatter(la, lb), None, mu, n_ph, bg, meta)
    assert torch.allclose(lp_v, lp_f, atol=1e-6, rtol=0)
    # chunked evaluation slices the metadata along with the data (compared
    # against the factor model chunked the same way, so only the gather is tested)
    obj_f.chunk_size = obj_v.chunk_size = 100
    chunked = _scatter(la, lb)
    chunked.log_alpha_beta_obs = lambda params, m: (la[m.obs_id], lb[m.obs_id])  # noqa: E731
    lp_fc = obj_f.log_prob(chunked, None, mu, n_ph, bg, meta)
    lp_vc = obj_v.log_prob(chunked, None, mu, n_ph, bg, meta)
    assert torch.allclose(lp_vc, lp_fc, atol=1e-6, rtol=0)
    assert torch.allclose(lp_vc, lp_f, atol=1e-3, rtol=1e-5)


def test_grid_cache_keys_the_two_models_apart():
    """Same ``(n_ph, bg)`` tensors, factor model then a different per-observation variance."""
    n_ph, bg, *_ = _batch("cpu", M=512)
    meta = _meta(512)
    q = SplitLogGrid(n_points=192)
    factor = Poisson(bg_var_factor=C)
    per_obs = Poisson(bg_var=4 * C * bg.clamp(min=0.0))
    a = q.grid_params(n_ph, bg, q._sig_b2(factor, bg, meta))["log_L0"].clone()
    b = q.grid_params(n_ph, bg, q._sig_b2(per_obs, bg, meta))["log_L0"]
    assert float((a - b).abs().max()) > 1e-3


def test_paths_that_hardcode_the_plain_poisson_refuse():
    n_ph, bg, *_, per_obs = _pair(M=64)
    q = FixedLogGrid(S_min=math.exp(-20), S_max=math.exp(20), n_points=64)
    q._reject_unmodelled_bg(per_obs)                    # the mean is untouched
    with pytest.raises(NotImplementedError, match="background variance"):
        q._reject_unmodelled_bg(per_obs, variance=True)


@pytest.mark.skipif(not (CUDA and HAVE_TRITON), reason="needs CUDA + triton")
def test_split_triton_matches_the_factor_model():
    n_ph, bg, la, lb, mu, factor, per_obs = _pair("cuda", seed=2)
    meta = _meta(n_ph.numel(), "cuda")
    ref = SplitLogGrid(n_points=192).marginalize_signal(_scatter(la, lb), factor, mu, n_ph, bg, None, meta)
    got = SplitLogGrid(n_points=192).marginalize_signal(_scatter(la, lb), per_obs, mu, n_ph, bg, None, meta)
    assert torch.allclose(got, ref, atol=1e-5, rtol=0)


# ── through the CLI stack and a checkpoint ────────────────────────────

def test_build_stack_uses_the_prepared_variance_and_marks_the_checkpoint(tmp_path):
    from torchsx.cli import _pipeline as P
    from torchsx.cli.merge_nn import _build_parser
    from tests.test_amortized_training import _synthetic
    from tests.test_cli_pipeline import _stack

    prep, params, *_ = _synthetic(n_hkl=20, n_xtal=12, per=10)
    prep.bg_var = 0.3 * prep.bg.clamp(min=0.0) + 0.1
    recipe = {"d_min": 0.9, "spacegroup": "P 1", "gain": 1.0, "radii": [7.0, 9.0, 11.0],
              "background": "model"}
    bundle = P.PrepBundle(prep, recipe, [], C, background="model")
    base = ["ck.pt", "x.sfx", "-o", str(tmp_path / "o.mtz"), "--device", "cpu"]

    args = _build_parser().parse_args(base)
    *_, obj, settings = P.build_stack(bundle, args, verbose=False)
    assert obj.likelihood.has_bg_var and obj.likelihood.bg_var_factor == 0.0
    assert obj.likelihood.bg_mean_offset == 0.0
    assert torch.equal(obj.likelihood.bg_var_obs, prep.bg_var)
    assert settings["background"] == "model" and settings["bg_var_factor"] == 0.0

    # the checkpoint carries both markers ...
    _scale, _o, tr = _stack(prep, params, tmp_path)
    tr.objective.likelihood = Poisson(bg_var=prep.bg_var)
    tr.data_recipe = dict(recipe)
    ck = torch.load(tr.save_best(tmp_path / "g.pt"), weights_only=False)
    assert ck["likelihood"]["kind"] == "Poisson+bgvar"
    assert ck["quadrature"]["kind"] == "SplitLogGrid+bgvar"
    assert ck["likelihood"]["bg_var"] == "per_observation"
    assert P.per_observation_bg_var_from_ckpt(ck)

    # ... a merge rebuilds the same likelihood from the prepared data
    *_, obj2, _s = P.build_stack(bundle, args, ckpt=ck, verbose=False)
    assert obj2.likelihood.has_bg_var
    # ... and refuses data prepared without the variance
    annulus = P.PrepBundle(_synthetic(n_hkl=20, n_xtal=12, per=10)[0], recipe, [], C)
    with pytest.raises(SystemExit, match="per-observation background"):
        P.build_stack(annulus, args, ckpt=ck, verbose=False)
    # a checkpoint whose markers disagree is a write nobody knows how to read
    bad = dict(ck, quadrature={"kind": "SplitLogGrid", "n_points": 64})
    with pytest.raises(SystemExit, match="markers disagree"):
        P.per_observation_bg_var_from_ckpt(bad)
    # an annulus checkpoint is untouched by the new markers
    tr.objective.likelihood = Poisson(bg_var_factor=C)
    ck0 = torch.load(tr.save_best(tmp_path / "h.pt"), weights_only=False)
    assert ck0["likelihood"]["bg_var"] is None and not P.per_observation_bg_var_from_ckpt(ck0)
