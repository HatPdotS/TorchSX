"""Ablation switches: own fold visible, scatter components, the real-space observation model."""
import json
import math

import numpy as np
import pytest
import torch

from torchsx.cli import _pipeline as P, train, merge_nn
from torchsx.merging.likelihood import Poisson
from torchsx.nn.objective import RealSpaceNLL
from torchsx.nn.pooling import fold_ids
from tests.test_amortized_training import _synthetic
from tests.test_fold_transformer import _encoder


@pytest.mark.parametrize("dof", [4.0, 32.0, None])
def test_real_space_log_prob_is_the_ev11_student_t(dof):
    lik = Poisson(bg_var_factor=0.2, bg_mean_offset=0.2)
    obj = RealSpaceNLL(lik, dof=dof, sd_fac=1.3, sd_b=0.05, sd_add=0.4)
    g = torch.Generator().manual_seed(2)
    n = torch.poisson(torch.full((50,), 30.0), generator=g)
    bg = torch.rand(50, generator=g) * 20
    log_mean = torch.randn(50, generator=g) + 2.0
    lp = obj.log_prob(None, None, log_mean, n, bg, None)
    mu = log_mean.exp()
    var = 1.3 ** 2 * ((n + 0.2 * bg).clamp(min=1.0) + 0.05 * mu + (0.4 * mu) ** 2)
    y = n - (bg + 0.2)
    ref = (torch.distributions.Normal(mu, var.sqrt()) if dof is None
           else torch.distributions.StudentT(dof, mu, var.sqrt())).log_prob(y)
    torch.testing.assert_close(lp, ref, rtol=1e-5, atol=1e-5)
    assert obj.terms() == pytest.approx({"sd_fac": 1.3, "sd_b": 0.05, "sd_add": 0.4})


def test_see_own_fold_lets_an_observation_read_its_own_measurement():
    enc = _encoder(see_own_fold=True)
    sizes = torch.tensor([1, 4, 9, 13, 6])
    gid = torch.repeat_interleave(torch.arange(sizes.numel()), sizes)
    xs = torch.randn(gid.numel(), 4)
    xc = torch.randn(gid.numel(), 7, requires_grad=True)
    z = enc(xs, xc, gid, sizes.numel())
    j = int((gid == 3).nonzero()[4])
    z[j].square().sum().backward()
    fold = fold_ids(gid, sizes.numel(), 3)
    own = (gid == gid[j]) & (fold == fold[j])
    assert xc.grad[j, 4:].abs().max() > 0
    assert xc.grad[own][:, 4:].abs().max() > 0
    assert torch.equal(xc.grad[gid != gid[j]], torch.zeros_like(xc.grad[gid != gid[j]]))


def test_ablation_options_parse_but_stay_out_of_help():
    parser = train._build_parser()
    text = parser.format_help()
    for option in ("--see-own-fold", "--scatter-components", "--observation-model", "--t-dof"):
        assert option not in text
    a = parser.parse_args(["x.stream", "-o", "run"])
    assert (a.see_own_fold, a.scatter_components, a.observation_model, a.t_dof) == (False, None, "counts", 32.0)
    a = parser.parse_args(["x.stream", "-o", "run", "--see-own-fold", "--scatter-components", "1",
                           "--observation-model", "student-t", "--t-dof", "8"])
    assert (a.see_own_fold, a.scatter_components, a.observation_model, a.t_dof) == (True, 1, "student-t", 8.0)


@pytest.mark.parametrize("argv,check", [
    (["--see-own-fold"], lambda ck: ck["encoder"]["see_own_fold"] is True),
    (["--scatter-components", "1"], lambda ck: ck["scatter"]["n_components"] == 1),
    (["--ctx-layers", "0", "--query-layers", "0", "--no-refine-geometry"],
     lambda ck: (ck["encoder"]["ctx_layers"], ck["encoder"]["query_layers"]) == (0, 0)),
    (["--observation-model", "student-t"],
     lambda ck: ck["observation"]["model"] == "student-t" and ck["observation"]["dof"] == 32.0
     and ck["scatter"]["miss_logit"] is None and ck["quadrature"]["kind"] is None),
    (["--observation-model", "normal"],
     lambda ck: ck["observation"]["model"] == "normal" and ck["observation"]["dof"] is None),
])
def test_ablations_train_and_merge_again_from_their_checkpoint(tmp_path, monkeypatch, argv, check):
    prep, _, *_ = _synthetic(n_hkl=20, n_xtal=40, per=10, n_cond=2)
    bundle = P.PrepBundle(prep, {"spacegroup": "P 1", "d_min": 0.9, "gain": 1.0, "radii": [7, 9, 11],
                                 "merge_friedel": True}, ["dark", "light"], 0.1)
    monkeypatch.setattr(P, "load_prep", lambda *a, **k: bundle)
    for key, value in dict(d_model=16, inner_points=32, k_coarse=15, m_fine=65).items():
        monkeypatch.setitem(P.DEFAULTS, key, value)
    run = tmp_path / "run"
    train.main(["x.stream", "-o", str(run), "--device", "cpu", "--steps", "4", "--seed", "17", *argv])
    ck = torch.load(run / "best.pt", weights_only=False)
    assert check(ck)
    if ck["observation"]["model"] != "counts":
        # the error terms were fitted and are carried by the checkpoint
        assert set(ck["observation"]["state"]) == {"log_sd_fac", "log_sd_b", "log_sd_add"}
        assert all(math.isfinite(v) for v in ck["observation"]["state"].values())
    merge_nn.main([str(run / "best.pt"), "x.stream", "-o", str(tmp_path / "again.mtz"), "--device", "cpu"])
    import gemmi
    trained = sorted(run.glob("merged_*.mtz"))
    assert trained
    for path in trained:
        a = np.asarray(gemmi.read_mtz_file(str(path)).column_with_label("F-obs"))
        b = np.asarray(gemmi.read_mtz_file(str(tmp_path / path.name.replace("merged", "again")))
                       .column_with_label("F-obs"))
        np.testing.assert_array_equal(np.isfinite(a), np.isfinite(b))
        assert np.isfinite(a).any()
        np.testing.assert_allclose(a[np.isfinite(a)], b[np.isfinite(b)], rtol=1e-4, atol=1e-6)
    summary = json.loads((run / "training_summary.json").read_text())
    assert summary["updates_per_state"] == [2, 2]
