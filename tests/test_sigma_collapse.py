"""A non-finite node in a reflection's only batch must not collapse its posterior width."""
import math

import torch

from torchsx.cli._pipeline import collapsed_sigma, report_merge
from torchsx.nn.posterior_target import PosteriorMergeTarget
from torchsx.nn.train_amortized import drop_nonfinite_rows


def _readout(acc, target):
    _, _, _, w, v, _ = target.integrate(acc)
    root = v.exp().sqrt()
    F = (w * root).sum(1)
    return F, (w * (root - F[:, None]) ** 2).sum(1).clamp(min=0).sqrt(), w


def _problem():
    target = PosteriorMergeTarget(torch.ones(1), k_coarse=63, m_fine=257, n_sigma=24.0, prior="wilson")
    target.wilson_sigma.fill_(1.0)
    target._loc = (torch.zeros(1), torch.full((1,), 0.25))
    v = target._loc[1][:, None] * target.u_coarse[None, :]
    batch = -30.0 * (v.exp() - 1.0 - v) - 200.0
    return target, batch


def test_one_nonfinite_node_collapses_the_width_when_only_that_node_is_zeroed():
    target, batch = _problem()
    F_ref, sd_ref, _ = _readout(batch, target)
    broken = batch.clone()
    broken[0, 20] = float("nan")
    F, sd, w = _readout(torch.nan_to_num(broken, nan=0.0), target)
    assert float(sd) < 1e-3 * float(sd_ref) and float(w.max()) > 0.999
    assert abs(float(F) - float(F_ref)) > 5 * float(sd_ref)


def test_dropping_the_row_keeps_the_width():
    target, batch = _problem()
    broken = batch.clone()
    broken[0, 20] = float("nan")
    kept, n_bad = drop_nonfinite_rows(broken)
    assert n_bad == 1 and not kept.any()
    F_ref, sd_ref, _ = _readout(batch, target)
    F, sd, w = _readout(batch + kept, target)
    assert float(F / sd) < 50
    torch.testing.assert_close(F, F_ref)
    torch.testing.assert_close(sd, sd_ref)
    assert float(w.max()) < 0.2
    clean, n_bad = drop_nonfinite_rows(batch)
    assert n_bad == 0 and clean is batch


def test_report_counts_and_lists_collapsed_widths(capsys):
    n = 1000
    F_sd = torch.full((n,), 0.04)
    F_sd[7], F_sd[11] = 1e-6, 2e-5
    res = {"estimator": "posterior", "seen": torch.ones(n, dtype=torch.bool), "F": torch.full((n,), 0.2),
           "F_sd": F_sd, "edge_mass": torch.zeros(n), "edge_low": torch.zeros(n),
           "edge_high": torch.zeros(n), "w_max": torch.full((n,), 0.05), "n_crystals": 10}
    res["w_max"][7] = 1.0
    hkl = torch.arange(3 * n).reshape(n, 3).numpy()
    low = collapsed_sigma(res, hkl)
    assert low["n"] == 2 and math.isclose(low["floor"], 4e-5, rel_tol=1e-5)
    assert [r["index"] for r in low["worst"]] == [7, 11] and low["worst"][0]["hkl"] == [21, 22, 23]
    assert low["worst"][0]["w_max"] == 1.0
    report_merge(res, hkl=hkl)
    out = capsys.readouterr().out
    assert "sigF floor  2 reflections" in out and "COLLAPSED" in out and "hkl [21, 22, 23]" in out
    res["F_sd"] = torch.full((n,), 0.04)
    report_merge(res, hkl=hkl)
    out = capsys.readouterr().out
    assert "sigF floor  0 reflections" in out and "COLLAPSED" not in out


def test_a_likelihood_narrower_than_the_node_spacing_is_resolved_by_refining_the_grid():
    from torchsx.nn.train_amortized import refine_grid

    target = PosteriorMergeTarget(torch.ones(3), k_coarse=63, m_fine=257, n_sigma=24.0, prior="wilson")
    target.wilson_sigma.fill_(1e6)
    peak = torch.tensor([0.07, -0.4, 9.0])
    count = torch.tensor([1e6, 30.0, 1e4])

    def sweep(v):
        x = v - peak[:, None]
        return -count[:, None] * (x.exp() - 1.0 - x) - 200.0

    centre, sd = torch.zeros(3), torch.full((3,), 0.25)
    target._loc = (centre, sd)
    _, sd_before, w_before = _readout(sweep(sd[:, None] * target.u_coarse[None, :]), target)
    assert float(w_before[0].max()) > 0.999 and float(sd_before[0]) < 1e-6

    centre, sd, acc, n = refine_grid(target, sweep, centre, sd)
    target._loc = (centre, sd)
    F, F_sd, w = _readout(acc, target)
    assert n == 2 and float(sd[1]) == 0.25
    assert float(w.max()) < 0.5
    expected = 0.5 * peak.exp().sqrt() / count.sqrt()
    torch.testing.assert_close(F_sd[[0, 2]], expected[[0, 2]], rtol=0.25, atol=0.0)
    torch.testing.assert_close(F[[0, 2]].log() * 2, peak[[0, 2]], rtol=0.0, atol=0.01)
