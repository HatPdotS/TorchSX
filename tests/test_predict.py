"""Tests for prediction-based integration (Phase C).

The q→pixel forward map and the chunk builder are tested on a constructed
geometry (always run). The full predict_reflections chain is exercised against
the real JF06T32V07 lyso geometry when present (skipped otherwise — the .geom
isn't in the repo).
"""

from __future__ import annotations

import os

import pytest
import torch

from torchsx.geometry import Geometry, Panel
from torchsx.datareduction.indexing import HEWL_CELL
from torchsx.datareduction.integrate import IntegrationResult
from torchsx.datareduction.predict import PredictedReflections, predict_reflections
from torchsx.datareduction.to_stream import predicted_to_chunk

GEOM_PATH = "/data/project/cls/standfuss/torchsx/raw_data/16M_v1_distance_93.8.geom"
_no_geom = not os.path.exists(GEOM_PATH)


def _two_panel_geom():
    """Two axis-aligned panels stacked along ss, a close detector."""
    panels = [
        Panel(name="p0", min_fs=0, max_fs=255, min_ss=0, max_ss=255,
              corner_x=-256.0, corner_y=-260.0,
              fs_vec=(1.0, 0.0, 0.0), ss_vec=(0.0, 1.0, 0.0), coffset=0.0),
        Panel(name="p1", min_fs=0, max_fs=255, min_ss=256, max_ss=511,
              corner_x=-256.0, corner_y=10.0,
              fs_vec=(1.0, 0.0, 0.0), ss_vec=(0.0, 1.0, 0.0), coffset=0.0),
    ]
    return Geometry(panels=panels, photon_energy_eV=12960.0, clen=0.1, res=5000.0)


# ── q→pixel forward map (C1) ──────────────────────────────────────────────


def test_recip_to_pixel_roundtrip():
    g = _two_panel_geom()
    pid_map = g._panel_id_map
    ss_all, fs_all = torch.where(pid_map >= 0)
    torch.manual_seed(0)
    sel = torch.randperm(ss_all.numel())[:500]
    ss, fs = ss_all[sel].float(), fs_all[sel].float()

    q = g.to_reciprocal(ss, fs)
    fs2, ss2, pid2 = g.recip_to_pixel(q)

    assert bool((pid2 >= 0).all())
    assert float((ss2 - ss).abs().max()) < 0.5
    assert float((fs2 - fs).abs().max()) < 0.5
    pid_true = pid_map[ss.long(), fs.long()].long()
    assert bool((pid2 == pid_true).all())


def test_recip_to_pixel_offdetector():
    g = _two_panel_geom()
    q = torch.tensor([[0.0, 0.0, -10.0],     # backward hemisphere
                      [500.0, 500.0, 0.0]])  # lands far off the panels
    _fs, _ss, pid = g.recip_to_pixel(q)
    assert bool((pid < 0).all())


# ── chunk builder (C3) ────────────────────────────────────────────────────


def test_predicted_to_chunk_columns():
    g = _two_panel_geom()
    n = 5
    pred = PredictedReflections(
        hkl=torch.randint(-3, 4, (n, 3)).to(torch.int16),
        panel=torch.zeros(n, dtype=torch.int64),
        ss=torch.linspace(10, 200, n), fs=torch.linspace(10, 200, n),
        excitation_error=torch.zeros(n), d_spacing=torch.linspace(2.0, 10.0, n),
    )
    integ = IntegrationResult(
        I=torch.full((n,), 100.0), sigma=torch.full((n,), 10.0),
        background=torch.full((n,), 5.0), peak=torch.full((n,), 120.0),
        npix=torch.full((n,), 29),
    )
    cell6 = torch.tensor([79.0, 79.0, 38.0, 90.0, 90.0, 90.0])
    chunk = predicted_to_chunk(pred, integ, HEWL_CELL, cell6, g, frame_key=("f.h5", 2))

    obs = chunk.levels["obs"]
    assert obs["I"].shape == (n,) and obs["hkl"].shape == (n, 3)
    assert torch.allclose(obs["sigma"], torch.full((n,), 10.0))      # from integration
    assert torch.allclose(obs["background"], torch.full((n,), 5.0))
    assert int(chunk.levels["crystal"]["n_refl"][0]) == n
    assert chunk.key == ("f.h5", 2)


# ── full prediction chain on the real geometry (C2) ───────────────────────


@pytest.mark.skipif(_no_geom, reason="real JF06T32V07 geom not present")
def test_predict_offset_cutoff_and_monotonicity():
    g = Geometry.from_geom_file(GEOM_PATH)
    counts = []
    for omax in (0.002, 0.005, 0.01):
        pred = predict_reflections(HEWL_CELL, g, d_min=2.0, offset_max=omax)
        # every kept reflection respects the cutoff and lands on a panel
        assert bool((pred.excitation_error.abs() <= omax + 1e-6).all())
        assert bool((pred.panel >= 0).all())
        assert bool((pred.d_spacing >= 2.0 - 1e-3).all())
        counts.append(len(pred))
    # tighter cutoff ⇒ strictly fewer predicted reflections
    assert counts[0] < counts[1] < counts[2]


@pytest.mark.skipif(_no_geom, reason="real JF06T32V07 geom not present")
def test_predict_spot_table_feeds_integrator():
    g = Geometry.from_geom_file(GEOM_PATH)
    pred = predict_reflections(HEWL_CELL, g, d_min=3.0, offset_max=0.004)
    st = pred.spot_table()
    assert st.shape == (len(pred), 4)
    # panel/ss/fs columns match the prediction; intensity column is a 0 placeholder
    assert torch.equal(st[:, 0].to(torch.int64), pred.panel)
    assert torch.allclose(st[:, 1], pred.ss) and torch.allclose(st[:, 2], pred.fs)
    assert float(st[:, 3].abs().max()) == 0.0
