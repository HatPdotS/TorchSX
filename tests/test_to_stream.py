"""Tests for folding indexed crystals back into an SFXStream.

These exercise the chunk-building + StreamBuilder path with synthetic indexing
results, so they need neither the peakfinder8 build nor the TORO clone.
"""

import pytest
import torch

from torchsx.datareduction import to_stream as ts
from torchsx.datareduction.indexing import IndexResult
from torchsx.datareduction.peakfinder import TorchPeakFinderParams
from torchsx.datareduction.spotfinding import PeakFinder8Params
from torchsx.datareduction.to_stream import (
    _index_frame,
    _resolve_params,
    crystal_to_chunk,
    image_only_chunk,
    peaks_level,
    reciprocal_vectors,
)
from torchsx._core import _sx_joins_with_peaks
from torchsx.datareduction._schema import FS, INTENSITY, PANEL, PROB, SS
from torchsx.geometry import Geometry, Panel
from torchsx.io._stream_builder import StreamBuilder
from torchsx.stream import SFXStream


def _geom():
    panel = Panel(name="p0", min_fs=0, max_fs=399, min_ss=0, max_ss=513,
                  corner_x=-200.0, corner_y=-256.0,
                  fs_vec=(1.0, 0.0, 0.0), ss_vec=(0.0, 1.0, 0.0), coffset=0.0)
    return Geometry(panels=[panel], photon_energy_eV=12960.0, clen=0.1, res=10000.0)


def _fake_result(n=12, basis=None):
    if basis is None:
        basis = torch.tensor([[79., 0, 0], [0, 79., 0], [0, 0, 38.]])
    spots = torch.zeros(n, 4)
    spots[:, PANEL] = 0
    spots[:, SS] = torch.linspace(20, 480, n)
    spots[:, FS] = torch.linspace(20, 380, n)
    spots[:, INTENSITY] = torch.linspace(50, 500, n)
    hkl = torch.randint(-5, 6, (n, 3))
    res = IndexResult(
        success=True, basis=basis, inlier_mask=torch.ones(n, dtype=torch.bool),
        hkl=hkl, n_indexed=n, n_spots=n, max_error=1e-3,
    )
    return spots, res


def test_reciprocal_vectors_inverse():
    B = torch.tensor([[79., 0, 0], [0, 79., 0], [0, 0, 38.]])
    Bs = reciprocal_vectors(B)
    assert torch.allclose(B @ Bs.T, torch.eye(3), atol=1e-5)


def test_crystal_to_chunk_columns():
    geom = _geom()
    spots, res = _fake_result(n=10)
    chunk = crystal_to_chunk(spots, res, geom, frame_key=("f.h5", 3))
    obs, crys, img = chunk.levels["obs"], chunk.levels["crystal"], chunk.levels["image"]

    assert obs["hkl"].shape == (10, 3) and obs["hkl"].dtype == torch.int16
    assert obs["I"].shape == (10,)
    assert bool((obs["sigma"] > 0).all())
    assert obs["crystal_id"].dtype == torch.int32 and int(obs["crystal_id"].max()) == 0

    # astar = first row of inv(basis).T
    assert torch.allclose(crys["astar"][0], reciprocal_vectors(res.basis)[0], atol=1e-5)
    assert crys["cell"].shape == (1, 6)
    assert float(crys["cell"][0, 0]) == torch.tensor(79.0)
    assert int(crys["n_refl"][0]) == 10
    assert int(crys["image_id"][0]) == 0

    assert float(img["camera_length"][0]) == pytest.approx(geom.clen)
    assert float(img["beam_energy"][0]) == pytest.approx(geom.photon_energy_eV)
    assert chunk.key == ("f.h5", 3)


def test_resolve_params_defaults():
    # pf8 backend → pf8 params; torch backends → torch params; explicit passthrough
    assert isinstance(_resolve_params("pf8", None), PeakFinder8Params)
    assert isinstance(_resolve_params("torch", None), TorchPeakFinderParams)
    assert isinstance(_resolve_params("torch-threshold", None), TorchPeakFinderParams)
    sentinel = PeakFinder8Params()
    assert _resolve_params("torch", sentinel) is sentinel


class _FakeReader:
    """Minimal reader: a flat synthetic frame so integration has pixels to read."""
    path = "fake.h5"
    _n_ss, _n_fs = 514, 400

    def panels(self, i):
        f = torch.full((1, self._n_ss, self._n_fs), 10.0)
        # add noise so radial_background isn't degenerate on a perfectly flat frame
        torch.manual_seed(i)
        f += torch.randn_like(f) * 1.0
        return f

    def mask_panels(self):
        return torch.ones((1, self._n_ss, self._n_fs), dtype=torch.bool)


def _patch_pipeline(monkeypatch, *, spots, res):
    """Stub the pf8/TORO-dependent steps inside to_stream; keep real spots_to_q
    (pure geometry, also used by crystal_to_chunk)."""
    monkeypatch.setattr(ts, "find_spots_on_frame", lambda *a, **k: spots)
    monkeypatch.setattr(ts, "index_spots", lambda *a, **k: res)


def test_index_frame_skips_too_few_spots(monkeypatch):
    geom = _geom()
    spots, res = _fake_result(n=3)
    _patch_pipeline(monkeypatch, spots=spots, res=res)
    out = _index_frame(_FakeReader(), geom, 0, PeakFinder8Params(), backend="pf8",
                       noise_model=None, null=None,
                       initial_cell=torch.eye(3), device="cpu", min_num_spots=8)
    assert out is None


def test_index_frame_unindexed_hit_kept(monkeypatch):
    """A hit that fails to index is kept as an image-only chunk with its peaks."""
    geom = _geom()
    spots, _ = _fake_result(n=12)
    fail = IndexResult(success=False, basis=None, inlier_mask=None,
                       hkl=None, n_indexed=0, n_spots=12, max_error=0.0)
    _patch_pipeline(monkeypatch, spots=spots, res=fail)
    out = _index_frame(_FakeReader(), geom, 4, PeakFinder8Params(), backend="pf8",
                       noise_model=None, null=None,
                       initial_cell=torch.eye(3), device="cpu", min_num_spots=8)
    assert out is not None
    chunk, n_obs, n_spots, _res = out
    assert n_obs == 0 and n_spots == 12            # hit, but no crystal
    assert set(chunk.levels) == {"image", "peak"}  # no obs/crystal
    assert chunk.levels["peak"]["peak_fs"].shape == (12,)
    assert int(chunk.levels["image"]["n_peaks"][0]) == 12
    assert chunk.key == ("fake.h5", 4)


def test_index_frame_drops_failed_index_when_index_only(monkeypatch):
    geom = _geom()
    spots, _ = _fake_result(n=12)
    fail = IndexResult(success=False, basis=None, inlier_mask=None,
                       hkl=None, n_indexed=0, n_spots=12, max_error=0.0)
    _patch_pipeline(monkeypatch, spots=spots, res=fail)
    out = _index_frame(_FakeReader(), geom, 0, PeakFinder8Params(), backend="pf8",
                       noise_model=None, null=None,
                       initial_cell=torch.eye(3), device="cpu", min_num_spots=8,
                       keep_unindexed_hits=False)
    assert out is None


def test_index_frame_attaches_peaks_to_indexed(monkeypatch):
    """An indexed chunk also carries the full raw peak search (all found spots)."""
    geom = _geom()
    spots, res = _fake_result(n=12)
    _patch_pipeline(monkeypatch, spots=spots, res=res)
    out = _index_frame(_FakeReader(), geom, 5, PeakFinder8Params(), backend="pf8",
                       noise_model=None, null=None,
                       initial_cell=torch.eye(3), device="cpu", min_num_spots=8)
    assert out is not None
    chunk, _n_obs, n_spots, _res = out
    assert "peak" in chunk.levels
    pk = chunk.levels["peak"]
    # one peak row per found spot (not just indexed inliers)
    assert pk["peak_fs"].shape == (n_spots,)
    assert pk["peak_recip"].shape == (n_spots,)
    assert pk["peak_panel"].dtype == torch.int16
    assert bool((pk["peak_image_id"] == 0).all())  # chunk-local


def test_index_frame_returns_chunk(monkeypatch):
    geom = _geom()
    spots, res = _fake_result(n=12)
    _patch_pipeline(monkeypatch, spots=spots, res=res)
    out = _index_frame(_FakeReader(), geom, 5, PeakFinder8Params(), backend="pf8",
                       noise_model=None, null=None,
                       initial_cell=torch.eye(3), device="cpu", min_num_spots=8)
    assert out is not None
    chunk, n_obs, n_spots, got = out
    assert n_obs == 12 and n_spots == 12 and got is res
    assert chunk.key == ("fake.h5", 5)
    # integration ran: obs carry a real propagated sigma + a background level,
    # not the sqrt(I) / 0 stubs.
    obs = chunk.levels["obs"]
    assert bool((obs["sigma"] > 0).all())
    assert "background" in obs and obs["background"].shape == (12,)
    assert bool((obs["background"] != 0).any())  # global-radial bg ≈ frame level


def test_index_frame_hit_gate_skips_low_sumprob(monkeypatch):
    """A near-blank frame (Σ peak prob < gate) skips the indexer entirely."""
    geom = _geom()
    n = 12
    spots = torch.zeros(n, 5)                 # probabilistic finder layout (PROB col)
    spots[:, SS] = torch.linspace(20, 480, n)
    spots[:, FS] = torch.linspace(20, 380, n)
    spots[:, PROB] = 0.05                      # Σ = 0.6 < 2.0 → gated
    monkeypatch.setattr(ts, "find_spots_on_frame", lambda *a, **k: spots)

    def _boom(*a, **k):
        raise AssertionError("index_spots must not run on a gated frame")
    monkeypatch.setattr(ts, "index_spots", _boom)

    out = _index_frame(_FakeReader(), geom, 0, TorchPeakFinderParams(), backend="torch",
                       noise_model=None, null=None, initial_cell=torch.eye(3),
                       device="cpu", min_num_spots=8, hit_min_sumprob=2.0)
    assert out is None


def test_index_frame_hit_gate_passes_high_sumprob(monkeypatch):
    """A frame with enough peak-probability mass proceeds to indexing."""
    geom = _geom()
    spots4, res = _fake_result(n=12)
    spots = torch.zeros(12, 5)
    spots[:, :4] = spots4
    spots[:, PROB] = 0.5                        # Σ = 6.0 ≥ 2.0 → passes
    _patch_pipeline(monkeypatch, spots=spots, res=res)
    out = _index_frame(_FakeReader(), geom, 0, TorchPeakFinderParams(), backend="torch",
                       noise_model=None, null=None, initial_cell=torch.eye(3),
                       device="cpu", min_num_spots=8, hit_min_sumprob=2.0)
    assert out is not None


def test_build_stream_populates_excise_metadata(monkeypatch):
    """build_stream writes image-level filename/event so excise can find frames."""
    geom = _geom()
    spots, res = _fake_result(n=10)
    _patch_pipeline(monkeypatch, spots=spots, res=res)
    stream, summary = ts.build_stream(
        _FakeReader(), geom, frames=[0, 1, 2], backend="pf8",
        calibrate=False, min_num_spots=8,
    )
    assert summary["crystals_indexed"] == 3
    img_meta = stream._core.metadata["image"]
    # Source paths are stored absolute so excise/finetuning can reload frames
    # regardless of the working directory at reload time.
    from torchsx._core import abspath_str
    abs_fake = abspath_str("fake.h5")
    assert img_meta["filename"] == [abs_fake, abs_fake, abs_fake]
    assert all(f.startswith("/") for f in img_meta["filename"])
    assert img_meta["event"] == ["//0", "//1", "//2"]
    # excise resolves obs -> frame via these; lengths must match the image rows
    assert len(img_meta["filename"]) == stream.n_images


def test_build_two_crystal_stream_joins_and_roundtrip(tmp_path):
    geom = _geom()
    builder = StreamBuilder.initialize(geometry=geom, spacegroup="P 43 21 2",
                                       async_writer=False)
    s0, r0 = _fake_result(n=8)
    s1, r1 = _fake_result(n=12)
    builder.add(crystal_to_chunk(s0, r0, geom, frame_key=("f.h5", 0)))
    builder.add(crystal_to_chunk(s1, r1, geom, frame_key=("f.h5", 1)))
    core = builder.finalize()

    assert type(core).__name__ == "SFXStreamCore"
    assert core.n_crystals == 2
    assert core.n_images == 2
    assert core.n_observations == 20
    # crystal_id offset across chunks: first 8 obs -> crystal 0, next 12 -> crystal 1
    assert int(core.crystal_id[:8].max()) == 0
    assert int(core.crystal_id[8:].min()) == 1
    # image_id offset on the crystal level
    assert int(core.image_id[0]) == 0 and int(core.image_id[1]) == 1

    stream = SFXStream(core)
    p = tmp_path / "indexed.sfx"
    stream.save(p)
    loaded = SFXStream.load(p, merge_friedel=False)
    assert loaded.n_observations == 20
    assert torch.equal(loaded.hkl, core.hkl)


def test_build_stream_attaches_and_roundtrips_peaks(monkeypatch, tmp_path):
    """build_stream emits a `peak` level; it survives the .sfx round-trip."""
    geom = _geom()
    spots, res = _fake_result(n=10)
    _patch_pipeline(monkeypatch, spots=spots, res=res)
    stream, summary = ts.build_stream(
        _FakeReader(), geom, frames=[0, 1, 2], backend="pf8",
        calibrate=False, min_num_spots=8,
    )
    assert summary["hits"] == 3 and summary["crystals_indexed"] == 3
    peak = stream._core.levels["peak"]
    assert peak.n == 30                                  # 3 frames × 10 found spots
    # peak_image_id points at the three images, 10 rows each
    assert sorted(peak["peak_image_id"].tolist()) == sorted([0] * 10 + [1] * 10 + [2] * 10)
    # join is recorded so it round-trips and stays consistent
    assert any(j.child_level == "peak" for j in stream._core.joins)

    p = tmp_path / "withpeaks.sfx"
    stream.save(p)
    loaded = SFXStream.load(p, merge_friedel=False)
    lp = loaded._core.levels["peak"]
    assert lp.n == 30
    assert lp["peak_panel"].dtype == torch.int16
    assert torch.equal(lp["peak_image_id"], peak["peak_image_id"])


def test_write_stream_emits_peaks_and_unindexed_hit(tmp_path):
    """write_stream writes a non-empty peak list, including a hit that didn't index."""
    geom = _geom()
    builder = StreamBuilder.initialize(geometry=geom, spacegroup="P 43 21 2",
                                       async_writer=False, joins=_sx_joins_with_peaks())
    # image 0: an indexed crystal (8 obs) + its 12 found peaks
    s0, r0 = _fake_result(n=8)
    peaks0 = torch.zeros(12, 4)
    peaks0[:, SS] = torch.linspace(10, 500, 12)
    peaks0[:, FS] = torch.linspace(10, 390, 12)
    peaks0[:, INTENSITY] = torch.linspace(100, 900, 12)
    ch0 = crystal_to_chunk(s0, r0, geom, frame_key=("f.h5", 0))
    ch0.levels["peak"] = peaks_level(peaks0, geom)
    builder.add(ch0)
    # image 1: a hit that failed to index — peaks only, no crystal
    peaks1 = torch.zeros(5, 4)
    peaks1[:, SS] = torch.linspace(20, 480, 5)
    peaks1[:, FS] = torch.linspace(20, 380, 5)
    peaks1[:, INTENSITY] = torch.linspace(50, 500, 5)
    builder.add(image_only_chunk(peaks1, geom, frame_key=("f.h5", 1)))
    core = builder.finalize()

    assert core.n_images == 2 and core.n_crystals == 1
    out = tmp_path / "out.stream"
    SFXStream(core).write_stream(str(out))
    text = out.read_text()

    chunks = text.split("----- Begin chunk -----")[1:]
    assert len(chunks) == 2
    # first chunk: indexed + 12 peaks + a reflection list
    peak_block0 = chunks[0].split("Peaks from peak search")[1].split("End of peak list")[0]
    peak_lines0 = [l for l in peak_block0.splitlines() if l.strip() and "fs/px" not in l]
    assert len(peak_lines0) == 12
    assert "Reflections measured after indexing" in chunks[0]
    assert "indexed_by = xgandalf" in chunks[0]
    # second chunk: a hit with 5 peaks and NO crystal
    assert "indexed_by = none" in chunks[1]
    assert "--- Begin crystal" not in chunks[1]
    peak_block1 = chunks[1].split("Peaks from peak search")[1].split("End of peak list")[0]
    peak_lines1 = [l for l in peak_block1.splitlines() if l.strip() and "fs/px" not in l]
    assert len(peak_lines1) == 5
