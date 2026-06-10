"""Viewer backend tests: synthetic SFXStream + raw .h5 → server endpoints.

The viewer's cache is a torchsx ``SFXStream`` (results) and frames come from the
raw ``.h5``. ``_synth.generate`` writes both; these tests cover the stream
round-trip, the per-frame slicing in ``StreamResults``, and the cached server
endpoints (geometry, manifest, reflections, counts incl. gzip transport). The
live matched-filter ``/score`` path is exercised manually on a GPU node (CPU
compute is too slow for CI).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

GEOM = Path("raw_data/16M_v1_distance_93.8.geom")
pytestmark = pytest.mark.skipif(not GEOM.exists(), reason="needs the detector .geom")


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    from torchsx.viewer import _synth
    out = tmp_path_factory.mktemp("synth") / "run"
    sfx = _synth.generate(out, n_frames=1, geom_path=str(GEOM), seed=1)
    return sfx


def test_stream_results(synth):
    from torchsx.viewer._source import StreamResults
    r = StreamResults(synth)
    assert r.n_images == 1
    assert r.geometry is not None and r.geometry.n_panels == 32
    path, fidx = r.frame_source(0)
    assert path.endswith(".h5") and fidx == 0
    refl = r.reflections(0)
    assert len(refl) > 0
    pk = refl[0]
    assert {"panel", "ss", "fs", "intensity", "hkl"} <= pk.keys()
    assert 0 <= pk["panel"] < 32 and len(pk["hkl"]) == 3
    assert r.crystals(0)[0]["cell_lengths"][0] == pytest.approx(79.1, abs=0.1)


def test_server_cached_endpoints(synth):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from torchsx.viewer.server import create_app

    client = fastapi_testclient.TestClient(create_app(synth, device="cpu"))

    man = client.get("/api/manifest").json()
    assert man["detector"]["n_panels"] == 32
    assert len(man["frames"]) == 1

    assert len(client.get("/api/geometry").json()["panels"]) == 32

    refl = client.get("/api/frames/0/reflections").json()
    assert len(refl) == man["frames"][0]["n_refl"]

    # counts come from the raw .h5, gzip-encoded; the TestClient (httpx)
    # transparently decompresses, so r.content is the raw uint16 buffer
    meta = client.get("/api/frames/0/counts/meta").json()
    r = client.get("/api/frames/0/counts")
    assert r.headers["content-encoding"] == "gzip"
    arr = np.frombuffer(r.content, dtype=np.uint16).reshape(meta["shape"])
    assert arr.shape == (32, 514, 1030)


def test_missing_frame_counts_404(synth):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from torchsx.viewer.server import create_app
    client = fastapi_testclient.TestClient(create_app(synth, device="cpu"))
    assert client.get("/api/frames/99/counts").status_code == 404


def test_mask_endpoint(synth):
    """/api/mask serves the stream's embedded good-pixel mask (1=good, 0=masked)
    as gzip uint8 matching the counts layout; _synth plants a masked block."""
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from torchsx.viewer.server import create_app

    client = fastapi_testclient.TestClient(create_app(synth, device="cpu"))
    det = client.get("/api/manifest").json()["detector"]
    r = client.get("/api/mask")
    assert r.status_code == 200 and r.headers["content-encoding"] == "gzip"
    P, nss, nfs = det["n_panels"], det["n_ss"], det["n_fs"]
    arr = np.frombuffer(r.content, dtype=np.uint8).reshape(P, nss, nfs)
    assert set(np.unique(arr)) <= {0, 1}
    assert arr[0, 10:30, 10:30].sum() == 0          # the planted masked block
    assert arr[0, 0, 0] == 1                          # good elsewhere


def test_reprocess_index_and_predict(synth):
    """Live reprocessing stages 2–3 (index, predict). The bayes ``find`` stage is
    GPU-practical only (calibration + matched filter), exercised manually on a node
    like ``/score``; here we cover the cell→basis→predict path and the Ewald lever.

    ``_synth`` plants Gaussian spots but no real Bragg lattice, so indexing won't
    succeed — assert the response *shape* (keys), not a physical solution. Prediction
    is checked from a cell-derived basis so it doesn't depend on indexing."""
    from torchsx.datareduction import JungfrauReader
    from torchsx.viewer._reprocess import Reprocessor
    from torchsx.viewer._source import StreamResults

    res = StreamResults(synth)

    def reader_for(i):
        p, f = res.frame_source(i)
        return JungfrauReader(p), f
    R = Reprocessor(res.geometry, reader_for, device="cpu")

    # index returns the documented keys even when the synth frame doesn't index
    spots = [{"panel": r["panel"], "ss": r["ss"], "fs": r["fs"],
              "intensity": r["intensity"], "prob": 1.0} for r in res.reflections(0)]
    ix = R.index(spots, cell=[79.1, 79.1, 38.0, 90, 90, 90], min_num_spots=8)
    assert {"success", "n_indexed", "basis", "cell"} <= ix.keys()

    # predict from a valid cell-derived basis; the Ewald lever monotonically
    # admits more reflections as offset_max grows
    basis = R._basis_from_cell([79.1, 79.1, 38.0, 90, 90, 90]).numpy().tolist()
    counts = [R.predict(basis, d_min=3.0, offset_max=o)["n_predicted"]
              for o in (0.002, 0.006, 0.02)]
    assert counts[0] <= counts[1] <= counts[2] and counts[2] > 0
    sample = R.predict(basis, d_min=3.0, offset_max=0.006)["reflections"][0]
    assert {"panel", "ss", "fs", "hkl", "excitation_error", "d_spacing"} <= sample.keys()


def test_reprocess_predict_endpoint(synth):
    """POST /predict over the HTTP layer (uses a cell-derived basis)."""
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from torchsx.viewer._reprocess import Reprocessor
    from torchsx.viewer.server import create_app

    basis = Reprocessor._basis_from_cell([79.1, 79.1, 38.0, 90, 90, 90]).numpy().tolist()
    client = fastapi_testclient.TestClient(create_app(synth, device="cpu"))
    r = client.post("/api/frames/0/predict",
                    json={"basis": basis, "d_min": 3.0, "offset_max": 0.006})
    assert r.status_code == 200
    body = r.json()
    assert body["n_predicted"] == len(body["reflections"]) > 0
