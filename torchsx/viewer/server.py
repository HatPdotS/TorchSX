"""Viewer server: cached results from an SFXStream, frames from the raw .h5,
matched-filter score computed on demand.

One server, no modes. It loads a torchsx ``SFXStream`` (the result cache), reads
detector frames from the raw ``.h5`` files the stream references, serves the
geometry, and computes the matched-filter score map for a frame on request (GPU
if available, else CPU). Intended to run on an interactive GPU node; the browser
connects over an SSH tunnel (the command is printed at startup).

Run::

    torchsx-viewer --stream results.sfx [--geom det.geom] [--device auto]
"""

from __future__ import annotations

import argparse
import functools
import getpass
import os
import socket
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import _cache as C
from ._source import StreamResults

FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"


# ── live-reprocess request bodies (see _reprocess.Reprocessor) ───────────────
class FindReq(BaseModel):
    top_n: int = 200
    sigma_radial: float | None = None
    sigma_azimuthal: float | None = None
    peak_min_distance: int = 2


class IndexReq(BaseModel):
    spots: list[dict] = []           # live Find peaks; empty → use the stream's
    cell: list[float] | None = None  # seed [a,b,c,α,β,γ]; None → HEWL
    min_num_spots: int = 8


class PredictReq(BaseModel):
    basis: list[list[float]]         # 3×3 real-space cell (rows a,b,c)
    d_min: float
    offset_max: float                # Ewald-offset cutoff |r| (Å⁻¹)


def create_app(stream_path: str | Path, geom_path: str | None = None,
               device: str = "auto", data_root: str | None = None) -> FastAPI:
    from torchsx.datareduction import JungfrauReader
    from torchsx.geometry import Geometry

    results = StreamResults(stream_path, data_root=data_root)

    # Geometry: prefer the one embedded in the stream; else --geom.
    geom = results.geometry
    if geom is None:
        if not geom_path:
            raise SystemExit("stream has no embedded geometry; pass --geom")
        geom = Geometry.from_geom_file(geom_path)
    geom_dict = C.geom_to_dict(geom)
    P, nss, nfs = geom.n_panels, 514, geom.detector_shape[1]

    @functools.lru_cache(maxsize=8)
    def _reader(path: str):
        return JungfrauReader(path)

    @functools.lru_cache(maxsize=16)
    def _frame_counts(i: int):
        """(uint16 (P,nss,nfs), p90, vmax) for image i, read from its raw .h5."""
        path, fidx = results.frame_source(i)
        panels = _reader(path).panels(fidx)  # (P,514,1030) int32
        counts = panels.clamp(0, 65535).cpu().numpy().astype(np.uint16)
        pos = counts[counts > 0]
        p90 = float(np.percentile(pos, 90)) if pos.size else 1.0
        return counts, p90, float(counts.max())

    # score engine is built lazily on first /score (imports torch finder bits)
    _score_state = {"engine": None}

    def _score_engine():
        if _score_state["engine"] is None:
            from ._score import ScoreEngine
            _score_state["engine"] = ScoreEngine(geom, device=device)
        return _score_state["engine"]

    # live reprocessor (find/index/predict) — lazy; calibrates on first find
    _reproc_state = {"r": None}

    def _reprocessor():
        if _reproc_state["r"] is None:
            from ._reprocess import Reprocessor

            def reader_for(i: int):
                path, fidx = results.frame_source(i)
                return _reader(path), fidx
            _reproc_state["r"] = Reprocessor(geom, reader_for, device=device)
        return _reproc_state["r"]

    app = FastAPI(title="torchsx viewer", docs_url=None, redoc_url=None)
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["GET", "POST"], allow_headers=["*"])

    def _gz(arr) -> Response:
        raw = C.gzip_bytes(arr)
        return Response(content=raw, media_type="application/octet-stream",
                        headers={"Content-Encoding": "gzip", "Content-Length": str(len(raw))})

    @app.get("/api/ping")
    def ping():
        """Tiny liveness probe — the client times the round-trip for its latency /
        connection indicator. No-store so intermediaries never serve it from cache."""
        return Response(content="{}", media_type="application/json",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/manifest")
    def manifest():
        return {
            "run_name": Path(results.path).stem,
            "detector": {"n_panels": P, "n_ss": nss, "n_fs": nfs},
            "frames": results.manifest_rows(),
            "score_default": {"sigma_radial": 2.0, "sigma_azimuthal": 0.7},
        }

    @app.get("/api/geometry")
    def geometry():
        return geom_dict

    # Detector good-pixel mask (per-run constant): (P,nss,nfs) uint8, 1=good,
    # 0=masked, gzip-encoded. 404 when the stream carries no mask (client then
    # renders everything unmasked).
    _mask_cache = {"arr": ..., "done": False}

    @app.get("/api/mask")
    def mask():
        if not _mask_cache["done"]:
            _mask_cache["arr"] = results.detector_mask()
            _mask_cache["done"] = True
        arr = _mask_cache["arr"]
        if arr is None:
            raise HTTPException(404, "stream has no detector mask")
        return _gz(arr)

    @app.get("/api/frames/{i}/counts")
    def counts(i: int):
        try:
            arr, _, _ = _frame_counts(i)
        except Exception as e:
            raise HTTPException(404, f"frame {i}: {e}")
        return _gz(arr)

    @app.get("/api/frames/{i}/counts/meta")
    def counts_meta(i: int):
        try:
            arr, p90, vmax = _frame_counts(i)
        except Exception as e:
            raise HTTPException(404, f"frame {i}: {e}")
        return {"shape": list(arr.shape), "p90": p90, "vmax": vmax}

    @app.get("/api/frames/{i}/reflections")
    def reflections(i: int):
        return results.reflections(i)

    @app.get("/api/frames/{i}/peaks")
    def peaks(i: int):
        return results.found_peaks(i)

    @app.get("/api/frames/{i}/crystal")
    def crystal(i: int):
        return results.crystals(i)

    @app.get("/api/frames/{i}/score")
    def score(i: int, sigma_radial: float | None = Query(None),
              sigma_azimuthal: float | None = Query(None)):
        import torch
        eng = _score_engine()
        path, fidx = results.frame_source(i)
        reader = _reader(path)
        panels = reader.panels_gpu(fidx, eng.device).to(torch.float32)
        good = reader.mask_panels().to(torch.bool).to(eng.device)
        arr, smin, smax = eng.score(panels, good, sigma_radial, sigma_azimuthal)
        resp = _gz(arr)
        resp.headers["X-Score-Min"] = repr(smin)
        resp.headers["X-Score-Max"] = repr(smax)
        return resp

    @app.get("/api/frames/{i}/score/meta")
    def score_meta(i: int):
        return {"shape": [P, nss, nfs], "device": str(_score_engine().device)}

    # ── live reprocessing (preview only; the stream on disk is never touched) ──
    @app.post("/api/frames/{i}/find-spots")
    def find_spots(i: int, req: FindReq):
        """Re-run the bayes finder on frame ``i`` with ``top_n`` (= hits fed to the
        indexer) + kernel σ. First call calibrates (a few seconds on GPU)."""
        try:
            return _reprocessor().find(
                i, top_n=req.top_n, sigma_radial=req.sigma_radial,
                sigma_azimuthal=req.sigma_azimuthal,
                peak_min_distance=req.peak_min_distance)
        except Exception as e:
            raise HTTPException(500, f"find-spots frame {i}: {e}")

    @app.post("/api/frames/{i}/index")
    def index(i: int, req: IndexReq):
        """Index a peak list (live Find result, else the stream's found peaks for
        frame ``i``) against a typed seed cell. Returns the refined basis + cell."""
        spots = req.spots if req.spots else results.found_peaks(i)
        try:
            return _reprocessor().index(spots, cell=req.cell,
                                        min_num_spots=req.min_num_spots)
        except Exception as e:
            raise HTTPException(500, f"index frame {i}: {e}")

    @app.post("/api/frames/{i}/predict")
    def predict(i: int, req: PredictReq):
        """Predict reflections from a (possibly hand-edited) basis, keeping those
        within the Ewald-offset cutoff ``offset_max`` and ``d_min``."""
        try:
            return _reprocessor().predict(req.basis, d_min=req.d_min,
                                          offset_max=req.offset_max)
        except Exception as e:
            raise HTTPException(500, f"predict frame {i}: {e}")

    if FRONTEND_DIST.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="ui")
    else:
        @app.get("/")
        def _no_ui():
            return JSONResponse({"msg": "frontend not built; run `npm run build` in "
                                 "torchsx/viewer/frontend, or use the Vite dev server",
                                 "api": "/api/manifest"})

    return app


# ── tunnel hint ──────────────────────────────────────────────────────────────

def _print_tunnel(port: int, login_host: str | None) -> None:
    user = getpass.getuser()
    node = os.environ.get("SLURMD_NODENAME") or socket.gethostname()
    on_compute = bool(os.environ.get("SLURM_JOB_ID"))
    login = login_host or os.environ.get("SLURM_SUBMIT_HOST")
    print("\n" + "=" * 70)
    print(f"  torchsx viewer — serving on {node}:{port}")
    print("=" * 70)
    print("  From your laptop:\n")
    if on_compute:
        login = login or "<login-node>"
        print(f"      ssh -J {user}@{login} -L {port}:localhost:{port} {user}@{node}")
        if login == "<login-node>":
            print("      (replace <login-node> with your cluster login host, or pass --login-host)")
    else:
        print(f"      ssh -L {port}:localhost:{port} {user}@{node}")
    print(f"\n  Then open:  http://localhost:{port}\n")
    print("=" * 70 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="torchsx result viewer (stream + raw frames + on-demand score).")
    ap.add_argument("--stream", required=True, help="path to a torchsx .sfx stream (the result cache)")
    ap.add_argument("--geom", default=None, help="detector .geom (optional if embedded in the stream)")
    ap.add_argument("--data-root", default=None,
                    help="base dir for resolving relative .h5 paths stored in the stream "
                         "(default: stream dir, then CWD)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="device for the matched-filter score (default auto → cuda if available)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--login-host", default=None)
    args = ap.parse_args()

    app = create_app(args.stream, args.geom, args.device, args.data_root)
    _print_tunnel(args.port, args.login_host)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
