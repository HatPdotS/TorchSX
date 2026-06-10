"""Live re-running of the per-frame pipeline (find → index → predict).

Live preview only — this never mutates the stream. It splits the work that
:func:`torchsx.datareduction.to_stream._index_frame` does on one frame into three
independently-callable stages so the UI can tune each:

* ``find``    — re-run the bayes matched-filter finder with a chosen ``top_n``
                (the number of hits that would feed the indexer) and kernel σ;
* ``index``   — index a peak list against a typed seed cell;
* ``predict`` — predict reflections from a (possibly hand-edited) real-space basis
                with a chosen Ewald-offset cutoff ``offset_max`` and ``d_min``.

The bayes finder needs a per-detector noise model + empirical null for its
probability column; these are calibrated once per kernel ``(σ_radial, σ_azimuthal)``
(cached) on a sample of frames from the referenced file — the first ``find`` of a
session pays a few seconds, the rest are fast.
"""

from __future__ import annotations

import numpy as np

from ._score import resolve_device


class Reprocessor:
    DEFAULT_SR = 2.0   # radial kernel σ (spectral bandwidth)
    DEFAULT_SA = 0.7   # azimuthal kernel σ (divergence/mosaicity)

    def __init__(self, geom, reader_for, device: str = "auto", n_calib: int = 12):
        """``reader_for(i)`` returns ``(JungfrauReader, frame_index)`` for image
        ``i`` — reuse the server's LRU-cached reader so frames aren't re-opened."""
        import torch

        self.geom = geom
        self._reader_for = reader_for
        self.device = torch.device(resolve_device(device))
        self.n_calib = int(n_calib)
        P = geom.n_panels
        nss, nfs = 514, geom.detector_shape[1]
        self._radius = geom.pixel_radius_px.reshape(P, nss, nfs).to(torch.float32).to(self.device)
        self._angle = geom.pixel_angle_rad.reshape(P, nss, nfs).to(torch.float32).to(self.device)
        self._calib: dict[tuple, tuple] = {}  # (sr,sa) -> (noise_model, null, profile)

    # ── calibration (lazy, per kernel) ──────────────────────────────────────
    def _calibrate(self, reader, sr: float, sa: float):
        key = (round(sr, 3), round(sa, 3))
        if key not in self._calib:
            from torchsx.datareduction.peakfinder import TorchPeakFinderParams
            from torchsx.datareduction.to_stream import _calibrate
            nm, null, profile, _ = _calibrate(
                reader, self.geom, TorchPeakFinderParams(),
                n_calib=self.n_calib, device=str(self.device),
                want_null=True, backend="bayes",
                sigma_radial=sr, sigma_azimuthal=sa,
            )
            self._calib[key] = (nm, null, profile)
        return self._calib[key]

    # ── stage 1: find peaks ──────────────────────────────────────────────────
    def find(self, i: int, *, top_n: int = 200, sigma_radial: float | None = None,
             sigma_azimuthal: float | None = None, peak_min_distance: int = 2,
             hit_min_sumprob: float = 2.0) -> dict:
        import torch

        from torchsx.datareduction._schema import FS, INTENSITY, PANEL, PROB, SS
        from torchsx.datareduction.peakfinder import TorchPeakFinderParams
        from torchsx.datareduction.spotfinding import find_spots_on_frame

        sr = self.DEFAULT_SR if sigma_radial is None else float(sigma_radial)
        sa = self.DEFAULT_SA if sigma_azimuthal is None else float(sigma_azimuthal)
        reader, fidx = self._reader_for(i)
        nm, null, profile = self._calibrate(reader, sr, sa)

        params = TorchPeakFinderParams(top_n=int(top_n),
                                       peak_min_distance=int(peak_min_distance))
        if hasattr(reader, "panels_gpu"):
            panels = reader.panels_gpu(fidx, self.device).to(torch.float32)
        else:
            panels = reader.panels(fidx).to(torch.float32).to(self.device)
        good = reader.mask_panels().to(torch.bool).to(self.device)
        spots = find_spots_on_frame(
            reader, fidx, self.geom, params, backend="bayes",
            noise_model=nm, null=null, profile=profile,
            panels=panels, mask=good, pix_r=self._radius, angle=self._angle,
        ).detach().cpu().numpy()

        has_prob = spots.shape[1] > PROB
        peaks = [{
            "panel": int(s[PANEL]), "ss": float(s[SS]), "fs": float(s[FS]),
            "intensity": float(s[INTENSITY]),
            "prob": float(s[PROB]) if has_prob else None,
        } for s in spots]
        sum_prob = float(spots[:, PROB].sum()) if (has_prob and spots.size) else 0.0
        return {"peaks": peaks, "n_found": len(peaks),
                "sum_prob": sum_prob, "hit": bool(sum_prob >= hit_min_sumprob)}

    # ── stage 2: index ────────────────────────────────────────────────────────
    def index(self, spots: list[dict], *, cell: list[float] | None = None,
              min_num_spots: int = 8) -> dict:
        import torch

        from torchsx.datareduction._schema import PROB
        from torchsx.datareduction.indexing import HEWL_CELL, index_spots
        from torchsx.datareduction.predict import _cell6_from_basis
        from torchsx.datareduction.spotfinding import spots_to_q

        if not spots:
            return {"success": False, "n_indexed": 0, "reason": "no peaks to index"}
        # (M,5) [panel, ss, fs, intensity, prob]; missing prob → weight 1.
        arr = np.array([[p["panel"], p["ss"], p["fs"],
                         p.get("intensity", 0.0),
                         1.0 if p.get("prob") is None else p["prob"]]
                        for p in spots], dtype=np.float32)
        spots_t = torch.from_numpy(arr)
        q = spots_to_q(spots_t, self.geom)
        has_prob = all(p.get("prob") is not None for p in spots)
        weights = spots_t[:, PROB] if has_prob else None
        initial_cell = self._basis_from_cell(cell) if cell else HEWL_CELL
        res = index_spots(q, initial_cell, weights=weights, device=str(self.device),
                          min_num_spots=int(min_num_spots), refine=True)
        basis = res.refined_basis if getattr(res, "refined_basis", None) is not None else res.basis
        basis = basis.detach().cpu()
        cell6 = _cell6_from_basis(basis)
        return {
            "success": bool(int(res.n_indexed) >= int(min_num_spots)),
            "n_indexed": int(res.n_indexed),
            "max_error": float(res.max_error),
            "basis": basis.numpy().tolist(),
            "cell": cell6.numpy().tolist(),
        }

    @staticmethod
    def _basis_from_cell(cell6: list[float]):
        """Real-space basis (rows a, b, c) from a ``[a,b,c,α,β,γ]`` cell.

        ``reciprocal_basis_matrix`` gives the reciprocal rows a*,b*,c*; the
        crystallographic ``B* = inv(B)ᵀ`` is involutive, so inverting it via
        :func:`reciprocal_vectors` returns the real-space basis."""
        import torch

        from torchref.base.reciprocal.basis import reciprocal_basis_matrix

        from torchsx.datareduction.to_stream import reciprocal_vectors
        c = torch.tensor([float(x) for x in cell6], dtype=torch.float32)
        return reciprocal_vectors(reciprocal_basis_matrix(c))

    # ── stage 3: predict ───────────────────────────────────────────────────────
    def predict(self, basis: list[list[float]], *, d_min: float, offset_max: float) -> dict:
        import torch

        from torchsx.datareduction.predict import predict_reflections

        b = torch.tensor([[float(x) for x in row] for row in basis], dtype=torch.float32)
        pred = predict_reflections(b, self.geom, d_min=float(d_min), offset_max=float(offset_max))
        n = len(pred)
        hkl = pred.hkl.cpu().numpy()
        panel = pred.panel.cpu().numpy()
        ss = pred.ss.cpu().numpy()
        fs = pred.fs.cpu().numpy()
        r = pred.excitation_error.cpu().numpy()
        d = pred.d_spacing.cpu().numpy()
        refl = [{
            "panel": int(panel[k]), "ss": float(ss[k]), "fs": float(fs[k]),
            "hkl": [int(x) for x in hkl[k]],
            "excitation_error": float(r[k]), "d_spacing": float(d[k]),
        } for k in range(n)]
        return {"reflections": refl, "n_predicted": n}
