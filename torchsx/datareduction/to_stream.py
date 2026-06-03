"""Fold indexed crystals back into an :class:`SFXStream`.

Closes the ingestion loop: detector frames → spots → TORO indexing → a columnar
``SFXStream`` (via :class:`~torchsx.io._stream_builder.StreamBuilder`), so the
data-reduction output flows straight into the existing merging pipeline.

One indexed crystal becomes one :class:`StreamChunk` (1 image + 1 crystal + its
indexed observations). The builder offsets ``crystal_id`` / ``image_id`` across
chunks and concatenates them into the three-level core.
"""

from __future__ import annotations

import torch

from ..geometry import Geometry
from ..io._stream_builder import StreamBuilder, StreamChunk
from ..stream import SFXStream
from ._schema import FS, INTENSITY, PANEL, PROB, SS
from .indexing import HEWL_CELL, IndexResult, index_spots
from .integrate import IntegrationResult, integrate_reflections
from .spotfinding import PeakFinder8Params, find_spots_on_frame, spots_to_q


def _make_pbar(total, *, enabled: bool, interval: float):
    """A progress bar that reads well in both a TTY and a redirected log.

    Uses ``tqdm_loggable.auto.tqdm`` — on a TTY it's a normal live bar; in a log
    / pipe it falls back to ``tqdm_logging``, which emits a throttled INFO log
    line (rate + ETA + our ``indexed``/``obs`` postfix) instead of carriage-return
    spam. **Callers that want the log lines to appear must configure logging at
    INFO** (e.g. ``logging.basicConfig(level=logging.INFO)``); on a TTY no logging
    setup is needed. Falls back to plain ``tqdm`` then to ``None`` (disabled).
    """
    if not enabled:
        return None
    try:
        from tqdm_loggable.auto import tqdm as _tqdm
        from tqdm_loggable.tqdm_logging import tqdm_logging
        from datetime import timedelta
        tqdm_logging.set_log_rate(timedelta(seconds=float(interval)))
    except Exception:  # noqa: BLE001 — tqdm_loggable absent → plain tqdm or off
        try:
            from tqdm.auto import tqdm as _tqdm
        except Exception:
            return None
    return _tqdm(total=total, desc="extract", unit="fr")


def reciprocal_vectors(basis: torch.Tensor) -> torch.Tensor:
    """Reciprocal lattice vectors (rows a*, b*, c*) from a real-space ``basis``.

    ``basis`` rows are the real-space cell vectors a, b, c (Å). With the
    crystallographic convention (no 2π), ``a*·a = 1`` and ``a*·b = a*·c = 0``,
    which gives ``B* = inv(B)ᵀ`` — its rows are a*, b*, c* in Å⁻¹.
    """
    return torch.linalg.inv(basis).T


def crystal_to_chunk(
    spots: torch.Tensor,
    result: IndexResult,
    geom: Geometry,
    *,
    frame_key=None,
    only_inliers: bool = True,
    beam_energy_eV: float | None = None,
    n_peaks_frame: int | None = None,
    profile_rad: float = 0.004,
    integration: IntegrationResult | None = None,
) -> StreamChunk:
    """Build a one-crystal :class:`StreamChunk` from an indexing result.

    Parameters
    ----------
    spots : (M, 4) tensor — ``[panel, ss, fs, intensity]`` for the frame.
    result : the :class:`IndexResult` for this frame.
    geom : detector geometry (for q-vectors / wavelength / camera length).
    only_inliers : keep only spots assigned to the lattice (recommended).
    beam_energy_eV : photon energy; defaults to ``geom.photon_energy_eV``.
    n_peaks_frame : total peaks found on the frame (image-level); defaults to M.
    integration : optional :class:`~torchsx.datareduction.integrate.IntegrationResult`
        for the same (masked, inlier-ordered) spots; when given, ``I``/``sigma``/
        ``background``/``peak`` come from the proper disk integration + noise-model
        propagation. When ``None``, the finder's quick-sum ``I`` is used with a
        Poisson ``sigma`` placeholder (back-compat).
    """
    mask = result.inlier_mask if only_inliers else torch.ones(
        result.n_spots, dtype=torch.bool
    )
    sp = spots[mask]
    hkl = result.hkl[mask].to(torch.int16)
    n = int(sp.shape[0])

    if integration is not None:
        if integration.I.shape[0] != n:
            raise ValueError(
                f"integration has {integration.I.shape[0]} reflections but "
                f"{n} (masked) spots — they must be aligned (same inlier order)."
            )
        I = integration.I.to(torch.float32)
        sigma = integration.sigma.to(torch.float32)
        background = integration.background.to(torch.float32)
        peak = integration.peak.to(torch.float32)
    else:
        I = sp[:, INTENSITY].to(torch.float32)
        sigma = torch.sqrt(I.clamp(min=1.0))             # Poisson placeholder
        background = torch.zeros(n, dtype=torch.float32)
        peak = I.clone()

    # per-spot probability/weight (probabilistic finder appends a PROB column);
    # persisted so refinement can reload `w` from the .sfx. 1.0 if not present.
    prob = (sp[:, PROB].to(torch.float32) if sp.shape[1] > PROB
            else torch.ones(n, dtype=torch.float32))
    obs = {
        "hkl": hkl,
        "I": I,
        "sigma": sigma,
        "background": background,
        "peak": peak,
        "fs": sp[:, FS].to(torch.float32),
        "ss": sp[:, SS].to(torch.float32),
        "panel_id": sp[:, PANEL].to(torch.int16),
        "crystal_id": torch.zeros(n, dtype=torch.int32),  # chunk-local
        "prob": prob,
    }

    bstar_mat = reciprocal_vectors(result.basis)         # rows a*, b*, c* (Å⁻¹)
    q = spots_to_q(sp, geom)
    d_min = 1.0 / float(q.norm(dim=1).max()) if n else 0.0
    cell6 = torch.cat([result.cell_lengths, result.cell_angles]).to(torch.float32)
    crystal = {
        "astar": bstar_mat[0:1].to(torch.float32),
        "bstar": bstar_mat[1:2].to(torch.float32),
        "cstar": bstar_mat[2:3].to(torch.float32),
        "cell": cell6.unsqueeze(0),
        "profile_rad": torch.tensor([profile_rad], dtype=torch.float32),
        "resolution": torch.tensor([d_min], dtype=torch.float32),
        "n_refl": torch.tensor([n], dtype=torch.int32),
        "image_id": torch.zeros(1, dtype=torch.int32),       # chunk-local
        "crystal_in_image": torch.zeros(1, dtype=torch.int32),
    }

    energy = geom.photon_energy_eV if beam_energy_eV is None else beam_energy_eV
    image = {
        "beam_energy": torch.tensor([energy], dtype=torch.float32),
        "det_shift_x": torch.zeros(1, dtype=torch.float32),
        "det_shift_y": torch.zeros(1, dtype=torch.float32),
        "n_peaks": torch.tensor([n_peaks_frame or n], dtype=torch.int32),
        "camera_length": torch.tensor([geom.clen], dtype=torch.float32),
    }

    return StreamChunk(levels={"obs": obs, "crystal": crystal, "image": image},
                       key=frame_key)


def predicted_to_chunk(
    pred,
    integration: IntegrationResult,
    basis: torch.Tensor,
    cell6: torch.Tensor,
    geom: Geometry,
    *,
    frame_key=None,
    beam_energy_eV: float | None = None,
    n_peaks_frame: int | None = None,
    profile_rad: float = 0.004,
) -> StreamChunk:
    """Build a one-crystal :class:`StreamChunk` from PREDICTED reflections.

    The prediction-based counterpart to :func:`crystal_to_chunk`: the ``hkl`` and
    detector positions come from :func:`~torchsx.datareduction.predict.predict_reflections`
    (the refined lattice), and ``I``/``sigma``/``background``/``peak`` from
    integrating at those predicted positions. ``cell6`` is the refined cell
    ``[a,b,c,α,β,γ]`` (from ``IndexResult.cell_lengths``/``cell_angles``).
    """
    n = len(pred)
    obs = {
        "hkl": pred.hkl.to(torch.int16),
        "I": integration.I.to(torch.float32),
        "sigma": integration.sigma.to(torch.float32),
        "background": integration.background.to(torch.float32),
        "peak": integration.peak.to(torch.float32),
        "fs": pred.fs.to(torch.float32),
        "ss": pred.ss.to(torch.float32),
        "panel_id": pred.panel.to(torch.int16),
        "crystal_id": torch.zeros(n, dtype=torch.int32),
        "prob": torch.ones(n, dtype=torch.float32),   # predicted: no per-peak weight
    }
    bstar_mat = reciprocal_vectors(basis)
    d_min = float(pred.d_spacing.min()) if n else 0.0
    crystal = {
        "astar": bstar_mat[0:1].to(torch.float32),
        "bstar": bstar_mat[1:2].to(torch.float32),
        "cstar": bstar_mat[2:3].to(torch.float32),
        "cell": cell6.to(torch.float32).unsqueeze(0),
        "profile_rad": torch.tensor([profile_rad], dtype=torch.float32),
        "resolution": torch.tensor([d_min], dtype=torch.float32),
        "n_refl": torch.tensor([n], dtype=torch.int32),
        "image_id": torch.zeros(1, dtype=torch.int32),
        "crystal_in_image": torch.zeros(1, dtype=torch.int32),
    }
    energy = geom.photon_energy_eV if beam_energy_eV is None else beam_energy_eV
    image = {
        "beam_energy": torch.tensor([energy], dtype=torch.float32),
        "det_shift_x": torch.zeros(1, dtype=torch.float32),
        "det_shift_y": torch.zeros(1, dtype=torch.float32),
        "n_peaks": torch.tensor([n_peaks_frame or n], dtype=torch.int32),
        "camera_length": torch.tensor([geom.clen], dtype=torch.float32),
    }
    return StreamChunk(levels={"obs": obs, "crystal": crystal, "image": image},
                       key=frame_key)


def _resolve_params(backend: str, params):
    """Pick a default finder-params object for ``backend`` when none is given."""
    if params is not None:
        return params
    if backend.startswith("torch") or backend == "bayes":
        from .peakfinder import TorchPeakFinderParams
        return TorchPeakFinderParams()
    return PeakFinder8Params()


def _calibrate(reader, geom: Geometry, params, *, n_calib: int, device: str,
               want_null: bool = True, backend: str = "pf8", verbose: bool = False,
               sigma_radial: float = 2.0, sigma_azimuthal: float = 0.7):
    """Per-run calibration pass: fit the noise model ``σ²(μ)`` (+ empirical null).

    Samples ``n_calib`` frames evenly across ``reader.n_frames``. The noise model
    is always fit (it feeds integration for *every* backend); the empirical null
    is fit only when ``want_null`` (the probabilistic finders need it, and it
    requires ``params.peak_min_distance``). For the ``"bayes"`` backend the
    matched-filter ``profile`` is a **constant anisotropic Gaussian** —
    ``σ_radial``/``σ_azimuthal`` (radially elongated): still-shot reflections are
    radially smeared by the beam's spectral bandwidth but azimuthally sharp, so this
    indexes ~+45% more crystals than an isotropic profile (and the bootstrap
    ``calibrate_spot_profile`` over-estimated σ ~3× and missed the anisotropy). The
    null is fit on that score. Returns ``(noise_model, null, profile, dbg)``.
    """
    import math
    from .noise_model import calibrate_noise, calibrate_null
    from .spot_profile import SpotProfile
    n = reader.n_frames
    step = max(1, n // max(1, n_calib))
    calib_idx = list(range(0, n, step))[:n_calib]
    noise_model, calib_dbg = calibrate_noise(reader, geom, calib_idx, device=device)
    profile = None
    if backend == "bayes":
        # NOTE: σ_azimuthal≈0.7 is robust (mosaicity/divergence); σ_radial≈2.0 is the
        # spectral-dispersion width at mid-resolution for THIS beam (~0.4% bandwidth) —
        # it scales with bandwidth/geometry, so revisit σ_radial per dataset.
        R = int(min(8, max(2, math.ceil(3 * sigma_radial))))
        profile = SpotProfile.constant(sigma_radial, sigma_azimuthal,
                                       kernel_radius=R, n_orient=12)
    null = None
    if want_null:
        null = calibrate_null(reader, geom, noise_model, calib_idx,
                              peak_min_distance=params.peak_min_distance, device=device,
                              profile=profile)
    if verbose:
        msg = (f"[calib] noise σ²(μ) coef a0..a3="
               f"{[round(c, 4) for c in calib_dbg['coef']]}")
        if profile is not None:
            sr, sa = profile.sigma_radial, profile.sigma_azimuthal
            msg += (f"  spot σ_rad={float(sr.min()):.2f}-{float(sr.max()):.2f} "
                    f"σ_azim={float(sa.min()):.2f}-{float(sa.max()):.2f} "
                    f"over {profile.n_bands} radius band(s)")
        if null is not None:
            msg += f"  null n_lmax/frame={null.n_lmax_per_frame:.0f}"
        print(msg)
    return noise_model, null, profile, calib_dbg


def _index_frame(reader, geom: Geometry, i, params, *, backend: str,
                 noise_model, null, initial_cell: torch.Tensor, device: str,
                 min_num_spots: int, int_radius: int = 4,
                 hit_min_sumprob: float = 2.0, profile=None, operator=None,
                 predict: bool = False, offset_max: float | None = None,
                 pred_d_min: float | None = None):
    """Find spots on one frame, index them, and integrate.

    Returns ``(chunk, n_obs, n_spots, result)`` for a successfully-indexed frame,
    or ``None`` if the frame has too few spots or fails to index. Two integration
    modes:

    * default (``predict=False``) — integrate the indexed **inliers** (found ∩
      indexed) at their found centroids;
    * ``predict=True`` — use the refined lattice to PREDICT every reflection within
      resolution with ``|Ewald offset| ≤ offset_max`` and integrate each at its
      predicted detector position (recovers weak/unobserved reflections).

    Both use the fixed-radius disk + global-radial background + per-pixel
    ``noise_model`` propagation of :func:`~torchsx.datareduction.integrate.integrate_reflections`.
    """
    # Read the frame once (on the working device) and reuse it for both the
    # finder and integration — the finder used to read it and integration re-read it.
    # On CUDA, panels_gpu decodes the bslz4 chunk on the GPU (nvcomp LZ4 + Triton
    # inverse-bitshuffle, ~9 ms vs ~35 ms CPU, ~6.6 MB across PCIe vs ~68 MB); it
    # transparently falls back to the CPU h5py path off-CUDA or without nvcomp/Triton.
    dev = torch.device(device)
    if hasattr(reader, "panels_gpu"):
        panels = reader.panels_gpu(int(i), dev).to(torch.float32)
    else:
        panels = reader.panels(int(i)).to(torch.float32).to(dev)
    good = reader.mask_panels().to(torch.bool).to(dev)
    P, n_ss, n_fs = panels.shape
    radius = geom.pixel_radius_px.reshape(P, n_ss, n_fs).to(torch.float32).to(dev)

    spots = find_spots_on_frame(reader, i, geom, params, backend=backend,
                                noise_model=noise_model, null=null, profile=profile,
                                panels=panels, mask=good, pix_r=radius, operator=operator)
    if spots.shape[0] < min_num_spots:
        return None
    # Hit gate: the probabilistic finder (PROB column) returns top-N peaks on
    # *every* frame, so Σ peak-probability ≈ expected # of real peaks separates
    # hits from background. Skip the (expensive) indexer on near-blank frames.
    if spots.shape[1] > PROB and float(spots[:, PROB].sum()) < hit_min_sumprob:
        return None
    q = spots_to_q(spots, geom)
    # probabilistic finder appends a PROB column → use it to weight indexing
    weights = spots[:, PROB] if spots.shape[1] > PROB else None
    res = index_spots(q, initial_cell, weights=weights, device=device,
                      min_num_spots=min_num_spots, refine=True)
    # Accept on the *refined* inlier count, NOT TORO's pre-refine `success` flag.
    # That flag rejects real crystals the ML refine recovers: blanks refine to ~0
    # inliers while real hits refine to ~100+ (a clean bimodal gap on lyso), so
    # `success` discarded >half the true hits. Refining every gated frame costs
    # more (refine is ~95 ms/frame on the A100) but ~doubles the indexed yield.
    if res.n_indexed < min_num_spots:
        return None

    # Frame pixels for integration: reuse the tensors loaded above (single read).
    # The per-frame compute (find/index/integrate) ran on `dev`; the chunk/stream
    # boundary is host-side, so we land results back on CPU before building chunks.
    inlier_dev = res.inlier_mask.to(dev)
    if predict:
        from .predict import predict_reflections  # lazy: predict imports to_stream
        # resolution from the indexed inliers unless overridden
        d_min = pred_d_min
        if d_min is None:
            qn = spots_to_q(spots[inlier_dev], geom).norm(dim=1).max()
            d_min = 1.0 / float(qn) if float(qn) > 0 else 0.0
        pred = predict_reflections(res.basis, geom, d_min=d_min,
                                   offset_max=offset_max)
        if len(pred) < min_num_spots:
            return None
        integ = integrate_reflections(panels, good, radius,
                                      pred.spot_table().to(dev),
                                      noise_model, r_int=int_radius)
        cell6 = torch.cat([res.cell_lengths, res.cell_angles])
        chunk = predicted_to_chunk(
            pred, integ.cpu(), res.basis, cell6, geom,
            frame_key=(reader.path, int(i)), n_peaks_frame=int(spots.shape[0]),
        )
        return chunk, len(pred), int(spots.shape[0]), res

    # default: integrate the indexed inliers at their found centroids
    sp_in = spots[inlier_dev]
    integ = integrate_reflections(panels, good, radius, sp_in, noise_model,
                                  r_int=int_radius)
    chunk = crystal_to_chunk(
        spots.cpu(), res, geom, frame_key=(reader.path, int(i)),
        n_peaks_frame=int(spots.shape[0]), integration=integ.cpu(),
    )
    return chunk, int(res.n_indexed), int(spots.shape[0]), res


def build_stream(
    reader,
    geom: Geometry,
    frames,
    *,
    params=None,
    initial_cell: torch.Tensor = HEWL_CELL,
    spacegroup: str = "P 43 21 2",
    device: str = "cpu",
    min_num_spots: int = 8,
    source_path: str = "",
    backend: str = "pf8",
    calibrate: bool = True,
    n_calib: int = 150,
    int_radius: int = 4,
    hit_min_sumprob: float = 2.0,
    sigma_radial: float = 2.0,
    sigma_azimuthal: float = 0.7,
    predict: bool = False,
    offset_max: float | None = None,
    pred_d_min: float | None = None,
    progress: bool = True,
    progress_interval: float = 10.0,
    verbose: bool = False,
) -> tuple[SFXStream, dict]:
    """Index a list of frames and assemble them into one :class:`SFXStream`.

    Thin orchestrator over :func:`_resolve_params`, :func:`_calibrate`, and
    :func:`_index_frame`. Returns ``(stream, summary)``. Frames that don't index
    (or yield too few indexed spots) are skipped; ``summary`` reports how many
    crystals made it in.

    When ``calibrate=True`` a one-time per-run pass fits the detector noise model
    ``σ²(μ)`` on ``n_calib`` sampled frames — needed by integration for *every*
    backend — plus the empirical null (probabilistic finder only). Each indexed
    frame's source ``(filename, event)`` is recorded in the stream's image-level
    metadata so the result can be re-opened by :meth:`SFXStream.excise_patches`.
    """
    params = _resolve_params(backend, params)
    if predict and offset_max is None:
        raise ValueError("predict=True requires offset_max (the Ewald-offset cutoff, Å⁻¹)")

    noise_model = null = profile = operator = None
    if calibrate:
        noise_model, null, profile, _ = _calibrate(
            reader, geom, params, n_calib=n_calib, device=device,
            want_null=(backend in ("torch", "bayes")), backend=backend, verbose=verbose,
            sigma_radial=sigma_radial, sigma_azimuthal=sigma_azimuthal,
        )
    # Build the fused matched-filter operator ONCE (geometry+mask+profile fixed) and
    # reuse it for every frame — the bayes finder's ~13x speedup over the conv path.
    if backend == "bayes" and profile is not None and torch.device(device).type == "cuda":
        from .spot_profile import build_matched_filter_operator
        P0 = reader.n_panels
        n_ss0, n_fs0 = 514, geom.detector_shape[1]
        good0 = reader.mask_panels().to(torch.bool).to(device)
        r0 = geom.pixel_radius_px.reshape(P0, n_ss0, n_fs0).to(torch.float32).to(device)
        ang0 = geom.pixel_angle_rad.reshape(P0, n_ss0, n_fs0).to(torch.float32).to(device)
        operator = build_matched_filter_operator(ang0, r0, good0, profile)

    builder = StreamBuilder.initialize(
        geometry=geom, spacegroup=spacegroup, source_path=source_path
    )

    total = len(frames) if hasattr(frames, "__len__") else None
    pbar = _make_pbar(total, enabled=progress, interval=progress_interval)

    n_tried = n_indexed = total_obs = 0
    for i in frames:
        n_tried += 1
        out = _index_frame(reader, geom, i, params, backend=backend,
                           noise_model=noise_model, null=null, profile=profile,
                           operator=operator,
                           initial_cell=initial_cell, device=device,
                           min_num_spots=min_num_spots, int_radius=int_radius,
                           hit_min_sumprob=hit_min_sumprob,
                           predict=predict, offset_max=offset_max,
                           pred_d_min=pred_d_min)
        if out is not None:
            chunk, n_obs, n_spots, res = out
            builder.add(chunk)
            n_indexed += 1
            total_obs += n_obs
            if verbose:
                print(f"  frame {i}: indexed {n_obs}/{n_spots} "
                      f"cell={[round(float(x),1) for x in res.cell_lengths]}")
        if pbar is not None:
            pbar.update(1)
            # indexed-vs-processed + accumulated observations; rate/ETA are native
            pbar.set_postfix(indexed=f"{n_indexed}/{n_tried}", obs=total_obs,
                             refresh=False)
    if pbar is not None:
        pbar.close()

    core = builder.finalize()
    _attach_frame_metadata(core, builder)
    stream = SFXStream(core)
    summary = {
        "frames_tried": n_tried,
        "crystals_indexed": n_indexed,
        "n_observations": total_obs,
        "spacegroup": spacegroup,
    }
    return stream, summary


def _attach_frame_metadata(core, builder) -> None:
    """Write per-image ``filename``/``event`` into ``core.metadata['image']``.

    :class:`~torchsx.io._excise.excise_patches` resolves each observation to a raw
    frame via ``metadata['image']['filename']`` + ``['event']`` (``"//N"`` → frame
    index ``N``). ``build_stream`` doesn't otherwise populate these, so excision on
    a datareduction-built stream would fail. We reconstruct them from the builder's
    ``key → row-range`` index (``key = (reader.path, frame_index)``), placing each
    frame's identity at its image row — robust to chunk ordering.
    """
    n_img = core.n_images
    filename = [""] * n_img
    event = [""] * n_img
    for key, ranges in builder.index.items():
        if "image" not in ranges or not isinstance(key, tuple) or len(key) != 2:
            continue
        start, _ = ranges["image"]
        path, frame_i = key
        if 0 <= start < n_img:
            filename[start] = str(path)
            event[start] = f"//{int(frame_i)}"
    core.metadata["image"] = {"filename": filename, "event": event}
