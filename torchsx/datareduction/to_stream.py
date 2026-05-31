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
from .indexing import HEWL_CELL, IndexResult, index_spots
from .peakfinder import PROB
from .spotfinding import FS, INTENSITY, PANEL, SS, PeakFinder8Params, find_spots_on_frame, spots_to_q


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
    """
    mask = result.inlier_mask if only_inliers else torch.ones(
        result.n_spots, dtype=torch.bool
    )
    sp = spots[mask]
    hkl = result.hkl[mask].to(torch.int16)
    n = int(sp.shape[0])

    I = sp[:, INTENSITY].to(torch.float32)
    # per-spot probability/weight (probabilistic finder appends a PROB column);
    # persisted so refinement can reload `w` from the .sfx. 1.0 if not present.
    prob = (sp[:, PROB].to(torch.float32) if sp.shape[1] > PROB
            else torch.ones(n, dtype=torch.float32))
    obs = {
        "hkl": hkl,
        "I": I,
        "sigma": torch.sqrt(I.clamp(min=1.0)),       # Poisson placeholder
        "background": torch.zeros(n, dtype=torch.float32),
        "peak": I.clone(),
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
    verbose: bool = False,
) -> tuple[SFXStream, dict]:
    """Index a list of frames and assemble them into one :class:`SFXStream`.

    Returns ``(stream, summary)``. Frames that don't index (or yield too few
    indexed spots) are skipped; ``summary`` reports how many crystals made it in.

    For ``backend="torch"`` with ``calibrate=True`` a one-time per-run calibration
    pass fits the noise model ``σ²(μ)`` and the empirical null on ``n_calib`` sampled
    frames, so the per-peak probabilities are calibrated and multiplicity-corrected.
    """
    if params is None:
        if backend.startswith("torch"):
            from .peakfinder import TorchPeakFinderParams
            params = TorchPeakFinderParams()
        else:
            params = PeakFinder8Params()

    noise_model = null = None
    calib_dbg = None
    if backend == "torch" and calibrate:
        from .noise_model import calibrate_noise, calibrate_null
        frames = list(frames)
        n = reader.n_frames
        step = max(1, n // max(1, n_calib))
        calib_idx = list(range(0, n, step))[:n_calib]
        noise_model, calib_dbg = calibrate_noise(reader, geom, calib_idx, device=device)
        null = calibrate_null(reader, geom, noise_model, calib_idx,
                              peak_min_distance=params.peak_min_distance, device=device)
        if verbose:
            print(f"[calib] noise σ²(μ) coef a0..a3="
                  f"{[round(c, 4) for c in calib_dbg['coef']]}  "
                  f"null n_lmax/frame={null.n_lmax_per_frame:.0f}")

    builder = StreamBuilder.initialize(
        geometry=geom, spacegroup=spacegroup, source_path=source_path
    )

    n_tried = n_indexed = total_obs = 0
    for i in frames:
        n_tried += 1
        spots = find_spots_on_frame(reader, i, geom, params, backend=backend,
                                    noise_model=noise_model, null=null)
        if spots.shape[0] < min_num_spots:
            continue
        q = spots_to_q(spots, geom)
        # probabilistic finder appends a PROB column → use it to weight indexing
        weights = spots[:, PROB] if spots.shape[1] > PROB else None
        res = index_spots(q, initial_cell, weights=weights, device=device,
                          min_num_spots=min_num_spots)
        if not res.success or res.n_indexed < min_num_spots:
            continue
        chunk = crystal_to_chunk(
            spots, res, geom, frame_key=(reader.path, int(i)),
            n_peaks_frame=int(spots.shape[0]),
        )
        builder.add(chunk)
        n_indexed += 1
        total_obs += res.n_indexed
        if verbose:
            print(f"  frame {i}: indexed {res.n_indexed}/{spots.shape[0]} "
                  f"cell={[round(float(x),1) for x in res.cell_lengths]}")

    core = builder.finalize()
    stream = SFXStream(core)
    summary = {
        "frames_tried": n_tried,
        "crystals_indexed": n_indexed,
        "n_observations": total_obs,
        "spacegroup": spacegroup,
    }
    return stream, summary
