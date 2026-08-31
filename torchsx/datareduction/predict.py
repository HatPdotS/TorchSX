"""Prediction-based reflection enumeration + the signal-vs-offset diagnostic.

The Phase-C complement to detection-based extraction: instead of integrating only
the peaks the hitfinder happened to find and the indexer confirmed, *predict*
every reflection the refined lattice should produce within resolution, keep those
near the Ewald sphere, and integrate each at its predicted detector position
(observed or not — this is how weak / unobserved reflections enter the dataset).

The inclusion lever is a single **Ewald-offset cutoff** ``|r| ≤ offset_max`` (Å⁻¹),
a per-beamline/sample hyperparameter. :func:`signal_vs_offset` integrates under a
deliberately large cutoff and bins mean I/σ by ``|r|`` so the cutoff can be read
off where signal decays into the noise floor.

Excitation error (the lever) is the same definition used everywhere else
(:func:`torchsx.merging.observations._ewald_offset`)::

    r = ‖q_lab + s₀‖ − 1/λ ,   q_lab = h·a⋆ + k·b⋆ + l·c⋆ ,   s₀ = (0, 0, 1/λ)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ._schema import FS, PANEL, SS
from .to_stream import reciprocal_vectors


@dataclass
class PredictedReflections:
    """Reflections predicted from a refined lattice and kept on-detector."""

    hkl: Tensor              # (M, 3) int16
    panel: Tensor            # (M,) int64
    ss: Tensor               # (M,) float32 — assembled slow-scan
    fs: Tensor               # (M,) float32 — fast-scan within panel
    excitation_error: Tensor  # (M,) float32 — signed r (Å⁻¹)
    d_spacing: Tensor        # (M,) float32 — d of each reflection (Å)

    def __len__(self) -> int:
        return int(self.hkl.shape[0])

    def spot_table(self) -> Tensor:
        """``(M, 4)`` ``[panel, ss, fs, 0]`` view for :func:`integrate_reflections`."""
        m = len(self)
        out = torch.zeros((m, 4), dtype=torch.float32)
        out[:, PANEL] = self.panel.to(torch.float32)
        out[:, SS] = self.ss
        out[:, FS] = self.fs
        return out


def _cell6_from_basis(basis: Tensor) -> Tensor:
    """``(6,)`` cell ``[a, b, c, α, β, γ]`` (Å, deg) from a real-space basis (rows a,b,c)."""
    a, b, c = basis[0], basis[1], basis[2]

    def ang(u, v):
        cos = torch.dot(u, v) / (u.norm() * v.norm())
        return torch.rad2deg(torch.arccos(cos.clamp(-1.0, 1.0)))

    return torch.tensor([
        float(a.norm()), float(b.norm()), float(c.norm()),
        float(ang(b, c)), float(ang(a, c)), float(ang(a, b)),
    ], dtype=torch.float32)


def predict_reflections(
    basis: Tensor,
    geom,
    *,
    d_min: float,
    offset_max: float,
    wavelength: float | None = None,
) -> PredictedReflections:
    """Enumerate hkl within ``d_min``, keep ``|r| ≤ offset_max``, project to pixels.

    Parameters
    ----------
    basis : (3, 3) refined real-space cell (rows a, b, c in Å) — e.g.
        ``IndexResult.basis``.
    geom : detector :class:`~torchsx.geometry.Geometry`.
    d_min : high-resolution limit (Å) for hkl enumeration.
    offset_max : Ewald-offset cutoff |r| (Å⁻¹) — the inclusion lever.
    wavelength : Å; defaults to ``geom.wavelength_A``.
    """
    from torchref.base.reciprocal.hkl import generate_possible_hkl, get_d_spacing

    lam = geom.wavelength_A if wavelength is None else float(wavelength)
    k0 = 1.0 / lam
    cell6 = _cell6_from_basis(basis)

    hkl = generate_possible_hkl(cell6, d_min).to(torch.float32)        # (M0, 3)
    bstar = reciprocal_vectors(basis).to(torch.float32)                # rows a*,b*,c*
    q = hkl @ bstar                                                    # (M0, 3) Å⁻¹

    s0 = torch.zeros_like(q)
    s0[:, 2] = k0
    r = (q + s0).norm(dim=1) - k0                                      # excitation error
    near = r.abs() <= float(offset_max)
    if not bool(near.any()):
        z = torch.zeros(0)
        return PredictedReflections(torch.zeros(0, 3, dtype=torch.int16),
                                    z.to(torch.int64), z, z, z, z)

    hkl_n = hkl[near]
    q_n = q[near]
    r_n = r[near]

    fs, ss, pid = geom.recip_to_pixel(q_n, lam)
    on = pid >= 0
    d = get_d_spacing(hkl_n[on], cell6).to(torch.float32)
    return PredictedReflections(
        hkl=hkl_n[on].to(torch.int16),
        panel=pid[on],
        ss=ss[on].to(torch.float32),
        fs=fs[on].to(torch.float32),
        excitation_error=r_n[on].to(torch.float32),
        d_spacing=d,
    )


def signal_vs_offset(
    reader,
    geom,
    frames,
    *,
    offset_max_large: float,
    n_bins: int = 20,
    backend: str = "torch",
    params=None,
    n_calib: int = 60,
    min_num_spots: int = 8,
    int_radius: int = 4,
    initial_cell: Tensor | None = None,
    device: str = "cpu",
) -> list[dict]:
    """Bin mean I/σ by ``|r|`` to pick ``offset_max`` (the diagnostic).

    Over a sample of indexable ``frames``: detect → index → predict with a large
    ``offset_max_large`` → integrate at predicted positions → collect
    ``(|r|, I/σ)``. Returns a per-bin table; the knee where mean I/σ falls toward
    ~1 is the noise floor and a sensible ``offset_max``. Reference-free; returns
    data (no plotting).
    """
    from .indexing import HEWL_CELL, index_spots
    from .integrate import integrate_reflections
    from .spotfinding import find_spots_on_frame, spots_to_q
    from .to_stream import _calibrate, _resolve_params
    from ._schema import PROB

    params = _resolve_params(backend, params)
    initial_cell = HEWL_CELL if initial_cell is None else initial_cell

    noise_model = null = profile = None
    nm, nl, prof, fp, good, _, _ = _calibrate(reader, geom, params, n_calib=n_calib, device=device,
                                              want_null=(backend in ("torch", "bayes")), backend=backend)
    noise_model, null, profile = nm, nl, prof
    fixed_pattern = fp.to(device) if fp is not None else None
    good_mask = good.to(device) if good is not None else None

    abs_r, i_over_sig, i_vals = [], [], []
    for i in frames:
        spots = find_spots_on_frame(reader, i, geom, params, backend=backend,
                                    noise_model=noise_model, null=null, profile=profile,
                                    mask=good_mask, fixed_pattern=fixed_pattern)
        if spots.shape[0] < min_num_spots:
            continue
        q = spots_to_q(spots, geom)
        weights = spots[:, PROB] if spots.shape[1] > PROB else None
        res = index_spots(q, initial_cell, weights=weights, device=device,
                          min_num_spots=min_num_spots)
        if not res.success or res.n_indexed < min_num_spots:
            continue
        # resolution limit from the indexed spots
        d_min = 1.0 / float(spots_to_q(spots[res.inlier_mask], geom).norm(dim=1).max())
        pred = predict_reflections(res.basis, geom, d_min=d_min,
                                   offset_max=offset_max_large)
        if len(pred) == 0:
            continue
        panels = reader.panels(int(i)).to(torch.float32)
        good = reader.mask_panels().to(torch.bool)
        P, n_ss, n_fs = panels.shape
        radius = geom.pixel_radius_px.reshape(P, n_ss, n_fs).to(torch.float32)
        integ = integrate_reflections(panels, good, radius, pred.spot_table(),
                                      noise_model, r_int=int_radius)
        abs_r.append(pred.excitation_error.abs())
        i_over_sig.append(integ.I / integ.sigma.clamp(min=1e-6))
        i_vals.append(integ.I)

    if not abs_r:
        return []
    r = torch.cat(abs_r)
    ios = torch.cat(i_over_sig)
    iv = torch.cat(i_vals)
    edges = torch.linspace(0.0, float(offset_max_large), n_bins + 1)
    table = []
    for b in range(n_bins):
        lo, hi = float(edges[b]), float(edges[b + 1])
        m = (r >= lo) & (r < hi)
        c = int(m.sum())
        table.append({
            "r_lo": lo, "r_hi": hi, "count": c,
            "mean_I_over_sigma": float(ios[m].mean()) if c else float("nan"),
            "mean_I": float(iv[m].mean()) if c else float("nan"),
        })
    return table
