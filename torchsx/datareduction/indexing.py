"""Index reciprocal-space spots with the SDSC PyTorch TORO indexer.

Wraps ``toro.models.indexer_model.ToroIndexer`` (installed from the
``submodules/toro_torch`` clone).  TORO needs an initial cell — it refines an
orientation of that cell against the spots; it cannot find a cell from scratch.

The indexer is torch-native and runs on CPU or GPU.  ``forward`` returns, for a
batch of frames, the best multi-crystal solutions ordered best-first; this
wrapper drives a single frame and reports the top solution.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

# Standard hen egg-white lysozyme (P 43 21 2), rows = real-space a, b, c (Å).
HEWL_CELL = torch.tensor(
    [[79.0, 0.0, 0.0], [0.0, 79.0, 0.0], [0.0, 0.0, 38.0]], dtype=torch.float32
)

# The SDSC PyTorch TORO clone is a namespace package (no toro/__init__.py), so a
# pip editable install maps nothing; import it straight off sys.path instead.
_TORO_DIR = Path(__file__).resolve().parents[2] / "submodules" / "toro_torch"


def _load_toro():
    if str(_TORO_DIR) not in sys.path:
        sys.path.insert(0, str(_TORO_DIR))
    try:
        from toro.models.indexer_model import ToroIndexer  # type: ignore
    except ImportError as e:  # pragma: no cover - depends on clone state
        raise ImportError(
            "PyTorch TORO clone not found. Expected it at:\n"
            f"  {_TORO_DIR}\n"
            "Clone it with:\n"
            "  git clone https://github.com/SwissDataScienceCenter/"
            "toro-indexer-for-serial-crystallography submodules/toro_torch\n"
            f"(original error: {e})"
        ) from e
    return ToroIndexer


def _load_weighted_toro():
    _load_toro()  # ensure the clone is importable (adds it to sys.path)
    from ._toro_weighted import WeightedToroIndexer
    return WeightedToroIndexer


@dataclass
class IndexResult:
    """Result of indexing one frame's spots."""

    success: bool
    basis: torch.Tensor          # (3, 3) refined real-space cell, rows a,b,c (Å)
    inlier_mask: torch.Tensor    # (N,) bool — spots belonging to the solution
    hkl: torch.Tensor            # (N, 3) int — assigned Miller indices
    n_indexed: int
    n_spots: int
    max_error: float
    peak_prob: torch.Tensor | None = None       # (N,) per-peak weight used, if any
    weighted_inlier_mass: float | None = None    # Σ weights over inliers (hit score)
    # max-likelihood refinement (refine.py), populated when index_spots(refine=True)
    refined_basis: torch.Tensor | None = None    # (3,3) refined real-space cell
    responsibilities: torch.Tensor | None = None  # (N,) P(spot ∈ lattice)
    loglik: float | None = None                  # mixture log-likelihood at optimum
    hit_score: float | None = None               # Σ log Bayes factor (lattice vs noise)

    @property
    def cell_lengths(self) -> torch.Tensor:
        """Refined cell edge lengths (|a|, |b|, |c|) in Å."""
        return self.basis.norm(dim=1)

    @property
    def cell_angles(self) -> torch.Tensor:
        """Refined cell angles (α, β, γ) in degrees."""
        a, b, c = self.basis[0], self.basis[1], self.basis[2]

        def ang(u, v):
            cos = torch.dot(u, v) / (u.norm() * v.norm())
            return torch.rad2deg(torch.arccos(cos.clamp(-1.0, 1.0)))

        return torch.stack([ang(b, c), ang(a, c), ang(a, b)])


def index_spots(
    q: torch.Tensor,
    initial_cell: torch.Tensor = HEWL_CELL,
    *,
    weights: torch.Tensor | None = None,
    refine: bool = False,
    sigma_d: float = 0.0015,
    refine_cell: bool = True,
    cell_restraint: float = 0.0,
    device: str | torch.device = "cpu",
    min_num_spots: int = 8,
    angle_resolution: int = 200,
    num_top_solutions: int = 200,
    lattice_size: int = 50000,
    num_iterations: int = 5,
    error_precision: float = 0.0012,
    filter_precision: float = 0.00075,
    filter_min_num_spots: int = 6,
) -> IndexResult:
    """Index a single frame's reciprocal-space spots.

    Parameters
    ----------
    q : (N, 3) tensor — reciprocal-space scattering vectors in Å⁻¹.
    initial_cell : (3, 3) tensor — initial real-space cell, rows a,b,c (Å).
    weights : optional ``(N,)`` per-peak weights ∈ [0,1] (e.g. peak probabilities).
        When given, the :class:`WeightedToroIndexer` ranks/fits lattices by a
        *weighted* inlier mass so low-confidence peaks contribute proportionally;
        ``None`` uses the standard equal-weight indexer.
    device : where to run the indexer (``"cpu"`` or ``"cuda"``).
    min_num_spots, angle_resolution, num_top_solutions :
        TORO ``params`` (search resolution + min spots for a viable crystal).
    lattice_size, num_iterations, error_precision, filter_precision,
    filter_min_num_spots : TORO constructor hyperparameters.

    Returns
    -------
    :class:`IndexResult` — the top solution (best-first), or a failure result
    when no viable solution is found.
    """
    device = torch.device(device)

    n_spots = int(q.shape[0])
    peaks = q.to(device=device, dtype=torch.float32).unsqueeze(0)  # (1, N, 3)
    initial_cell = initial_cell.to(device=device, dtype=torch.float32)

    w = None
    if weights is not None:
        w = weights.to(device=device, dtype=torch.float32).reshape(-1)
        Indexer = _load_weighted_toro()
    else:
        Indexer = _load_toro()

    im = Indexer(
        lattice_size=lattice_size,
        num_iterations=num_iterations,
        error_precision=error_precision,
        filter_precision=filter_precision,
        filter_min_num_spots=filter_min_num_spots,
    ).to(device)
    if w is not None:
        im.set_peak_weights(w.unsqueeze(0))  # (1, N)

    params = [
        torch.tensor(min_num_spots),
        torch.tensor(angle_resolution),
        torch.tensor(num_top_solutions),
    ]
    successes, bases, masks, errors, _pen = im(peaks, initial_cell, params)

    # no-solution sentinel: forward returns 1-D zero tensors
    if bases.ndim < 4 or bases.shape[1] == 0:
        return IndexResult(
            success=False,
            basis=torch.zeros(3, 3),
            inlier_mask=torch.zeros(n_spots, dtype=torch.bool),
            hkl=torch.zeros(n_spots, 3, dtype=torch.long),
            n_indexed=0,
            n_spots=n_spots,
            max_error=float("nan"),
            peak_prob=w.detach().cpu() if w is not None else None,
            weighted_inlier_mass=0.0 if w is not None else None,
        )

    # crystal 0 is the best (solutions are appended best-first)
    basis = bases[0, 0].detach().cpu()              # (3, 3)
    inlier_mask = masks[0, 0].detach().cpu().bool() # (N,)
    success = bool(successes[0, 0])
    max_error = float(errors[0, 0])

    # hkl = round(q · a/b/c) with rows of basis being the real-space cell vectors
    proj = q.to(torch.float32) @ basis.T            # (N, 3)
    hkl = torch.round(proj).long()

    peak_prob = w.detach().cpu() if w is not None else None
    weighted_mass = (
        float((inlier_mask.to(torch.float32) * peak_prob).sum())
        if peak_prob is not None
        else None
    )

    refined_basis = responsibilities = loglik = hit_score = None
    if refine:
        from .refine import refine_lattice
        rw = peak_prob if peak_prob is not None else torch.ones(n_spots)
        rr = refine_lattice(
            q.to(torch.float32).cpu(), rw, basis,
            sigma_d=sigma_d, refine_cell=refine_cell,
            cell_restraint=cell_restraint, ref_cell=initial_cell.detach().cpu(),
        )
        refined_basis = rr.basis
        responsibilities = rr.responsibilities
        loglik, hit_score = rr.loglik, rr.hit_score
        # re-assign hkl / inliers from the refined basis
        hkl = torch.round(q.to(torch.float32) @ refined_basis.T).long()
        inlier_mask = responsibilities > 0.5

    return IndexResult(
        success=success,
        basis=basis,
        inlier_mask=inlier_mask,
        hkl=hkl,
        n_indexed=int(inlier_mask.sum()),
        n_spots=n_spots,
        max_error=max_error,
        peak_prob=peak_prob,
        weighted_inlier_mass=weighted_mass,
        refined_basis=refined_basis,
        responsibilities=responsibilities,
        loglik=loglik,
        hit_score=hit_score,
    )
