"""Weighted variant of the SDSC PyTorch TORO indexer.

Subclasses :class:`toro.models.indexer_model.ToroIndexer` so a per-peak weight
``w ∈ [0,1]`` (e.g. the probabilistic finder's peak probability) drives lattice
scoring: low-confidence peaks contribute proportionally instead of all-or-nothing.

Only the two per-peak-sum methods are overridden — their bodies mirror the pinned
vendored versions at ``submodules/toro_torch/toro/models/indexer_model.py`` (the
2026 clone), with the weight injected:

* ``compute_scores_and_hkl`` — candidate lattices ranked by the *weighted* inlier
  mass ``Σ(is_inlier · w)`` instead of the raw inlier count ``Σ is_inlier``.
* ``index_candidate_solutions`` — the trimmed least-squares refine becomes a
  *weighted* least squares (rows scaled by ``√w``).

The geometric path keeps the 3-column ``peaks`` (no weight packed into matmuls);
``self._peak_weights`` aligns to peaks by index and is multiplied only after the
non-zero / inlier masks, so it stays valid through TORO's multi-crystal loop
(used peaks are zeroed, hence contribute 0). With uniform weights the results
match the base indexer.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

_TORO_DIR = Path(__file__).resolve().parents[2] / "submodules" / "toro_torch"
if str(_TORO_DIR) not in sys.path:
    sys.path.insert(0, str(_TORO_DIR))

from toro.models.indexer_model import (  # type: ignore  # noqa: E402
    ToroIndexer,
    batched_invert_matrix,
    batched_regression,
    bcompute_penalization,
)


class WeightedToroIndexer(ToroIndexer):
    """``ToroIndexer`` with optional per-peak weighting.

    Set :attr:`_peak_weights` (shape ``(bs, N)``, in ``[0,1]``) before calling
    ``forward``; leave it ``None`` to recover the unweighted base behaviour.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._peak_weights: torch.Tensor | None = None

    def set_peak_weights(self, weights: torch.Tensor | None) -> None:
        self._peak_weights = weights

    def compute_scores_and_hkl(self, peaks, rotated_bases, dist_to_integer: float = 0.12):
        rearanged_bases = rotated_bases.permute(0, 2, 3, 1)
        projections = torch.stack([
            torch.bmm(peaks, rearanged_bases[:, 0, :, :]),
            torch.bmm(peaks, rearanged_bases[:, 1, :, :]),
            torch.bmm(peaks, rearanged_bases[:, 2, :, :]),
        ], -1).permute(0, 2, 1, 3)

        hkl = torch.round(projections)
        distance = torch.norm(projections - hkl, dim=-1, p=2)
        is_inlier = distance <= dist_to_integer

        w = self._peak_weights
        if w is None:
            scores = torch.sum(is_inlier, dim=-1)
        else:
            # is_inlier (bs, num_bases, N) · w (bs, N) → weighted inlier mass
            scores = torch.sum(
                is_inlier.to(self.dtype) * w[:, None, :].to(self.dtype), dim=-1
            )
        return scores, hkl, is_inlier

    def index_candidate_solutions(
        self, bases, peaks, peaks_mask, initial_cell, num_iterations: int
    ):
        start = 0.01
        base = 2 ** (math.log(self.error_precision * 100) / ((num_iterations - 1) * math.log(2)))
        error_bounds = [float(start * base ** i) for i in range(num_iterations)]
        error = torch.zeros_like(peaks_mask, dtype=self.dtype)
        non_zero_mask = torch.sum(torch.abs(peaks), dim=-1) != 0

        # √w broadcast over the candidate (num_top) axis → (bs, 1, N, 1)
        w = self._peak_weights
        sqrt_w = None
        if w is not None:
            sqrt_w = torch.sqrt(w[:, None, :].clamp(min=0.0).to(self.dtype))[..., None]

        for rnd in range(num_iterations):
            masked_peaks = peaks * peaks_mask[..., None]
            targets = torch.round(masked_peaks @ bases.transpose(-2, -1))
            X, B = masked_peaks, targets
            if sqrt_w is not None:
                X, B = masked_peaks * sqrt_w, targets * sqrt_w
            transposed_bases = (
                torch.linalg.lstsq(X, B)[0]
                if peaks.device == torch.device("cpu")
                else batched_regression(X, B)
            )
            bases = transposed_bases.transpose(-2, -1)

            predictions = peaks @ transposed_bases
            back_points = torch.round(predictions) @ batched_invert_matrix(transposed_bases)
            error = torch.norm(peaks - back_points, dim=-1, p=2)

            peaks_mask = error < error_bounds[rnd]
            peaks_mask &= non_zero_mask

        penalization = bcompute_penalization(bases, initial_cell)
        return bases, peaks_mask, error * peaks_mask, penalization
