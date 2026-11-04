"""The tether's terms: each one a direction the likelihood leaves to the model, held at the counts.

A term is a :class:`KMeanTerm`: the arithmetic mean of ``K`` over the
observations carries no trend along the term's columns.  All terms' columns
enter one fit (:func:`~torchsx.merging.tether.kmean.integral_step`), so their
coefficients are separated from each other and from the crystal constants.  A
term's columns are given per reflection (``per_reflection``) or per
observation.

A new direction is a subclass with a ``name``, its columns and its report;
adding it to :data:`TERMS` makes it available to training
(``TrainConfig.tether``, ``torchsx.train --tether``) and to every merge's
readout.  A term's columns must come from the counts or the geometry, never
from ``K`` or the merged intensities: then the term is zero where the data put
it.
"""

from __future__ import annotations

import torch
from torch import Tensor

from torchsx.merging.tether.batch import TetherBatch

__all__ = ["TERMS", "KMeanTerm", "SmoothGauge"]


class KMeanTerm:
    """The arithmetic mean of ``K`` carries no trend along :meth:`columns`."""

    name: str = ""
    #: ``columns`` is ``n_hkl x q`` (indexed by each observation's reflection)
    #: instead of ``n_obs x q``
    per_reflection: bool = False

    def columns(self, batch: TetherBatch) -> Tensor:
        raise NotImplementedError

    def summary(self, coef: Tensor, cols: Tensor, batch: TetherBatch) -> float:
        """One number for the training record, from this term's coefficients."""
        raise NotImplementedError

    def report(self, coef: Tensor, cols: Tensor, batch: TetherBatch, **kw) -> dict:
        """The merge readout, from this term's coefficients over all observations."""
        return {"value": float(self.summary(coef, cols, batch))}


class SmoothGauge(KMeanTerm):
    """The rocking-curve integral of ``K`` is the same at every resolution and in every
    direction: the smooth reciprocal basis (without its constant column) as columns.

    The likelihood is exactly flat along ``K -> K g(h), I0 -> I0 / g(h)`` for a
    smooth ``g``; this term is what fixes it, with the scale model pinning the
    scalar gauge only (``anchor_stat="median"``).
    """

    name = "gauge"
    per_reflection = True

    def columns(self, batch: TetherBatch) -> Tensor:
        return batch.phi

    def summary(self, coef, cols, batch) -> float:
        """RMS over the observations of the log factor the gauge would still apply, in nats."""
        m = (cols.to(coef) @ coef.detach()).index_select(0, batch.hkl_idx)
        return float((m * m).mean().sqrt()) if m.numel() else 0.0

    def report(self, coef, cols, batch, *, d_spacing: Tensor | None = None, n_shells: int = 10,
               weights: Tensor | None = None) -> dict:
        """RMS over the observations of ``phi b`` -- the log factor by which the sum-of-spots
        gauge would raise each reflection's intensity -- its largest magnitude, the
        observations the runaway guard left out and, with ``d_spacing``, its observation
        mean in ``n_shells`` shells of equal reflection count, low resolution first."""
        mode = cols.to(coef) @ coef
        n_h = torch.bincount(batch.hkl_idx, minlength=batch.n_hkl).to(coef.dtype)
        seen = n_h > 0
        out = {"rms": float(((n_h * mode ** 2).sum() / n_h.sum().clamp(min=1.0)).sqrt()),
               "max_abs": float(mode[seen].abs().max()) if bool(seen.any()) else 0.0}
        if weights is not None:
            out["n_runaway"] = int((weights == 0).sum())
        if d_spacing is not None:
            d = d_spacing.to(coef.device, torch.float64)
            idx = seen.nonzero(as_tuple=True)[0]
            order = idx[torch.argsort(d[idx], descending=True)]
            out["by_shell"] = [
                {"d_lo": float(d[s].max()), "d_hi": float(d[s].min()),
                 "mean": float((n_h[s] * mode[s]).sum() / n_h[s].sum()), "n_obs": int(n_h[s].sum())}
                for s in torch.tensor_split(order, n_shells) if s.numel()
            ]
        return out


#: Every term, by name.  The order is the order of their columns.
TERMS = {t.name: t for t in (SmoothGauge(),)}
