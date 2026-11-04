"""The tether: a set of terms, their penalty for a training batch and their readout over a merge."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

from torchsx.merging.tether.batch import TetherBatch
from torchsx.merging.tether.kmean import integral_mode, integral_step, step_mean_square
from torchsx.merging.tether.terms import TERMS

__all__ = ["Tether", "parse_terms"]


def parse_terms(spec: str | Iterable[str]) -> tuple[str, ...]:
    """Term names from ``"gauge"``, a comma-separated list, ``"none"`` or an iterable, in
    :data:`~torchsx.merging.tether.terms.TERMS` order; unknown names raise."""
    names = spec.split(",") if isinstance(spec, str) else list(spec)
    names = {n.strip() for n in names if n.strip() and n.strip() != "none"}
    unknown = names - set(TERMS)
    if unknown:
        raise ValueError(f"unknown tether term(s) {sorted(unknown)}; known: {', '.join(TERMS)}")
    return tuple(n for n in TERMS if n in names)


class Tether:
    """The sum-of-spots tether over the named terms.

    :meth:`penalty` is the training term for one batch: the mean square, in
    nats^2, of the log-intensity change that the terms' closed-form step stands
    for.  At any positive weight its minimum is where every term is zero, which
    the data put there: it fixes directions the likelihood leaves free without
    biasing the fit, and its weight sets only how stiffly they are held.
    :meth:`readout` reports every term over a whole merge.
    """

    def __init__(self, terms: Iterable[str], *, runaway: float = 10.0):
        self.names = parse_terms(terms)
        self.terms = [TERMS[n] for n in self.names]
        self.runaway = float(runaway)

    def __bool__(self) -> bool:
        return bool(self.terms)

    def _columns(self, batch: TetherBatch):
        """``(phi, x, layout)``: the per-reflection and per-observation columns of all terms
        and each term's ``(slice of b, columns)``."""
        refl = [(t, t.columns(batch)) for t in self.terms if t.per_reflection]
        obs = [(t, t.columns(batch)) for t in self.terms if not t.per_reflection]
        dev = batch.log_K.device
        phi = (torch.cat([c for _, c in refl], 1) if refl
               else torch.zeros(batch.n_hkl, 0, device=dev, dtype=torch.float64))
        x = torch.cat([c for _, c in obs], 1) if obs else None
        layout, at = [], 0
        for t, c in refl + obs:
            layout.append((t, slice(at, at + c.shape[1]), c))
            at += c.shape[1]
        return phi, x, layout

    def penalty(self, batch: TetherBatch) -> tuple[Tensor, dict]:
        """The batch's penalty (in the graph through ``batch.log_K``) and a record:
        ``tether_<name>`` per term and ``tether_penalty``."""
        phi, x, layout = self._columns(batch)
        b = integral_step(phi, batch.hkl_idx, batch.log_K, batch.crystal_id, batch.n_crystals,
                          extra=x, runaway=self.runaway)
        pen = step_mean_square(phi, batch.hkl_idx, b, x)
        rec = {f"tether_{t.name}": t.summary(b[sl], cols, batch) for t, sl, cols in layout}
        rec["tether_penalty"] = float(pen.detach())
        return pen, rec

    @torch.no_grad()
    def readout(self, batch: TetherBatch, *, d_spacing: Tensor | None = None) -> dict:
        """Every term over a whole set of observations (a merge): the per-reflection terms by
        the exact fit, the per-observation ones by one step once those are removed."""
        out = {"n_obs": int(batch.hkl_idx.numel())}
        if not out["n_obs"] or not self.terms:
            return out
        phi, x, layout = self._columns(batch)
        lk = batch.log_K.detach().to(torch.float64)
        b = torch.zeros(phi.shape[1], device=lk.device, dtype=torch.float64)
        w = None
        if phi.shape[1]:
            b, w = integral_mode(phi, batch.hkl_idx, lk, batch.crystal_id, batch.n_crystals,
                                 runaway=self.runaway, return_weights=True)
        if x is not None:
            lk0 = lk - (phi @ b).index_select(0, batch.hkl_idx)
            step = integral_step(phi, batch.hkl_idx, lk0, batch.crystal_id, batch.n_crystals,
                                 extra=x, runaway=self.runaway)
            b = torch.cat([b, step[phi.shape[1]:]])
        for t, sl, cols in layout:
            kw = {"d_spacing": d_spacing, "weights": w} if t.per_reflection else {}
            out[t.name] = t.report(b[sl], cols, batch, **kw)
        return out
