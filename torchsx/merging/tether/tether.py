"""The tether: a set of terms, their penalty for a training batch and their readout over a merge."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import Tensor

from torchsx.merging.tether.batch import TetherBatch
from torchsx.merging.tether.kmean import integral_mode, integral_step, step_mean_square
from torchsx.merging.tether.response import STEPS, response_slope, slope_curvature
from torchsx.merging.tether.terms import TERMS, KMeanTerm, ResponseTerm

__all__ = ["Curves", "Tether", "parse_terms"]

#: Evaluates each request ``(group of every observation, number of groups, nodes)``,
#: with nodes ``n_groups x 3`` in ``log I``, as the summed log-likelihood of every
#: group at its nodes (``n_groups x 3``), in the graph of the scale and scatter
#: predictions where training needs it.
Curves = Callable[[list[tuple[Tensor, int, Tensor]]], list[Tensor]]


def parse_terms(spec: str | Iterable[str]) -> tuple[str, ...]:
    """Term names from ``"gauge,strength,response"``, ``"none"`` or an iterable, in
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
    nats^2, of the log-intensity change that the K-mean terms' closed-form
    step and the response terms' slopes stand for.  At any positive weight its
    minimum is where every term is zero, which the data put there: it fixes
    directions the likelihood leaves free (or nearly so) without biasing the
    fit, and its weight sets how stiffly they are held.  :meth:`readout` reports
    every term over a whole merge.
    """

    #: node spacing in ``log I`` of the response curve
    DELTA = 0.05
    #: observations a (reflection, half) group needs to enter a response term
    #: (a quarter needs half as many)
    MIN_OBS = 10
    #: groups a batch needs for its response terms to count
    MIN_GROUPS = 200

    def __init__(self, terms: Iterable[str], *, min_obs: int | None = None, min_groups: int | None = None,
                 runaway: float = 10.0):
        self.names = parse_terms(terms)
        self.terms = [TERMS[n] for n in self.names]
        self.kmean = [t for t in self.terms if isinstance(t, KMeanTerm)]
        self.response = [t for t in self.terms if isinstance(t, ResponseTerm)]
        self.min_obs = int(self.MIN_OBS if min_obs is None else min_obs)
        self.min_groups = int(self.MIN_GROUPS if min_groups is None else min_groups)
        self.runaway = float(runaway)

    def __bool__(self) -> bool:
        return bool(self.terms)

    # ── K-mean terms ──────────────────────────────────────────────────

    def _kmean_columns(self, batch: TetherBatch):
        """``(phi, x, layout)``: the per-reflection and per-observation columns of all
        K-mean terms and each term's ``(slice of b, columns)``."""
        refl = [(t, t.columns(batch)) for t in self.kmean if t.per_reflection]
        obs = [(t, t.columns(batch)) for t in self.kmean if not t.per_reflection]
        dev = batch.n_signal.device
        phi = (torch.cat([c for _, c in refl], 1) if refl
               else torch.zeros(batch.n_hkl, 0, device=dev, dtype=torch.float64))
        x = torch.cat([c for _, c in obs], 1) if obs else None
        layout, at = [], 0
        for t, c in refl + obs:
            layout.append((t, slice(at, at + c.shape[1]), c))
            at += c.shape[1]
        return phi, x, layout

    # ── response terms ────────────────────────────────────────────────

    def _response_requests(self, batch: TetherBatch, min_obs: int):
        """Halves and quarters of the crystals: each part's groups, the selection, the
        covariates and the nodes ``log I_lin + DELTA * (-1, 0, 1)``."""
        cov = [t.covariate(batch) for t in self.response]
        out = []
        for n_parts, need in ((2, min_obs), (4, max(1, min_obs // 2))):
            g = batch.groups(n_parts)
            half = batch.half_of_row(n_parts)
            sel = (g["count"] >= need) & (g["I_lin"] > 0)
            for _, s in cov:
                sel = sel & s[half]
            X = torch.stack([x[half] for x, _ in cov], 1)
            v0 = torch.maximum(g["I_lin"], g["sd"]).clamp(min=1e-12).log()
            steps = torch.tensor(STEPS, device=v0.device, dtype=torch.float64)
            v = (v0[:, None] + self.DELTA * steps).to(torch.float32).contiguous()
            out.append({"n_parts": n_parts, "rows": g["rows"], "n_rows": g["n_rows"], "sel": sel, "X": X, "v": v})
        return out

    def _response_fit(self, batch: TetherBatch, req: dict, S: Tensor, min_groups: int):
        s, F = slope_curvature(S, req["v"])
        with torch.no_grad():
            idx = (req["sel"] & torch.isfinite(s) & torch.isfinite(F) & (F > 0)).nonzero(as_tuple=True)[0]
        if idx.numel() < min_groups:
            return None, None, int(idx.numel())
        A = torch.cat([torch.ones(idx.numel(), 1, device=s.device, dtype=torch.float64),
                       batch.phi.index_select(0, idx % batch.n_hkl)], 1)
        kappa, C = response_slope(s[idx], F[idx], A, req["X"][idx])
        return kappa, C, int(idx.numel())

    def _response(self, batch: TetherBatch, curves: Curves, min_obs: int, min_groups: int):
        reqs = self._response_requests(batch, min_obs)
        S = curves([(r["rows"], r["n_rows"], r["v"]) for r in reqs])
        fits = [self._response_fit(batch, r, s, min_groups) for r, s in zip(reqs, S)]
        rec = {"n_groups": fits[0][2]}
        if any(f[0] is None for f in fits):
            return None, None, rec
        (k2, C, _), (k4, _, _) = fits
        rec.update(half=k2.detach(), quarter=k4.detach())
        return 2 * k2 - k4, C, rec

    # ── training and readout ──────────────────────────────────────────

    def penalty(self, batch: TetherBatch, curves: Curves | None = None) -> tuple[Tensor, dict]:
        """The batch's penalty (in the graph through ``batch.log_K`` and, for response
        terms, the curves) and a record: ``tether_<name>`` per term (the RMS mode in
        nats for the gauge, the slope otherwise), ``tether_<name>_half`` and
        ``_quarter`` for response terms, ``tether_groups``, ``tether_penalty``."""
        dev = batch.n_signal.device
        pen = torch.zeros((), device=dev, dtype=torch.float64)
        rec = {}
        if self.kmean:
            phi, x, layout = self._kmean_columns(batch)
            b = integral_step(phi, batch.hkl_idx, batch.log_K, batch.crystal_id, batch.n_crystals,
                              extra=x, runaway=self.runaway)
            pen = pen + step_mean_square(phi, batch.hkl_idx, b, x)
            for t, sl, cols in layout:
                rec[f"tether_{t.name}"] = t.summary(b[sl], cols, batch)
        if self.response:
            if curves is None:
                raise ValueError("response terms need the likelihood's curves")
            kappa, C, r = self._response(batch, curves, self.min_obs, self.min_groups)
            rec["tether_groups"] = r["n_groups"]
            if kappa is not None:
                pen = pen + kappa @ C @ kappa
                for i, t in enumerate(self.response):
                    rec[f"tether_{t.name}"] = float(kappa.detach()[i])
                    rec[f"tether_{t.name}_half"] = float(r["half"][i])
                    rec[f"tether_{t.name}_quarter"] = float(r["quarter"][i])
        rec["tether_penalty"] = float(pen.detach())
        return pen, rec

    @torch.no_grad()
    def readout(self, batch: TetherBatch, curves: Curves | None = None, *,
                d_spacing: Tensor | None = None) -> dict:
        """Every term over a whole set of observations (a merge): the per-reflection K-mean
        terms by the exact fit, the per-observation ones by one step once those are
        removed, the response terms over halves and quarters of all the crystals."""
        out = {"n_obs": int(batch.hkl_idx.numel())}
        if not out["n_obs"]:
            return out
        if self.kmean:
            phi, x, layout = self._kmean_columns(batch)
            lk = batch.log_K.detach().to(torch.float64)
            b_r = torch.zeros(phi.shape[1], device=lk.device, dtype=torch.float64)
            w = None
            if phi.shape[1]:
                b_r, w = integral_mode(phi, batch.hkl_idx, lk, batch.crystal_id, batch.n_crystals,
                                       runaway=self.runaway, return_weights=True)
            b = b_r
            if x is not None:
                lk0 = lk - (phi @ b_r).index_select(0, batch.hkl_idx)
                step = integral_step(phi, batch.hkl_idx, lk0, batch.crystal_id, batch.n_crystals,
                                     extra=x, runaway=self.runaway)
                b = torch.cat([b_r, step[phi.shape[1]:]])
            for t, sl, cols in layout:
                kw = {"d_spacing": d_spacing, "weights": w} if t.per_reflection else {}
                out[t.name] = t.report(b[sl], cols, batch, **kw)
        if self.response and curves is not None:
            kappa, _, r = self._response(batch, curves, self.min_obs, 1)
            for i, t in enumerate(self.response):
                out[t.name] = ({"kappa": None, "n_groups": r["n_groups"]} if kappa is None else
                               {"kappa": float(kappa[i]), "kappa_half": float(r["half"][i]),
                                "kappa_quarter": float(r["quarter"][i]), "n_groups": r["n_groups"]})
        return out
