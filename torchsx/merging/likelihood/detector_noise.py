"""Learned detector noise: a per-panel variance polynomial in the count.

The detection likelihood is a deviance-scaled Poisson in the counts,
``log p = (-D/2) * iphi + 1/2 log iphi + logpeak`` with ``iphi = lam / V`` and
``V = lam + sig_b2`` in photon units (``lam = S + bg``: the expected count of the
peak at signal ``S``; ``sig_b2``: the variance of the background estimate).  To the
order the Poisson saddlepoint holds this is a Gaussian of variance ``V`` in the
counts, with the Poisson's skew at low counts.  Everything a detector does beyond
Poisson at its declared gain is absent from it.  This module replaces ``V`` by::

    V = A_p + B_p (lam + sig_b2) + C_p lam^2          (p = the observation's panel)

- ``A``  additive variance (read noise, pedestal; photons^2);
- ``B``  dispersion of the counting term: a gain that is off (B = true/declared);
- ``C``  relative variance proportional to the count squared (gain nonuniformity).

The variance depends on the observation's **count** and **panel** only: nothing
per shot, crystal or geometry.  The count is the expected count ``lam`` at each
node of the signal integral, i.e. the detector's variance given the true count --
never the measured ``n``, whose use in a variance biases the fit the way
count-dependent weights bias a weighted mean.  It is condition-blind by
construction.  ``A = 0, B = 1, C = 0`` is the plain model; the initial values are
that up to negligible ``A0``, ``C0``.

Parameters are ``log A, log B, log C`` per panel, **projected** into bounds after
every optimizer step (:meth:`project`), not squashed by a saturating function:
Adam normalises a small persistent gradient to a full step, and behind a
saturating map the raw value then walks without limit.  ``C <= 0.04`` (20 %
relative SD) also keeps the likelihood's decay in ``lam`` (it levels off at
``-1/(2C)``) deep enough for the split grid's window.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["DetectorNoise"]

#: ``(min, max)`` of each term: A in photons^2, B and C dimensionless
BOUNDS = {"A": (1e-4, 1e4), "B": (0.02, 100.0), "C": (1e-8, 0.04)}
INIT = {"A": 1e-2, "B": 1.0, "C": 1e-5}


class DetectorNoise(nn.Module):
    """``V = A[p] + B[p] (lam + sig_b2) + C[p] lam^2`` per detector panel ``p``.

    Parameters
    ----------
    n_groups
        Detector panels; observations carry ``panel_id`` in ``[0, n_groups)``.
    terms
        Which of ``"A"``, ``"B"``, ``"C"`` are fitted; the others stay at their
        initial value (``"B"`` alone is a per-panel gain correction).
    """

    def __init__(self, n_groups: int = 1, *, terms: str = "ABC") -> None:
        super().__init__()
        if n_groups < 1:
            raise ValueError("n_groups must be positive")
        bad = set(terms) - set("ABC")
        if bad or not terms:
            raise ValueError(f"terms must be a non-empty subset of 'ABC', got {terms!r}")
        self.n_groups, self.terms = int(n_groups), "".join(t for t in "ABC" if t in terms)
        for t in "ABC":
            p = nn.Parameter(torch.full((self.n_groups,), math.log(INIT[t])), requires_grad=t in self.terms)
            self.register_parameter(f"log_{t}", p)

    def config(self) -> dict:
        return {"n_groups": self.n_groups, "terms": self.terms}

    @classmethod
    def from_config(cls, cfg: dict) -> "DetectorNoise":
        return cls(int(cfg["n_groups"]), terms=str(cfg.get("terms", "ABC")))

    @torch.no_grad()
    def project(self) -> None:
        """Clamp every term into :data:`BOUNDS` (call after each optimizer step)."""
        for t, (lo, hi) in BOUNDS.items():
            getattr(self, f"log_{t}").clamp_(math.log(lo), math.log(hi))

    def terms_obs(self, group: Tensor | None, n: int, device=None) -> tuple[Tensor, Tensor, Tensor]:
        """``(A, B, C)`` per observation, differentiable in the fitted terms."""
        if group is None:
            group = torch.zeros(n, dtype=torch.long, device=device or self.log_A.device)
        g = group.long()
        return self.log_A[g].exp(), self.log_B[g].exp(), self.log_C[g].exp()

    def values(self) -> dict:
        """Current ``A, B, C`` per panel, for logging."""
        return {t: getattr(self, f"log_{t}").detach().exp().cpu().tolist() for t in "ABC"}
