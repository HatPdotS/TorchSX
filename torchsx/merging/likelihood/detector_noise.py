"""Learned detector noise: an additive term and a gain dispersion of the counts, per panel or per resolution bin.

The detection likelihood is a deviance-scaled Poisson in the counts,
``log p = (-D/2) * iphi + 1/2 log iphi + logpeak`` with ``iphi = lam / V`` and
``V = lam + sig_b2`` in photon units (``lam = S + bg``: the expected count of the
peak at signal ``S``; ``sig_b2``: the variance of the background estimate).  To the
order the Poisson saddlepoint holds this is a Gaussian of variance ``V`` in the
counts, with the Poisson's skew at low counts.  Everything a detector does beyond
Poisson at its declared gain is absent from it.  This module replaces ``V`` by::

    V = A_p + B_p (lam + sig_b2)          (p = the observation's panel)

- ``A``  additive variance (read noise, pedestal; photons^2);
- ``B``  dispersion of the counting term: a gain that is off (B = true/declared).

The variance depends on the observation's **count** and **panel** only: nothing
per shot, crystal or geometry.  The count is the expected count ``lam`` at each
node of the signal integral, i.e. the detector's variance given the true count --
never the measured ``n``, whose use in a variance biases the fit the way
count-dependent weights bias a weighted mean.  It is condition-blind by
construction.  ``A = 0, B = 1`` is the plain model; the initial values are that up
to a negligible ``A0``.

There is no term in ``lam^2``: that is the multiplicative scatter's variance law,
and the two cannot be told apart.

Per panel, A and B are not hardware constants.  Within one panel the observations
span a narrow range of V, so only ``A + B V`` is identified and the two trade; and
a panel's position fixes its count level and resolution, which a per-panel term
therefore absorbs as well.

``--detector-noise auto`` fits **one global B** (``n_groups = 1``, ``terms = "B"``)
when preparation replaced a wrong declared gain by its calibration, whose estimate
is biased: B corrects the gain on the integrated sums.
``V = B (lam + sig_b2)`` is exactly a global gain error.

With ``edges`` the groups are resolution bins instead of panels (``--detector-noise
resolution``): group ``g`` holds the observations with ``edges[g-1] <= s^2 < edges[g]``,
``s^2 = 1 / (4 d^2)``.  A B per bin lets the counting variance scale with resolution,
which one global scale of the background variance cannot.

Parameters are ``log A, log B`` per group, **projected** into bounds after every
optimizer step (:meth:`project`), not squashed by a saturating function: Adam
normalises a small persistent gradient to a full step, and behind a saturating map
the raw value then walks without limit.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["DetectorNoise"]

#: ``(min, max)`` of each term: A in photons^2, B dimensionless
BOUNDS = {"A": (1e-4, 1e4), "B": (0.02, 100.0)}
INIT = {"A": 1e-2, "B": 1.0}
TERMS = "AB"


class DetectorNoise(nn.Module):
    """``V = A[p] + B[p] (lam + sig_b2)`` per group ``p``: a detector panel or a resolution bin.

    Parameters
    ----------
    n_groups
        Detector panels (observations carry ``panel_id`` in ``[0, n_groups)``), or
        resolution bins when ``edges`` is given.
    terms
        Which of ``"A"``, ``"B"`` are fitted; the other stays at its initial value
        (``"B"`` alone is a gain correction per group).
    edges
        ``n_groups - 1`` ascending bin edges in ``s^2 = 1 / (4 d^2)``.  The groups are
        then resolution bins and ``panel_id`` is not used.
    """

    def __init__(self, n_groups: int = 1, *, terms: str = TERMS, edges=None) -> None:
        super().__init__()
        if n_groups < 1:
            raise ValueError("n_groups must be positive")
        bad = set(terms) - set(TERMS)
        if bad or not terms:
            raise ValueError(f"terms must be a non-empty subset of {TERMS!r}, got {terms!r}")
        self.n_groups, self.terms = int(n_groups), "".join(t for t in TERMS if t in terms)
        for t in TERMS:
            p = nn.Parameter(torch.full((self.n_groups,), math.log(INIT[t])), requires_grad=t in self.terms)
            self.register_parameter(f"log_{t}", p)
        e = None if edges is None else torch.as_tensor(edges, dtype=torch.float32).flatten()
        if e is not None and (e.numel() != self.n_groups - 1 or bool((e[1:] <= e[:-1]).any())):
            raise ValueError(f"edges must be {self.n_groups - 1} ascending values, got {e.tolist()}")
        # configuration, not a fitted value: kept out of the state dict, written by config()
        self.register_buffer("edges", e, persistent=False)

    def config(self) -> dict:
        cfg = {"n_groups": self.n_groups, "terms": self.terms}
        if self.edges is not None:
            cfg["edges"] = self.edges.tolist()
        return cfg

    @classmethod
    def from_config(cls, cfg: dict) -> "DetectorNoise":
        return cls(int(cfg["n_groups"]), terms=str(cfg.get("terms", TERMS)), edges=cfg.get("edges"))

    def groups(self, meta) -> Tensor | None:
        """Group of each observation of ``meta``: its resolution bin, else its panel (``None``: one group)."""
        if self.edges is not None:
            return torch.bucketize(meta.s_sq.to(self.edges.dtype), self.edges, right=True)
        group = getattr(meta, "panel_id", None)
        return None if self.n_groups == 1 else group

    @torch.no_grad()
    def project(self) -> None:
        """Clamp every term into :data:`BOUNDS` (call after each optimizer step)."""
        for t, (lo, hi) in BOUNDS.items():
            getattr(self, f"log_{t}").clamp_(math.log(lo), math.log(hi))

    def terms_obs(self, group: Tensor | None, n: int, device=None) -> tuple[Tensor, Tensor]:
        """``(A, B)`` per observation, differentiable in the fitted terms."""
        if group is None:
            group = torch.zeros(n, dtype=torch.long, device=device or self.log_A.device)
        g = group.long()
        return self.log_A[g].exp(), self.log_B[g].exp()

    def values(self) -> dict:
        """Current ``A, B`` per group, for logging."""
        return {t: getattr(self, f"log_{t}").detach().exp().cpu().tolist() for t in TERMS}
