"""The mean intensity's modulation by translational pseudo-symmetry, in the Wilson prior.

A translation ``t`` relating copies of the structure (translational NCS, a
lattice translocation defect) multiplies the expected intensity by a function of
the phase ``h.t``: with the copies' symmetry images included, of

    c(h) = (1 / |G|) sum_R cos(2 pi h.R t)

(:func:`~torchsx.merging.scale.pseudo_translation.phase_cos`) and of
resolution.  The prior's mean becomes

    Sigma(h) = epsilon(h) g(h) exp(phi(h) @ coef),

the symmetry factor and ``g`` fixed before training, the smooth surface fitted
by the evidence as before (:mod:`torchsx.nn.posterior_target`).

``g`` is fitted to the data's normalised plain means ``E2``
(:func:`~torchsx.merging.scale.pseudo_translation.normalised_intensities`):
``eta = sum_p a_p(s) c_p(h)`` over the translations, the amplitudes ``a_p`` by
least squares of ``E2`` on ``1, c_1 .. c_P`` per resolution shell (each slope
over the shell's intercept, interpolated linearly in ``s^2`` and clamped at the
ends), and ``g = f(eta)`` with ``f`` the non-decreasing fit of ``E2`` on
equal-count bins of ``eta``, scaled to mean 1, constant beyond its ends and
floored.  For a pure pseudo-translation ``f`` is ``1 + eta``.

A factor of ``h`` is degenerate between the prior and the scale model: the
evidence is unchanged when the prior's mean of a family of reflections and the
scale of their observations move by the same factor.  The scale model can hold
one only through the reflection profiles of
:mod:`~torchsx.merging.scale.pseudo_translation`, whose phase-dependent width
and broad part change how bright a family looks where the observed Ewald
offsets do not reach the profile's edges.  Fixing ``g`` from the data pins that
direction, so the profiles are left the shape of the reflections.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from torchsx.merging.scale.pseudo_translation import (
    E2_CLIP,
    epsilon_and_centric,
    normalised_intensities,
    phase_cos,
    point_group_rotations,
)

__all__ = ["PhaseModulation"]


def _interp_weights(x, centres):
    """Linear interpolation of ``x`` between sorted ``centres``, clamped: ``(lo, hi, w_hi)``."""
    if len(centres) == 1:
        z = np.zeros(len(x), dtype=np.int64)
        return z, z, np.zeros(len(x))
    hi = np.clip(np.searchsorted(centres, x), 1, len(centres) - 1)
    lo = hi - 1
    return lo, hi, np.clip((x - centres[lo]) / (centres[hi] - centres[lo]), 0.0, 1.0)


def _isotonic(y, w):
    """Weighted least-squares non-decreasing fit of ``y`` (pool adjacent violators)."""
    blocks = []                                   # [mean, weight, length]
    for yi, wi in zip(y, w):
        blocks.append([yi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, w2, n2 = blocks.pop()
            m1, w1, n1 = blocks.pop()
            blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    return np.concatenate([np.full(n, m) for m, _, n in blocks])


@dataclass
class PhaseModulation:
    """``g(h) = f(eta)``, ``eta = sum_p a_p(s) c_p(h)``: the mean factor of the intensity statistics.

    ``coef[j, p]`` is ``a_p`` in the resolution shell centred at
    ``s2_centres[j]`` (``s2 = 1/d^2``); ``f`` is piecewise linear through
    ``(link_x, link_y)``.  Without translations ``g = 1``.
    """
    translations: np.ndarray                    # (P, 3) fractional
    spacegroup: str | None = None
    s2_centres: np.ndarray = field(default_factory=lambda: np.zeros(1))
    coef: np.ndarray = field(default_factory=lambda: np.zeros((1, 0)))
    link_x: np.ndarray = field(default_factory=lambda: np.zeros(1))
    link_y: np.ndarray = field(default_factory=lambda: np.ones(1))
    floor: float = 0.02

    def cosines(self, hkl) -> np.ndarray:
        """``c_p(h)`` for every translation, ``(N, P)``."""
        hkl = np.asarray(hkl, dtype=np.int64)
        if not len(self.translations):
            return np.zeros((len(hkl), 0))
        rotations = point_group_rotations(self.spacegroup or "P 1")
        return np.stack([phase_cos(hkl, t, rotations) for t in self.translations], axis=1)

    def predictor(self, hkl, s2) -> np.ndarray:
        """``eta(h)``."""
        s2 = np.atleast_1d(np.asarray(s2, dtype=np.float64))
        lo, hi, w = _interp_weights(s2, self.s2_centres)
        a = (1 - w)[:, None] * self.coef[lo] + w[:, None] * self.coef[hi]
        return (a * self.cosines(hkl)).sum(axis=1)

    def log_factor(self, hkl, s2) -> np.ndarray:
        """``log g(h)``."""
        if not len(self.translations):
            return np.zeros(len(np.asarray(hkl)))
        return np.log(np.maximum(np.interp(self.predictor(hkl, s2), self.link_x, self.link_y), self.floor))

    def log_prior_offset(self, hkl, s2) -> np.ndarray:
        """``log epsilon(h) + log g(h)``, the prior's offset from the smooth surface."""
        hkl = np.asarray(hkl, dtype=np.int64)
        eps = epsilon_and_centric(hkl, point_group_rotations(self.spacegroup or "P 1"))[0].clip(min=1)
        return np.log(eps) + self.log_factor(hkl, s2)

    @classmethod
    def fit(cls, hkl, s2, E2, translations, spacegroup: str | None, *, n_shells: int = 6,
            min_per_shell: int = 1000, n_knots: int = 40, min_per_knot: int = 500,
            floor: float = 0.02) -> "PhaseModulation":
        """Fit to ``E2`` (the reflections' intensities over their symmetry factor
        and shell mean); shells and link bins hold at least ``min_per_shell`` /
        ``min_per_knot`` reflections."""
        hkl = np.asarray(hkl, dtype=np.int64)
        s2 = np.asarray(s2, dtype=np.float64)
        y = np.clip(np.asarray(E2, dtype=np.float64), *E2_CLIP)
        t = np.asarray(translations, dtype=np.float64).reshape(-1, 3)
        mod = cls(translations=t, spacegroup=spacegroup, floor=float(floor))
        if not len(t):
            return mod
        n_shells = max(1, min(n_shells, len(hkl) // max(1, min_per_shell)))
        shells = [ix for ix in np.array_split(np.argsort(s2, kind="stable"), n_shells) if len(ix)]
        mod.s2_centres = np.array([np.median(s2[ix]) for ix in shells])
        C = mod.cosines(hkl)
        mod.coef = np.zeros((len(shells), len(t)))
        for j, ix in enumerate(shells):
            X = np.concatenate([np.ones((len(ix), 1)), C[ix]], axis=1)
            beta = np.linalg.lstsq(X, y[ix], rcond=None)[0]
            mod.coef[j] = beta[1:] / max(beta[0], 0.1)
        eta = mod.predictor(hkl, s2)
        n_bins = max(2, min(n_knots, len(hkl) // max(1, min_per_knot)))
        bins = np.array_split(np.argsort(eta, kind="stable"), n_bins)
        mod.link_x = np.array([eta[ix].mean() for ix in bins])
        w = np.array([len(ix) for ix in bins], dtype=np.float64)
        mod.link_y = _isotonic(np.array([y[ix].mean() for ix in bins]) / y.mean(), w)
        return mod

    @classmethod
    def from_prep(cls, prep, translations, spacegroup: str | None, **kw) -> "PhaseModulation":
        """Fit to the prepared reflections' plain means, the ones with observations."""
        hkl, s2, E2, seen = normalised_intensities(prep, spacegroup)
        return cls.fit(hkl[seen], s2[seen], E2[seen], translations, spacegroup, **kw)

    def summary(self) -> list[dict]:
        """Per translation, its amplitude per resolution shell."""
        return [{"t": np.round(t, 4).tolist(), "amplitude": np.round(self.coef[:, p], 3).tolist(),
                 "d": np.round(1.0 / np.sqrt(self.s2_centres), 2).tolist()}
                for p, t in enumerate(self.translations)]

    def definition(self) -> dict:
        return {"translations": np.asarray(self.translations).tolist(), "spacegroup": self.spacegroup,
                "s2_centres": np.asarray(self.s2_centres).tolist(), "coef": np.asarray(self.coef).tolist(),
                "link_x": np.asarray(self.link_x).tolist(), "link_y": np.asarray(self.link_y).tolist(),
                "floor": float(self.floor)}

    @classmethod
    def from_definition(cls, d: dict) -> "PhaseModulation":
        t = np.asarray(d["translations"], dtype=np.float64).reshape(-1, 3)
        return cls(translations=t, spacegroup=d.get("spacegroup"),
                   s2_centres=np.asarray(d["s2_centres"], dtype=np.float64),
                   coef=np.asarray(d["coef"], dtype=np.float64).reshape(-1, len(t)),
                   link_x=np.asarray(d["link_x"], dtype=np.float64),
                   link_y=np.asarray(d["link_y"], dtype=np.float64), floor=float(d["floor"]))
