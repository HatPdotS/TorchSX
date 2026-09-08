"""Scale calibration of the posterior merge onto the pooled mean.

The posterior readout is the most precise merge the trained scale model
gives, but the shape likelihood it comes from estimates the location of a
skewed observation distribution rather than its mean, and refinement compares
against the mean.  The pooled readout (``sum(n - bg) / sum K`` on the same
trained K) is a mean estimate with no shape assumption.  The offset between
the two is a smooth function of resolution and intensity.

This module fits that offset as one multiplicative surface

    I_pooled  ~  f(s^2, log I_post) * I_post,     f = Phi(s^2, log I_post) beta

with ``Phi`` a tensor-product Chebyshev basis.  ``f`` is linear in ``beta``,
so the fit is a single weighted least-squares solve with weights from the
pooled sigma plus a relative floor and a weak ridge toward ``f == 1``.
Applied to the posterior it keeps the posterior's precision on the pooled
scale and dynamic range.  It is fitted once per merge run on the full merges
of every split jointly and applied to every split and half-set.  Nothing here
depends on the scatter model: any two readouts of the same reflections can be
mapped.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

_EPS = 1e-6


def _as_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


@dataclass
class ChebyshevScaleSurface:
    """Multiplicative correction ``f(s^2, log I)`` on a tensor Chebyshev basis.

    Parameters
    ----------
    nx, ny
        Chebyshev degrees in ``s^2 = 1 / (4 d^2)`` and in ``log I``; the basis
        has ``(nx + 1) (ny + 1)`` coefficients.
    ridge
        Ridge strength relative to ``trace(A^T W A) / p``, pulling ``f`` toward 1.
    rel_floor
        Relative floor added in quadrature to the pooled sigma in the weights,
        so no reflection with a tiny stated sigma dominates the fit.
    f_min, f_max
        Clamps on the applied factor.
    """

    nx: int = 9
    ny: int = 9
    ridge: float = 1e-4
    rel_floor: float = 0.02
    f_min: float = 0.05
    f_max: float = 20.0
    s2_range: tuple[float, float] | None = None
    logI_range: tuple[float, float] | None = None
    beta: list[float] | None = None
    fit_info: dict = field(default_factory=dict)

    # ── basis ──────────────────────────────────────────────────────────────

    @property
    def n_coef(self) -> int:
        return (self.nx + 1) * (self.ny + 1)

    @property
    def is_fitted(self) -> bool:
        return self.beta is not None

    def _scaled(self, v, rng) -> np.ndarray:
        lo, hi = rng
        return np.clip(2.0 * (v - lo) / max(hi - lo, 1e-12) - 1.0, -1.0, 1.0)

    def _phi(self, d, I) -> np.ndarray:
        d, I = _as_np(d), _as_np(I)
        s2 = 1.0 / (4.0 * d * d)
        y = np.log(np.clip(I, _EPS, None))
        tx = np.polynomial.chebyshev.chebvander(self._scaled(s2, self.s2_range), self.nx)
        ty = np.polynomial.chebyshev.chebvander(self._scaled(y, self.logI_range), self.ny)
        return (tx[:, :, None] * ty[:, None, :]).reshape(len(d), -1)

    # ── fit / apply ────────────────────────────────────────────────────────

    def fit(self, I_post, I_pooled, sd_pooled, d) -> dict:
        """Weighted least squares of ``I_pooled`` on ``I_post * Phi``.

        Rows with non-finite values or ``I_post <= 0`` are dropped.  Returns
        the fit statistics (also kept on ``fit_info``).
        """
        P, Q, S, D = (_as_np(v) for v in (I_post, I_pooled, sd_pooled, d))
        m = np.isfinite(P) & np.isfinite(Q) & np.isfinite(S) & np.isfinite(D) & (P > 0) & (D > 0)
        if m.sum() < 2 * self.n_coef:
            raise ValueError(
                f"scale calibration needs at least {2 * self.n_coef} usable "
                f"reflections, got {int(m.sum())}"
            )
        P, Q, S, D = P[m], Q[m], S[m], D[m]
        s2 = 1.0 / (4.0 * D * D)
        self.s2_range = (float(s2.min()), float(s2.max()))
        lo, hi = np.quantile(np.log(np.clip(P, _EPS, None)), [0.002, 0.998])
        self.logI_range = (float(lo), float(hi))

        Phi = self._phi(D, P)
        A = Phi * P[:, None]
        w = 1.0 / (S ** 2 + (self.rel_floor * np.abs(Q)) ** 2)
        Aw = A * w[:, None]
        AtA = Aw.T @ A
        Atb = Aw.T @ Q
        lam = self.ridge * np.trace(AtA) / A.shape[1]
        prior = np.zeros(A.shape[1]); prior[0] = 1.0          # T0*T0 == 1 -> f == 1
        beta = np.linalg.solve(AtA + lam * np.eye(A.shape[1]), Atb + lam * prior)
        self.beta = [float(b) for b in beta]
        r = (Q - A @ beta) * np.sqrt(w)
        f = self.factor(D, P)
        self.fit_info = {
            "n_fit": int(m.sum()),
            "n_coef": int(A.shape[1]),
            "chi2_per_dof": float((r ** 2).sum() / max(m.sum() - A.shape[1], 1)),
            "factor_min": float(f.min()),
            "factor_max": float(f.max()),
            "mean_ratio_raw": float(P.mean() / Q.mean()),
            "mean_ratio_corrected": float((f * P).mean() / Q.mean()),
        }
        return dict(self.fit_info)

    def factor(self, d, I) -> np.ndarray:
        """The multiplicative factor on intensities, clamped to ``[f_min, f_max]``."""
        if not self.is_fitted:
            raise RuntimeError("surface is not fitted")
        f = self._phi(d, I) @ np.asarray(self.beta)
        return np.clip(f, self.f_min, self.f_max)

    def apply(self, d, I, I_sd=None, F=None, F_sd=None) -> dict:
        """Scale a merge: ``I f``, ``I_sd f``, ``F sqrt(f)``, ``F_sd sqrt(f)``.

        The posterior ``F`` is a posterior mean of ``sqrt(I0)``, so a factor
        on intensities is its square root on amplitudes.  Returns a dict with
        the scaled arrays that were given plus ``factor``.
        """
        f = self.factor(d, I)
        out = {"factor": f, "I": _as_np(I) * f}
        if I_sd is not None:
            out["I_sd"] = _as_np(I_sd) * f
        if F is not None:
            out["F"] = _as_np(F) * np.sqrt(f)
        if F_sd is not None:
            out["F_sd"] = _as_np(F_sd) * np.sqrt(f)
        return out

    # ── diagnostics / persistence ──────────────────────────────────────────

    def shell_check(self, d, I_post, I_pooled, edges) -> list[dict]:
        """Per-shell mean ratio posterior/pooled, raw and corrected, plus the
        weakest octile of each shell -- the transfer check for a split the
        surface was or was not fitted on."""
        D, P, Q = (_as_np(v) for v in (d, I_post, I_pooled))
        f = self.factor(D, P)
        rows = []
        for hi, lo in zip(edges[:-1], edges[1:]):
            s = (D <= hi) & (D > lo) & np.isfinite(P) & np.isfinite(Q)
            if s.sum() < 8:
                continue
            w = s & (P <= np.quantile(P[s], 0.125))
            rows.append({
                "d_hi": float(hi), "d_lo": float(lo), "n": int(s.sum()),
                "raw": float(P[s].mean() / Q[s].mean()),
                "corrected": float((f * P)[s].mean() / Q[s].mean()),
                "weak_raw": float(P[w].mean() / Q[w].mean()) if w.sum() else None,
                "weak_corrected": float((f * P)[w].mean() / Q[w].mean()) if w.sum() else None,
            })
        return rows

    def to_dict(self) -> dict:
        d = asdict(self)
        d["basis"] = "tensor Chebyshev in (s^2, log I); f = Phi beta, I_pooled ~ f I_post"
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ChebyshevScaleSurface":
        d = dict(d); d.pop("basis", None)
        obj = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if obj.s2_range is not None:
            obj.s2_range = tuple(obj.s2_range)
        if obj.logI_range is not None:
            obj.logI_range = tuple(obj.logI_range)
        return obj


def default_shell_edges(d, n: int = 8) -> list[float]:
    """Equal-count resolution shell edges over ``d`` (descending), for the transfer check."""
    D = _as_np(d); D = D[np.isfinite(D)]
    q = np.quantile(D, np.linspace(1.0, 0.0, n + 1))
    q[0] = max(q[0], D.max()) * 1.001; q[-1] = min(q[-1], D.min()) * 0.999
    return [float(v) for v in q]
