"""Reflection profiles that depend on the structure's translational pseudo-symmetry.

A translation ``t`` relating parts of the crystal -- translational NCS, a
lattice translocation defect, stacking faults -- makes how a reflection is
recorded depend on its phase ``h.t``.  Reflections with integer ``h.t`` stay
sharp; the others spread along the fault direction, and the spread part may
be broad and shallow while the rest stays sharp.  One partiality profile
shared by every reflection then fits none of these families.

**Detection.**  :func:`find_pseudo_translations` finds the translations in the
data: the Patterson function of ``E2 - 1`` (each reflection's plain mean over
its symmetry factor and its resolution shell's mean) over the full sphere, its
significant off-origin peaks refined to the translation that best explains the
intensities, harmonics ``n t`` of a stronger translation placed exactly at
``n t``.  Each peak carries the amplitude of the intensity modulation it
implies.  :func:`select_translations` keeps a translation that dominates the
Patterson function -- at least :data:`MIN_AMPLITUDE`, and :data:`DOMINANCE`
times any other translation that is not one of its harmonics -- together with
its harmonics above :data:`MIN_AMPLITUDE`.  A copy of the whole structure gives
such a peak; the interatomic vectors of a small structure give several peaks
of similar height, and nothing is kept.

**Profile.**  With ``c_j(h)`` the cosine of ``2 pi h.R t_j`` averaged over the
point-group rotations ``R`` and ``z`` the standardised resolution, each
reflection gets a core width ``w(h)`` (:class:`PhaseWidth`) and a broad
component of relative area ``rho(h)`` and width ``W(h)`` times the core's
(:class:`PhaseBroad`), all linear in

    c_j,  c_j z,  c_j^2,  c_j^2 z      (per translation j)

and bounded.  The scale model reads the observation's Ewald offset in units of
``w(h)`` for the core and of ``w(h) W(h)`` for the broad part (a second pass of
the per-observation branch of the encoder, sharing the crystal context), and

    K = [K_core(e / w) / w + rho K_core(e / (w W)) / (w W)] / (1 + rho),

a profile with the area of the shared one.  The merged intensity of a
reflection is therefore its total recorded intensity, core and broad part
together.  The coefficients start at the shared profile with a small broad
part and are fitted with the network.

Rotations act on Miller indices as rows, ``h' = h @ R``.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["DOMINANCE", "MIN_AMPLITUDE", "PhaseBroad", "PhaseWidth", "find_pseudo_translations",
           "find_translations", "select_translations"]

#: Translations whose modulation amplitude is below this are not used.
MIN_AMPLITUDE: float = 0.1
#: The used translation's amplitude over that of any other non-harmonic translation.
DOMINANCE: float = 3.0

E2_CLIP = (-5.0, 50.0)


# ── detection ──────────────────────────────────────────────────────────────

def metric(cell):
    """Reciprocal basis rows ``B`` (``s = h @ B``, 1/A) and real-space rows ``A`` (A)."""
    from torchsx.merging.priors.reciprocal_basis import _reciprocal_basis_matrix
    B = _reciprocal_basis_matrix(torch.as_tensor(np.asarray(cell, dtype=np.float64))).numpy()
    return B, np.linalg.inv(B).T


def point_group_rotations(spacegroup) -> list[np.ndarray]:
    """The distinct rotations of a space group, acting on Miller indices as rows."""
    from torchref.symmetry import SpaceGroup
    m = SpaceGroup(spacegroup).matrices.detach().cpu().round().to(torch.int64).numpy()
    return [np.array(k, dtype=np.int64).reshape(3, 3) for k in sorted({tuple(r.ravel()) for r in m})]


def _key(hkl):
    h = np.asarray(hkl, dtype=np.int64) + 1024
    return (h[:, 0] << 22) | (h[:, 1] << 11) | h[:, 2]


def expand(hkl, rotations=()):
    """Every distinct symmetry image and Friedel mate of each row; the source row of each."""
    rotations = list(rotations) or [np.eye(3, dtype=np.int64)]
    images = np.concatenate([hkl @ R for R in rotations] + [-(hkl @ R) for R in rotations])
    owner = np.tile(np.arange(len(hkl)), 2 * len(rotations))
    _, first = np.unique(_key(images), return_index=True)
    return images[first], owner[first]


def epsilon_and_centric(hkl, rotations):
    """Symmetry factor ``#{R : hR = h}`` and centricity (``hR = -h`` for some R)."""
    eps = np.zeros(len(hkl), dtype=np.int64)
    centric = np.zeros(len(hkl), dtype=bool)
    for R in rotations:
        img = hkl @ R
        eps += (img == hkl).all(axis=1)
        centric |= (img == -hkl).all(axis=1)
    return eps, centric


def phase_cos(hkl, t, rotations=()):
    """``c(h)``: the cosine of ``2 pi h.R t`` averaged over the rotations."""
    rotations = list(rotations) or [np.eye(3)]
    t = np.asarray(t, dtype=np.float64)
    return np.mean([np.cos(2 * np.pi * hkl @ (R @ t)) for R in rotations], axis=0)


def patterson(hkl, values, oversample=3):
    """Patterson function of ``values`` on the listed indices (already expanded),
    on a grid that places a peak to ``1 / (2 oversample max|h|)``; and the grid shape."""
    hmax = np.abs(hkl).max(axis=0)
    shape = tuple(int(2 * oversample * m + 2) for m in hmax)
    grid = np.zeros(shape)
    idx = hkl % np.array(shape)
    np.add.at(grid, (idx[:, 0], idx[:, 1], idx[:, 2]), values)
    return np.real(np.fft.ifftn(grid)) * grid.size, shape


def _local_maxima(P):
    """``P`` equals the maximum of its 3x3x3 periodic neighbourhood."""
    m = P.copy()
    for axis in range(3):
        m = np.maximum(m, np.maximum(np.roll(m, 1, axis=axis), np.roll(m, -1, axis=axis)))
    return P == m


def wrap(u):
    """``u`` mod 1 into ``[-1/2, 1/2)``."""
    return np.mod(np.asarray(u, dtype=np.float64) + 0.5, 1.0) - 0.5


def canonical_shift(u):
    """``u`` or ``-u`` mod 1, whichever has its first non-zero coordinate in (0, 1/2]."""
    u = wrap(u)
    for x in u:
        if abs(x) > 1e-6:
            return u if x > 0 else -u
    return u


def refine_translation(hkl, v, t0, n_iter: int = 30):
    """The translation near ``t0`` maximising ``f(t) = sum v cos(2 pi h.t)`` over expanded indices
    (Newton steps on ``f`` with the step halved until ``f`` does not fall)."""
    h = np.asarray(hkl, dtype=np.float64)
    t = np.asarray(t0, dtype=np.float64)
    f = lambda x: float(np.dot(v, np.cos(2 * np.pi * h @ x)))
    best = f(t)
    for _ in range(n_iter):
        ph = 2 * np.pi * h @ t
        g = -2 * np.pi * (h * (v * np.sin(ph))[:, None]).sum(0)
        H = -4 * np.pi ** 2 * (h * (v * np.cos(ph))[:, None]).T @ h
        try:
            step = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)) or np.dot(step, g) <= 0:
            break
        lam = 1.0
        while lam > 1e-3:
            cand = t + lam * step
            fc = f(cand)
            if fc >= best:
                break
            lam /= 2
        if lam <= 1e-3 or np.max(np.abs(lam * step)) < 1e-7:
            break
        t, best = cand, fc
    return wrap(t)


def _distance(u, w, A, rotations):
    """Smallest Cartesian distance between ``u`` and the images ``+-R w`` modulo the lattice."""
    rotations = list(rotations) or [np.eye(3)]
    return min(np.linalg.norm(wrap(u - s * (R @ w)) @ A) for R in rotations for s in (1, -1))


def find_translations(hkl, cell, E2, rotations=(), *, max_peaks=6, min_sigma=8.0, min_amplitude=0.02,
                      min_distance=8.0, min_separation=6.0, max_harmonic=4, harmonic_tolerance=0.25):
    """Significant off-origin Patterson peaks of ``E2 - 1``, refined.

    ``hkl`` and ``E2`` describe the unique reflections; the Patterson sum runs
    over their images under ``rotations`` and Friedel.  Each peak carries the
    modulation amplitude it implies, ``P(u) / sum c_u(h)^2`` over the expanded
    set, its significance against the Patterson's robust spread, and
    ``harmonic_of`` / ``order`` when it lies within ``harmonic_tolerance``
    times the resolution limit of ``order`` times an earlier peak, and then sits
    exactly there.
    """
    full, owner = expand(hkl, rotations)
    v = np.clip(E2, *E2_CLIP)[owner] - 1.0
    P, shape = patterson(full, v)
    noise = 1.4826 * np.median(np.abs(P - np.median(P)))
    is_peak = _local_maxima(P) & (P >= min_sigma * noise)
    cand = np.argwhere(is_peak)
    heights = P[tuple(cand.T)]
    B, A = metric(cell)
    d_min = 1.0 / np.linalg.norm(full @ B, axis=1).max()
    peaks = []
    for c, ht in sorted(zip(cand, heights), key=lambda x: -x[1]):
        u = canonical_shift(c / np.array(shape))
        if np.linalg.norm(u @ A) < min_distance:
            continue
        amplitude = ht / np.sum(phase_cos(full, u, rotations) ** 2)
        if amplitude < min_amplitude:
            continue
        if any(_distance(u, np.array(p['t']), A, rotations) < min_separation for p in peaks):
            continue
        t = canonical_shift(refine_translation(full, v, u))
        if any(_distance(t, np.array(p['t']), A, rotations) < min_separation for p in peaks):
            continue
        rec = {'t': t.tolist(), 'amplitude': float(amplitude), 'significance': float(ht / noise),
               'distance_A': float(np.linalg.norm(t @ A)), 'harmonic_of': None, 'order': 1}
        for j, p in enumerate(peaks):
            if p['harmonic_of'] is not None:
                continue
            for n in range(2, max_harmonic + 1):
                if _distance(t, n * np.array(p['t']), A, rotations) < harmonic_tolerance * d_min:
                    # the harmonic is the same phase: n t of the better-determined fundamental
                    nt = canonical_shift(n * np.array(p['t']))
                    rec.update(t=nt.tolist(), distance_A=float(np.linalg.norm(nt @ A)), harmonic_of=j, order=n)
                    break
            if rec['harmonic_of'] is not None:
                break
        peaks.append(rec)
        if len(peaks) == max_peaks:
            break
    return peaks


def _reflection_means(prep) -> tuple[np.ndarray, np.ndarray]:
    """Each reflection's plain mean of the observed intensities (unfloored) and observation count."""
    from torchsx.merging.store import PreparedStore
    if isinstance(prep, PreparedStore):
        num = prep.tables["sum_I"].double().cpu().numpy()
        cnt = prep.tables["count"].double().cpu().numpy()
    else:
        idx = prep.obs_info.hkl_idx.long().cpu().numpy()
        num = np.bincount(idx, prep.I_obs.double().cpu().numpy(), minlength=prep.n_valid)
        cnt = np.bincount(idx, minlength=prep.n_valid).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, num / np.maximum(cnt, 1.0), np.nan), cnt


def find_pseudo_translations(prep, spacegroup: str | None, *, n_shells: int = 20, **kw) -> list[dict]:
    """Translations of the structure in the Patterson function of ``E2 - 1``.

    ``E2`` is each reflection's plain mean over its symmetry factor and the mean
    of that ratio in its resolution shell (equal-count shells in ``1/d^2``).
    Keyword arguments go to :func:`find_translations`.  Every peak found is
    returned; :data:`MIN_AMPLITUDE` decides which are used.
    """
    hkl = prep.hkl_valid.long().cpu().numpy()
    s2 = (1.0 / prep.d_spacing.double() ** 2).cpu().numpy()
    rotations = point_group_rotations(spacegroup or "P 1")
    eps = epsilon_and_centric(hkl, rotations)[0].clip(min=1)
    mean, cnt = _reflection_means(prep)
    seen = np.isfinite(mean) & (cnt > 0)
    I = np.where(seen, mean, 0.0) / eps
    edges = np.quantile(s2[seen], np.linspace(0.0, 1.0, n_shells + 1))
    shell = np.clip(np.searchsorted(edges, s2, side="right") - 1, 0, n_shells - 1)
    shell_mean = np.array([I[seen & (shell == i)].mean() if (seen & (shell == i)).any() else np.nan
                           for i in range(n_shells)])
    E2 = np.where(seen, I / shell_mean[shell], 1.0)
    cell = prep.median_cell.double().cpu().numpy()
    return find_translations(hkl, cell, E2, rotations, **kw)


def select_translations(peaks: list[dict]) -> tuple[list[dict], str]:
    """The translations to use from :func:`find_pseudo_translations`' peaks, and why.

    The strongest peak that is not a harmonic is used when its amplitude is at
    least :data:`MIN_AMPLITUDE` and :data:`DOMINANCE` times that of every other
    non-harmonic peak; with it, its harmonics of amplitude at least
    :data:`MIN_AMPLITUDE`.
    """
    fundamentals = sorted((i for i, p in enumerate(peaks) if p["harmonic_of"] is None),
                          key=lambda i: -peaks[i]["amplitude"])
    if not fundamentals or peaks[fundamentals[0]]["amplitude"] < MIN_AMPLITUDE:
        return [], f"no translation with amplitude {MIN_AMPLITUDE} or more"
    first = fundamentals[0]
    others = [peaks[i]["amplitude"] for i in fundamentals[1:]]
    if others and peaks[first]["amplitude"] < DOMINANCE * max(others):
        return [], (f"no dominant translation (amplitudes {peaks[first]['amplitude']:.3f} and "
                    f"{max(others):.3f}: several comparable peaks, as from the interatomic vectors of a "
                    f"small structure)")
    used = [peaks[first]] + [p for p in peaks if p["harmonic_of"] == first and p["amplitude"] >= MIN_AMPLITUDE]
    return used, f"{len(used)} translation(s): one dominant translation and its harmonics"


# ── profile ────────────────────────────────────────────────────────────────

def _phase_table(hkl, s2, translations, spacegroup, s_centre, s_scale):
    """``(table, translations, s_centre, s_scale)``: per translation the columns ``c, c z, c^2, c^2 z``."""
    hkl = np.asarray(hkl, dtype=np.int64)
    t = np.asarray(translations, dtype=np.float64).reshape(-1, 3)
    rotations = point_group_rotations(spacegroup or "P 1")
    s = np.sqrt(np.asarray(s2, dtype=np.float64)) / 2.0
    s_centre = float(np.mean(s)) if s_centre is None else float(s_centre)
    s_scale = float(np.std(s) or 1.0) if s_scale is None else float(s_scale)
    z = (s - s_centre) / s_scale
    cols = []
    for tj in t:
        c = phase_cos(hkl, tj, rotations)
        cols += [c, c * z, c * c, c * c * z]
    X = np.stack(cols, axis=1) if cols else np.zeros((len(hkl), 0))
    return torch.as_tensor(X, dtype=torch.float32), t.tolist(), s_centre, s_scale


class PhaseWidth(nn.Module):
    """``log w(h)``, the core width of each reflection relative to the shared profile.

    ``hkl`` and ``s2`` (``1/d^2``) describe the reflections the scale model
    indexes by ``hkl_idx``; ``translations`` are fractional ``t_j``,
    ``spacegroup`` sets the rotations the cosines are averaged over.  The
    resolution is standardised by ``s_centre`` / ``s_scale`` (fitted on
    ``s2`` when not given), so a model restored on another reflection set
    evaluates the same function.
    """

    #: Largest ``|log w|``: the profile is stretched or narrowed at most e^BOUND-fold.
    BOUND: float = 3.0

    def __init__(self, hkl, s2, translations, spacegroup: str | None, *,
                 s_centre: float | None = None, s_scale: float | None = None) -> None:
        super().__init__()
        X, self.translations, self.s_centre, self.s_scale = _phase_table(
            hkl, s2, translations, spacegroup, s_centre, s_scale)
        self.spacegroup = spacegroup
        self.register_buffer("table", X, persistent=False)
        self.coef = nn.Parameter(torch.zeros(X.shape[1]))

    @classmethod
    def from_prep(cls, prep, translations, spacegroup: str | None, **kw) -> "PhaseWidth":
        return cls(prep.hkl_valid.long().cpu().numpy(), (1.0 / prep.d_spacing.double() ** 2).cpu().numpy(),
                   translations, spacegroup, **kw)

    @classmethod
    def from_definition(cls, d: dict, prep) -> "PhaseWidth":
        return cls.from_prep(prep, d["translations"], d.get("spacegroup"),
                             s_centre=d["s_centre"], s_scale=d["s_scale"])

    def definition(self) -> dict:
        return {"translations": self.translations, "spacegroup": self.spacegroup,
                "s_centre": self.s_centre, "s_scale": self.s_scale,
                "coef": self.coef.detach().cpu().tolist(), "bound": self.BOUND}

    def log_width(self, hkl_idx: Tensor) -> Tensor:
        """``log w`` of the reflections ``hkl_idx`` (on the module's device)."""
        if self.table.shape[1] == 0:
            return torch.zeros(hkl_idx.shape, device=self.coef.device)
        eta = self.table.index_select(0, hkl_idx.to(self.table.device).long()) @ self.coef
        return self.BOUND * torch.tanh(eta / self.BOUND)


class PhaseBroad(nn.Module):
    """The broad component of each reflection's profile: relative area ``rho(h)`` and width ``W(h)``.

    ``log rho = RHO_CENTRE + RHO_HALF tanh(eta_rho / RHO_HALF)`` and
    ``log W = log W_MIN + W_SPAN sigmoid(eta_W)``, with each ``eta`` an
    intercept plus the phase table times coefficients.  The intercepts start
    at ``rho = exp(RHO_CENTRE)`` and ``W = W_INIT``, the coefficients at zero;
    ``W >= W_MIN`` keeps the two components distinct.
    """

    RHO_CENTRE: float = -2.5
    RHO_HALF: float = 5.0
    W_MIN: float = 2.0
    W_SPAN: float = 4.0
    W_INIT: float = 5.0

    def __init__(self, hkl, s2, translations, spacegroup: str | None, *,
                 s_centre: float | None = None, s_scale: float | None = None) -> None:
        super().__init__()
        X, self.translations, self.s_centre, self.s_scale = _phase_table(
            hkl, s2, translations, spacegroup, s_centre, s_scale)
        self.spacegroup = spacegroup
        self.register_buffer("table", X, persistent=False)
        p = X.shape[1]
        frac = np.log(self.W_INIT / self.W_MIN) / self.W_SPAN
        self.rho_coef = nn.Parameter(torch.zeros(p + 1))
        self.width_coef = nn.Parameter(torch.zeros(p + 1))
        with torch.no_grad():
            self.width_coef[0] = float(np.log(frac / (1.0 - frac)))

    @classmethod
    def from_prep(cls, prep, translations, spacegroup: str | None, **kw) -> "PhaseBroad":
        return cls(prep.hkl_valid.long().cpu().numpy(), (1.0 / prep.d_spacing.double() ** 2).cpu().numpy(),
                   translations, spacegroup, **kw)

    @classmethod
    def from_definition(cls, d: dict, prep) -> "PhaseBroad":
        return cls.from_prep(prep, d["translations"], d.get("spacegroup"),
                             s_centre=d["s_centre"], s_scale=d["s_scale"])

    def definition(self) -> dict:
        return {"translations": self.translations, "spacegroup": self.spacegroup,
                "s_centre": self.s_centre, "s_scale": self.s_scale,
                "rho_coef": self.rho_coef.detach().cpu().tolist(),
                "width_coef": self.width_coef.detach().cpu().tolist()}

    def _eta(self, coef: Tensor, hkl_idx: Tensor) -> Tensor:
        eta = coef[0].expand(hkl_idx.shape)
        if self.table.shape[1]:
            eta = eta + self.table.index_select(0, hkl_idx.to(self.table.device).long()) @ coef[1:]
        return eta

    def log_fraction(self, hkl_idx: Tensor) -> Tensor:
        """``log rho``: the broad part's area relative to the core's."""
        eta = self._eta(self.rho_coef, hkl_idx)
        return self.RHO_CENTRE + self.RHO_HALF * torch.tanh(eta / self.RHO_HALF)

    def log_width(self, hkl_idx: Tensor) -> Tensor:
        """``log W``: the broad part's width relative to the core's (``>= log W_MIN``)."""
        return float(np.log(self.W_MIN)) + self.W_SPAN * torch.sigmoid(self._eta(self.width_coef, hkl_idx))
