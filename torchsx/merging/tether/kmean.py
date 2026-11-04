"""The arithmetic mean of the scales, held free of a trend along given columns.

The likelihood sees an observation only through ``K I0``, so
``K -> K g(h), I0 -> I0 / g(h)`` is an exact symmetry for every smooth function
``g`` of the reflection, and something has to fix it.  In rotation data the
full intensity is the sum of a reflection's partials.  The observations of a
reflection in still images sample the excitation error evenly over a window
that covers the whole rocking curve, so the **arithmetic** mean of ``K`` over
them is that integral: the sum of the spots.  Pinned at every resolution and in
every direction, it leaves partiality in ``K`` and the fall-off of the
integrated intensities in ``I0``.

The statistic is the log-link fit

    K_j  ~  exp(c_k(j) + A_j @ b),

a constant per crystal and a trend along columns ``A`` -- the smooth reciprocal
basis ``phi`` of the observation's reflection (without its constant column,
which the crystal constants carry) and, optionally, columns given per
observation.  ``b = 0`` is the constraint: in every crystal and every column
the mean of ``K exp(-c - A b)`` is one.  Profiled over the crystal constants
the objective in ``b`` is convex:

    f(b) = sum_k W_k c_k(b) + sum_j w_j A_j @ b,
    c_k(b) = log of crystal k's w-weighted mean of exp(log K_j - A_j @ b).

:func:`integral_step` is the training statistic: one Newton step from
``b = 0`` in closed form.  It vanishes exactly where the solution does, with the
same zero gradient, and agrees with it to first order near the constraint, so a
penalty on it holds the same gauge without a solve inside the update.
:func:`integral_mode` solves the fit exactly, for a readout over a whole merge:
damped Newton from the least-squares fit of ``log K`` (the geometric-mean
gauge), so a start drifted many nats along the gauge converges, then one step in
the graph at the solution, which carries the exact implicit derivative.

Observations more than ``runaway`` nats above the median of their crystal are
left out of the fit.  The arithmetic mean itself cannot be the reference: one
runaway carries its crystal's mean with it and sits only about ``log n`` above
it.  The test is invariant under a smooth shift and under crystal constants.

The basis is given per reflection (``n_hkl x p``) and indexed by each
observation's reflection, and every sum over observations is formed per
reflection or per crystal first: the working memory is a few scalars per
observation.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "integral_mode",
    "integral_step",
    "mode_mean_square",
    "smooth_columns",
    "step_mean_square",
]

#: Observations per block when a sum needs the basis row of every observation.
_BLOCK = 1 << 20


def smooth_columns(phi: Tensor, tol: float = 1e-9) -> Tensor:
    """The columns of a per-reflection basis that vary over the reflections.

    A constant column is the same for every observation of a crystal, so the
    crystal constants carry it and the statistic leaves it out.
    """
    return phi[:, phi.std(dim=0) > tol]


def _crystal_log_mean(e: Tensor, w: Tensor, cid: Tensor, n_cry: int) -> Tensor:
    """log of each crystal's ``w``-weighted mean of ``exp(e)``.

    The shift ``m`` only keeps the exponentials finite; neither the value nor
    its derivative depends on it, so it is taken outside the graph.
    """
    with torch.no_grad():
        m = torch.full((n_cry,), -torch.inf, device=e.device, dtype=e.dtype).scatter_reduce(
            0, cid, torch.where(w > 0, e, torch.full_like(e, -torch.inf)), "amax")
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    s = torch.zeros(n_cry, device=e.device, dtype=e.dtype).index_add(0, cid, w * torch.exp(e - m[cid]))
    n = torch.zeros(n_cry, device=e.device, dtype=e.dtype).index_add(0, cid, w)
    return m + torch.log(s.clamp(min=1e-300) / n.clamp(min=1.0))


def _crystal_median(e: Tensor, cid: Tensor, n_cry: int) -> Tensor:
    """Each crystal's (lower) median of ``e``; 0 for a crystal without observations."""
    order = torch.argsort(e, stable=True)
    order = order[torch.argsort(cid[order], stable=True)]
    counts = torch.bincount(cid, minlength=n_cry)
    mid = torch.cumsum(counts, 0) - counts + (counts - 1).clamp(min=0) // 2
    has = counts > 0
    med = torch.zeros(n_cry, device=e.device, dtype=e.dtype)
    med[has] = e[order[mid[has]]]
    return med


def psd_solve(H: Tensor, g: Tensor, rcond: float = 1e-10) -> Tensor:
    """``H^+ g`` for a positive semi-definite ``H`` and a vector or matrix ``g``:
    exact in every direction the data determine, zero in a direction they do
    not (an eigenvalue below ``rcond`` times the largest).  Differentiable in
    ``g``."""
    if H.numel() == 0:
        return torch.zeros_like(g)
    lam, V = torch.linalg.eigh(H)
    inv = torch.where(lam > rcond * lam.max().clamp(min=1e-300), 1.0 / lam, torch.zeros_like(lam))
    return V @ ((inv[:, None] if g.dim() == 2 else inv) * (V.T @ g))


class _Fit:
    """One set of observations' fit: the basis, their reflections and crystals, the weights."""

    def __init__(self, phi: Tensor, h: Tensor, cid: Tensor, n_cry: int, w: Tensor):
        self.phi, self.h, self.cid, self.n_cry, self.w = phi, h, cid, n_cry, w
        n_hkl = phi.shape[0]
        self.W = torch.zeros(n_cry, device=w.device, dtype=w.dtype).index_add_(0, cid, w)
        self.w_h = torch.zeros(n_hkl, device=w.device, dtype=w.dtype).index_add_(0, h, w)
        self.wphi = phi.T @ self.w_h

    def per_hkl(self, v: Tensor) -> Tensor:
        return torch.zeros(self.phi.shape[0], device=v.device, dtype=v.dtype).index_add(0, self.h, v)

    def residual(self, lk: Tensor, b: Tensor) -> Tensor:
        """``log K - phi b`` per observation."""
        return lk - (self.phi @ b)[self.h]

    def ratios(self, lk: Tensor, b: Tensor) -> Tensor:
        """``r_j = K_j exp(-c_k(b) - phi_j b)``; each crystal's weighted mean is one."""
        e = self.residual(lk, b)
        return torch.exp(e - _crystal_log_mean(e, self.w, self.cid, self.n_cry)[self.cid])

    def gradient(self, r: Tensor) -> Tensor:
        """``sum_j w_j phi_j (1 - r_j)``, the gradient of the profiled objective."""
        return self.phi.T @ self.per_hkl(self.w * (1.0 - r))

    def objective(self, lk: Tensor, b: Tensor) -> Tensor:
        e = self.residual(lk, b)
        return (self.W * _crystal_log_mean(e, self.w, self.cid, self.n_cry)).sum() + self.wphi @ b

    def crystal_sums(self, wr: Tensor) -> tuple[Tensor, Tensor]:
        """Each crystal's sum of ``wr`` and its ``wr``-weighted basis sum (``n_cry x p``)."""
        phi, h, cid = self.phi, self.h, self.cid
        sr = torch.zeros(self.n_cry, device=wr.device, dtype=wr.dtype).index_add_(0, cid, wr).clamp(min=1e-12)
        srp = torch.zeros(self.n_cry, phi.shape[1], device=wr.device, dtype=wr.dtype)
        for a in range(0, h.numel(), _BLOCK):
            sl = slice(a, a + _BLOCK)
            srp.index_add_(0, cid[sl], phi.index_select(0, h[sl]) * wr[sl, None])
        return sr, srp

    def hessian(self, r: Tensor, x: Tensor | None = None) -> Tensor:
        """``sum_j w_j r_j A_j A_j^T`` less each crystal's outer product of its
        ``w r``-weighted column sum over its ``w r`` sum (the profiled crystal
        constants); ``A = [phi, x]`` with ``x`` columns given per observation
        (``n_obs x q``), the basis block first."""
        wr = self.w * r
        sr, srp = self.crystal_sums(wr)
        H = (self.phi * self.per_hkl(wr)[:, None]).T @ self.phi - (srp / sr[:, None]).T @ srp
        if x is None:
            return H
        q = x.shape[1]
        srx = torch.zeros(self.n_cry, q, device=r.device, dtype=r.dtype).index_add_(0, self.cid, wr[:, None] * x)
        hx = torch.zeros(self.phi.shape[0], q, device=r.device, dtype=r.dtype).index_add_(0, self.h, wr[:, None] * x)
        H_px = self.phi.T @ hx - (srp / sr[:, None]).T @ srx
        H_xx = (x * wr[:, None]).T @ x - (srx / sr[:, None]).T @ srx
        return torch.cat([torch.cat([H, H_px], 1), torch.cat([H_px.T, H_xx], 1)], 0)

    def start(self, lk: Tensor, sweeps: int = 3) -> Tensor:
        """Least-squares fit of ``log K ~ c_crystal + phi b`` (the geometric-mean gauge): a start
        near the solution however far the network's output has drifted along the gauge."""
        A = (self.phi * self.w_h[:, None]).T @ self.phi
        W = self.W.clamp(min=1.0)
        b = torch.zeros(self.phi.shape[1], device=lk.device, dtype=lk.dtype)
        for _ in range(sweeps):
            c = torch.zeros(self.n_cry, device=lk.device, dtype=lk.dtype).index_add_(
                0, self.cid, self.w * self.residual(lk, b)) / W
            b = psd_solve(A, self.phi.T @ self.per_hkl(self.w * (lk - c[self.cid])))
        return b

    def solve(self, lk: Tensor, sweeps: int, tol: float = 1e-10) -> Tensor:
        """Damped Newton from the least-squares start: each step halved until the
        objective does not increase; stops when a step no longer moves ``b``."""
        b = self.start(lk)
        f = self.objective(lk, b)
        for _ in range(sweeps):
            r = self.ratios(lk, b)
            step = psd_solve(self.hessian(r), self.gradient(r))
            t = 1.0
            for _ in range(30):
                b_new = b - t * step
                f_new = self.objective(lk, b_new)
                if bool(torch.isfinite(f_new)) and f_new <= f:
                    break
                t *= 0.5
            else:
                break
            b, f = b_new, f_new
            if float((t * step).abs().max()) < tol:
                break
        return b


def _runaway_weights(e: Tensor, cid: Tensor, n_cry: int, runaway: float) -> Tensor:
    return (e - _crystal_median(e, cid, n_cry)[cid] < runaway).to(e.dtype)


def integral_mode(phi: Tensor, hkl_idx: Tensor, log_K: Tensor, crystal_id: Tensor, n_crystals: int,
                  *, sweeps: int = 12, runaway: float = 10.0, guard_rounds: int = 3,
                  return_weights: bool = False):
    """The smooth mode ``b`` of the arithmetic mean of ``K``, solved exactly (float64).

    ``phi`` is the per-reflection basis without constant columns
    (:func:`smooth_columns`), ``hkl_idx`` and ``crystal_id`` index each
    observation's reflection and crystal.  ``b`` is in the graph through
    ``log_K``, with the exact derivative of the solution.  The runaway guard is
    repeated against the fitted residual until the excluded set is stable.
    With ``return_weights`` the observations' fit weights come back too (0 for
    an observation the guard left out).
    """
    phi = phi.to(device=log_K.device, dtype=torch.float64)
    h, cid = hkl_idx.long(), crystal_id.long()
    lk = log_K.to(torch.float64)
    n_cry = int(n_crystals)
    with torch.no_grad():
        x = lk.detach()
        fit = _Fit(phi, h, cid, n_cry, torch.ones_like(x))
        b = fit.solve(x, sweeps)
        for _ in range(guard_rounds):
            w = _runaway_weights(fit.residual(x, b), cid, n_cry, runaway)
            if torch.equal(w, fit.w):
                break
            fit = _Fit(phi, h, cid, n_cry, w)
            b = fit.solve(x, sweeps)
        H = fit.hessian(fit.ratios(x, b))
        if not (bool(torch.isfinite(b).all()) and bool(torch.isfinite(H).all())):
            raise FloatingPointError("the sum-of-spots fit did not reach finite values")
    b = b - psd_solve(H, fit.gradient(fit.ratios(lk, b)))
    return (b, fit.w) if return_weights else b


def integral_step(phi: Tensor, hkl_idx: Tensor, log_K: Tensor, crystal_id: Tensor, n_crystals: int,
                  *, extra: Tensor | None = None, runaway: float = 10.0) -> Tensor:
    """One Newton step of the fit from ``b = 0``, in closed form (float64).

    The crystal constants are profiled in closed form (each crystal's
    arithmetic mean of ``K``), the Hessian is taken outside the graph, and the
    runaway guard is one pass against each crystal's median of ``log K``.
    ``extra`` (``n_obs x q``) adds columns given per observation after the
    basis; ``b`` has ``p + q`` entries, the basis part first.  In the graph
    through ``log_K``.
    """
    phi = phi.to(device=log_K.device, dtype=torch.float64)
    h, cid = hkl_idx.long(), crystal_id.long()
    lk = log_K.to(torch.float64)
    n_cry = int(n_crystals)
    with torch.no_grad():
        w = _runaway_weights(lk.detach(), cid, n_cry, runaway)
    fit = _Fit(phi, h, cid, n_cry, w)
    r = fit.ratios(lk, torch.zeros(phi.shape[1], device=lk.device, dtype=lk.dtype))
    g = fit.gradient(r)
    x = None if extra is None else extra.to(device=lk.device, dtype=torch.float64).reshape(lk.numel(), -1)
    with torch.no_grad():
        H = fit.hessian(r.detach(), x)
    if x is not None:
        g = torch.cat([g, x.T @ (w * (1.0 - r))])
    if not bool(torch.isfinite(H).all()):
        raise FloatingPointError("the sum-of-spots step did not reach finite values")
    return -psd_solve(H, g)


def mode_mean_square(phi: Tensor, hkl_idx: Tensor, b: Tensor) -> Tensor:
    """``mean_j (phi(h_j) @ b)^2`` over the observations, in nats^2."""
    phi = phi.to(device=b.device, dtype=b.dtype)
    n_h = torch.bincount(hkl_idx.long(), minlength=phi.shape[0]).to(b.dtype)
    return (n_h * (phi @ b) ** 2).sum() / n_h.sum().clamp(min=1.0)


def step_mean_square(phi: Tensor, hkl_idx: Tensor, b: Tensor, extra: Tensor | None = None) -> Tensor:
    """``mean_j (phi(h_j) @ b_phi + extra_j @ b_x)^2`` over the observations, in nats^2:
    the log-intensity change the step ``b`` of :func:`integral_step` stands for."""
    phi = phi.to(device=b.device, dtype=b.dtype)
    p = phi.shape[1]
    m = (phi @ b[:p]).index_select(0, hkl_idx.long())
    if extra is not None:
        m = m + extra.to(b).reshape(m.numel(), -1) @ b[p:]
    return (m * m).mean() if m.numel() else m.sum()
