"""Maximum-likelihood lattice (+ geometry) refinement — "explain the spots".

TORO proposes an orientation/cell by grid-search + trimmed-LSQ (a discrete argmax
of a hard inlier count). This module *refines* that proposal by gradient-optimising
the probability of explaining the observed spots under a lattice + outlier mixture
— turning the vote-count into a differentiable likelihood that uses every spot
softly (weighted by its probability ``w``), and that can also refine the cell and
the detector geometry.

Convention (matches ``to_stream.reciprocal_vectors`` and ``indexing``):
``basis`` rows = real a,b,c (Å); ``UB = inv(basis)`` has **columns** a*,b*,c* (Å⁻¹);
``q_pred(hkl) = hkl @ UBᵀ``; ``frac = q @ basisᵀ = solve(UB, qᵀ)ᵀ``.

The objective, for spots ``q`` (N,3) with weights ``w`` (N,) under setting ``UB``:

    p_lat_i = Σ_{δ∈nbhd} 𝒩(q_i − (hkl0+δ)@UBᵀ ; σ_d² I)        # soft lattice density
    p_i     = π·w_i·p_lat_i + (1−π)·u_out                       # + uniform outlier
    NLL     = −Σ log p_i ,  responsibility r_i = π·w_i·p_lat_i / p_i

Summing 𝒩 over a small integer neighbourhood makes the ``round`` assignment smooth.
σ_d (lattice spread) and π (mixture weight) are fixed (physical); only the
lattice/geometry parameters are optimised.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


# ±1 in each Miller axis → 27 nearest lattice points (smooths the round assignment)
_NBHD = torch.tensor(
    [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
    dtype=torch.float64,
)


@dataclass
class RefineResult:
    basis: torch.Tensor            # (3,3) refined real-space cell, rows a,b,c (Å)
    responsibilities: torch.Tensor  # (N,) P(spot ∈ lattice) ∈ [0,1]
    loglik: float                   # Σ log p_i at the optimum
    hit_score: float                # Σ log[1 + π w p_lat /((1−π)u_out)] (log Bayes factor)
    n_inlier: int                   # responsibilities > 0.5

    @property
    def cell_lengths(self) -> torch.Tensor:
        return self.basis.norm(dim=1)


def _skew(v: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(v[..., 0])
    return torch.stack([z, -v[..., 2], v[..., 1],
                        v[..., 2], z, -v[..., 0],
                        -v[..., 1], v[..., 0], z], dim=-1).reshape(*v.shape[:-1], 3, 3)


def _q_shell_volume(q: torch.Tensor) -> float:
    """Volume (Å⁻³) of the sampled reciprocal shell, for the uniform outlier density."""
    n = q.detach().norm(dim=1)                 # u_out is a constant, not refined
    qmax, qmin = float(n.max()), float(n.min())
    return (4.0 / 3.0) * math.pi * (qmax ** 3 - qmin ** 3) or 1.0


def lattice_log_likelihood(
    q: torch.Tensor,
    w: torch.Tensor,
    UB: torch.Tensor,
    hkl0: torch.Tensor,
    *,
    sigma_d: float,
    pi: float,
    u_out: float,
    nbhd: torch.Tensor | None = None,
):
    """Mixture log-likelihood + per-spot responsibilities for a fixed assignment ``hkl0``.

    ``q,w,UB,hkl0`` float64; gradients flow through ``UB`` (``hkl0`` is the detached
    EM assignment). Returns ``(loglik, responsibilities, p_lat)``.
    """
    nbhd = _NBHD.to(q.device) if nbhd is None else nbhd
    inv_2s2 = 1.0 / (2.0 * sigma_d * sigma_d)
    norm = (2.0 * math.pi * sigma_d * sigma_d) ** (-1.5)

    hk = hkl0[:, None, :] + nbhd[None, :, :]           # (N, K, 3)
    q_pred = hk @ UB.T                                  # (N, K, 3)
    r2 = ((q[:, None, :] - q_pred) ** 2).sum(-1)        # (N, K)
    p_lat = norm * torch.exp(-inv_2s2 * r2).sum(1)      # (N,)

    sig = pi * w * p_lat
    p = (sig + (1.0 - pi) * u_out).clamp_min(1e-300)
    loglik = torch.log(p).sum()
    resp = sig / p
    return loglik, resp, p_lat


def refine_lattice(
    q: torch.Tensor,
    w: torch.Tensor,
    basis0: torch.Tensor,
    *,
    sigma_d: float = 0.0015,
    pi: float = 0.5,
    refine_cell: bool = True,
    cell_restraint: float = 0.0,
    ref_cell: torch.Tensor | None = None,
    n_outer: int = 4,
    max_iter: int = 60,
) -> RefineResult:
    """Refine a TORO proposal by maximum likelihood (mode A — geometry fixed).

    EM-style: assign ``hkl0`` at the current setting (E-step), then LBFGS-optimise the
    basis with that assignment fixed (M-step); repeat ``n_outer`` rounds.

    ``refine_cell=True`` optimises the full 3×3 ``UB`` (orientation + cell); ``False``
    optimises only a 3-DOF orientation ``R(δω)@UB0``. ``cell_restraint`` (with
    ``ref_cell``) adds a soft quadratic pull of the refined cell edge lengths toward the
    reference, anchoring scale (set high ⇒ effectively orientation-only).
    """
    dev = q.device
    q = q.to(torch.float64)
    w = w.to(torch.float64)
    UB0 = torch.linalg.inv(basis0.to(torch.float64))    # columns a*,b*,c*
    u_out = 1.0 / _q_shell_volume(q)
    ref_len = ref_cell.to(torch.float64).norm(dim=1) if ref_cell is not None else None

    if refine_cell:
        P = nn.Parameter(UB0.clone())
        ub = lambda: P                                   # noqa: E731
        params = [P]
    else:
        dw = nn.Parameter(torch.zeros(3, dtype=torch.float64, device=dev))
        ub = lambda: torch.linalg.matrix_exp(_skew(dw)) @ UB0   # noqa: E731
        params = [dw]

    for _ in range(n_outer):
        with torch.no_grad():
            hkl0 = torch.round(torch.linalg.solve(ub(), q.T).T)
        opt = torch.optim.LBFGS(params, max_iter=max_iter, line_search_fn="strong_wolfe",
                                history_size=10, tolerance_grad=1e-9, tolerance_change=1e-12)

        def closure():
            opt.zero_grad()
            UB = ub()
            ll, _, _ = lattice_log_likelihood(q, w, UB, hkl0, sigma_d=sigma_d,
                                              pi=pi, u_out=u_out)
            loss = -ll
            if cell_restraint > 0 and ref_len is not None:
                basis = torch.linalg.inv(UB)
                loss = loss + cell_restraint * ((basis.norm(dim=1) - ref_len) ** 2).sum()
            loss.backward()
            for p in params:                              # LBFGS needs contiguous grads
                if p.grad is not None and not p.grad.is_contiguous():
                    p.grad = p.grad.contiguous()
            return loss

        opt.step(closure)

    with torch.no_grad():
        UB = ub()
        hkl0 = torch.round(torch.linalg.solve(UB, q.T).T)
        ll, resp, p_lat = lattice_log_likelihood(q, w, UB, hkl0, sigma_d=sigma_d,
                                                 pi=pi, u_out=u_out)
        basis = torch.linalg.inv(UB)
        hit = torch.log1p(pi * w * p_lat / ((1.0 - pi) * u_out)).sum()

    return RefineResult(
        basis=basis.to(torch.float32).to(dev),
        responsibilities=resp.to(torch.float32).to(dev),
        loglik=float(ll),
        hit_score=float(hit),
        n_inlier=int((resp > 0.5).sum()),
    )


# ───────────────────────── stream persistence + geometry ─────────────────────
def crystals_from_stream(stream, geom):
    """Reload per-crystal records from an :class:`SFXStream` for re-refinement.

    Returns a list of dicts ``{ss, fs, hkl, w, basis0}`` per crystal (q is
    re-derived from the pixels through ``geom`` at refine time, so geometry can be
    varied). ``basis0 = inv(UB)`` with ``UB`` columns ``= (a*,b*,c*)``.
    """
    core = stream._core
    cid = core.crystal_id.long()
    has_prob = getattr(core, "prob", None) is not None
    out = []
    for c in range(core.n_crystals):
        m = cid == c
        M_rows = torch.stack([core.astar[c], core.bstar[c], core.cstar[c]], 0)  # rows a*,b*,c*
        basis0 = torch.linalg.inv(M_rows.T)                  # UB = M_rowsᵀ ⇒ basis = inv(UB)
        out.append({
            "ss": core.ss[m].to(torch.float64),
            "fs": core.fs[m].to(torch.float64),
            "hkl": core.hkl[m].to(torch.float64),
            "w": (core.prob[m] if has_prob else torch.ones(int(m.sum()))).to(torch.float64),
            "basis0": basis0.to(torch.float64),
        })
    return out


def refine_geometry(
    crystals,
    geom,
    *,
    sigma_d: float = 0.0015,
    pi: float = 0.5,
    refine_cell: bool = False,
    refine_clen: bool = True,
    refine_beam: bool = True,
    n_rounds: int = 3,
    max_iter: int = 50,
):
    """Block-coordinate joint refinement of global detector geometry + per-crystal
    lattices (mode B). ``crystals`` from :func:`crystals_from_stream`.

    Alternates: (a) per-crystal :func:`refine_lattice` at the current geometry, then
    (b) global geometry (``dclen`` scalar m, ``dbeam`` (2,) m) with lattices fixed —
    for ``n_rounds`` rounds. Needs ≥ several crystals to break the orientation↔geometry
    degeneracy. Returns ``{dclen, dbeam, bases, loglik, loglik0}``.
    """
    dclen = torch.zeros((), dtype=torch.float64, requires_grad=True)
    dbeam = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    gp = ([dclen] if refine_clen else []) + ([dbeam] if refine_beam else [])
    bases = [c["basis0"].clone() for c in crystals]

    def q_of(c):
        return geom.to_reciprocal(
            c["ss"], c["fs"],
            dclen=dclen if refine_clen else None,
            dbeam=dbeam if refine_beam else None,
        )

    def total_loglik():
        with torch.no_grad():
            tot = 0.0
            for c, b in zip(crystals, bases):
                q = q_of(c)
                UB = torch.linalg.inv(b)
                hkl0 = torch.round(torch.linalg.solve(UB, q.T).T)
                ll, _, _ = lattice_log_likelihood(q, c["w"], UB, hkl0, sigma_d=sigma_d,
                                                  pi=pi, u_out=1.0 / _q_shell_volume(q))
                tot += float(ll)
            return tot

    loglik0 = total_loglik()
    for _ in range(n_rounds):
        # (a) per-crystal lattice refine at current geometry
        with torch.no_grad():
            qs = [q_of(c).detach() for c in crystals]
        for i, c in enumerate(crystals):
            rr = refine_lattice(qs[i].to(torch.float32), c["w"].to(torch.float32),
                                bases[i].to(torch.float32), sigma_d=sigma_d, pi=pi,
                                refine_cell=refine_cell)
            bases[i] = rr.basis.to(torch.float64)
        if not gp:
            break
        # (b) global geometry, lattices fixed
        UBs = [torch.linalg.inv(b) for b in bases]
        with torch.no_grad():
            hkl0s = [torch.round(torch.linalg.solve(UBs[i], q_of(crystals[i]).T).T)
                     for i in range(len(crystals))]
        opt = torch.optim.LBFGS(gp, max_iter=max_iter, line_search_fn="strong_wolfe",
                                history_size=10, tolerance_grad=1e-12, tolerance_change=1e-14)

        def closure():
            opt.zero_grad()
            tot = 0.0
            for i, c in enumerate(crystals):
                q = q_of(c)
                ll, _, _ = lattice_log_likelihood(q, c["w"], UBs[i], hkl0s[i],
                                                  sigma_d=sigma_d, pi=pi,
                                                  u_out=1.0 / _q_shell_volume(q))
                tot = tot - ll
            tot.backward()
            for p in gp:
                if p.grad is not None and not p.grad.is_contiguous():
                    p.grad = p.grad.contiguous()
            return tot

        opt.step(closure)

    return {
        "dclen": float(dclen.detach()),
        "dbeam": dbeam.detach().tolist(),
        "bases": [b.to(torch.float32) for b in bases],
        "loglik": total_loglik(),
        "loglik0": loglik0,
    }
