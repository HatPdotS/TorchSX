"""Smooth, symmetry-adapted basis on reciprocal space.

One low-dimensional function class ``phi(h)`` serves two jobs that must share
it:

* the Wilson prior's ``log Sigma(h)`` surface (replacing equal-count
  resolution shells), fitted by :func:`fit_wilson_log_sigma`;
* the gauge pin on the scale network: the ensemble-mean component of
  ``log K`` that lies in ``span(phi)`` is projected out before the
  likelihood sees it.

The likelihood is exactly invariant under ``K(h) -> K(h) g(h)``,
``I0(h) -> I0(h) / g(h)`` for any function ``g`` of reciprocal space.  A prior
whose ``Sigma`` is estimated from the current merge cannot anchor that
direction (``Sigma`` moves with ``I0``).  Pinning the mean of ``log K`` over
``span(phi)`` removes the smooth part of the freedom exactly; what is left is
orthogonal to ``phi`` and is precisely what a smooth ``Sigma`` penalises.

Columns
-------
* Chebyshev ``T_0 .. T_degree`` in ``x = 2 s2 / s2_max - 1`` with
  ``s2 = (sin theta / lambda)^2 = |s|^2 / 4`` (the ``ObsInfo.s_sq``
  convention).  ``T_1`` carries the isotropic Wilson B, the higher orders the
  curvature of a real Wilson plot (the 4-6 A solvent bump included).
* Quadratic forms ``s^T A_i s`` for an orthonormal basis ``{A_i}`` of the
  point-group-invariant **traceless** symmetric tensors, in the Cartesian
  reciprocal frame.  Traceless because the trace is ``|s|^2`` and already in
  ``T_1``.  P 63 has one such tensor (c* against ab), orthorhombic two,
  cubic none, P 1 five.

Every column except ``T_0`` is standardised over the reflections the basis was
built on, so the normal matrix of a least-squares fit is well conditioned.

The classical ``WilsonAnisoPrior`` (``wilson_aniso.py``) fits an
unsymmetrised 6-vector ``U``; this module's invariant tensors could replace it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "ReciprocalBasis",
    "chebyshev",
    "invariant_traceless_tensors",
    "fit_wilson_log_sigma",
    "wilson_b_equivalent",
]


def chebyshev(x: Tensor, degree: int) -> Tensor:
    """``T_0(x) .. T_degree(x)`` by the three-term recurrence, shape ``(n, degree+1)``."""
    cols = [torch.ones_like(x)]
    if degree >= 1:
        cols.append(x)
    for _ in range(2, degree + 1):
        cols.append(2.0 * x * cols[-1] - cols[-2])
    return torch.stack(cols, dim=1)


def _reciprocal_basis_matrix(cell: Tensor) -> Tensor:
    from torchref.base.reciprocal import reciprocal_basis_matrix

    return reciprocal_basis_matrix(cell.detach().to(torch.float64).cpu())


def _hkl_rotations(spacegroup: str | None) -> Tensor:
    """Rotation parts acting on Miller indices, row convention ``h' = h @ R``.

    Returns ``(n_ops, 3, 3)`` float64; the identity alone for ``None``/``P 1``.
    """
    if not spacegroup or spacegroup.replace(" ", "") == "P1":
        return torch.eye(3, dtype=torch.float64)[None]
    from torchref.symmetry import SpaceGroup

    sg = SpaceGroup(spacegroup)
    eye = torch.eye(3, dtype=torch.int64, device=sg.device)
    # images[o, j, :] = e_j @ R_o  ->  rows of R_o
    images = sg.expand_reciprocal(eye)
    return images.to(torch.float64).cpu()


def invariant_traceless_tensors(spacegroup: str | None, cell: Tensor) -> Tensor:
    """Orthonormal basis of point-group-invariant traceless symmetric tensors.

    The tensors act on the Cartesian reciprocal vector ``s = h @ B`` (rows of
    ``B`` are a*, b*, c*).  With ``h' = h @ R`` the vector transforms as
    ``s' = s @ M``, ``M = B^-1 R B``, and a quadratic form ``s A s^T`` is
    invariant iff ``M A M^T = A`` for every operation.  The group average
    ``A -> mean_o M_o A M_o^T`` is the projector onto the invariant subspace;
    its range is read off by SVD.

    Returns ``(k, 3, 3)`` float64 with ``k`` in ``0..5``.
    """
    B = _reciprocal_basis_matrix(cell)
    Binv = torch.linalg.inv(B)
    R = _hkl_rotations(spacegroup)
    M = Binv[None] @ R @ B[None]                       # (n_ops, 3, 3)

    # Orthonormal (Frobenius) basis of traceless symmetric 3x3 tensors.
    r2, r6 = 1.0 / 2 ** 0.5, 1.0 / 6 ** 0.5
    E = torch.tensor([
        [[r2, 0, 0], [0, -r2, 0], [0, 0, 0]],
        [[r6, 0, 0], [0, r6, 0], [0, 0, -2 * r6]],
        [[0, r2, 0], [r2, 0, 0], [0, 0, 0]],
        [[0, 0, r2], [0, 0, 0], [r2, 0, 0]],
        [[0, 0, 0], [0, 0, r2], [0, r2, 0]],
    ], dtype=torch.float64)                            # (5, 3, 3)

    averaged = torch.einsum("oij,ejk,olk->eil", M, E, M) / M.shape[0]   # (5, 3, 3)
    averaged = 0.5 * (averaged + averaged.transpose(1, 2))
    # Coordinates of the averaged tensors in the E basis: P[e, f] = <avg_e, E_f>.
    P = torch.einsum("eij,fij->ef", averaged, E)      # (5, 5), the projector
    U, S, _ = torch.linalg.svd(P)
    keep = S > 0.5
    if not bool(keep.any()):
        return torch.zeros(0, 3, 3, dtype=torch.float64)
    coords = U[:, keep].T                              # (k, 5)
    return torch.einsum("ke,eij->kij", coords, E)


@dataclass
class ReciprocalBasis:
    """``phi(h)`` on a fixed reflection set, plus what is needed to evaluate it elsewhere."""

    phi: Tensor            # (n_hkl, p) float32, standardised columns (T_0 = 1)
    s2: Tensor             # (n_hkl,) (sin theta / lambda)^2 of the reflections
    cell: Tensor           # (6,) float64
    spacegroup: str | None
    degree: int
    s2_max: float
    aniso_tensors: Tensor  # (k, 3, 3) float64
    col_mean: Tensor       # (p,) float64, zero for T_0
    col_std: Tensor        # (p,) float64, one for T_0

    @property
    def n_aniso(self) -> int:
        return int(self.aniso_tensors.shape[0])

    @property
    def p(self) -> int:
        return int(self.phi.shape[1])

    @classmethod
    def from_hkl(
        cls,
        hkl: Tensor,
        cell: Tensor,
        spacegroup: str | None,
        *,
        degree: int = 4,
        aniso: bool = True,
    ) -> "ReciprocalBasis":
        cell64 = cell.detach().to(torch.float64).cpu()
        tensors = (invariant_traceless_tensors(spacegroup, cell64) if aniso
                   else torch.zeros(0, 3, 3, dtype=torch.float64))
        raw, s2 = cls._raw_columns(hkl, cell64, degree, tensors, s2_max=None)
        s2_max = float(s2.max().clamp(min=1e-12))
        raw, s2 = cls._raw_columns(hkl, cell64, degree, tensors, s2_max=s2_max)
        col_mean = raw.mean(dim=0)
        col_std = raw.std(dim=0, unbiased=False)
        col_mean[0] = 0.0
        col_std[0] = 1.0
        col_std = torch.where(col_std > 1e-12, col_std, torch.ones_like(col_std))
        phi = ((raw - col_mean) / col_std).to(torch.float32)
        return cls(phi=phi, s2=s2.to(torch.float32), cell=cell64, spacegroup=spacegroup,
                   degree=int(degree), s2_max=s2_max, aniso_tensors=tensors,
                   col_mean=col_mean, col_std=col_std)

    @staticmethod
    def _raw_columns(hkl: Tensor, cell64: Tensor, degree: int, tensors: Tensor,
                     *, s2_max: float | None) -> tuple[Tensor, Tensor]:
        B = _reciprocal_basis_matrix(cell64)
        s_vec = hkl.detach().to(torch.float64).cpu() @ B          # (n, 3), 1/A
        s2 = (s_vec * s_vec).sum(dim=1) / 4.0
        if s2_max is None:
            return s2, s2
        x = (2.0 * s2 / s2_max - 1.0).clamp(-1.0, 1.0)
        cols = [chebyshev(x, degree)]
        if tensors.shape[0]:
            cols.append(torch.einsum("ni,kij,nj->nk", s_vec, tensors, s_vec))
        return torch.cat(cols, dim=1), s2

    def evaluate(self, hkl: Tensor) -> Tensor:
        """``phi`` for arbitrary Miller indices, on the stored standardisation."""
        raw, _ = self._raw_columns(hkl, self.cell, self.degree, self.aniso_tensors,
                                   s2_max=self.s2_max)
        return ((raw - self.col_mean) / self.col_std).to(torch.float32)

    def definition(self) -> dict:
        """Everything :meth:`evaluate` needs, without the reflection set."""
        return {"cell": self.cell.tolist(), "spacegroup": self.spacegroup, "degree": self.degree,
                "s2_max": self.s2_max, "aniso_tensors": self.aniso_tensors.clone(),
                "col_mean": self.col_mean.clone(), "col_std": self.col_std.clone()}

    @classmethod
    def from_definition(cls, d: dict) -> "ReciprocalBasis":
        """A basis that evaluates like the one :meth:`definition` was taken from;
        its own reflection set is empty."""
        p = int(torch.as_tensor(d["col_mean"]).numel())
        return cls(phi=torch.zeros(0, p), s2=torch.zeros(0),
                   cell=torch.as_tensor(d["cell"], dtype=torch.float64), spacegroup=d["spacegroup"],
                   degree=int(d["degree"]), s2_max=float(d["s2_max"]),
                   aniso_tensors=torch.as_tensor(d["aniso_tensors"], dtype=torch.float64),
                   col_mean=torch.as_tensor(d["col_mean"], dtype=torch.float64),
                   col_std=torch.as_tensor(d["col_std"], dtype=torch.float64))

    def to(self, device) -> "ReciprocalBasis":
        return ReciprocalBasis(phi=self.phi.to(device), s2=self.s2.to(device), cell=self.cell,
                               spacegroup=self.spacegroup, degree=self.degree, s2_max=self.s2_max,
                               aniso_tensors=self.aniso_tensors, col_mean=self.col_mean,
                               col_std=self.col_std)


def fit_wilson_log_sigma(
    phi: Tensor,
    I: Tensor,
    is_centric: Tensor | None = None,
    coef0: Tensor | None = None,
    *,
    sample_weight: Tensor | None = None,
    n_iter: int = 8,
    ridge: float = 1e-6,
    tol: float = 1e-8,
) -> tuple[Tensor, int]:
    """Maximum-likelihood Wilson fit ``log Sigma(h) = phi(h) @ c``.

    Minimises ``sum_h w_h (log Sigma_h + I_h / Sigma_h)`` with ``w = 1``
    (acentric) or ``1/2`` (centric) -- the negative log of Wilson's
    distributions up to constants -- by damped Newton steps from ``coef0``
    (warm start) or from the flat ``log mean(I)`` solution.

    Optional ``sample_weight`` multiplies the centric/acentric weight; zero
    weight excludes a reflection from both fitting and initialization.

    Returns ``(coef, n_iterations_used)``; ``coef`` has ``phi.dtype``.
    """
    phi64 = phi.detach().to(torch.float64)
    I64 = I.detach().to(torch.float64).clamp(min=1e-30)
    w = torch.ones_like(I64)
    if is_centric is not None:
        w = torch.where(is_centric.to(I64.device), torch.full_like(I64, 0.5), w)
    if sample_weight is not None:
        weights = sample_weight.detach().to(I64)
        if weights.shape != I64.shape or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()) or not bool((weights > 0).any()):
            raise ValueError("Wilson sample weights must be finite, nonnegative and nonempty")
        keep = weights > 0
        phi64, I64, w = phi64[keep], I64[keep], (w * weights)[keep]
    p = phi64.shape[1]
    if coef0 is None:
        c = torch.zeros(p, dtype=torch.float64, device=phi64.device)
        c[0] = (I64.mean() if sample_weight is None else (w * I64).sum() / w.sum()).log()
    else:
        c = coef0.detach().to(torch.float64).clone()

    def objective(cc: Tensor) -> Tensor:
        ls = phi64 @ cc
        return (w * (ls + I64 * torch.exp(-ls))).sum()

    f = objective(c)
    used = 0
    eye = torch.eye(p, dtype=torch.float64, device=phi64.device)
    for it in range(n_iter):
        ls = phi64 @ c
        r = I64 * torch.exp(-ls)                       # I / Sigma
        grad = phi64.T @ (w * (1.0 - r))
        H = phi64.T @ (phi64 * (w * r)[:, None]) + ridge * eye
        step = torch.linalg.solve(H, grad)
        t = 1.0
        while True:
            c_new = c - t * step
            f_new = objective(c_new)
            if f_new <= f or t < 1e-4:
                break
            t *= 0.5
        used = it + 1
        converged = (f - f_new).abs() <= tol * max(1.0, float(f.abs()))
        c, f = c_new, f_new
        if converged:
            break
    return c.to(phi.dtype), used


def wilson_b_equivalent(basis: ReciprocalBasis, coef: Tensor) -> float:
    """Isotropic B (A^2) of the fitted surface: ``log Sigma ~ -2 B s2`` slope over the data range."""
    ls = (basis.phi.to(torch.float64) @ coef.detach().to(torch.float64).to(basis.phi.device))
    s2 = basis.s2.to(torch.float64)
    x = s2 - s2.mean()
    slope = float((x * (ls - ls.mean())).sum() / (x * x).sum().clamp(min=1e-30))
    return -0.5 * slope
