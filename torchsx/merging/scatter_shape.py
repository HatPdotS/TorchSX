"""Reparametrizations for the GenGauss scatter shape (α, β).

`ScatterShape` is an ``nn.Module`` whose forward — split as ``log_alpha()``
and ``log_beta()`` — returns broadcast ``(n_hkl,)`` tensors that downstream
loss code can index with ``meta.hkl_idx`` exactly as before.  Concrete
subclasses own the trainable storage and decide how it broadcasts.

This is the seam through which the per-HKL parametrization (``Free``), a
single global pair (``Global``), or a frozen-at-truth value (``Fixed``) can
be swapped without rewiring the loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ScatterShape(nn.Module):
    """Base for (α, β) reparametrizations.  Subclasses set ``n_hkl`` and
    implement ``log_alpha`` / ``log_beta``."""

    n_hkl: int

    def log_alpha(self) -> Tensor:
        raise NotImplementedError

    def log_beta(self) -> Tensor:
        raise NotImplementedError


class FreePerHKLShape(ScatterShape):
    """Per-HKL ``nn.Parameter`` log_α, log_β — the legacy default."""

    def __init__(self, log_alpha_init: Tensor, log_beta_init: Tensor) -> None:
        super().__init__()
        if log_alpha_init.shape != log_beta_init.shape or log_alpha_init.dim() != 1:
            raise ValueError(
                "FreePerHKLShape expects 1-D log_alpha_init / log_beta_init of equal length"
            )
        self.n_hkl = int(log_alpha_init.numel())
        self._log_alpha = nn.Parameter(log_alpha_init.clone())
        self._log_beta = nn.Parameter(log_beta_init.clone())

    def log_alpha(self) -> Tensor:
        return self._log_alpha

    def log_beta(self) -> Tensor:
        return self._log_beta


class GlobalShape(ScatterShape):
    """Single scalar log_α, log_β broadcast to all HKLs."""

    def __init__(
        self, n_hkl: int, log_alpha_init: float, log_beta_init: float,
    ) -> None:
        super().__init__()
        self.n_hkl = int(n_hkl)
        self._log_alpha = nn.Parameter(torch.tensor(float(log_alpha_init)))
        self._log_beta = nn.Parameter(torch.tensor(float(log_beta_init)))

    def log_alpha(self) -> Tensor:
        return self._log_alpha.expand(self.n_hkl)

    def log_beta(self) -> Tensor:
        return self._log_beta.expand(self.n_hkl)


class AnisoQuadShape(ScatterShape):
    """log α(h) = a₀ + sᵀ · U_α · s   on physical reciprocal-space vectors
    s = hkl @ B_orth (Å⁻¹).

    Friedel symmetry zeroes any odd-order terms in (h, k, l), so the most
    general 2nd-order 3D polynomial reduces to a constant plus a symmetric
    quadratic form.  Stores 6 unique entries of U as a flat vector
    ``[Uxx, Uyy, Uzz, Uxy, Uxz, Uyz]`` (same packing as
    :func:`torchref.base.math_torch.U_to_matrix`).  Same parametrization
    for log β with independent parameters: 7 params per shape, 14 total.
    """

    def __init__(
        self,
        s_vec: Tensor,
        log_alpha_init: float = 0.0,
        log_beta_init: float = 0.0,
        U_alpha_init: Tensor | None = None,
        U_beta_init: Tensor | None = None,
    ) -> None:
        super().__init__()
        if s_vec.dim() != 2 or s_vec.shape[1] != 3:
            raise ValueError(
                f"AnisoQuadShape expects s_vec of shape (n_hkl, 3); got {tuple(s_vec.shape)}"
            )
        self.n_hkl = int(s_vec.shape[0])
        self.register_buffer("_s_vec", s_vec.clone().float())
        self._a0_alpha = nn.Parameter(torch.tensor(float(log_alpha_init)))
        self._a0_beta = nn.Parameter(torch.tensor(float(log_beta_init)))
        if U_alpha_init is None:
            U_alpha_init = torch.zeros(6)
        if U_beta_init is None:
            U_beta_init = torch.zeros(6)
        if U_alpha_init.numel() != 6 or U_beta_init.numel() != 6:
            raise ValueError("U_*_init must be 6-vectors (Uxx,Uyy,Uzz,Uxy,Uxz,Uyz)")
        self._U_alpha = nn.Parameter(U_alpha_init.detach().clone().float())
        self._U_beta = nn.Parameter(U_beta_init.detach().clone().float())

    def _quad(self, U: Tensor) -> Tensor:
        sx, sy, sz = self._s_vec.unbind(dim=1)
        return (
            U[0] * sx * sx + U[1] * sy * sy + U[2] * sz * sz
            + 2.0 * (U[3] * sx * sy + U[4] * sx * sz + U[5] * sy * sz)
        )

    def log_alpha(self) -> Tensor:
        return self._a0_alpha + self._quad(self._U_alpha)

    def log_beta(self) -> Tensor:
        return self._a0_beta + self._quad(self._U_beta)


class FixedShape(ScatterShape):
    """Non-trainable log_α, log_β stored as buffers.  For closure tests
    where (α, β) are pinned to truth."""

    def __init__(self, log_alpha: Tensor, log_beta: Tensor) -> None:
        super().__init__()
        if log_alpha.shape != log_beta.shape or log_alpha.dim() != 1:
            raise ValueError(
                "FixedShape expects 1-D log_alpha / log_beta of equal length"
            )
        self.n_hkl = int(log_alpha.numel())
        self.register_buffer("_log_alpha", log_alpha.clone())
        self.register_buffer("_log_beta", log_beta.clone())

    def log_alpha(self) -> Tensor:
        return self._log_alpha

    def log_beta(self) -> Tensor:
        return self._log_beta


def _design_matrix_aniso_quad(s_vec: Tensor) -> Tensor:
    """``(n_hkl, 7)`` design matrix for the AnisoQuadShape basis."""
    sx, sy, sz = s_vec.unbind(dim=1)
    return torch.stack(
        [
            torch.ones_like(sx),
            sx * sx, sy * sy, sz * sz,
            2.0 * sx * sy, 2.0 * sx * sz, 2.0 * sy * sz,
        ],
        dim=1,
    )


@torch.no_grad()
def project_to_shape(
    new_shape: ScatterShape,
    old_log_alpha: Tensor,
    old_log_beta: Tensor,
) -> None:
    """Project per-HKL ``log_α`` / ``log_β`` onto ``new_shape``'s parameters in place.

    Used by :meth:`LogSpaceLoss.swap_scatter_shape` to warm-start a new
    parametrization from an existing per-HKL fit:

    * :class:`FreePerHKLShape` — copy directly (identity).
    * :class:`GlobalShape`     — scalar = median of per-HKL log values.
    * :class:`AnisoQuadShape`  — least-squares projection onto
      ``[1, sx², sy², sz², 2sxsy, 2sxsz, 2sysz]``.
    * :class:`FixedShape`      — buffer values are not trainable; left alone.
    """
    if isinstance(new_shape, FreePerHKLShape):
        new_shape._log_alpha.data.copy_(
            old_log_alpha.to(new_shape._log_alpha.device)
        )
        new_shape._log_beta.data.copy_(
            old_log_beta.to(new_shape._log_beta.device)
        )
        return
    if isinstance(new_shape, GlobalShape):
        new_shape._log_alpha.data.fill_(float(old_log_alpha.median()))
        new_shape._log_beta.data.fill_(float(old_log_beta.median()))
        return
    if isinstance(new_shape, AnisoQuadShape):
        s_vec = new_shape._s_vec.detach().cpu().double()
        design = _design_matrix_aniso_quad(s_vec)
        # Use the normal-equation pseudoinverse on float64.  ``torch.linalg.lstsq``
        # has hit cuSOLVER / LAPACK ``Argument 6`` failures on some HKL
        # distributions; ``pinv`` of the (n_hkl, 7) matrix is cheap (7×7 SVD)
        # and unconditionally stable.
        a_log = old_log_alpha.detach().cpu().double().unsqueeze(-1)
        b_log = old_log_beta.detach().cpu().double().unsqueeze(-1)
        XtX = design.T @ design
        Xt_a = design.T @ a_log
        Xt_b = design.T @ b_log
        sol_a = torch.linalg.pinv(XtX) @ Xt_a
        sol_b = torch.linalg.pinv(XtX) @ Xt_b
        sol_a = sol_a.squeeze(-1).float()
        sol_b = sol_b.squeeze(-1).float()
        dev_a = new_shape._a0_alpha.device
        dev_b = new_shape._a0_beta.device
        new_shape._a0_alpha.data.fill_(float(sol_a[0]))
        new_shape._U_alpha.data.copy_(sol_a[1:7].to(dev_a))
        new_shape._a0_beta.data.fill_(float(sol_b[0]))
        new_shape._U_beta.data.copy_(sol_b[1:7].to(dev_b))
        return
    if isinstance(new_shape, FixedShape):
        return
    raise TypeError(
        f"project_to_shape: no projection rule for {type(new_shape).__name__}"
    )
