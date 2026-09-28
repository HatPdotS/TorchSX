"""Learned detector noise: a per-observation dispersion of the counting term.

The detection likelihood is a deviance-scaled Poisson in the counts,
``log p = iphi * (-D/2) + 1/2 log iphi + logpeak`` with ``iphi = lam / (lam + sig_b2)``.
Its variance is ``lam + sig_b2`` in photon units, so everything the detector does
beyond Poisson -- a gain that is off, a gain that depends on the flux (JUNGFRAU's
gain stages, a counting detector's pile-up), a module whose calibration differs
-- is simply absent from it.  This module adds one factor per observation::

    Var = phi_i * (lam + sig_b2),   log p = iphi * (-D/2) / phi_i + 1/2 log iphi - 1/2 log phi_i + logpeak

under the extended quasi-likelihood.  ``phi`` is exactly a gain correction: dividing
the counts, the background and ``sig_b2`` by ``phi`` gives ``D / phi`` with ``iphi``
unchanged, and the Jacobian of that change of units supplies the ``-1/2 log phi``
(to the order the Poisson saddlepoint holds).  That term is also what makes ``phi``
learnable: without it the loss would fall without bound as ``phi`` grows.

``phi`` is a function of the detector and of the observation's expected count::

    log phi_i = a[group_i] + f(log lam_hat_i)

``group`` is the detector panel (one group when the stream has none).  ``f`` is
piecewise linear on fixed knots in ``log lam``, zero at the lowest knot, so ``a``
is the dispersion of background-dominated observations of each panel and ``f``
how it changes with the flux.  ``lam_hat = K * I_ref + bg`` is the model's own
expected count (the running reference intensity, detached), never the
observation's count -- a variance that depends on the observed count would bias
the fit, as it does for weighted means.  It must also stay blind to the
condition: the detector does not know whether the pump fired.

At initialisation ``a = f = 0`` and the likelihood is exactly the current one.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["DetectorNoise"]


class DetectorNoise(nn.Module):
    """``log phi = a[group] + f(log lam_hat)``, installed per batch by ``obs_id``.

    Parameters
    ----------
    n_groups
        Detector groups (panels); observations carry ``group`` in ``[0, n_groups)``.
    knots
        Knot positions of ``f`` in ``log lam`` (photons in the peak).  ``None`` gives
        ``mean``-free noise (``a`` only).
    bound
        ``|log phi| <= bound`` through a soft ``tanh`` clamp.
    """

    def __init__(self, n_groups: int = 1, knots: Tensor | None = None, *,
                 bound: float = math.log(50.0)) -> None:
        super().__init__()
        if n_groups < 1:
            raise ValueError("n_groups must be positive")
        self.n_groups = int(n_groups)
        self.a = nn.Parameter(torch.zeros(self.n_groups))
        if knots is not None:
            knots = torch.as_tensor(knots, dtype=torch.float32).flatten()
            if knots.numel() < 2 or not bool((knots[1:] > knots[:-1]).all()):
                raise ValueError("knots must be increasing, at least two")
            self.register_buffer("knots", knots.clone())
            self.f = nn.Parameter(torch.zeros(knots.numel() - 1))   # f(knot 0) = 0
        else:
            self.register_buffer("knots", torch.zeros(0))
            self.register_parameter("f", None)
        self.bound = float(bound)
        self._sorted_ids: Tensor | None = None
        self._perm: Tensor | None = None
        self._log_u: Tensor | None = None
        self._key: tuple | None = None

    @classmethod
    def default(cls, n_groups: int, mean_specific: bool = True) -> "DetectorNoise":
        """Knots every 1.5 nats from 1 to ~2e6 photons in the peak."""
        knots = torch.arange(0.0, 15.0, 1.5) if mean_specific else None
        return cls(n_groups, knots)

    @property
    def mean_specific(self) -> bool:
        return self.f is not None

    def config(self) -> dict:
        return {"n_groups": self.n_groups, "knots": self.knots.tolist() or None, "bound": self.bound}

    @classmethod
    def from_config(cls, cfg: dict) -> "DetectorNoise":
        knots = cfg.get("knots")
        return cls(int(cfg["n_groups"]), None if not knots else torch.tensor(knots), bound=float(cfg["bound"]))

    # ── the function ───────────────────────────────────────────────────

    def _f(self, log_lam: Tensor) -> Tensor:
        k = self.knots.to(log_lam.dtype)
        vals = torch.cat([self.f.new_zeros(1), self.f]).to(log_lam.dtype)
        x = log_lam.clamp(float(k[0]), float(k[-1]))
        i = torch.searchsorted(k, x.detach().contiguous(), right=True).clamp(1, k.numel() - 1)
        x0, x1 = k[i - 1], k[i]
        t = (x - x0) / (x1 - x0)
        return vals[i - 1] * (1.0 - t) + vals[i] * t

    def log_phi(self, group: Tensor | None, log_lam_hat: Tensor) -> Tensor:
        raw = (self.a[group] if group is not None else self.a[0].expand_as(log_lam_hat)).to(log_lam_hat.dtype)
        if self.f is not None:
            raw = raw + self._f(log_lam_hat.detach())
        return self.bound * torch.tanh(raw / self.bound)

    # ── per-batch installation, addressed by obs_id like the scatter ──

    def prime(self, obs_id: Tensor, group: Tensor | None, log_lam_hat: Tensor, *, key=None) -> Tensor:
        """Compute and install ``log u = -log phi`` for the observations ``obs_id``.

        ``key`` identifies the batch as a tuple of objects compared by identity
        (the batch's scale prediction and metadata); they are held until the next
        batch, so no later object can reuse their ids.  A second call with the same
        key returns the installed tensor, so the outer-node and locate passes of
        one step share it (and its autograd graph).
        """
        if self.matches(key):
            return self._log_u
        log_u = -self.log_phi(group, log_lam_hat)
        sorted_ids, perm = torch.sort(obs_id)
        self._sorted_ids, self._perm, self._log_u, self._key = sorted_ids, perm, log_u, key
        return log_u

    def matches(self, key) -> bool:
        """Whether the installed prediction is this batch's and usable here.

        A prediction made under ``no_grad`` (the locate scan runs first in each
        step) has no graph, so it is not reused once gradients are on -- the
        loss pass would otherwise train nothing.
        """
        k = self._key
        return (key is not None and k is not None and self._log_u is not None
                and len(k) == len(key) and all(a is b for a, b in zip(k, key))
                and (self._log_u.requires_grad or not torch.is_grad_enabled()))

    def clear(self) -> None:
        self._sorted_ids = self._perm = self._log_u = None
        self._key = None

    @property
    def active(self) -> bool:
        return self._log_u is not None

    def log_u_obs(self, obs_id: Tensor) -> Tensor:
        """The installed ``log u`` for ``obs_id`` (any subset or order of the batch)."""
        if self._log_u is None:
            raise RuntimeError("DetectorNoise has no prediction installed; call prime() first")
        loc = torch.searchsorted(self._sorted_ids, obs_id).clamp(max=self._sorted_ids.numel() - 1)
        if not bool((self._sorted_ids[loc] == obs_id).all()):
            raise KeyError("obs_id outside the installed detector-noise prediction")
        return self._log_u[self._perm[loc]]
