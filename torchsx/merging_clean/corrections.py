"""Differentiable per-observation log-space corrections.

Every concrete correction is an ``nn.Module``:
    - declares its parameters as ``nn.Parameter`` in ``__init__``
    - returns a per-obs additive log-space correction from ``forward(obs_info)``
    - (optionally) returns a scalar penalty from ``regularizer()``

Corrections are registered in an ``nn.ModuleList`` on ``UnifiedMergeLoss``,
so all their parameters are discovered by ``optimizer = LBFGS(loss.parameters(), ...)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


# ── Per-observation metadata ────────────────────────────────


@dataclass
class ObsInfo:
    """Per-observation metadata, sorted in the same order as the
    observations inside ``UnifiedMergeLoss``."""

    crystal_id: Tensor      # (N,) long
    s_sq: Tensor            # (N,) float  — 1/(4 d²)
    hkl_idx: Tensor         # (N,) long   — index into valid-HKL array
    n_crystals: int
    panel_id: Tensor | None = None
    fs: Tensor | None = None
    ss: Tensor | None = None
    peak: Tensor | None = None
    background: Tensor | None = None

    def slice(self, s: int, e: int) -> "ObsInfo":
        """Return a view restricted to obs[s:e] (for chunked closures)."""
        return ObsInfo(
            crystal_id=self.crystal_id[s:e],
            s_sq=self.s_sq[s:e],
            hkl_idx=self.hkl_idx[s:e],
            n_crystals=self.n_crystals,
            panel_id=None if self.panel_id is None else self.panel_id[s:e],
            fs=None if self.fs is None else self.fs[s:e],
            ss=None if self.ss is None else self.ss[s:e],
            peak=None if self.peak is None else self.peak[s:e],
            background=None if self.background is None else self.background[s:e],
        )

    def to(self, device) -> "ObsInfo":
        def _m(t):
            return None if t is None else t.to(device)
        return ObsInfo(
            crystal_id=self.crystal_id.to(device),
            s_sq=self.s_sq.to(device),
            hkl_idx=self.hkl_idx.to(device),
            n_crystals=self.n_crystals,
            panel_id=_m(self.panel_id),
            fs=_m(self.fs),
            ss=_m(self.ss),
            peak=_m(self.peak),
            background=_m(self.background),
        )


# ── Base class ──────────────────────────────────────────────


class Correction(nn.Module):
    """Differentiable per-observation log-space correction.

    Subclasses implement ``forward(obs_info, obs_range=None) → (N_obs,)``.
    ``obs_info`` is already sliced to the current chunk (length == chunk
    size).  ``obs_range`` is the `(start, end)` of the chunk in the
    original obs order — only needed by corrections that depend on the
    original obs index (e.g. ``FixedCorrection``).

    May override ``regularizer() → scalar`` (default: 0).
    """

    def forward(
        self, obs_info: ObsInfo, obs_range: tuple[int, int] | None = None
    ) -> Tensor:  # pragma: no cover
        raise NotImplementedError

    def regularizer(self) -> Tensor:
        """Scalar penalty on this correction's parameters.  Default: 0."""
        try:
            p = next(self.parameters())
            return torch.zeros((), device=p.device, dtype=p.dtype)
        except StopIteration:
            return torch.zeros(())


# ── Scale + B-factor (per crystal) ──────────────────────────


class ScaleBCorrection(Correction):
    """Per-crystal overall scale (log G) and B-factor.

    Log-space model::

        c_j = (log_G − mean(log_G))[cid_j]  −  2 · B[cid_j] · s²_j

    The ``log_G − mean`` re-parameterisation hard-enforces zero-mean on the
    crystal scales, which otherwise trade off against ``log_I0`` (they are
    only individually identifiable up to a global constant).  The ridge
    penalty on the centred ``log_G`` and on ``B`` anchors the magnitudes.
    """

    def __init__(
        self,
        n_crystals: int = 0,
        *,
        scale_prior: float = 10.0,
        bfactor_prior: float = 1.0,
    ):
        super().__init__()
        self.n_crystals = n_crystals
        self.scale_prior = scale_prior
        self.bfactor_prior = bfactor_prior
        self.log_G_raw = nn.Parameter(torch.zeros(n_crystals))
        self.B = nn.Parameter(torch.zeros(n_crystals))

    @classmethod
    def from_stream(
        cls,
        stream,
        obs_info: ObsInfo,
        *,
        scale_prior: float = 10.0,
        bfactor_prior: float = 1.0,
    ) -> "ScaleBCorrection":
        n = int(obs_info.n_crystals)
        return cls(n_crystals=n, scale_prior=scale_prior, bfactor_prior=bfactor_prior)

    # Centred log_G (view, used by forward and regularizer)
    def _log_G_centred(self) -> Tensor:
        return self.log_G_raw - self.log_G_raw.mean()

    def forward(
        self, obs_info: ObsInfo, obs_range: tuple[int, int] | None = None
    ) -> Tensor:
        del obs_range  # not needed — scattering via crystal_id is slice-agnostic
        cid = obs_info.crystal_id
        s2 = obs_info.s_sq
        return self._log_G_centred()[cid] - 2.0 * self.B[cid] * s2

    def regularizer(self) -> Tensor:
        lg = self._log_G_centred()
        return self.scale_prior * (lg * lg).sum() + self.bfactor_prior * (self.B * self.B).sum()


# ── Panel gain (optional linear gradient) ───────────────────


class PanelCorrection(Correction):
    """Per-panel gain offset (+ optional fs/ss linear gradient)."""

    def __init__(
        self,
        n_panels: int = 0,
        *,
        fit_gradient: bool = False,
        panel_prior: float = 1.0,
    ):
        super().__init__()
        self.n_panels = n_panels
        self.fit_gradient = fit_gradient
        self.panel_prior = panel_prior
        self.gain = nn.Parameter(torch.zeros(n_panels))
        if fit_gradient:
            self.grad_fs = nn.Parameter(torch.zeros(n_panels))
            self.grad_ss = nn.Parameter(torch.zeros(n_panels))
        else:
            self.register_parameter("grad_fs", None)
            self.register_parameter("grad_ss", None)

    @classmethod
    def from_stream(
        cls,
        stream,
        obs_info: ObsInfo,
        *,
        fit_gradient: bool = False,
        panel_prior: float = 1.0,
    ) -> "PanelCorrection":
        if obs_info.panel_id is None:
            raise ValueError("PanelCorrection requires obs_info.panel_id")
        n_panels = int(obs_info.panel_id.max().item()) + 1
        return cls(n_panels=n_panels, fit_gradient=fit_gradient, panel_prior=panel_prior)

    def forward(
        self, obs_info: ObsInfo, obs_range: tuple[int, int] | None = None
    ) -> Tensor:
        del obs_range
        pid = obs_info.panel_id
        assert pid is not None, "PanelCorrection requires panel_id"
        pred = self.gain[pid]
        if self.grad_fs is not None:
            pred = pred + self.grad_fs[pid] * obs_info.fs + self.grad_ss[pid] * obs_info.ss
        return pred

    def regularizer(self) -> Tensor:
        reg = self.panel_prior * (self.gain * self.gain).sum()
        if self.grad_fs is not None:
            reg = reg + self.panel_prior * (self.grad_fs * self.grad_fs).sum()
            reg = reg + self.panel_prior * (self.grad_ss * self.grad_ss).sum()
        return reg


# ── Fixed (non-trainable) correction ────────────────────────


class FixedCorrection(Correction):
    """Pre-computed per-observation correction (e.g. from a neural net)."""

    def __init__(self, log_c_sorted: Tensor):
        super().__init__()
        # Buffer, not parameter
        self.register_buffer("log_c", log_c_sorted.clone())

    def forward(
        self, obs_info: ObsInfo, obs_range: tuple[int, int] | None = None
    ) -> Tensor:
        if obs_range is None:
            return self.log_c
        s, e = obs_range
        return self.log_c[s:e]

    def regularizer(self) -> Tensor:
        return torch.zeros((), device=self.log_c.device, dtype=self.log_c.dtype)
