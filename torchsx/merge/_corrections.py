"""Modular per-observation corrections for the GenGauss EM pipeline.

Corrections model systematic effects on individual observations (per-crystal
scale, B-factor, partiality, absorption, ...).  Each correction is fit on a
**work set** of observations; a held-out **free set** monitors overfitting.

Architecture
------------
1. ``Correction`` — abstract base class.  Subclass and implement ``fit`` /
   ``predict`` / ``n_params``.
2. ``ObsInfo`` — per-observation metadata bundle passed to every correction.
3. ``CorrectionPipeline`` — holds a list of corrections, manages the
   work/free split, orchestrates fitting, and computes monitoring stats.

Adding a new correction
-----------------------
Subclass ``Correction``, implement the three required methods, and append an
instance to the pipeline's ``corrections`` list.  The EM loop in
``fit_gengauss`` will pick it up automatically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import Tensor


# ── Per-observation metadata ────────────────────────────────


@dataclass
class ObsInfo:
    """Per-observation metadata for correction fitting.

    All tensors are ``(N_obs,)`` and sorted in the same order as
    ``log_I_sorted`` in the EM loop.  Extend with additional fields as
    new corrections require them (e.g. excitation_error, orientation).
    """

    crystal_id: Tensor   # (N,) long — crystal index per observation
    s_sq: Tensor         # (N,) float — 1/(4d²) per observation
    hkl_idx: Tensor      # (N,) long — index into valid-HKL array
    n_crystals: int      # total number of distinct crystals
    # Optional fields for additional corrections
    panel_id: Tensor | None = None   # (N,) long — detector panel index
    fs: Tensor | None = None         # (N,) float — fast-scan position
    ss: Tensor | None = None         # (N,) float — slow-scan position
    peak: Tensor | None = None       # (N,) float — peak counts
    background: Tensor | None = None # (N,) float — background counts


# ── Abstract correction ─────────────────────────────────────


class Correction(ABC):
    """Base class for per-observation log-space intensity corrections.

    A correction models a systematic effect ``c_j`` such that

        log(I_obs_j) = log(I_true_h) + c_j + noise

    The correction is fit from residuals ``r_j = log(I_obs_j) - μ_h`` on
    the work set and applied to **all** observations via ``predict()``.
    """

    @abstractmethod
    def fit(
        self,
        residuals: Tensor,
        obs_info: ObsInfo,
        work_mask: Tensor,
    ) -> None:
        """Fit correction parameters from work-set residuals.

        Parameters
        ----------
        residuals : (N,) log-space residuals after subtracting μ_h and
            all previously-fit corrections in the pipeline.
        obs_info : per-observation metadata.
        work_mask : (N,) bool — True for work-set observations.
        """

    @abstractmethod
    def predict(self, obs_info: ObsInfo) -> Tensor:
        """Return per-observation correction in log-space.

        Returns (N,) tensor to **subtract** from log(I_obs).
        """

    @abstractmethod
    def n_params(self) -> int:
        """Number of fitted parameters (for monitoring)."""

    def summary(self) -> str:
        """One-line human-readable summary (override for detail)."""
        return f"{self.__class__.__name__}: {self.n_params()} params"


# ── Fixed (pre-computed) correction ────────────────────────


class FixedCorrection(Correction):
    """Pre-computed per-observation correction (e.g. from a neural net).

    Stores a fixed log-space correction vector.  ``fit()`` is a no-op.
    The correction vector must be in the same order as the observations
    in the stream (i.e. ``core.I`` order, before any sorting).

    Parameters
    ----------
    log_c : (N_obs,) tensor — log-space correction per observation.
        Will be reindexed to match the sorted order in the EM loop.
    """

    def __init__(self, log_c: Tensor) -> None:
        self._log_c_raw = log_c  # original order (stream order)
        self._log_c_sorted: Tensor | None = None

    def fit(self, residuals: Tensor, obs_info: ObsInfo, work_mask: Tensor) -> None:
        """No-op — corrections are pre-computed."""
        pass

    def predict(self, obs_info: ObsInfo) -> Tensor:
        """Return the pre-computed corrections.

        Note: on first call, the raw corrections are assumed to already
        be in the correct sorted order (the caller must reindex).
        """
        if self._log_c_sorted is not None:
            return self._log_c_sorted
        # Assume corrections are already in the right order
        return self._log_c_raw[:obs_info.crystal_id.shape[0]]

    def n_params(self) -> int:
        return 0  # not fitted

    def summary(self) -> str:
        return f"FixedCorrection: {self._log_c_raw.shape[0]} obs"


# ── Scale + B-factor correction ─────────────────────────────


class ScaleBCorrection(Correction):
    """Per-crystal overall scale and B-factor.

    Model::

        c_j = log(G_c) − 2·B_c·s²_j

    Fit via ridge-regularised OLS per crystal (closed-form scatter ops).
    """

    def __init__(
        self,
        scale_prior: float = 10.0,
        bfactor_prior: float = 1.0,
    ):
        self.scale_prior = scale_prior
        self.bfactor_prior = bfactor_prior
        self.log_G: Tensor | None = None
        self.B: Tensor | None = None
        self._n_crystals = 0

    def set_priors(
        self,
        scale_prior: float | None = None,
        bfactor_prior: float | None = None,
    ) -> None:
        """Update ridge regularisation strengths (e.g. warm-up schedule)."""
        if scale_prior is not None:
            self.scale_prior = scale_prior
        if bfactor_prior is not None:
            self.bfactor_prior = bfactor_prior

    # ── interface ──

    def fit(
        self,
        residuals: Tensor,
        obs_info: ObsInfo,
        work_mask: Tensor,
    ) -> None:
        cid = obs_info.crystal_id
        s2 = obs_info.s_sq
        n_cryst = obs_info.n_crystals
        self._n_crystals = n_cryst

        # Only work-set observations contribute to the fit
        w = work_mask.float()

        _dev = cid.device
        cnt = torch.zeros(n_cryst, device=_dev).scatter_add_(0, cid, w)
        sum_r = torch.zeros(n_cryst, device=_dev).scatter_add_(0, cid, residuals * w)
        sum_s = torch.zeros(n_cryst, device=_dev).scatter_add_(0, cid, s2 * w)
        sum_rs = torch.zeros(n_cryst, device=_dev).scatter_add_(0, cid, residuals * s2 * w)
        sum_ss = torch.zeros(n_cryst, device=_dev).scatter_add_(0, cid, (s2 ** 2) * w)

        # 2×2 ridge: (X'X + Λ)β = X'y
        lam_a = self.scale_prior
        lam_b = self.bfactor_prior
        a11 = cnt + lam_a
        a12 = sum_s
        a22 = sum_ss + lam_b
        det = (a11 * a22 - a12 * a12).clamp(min=1e-12)

        a_coeff = (a22 * sum_r - a12 * sum_rs) / det
        b_coeff = (a11 * sum_rs - a12 * sum_r) / det

        self.log_G = a_coeff.clamp(-10, 10)
        self.B = (-b_coeff / 2).clamp(-50, 50)

        # Zero for crystals with no work-set observations
        no_obs = cnt < 0.5
        self.log_G[no_obs] = 0.0
        self.B[no_obs] = 0.0

        # Normalize: zero-mean log_G (prevents global scale drift)
        has_obs = ~no_obs
        if has_obs.sum() > 0:
            self.log_G[has_obs] -= self.log_G[has_obs].mean()

    def predict(self, obs_info: ObsInfo) -> Tensor:
        if self.log_G is None:
            return torch.zeros(obs_info.crystal_id.shape[0], device=obs_info.crystal_id.device)
        return (
            self.log_G[obs_info.crystal_id]
            - 2.0 * self.B[obs_info.crystal_id] * obs_info.s_sq
        )

    def n_params(self) -> int:
        return 2 * self._n_crystals

    def summary(self) -> str:
        if self.log_G is None:
            return "ScaleB: not yet fit"
        return (
            f"ScaleB: {self._n_crystals} crystals, "
            f"med |log G|={self.log_G.abs().median():.3f}, "
            f"med |B|={self.B.abs().median():.1f} Å²"
        )


# ── Panel gain correction ──────────────────────────────────


class PanelCorrection(Correction):
    """Per-panel detector gain offset (+ optional linear gradient).

    Model (constant only)::

        c_j = g_p            (1 param per panel)

    Model (with gradient)::

        c_j = g_p + a_p·fs_j + b_p·ss_j   (3 params per panel)

    Only **32 (or 96)** global parameters for the full dataset.
    """

    def __init__(self, fit_gradient: bool = False):
        self.fit_gradient = fit_gradient
        self.gain: Tensor | None = None        # (n_panels,)
        self.grad_fs: Tensor | None = None     # (n_panels,) if fit_gradient
        self.grad_ss: Tensor | None = None     # (n_panels,) if fit_gradient
        self._n_panels = 0

    def fit(
        self,
        residuals: Tensor,
        obs_info: ObsInfo,
        work_mask: Tensor,
    ) -> None:
        assert obs_info.panel_id is not None, "PanelCorrection requires panel_id"
        pid = obs_info.panel_id
        n_panels = int(pid.max().item()) + 1
        self._n_panels = n_panels
        w = work_mask.float()

        _dev = pid.device
        cnt = torch.zeros(n_panels, device=_dev).scatter_add_(0, pid, w)
        sum_r = torch.zeros(n_panels, device=_dev).scatter_add_(0, pid, residuals * w)

        safe_cnt = cnt.clamp(min=1)
        self.gain = sum_r / safe_cnt

        if self.fit_gradient and obs_info.fs is not None and obs_info.ss is not None:
            fs, ss = obs_info.fs, obs_info.ss
            # Per-panel means
            sum_fs = torch.zeros(n_panels, device=_dev).scatter_add_(0, pid, fs * w)
            sum_ss = torch.zeros(n_panels, device=_dev).scatter_add_(0, pid, ss * w)
            mean_fs = sum_fs / safe_cnt
            mean_ss = sum_ss / safe_cnt

            # Center features
            dfs = fs - mean_fs[pid]
            dss = ss - mean_ss[pid]

            # Residuals after constant offset
            r2 = residuals - self.gain[pid]

            # Per-panel OLS for gradient terms
            sum_r_dfs = torch.zeros(n_panels, device=_dev).scatter_add_(0, pid, r2 * dfs * w)
            sum_r_dss = torch.zeros(n_panels, device=_dev).scatter_add_(0, pid, r2 * dss * w)
            sum_dfs2 = torch.zeros(n_panels, device=_dev).scatter_add_(0, pid, (dfs ** 2) * w)
            sum_dss2 = torch.zeros(n_panels, device=_dev).scatter_add_(0, pid, (dss ** 2) * w)

            self.grad_fs = sum_r_dfs / sum_dfs2.clamp(min=1e-12)
            self.grad_ss = sum_r_dss / sum_dss2.clamp(min=1e-12)
        else:
            self.grad_fs = None
            self.grad_ss = None

        # Zero panels with no observations
        no_obs = cnt < 0.5
        self.gain[no_obs] = 0.0
        if self.grad_fs is not None:
            self.grad_fs[no_obs] = 0.0
            self.grad_ss[no_obs] = 0.0

    def predict(self, obs_info: ObsInfo) -> Tensor:
        if self.gain is None:
            return torch.zeros(obs_info.crystal_id.shape[0], device=obs_info.crystal_id.device)
        pid = obs_info.panel_id
        pred = self.gain[pid]
        if self.grad_fs is not None and obs_info.fs is not None:
            pred = pred + self.grad_fs[pid] * obs_info.fs
            pred = pred + self.grad_ss[pid] * obs_info.ss
        return pred

    def n_params(self) -> int:
        return self._n_panels * (3 if self.fit_gradient else 1)

    def summary(self) -> str:
        if self.gain is None:
            return "Panel: not yet fit"
        g = self.gain
        s = f"Panel: {self._n_panels} panels, gain range [{g.min():.4f}, {g.max():.4f}]"
        if self.grad_fs is not None:
            s += f", grad_fs range [{self.grad_fs.min():.2e}, {self.grad_fs.max():.2e}]"
        return s


# ── Correction pipeline ─────────────────────────────────────


@dataclass
class FreeSetStats:
    """Monitoring stats from the free-set comparison."""

    alpha_work: float    # median α from work-set residuals
    alpha_free: float    # median α from free-set residuals
    ratio: float         # alpha_free / alpha_work (>1 → overfitting)
    n_work: int
    n_free: int


class CorrectionPipeline:
    """Manages a list of corrections with work/free split.

    Parameters
    ----------
    corrections : sequence of Correction instances (applied in order).
    free_fraction : fraction of observations held out for monitoring.
    seed : random seed for reproducible work/free split.
    """

    def __init__(
        self,
        corrections: Sequence[Correction],
        free_fraction: float = 0.1,
        seed: int = 42,
    ):
        self.corrections = list(corrections)
        self.free_fraction = free_fraction
        self.seed = seed
        self._work_mask: Tensor | None = None
        self._free_mask: Tensor | None = None

    def init_masks(self, n_obs: int, device: str | torch.device = "cpu") -> None:
        """Create work/free masks (call once before EM loop)."""
        gen = torch.Generator().manual_seed(self.seed)
        self._free_mask = (torch.rand(n_obs, generator=gen) < self.free_fraction).to(device)
        self._work_mask = ~self._free_mask

    @property
    def work_mask(self) -> Tensor:
        assert self._work_mask is not None, "call init_masks() first"
        return self._work_mask

    @property
    def free_mask(self) -> Tensor:
        assert self._free_mask is not None, "call init_masks() first"
        return self._free_mask

    def fit_and_predict(
        self,
        residuals: Tensor,
        obs_info: ObsInfo,
    ) -> Tensor:
        """Fit all corrections sequentially on work set, return total correction.

        Each correction sees residuals with earlier corrections already
        subtracted, so they are fit on the *remaining* residual.

        Returns
        -------
        total_correction : (N,) tensor to subtract from log(I_obs).
        """
        total = torch.zeros_like(residuals)
        remaining = residuals.clone()

        for corr in self.corrections:
            corr.fit(remaining, obs_info, self.work_mask)
            pred = corr.predict(obs_info)
            total += pred
            remaining = remaining - pred

        return total

    def predict(self, obs_info: ObsInfo) -> Tensor:
        """Apply all corrections (no fitting)."""
        total = torch.zeros(obs_info.crystal_id.shape[0], device=obs_info.crystal_id.device)
        for corr in self.corrections:
            total += corr.predict(obs_info)
        return total

    def monitor(
        self,
        log_I: Tensor,
        mu_expanded: Tensor,
        total_correction: Tensor,
        obs_counts: Tensor,
        beta_per_hkl: Tensor,
    ) -> FreeSetStats:
        """Compare work vs free residual distributions.

        Computes per-HKL α separately for work and free observations,
        returns median ratio.  A ratio >> 1 indicates overfitting.
        """
        corrected = log_I - total_correction
        residuals = (corrected - mu_expanded).abs().clamp(min=1e-8)
        beta_exp = beta_per_hkl.repeat_interleave(obs_counts)

        # Per-HKL α for work and free subsets
        hkl_idx = torch.arange(len(obs_counts)).repeat_interleave(obs_counts)
        n_hkl = len(obs_counts)

        def _alpha_subset(mask: Tensor) -> Tensor:
            r_masked = residuals * mask.float()
            r_beta = r_masked.pow(beta_exp) * mask.float()
            sum_rb = torch.zeros(n_hkl, device=hkl_idx.device).scatter_add_(0, hkl_idx, r_beta)
            cnt = torch.zeros(n_hkl, device=hkl_idx.device).scatter_add_(0, hkl_idx, mask.float())
            cnt = cnt.clamp(min=1)
            beta_hkl = beta_per_hkl
            return (beta_hkl / cnt * sum_rb).pow(1.0 / beta_hkl).clamp(min=0.01)

        alpha_w = _alpha_subset(self.work_mask)
        alpha_f = _alpha_subset(self.free_mask)

        # Only consider HKLs with enough free-set observations
        n_free_per_hkl = torch.zeros(n_hkl, device=hkl_idx.device).scatter_add_(
            0, hkl_idx, self.free_mask.float()
        )
        valid = n_free_per_hkl >= 3
        med_w = alpha_w[valid].median().item() if valid.any() else 0.0
        med_f = alpha_f[valid].median().item() if valid.any() else 0.0
        ratio = med_f / max(med_w, 1e-8)

        return FreeSetStats(
            alpha_work=med_w,
            alpha_free=med_f,
            ratio=ratio,
            n_work=int(self.work_mask.sum()),
            n_free=int(self.free_mask.sum()),
        )

    def summary(self) -> str:
        lines = [f"CorrectionPipeline ({len(self.corrections)} corrections):"]
        for c in self.corrections:
            lines.append(f"  {c.summary()}")
        return "\n".join(lines)
