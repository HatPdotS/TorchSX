"""Precomputed-grid quadrature — *for sub-photon (negatives-Mandel) data only*.

Tabulates the log marginal of ``∫ Poisson(N | M·e^u + bg) · GG(u; α, β) du``
on a 5D grid in ``(log α, β, log μ, log λ, log f)`` where:

    λ      = N + bg + ε_anchor
    f      = bg / λ                    (∈ [F_FLOOR, 1])
    log μ  = log_mean − log λ          (Poisson-anchored offset)
    log α  = log of GenGauss scale α
    β      = GenGauss shape

Stores ``log p`` directly (no analytic prefactor split — the
prefactor-strip approach has fatal cell-blend errors for data with
wide ``log λ`` range).

**Regime of applicability.**  The integrand has ``∂log_p / ∂log_λ ≈ N``,
so cell-to-cell variation grows as ``O(N · h)``.  Multilinear (or
cubic) interpolation can only resolve a function whose cell-to-cell
variation is small relative to the cell size.  Concretely:

* ``n_ph ≲ 1`` (negatives-Mandel sub-photon regime, e.g. the SFX
  streams in ``experiments/data/`` at native gain): table works,
  median 2.5e-3 nat / max 5.6e-2 nat error vs brute force.
* ``n_ph ≫ 1`` (high-flux photon-counting data, e.g. BR at gain
  ≈ 30 ADU/photon with ``n_ph`` up to 10⁵): cell variations of
  10⁴ nat per cell — table cannot represent the function regardless
  of resolution.  Use ``HybridLogGrid``, ``LaplaceSaddleGrid``, or
  ``DeltaAtESQuadrature`` instead.

The ``from_prep`` constructor below raises if ``prep`` looks
photon-counting (max ``n_ph + bg`` exceeds a threshold).

Per-obs forward cost (when applicable): 32-corner gather + 5 mul-adds.
At production batch sizes (≥ 256 k obs) this is ~4× faster than
``HybridLogGrid`` while matching its accuracy on median/q90/q99.

See ``experiments/Precompute_grid/FINDINGS.md`` for the sizing study
and the BR-divergence post-mortem.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from torchsx.merging.config import MergeConfig
from torchsx.merging.likelihood.base import DetectionLikelihood
from torchsx.merging.observations import ObsInfo, PreparedObs
from torchsx.merging.parameters import Parameters
from torchsx.merging.quadrature.base import Quadrature
from torchsx.merging.scatter.base import ScatterModel


EPS_ANCHOR = 1e-3
F_FLOOR = 1e-4


# ── Grid spec ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PrecomputedGridSpec:
    """Axes of the 5D table.

    Defaults match the validated production recipe (`bench_axis_isolate.py`):
    median 2.5e-3 nat, q99 8.0e-3 nat, max 5.6e-2 nat at 12 MB float32.
    """
    n_alpha: int = 12
    alpha_lo: float = 0.05
    alpha_hi: float = 8.0
    n_beta: int = 10
    beta_lo: float = 0.5
    beta_hi: float = 4.0
    n_logmu: int = 32
    logmu_lo: float = -20.0
    logmu_hi: float = +5.0
    n_loglam: int = 48              # ← critical, see FINDINGS.md
    loglam_lo: float = -10.0
    loglam_hi: float = +5.0
    n_logf: int = 16
    logf_lo: float = math.log(F_FLOOR)
    logf_hi: float = 0.0

    def alpha_axis(self) -> Tensor:
        return torch.logspace(
            math.log10(self.alpha_lo), math.log10(self.alpha_hi),
            self.n_alpha, dtype=torch.float64,
        )

    def beta_axis(self) -> Tensor:
        return torch.linspace(self.beta_lo, self.beta_hi, self.n_beta,
                              dtype=torch.float64)

    def logmu_axis(self) -> Tensor:
        return torch.linspace(self.logmu_lo, self.logmu_hi, self.n_logmu,
                              dtype=torch.float64)

    def loglam_axis(self) -> Tensor:
        return torch.linspace(self.loglam_lo, self.loglam_hi, self.n_loglam,
                              dtype=torch.float64)

    def logf_axis(self) -> Tensor:
        return torch.linspace(self.logf_lo, self.logf_hi, self.n_logf,
                              dtype=torch.float64)

    def total_cells(self) -> int:
        return (self.n_alpha * self.n_beta * self.n_logmu *
                self.n_loglam * self.n_logf)


# ── Reference 1D quadrature (used to fill the table at build time) ───


def _gg_quad_range(beta: Tensor, p_tail: float = 1e-12) -> Tensor:
    return (-math.log(p_tail)) ** (1.0 / beta.clamp(min=0.1))


def _log_gg(eps: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
    z = eps.abs() / alpha.clamp(min=1e-8)
    log_norm = (
        torch.log(beta) - math.log(2.0)
        - torch.log(alpha) - torch.lgamma(1.0 / beta)
    )
    return -z.clamp(min=1e-12).pow(beta) + log_norm


def _log_poisson(N: Tensor, lam: Tensor) -> Tensor:
    lam = lam.clamp(min=1e-30)
    return N * torch.log(lam) - lam - torch.lgamma(N + 1.0)


_EPS_ABS_MAX = 50.0   # log_S window ceiling — exp(50) ≈ 5e21 < float64 max


def _log_marginal_ref(
    N: Tensor, bg: Tensor, log_mean: Tensor,
    alpha: Tensor, beta: Tensor,
    n_eps: int = 2001, eps_max_mult: float = 1.5,
) -> Tensor:
    """Per-obs quadrature with a SADDLE-POINT-CENTERED ε grid.

    For high-N obs the Poisson factor in the integrand has width
    ``σ_ε ≈ 1/√(N − bg)`` which can be 1000× tighter than the GG mode
    width ``α``.  A uniform trapezoid on the union window
    drastically under-resolves the Poisson peak, biasing the integral
    by tens of nats on BR-like data.

    Strategy: split the n_eps nodes between two zones:
      * a "Poisson zone" of ±n_sigma_pois·σ_ε around ``ε* = log((N-bg)/M)``
        — covers the data peak with adequate sub-σ resolution
      * a "GG zone" of ±α·R(β) around ε=0 — covers the prior peak
    Trapezoid weights come from neighbour spacing in the sorted union
    of the two grids (same trick HybridLogGrid uses for its data+prior
    union).  When N ≤ bg the Poisson factor has no peak in ε and we
    fall back to the GG-only grid.
    """
    M = torch.exp(log_mean)
    safe_pos = (N - bg).clamp(min=1e-30)
    eps_star = torch.log(safe_pos / M.clamp(min=1e-30))
    has_pois = N > bg

    # Poisson width in ε: σ_ε = √Var[Poisson]/peak = 1/√(N-bg) (Stirling).
    sigma_eps = (safe_pos.clamp(min=1.0)).rsqrt()
    n_sigma_pois = 8.0
    pois_lo = torch.where(has_pois,
                          (eps_star - n_sigma_pois * sigma_eps).clamp(min=-_EPS_ABS_MAX),
                          torch.full_like(eps_star, _EPS_ABS_MAX))
    pois_hi = torch.where(has_pois,
                          (eps_star + n_sigma_pois * sigma_eps).clamp(max=_EPS_ABS_MAX),
                          torch.full_like(eps_star, -_EPS_ABS_MAX))

    # GG zone — wide, covers the prior peak.
    R = _gg_quad_range(beta) * float(eps_max_mult)
    gg_hi = (alpha * R).clamp(max=_EPS_ABS_MAX)
    gg_lo = (-alpha * R).clamp(min=-_EPS_ABS_MAX)

    # Allocate n_eps total nodes between the two zones.  Half each by
    # default, but if no Poisson peak give all to GG.
    n_pois = n_eps // 2
    n_gg = n_eps - n_pois

    frac_p = torch.linspace(0.0, 1.0, n_pois, device=N.device, dtype=log_mean.dtype)
    frac_g = torch.linspace(0.0, 1.0, n_gg, device=N.device, dtype=log_mean.dtype)

    # (M, n_pois) and (M, n_gg) per-obs grids.
    eps_pois = pois_lo[:, None] + (pois_hi - pois_lo)[:, None] * frac_p[None, :]
    eps_gg = gg_lo[:, None] + (gg_hi - gg_lo)[:, None] * frac_g[None, :]

    # Sorted union; neighbour-spacing trapezoid weights.
    eps_all = torch.cat([eps_pois, eps_gg], dim=1)
    eps_sorted, _ = eps_all.sort(dim=1)
    diffs = eps_sorted[:, 1:] - eps_sorted[:, :-1]
    w = torch.zeros_like(eps_sorted)
    w[:, 0] = diffs[:, 0] * 0.5
    w[:, -1] = diffs[:, -1] * 0.5
    w[:, 1:-1] = (diffs[:, :-1] + diffs[:, 1:]) * 0.5
    log_w = torch.log(w.clamp(min=1e-30))

    log_gg_v = _log_gg(eps_sorted, alpha[:, None], beta[:, None])
    lam = M[:, None] * torch.exp(eps_sorted) + bg[:, None]
    log_poi = _log_poisson(N[:, None], lam)
    return torch.logsumexp(log_gg_v + log_poi + log_w, dim=1)


def _analytic_prefactor(N: Tensor, bg: Tensor) -> Tensor:
    return N * (N + bg + EPS_ANCHOR).log() - torch.lgamma(N + 1.0)


def build_table(
    spec: PrecomputedGridSpec,
    *,
    device: torch.device | str = "cpu",
    n_eps: int = 2001,
    chunk: int = 100_000,
) -> Tensor:
    """Build the 5D ``log p`` table.

    Stores ``log p`` directly — no analytic prefactor split.  The
    prefactor-strip approach (storing ``log p − N·log λ +
    lgamma(N+1)``) made interpolation inconsistent across cells with
    different ``log_lam``: at lookup the obs's ``log_lam`` is used to
    add the prefactor back, but interpolated ``log_z`` blends cell
    values whose prefactors were *different*, producing 30+ nat
    blend errors on data with wide ``log_lam`` range.

    Returns a float64 tensor of shape ``(n_alpha, n_beta, n_logmu,
    n_loglam, n_logf)``.  Cast to float32 before registering as a buffer.
    """
    dev = torch.device(device)
    a = spec.alpha_axis().to(dev)
    b = spec.beta_axis().to(dev)
    lmu = spec.logmu_axis().to(dev)
    llam = spec.loglam_axis().to(dev)
    lf = spec.logf_axis().to(dev)

    Aa, Bb, LMU, LLAM, LF = torch.meshgrid(a, b, lmu, llam, lf, indexing="ij")
    shape = Aa.shape

    Aaf, Bbf, LMUf, LLAMf, LFf = (t.flatten() for t in (Aa, Bb, LMU, LLAM, LF))
    f = LFf.exp().clamp(max=1.0)
    lam = LLAMf.exp()
    bg = (f * lam).clamp(min=0.0)
    N = (lam - bg).clamp(min=0.0)
    log_mean = LMUf + LLAMf

    n_total = Aaf.numel()
    out = torch.empty(n_total, device=dev, dtype=torch.float64)
    for s in range(0, n_total, chunk):
        e = min(s + chunk, n_total)
        out[s:e] = _log_marginal_ref(
            N[s:e], bg[s:e], log_mean[s:e],
            Aaf[s:e], Bbf[s:e], n_eps=n_eps,
        )
    return out.view(shape)


# ── Multilinear interpolation ────────────────────────────────────────


def _bracket(axis: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
    n = axis.numel()
    idx = torch.searchsorted(axis, x).clamp(min=1, max=n - 1) - 1
    lo = axis[idx]
    hi = axis[idx + 1]
    frac = ((x - lo) / (hi - lo)).clamp(min=0.0, max=1.0)
    return idx, frac


def _interp5d(table: Tensor, axes: tuple, pts: tuple) -> Tensor:
    """Multilinear interpolation — fast but C⁰ only (gradient discontinuous
    across cell boundaries).  Suitable for forward-only evaluation; for
    use inside an L-BFGS objective prefer ``_interp5d_cubic``."""
    assert table.ndim == len(axes) == len(pts) == 5
    M = pts[0].numel()
    device = table.device
    dtype = table.dtype
    idxs, fracs = [], []
    for ax, p in zip(axes, pts):
        i, ff = _bracket(ax, p)
        idxs.append(i)
        fracs.append(ff.to(dtype))
    out = torch.zeros(M, dtype=dtype, device=device)
    for corner in range(32):
        weight = torch.ones(M, dtype=dtype, device=device)
        gather_idx = []
        for d in range(5):
            bit = (corner >> d) & 1
            if bit:
                gather_idx.append(idxs[d] + 1)
                weight = weight * fracs[d]
            else:
                gather_idx.append(idxs[d])
                weight = weight * (1.0 - fracs[d])
        out = out + weight * table[tuple(gather_idx)]
    return out


def _catmull_rom_weights(t: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Catmull-Rom weights at fractional ``t`` ∈ [0, 1] within a cell.

    Returns (w_{-1}, w_0, w_1, w_2) for nodes at indices (i−1, i, i+1, i+2).
    The interp is C¹ continuous (so gradients across cell boundaries are
    smooth) and passes through the control points.
    """
    t2 = t * t
    t3 = t2 * t
    w_m1 = -0.5 * t3 + t2 - 0.5 * t
    w_0 = 1.5 * t3 - 2.5 * t2 + 1.0
    w_1 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
    w_2 = 0.5 * t3 - 0.5 * t2
    return w_m1, w_0, w_1, w_2


def _interp5d_cubic(table: Tensor, axes: tuple, pts: tuple) -> Tensor:
    """5D Catmull-Rom interpolation, fully vectorised via broadcast gather.

    Per obs we need 4^5 = 1024 corner values; we compute them in one big
    advanced-index gather (shape (M, 4, 4, 4, 4, 4)) then weight + sum.
    Memory per obs: 1024 * dtype.itemsize bytes (8 KB for fp64).  Chunk
    the obs axis to keep peak memory bounded.
    """
    assert table.ndim == len(axes) == len(pts) == 5
    M = pts[0].numel()
    device = table.device
    dtype = table.dtype
    sizes = table.shape

    # Per-axis: (M, 4) integer indices and (M, 4) Catmull-Rom weights.
    all_idxs: list[Tensor] = []
    all_weights: list[Tensor] = []
    for d, (ax, p) in enumerate(zip(axes, pts)):
        i, ff = _bracket(ax, p)
        i_stack = torch.stack(
            [(i - 1).clamp(min=0, max=sizes[d] - 1),
             i.clamp(min=0, max=sizes[d] - 1),
             (i + 1).clamp(min=0, max=sizes[d] - 1),
             (i + 2).clamp(min=0, max=sizes[d] - 1)],
            dim=-1,
        )                                                  # (M, 4)
        w0, w1, w2, w3 = _catmull_rom_weights(ff.to(dtype))
        w_stack = torch.stack([w0, w1, w2, w3], dim=-1)    # (M, 4)
        all_idxs.append(i_stack)
        all_weights.append(w_stack)

    # Broadcast-shape advanced indexing.  For each obs, gather shape
    # (4, 4, 4, 4, 4) at the table → flat (M, 4^5).
    chunk = 200_000
    out = torch.empty(M, dtype=dtype, device=device)
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        sub_M = e - s

        # Reshape each axis index to (sub_M, 4, 1, 1, 1, 1) etc. with
        # the right broadcast position.
        i0 = all_idxs[0][s:e, :, None, None, None, None]
        i1 = all_idxs[1][s:e, None, :, None, None, None]
        i2 = all_idxs[2][s:e, None, None, :, None, None]
        i3 = all_idxs[3][s:e, None, None, None, :, None]
        i4 = all_idxs[4][s:e, None, None, None, None, :]
        gathered = table[i0, i1, i2, i3, i4]   # (sub_M, 4, 4, 4, 4, 4)

        w0 = all_weights[0][s:e, :, None, None, None, None]
        w1 = all_weights[1][s:e, None, :, None, None, None]
        w2 = all_weights[2][s:e, None, None, :, None, None]
        w3 = all_weights[3][s:e, None, None, None, :, None]
        w4 = all_weights[4][s:e, None, None, None, None, :]

        out[s:e] = (gathered * w0 * w1 * w2 * w3 * w4).sum(dim=(1, 2, 3, 4, 5))
    return out


# ── Quadrature subclass ──────────────────────────────────────────────


class PrecomputedGridQuadrature(Quadrature):
    """5D multilinear-interp lookup of the precomputed log marginal.

    Parameters
    ----------
    spec
        ``PrecomputedGridSpec`` controlling axis ranges + node counts.
        Default = the validated production recipe (12 MB, max 5.6e-2 nat).
    table
        Optional precomputed 5D float32 buffer.  When ``None`` the table
        is built on the device of the surrounding module via
        ``build_table``.
    n_eps
        Reference resolution per cell at build time (default 2001 → ~1e-6
        nat self-consistency).
    eps_anchor
        Floor to avoid ``log(0)`` when ``N + bg = 0`` (default 1e-3).
    f_floor
        Lower clip on ``f = bg / λ`` at lookup (default 1e-4).
    """

    def __init__(
        self,
        spec: PrecomputedGridSpec | None = None,
        table: Tensor | None = None,
        n_eps: int = 2001,
        eps_anchor: float = EPS_ANCHOR,
        f_floor: float = F_FLOOR,
        interp: str = "cubic",
    ) -> None:
        super().__init__()
        self.spec = spec or PrecomputedGridSpec()
        self.eps_anchor = float(eps_anchor)
        self.f_floor = float(f_floor)
        self.n_eps = int(n_eps)
        if interp not in ("multilinear", "cubic"):
            raise ValueError(f"interp must be 'multilinear' or 'cubic', got {interp!r}")
        self.interp = interp

        if table is None:
            table = build_table(self.spec, device="cpu", n_eps=n_eps)
        if table.dtype != torch.float32:
            table = table.to(torch.float32)

        self.register_buffer("table", table)
        self.register_buffer("ax_alpha",  self.spec.alpha_axis())
        self.register_buffer("ax_beta",   self.spec.beta_axis())
        self.register_buffer("ax_logmu",  self.spec.logmu_axis())
        self.register_buffer("ax_loglam", self.spec.loglam_axis())
        self.register_buffer("ax_logf",   self.spec.logf_axis())

    @classmethod
    def from_prep(
        cls, prep: PreparedObs, config: MergeConfig,
    ) -> "PrecomputedGridQuadrature":
        """Auto-size the grid axes from ``prep`` so the table covers the
        actual stream's dynamic range with a margin for LBFGS line-search.

        Raises ``ValueError`` if ``prep`` is photon-counting data
        (``max n_ph > n_ph_max``) — the precomputed-grid approach is
        not accurate in that regime.  See the module docstring.
        """
        n_ph_max = float(
            (prep.I_obs + prep.bg).clamp(min=0).max().item()
        )
        n_ph_threshold = getattr(config, "precomp_max_n_ph", 50.0)
        if n_ph_max > n_ph_threshold:
            raise ValueError(
                f"PrecomputedGridQuadrature is designed for sub-photon "
                f"(negatives-Mandel) data; prep has max n_ph = "
                f"{n_ph_max:.1f} which exceeds the threshold "
                f"{n_ph_threshold}.  For high-flux photon-counting data "
                f"use HybridLogGrid, LaplaceSaddleGrid, or "
                f"DeltaAtESQuadrature.  To override (not recommended), "
                f"set ``MergeConfig.precomp_max_n_ph = inf``."
            )
        bg_safe = prep.bg.clamp(min=0.0)
        n_ph = (prep.I_obs + prep.bg).clamp(min=0)
        lam = (n_ph + bg_safe + EPS_ANCHOR).clamp(min=EPS_ANCHOR)
        log_lam = lam.log()
        loglam_lo = float(log_lam.min().item()) - 2.0
        loglam_hi = float(log_lam.max().item()) + 2.0

        log_mean = prep.mu_init[prep.hkl_id]      # init log_I0 at each obs
        log_mu = log_mean - log_lam
        logmu_lo = float(log_mu.min().item()) - 5.0       # LBFGS headroom
        logmu_hi = float(log_mu.max().item()) + 5.0

        # Adapt α and β to the per-HKL init values too — wide axis cells
        # at extreme α/β (where the GG-window-cap at ε_max biases log_z
        # upward) let cubic interp overshoot into pathological regions
        # during LBFGS line search.
        alpha_init = prep.alpha_init.clamp(min=0.05).double()
        beta_init = prep.beta_perhkl.clamp(min=0.5).double()
        a_lo = float(alpha_init.min().item()) * 0.5       # 2× headroom each side
        a_hi = float(alpha_init.max().item()) * 2.0
        b_lo = max(0.5, float(beta_init.min().item()) - 0.5)
        b_hi = float(beta_init.max().item()) + 1.0

        spec = PrecomputedGridSpec(
            alpha_lo=a_lo, alpha_hi=a_hi,
            beta_lo=b_lo, beta_hi=b_hi,
            loglam_lo=loglam_lo, loglam_hi=loglam_hi,
            logmu_lo=logmu_lo,   logmu_hi=logmu_hi,
        )
        return cls(
            spec=spec,
            n_eps=getattr(config, "precomp_n_eps", 2001),
            interp=getattr(config, "precomp_interp", "cubic"),
        )

    # ── Quadrature API ───────────────────────────────────────────────
    def marginalize_signal(
        self,
        scatter: ScatterModel,
        likelihood: DetectionLikelihood,
        log_mean: Tensor,
        n_ph: Tensor,
        bg: Tensor,
        params: Parameters,
        meta: ObsInfo,
        obs_range: tuple[int, int] | None = None,
    ) -> Tensor:
        self._reject_unmodelled_bg(likelihood, variance=True)
        h = meta.hkl_idx
        alpha = params.log_alpha[h].exp()
        beta = params.log_beta[h].exp()

        # Clamp inputs to the table support — out-of-range obs would
        # otherwise produce flat extrapolation.
        alpha = alpha.clamp(min=float(self.ax_alpha[0]),
                            max=float(self.ax_alpha[-1]))
        beta = beta.clamp(min=float(self.ax_beta[0]),
                          max=float(self.ax_beta[-1]))

        # Streams can carry negative ``bg`` (over-subtracted dark-current);
        # the Poisson likelihood handles those via internal lam-clamp, but
        # the table only covers ``bg ≥ 0``.  Map oob obs to the nearest
        # valid (bg, lam) by clamping bg≥0 and keeping λ above ``eps_anchor``.
        bg_safe = bg.clamp(min=0.0)
        lam_safe = (n_ph + bg_safe + self.eps_anchor).clamp(min=self.eps_anchor)
        log_lam_raw = lam_safe.log()
        f = (bg_safe / lam_safe).clamp(min=self.f_floor, max=1.0)
        log_f_raw = f.log()
        log_mu_raw = log_mean - log_lam_raw

        log_lam = log_lam_raw.clamp(min=float(self.ax_loglam[0]),
                                     max=float(self.ax_loglam[-1]))
        log_mu = log_mu_raw.clamp(min=float(self.ax_logmu[0]),
                                   max=float(self.ax_logmu[-1]))
        log_f = log_f_raw.clamp(min=float(self.ax_logf[0]),
                                 max=float(self.ax_logf[-1]))

        # Diagnose: how many obs got clamped?  Out-of-range obs use the
        # nearest valid table cell, so the prediction is biased; flag
        # this so callers can spot a misconfigured spec.
        if not getattr(self, "_clip_warned", False):
            n_clip = (
                (log_lam_raw != log_lam).sum().item()
                + (log_mu_raw != log_mu).sum().item()
                + (log_f_raw != log_f).sum().item()
            )
            if n_clip > 0:
                import warnings
                warnings.warn(
                    f"PrecomputedGridQuadrature: {n_clip} clamps on "
                    f"(log_lam, log_mu, log_f) — table spec doesn't cover "
                    f"the data fully.  Use ``from_prep(prep, config)`` "
                    f"to auto-size the axes.", RuntimeWarning,
                )
                self._clip_warned = True

        # Cast the table to log_mean's dtype so the interp accumulates
        # in the same precision the rest of the loss uses.
        dtype = log_mean.dtype
        pts = (alpha.to(dtype), beta.to(dtype),
               log_mu.to(dtype), log_lam.to(dtype), log_f.to(dtype))
        axes = (
            self.ax_alpha.to(dtype),  self.ax_beta.to(dtype),
            self.ax_logmu.to(dtype),  self.ax_loglam.to(dtype),
            self.ax_logf.to(dtype),
        )
        if self.interp == "cubic":
            log_p = _interp5d_cubic(self.table.to(dtype), axes, pts)
        else:
            log_p = _interp5d(self.table.to(dtype), axes, pts)
        return log_p
