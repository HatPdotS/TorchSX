"""Collapse-free scale-consistency loss for neural reflection scaling.

This is the training objective for the (future) network that treats each frame
as a context and each reflection as a token and predicts a **distribution over
the per-observation scale**.  It is *reference-free*: there is no externally
merged intensity to regress against (the careless approach).  Instead, one
latent intensity ``J_h`` per unique reflection couples all of that reflection's
observations, and each observation is scored by its **marginal predictive
density** under the network's scale distribution and the (fixed, calibrated)
detector-noise distribution::

    log_mean_k   = log_I0[h] + log_scale_mean_k                # J_h · predicted scale
    p(I_k | J_h) = ∫ N( I_k ; S , σ²_det,k ) · LogNormal( S ; log_mean_k , σ_scale_k² ) dS
    NLL          = − Σ_k log p(I_k | J_h)  +  prior(log_I0)  [+ scale-mean centering]

Summing ``−log p`` over a shared ``J_h`` *is* the all-vs-all overlap between the
observations of HKL ``h`` — expressed as an O(n) shared-latent marginal
likelihood rather than O(n²) pairs — and, crucially, as a **proper scoring
rule**.  That is what makes it collapse-free:

* **Uncertainty inflation** is self-penalising — a normalised predictive density
  drops when ``σ_scale`` is broader than the data warrant, and the fixed
  detector σ is a hard floor it cannot undercut.
* **The global scale→0 / J→∞ gauge** is pinned by the Wilson ``prior`` on
  ``log_I0`` (and the optional ``log_scale_mean`` centering term).

The remaining degeneracies are *information leaks* that live in the **feature
contract of the network feeding this loss**, not in the loss itself (see
:mod:`torchsx.nn.features`):

* the scale head must never see an observation's own ``I_k`` (else
  ``s_k = I_k/J_h`` fits perfectly and ``J_h`` goes free), and
* the discrete ``(h,k,l)`` may be fed to the net only under **multi-system**
  training (a shared net cannot memorise ``(h,k,l)→intensity`` across systems,
  so per-HKL signal is forced into the per-system ``log_I0``).

The integral is a single 1-D quadrature over ``log S`` — the same form as the
engine's :meth:`FixedLogGrid.marginalize_signal`, with the per-HKL GenGauss
``(α, β)`` replaced by the network's **per-observation** width.  A
generalised-Gaussian with ``β = 2`` is exactly a Gaussian in log-space, i.e. a
log-normal scale distribution, so we reuse
:func:`torchsx.merging.scatter.gengauss.gengauss_log_prob_from_obs` for the
scatter kernel.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from torchsx.merging.scatter.gengauss import gengauss_log_prob_from_obs

# log-normal scale ⇒ GenGauss with β=2.  Match exp(−(ε/α)²) to N(0,σ²) ∝
# exp(−ε²/2σ²): α² = 2σ² ⇒ log α = log σ + ½·log 2.  With this α the GenGauss
# normaliser −½logπ − logα equals the Gaussian −½log(2π) − logσ exactly.
_HALF_LOG2 = 0.5 * math.log(2.0)
_LOG2 = math.log(2.0)
_LOG_2PI = math.log(2.0 * math.pi)


class ScaleConsistencyNLL(nn.Module):
    """Shared-latent marginal NLL over per-observation scale distributions.

    Holds the per-HKL latent intensities ``log_I0`` as an ``nn.Parameter`` so
    they are optimised jointly with the (future) network weights — *end-to-end
    joint* training.  For **multi-system** training, ``log_I0`` is keyed per
    ``(system, hkl)``: concatenate the systems' valid-HKL tables into one flat
    index space and pass the matching global ``hkl_id`` to :meth:`forward`.

    Parameters
    ----------
    n_hkl
        Number of (global) unique reflections — the length of ``log_I0``.
    mu_init
        Optional ``(n_hkl,)`` seed for ``log_I0`` (e.g. each system's
        ``PreparedObs.mu_init``); ``None`` → zeros.
    prior
        Optional intensity prior over ``log_I0`` — an
        :class:`~torchsx.merging.priors.base.IntensityPrior` whose ``forward``
        returns a per-HKL negative-log-prior vector.  Use
        :class:`~torchsx.merging.priors.wilson.WilsonPrior` to pin the absolute
        scale gauge.  ``None`` → no prior (the gauge is then unconstrained; only
        for tests that supply their own anchoring).
    center_scale_mean
        If ``True``, add ``lambda_center · mean_h(⟨log_scale_mean⟩_h)²`` style
        penalty pulling the per-HKL mean of ``log_scale_mean`` toward zero — a
        soft gauge stabiliser that keeps per-HKL signal in ``log_I0`` rather
        than the scale.  Off by default (the Wilson prior is the primary anchor).
    lambda_center
        Strength of the centering penalty (only used when
        ``center_scale_mean=True``).
    S_min, S_max, n_points
        Linear-S range and node count of the fixed ``log S`` quadrature grid,
        mirroring :class:`~torchsx.merging.quadrature.fixed_log_grid.FixedLogGrid`.
    chunk_size
        Max observations per quadrature chunk, to bound the ``(M, K)`` grid
        tensor.  The loss is summed across chunks (exact, not an approximation).
    var_floor
        Lower clamp on the detector variance ``σ²_det`` for numerical safety.
    """

    def __init__(
        self,
        n_hkl: int,
        mu_init: Tensor | None = None,
        *,
        prior: nn.Module | None = None,
        center_scale_mean: bool = False,
        lambda_center: float = 0.0,
        S_min: float = 1e-3,
        S_max: float = 1e6,
        n_points: int = 2001,
        chunk_size: int = 200_000,
        var_floor: float = 1e-12,
    ) -> None:
        super().__init__()
        if not (S_min > 0.0 and S_max > S_min):
            raise ValueError(f"need 0 < S_min < S_max, got {S_min=} {S_max=}")
        if mu_init is None:
            log_I0 = torch.zeros(n_hkl)
        else:
            if mu_init.shape != (n_hkl,):
                raise ValueError(
                    f"mu_init must have shape ({n_hkl},), got {tuple(mu_init.shape)}"
                )
            log_I0 = mu_init.detach().clone().float()
        self.log_I0 = nn.Parameter(log_I0)
        self.prior = prior
        self.center_scale_mean = bool(center_scale_mean)
        self.lambda_center = float(lambda_center)
        self.n_points = int(n_points)
        self.chunk_size = int(chunk_size)
        self.var_floor = float(var_floor)

        log_S = torch.linspace(math.log(S_min), math.log(S_max), self.n_points)
        self.register_buffer("log_S_grid", log_S)
        self.dlog_S = float((math.log(S_max) - math.log(S_min)) / (self.n_points - 1))

    # ── quadrature ──────────────────────────────────────────────────────

    def _trap_log_w(self, ref: Tensor) -> Tensor:
        """Trapezoidal log-weights with half-weight endpoints (matches FixedLogGrid)."""
        log_w = ref.new_full((self.n_points,), math.log(self.dlog_S))
        log_w[0] = log_w[0] - _LOG2
        log_w[-1] = log_w[-1] - _LOG2
        return log_w

    def _marginal_log_p_chunk(
        self,
        log_mean: Tensor,            # (m,)
        log_scale_logsigma: Tensor,  # (m,)  log σ_scale
        I_obs: Tensor,               # (m,)
        sigma_det: Tensor,           # (m,)
    ) -> Tensor:                     # (m,)
        """``log ∫ N(I; S, σ²_det) · LogNormal(S; log_mean, σ_scale²) dS`` per obs."""
        log_S = self.log_S_grid                                    # (K,)
        S = log_S.exp()                                            # (K,)
        eps = log_S[None, :] - log_mean[:, None]                   # (m, K)

        # Scale distribution: Gaussian-in-log ⇒ GenGauss(α = σ√2, β = 2).
        log_alpha = log_scale_logsigma + _HALF_LOG2                # (m,)
        log_beta = log_scale_logsigma.new_full(log_scale_logsigma.shape, _LOG2)
        log_scatter = gengauss_log_prob_from_obs(eps, log_alpha, log_beta)  # (m, K)

        # Detector likelihood: N(I_obs; S, σ²_det), σ_det fixed (per-obs input).
        var = (sigma_det * sigma_det).clamp(min=self.var_floor)    # (m,)
        resid = I_obs[:, None] - S[None, :]                        # (m, K)
        log_lik = -0.5 * (
            _LOG_2PI + var.log()[:, None] + resid * resid / var[:, None]
        )                                                          # (m, K)

        log_w = self._trap_log_w(log_scatter)                      # (K,)
        return torch.logsumexp(log_scatter + log_lik + log_w[None, :], dim=1)  # (m,)

    def data_nll_per_obs(
        self,
        log_scale_mean: Tensor,      # (M,)
        log_scale_logsigma: Tensor,  # (M,)
        I_obs: Tensor,               # (M,)
        sigma_det: Tensor,           # (M,)
        hkl_id: Tensor,              # (M,) int64, index into log_I0
    ) -> Tensor:                     # (M,)
        """Per-observation marginal NLL (no prior, no reduction)."""
        log_mean_full = self.log_I0[hkl_id] + log_scale_mean
        M = I_obs.shape[0]
        if M <= self.chunk_size:
            return -self._marginal_log_p_chunk(
                log_mean_full, log_scale_logsigma, I_obs, sigma_det
            )
        out = I_obs.new_empty(M)
        for s in range(0, M, self.chunk_size):
            e = min(s + self.chunk_size, M)
            out[s:e] = -self._marginal_log_p_chunk(
                log_mean_full[s:e], log_scale_logsigma[s:e], I_obs[s:e], sigma_det[s:e]
            )
        return out

    # ── objective ───────────────────────────────────────────────────────

    def forward(
        self,
        log_scale_mean: Tensor,      # (M,) network output
        log_scale_logsigma: Tensor,  # (M,) network output: log σ_scale
        I_obs: Tensor,               # (M,)
        sigma_det: Tensor,           # (M,) calibrated detector noise (fixed)
        hkl_id: Tensor,              # (M,) int64, global (system,hkl) index
    ) -> Tensor:
        """Total scalar loss: data NLL + prior [+ scale-mean centering]."""
        loss = self.data_nll_per_obs(
            log_scale_mean, log_scale_logsigma, I_obs, sigma_det, hkl_id
        ).sum()

        if self.prior is not None:
            loss = loss + self.prior(self.log_I0).sum()

        if self.center_scale_mean and self.lambda_center > 0.0:
            # Pull each HKL's mean predicted log-scale toward 0 (gauge stabiliser).
            n_hkl = self.log_I0.shape[0]
            sums = torch.zeros(n_hkl, dtype=log_scale_mean.dtype, device=log_scale_mean.device)
            sums = sums.scatter_add(0, hkl_id, log_scale_mean)
            cnt = torch.bincount(hkl_id, minlength=n_hkl).to(sums.dtype).clamp(min=1.0)
            per_hkl_mean = sums / cnt
            loss = loss + self.lambda_center * per_hkl_mean.pow(2).sum()

        return loss

    # ── posterior readout ───────────────────────────────────────────────

    @torch.enable_grad()
    def posterior(
        self,
        log_scale_mean: Tensor,
        log_scale_logsigma: Tensor,
        I_obs: Tensor,
        sigma_det: Tensor,
        hkl_id: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Merged intensity and its **effective σ** per HKL (Laplace readout).

        The merged intensity is ``I0_h = exp(log_I0_h)``.  The effective σ is a
        Laplace approximation: the data NLL's curvature w.r.t. each ``log_I0_h``
        gives ``σ_logI0,h = 1/√H_hh``, propagated to intensity by
        ``σ_eff,h = I0_h · σ_logI0,h``.  This is the "effective σ from the
        integration" — it folds the per-observation scale spread and detector
        noise into the posterior width of the shared latent, and shrinks as a
        reflection accrues more (and more consistent) observations.

        The data term's Hessian in ``log_I0`` is diagonal (each observation of
        HKL ``h`` depends only on ``log_I0_h``), so the full diagonal comes from
        two backward passes — exact for the data term (the prior's weak
        cross-shell coupling is omitted from the curvature).
        """
        log_I0 = self.log_I0
        had_grad = log_I0.requires_grad
        log_I0.requires_grad_(True)
        data_nll = self.data_nll_per_obs(
            log_scale_mean, log_scale_logsigma, I_obs, sigma_det, hkl_id
        ).sum()
        (g,) = torch.autograd.grad(data_nll, log_I0, create_graph=True)
        (h_diag,) = torch.autograd.grad(g.sum(), log_I0)
        log_I0.requires_grad_(had_grad)

        I0 = log_I0.detach().exp()
        sigma_log = h_diag.detach().clamp(min=1e-12).rsqrt()
        sigma_eff = I0 * sigma_log
        return I0, sigma_eff
