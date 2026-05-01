"""Per-observation residuals to check the merging model fits as it should.

Three orthogonal residuals aligned with the generative model layers:

* **Pearson residual** ``(n - μ_tot) / √μ_tot`` with ``μ_tot = K·I + bg``.
  Tests the Poisson detection layer — ~N(0, σ²) with σ ≥ 1 if the scatter
  component is needed (i.e., σ > 1 means data has more dispersion than
  pure Poisson predicts).
* **Log-scatter** ``ε = log((n − bg) / (K·I))`` — only defined for obs
  where ``n > bg``.  Tests the multiplicative-scatter density
  (GenGauss / Gamma).  Should be distributed as the chosen scatter
  density per HKL.
* **Standardised log-scatter** ``z = ε / α_h``.  Aggregating across HKLs
  the empirical distribution should match the per-HKL scatter shape (β).

Plus a Bayes-factor diagnostic for the bg mixture::

    BF = log p(n | signal+bg) − log p(n | bg-only)

Should be large positive for bright-HKL obs (signal hypothesis wins) and
negative for dark-HKL obs (bg hypothesis wins).  A ~zero BF with no
clear separation means the per-obs evidence isn't pinning down which
regime the obs belongs to — indicates a noisy/borderline observation.

The full per-obs marginal log-likelihood and the post-mixture log p (i.e.
the per-obs contribution to the data NLL the LBFGS minimised) are saved
alongside the residuals so downstream plots don't have to reconstruct
them from the Bayes factor + bg-Poisson formula:

* ``log_p_sig_marginal`` = ``log ∫ p(n|S,bg) p_GG(log_S − log_mean) dlog_S``
  — the signal-hypothesis marginal; recoverable as ``log_bf + log_p_bg``
  but cached here for convenience.
* ``log_p`` = post bg-mixture combine.  Equals ``log_p_sig_marginal``
  for ``bg_mode="off"``; for the bg-mixture modes it's
  ``logaddexp(log(1−π) + log_p_sig, log(π) + log_p_bg)``.  Sum over obs
  reproduces ``final_data_nll`` up to fp32 jitter.

These quantities are aggregated per resolution shell and per intensity
decile so a plot can spot bias versus those covariates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from torchsx.merging.merger import Merger
    from torchsx.merging.result import MergedResult


@dataclass
class Residuals:
    """Per-observation + per-stratum residual summary."""

    # ── Per-obs (n_obs,) tensors ──
    pearson: Tensor              # (n - μ_tot) / √μ_tot
    log_scatter: Tensor          # log((n − bg) / (K · I))  (NaN if n ≤ bg)
    z_scatter: Tensor            # log_scatter / α_h         (NaN if undefined)
    log_bf: Tensor               # log p(signal+bg) − log p(bg-only)
    log_p_sig_marginal: Tensor   # log ∫ p(n|S,bg) · p_GG(log S − log_mean) dlog S
    log_p: Tensor                # post bg-mixture combine; per-obs contribution
                                 # to the data NLL.  For bg_mode="off" this
                                 # equals ``log_p_sig_marginal``; for the
                                 # bg-mixture modes it is
                                 # ``log_p_sig_marginal``-vs-``log_p_bg`` mixed.
    # ── Per-obs context (for grouping in plots) ──
    obs_hkl_id: Tensor           # which HKL each obs belongs to (n_obs,)
    obs_d_spacing: Tensor        # (n_obs,)
    obs_log_I_predicted: Tensor  # log(K · I)  (n_obs,)
    obs_n: Tensor                # raw photon counts
    obs_bg: Tensor               # bg estimate
    obs_mu_tot: Tensor           # K·I + bg
    # ── Per-shell aggregates ──
    shell_id: Tensor             # (n_obs,) — already in PreparedObs
    shell_summary: dict          # keyed by metric → (n_shells,) tensor


@torch.no_grad()
def compute_residuals(merger: "Merger", result: "MergedResult") -> Residuals:
    """Compute per-observation residuals using the converged merger state.

    Must be called *after* ``merger.merge()`` so ``merger.params``,
    ``merger.scale`` etc. carry LBFGS-converged values.
    """
    prep = merger.prep
    params = merger.params
    scale = merger.scale
    likelihood = merger.likelihood
    device = merger.device

    n_obs = prep.n_obs_total
    h = prep.hkl_id                                       # (n_obs,)
    n = (prep.I_obs + prep.bg).clamp(min=0)               # photon counts
    bg = prep.bg

    # ── per-obs predictions ──
    log_K = scale.log_k(params, prep.obs_info, obs_range=None)   # (n_obs,)
    log_I = params.log_I0[h]                                       # (n_obs,)
    log_mu_sig = log_K + log_I                                     # log(K·I)
    mu_sig = log_mu_sig.exp()
    mu_tot = mu_sig + bg                                           # E[n]

    # ── Pearson residuals ──
    pearson = (n - mu_tot) / mu_tot.clamp(min=1e-6).sqrt()

    # ── Log-scatter (only valid where n > bg) ──
    n_minus_bg = (n - bg)
    valid_log = n_minus_bg > 0
    log_scatter = torch.full_like(n, float("nan"))
    log_scatter[valid_log] = (
        torch.log(n_minus_bg[valid_log]) - log_mu_sig[valid_log]
    )

    # Standardised by per-HKL α
    alpha_h = params.log_alpha[h].exp()
    z_scatter = log_scatter / alpha_h.clamp(min=1e-6)

    # ── Bayes factor for signal hypothesis vs bg-only ──
    # log p(n|bg) = n log bg − bg − log Γ(n+1)
    log_p_bg = (
        n * torch.log(bg.clamp(min=1e-12))
        - bg.clamp(min=1e-12)
        - torch.lgamma(n + 1)
    )
    # Proper marginal log p(n | signal+bg) — integrate over the scatter
    # latent S using the same quadrature LBFGS used.  Chunked to keep
    # memory bounded.  We also fold the bg-mixture combine pass through
    # ``likelihood.combine_with_marginal`` so we get the full per-obs
    # NLL contribution (= -log_p) the LBFGS objective minimised, not
    # just the signal-only marginal.
    log_p_sig_marginal = torch.empty_like(n)
    log_p_total = torch.empty_like(n)
    chunk = 1_000_000
    for s_lo in range(0, n_obs, chunk):
        s_hi = min(s_lo + chunk, n_obs)
        oi_slice = prep.obs_info.slice(s_lo, s_hi)
        log_p_sig_chunk = merger.quadrature.marginalize_signal(
            merger.scatter, likelihood,
            log_mu_sig[s_lo:s_hi],
            n[s_lo:s_hi], bg[s_lo:s_hi],
            params, oi_slice,
            obs_range=(s_lo, s_hi),
        )
        log_p_sig_marginal[s_lo:s_hi] = log_p_sig_chunk
        log_p_total[s_lo:s_hi] = likelihood.combine_with_marginal(
            n[s_lo:s_hi], bg[s_lo:s_hi], log_p_sig_chunk, params, oi_slice,
        )
    log_bf = log_p_sig_marginal - log_p_bg

    # ── Per-shell aggregates ──
    shell_id = prep.shell_id_fw.to(device)
    n_shells = int(shell_id.max().item()) + 1 if n_obs > 0 else 0

    def _shell_med(x: Tensor) -> Tensor:
        out = torch.full((n_shells,), float("nan"))
        # Use bincount + scatter: median per shell needs sort
        for s in range(n_shells):
            mask = shell_id[h] == s
            if mask.any():
                out[s] = torch.median(x[mask][torch.isfinite(x[mask])])
        return out

    def _shell_iqr(x: Tensor) -> Tensor:
        out = torch.full((n_shells,), float("nan"))
        q_levels = torch.tensor([0.25, 0.75], device=x.device, dtype=x.dtype)
        for s in range(n_shells):
            mask = shell_id[h] == s
            if mask.any():
                vals = x[mask][torch.isfinite(x[mask])]
                if vals.numel() >= 4:
                    q25, q75 = vals.quantile(q_levels)
                    out[s] = q75 - q25
        return out

    shell_summary = {
        "pearson_med": _shell_med(pearson),
        "pearson_iqr": _shell_iqr(pearson),
        "z_med": _shell_med(z_scatter),
        "z_iqr": _shell_iqr(z_scatter),
        "log_bf_med": _shell_med(log_bf),
        "frac_signal_won": _shell_med((log_bf > 0).float()),  # sign-only summary
    }

    return Residuals(
        pearson=pearson.cpu(),
        log_scatter=log_scatter.cpu(),
        z_scatter=z_scatter.cpu(),
        log_bf=log_bf.cpu(),
        log_p_sig_marginal=log_p_sig_marginal.cpu(),
        log_p=log_p_total.cpu(),
        obs_hkl_id=h.cpu(),
        obs_d_spacing=prep.d_spacing[h].cpu(),
        obs_log_I_predicted=log_mu_sig.cpu(),
        obs_n=n.cpu(),
        obs_bg=bg.cpu(),
        obs_mu_tot=mu_tot.cpu(),
        shell_id=shell_id[h].cpu(),
        shell_summary={k: v.cpu() for k, v in shell_summary.items()},
    )


def save_residuals(res: Residuals, path) -> None:
    """Persist the residuals dataclass as a torch .pt for later plotting.

    Stores everything in a flat dict so the plotting script doesn't need
    the dataclass definition imported.
    """
    out = {
        "pearson": res.pearson, "log_scatter": res.log_scatter,
        "z_scatter": res.z_scatter, "log_bf": res.log_bf,
        "log_p_sig_marginal": res.log_p_sig_marginal,
        "log_p": res.log_p,
        "obs_hkl_id": res.obs_hkl_id, "obs_d_spacing": res.obs_d_spacing,
        "obs_log_I_predicted": res.obs_log_I_predicted,
        "obs_n": res.obs_n, "obs_bg": res.obs_bg, "obs_mu_tot": res.obs_mu_tot,
        "shell_id": res.shell_id,
    }
    out.update({f"shell_{k}": v for k, v in res.shell_summary.items()})
    torch.save(out, path)
