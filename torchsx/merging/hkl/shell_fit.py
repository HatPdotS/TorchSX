"""L-BFGS fit of a GenGauss + Gauss mixture to ``log(I/⟨I⟩)`` per shell.

Used by ``prepare_observations`` to seed per-HKL GenGauss β from shell-pooled
data.  Extracted from the legacy ``torchsx.merge._distribution._fit_shell_lbfgs``
(renamed here to ``fit_shell_gengauss_gauss`` for clarity).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def fit_shell_gengauss_gauss(
    data: torch.Tensor,
) -> tuple[float, ...] | None:
    """Fit ``GenGauss(β, μ₁, α) + (1−w)·Gauss(μ₂, σ₂)`` to a 1D sample via L-BFGS.

    Parameters
    ----------
    data : 1D float tensor of log(I/mean_I) values pooled across one shell.

    Returns
    -------
    (beta, mu1, alpha, w1, mu2, sigma2) or ``None`` on optimiser failure.
    The GenGauss component is the signal branch; the Gaussian component
    absorbs heavy-tailed or outlier mass.  Guaranteed μ₁ ≤ μ₂.
    """
    y = data.double()
    mu_all = float(y.mean())
    sig_all = float(y.std())

    beta_unc = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    mu1 = torch.tensor(mu_all - 0.3, dtype=torch.float64, requires_grad=True)
    log_alpha = torch.tensor(
        math.log(max(sig_all * 0.8, 0.05)), dtype=torch.float64, requires_grad=True
    )
    w_logit = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    mu2 = torch.tensor(mu_all + 1.0, dtype=torch.float64, requires_grad=True)
    log_sigma2 = torch.tensor(
        math.log(max(sig_all * 0.7, 0.05)), dtype=torch.float64, requires_grad=True
    )

    params = [beta_unc, mu1, log_alpha, w_logit, mu2, log_sigma2]
    optimizer = torch.optim.LBFGS(
        params,
        lr=1.0,
        max_iter=100,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
    )

    def closure():
        optimizer.zero_grad()
        beta = F.softplus(beta_unc) + 0.5
        alpha = torch.exp(log_alpha)
        sigma2 = torch.exp(log_sigma2)
        w1 = torch.sigmoid(w_logit)

        z1 = torch.abs(y - mu1) / alpha.clamp(min=1e-8)
        log_pdf1 = (
            torch.log(beta)
            - torch.log(2.0 * alpha)
            - torch.lgamma(1.0 / beta)
            - z1.clamp(min=1e-8).pow(beta)
        )

        log_pdf2 = (
            -0.5 * math.log(2 * math.pi)
            - log_sigma2
            - 0.5 * ((y - mu2) / sigma2.clamp(min=1e-8)) ** 2
        )

        log_mix = torch.logaddexp(
            torch.log(w1.clamp(min=1e-8)) + log_pdf1,
            torch.log((1 - w1).clamp(min=1e-8)) + log_pdf2,
        )
        loss = -log_mix.mean()
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except Exception:
        return None

    with torch.no_grad():
        beta_val = (F.softplus(beta_unc) + 0.5).item()
        mu1_val, mu2_val = mu1.item(), mu2.item()
        alpha_val = torch.exp(log_alpha).item()
        sigma2_val = torch.exp(log_sigma2).item()
        w1_val = torch.sigmoid(w_logit).item()

    if mu1_val > mu2_val:
        # Swap so μ₁ ≤ μ₂ (GenGauss component is the signal branch).
        beta_val, mu1_val, alpha_val, w1_val, mu2_val, sigma2_val = (
            beta_val, mu2_val, sigma2_val, 1 - w1_val, mu1_val, alpha_val,
        )

    return beta_val, mu1_val, alpha_val, w1_val, mu2_val, sigma2_val
