"""Central orchestration class: stream → `MergedResult` via one LBFGS."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor

from torchsx.merge._noise import PoissonNoiseModel
from torchsx.merging_clean.config import MergeConfig
from torchsx.merging_clean.corrections import (
    Correction,
    ScaleBCorrection,
)
from torchsx.merging_clean.loss import UnifiedMergeLoss
from torchsx.merging_clean.observations import PreparedObs, prepare_observations
from torchsx.merging_clean.result import MergedResult


def _q_moments(
    alpha: Tensor, beta: Tensor, *, n_grid: int = 201, u_max: float = 10.0
) -> tuple[Tensor, Tensor]:
    """E[q] and Var[q] for q = exp(ε), ε ~ GenGauss(0, α, β), per HKL.

    Evaluated by numerical integration over standardised u = ε/α with a
    uniform grid on [-u_max, u_max].  With α ≈ 1.5, β ≈ 2, the true Var[q]
    is roughly 5× larger than the linear approximation α²·c(β)² — so using
    the correct moments matters quantitatively for σ(I_h) at bright HKLs.

    Shapes: alpha, beta (n_HKL,).  Returns (E[q], Var[q]), both (n_HKL,).
    """
    device = alpha.device
    u = torch.linspace(-u_max, u_max, n_grid, device=device)
    du = (2.0 * u_max) / (n_grid - 1)
    # log density (standardised GenGauss): log β − log 2 − lgamma(1/β) − |u|^β
    log_norm = torch.log(beta) - math.log(2.0) - torch.lgamma(1.0 / beta)   # (n_HKL,)
    abs_u_pow_beta = u.abs().unsqueeze(0).pow(beta.unsqueeze(1))            # (n_HKL, K)
    log_p = log_norm.unsqueeze(1) - abs_u_pow_beta                          # (n_HKL, K)
    p = log_p.exp()                                                         # density in u
    # q = exp(ε) = exp(α·u)
    q = torch.exp(alpha.unsqueeze(1) * u.unsqueeze(0))                      # (n_HKL, K)
    E_q = (q * p).sum(dim=1) * du                                            # (n_HKL,)
    E_q2 = (q * q * p).sum(dim=1) * du
    Var_q = (E_q2 - E_q * E_q).clamp(min=0.0)
    return E_q, Var_q


class Merger:
    """Merge unmerged SFX observations with a single joint LBFGS minimization.

    Parameters
    ----------
    stream : torchsx SFXStream
    config : MergeConfig (defaults applied if None)
    corrections : list of Correction classes or instances, or None.
        - None → defaults to ``[ScaleBCorrection]``
        - class → built from the stream via ``.from_stream(stream, obs_info)``
        - instance → used as-is (the caller pre-sized the params)
    partiality : (N_obs,) tensor in stream order, optional.
    device : 'cpu' or 'cuda'.  All tensors move to this device before fit.
    poisson_noise : PoissonNoiseModel, optional.  Sets integration-region
        pixel counts used to build the total background.
    """

    def __init__(
        self,
        stream,
        *,
        config: MergeConfig | None = None,
        corrections: Sequence[Correction] | None = None,
        partiality: Tensor | None = None,
        device: str | torch.device = "cpu",
        poisson_noise: PoissonNoiseModel | None = None,
        init_from: MergedResult | None = None,
    ):
        self.stream = stream
        self.config = config or MergeConfig()
        self.device = torch.device(device)
        self.partiality = partiality
        self.poisson_noise = poisson_noise or PoissonNoiseModel()
        self._user_corrections = corrections  # None → default later
        self.init_from = init_from  # optional MergedResult to seed log_I0/α/β

        # Filled by merge()
        self.prep = None
        self.loss: UnifiedMergeLoss | None = None
        self.diagnostics: dict = {}

    # ── public API ──

    def merge(self) -> MergedResult:
        self._prepare()
        self._build_loss()
        self._run_lbfgs()
        return self._package_result()

    # ── internals ──

    def _prepare(self) -> None:
        prep = prepare_observations(
            self.stream,
            self.config,
            partiality=self.partiality,
            poisson_noise=self.poisson_noise,
        )
        if self.init_from is not None:
            self._seed_init_from(prep, self.init_from)
        if self.device.type != "cpu":
            prep = prep.to(self.device)
        self.prep = prep

    @staticmethod
    def _seed_init_from(prep, source: MergedResult) -> None:
        """Override prep's mu_init / alpha_init / beta_perhkl for HKLs matched
        in `source` by (h, k, l).  HKLs absent from source keep the shell-fit
        init.  Modifies `prep` in place (CPU tensors)."""
        src_hkl = source.hkl.long()
        lookup: dict[tuple[int, int, int], int] = {}
        for i in range(src_hkl.shape[0]):
            lookup[(int(src_hkl[i, 0]), int(src_hkl[i, 1]), int(src_hkl[i, 2]))] = i

        tgt_hkl = prep.hkl_valid.long()
        match = torch.full((prep.n_valid,), -1, dtype=torch.long)
        for i in range(prep.n_valid):
            key = (int(tgt_hkl[i, 0]), int(tgt_hkl[i, 1]), int(tgt_hkl[i, 2]))
            j = lookup.get(key)
            if j is not None:
                match[i] = j
        has = match >= 0
        n_match = int(has.sum())
        if n_match == 0:
            return
        idx = match[has]
        prep.mu_init[has] = torch.log(source.I0[idx].clamp(min=1e-6))
        prep.alpha_init[has] = source.alpha[idx].clamp(min=0.01)
        prep.beta_perhkl[has] = source.beta[idx].clamp(min=0.5)
        print(f"  init_from: matched {n_match}/{prep.n_valid} HKLs", flush=True)

    def _build_corrections(self) -> list[Correction]:
        user = self._user_corrections
        if user is None:
            user = [ScaleBCorrection]
        corrs: list[Correction] = []
        for item in user:
            if isinstance(item, type) and issubclass(item, Correction):
                c = item.from_stream(self.stream, self.prep.obs_info)
            elif isinstance(item, Correction):
                c = item
            else:
                raise TypeError(
                    f"corrections entry must be a Correction class or instance, got {type(item)!r}"
                )
            corrs.append(c.to(self.device))
        return corrs

    def _build_loss(self) -> None:
        corrections = self._build_corrections()
        loss = UnifiedMergeLoss(self.prep, corrections, self.config)
        loss.to(self.device)
        self.loss = loss

    def _run_lbfgs(self) -> None:
        cfg = self.config
        loss = self.loss
        dev = self.device
        opt = torch.optim.LBFGS(
            [p for p in loss.parameters() if p.requires_grad],
            lr=cfg.lr,
            max_iter=cfg.max_iter,
            history_size=cfg.history_size,
            tolerance_grad=cfg.tolerance_grad,
            tolerance_change=cfg.tolerance_change,
            line_search_fn="strong_wolfe",
        )

        n_calls = {"n": 0}

        def closure():
            opt.zero_grad(set_to_none=False)
            total = torch.zeros((), device=dev)
            for s, e in loss.obs_chunks(cfg.obs_chunk_size):
                nll = loss.data_nll(s, e)
                nll.backward()
                total = total + nll.detach()
            reg = loss.regularizers_only()
            reg.backward()
            total = total + reg.detach()
            n_calls["n"] += 1
            return total

        final_loss = opt.step(closure)
        with torch.no_grad():
            reg_final = loss.regularizers_only()
            data_final = loss.data_nll(0, loss.n_obs_total)
            # Max gradient norm across parameters — useful for convergence check
            max_grad = 0.0
            for p in loss.parameters():
                if p.grad is not None:
                    g = p.grad.detach().abs().max().item()
                    if g > max_grad:
                        max_grad = g
        self.diagnostics.update(
            final_loss=float(final_loss) if final_loss is not None else None,
            final_data_nll=float(data_final),
            final_reg=float(reg_final),
            n_closure_calls=n_calls["n"],
            max_grad=max_grad,
            max_iter=cfg.max_iter,
            tolerance_grad=cfg.tolerance_grad,
        )

    def _package_result(self) -> MergedResult:
        loss = self.loss
        prep = self.prep
        cfg = self.config

        # ── Hessian-diagonal σ(log_I0) from data NLL + regularizers ──
        # Data NLL's Hessian w.r.t. log_I0 is diagonal (log p_j only depends
        # on log_I0[h_j]), so H·1 = diag(H).  Compute per chunk to keep the
        # autograd graph small.  Regularizer Hessian on log_I0 is diagonal
        # with entries 2·λ_floor·s (from the weak floor prior).  Dirichlet
        # smoothing on log_I0 couples neighbours but is off by default.
        # σ(I_h) = σ(log_I₀_h) · I_h_MAP   (delta method).
        H_diag = torch.zeros_like(loss.log_I0)
        for s, e in loss.obs_chunks(cfg.obs_chunk_size):
            nll = loss.data_nll(s, e)
            grad_logI0 = torch.autograd.grad(nll, loss.log_I0, create_graph=True)[0]
            H_chunk = torch.autograd.grad(
                grad_logI0, loss.log_I0,
                grad_outputs=torch.ones_like(loss.log_I0),
                retain_graph=False,
            )[0]
            H_diag = H_diag + H_chunk.detach()

        # Regularizer diagonal contribution (only floor prior touches log_I0
        # with a diagonal term; smoothing on log_I0 would add off-diagonal
        # but is disabled by default).
        reg_scale = float(prep.n_obs_total) / float(cfg.reg_scale_ref_obs)
        if cfg.lam_I0_floor > 0:
            H_diag = H_diag + 2.0 * cfg.lam_I0_floor * reg_scale
        if cfg.lam_I0_smooth > 0:
            # Dirichlet on log_I0 contributes 2·λ_smooth·(deg + 1) approx
            # to each diagonal (deg = neighbour count).  Coarse; off by default.
            H_diag = (
                H_diag
                + 2.0 * cfg.lam_I0_smooth * reg_scale
                * loss.edge_src.bincount(minlength=loss.n_valid).float().clamp(min=0)
            )

        with torch.no_grad():
            I0 = torch.exp(loss.log_I0)
            alpha = torch.exp(loss.log_alpha)
            beta = torch.exp(loss.log_beta)

            sigma_logI0 = (1.0 / H_diag.clamp(min=1e-6)).sqrt()
            sigma_I0 = sigma_logI0 * I0

            # Rebuild log_scale on all observations (used for diagnostics below
            # and for bias correction if enabled).
            log_scale = torch.zeros(prep.n_obs_total, device=self.device)
            for c in loss.corrections:
                log_scale = log_scale + c(loss._obs_info_slice(0, prep.n_obs_total),
                                          obs_range=(0, prep.n_obs_total))
            if loss.has_partiality:
                log_scale = log_scale + torch.log(loss.partiality.clamp(min=1e-12))
            scale = torch.exp(log_scale)

            # ── Truncation-bias correction (port from fit_em) ──
            if self.config.apply_bias_correction:
                bg_sum = torch.zeros(prep.n_valid, device=self.device).scatter_add_(
                    0, loss.hkl_id, prep.bg.clamp(min=0)
                )
                bg_per_hkl = (bg_sum / prep.obs_counts.float().clamp(min=1)).clamp(min=1.0)
                sigma_floor = (bg_per_hkl / prep.obs_counts.float().clamp(min=1)).sqrt()
                sig_lo_hkl = (sigma_floor * 0.5).clamp(min=0.01)

                log_I0_raw = torch.log(I0.clamp(min=1e-8))
                eps_min = torch.log(sig_lo_hkl.clamp(min=1e-8)) - log_I0_raw
                z_min = eps_min / alpha.clamp(min=1e-8)
                phi = torch.exp(-0.5 * z_min * z_min) / math.sqrt(2 * math.pi)
                Phi_c = (0.5 * torch.erfc(z_min / math.sqrt(2))).clamp(min=1e-8)
                bias = alpha * phi / Phi_c
                I0 = torch.exp(log_I0_raw - bias)

            mu = torch.log(I0.clamp(min=1e-8))
            sigma_mu = sigma_I0 / I0.clamp(min=1e-8)

        return MergedResult(
            hkl=prep.hkl_valid.cpu(),
            I0=I0.cpu(),
            alpha=alpha.cpu(),
            beta=beta.cpu(),
            mu=mu.cpu(),
            d_spacing=prep.d_spacing.cpu(),
            n_obs=prep.obs_counts.cpu(),
            sigma_I0=sigma_I0.cpu(),
            sigma_mu=sigma_mu.cpu(),
            diagnostics=dict(self.diagnostics),
        )
