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

        with torch.no_grad():
            I0 = torch.exp(loss.log_I0)
            alpha = torch.exp(loss.log_alpha)
            beta = torch.exp(loss.log_beta)

            # ── Empirical σ(I0) from residuals of (I_obs − I0·scale) ──
            # Rebuild log_scale on all observations
            log_scale = torch.zeros(prep.n_obs_total, device=self.device)
            for c in loss.corrections:
                log_scale = log_scale + c(loss._obs_info_slice(0, prep.n_obs_total),
                                          obs_range=(0, prep.n_obs_total))
            if loss.has_partiality:
                log_scale = log_scale + torch.log(loss.partiality.clamp(min=1e-12))
            scale = torch.exp(log_scale)

            I0_po = I0[loss.hkl_id]
            resid_sq = (prep.I_obs - I0_po * scale) ** 2
            sum_resid_sq = torch.zeros(prep.n_valid, device=self.device).scatter_add_(
                0, loss.hkl_id, resid_sq
            )
            n = prep.obs_counts.float()
            sigma_I0 = (sum_resid_sq / (n * (n - 1).clamp(min=1))).clamp(min=1e-12).sqrt()

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
