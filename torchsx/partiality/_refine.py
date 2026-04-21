"""Optimisation loop for orientation refinement."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ._excitation import ExcitationError
from ._concordance import RankConcordanceLoss, ExplainedVarianceLoss
from ._pairs import PairData, RefineData


@dataclass
class RefinementConfig:
    n_steps: int = 300
    lr_omega: float = 1e-4
    lr_lambda: float = 1e-5
    beta_start: float = 5.0
    beta_end: float = 30.0
    restraint_omega: float = 1.0
    restraint_lambda: float = 1e4
    free_fraction: float = 0.1
    patience: int = 50       # early stop if free-set metric doesn't improve
    log_every: int = 20
    device: str = "cpu"


@dataclass
class RefinementResult:
    delta_omega_xy: Tensor
    delta_lambda: Tensor
    loss_history: list
    concordance_history: list      # work-set metric
    free_concordance_history: list  # free-set metric
    model: ExcitationError


def refine_orientations(
    pair_data: PairData,
    config: RefinementConfig | None = None,
    init_model: ExcitationError | None = None,
) -> RefinementResult:
    """Rank-concordance orientation refinement with free-set monitoring."""
    if config is None:
        config = RefinementConfig()

    device = pair_data.pair_i.device

    model = ExcitationError(
        n_crystals=pair_data.n_crystals,
        UB=pair_data.UB,
        lambda_0=pair_data.lambda_0,
    ).to(device)

    if init_model is not None:
        with torch.no_grad():
            model.delta_omega_xy.copy_(init_model.delta_omega_xy.to(device))
            model.delta_lambda.copy_(init_model.delta_lambda.to(device))

    loss_fn = RankConcordanceLoss(beta=config.beta_start).to(device)

    optimizer = torch.optim.Adam([
        {"params": [model.delta_omega_xy], "lr": config.lr_omega},
        {"params": [model.delta_lambda], "lr": config.lr_lambda},
    ])

    # Work/free split on pairs
    n_pairs = pair_data.n_pairs
    gen = torch.Generator(device=device).manual_seed(42)
    free_mask = torch.rand(n_pairs, generator=gen, device=device) < config.free_fraction
    work_mask = ~free_mask

    w_i, w_j = pair_data.pair_i[work_mask], pair_data.pair_j[work_mask]
    w_sign, w_weight = pair_data.pair_sign[work_mask], pair_data.pair_weight[work_mask]
    f_i, f_j = pair_data.pair_i[free_mask], pair_data.pair_j[free_mask]
    f_sign = pair_data.pair_sign[free_mask]

    print(f"Refining {pair_data.n_crystals:,} crystals, "
          f"{work_mask.sum():,} work / {free_mask.sum():,} free pairs, "
          f"{config.n_steps} steps")
    print(f"  Soft restraints: λ_ω={config.restraint_omega}, "
          f"λ_λ={config.restraint_lambda}")

    loss_history, concordance_history, free_concordance_history = [], [], []
    best_free, best_step, best_state = -1.0, 0, None

    for step in range(config.n_steps):
        frac = step / max(config.n_steps - 1, 1)
        loss_fn.beta = config.beta_start + (config.beta_end - config.beta_start) * frac

        optimizer.zero_grad()
        abs_r = model.abs_excitation_error(pair_data.hkl_obs, pair_data.crystal_id)
        conc_loss = loss_fn(abs_r, w_i, w_j, w_sign, w_weight)

        omega_pen = config.restraint_omega * (model.delta_omega_xy ** 2).mean()
        lambda_pen = config.restraint_lambda * (model.delta_lambda ** 2).mean()
        (conc_loss + omega_pen + lambda_pen).backward()
        optimizer.step()

        loss_history.append(conc_loss.item())

        if step % config.log_every == 0 or step == config.n_steps - 1:
            with torch.no_grad():
                abs_r_d = model.abs_excitation_error(
                    pair_data.hkl_obs, pair_data.crystal_id)
                cf_w = loss_fn.concordance_fraction(abs_r_d, w_i, w_j, w_sign)
                cf_f = loss_fn.concordance_fraction(abs_r_d, f_i, f_j, f_sign)
            concordance_history.append(cf_w)
            free_concordance_history.append(cf_f)

            if cf_f > best_free:
                best_free, best_step = cf_f, step
                best_state = {k: v.detach().clone() for k, v in
                              [("dw", model.delta_omega_xy), ("dl", model.delta_lambda)]}

            dw = model.delta_omega_xy.detach().norm(dim=1)
            print(f"  step {step:>4}: loss={conc_loss.item():+.6f}  "
                  f"work={cf_w:.4f} free={cf_f:.4f}  β={loss_fn.beta:.1f}  "
                  f"|δω|={dw.median()*1e3:.2f} mdeg  pen_ω={omega_pen.item():.4e}")

            if config.patience > 0 and step - best_step > config.patience:
                print(f"  Early stop: free-set peaked at step {best_step}")
                break

    if best_state is not None:
        with torch.no_grad():
            model.delta_omega_xy.copy_(best_state["dw"])
            model.delta_lambda.copy_(best_state["dl"])
        print(f"  Restored best (step {best_step}, free={best_free:.4f})")

    return RefinementResult(
        delta_omega_xy=model.delta_omega_xy.detach().cpu(),
        delta_lambda=model.delta_lambda.detach().cpu(),
        loss_history=loss_history,
        concordance_history=concordance_history,
        free_concordance_history=free_concordance_history,
        model=model,
    )


def refine_orientations_ev(
    data: RefineData,
    config: RefinementConfig | None = None,
    init_model: ExcitationError | None = None,
) -> RefinementResult:
    """Explained-variance orientation refinement with free-set monitoring.

    Uses per-HKL Corr²(|r|, log_resid) as the loss — maximises the
    variance of log-residuals explained by |r|.  No partiality model
    assumed; Poisson+GenGauss noise weighting via the ``weights`` field.
    """
    if config is None:
        config = RefinementConfig()

    device = data.hkl_obs.device

    model = ExcitationError(
        n_crystals=data.n_crystals,
        UB=data.UB,
        lambda_0=data.lambda_0,
    ).to(device)

    if init_model is not None:
        with torch.no_grad():
            model.delta_omega_xy.copy_(init_model.delta_omega_xy.to(device))
            model.delta_lambda.copy_(init_model.delta_lambda.to(device))

    loss_fn = ExplainedVarianceLoss()

    optimizer = torch.optim.Adam([
        {"params": [model.delta_omega_xy], "lr": config.lr_omega},
        {"params": [model.delta_lambda], "lr": config.lr_lambda},
    ])

    # Work/free split on observations (per-obs random mask)
    N = data.hkl_obs.shape[0]
    gen = torch.Generator(device=device).manual_seed(42)
    free_mask = torch.rand(N, generator=gen, device=device) < config.free_fraction
    work_mask = ~free_mask

    work_w = data.weights * work_mask.float()
    free_w = data.weights * free_mask.float()

    n_work = work_mask.sum().item()
    n_free = free_mask.sum().item()

    print(f"Refining {data.n_crystals:,} crystals, "
          f"{n_work:,} work / {n_free:,} free obs, {config.n_steps} steps")
    print(f"  Soft restraints: λ_ω={config.restraint_omega}, "
          f"λ_λ={config.restraint_lambda}")

    loss_history, r2_work_history, r2_free_history = [], [], []
    best_free_r2, best_step, best_state = -1.0, 0, None

    for step in range(config.n_steps):
        optimizer.zero_grad()

        abs_r = model.abs_excitation_error(data.hkl_obs, data.crystal_id)

        # Loss on work set only (free-set weights are zero)
        ev_loss = loss_fn(abs_r, data.hkl_id, data.log_resid, work_w, data.n_hkl)

        omega_pen = config.restraint_omega * (model.delta_omega_xy ** 2).mean()
        lambda_pen = config.restraint_lambda * (model.delta_lambda ** 2).mean()

        (ev_loss + omega_pen + lambda_pen).backward()
        optimizer.step()

        loss_history.append(ev_loss.item())

        if step % config.log_every == 0 or step == config.n_steps - 1:
            with torch.no_grad():
                abs_r_d = model.abs_excitation_error(data.hkl_obs, data.crystal_id)
                r2_w = loss_fn.mean_r_squared(
                    abs_r_d, data.hkl_id, data.log_resid, work_w, data.n_hkl)
                r2_f = loss_fn.mean_r_squared(
                    abs_r_d, data.hkl_id, data.log_resid, free_w, data.n_hkl)
            r2_work_history.append(r2_w)
            r2_free_history.append(r2_f)

            if r2_f > best_free_r2:
                best_free_r2, best_step = r2_f, step
                best_state = {k: v.detach().clone() for k, v in
                              [("dw", model.delta_omega_xy), ("dl", model.delta_lambda)]}

            dw = model.delta_omega_xy.detach().norm(dim=1)
            print(f"  step {step:>4}: loss={ev_loss.item():+.8f}  "
                  f"R²_work={r2_w:.6f} R²_free={r2_f:.6f}  "
                  f"|δω|={dw.median()*1e3:.2f} mdeg  pen_ω={omega_pen.item():.4e}")

            if config.patience > 0 and step - best_step > config.patience:
                print(f"  Early stop: free-set R² peaked at step {best_step}")
                break

    if best_state is not None:
        with torch.no_grad():
            model.delta_omega_xy.copy_(best_state["dw"])
            model.delta_lambda.copy_(best_state["dl"])
        print(f"  Restored best (step {best_step}, free R²={best_free_r2:.6f})")

    return RefinementResult(
        delta_omega_xy=model.delta_omega_xy.detach().cpu(),
        delta_lambda=model.delta_lambda.detach().cpu(),
        loss_history=loss_history,
        concordance_history=r2_work_history,
        free_concordance_history=r2_free_history,
        model=model,
    )
