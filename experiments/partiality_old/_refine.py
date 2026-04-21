"""Optimisation loop for rank-concordance orientation refinement."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from partiality._excitation import ExcitationErrorModel
from partiality._concordance import RankConcordanceLoss
from partiality._data import PairData


@dataclass
class RefinementConfig:
    n_steps: int = 300
    lr_omega: float = 1e-4
    lr_lambda: float = 1e-5
    beta_start: float = 5.0
    beta_end: float = 30.0
    # Soft restraints (L2 penalty) instead of hard clamps
    restraint_omega: float = 1.0    # penalty weight for ||δω||²
    restraint_lambda: float = 1e4   # penalty weight for ||δλ||² (δλ is small in Å)
    log_every: int = 20
    device: str = "cpu"


@dataclass
class RefinementResult:
    delta_omega_xy: Tensor
    delta_lambda: Tensor
    loss_history: list
    concordance_history: list
    model: ExcitationErrorModel


def refine_orientations(
    pair_data: PairData,
    config: RefinementConfig | None = None,
) -> RefinementResult:
    """Run rank-concordance refinement with soft restraints."""
    if config is None:
        config = RefinementConfig()

    device = pair_data.pair_i.device

    model = ExcitationErrorModel(
        n_crystals=pair_data.n_crystals,
        UB=pair_data.UB,
        lambda_0=pair_data.lambda_0,
    ).to(device)

    loss_fn = RankConcordanceLoss(beta=config.beta_start).to(device)

    optimizer = torch.optim.Adam([
        {"params": [model.delta_omega_xy], "lr": config.lr_omega},
        {"params": [model.delta_lambda], "lr": config.lr_lambda},
    ])

    loss_history = []
    concordance_history = []

    print(f"Refining {pair_data.n_crystals:,} crystals, "
          f"{pair_data.n_pairs:,} pairs, {config.n_steps} steps")
    print(f"  Soft restraints: λ_ω={config.restraint_omega}, "
          f"λ_λ={config.restraint_lambda}")

    for step in range(config.n_steps):
        frac = step / max(config.n_steps - 1, 1)
        loss_fn.beta = config.beta_start + (config.beta_end - config.beta_start) * frac

        optimizer.zero_grad()

        abs_r = model.abs_excitation_error(pair_data.hkl_obs, pair_data.crystal_id)
        concordance_loss = loss_fn(
            abs_r,
            pair_data.pair_i, pair_data.pair_j,
            pair_data.pair_sign, pair_data.pair_weight,
        )

        # Soft restraints (L2 penalty)
        omega_penalty = config.restraint_omega * (model.delta_omega_xy ** 2).mean()
        lambda_penalty = config.restraint_lambda * (model.delta_lambda ** 2).mean()

        loss = concordance_loss + omega_penalty + lambda_penalty
        loss.backward()
        optimizer.step()

        loss_history.append(concordance_loss.item())

        if step % config.log_every == 0 or step == config.n_steps - 1:
            with torch.no_grad():
                abs_r_d = model.abs_excitation_error(
                    pair_data.hkl_obs, pair_data.crystal_id)
                cf = loss_fn.concordance_fraction(
                    abs_r_d, pair_data.pair_i, pair_data.pair_j, pair_data.pair_sign)
            concordance_history.append(cf)
            omega_mag = model.delta_omega_xy.detach().norm(dim=1)
            dlam = model.delta_lambda.detach()
            print(f"  step {step:>4}: conc_loss={concordance_loss.item():+.6f}  "
                  f"concordance={cf:.4f}  β={loss_fn.beta:.1f}  "
                  f"|δω| med={omega_mag.median()*1e3:.2f} mdeg  "
                  f"|δλ| med={dlam.abs().median()*1e4:.2f}×10⁻⁴Å  "
                  f"pen_ω={omega_penalty.item():.4e} pen_λ={lambda_penalty.item():.4e}")

    return RefinementResult(
        delta_omega_xy=model.delta_omega_xy.detach().cpu(),
        delta_lambda=model.delta_lambda.detach().cpu(),
        loss_history=loss_history,
        concordance_history=concordance_history,
        model=model,
    )
