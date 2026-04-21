"""Iterative partiality-aware merging pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from partiality._excitation import ExcitationErrorModel
from partiality._concordance import RankConcordanceLoss
from partiality._refine import refine_orientations, RefinementConfig
from partiality._partiality_model import GenLorentzPartiality, fit_partiality_model
from partiality._data import _build_pairs, PairData


@dataclass
class PipelineConfig:
    n_outer_cycles: int = 4
    # EM
    n_em_cycles: int = 10
    n_shells: int = 20
    min_obs: int = 20
    # Rank refinement
    rank_n_steps: int = 200
    rank_lr_omega: float = 1e-4
    rank_lr_lambda: float = 1e-5
    rank_restraint_omega: float = 300.0
    rank_restraint_lambda: float = 2e4
    rank_beta_range: tuple[float, float] = (5.0, 30.0)
    # Partiality
    part_poly_order: int = 3
    part_n_steps: int = 500
    part_lr: float = 1e-3
    part_subsample: int = 500_000
    # Pairs
    pair_I_threshold: float = 20.0
    pair_min_mult: int = 3
    pair_max_per_hkl: int = 500


@dataclass
class CycleState:
    excitation_model: ExcitationErrorModel
    partiality_model: GenLorentzPartiality | None
    dist: object | None  # HKLDistribution from fit_em


@dataclass
class PipelineResult:
    state: CycleState
    cycle_diagnostics: list[dict]


def run_pipeline(
    streams,
    config: PipelineConfig | None = None,
    device: str = "cpu",
) -> PipelineResult:
    """Iterative partiality-aware merging.

    Parameters
    ----------
    streams : SFXStream or list[SFXStream]
        Pre-loaded stream(s). If multiple, concatenated internally.
    """
    if config is None:
        config = PipelineConfig()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from torchsx.merge._em import fit_em, EMConfig
    from torchsx.merge._corrections import ScaleBCorrection, FixedCorrection
    from torchsx.merge._noise import PoissonNoiseModel, compute_sigma2_LN
    from torchsx.merge._hkl import compute_hkl_index
    from torchsx.merge._neural_model import compute_excitation_error

    # Handle single or multiple streams
    if isinstance(streams, (list, tuple)):
        if len(streams) > 1:
            from torchsx.stream import SFXStream
            stream = SFXStream.concat(streams)
        else:
            stream = streams[0]
    else:
        stream = streams

    core = stream._core
    N = core.hkl.shape[0]
    C = core.astar.shape[0]
    print(f"Pipeline: {N:,} obs, {C:,} crystals")

    # Precompute geometry
    hkl_idx = compute_hkl_index(core.hkl)
    hkl_id = hkl_idx.hkl_id

    wavelength = 12398.42 / core.beam_energy.median().item()
    _, q_lab = compute_excitation_error(
        core.hkl, core.astar, core.bstar, core.cstar,
        core.crystal_id, wavelength)
    q_mag = q_lab.norm(dim=1)
    s_sq = (q_mag / 2) ** 2
    inv_d = q_mag
    cid = core.crystal_id

    UB = torch.stack([core.astar, core.bstar, core.cstar], dim=2)
    beam_E = core.beam_energy[core.image_id.long()]
    lambda_0 = 12398.42 / beam_E

    # Initialise state
    state = CycleState(
        excitation_model=ExcitationErrorModel(C, UB, lambda_0),
        partiality_model=None,
        dist=None,
    )

    diagnostics = []

    for cycle in range(config.n_outer_cycles):
        print(f"\n{'='*60}")
        print(f"OUTER CYCLE {cycle}")
        print(f"{'='*60}")

        # ── Step 1: Full EM (quadrature D-step + BG S-step) ──
        print("\nStep 1: Full EM...")

        # Build corrections list
        if state.partiality_model is not None:
            with torch.no_grad():
                abs_r = state.excitation_model.abs_excitation_error(core.hkl, cid)
                log_part = state.partiality_model.log_partiality(abs_r, inv_d)
                log_part = log_part - log_part.mean()  # shape only
            corrections = [ScaleBCorrection(), FixedCorrection(log_part)]
        else:
            corrections = [ScaleBCorrection()]

        em_config = EMConfig(
            n_em_cycles=config.n_em_cycles,
            n_shells=config.n_shells,
            min_obs=config.min_obs,
        )
        dist = fit_em(stream, config=em_config, corrections=corrections)
        state.dist = dist
        print(f"  I₀ median = {dist.I0.median():.1f}, "
              f"α median = {dist.alpha.median():.4f}")

        # Get BG corrections from the fitted ScaleBCorrection
        bg_corr = corrections[0]
        if hasattr(bg_corr, 'log_G'):
            log_bg = bg_corr.log_G[cid.long()] - 2 * bg_corr.B[cid.long()] * s_sq
        else:
            log_bg = torch.zeros(N)

        # ── Step 2: Rank refinement (cold start from original UB) ──
        print("\nStep 2: Rank-concordance orientation refinement...")
        I_bg_corr = core.I / torch.exp(log_bg).clamp(min=1e-6)

        # Build pairs from BG-corrected intensities
        n_unique = hkl_idx.n_unique
        pair_i, pair_j, pair_sign, pair_weight = _build_pairs(
            I_bg_corr, hkl_id, cid, n_unique,
            I_threshold=config.pair_I_threshold,
            min_mult=config.pair_min_mult,
            max_pairs_per_hkl=config.pair_max_per_hkl)

        pair_data = PairData(
            hkl_obs=core.hkl, crystal_id=cid, hkl_id=hkl_id,
            I_corr=I_bg_corr, s_sq=s_sq,
            UB=UB, lambda_0=lambda_0,
            pair_i=pair_i, pair_j=pair_j,
            pair_sign=pair_sign, pair_weight=pair_weight,
            n_crystals=C, n_obs=N, n_pairs=len(pair_i))

        rank_config = RefinementConfig(
            n_steps=config.rank_n_steps,
            lr_omega=config.rank_lr_omega,
            lr_lambda=config.rank_lr_lambda,
            beta_start=config.rank_beta_range[0],
            beta_end=config.rank_beta_range[1],
            restraint_omega=config.rank_restraint_omega,
            restraint_lambda=config.rank_restraint_lambda,
            log_every=50,
        )
        rank_result = refine_orientations(pair_data, rank_config)
        state.excitation_model = rank_result.model

        print(f"  |δω| median: "
              f"{rank_result.delta_omega_xy.norm(dim=1).median() * 1e3:.2f} mdeg")

        # ── Step 3: Fit partiality model ──
        print("\nStep 3: Fitting GenLorentz partiality...")
        with torch.no_grad():
            abs_r = state.excitation_model.abs_excitation_error(core.hkl, cid)

        # Map dist I₀ to per-observation
        # dist uses a subset of HKLs (n_valid) — need to map back
        # For now use a simple merged I₀ from BG-corrected intensities
        pos = I_bg_corr > 0
        I0_sum = torch.zeros(n_unique)
        I0_cnt = torch.zeros(n_unique)
        I0_sum.scatter_add_(0, hkl_id[pos], I_bg_corr[pos])
        I0_cnt.scatter_add_(0, hkl_id[pos], torch.ones_like(I_bg_corr[pos]))
        I0_merged = I0_sum / I0_cnt.clamp(min=1)
        I0_per_obs = I0_merged[hkl_id]

        # σ²_LN per obs from fitted α, β
        sigma2_LN = compute_sigma2_LN(dist.alpha, dist.beta)
        # Map dist's per-valid-HKL to per-unique-HKL
        sigma2_LN_full = torch.full((n_unique,), sigma2_LN.median().item())
        # dist.hkl gives us the valid HKLs — but mapping is complex
        # Use per-shell median as approximation
        sigma2_LN_per_obs = torch.full((N,), sigma2_LN.median().item())

        # Poisson noise (from detector, already calibrated)
        noise = PoissonNoiseModel()
        sigma2_poi = noise.sigma2(core.I, core.background)
        # Scale by BG correction
        sigma2_poi_bg = sigma2_poi / torch.exp(log_bg).clamp(min=1e-6) ** 2

        new_part_model = fit_partiality_model(
            I_bg_corr, I0_per_obs, sigma2_poi_bg, sigma2_LN_per_obs,
            abs_r, inv_d,
            n_steps=config.part_n_steps,
            lr=config.part_lr,
            subsample=config.part_subsample,
            poly_order=config.part_poly_order,
        )
        # Only accept if the model didn't go NaN
        with torch.no_grad():
            test_r = torch.linspace(0, 0.005, 10)
            test_inv_d = torch.full((10,), 0.5)
            test_lp = new_part_model.log_partiality(test_r, test_inv_d)
        if torch.isfinite(test_lp).all():
            state.partiality_model = new_part_model
            print("  Partiality model accepted")
        else:
            print("  WARNING: Partiality model has NaN — keeping previous")

        # ── Diagnostics ──
        diag = _diagnostics(core.I.numpy(), abs_r.numpy(), hkl_id.numpy(),
                            dist, cycle)
        diagnostics.append(diag)

    return PipelineResult(state=state, cycle_diagnostics=diagnostics)


def _diagnostics(I_obs, abs_r, hkl_id, dist, cycle):
    from scipy.stats import spearmanr

    n_unique = hkl_id.max() + 1
    order = np.argsort(hkl_id)
    boundaries = np.searchsorted(hkl_id[order], np.arange(n_unique + 1))

    rhos = []
    for uid in range(n_unique):
        lo, hi = boundaries[uid], boundaries[uid + 1]
        idx = order[lo:hi]
        I_h, r_h = I_obs[idx], abs_r[idx]
        keep = I_h > 50
        if keep.sum() < 10:
            continue
        if np.ptp(I_h[keep]) < 1e-10 or np.ptp(r_h[keep]) < 1e-10:
            continue
        rho, _ = spearmanr(I_h[keep], r_h[keep])
        rhos.append(rho)

    rhos = np.array(rhos)
    med_rho = np.median(rhos) if len(rhos) > 0 else np.nan
    frac_neg = (rhos < 0).mean() * 100 if len(rhos) > 0 else 0
    sigma_LN_med = dist.alpha.median().item()

    print(f"\n  ── Cycle {cycle} ──")
    print(f"  ρ(I, |r|): median = {med_rho:.4f}, {frac_neg:.0f}% neg ({len(rhos)} HKLs)")
    print(f"  σ_LN median = {sigma_LN_med:.4f}")

    return {"cycle": cycle, "med_rho": med_rho, "frac_neg": frac_neg,
            "sigma_LN_med": sigma_LN_med}
