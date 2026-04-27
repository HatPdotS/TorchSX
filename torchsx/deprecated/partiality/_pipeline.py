"""Iterative partiality-aware merging pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from torchsx.deprecated.merge._em import fit_em, EMConfig
from torchsx.deprecated.merge._corrections import ScaleBCorrection, FixedCorrection
from torchsx.deprecated.merge._noise import PoissonNoiseModel, compute_sigma2_LN
from torchsx.deprecated.merge._hkl import compute_hkl_index

from ._excitation import ExcitationError
from ._profile import GenLorentzProfile, GaussianProfile
from ._concordance import RankConcordanceLoss
from ._refine import refine_orientations, RefinementConfig, RefinementResult
from ._pairs import PairData, build_pairs
from ._fit import fit_partiality_model


@dataclass
class PipelineConfig:
    n_outer_cycles: int = 4
    em: EMConfig = field(default_factory=EMConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    profile_type: str = "genlorentz"  # or "gaussian"
    # Partiality fitting
    part_n_steps: int = 500
    part_lr: float = 1e-3
    part_subsample: int = 500_000
    part_poly_order: int = 3
    # Pairs
    pair_I_threshold: float = 20.0
    pair_min_mult: int = 3
    pair_max_per_hkl: int = 500


@dataclass
class CycleState:
    excitation_model: ExcitationError
    profile: GenLorentzProfile | GaussianProfile | None
    dist: object | None


@dataclass
class PipelineResult:
    state: CycleState
    cycle_diagnostics: list[dict]


def run_partiality_pipeline(
    stream,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """Iterative partiality-aware merging.

    Each outer cycle:
      1. EM merge with current corrections (ScaleB + partiality)
      2. Rank-concordance orientation refinement (cold start)
      3. Fit partiality profile (GenLorentz or Gaussian)

    Parameters
    ----------
    stream : SFXStream or list[SFXStream]
        Pre-loaded stream(s). If multiple, concatenated internally.
    """
    if config is None:
        config = PipelineConfig()

    from torchsx.stream import SFXStream

    # Handle single or multiple streams
    if isinstance(stream, (list, tuple)):
        if len(stream) > 1:
            stream = SFXStream.concat(stream)
        else:
            stream = stream[0]

    core = stream._core
    N = core.hkl.shape[0]
    C = core.astar.shape[0]
    print(f"Pipeline: {N:,} obs, {C:,} crystals")

    # Precompute geometry
    hkl_idx = compute_hkl_index(core.hkl)
    hkl_id = hkl_idx.hkl_id
    n_unique = hkl_idx.n_unique

    wavelength = 12398.42 / core.beam_energy.median().item()

    # q-vector and resolution
    UB = torch.stack([core.astar, core.bstar, core.cstar], dim=2)
    beam_E = core.beam_energy[core.image_id.long()]
    lambda_0 = 12398.42 / beam_E

    # Compute q_lab for resolution
    _exc_model = ExcitationError(C, UB, lambda_0)
    with torch.no_grad():
        r_init = _exc_model.excitation_error(core.hkl, core.crystal_id)
        # q_mag from RUB @ h
        RUB = _exc_model._all_RUB()
        q = torch.einsum(
            "nij,nj->ni",
            RUB[core.crystal_id.long()],
            core.hkl.float(),
        )
    q_mag = q.norm(dim=1)
    s_sq = (q_mag / 2) ** 2
    inv_d = q_mag
    cid = core.crystal_id

    # Initialise state
    state = CycleState(
        excitation_model=ExcitationError(C, UB, lambda_0),
        profile=None,
        dist=None,
    )

    diagnostics = []

    for cycle in range(config.n_outer_cycles):
        print(f"\n{'='*60}")
        print(f"OUTER CYCLE {cycle}")
        print(f"{'='*60}")

        # ── Step 1: Full EM ──
        print("\nStep 1: Full EM...")

        corrections = [ScaleBCorrection()]
        if state.profile is not None:
            with torch.no_grad():
                abs_r = state.excitation_model.abs_excitation_error(core.hkl, cid)
                log_part = state.profile.log_partiality(abs_r, inv_d)
                log_part = log_part - log_part.mean()  # shape only
            corrections.append(FixedCorrection(log_part))

        dist = fit_em(stream, config=config.em, corrections=corrections)
        state.dist = dist
        print(f"  I0 median = {dist.I0.median():.1f}, "
              f"alpha median = {dist.alpha.median():.4f}")

        # Get BG corrections from the fitted ScaleBCorrection
        bg_corr = corrections[0]
        if hasattr(bg_corr, 'log_G'):
            log_bg = bg_corr.log_G[cid.long()] - 2 * bg_corr.B[cid.long()] * s_sq
        else:
            log_bg = torch.zeros(N)

        # ── Step 2: Rank refinement (cold start) ──
        print("\nStep 2: Rank-concordance orientation refinement...")
        I_bg_corr = core.I / torch.exp(log_bg).clamp(min=1e-6)

        pair_i, pair_j, pair_sign, pair_weight = build_pairs(
            I_bg_corr, hkl_id, cid, n_unique,
            I_threshold=config.pair_I_threshold,
            min_mult=config.pair_min_mult,
            max_pairs_per_hkl=config.pair_max_per_hkl,
        )

        pair_data = PairData(
            hkl_obs=core.hkl, crystal_id=cid, hkl_id=hkl_id,
            I_corr=I_bg_corr, s_sq=s_sq,
            UB=UB, lambda_0=lambda_0,
            pair_i=pair_i, pair_j=pair_j,
            pair_sign=pair_sign, pair_weight=pair_weight,
            n_crystals=C, n_obs=N, n_pairs=len(pair_i),
        )

        rank_result = refine_orientations(pair_data, config.refinement)
        state.excitation_model = rank_result.model

        print(f"  |dw| median: "
              f"{rank_result.delta_omega_xy.norm(dim=1).median() * 1e3:.2f} mdeg")

        # ── Step 3: Fit partiality profile ──
        print("\nStep 3: Fitting partiality profile...")
        with torch.no_grad():
            abs_r = state.excitation_model.abs_excitation_error(core.hkl, cid)

        # Map I0 to per-obs
        pos = I_bg_corr > 0
        I0_sum = torch.zeros(n_unique)
        I0_cnt = torch.zeros(n_unique)
        I0_sum.scatter_add_(0, hkl_id[pos], I_bg_corr[pos])
        I0_cnt.scatter_add_(0, hkl_id[pos], torch.ones_like(I_bg_corr[pos]))
        I0_merged = I0_sum / I0_cnt.clamp(min=1)
        I0_per_obs = I0_merged[hkl_id]

        # sigma2_LN per obs: map from dist where possible, else median
        sigma2_LN_vals = compute_sigma2_LN(dist.alpha, dist.beta)
        sigma2_LN_med = sigma2_LN_vals.median().item()

        # Build proper mapping from dist HKLs to full HKL space
        sigma2_LN_full = torch.full((n_unique,), sigma2_LN_med)
        dist_hkl_idx = compute_hkl_index(dist.hkl)
        # Match dist HKLs to observation HKLs
        # dist.hkl is a subset of the unique HKLs (those with min_obs)
        # Find which unique HKL each dist HKL corresponds to
        for i in range(len(dist.hkl)):
            h = dist.hkl[i]
            # Find matching unique HKL
            match = (hkl_idx.unique_hkl == h).all(dim=1)
            idx = torch.where(match)[0]
            if len(idx) > 0:
                sigma2_LN_full[idx[0]] = sigma2_LN_vals[i]
        sigma2_LN_per_obs = sigma2_LN_full[hkl_id]

        # Poisson noise
        noise = PoissonNoiseModel()
        sigma2_poi = noise.sigma2(core.I, core.background)
        sigma2_poi_bg = sigma2_poi / torch.exp(log_bg).clamp(min=1e-6) ** 2

        new_profile = fit_partiality_model(
            I_bg_corr, I0_per_obs, sigma2_poi_bg, sigma2_LN_per_obs,
            abs_r, inv_d,
            n_steps=config.part_n_steps,
            lr=config.part_lr,
            subsample=config.part_subsample,
            poly_order=config.part_poly_order,
        )

        # NaN guard
        with torch.no_grad():
            test_r = torch.linspace(0, 0.005, 10)
            test_inv_d = torch.full((10,), 0.5)
            test_lp = new_profile.log_partiality(test_r, test_inv_d)
        if torch.isfinite(test_lp).all():
            state.profile = new_profile
            print("  Partiality model accepted")
        else:
            print("  WARNING: Partiality model has NaN -- keeping previous")

        # ── Diagnostics ──
        diag = _compute_diagnostics(
            core.I, abs_r.detach(), hkl_id, dist, cycle,
        )
        diagnostics.append(diag)

        # Early stopping
        if len(diagnostics) >= 3:
            d0 = diagnostics[-2]["med_rho"]
            d1 = diagnostics[-1]["med_rho"]
            if abs(d1 - d0) < 0.001:
                print(f"\n  Converged (delta rho = {d1-d0:.4f})")
                break

    return PipelineResult(state=state, cycle_diagnostics=diagnostics)


def _spearman_rho(x: Tensor, y: Tensor) -> float:
    """Spearman rank correlation (pure PyTorch, no scipy)."""
    n = len(x)
    if n < 3:
        return float("nan")
    rx = x.argsort().argsort().float()
    ry = y.argsort().argsort().float()
    d = rx - ry
    return float(1.0 - 6.0 * (d ** 2).sum() / (n * (n ** 2 - 1)))


def _compute_diagnostics(I_obs, abs_r, hkl_id, dist, cycle):
    """Per-HKL Spearman rho(I, |r|) diagnostics."""
    import numpy as np

    I_np = I_obs.numpy()
    r_np = abs_r.numpy()
    hkl_np = hkl_id.numpy()
    n_unique = hkl_np.max() + 1

    order = np.argsort(hkl_np)
    boundaries = np.searchsorted(hkl_np[order], np.arange(n_unique + 1))

    rhos = []
    for uid in range(n_unique):
        lo, hi = boundaries[uid], boundaries[uid + 1]
        idx = order[lo:hi]
        I_h, r_h = I_np[idx], r_np[idx]
        keep = I_h > 50
        if keep.sum() < 10:
            continue
        I_t = torch.from_numpy(I_h[keep])
        r_t = torch.from_numpy(r_h[keep])
        if I_t.std() < 1e-10 or r_t.std() < 1e-10:
            continue
        rho = _spearman_rho(I_t, r_t)
        if np.isfinite(rho):
            rhos.append(rho)

    rhos = np.array(rhos)
    med_rho = np.median(rhos) if len(rhos) > 0 else np.nan
    frac_neg = (rhos < 0).mean() * 100 if len(rhos) > 0 else 0
    sigma_LN_med = dist.alpha.median().item()

    print(f"\n  -- Cycle {cycle} --")
    print(f"  rho(I, |r|): median = {med_rho:.4f}, {frac_neg:.0f}% neg ({len(rhos)} HKLs)")
    print(f"  sigma_LN median = {sigma_LN_med:.4f}")

    return {"cycle": cycle, "med_rho": med_rho, "frac_neg": frac_neg,
            "sigma_LN_med": sigma_LN_med}
