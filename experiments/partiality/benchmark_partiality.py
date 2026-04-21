#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -p gpu-day
#SBATCH -t 4:00:00
#SBATCH -o /das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/partiality/benchmark_partiality.out

"""Benchmark empirical partiality correction with the validated Poisson EM model.

Strategy:
  1. EM merge on full dataset → I0, alpha
  2. Rank-concordance orientation refinement → refined |r|
  3. Bin (resolution, |r|) → empirical log(p) per bin (no parametric model)
  4. Pass partiality into fit_em() via the `partiality` parameter,
     which scales the distribution: I_obs ~ Poisson(I0 * p * scale + bg)
  5. Split-half validation: apply shared partiality to halves, merge independently

Key: we scale the DISTRIBUTION, not the observation. This keeps Poisson
variance correct — dividing I_obs by p would amplify noise 1/p² times.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from torchsx.stream import SFXStream
from torchsx.merge import (
    fit_em, EMConfig, ScaleBCorrection,
    match_halves, shell_stats, cc_half, cc_star, r_split,
)
from torchsx.merge._hkl import compute_hkl_index
from torchsx.merge._noise import PoissonNoiseModel, compute_sigma2_LN
from torchsx.partiality._excitation import ExcitationError
from torchsx.partiality._pairs import PairData, RefineData, build_pairs, fit_BG
from torchsx.partiality._refine import refine_orientations_ev, RefinementConfig

# ── Configuration ────────────────────────────────────────────
STREAM_DIR = Path("/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams")
OUTDIR = Path("/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/partiality/results")
OUTDIR.mkdir(exist_ok=True)

DELAYS = [-91, -136, -182, -227]
CELL = torch.tensor([14.97, 18.85, 18.89, 89.37, 84.94, 67.84], dtype=torch.float32)
SEED = 42
N_OUTER = 1  # single pass: EM → refine → empirical table (no compounding)
N_SHELLS = 20
D_MIN, D_MAX = 0.9, 10.0

EM_CFG = EMConfig(n_em_cycles=10, min_obs=5, alpha_prior_lambda=0.0)
REFINE_CFG = RefinementConfig(
    n_steps=500,
    lr_omega=1e-4,
    lr_lambda=1e-5,
    beta_start=5.0,
    beta_end=30.0,
    restraint_omega=100.0,  # optimal from restraint scan (was 300)
    restraint_lambda=2e4,
    free_fraction=0.1,
    patience=100,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# ── Helpers ──────────────────────────────────────────────────
to_F = lambda I: I.clamp(min=0).sqrt()


def d_to_x(d):
    return 1.0 / np.asarray(d) ** 2


def setup_res_axis(ax):
    tick_d = np.array([10, 5, 3, 2, 1.5, 1.2, 1.0, 0.9])
    ax.set_xticks(d_to_x(tick_d))
    ax.set_xticklabels([f"{d:g}" for d in tick_d])
    ax.set_xlim(d_to_x(D_MAX), d_to_x(D_MIN))
    ax.set_xlabel("Resolution (Å)")


def compute_geometry(core):
    """Return UB, lambda_0, inv_d, s_sq, d_spacing from stream core."""
    UB = torch.stack([core.astar, core.bstar, core.cstar], dim=2)
    beam_E = core.beam_energy[core.image_id.long()]
    lambda_0 = 12398.42 / beam_E

    exc = ExcitationError(core.astar.shape[0], UB, lambda_0)
    with torch.no_grad():
        RUB = exc._all_RUB()
        q = torch.einsum("nij,nj->ni", RUB[core.crystal_id.long()], core.hkl.float())
    q_mag = q.norm(dim=1)
    inv_d = q_mag
    s_sq = (q_mag / 2) ** 2
    d_spacing = 1.0 / inv_d.clamp(min=1e-6)
    return UB, lambda_0, inv_d, s_sq, d_spacing


def fit_partiality_surface(
    I_obs, I0_per_obs, scale_per_obs, abs_r, s_sq,
    sigma2_poi, sigma2_LN_per_obs,
    max_r_order=3, max_s_order=2,
):
    """Fit smooth partiality surface log(p) = f(|r|, s²) via weighted LS.

    Basis: r_norm^k · s_norm^l for k=1..max_r_order, l=0..max_s_order.
    Inputs are normalized to [0, 1] for numerical stability.
    No constant term (absorbed by scale) and no k=0 terms (p=1 at |r|=0).

    Weight = 1/Var(log_residual) where:
        Var ≈ σ²_poi / (I0·scale)² + σ²_LN

    This downweights Poisson-noise-dominated weak observations.

    Returns
    -------
    partiality : (N_obs,) tensor, normalized so median=1
    coeffs : fitted polynomial coefficients
    basis_spec : list of (k, l) tuples describing the basis
    r_scale, s_scale : normalization constants (for evaluating on new data)
    """
    N = len(I_obs)

    # Valid: positive obs with reasonable I0 and scale
    c1 = I_obs > 0
    c2 = I0_per_obs > 0.5
    c3 = scale_per_obs > 0.01
    c4 = torch.isfinite(abs_r)
    c5 = torch.isfinite(I0_per_obs)
    c6 = abs_r > 0
    c7 = abs_r < 0.01
    print(f"  Filters: I>0={c1.sum():,} I0>0.5={c2.sum():,} scale>0.01={c3.sum():,} "
          f"r_finite={c4.sum():,} I0_finite={c5.sum():,} r>0={c6.sum():,} r<0.01={c7.sum():,}")
    valid = c1 & c2 & c3 & c4 & c5 & c6 & c7
    print(f"  Valid obs: {valid.sum():,} / {N:,}")

    # Log-space residuals
    log_resid = torch.zeros(N)
    log_resid[valid] = torch.log(
        I_obs[valid] / (I0_per_obs[valid] * scale_per_obs[valid])
    )

    # Weights: 1 / Var(log_residual)
    signal_sq = (I0_per_obs * scale_per_obs).clamp(min=1.0) ** 2
    var_log = sigma2_poi / signal_sq + sigma2_LN_per_obs.clamp(min=0.01)
    weights = torch.zeros(N)
    weights[valid] = 1.0 / var_log[valid].clamp(min=1e-6)

    # Normalize |r| and s² to [0, 1] for numerical stability
    r_scale = abs_r[valid].quantile(0.95).item()
    s_scale = s_sq[valid].quantile(0.95).item()
    r_scale = max(r_scale, 1e-6)
    s_scale = max(s_scale, 1e-6)
    r_norm = (abs_r / r_scale).clamp(max=2.0)
    s_norm = (s_sq / s_scale).clamp(max=2.0)

    print(f"  Normalization: r_scale={r_scale:.5f} Å⁻¹, s_scale={s_scale:.5f} Å⁻²")

    # Build basis: r_norm^k · s_norm^l for k ≥ 1
    basis_spec = []
    columns = []
    for k in range(1, max_r_order + 1):
        for l in range(0, max_s_order + 1):
            basis_spec.append((k, l))
            columns.append(r_norm.pow(k) * s_norm.pow(l))

    n_basis = len(columns)
    print(f"  Partiality surface: {n_basis} basis functions, "
          f"{valid.sum():,} valid obs")

    # Weighted normal equations: (X'WX) c = X'Wy
    w_v = weights[valid]
    y_v = log_resid[valid]
    cols_v = [col[valid] for col in columns]

    XtWX = torch.zeros(n_basis, n_basis)
    XtWy = torch.zeros(n_basis)
    for i in range(n_basis):
        wxi = w_v * cols_v[i]
        XtWy[i] = (wxi * y_v).sum()
        for j in range(i, n_basis):
            val = (wxi * cols_v[j]).sum()
            XtWX[i, j] = val
            XtWX[j, i] = val

    # Ridge regularization
    ridge = 1e-4 * XtWX.diag().mean()
    XtWX += ridge * torch.eye(n_basis)

    coeffs = torch.linalg.solve(XtWX, XtWy)

    for (k, l), c in zip(basis_spec, coeffs):
        label = f"r^{k}" + (f"·s^{l}" if l > 0 else "")
        print(f"    {label:>10s}: {c.item():+.4f}")

    # Evaluate on ALL observations
    log_p = torch.zeros(N)
    for idx, col in enumerate(columns):
        log_p += coeffs[idx] * col

    # Physical constraint: partiality ≤ 1 → clamp log(p) ≤ 0
    log_p = log_p.clamp(min=-3.0, max=0.0)

    # Normalize: subtract mean so it's shape-only
    log_p = log_p - log_p.mean()

    partiality = torch.exp(log_p)
    partiality = partiality / partiality.median()
    partiality = partiality.clamp(min=0.1, max=5.0)

    return partiality, coeffs, basis_spec, r_scale, s_scale


# ══════════════════════════════════════════════════════════════
# Load streams
# ══════════════════════════════════════════════════════════════
print("── Loading streams ──")
streams = [SFXStream.load(STREAM_DIR / f"delay_{d}.sfx") for d in DELAYS]
s = SFXStream.concat(streams)
print(f"Total: {s.n_observations:,} obs, {s.n_crystals:,} crystals")

h1, h2 = s.split_random(n_parts=2, seed=SEED)
print(f"Split: {h1.n_crystals}/{h2.n_crystals} crystals")

# ── Precompute geometry for full dataset ──
core = s._core
C = core.astar.shape[0]
N = core.hkl.shape[0]
UB, lambda_0, inv_d, s_sq, d_obs = compute_geometry(core)
hkl_idx = compute_hkl_index(core.hkl)
hkl_id = hkl_idx.hkl_id
n_unique = hkl_idx.n_unique
cid = core.crystal_id

print(f"Geometry: {n_unique} unique HKLs")

# ══════════════════════════════════════════════════════════════
# BASELINE: EM + ScaleB, no partiality
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BASELINE: EM Poisson without partiality")
print("=" * 60)

dist_base = {}
for label, half in [("half1", h1), ("half2", h2)]:
    print(f"\n  {label}...")
    d = fit_em(half, config=EM_CFG, corrections=[ScaleBCorrection()], device=device)
    dist_base[label] = d
    print(f"  med(I₀)={d.I0.median():.1f}, med(α)={d.alpha.median():.4f}")

m_base = match_halves(
    dist_base["half1"].hkl, to_F(dist_base["half1"].I0),
    dist_base["half2"].hkl, to_F(dist_base["half2"].I0),
    CELL, d_min=D_MIN, d_max=D_MAX,
)
cc_base = cc_half(m_base)
r_base = r_split(m_base)
print(f"\n  Baseline: CC1/2={cc_base:.4f}, R_split={r_base:.4f}, N={m_base.F1.numel()}")

# ══════════════════════════════════════════════════════════════
# PARTIALITY PIPELINE on full dataset
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PARTIALITY PIPELINE (empirical, full dataset)")
print("=" * 60)

exc_model = None
partiality = None  # (N_obs,) tensor in stream order
alpha_trajectory = []
concordance_trajectory = []
coeffs = None
basis_spec = None

for cycle in range(N_OUTER):
    print(f"\n{'─'*60}")
    print(f"OUTER CYCLE {cycle}")
    print(f"{'─'*60}")

    # ── Step 1: EM merge with partiality in the model ──
    print(f"\nStep 1: EM merge (partiality={'yes' if partiality is not None else 'no'})...")
    dist_full = fit_em(
        s, config=EM_CFG, corrections=[ScaleBCorrection()],
        device=device, partiality=partiality,
    )
    med_alpha = dist_full.alpha.median().item()
    alpha_trajectory.append(med_alpha)
    print(f"  med(I₀)={dist_full.I0.median():.1f}, med(α)={med_alpha:.4f}")

    # ── Step 2: Build observation data for explained-variance refinement ──
    print(f"\nStep 2: Building RefineData (log-residuals + Poisson weights)...")
    I_corr, log_G, B = fit_BG(core.I, hkl_id, cid, s_sq, n_unique, C)
    print(f"  median |log G|={log_G.abs().median():.3f}, median B={B.median():.1f}")

    # Map dist.I0 to per-obs
    _dist_hkl = dist_full.hkl.to(torch.int16)
    _I0_full = torch.full((n_unique,), dist_full.I0.median().item())
    _alpha_full = torch.full((n_unique,), dist_full.alpha.median().item())
    _lookup = {}
    for i in range(len(_dist_hkl)):
        _lookup[(_dist_hkl[i, 0].item(), _dist_hkl[i, 1].item(), _dist_hkl[i, 2].item())] = i
    for j in range(n_unique):
        idx = _lookup.get((hkl_idx.unique_hkl[j, 0].item(),
                           hkl_idx.unique_hkl[j, 1].item(),
                           hkl_idx.unique_hkl[j, 2].item()))
        if idx is not None:
            _I0_full[j] = dist_full.I0[idx]
            _alpha_full[j] = dist_full.alpha[idx]
    I0_per_obs = _I0_full[hkl_id]
    alpha_per_obs = _alpha_full[hkl_id]

    # Scale per obs from B/G correction
    scale_per_obs = torch.exp(log_G[cid.long()] + B[cid.long()] * (-2 * s_sq)).clamp(min=0.01)

    # Log-residuals (positive obs only, valid I0)
    pos_valid = (core.I > 0) & (I0_per_obs > 0.5) & (scale_per_obs > 0.01)
    log_resid = torch.zeros(N)
    log_resid[pos_valid] = torch.log(core.I[pos_valid] / (I0_per_obs[pos_valid] * scale_per_obs[pos_valid]))

    # Poisson+GenGauss weights: w = 1 / Var(log_resid)
    noise = PoissonNoiseModel()
    noise.gain = 1.0
    sigma2_poi = noise.sigma2(core.I, core.background)
    signal_sq = (I0_per_obs * scale_per_obs).clamp(min=1.0) ** 2
    var_log = sigma2_poi / signal_sq + alpha_per_obs.clamp(min=0.1) ** 2
    weights = torch.zeros(N)
    weights[pos_valid] = 1.0 / var_log[pos_valid].clamp(min=1e-6)

    n_valid_obs = pos_valid.sum().item()
    print(f"  {n_valid_obs:,} valid obs, median weight={weights[pos_valid].median():.4f}")

    refine_data = RefineData(
        hkl_obs=core.hkl, crystal_id=cid, hkl_id=hkl_id,
        log_resid=log_resid, weights=weights,
        UB=UB, lambda_0=lambda_0,
        n_crystals=C, n_hkl=n_unique,
    )

    # ── Step 3: Explained-variance orientation refinement ──
    print(f"\nStep 3: Explained-variance refinement (warm_start={exc_model is not None})...")
    REFINE_CFG.device = device
    refine_data_dev = refine_data.to(device)
    ref_result = refine_orientations_ev(refine_data_dev, REFINE_CFG, init_model=exc_model)
    exc_model = ref_result.model
    r2_work = ref_result.concordance_history[-1] if ref_result.concordance_history else 0
    r2_free = ref_result.free_concordance_history[-1] if ref_result.free_concordance_history else 0
    concordance_trajectory.append((r2_work, r2_free))
    print(f"  |δω| median: {ref_result.delta_omega_xy.norm(dim=1).median() * 1e3:.2f} mdeg")

    # ── Step 4: Fit smooth partiality surface ──
    print(f"\nStep 4: Fitting partiality surface log(p) = f(|r|, s²)...")
    with torch.no_grad():
        abs_r = exc_model.abs_excitation_error(
            core.hkl.to(device), cid.to(device),
        ).cpu()

    # Map dist.I0 and sigma2_LN to per-obs
    dist_hkl = dist_full.hkl.to(torch.int16)
    I0_full = torch.full((n_unique,), dist_full.I0.median().item())
    alpha_full = torch.full((n_unique,), dist_full.alpha.median().item())
    dist_lookup = {}
    for i in range(len(dist_hkl)):
        key = (dist_hkl[i, 0].item(), dist_hkl[i, 1].item(), dist_hkl[i, 2].item())
        dist_lookup[key] = i
    for j in range(n_unique):
        key = (hkl_idx.unique_hkl[j, 0].item(), hkl_idx.unique_hkl[j, 1].item(),
               hkl_idx.unique_hkl[j, 2].item())
        idx = dist_lookup.get(key)
        if idx is not None:
            I0_full[j] = dist_full.I0[idx]
            alpha_full[j] = dist_full.alpha[idx]
    I0_per_obs = I0_full[hkl_id]
    alpha_per_obs = alpha_full[hkl_id]

    # sigma2_LN from alpha (approx for beta~1.5: sigma2_LN ≈ alpha²)
    sigma2_LN_per_obs = alpha_per_obs ** 2

    # Scale from B/G correction
    scale_per_obs = torch.exp(
        log_G[cid.long()] + B[cid.long()] * (-2 * s_sq)
    ).clamp(min=0.01)

    # Poisson variance
    noise = PoissonNoiseModel()
    noise.gain = 1.0
    sigma2_poi = noise.sigma2(core.I, core.background)

    partiality, coeffs, basis_spec, r_scale, s_scale = fit_partiality_surface(
        core.I, I0_per_obs, scale_per_obs, abs_r, s_sq,
        sigma2_poi, sigma2_LN_per_obs,
    )

    print(f"  Partiality: median={partiality.median():.3f}, "
          f"p10={partiality.quantile(0.1):.3f}, p90={partiality.quantile(0.9):.3f}")

print(f"\nAlpha trajectory: {[f'{a:.4f}' for a in alpha_trajectory]}")
print(f"R² trajectory: {[f'work={w:.6f} free={f:.6f}' for w, f in concordance_trajectory]}")

# ══════════════════════════════════════════════════════════════
# SPLIT-HALF VALIDATION with partiality
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SPLIT-HALF: EM Poisson + Empirical Partiality")
print("=" * 60)

# Evaluate the fitted surface on each half's (|r|, s²) to get per-obs partiality.
# Use the half's own initial orientations for |r| (corrections are ~1 mdeg,
# negligible for the smooth surface).

dist_part = {}
for label, half in [("half1", h1), ("half2", h2)]:
    print(f"\n  {label}...")
    half_core = half._core

    half_UB, half_lam, half_inv_d, half_s_sq, half_d = compute_geometry(half_core)
    half_exc = ExcitationError(half_core.astar.shape[0], half_UB, half_lam)
    with torch.no_grad():
        half_abs_r = half_exc.abs_excitation_error(
            half_core.hkl, half_core.crystal_id
        )

    # Evaluate polynomial surface with same normalization
    r_n = (half_abs_r.float() / r_scale).clamp(max=2.0)
    s_n = (half_s_sq.float() / s_scale).clamp(max=2.0)
    log_p = torch.zeros(len(half_core.I))
    for idx, (k, l) in enumerate(basis_spec):
        log_p += coeffs[idx] * r_n.pow(k) * s_n.pow(l)
    log_p = log_p.clamp(min=-3.0, max=0.0)
    log_p = log_p - log_p.mean()

    half_partiality = torch.exp(log_p)
    half_partiality = half_partiality / half_partiality.median()
    half_partiality = half_partiality.clamp(min=0.1, max=5.0)
    print(f"  partiality: p10={half_partiality.quantile(0.1):.3f}, "
          f"p90={half_partiality.quantile(0.9):.3f}")

    d = fit_em(
        half, config=EM_CFG, corrections=[ScaleBCorrection()],
        device=device, partiality=half_partiality,
    )
    dist_part[label] = d
    print(f"  med(I₀)={d.I0.median():.1f}, med(α)={d.alpha.median():.4f}")

m_part = match_halves(
    dist_part["half1"].hkl, to_F(dist_part["half1"].I0),
    dist_part["half2"].hkl, to_F(dist_part["half2"].I0),
    CELL, d_min=D_MIN, d_max=D_MAX,
)
cc_part = cc_half(m_part)
r_part = r_split(m_part)
print(f"\n  Partiality: CC1/2={cc_part:.4f}, R_split={r_part:.4f}, N={m_part.F1.numel()}")

# ══════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"COMPARISON ({D_MIN}–{D_MAX} Å)")
print("=" * 65)
print(f"  {'Method':>30s}  {'CC1/2':>7s}  {'CC*':>7s}  {'R_split':>8s}  {'N':>6s}  {'med(α)':>8s}")
print(f"  {'-'*67}")

base_alpha = 0.5 * (dist_base["half1"].alpha.median().item() +
                     dist_base["half2"].alpha.median().item())
part_alpha = 0.5 * (dist_part["half1"].alpha.median().item() +
                     dist_part["half2"].alpha.median().item())

for name, m, alpha in [
    ("EM Poisson (baseline)", m_base, base_alpha),
    ("EM + empirical partiality", m_part, part_alpha),
]:
    cc = cc_half(m)
    print(f"  {name:>30s}  {cc:>7.4f}  {cc_star(cc):>7.4f}  {r_split(m):>8.4f}  "
          f"{m.F1.numel():>6d}  {alpha:>8.4f}")

delta_cc = cc_part - cc_base
delta_r = r_part - r_base
delta_alpha = part_alpha - base_alpha
print(f"\n  Delta: CC1/2={delta_cc:+.4f}, R_split={delta_r:+.4f}, alpha={delta_alpha:+.4f}")

# ══════════════════════════════════════════════════════════════
# Per-shell comparison
# ══════════════════════════════════════════════════════════════
shells_base = shell_stats(m_base, n_shells=N_SHELLS)
shells_part = shell_stats(m_part, n_shells=N_SHELLS)

print(f"\n{'='*80}")
print(f"PER-SHELL CC1/2")
print(f"{'='*80}")
print(f"  {'d_max':>7s} {'d_min':>7s}  {'Baseline':>10s}  {'Partiality':>10s}  {'Delta':>8s}")
print(f"  {'-'*50}")
for sb, sp in zip(shells_base, shells_part):
    delta = sp.cc_half - sb.cc_half
    print(f"  {sb.d_max:>7.2f} {sb.d_min:>7.2f}  {sb.cc_half:>10.4f}  {sp.cc_half:>10.4f}  {delta:>+8.4f}")

print(f"\n{'='*80}")
print(f"PER-SHELL R_split")
print(f"{'='*80}")
print(f"  {'d_max':>7s} {'d_min':>7s}  {'Baseline':>10s}  {'Partiality':>10s}  {'Delta':>8s}")
print(f"  {'-'*50}")
for sb, sp in zip(shells_base, shells_part):
    delta = sp.r_split - sb.r_split
    print(f"  {sb.d_max:>7.2f} {sb.d_min:>7.2f}  {sb.r_split:>10.4f}  {sp.r_split:>10.4f}  {delta:>+8.4f}")

# CSV
csv_path = OUTDIR / "benchmark_partiality_shells.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "d_max", "d_min", "n_matched", "cc_half", "cc_star", "r_split"])
    for name, shells in [("Baseline", shells_base), ("Partiality", shells_part)]:
        for sh in shells:
            w.writerow([name, f"{sh.d_max:.4f}", f"{sh.d_min:.4f}",
                        sh.n_matched, f"{sh.cc_half:.6f}", f"{sh.cc_star:.6f}",
                        f"{sh.r_split:.6f}"])
print(f"\nWrote {csv_path}")

# ══════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════

# Plot 1: CC1/2 and R_split vs resolution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
for name, shells, color, marker in [
    ("Baseline", shells_base, "C0", "o"),
    ("+ Emp. Partiality", shells_part, "C3", "s"),
]:
    d_mid = [0.5 * (s.d_max + s.d_min) for s in shells]
    kw = dict(color=color, marker=marker, ms=5, lw=1.8, label=name)
    ax1.plot(d_to_x(d_mid), [s.cc_half for s in shells], "-", **kw)
    ax2.plot(d_to_x(d_mid), [s.r_split for s in shells], "-", **kw)

ax1.set_ylabel("CC1/2"); ax1.set_title("CC1/2 vs Resolution")
ax1.legend(loc="lower left"); ax1.set_ylim(-0.05, 1.05); ax1.grid(True, alpha=0.3)
setup_res_axis(ax1)
ax2.set_ylabel("R_split"); ax2.set_title("R_split vs Resolution")
ax2.legend(loc="upper left"); ax2.set_ylim(0, None); ax2.grid(True, alpha=0.3)
setup_res_axis(ax2)
fig.suptitle("Empirical Partiality: Precision Impact", fontsize=13)
fig.tight_layout()
fig.savefig(OUTDIR / "benchmark_partiality.png", dpi=150)
print(f"Saved benchmark_partiality.png")
plt.close(fig)

# Plot 2: Fitted partiality surface
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Evaluate surface on a grid for visualization (using normalized coords)
r_grid = np.linspace(0, 0.006, 100)
s2_vals = [0.01, 0.04, 0.09, 0.16, 0.25]  # ~ d=5, 2.5, 1.7, 1.25, 1.0 Å
for s2v in s2_vals:
    d_approx = 1.0 / (2 * np.sqrt(s2v))
    r_n = np.clip(r_grid / r_scale, 0, 2)
    s_n = np.clip(s2v / s_scale, 0, 2)
    log_p_curve = np.zeros(len(r_grid))
    for idx, (k, l) in enumerate(basis_spec):
        log_p_curve += coeffs[idx].item() * r_n**k * s_n**l
    log_p_curve = np.clip(log_p_curve, -3, 0)
    log_p_curve -= log_p_curve[0]  # normalize so p(0)=1
    ax1.plot(r_grid * 1e3, log_p_curve, lw=2,
             label=f"d={d_approx:.1f} Å")
ax1.set_xlabel("|r| (×10⁻³ Å⁻¹)"); ax1.set_ylabel("log(p)")
ax1.set_title("Fitted Partiality Surface")
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
ax1.set_ylim(-3, 0.5)

# Partiality distribution histogram
ax2.hist(partiality.numpy().clip(0, 3), bins=100, color="C0", alpha=0.7)
ax2.axvline(1.0, color="red", ls="--", lw=1.5)
ax2.set_xlabel("Partiality (normalized)")
ax2.set_ylabel("Count")
ax2.set_title(f"Partiality Distribution\nmed={partiality.median():.3f}, "
              f"p10={partiality.quantile(0.1):.3f}, p90={partiality.quantile(0.9):.3f}")

fig.tight_layout()
fig.savefig(OUTDIR / "partiality_trajectory.png", dpi=150)
print(f"Saved partiality_trajectory.png")
plt.close(fig)

# Plot 3: Alpha vs resolution comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for ax, (d1, label) in zip([ax1, ax2], [
    (dist_base["half1"], "Baseline"),
    (dist_part["half1"], "With Partiality"),
]):
    d_sp = d1.d_spacing.numpy()
    alpha = d1.alpha.numpy()
    I0 = d1.I0.numpy()

    sc = ax.scatter(d_to_x(d_sp), alpha, c=np.log10(I0.clip(0.1)),
                    s=3, alpha=0.4, cmap="viridis", vmin=-1, vmax=4, rasterized=True)
    plt.colorbar(sc, ax=ax, label="log₁₀(I₀)", shrink=0.8)

    edges = np.percentile(d_sp, np.linspace(0, 100, N_SHELLS + 1))
    d_mid, a_med = [], []
    for i in range(N_SHELLS):
        mask = (d_sp >= edges[i]) & (d_sp < edges[i + 1])
        if mask.sum() < 2:
            continue
        d_mid.append(0.5 * (edges[i] + edges[i + 1]))
        a_med.append(np.median(alpha[mask]))
    ax.plot(d_to_x(d_mid), a_med, "r-o", ms=4, lw=2, label=f"median α={np.median(alpha):.3f}")
    ax.set_ylabel("α"); ax.set_title(label, fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 3.5)
    setup_res_axis(ax)

fig.suptitle("Alpha vs Resolution: Effect of Partiality", fontsize=13)
fig.tight_layout()
fig.savefig(OUTDIR / "partiality_alpha_comparison.png", dpi=150)
print(f"Saved partiality_alpha_comparison.png")
plt.close(fig)

# Save results
torch.save({
    "alpha_trajectory": alpha_trajectory,
    "concordance_trajectory": concordance_trajectory,
    "cc_base": cc_base, "r_base": r_base,
    "cc_part": cc_part, "r_part": r_part,
    "base_alpha": base_alpha, "part_alpha": part_alpha,
    "surface_coeffs": coeffs,
    "basis_spec": basis_spec,
}, OUTDIR / "benchmark_partiality_summary.pt")

print("\nDone.")
