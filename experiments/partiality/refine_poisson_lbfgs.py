#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -p gpu-day
#SBATCH -t 4:00:00
#SBATCH -o /das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/partiality/refine_poisson_lbfgs.out

"""Joint orientation + free-form partiality refinement via Poisson+GenGauss marginal.

Model per observation j of HKL h from crystal c:

    |r_j| = excitation_error(δω_c, hkl_j)          [differentiable]
    σ_r_j = |q_j| × σ_ω                             [resolution-dependent]
    p_eff_j = ∫ p_relp(r) × G(r; |r_j|, σ_r_j) dr  [Gauss-Hermite quadrature]
    λ_j = I0_h × p_eff_j × scale_j + bg_j           [expected photon count]
    n_obs_j ~ Poisson(λ_j)                           [observation model]

Jointly optimise:
    δω      (per crystal, 2 params each)
    p_relp  (free-form monotone profile, K control points)
    σ_ω     (global angular spread)

I0 is fixed from the EM merge (no circularity).
Free-set monitoring (10% held-out observations) for early stopping.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torchsx.stream import SFXStream
from torchsx.merge import (
    fit_em, EMConfig, ScaleBCorrection,
    match_halves, shell_stats, cc_half, cc_star, r_split,
)
from torchsx.merge._hkl import compute_hkl_index
from torchsx.merge._noise import PoissonNoiseModel
from torchsx.partiality._excitation import ExcitationError
from torchsx.partiality._pairs import fit_BG

# ── Configuration ────────────────────────────────────────────
STREAM_DIR = Path("/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams")
OUTDIR = Path("/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/partiality/results")
OUTDIR.mkdir(exist_ok=True)

DELAYS = [-91, -136, -182, -227]
CELL = torch.tensor([14.97, 18.85, 18.89, 89.37, 84.94, 67.84], dtype=torch.float32)
SEED = 42
D_MIN, D_MAX = 0.9, 10.0
N_SHELLS = 20

EM_CFG = EMConfig(n_em_cycles=10, min_obs=5, alpha_prior_lambda=0.0)

# Refinement
N_STEPS = 500
LR_OMEGA = 1e-4
LR_PROFILE = 1e-2
LR_SIGMA = 1e-3
LR_ALPHA = 1e-2
LAMBDA_ALPHA = 0.1    # restraint encouraging alpha to sharpen
RESTRAINT_OMEGA = 100.0
RESTRAINT_LAMBDA = 2e4
FREE_FRACTION = 0.1
PATIENCE = 100
LOG_EVERY = 20
N_QUAD = 11        # Gauss-Hermite quadrature points
N_R_CONTROL = 20   # control points for relp profile
R_MAX = 0.008      # max |r| for profile (Å⁻¹)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# ══════════════════════════════════════════════════════════════
# Free-form monotone relp profile
# ══════════════════════════════════════════════════════════════

class FreeFormProfile(nn.Module):
    """Monotone decreasing relp profile p(r) with p(0)=1, p(∞)→0.

    Parameterised at K control points r_1 < r_2 < ... < r_K.
    log(p) at control point k = -Σ_{i=1}^{k} softplus(δ_i),
    ensuring monotonicity.  Linear interpolation between points.
    """

    def __init__(self, n_points: int = 20, r_max: float = 0.008):
        super().__init__()
        self.register_buffer(
            "r_control",
            torch.linspace(0, r_max, n_points + 1),  # includes r=0
        )
        # Free parameters: increments between control points.
        # Initialise BROAD (p ≈ 1 everywhere) so the model starts near
        # the no-partiality baseline.  softplus(-3) ≈ 0.05, so cumulative
        # log_p at r_max ≈ -0.05 * K ≈ -1 → p ≈ 0.37 at r_max.
        self.raw_increments = nn.Parameter(
            torch.full((n_points,), -3.0)
        )

    def log_p_at_control(self) -> Tensor:
        """(K+1,) log(p) at control points, starting with 0 at r=0."""
        increments = F.softplus(self.raw_increments)
        cum = increments.cumsum(0)
        return torch.cat([torch.zeros(1, device=cum.device), -cum])

    def log_partiality(self, r: Tensor) -> Tensor:
        """Evaluate log(p) at arbitrary r >= 0 via linear interpolation."""
        log_p = self.log_p_at_control()
        rp = self.r_control

        # Clamp r to [0, r_max] and find interval
        r_clamped = r.clamp(0, rp[-1])
        # Index into intervals
        idx = torch.searchsorted(rp, r_clamped) - 1
        idx = idx.clamp(0, len(rp) - 2)

        # Linear interpolation weight
        dr = (rp[idx + 1] - rp[idx]).clamp(min=1e-10)
        t = (r_clamped - rp[idx]) / dr
        t = t.clamp(0, 1)

        return log_p[idx] * (1 - t) + log_p[idx + 1] * t


# ══════════════════════════════════════════════════════════════
# Gauss-Hermite quadrature for orientation uncertainty
# ══════════════════════════════════════════════════════════════

def gauss_hermite_points(n: int):
    """Return (t, w) for Gauss-Hermite quadrature of ∫f(x)N(x;0,1)dx.

    ∫f(x) N(x;μ,σ) dx ≈ Σ_k w_k × f(μ + σ√2 × t_k)
    """
    from numpy.polynomial.hermite import hermgauss
    t_np, w_np = hermgauss(n)
    # Normalise: standard Hermite weights include exp(-t²),
    # but we want the N(0,1) weighting → divide by √π
    w_np = w_np / math.sqrt(math.pi)
    return torch.from_numpy(t_np).float(), torch.from_numpy(w_np).float()


def compute_p_eff(
    profile: FreeFormProfile,
    abs_r: Tensor,
    sigma_r: Tensor,
    quad_t: Tensor,
    quad_w: Tensor,
) -> Tensor:
    """Effective partiality = ∫ p(r) × G(r; |r₀|, σ_r) dr.

    Uses Gauss-Hermite quadrature:
        p_eff = Σ_k w_k × p(|r₀| + σ_r × √2 × t_k)

    Parameters
    ----------
    abs_r : (N,) nominal |r| (differentiable)
    sigma_r : (N,) per-obs σ_r = |q| × σ_ω
    quad_t, quad_w : (Q,) quadrature points and weights

    Returns (N,) effective partiality (linear scale, 0 < p ≤ 1).
    """
    # (N, Q) quadrature grid: r_0 + σ_r * √2 * t_k
    sqrt2 = math.sqrt(2.0)
    r_grid = abs_r[:, None] + sigma_r[:, None] * sqrt2 * quad_t[None, :]
    r_grid = r_grid.clamp(min=0)  # |r| can't be negative

    # Evaluate profile at all quadrature points
    log_p_grid = profile.log_partiality(r_grid)  # (N, Q)

    # Weighted sum in log-space for numerical stability
    # p_eff = Σ w_k exp(log_p_k)
    # Use logsumexp trick: log(p_eff) = logsumexp(log_p_k + log(w_k))
    log_w = torch.log(quad_w.clamp(min=1e-30))[None, :]  # (1, Q)
    log_p_eff = torch.logsumexp(log_p_grid + log_w, dim=1)  # (N,)

    return torch.exp(log_p_eff).clamp(min=1e-6, max=1.0)


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

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
    """Return UB, lambda_0, inv_d, s_sq, q_mag."""
    UB = torch.stack([core.astar, core.bstar, core.cstar], dim=2)
    beam_E = core.beam_energy[core.image_id.long()]
    lambda_0 = 12398.42 / beam_E
    exc = ExcitationError(core.astar.shape[0], UB, lambda_0)
    with torch.no_grad():
        RUB = exc._all_RUB()
        q = torch.einsum("nij,nj->ni", RUB[core.crystal_id.long()], core.hkl.float())
    q_mag = q.norm(dim=1)
    return UB, lambda_0, q_mag


# ══════════════════════════════════════════════════════════════
# Load data and run baseline EM
# ══════════════════════════════════════════════════════════════

print("── Loading streams ──")
streams = [SFXStream.load(STREAM_DIR / f"delay_{d}.sfx") for d in DELAYS]
s = SFXStream.concat(streams)
h1, h2 = s.split_random(n_parts=2, seed=SEED)
print(f"Total: {s.n_observations:,} obs, {s.n_crystals:,} crystals")
print(f"Split: {h1.n_crystals}/{h2.n_crystals}")

core = s._core
C = core.astar.shape[0]
N = core.hkl.shape[0]
UB, lambda_0, q_mag = compute_geometry(core)
hkl_idx = compute_hkl_index(core.hkl)
hkl_id = hkl_idx.hkl_id
n_unique = hkl_idx.n_unique
cid = core.crystal_id
s_sq = (q_mag / 2) ** 2

print(f"Geometry: {n_unique} unique HKLs, q_mag range [{q_mag.min():.3f}, {q_mag.max():.3f}] Å⁻¹")

# ── Baseline EM (split-half) ──
print("\n" + "=" * 60)
print("BASELINE: EM Poisson (no partiality)")
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
cc_base, r_base = cc_half(m_base), r_split(m_base)
base_alpha = 0.5 * (dist_base["half1"].alpha.median().item() +
                     dist_base["half2"].alpha.median().item())
print(f"\n  Baseline: CC1/2={cc_base:.4f}, R_split={r_base:.4f}, α={base_alpha:.4f}")

# ── Full-dataset EM for I0 reference ──
print("\n── Full-dataset EM for I0 ──")
dist_full = fit_em(s, config=EM_CFG, corrections=[ScaleBCorrection()], device=device)
print(f"  med(I₀)={dist_full.I0.median():.1f}, med(α)={dist_full.alpha.median():.4f}")

# Map I0 to per-obs
dist_hkl = dist_full.hkl.to(torch.int16)
I0_full = torch.full((n_unique,), dist_full.I0.median().item())
_lookup = {}
for i in range(len(dist_hkl)):
    _lookup[(dist_hkl[i, 0].item(), dist_hkl[i, 1].item(), dist_hkl[i, 2].item())] = i
for j in range(n_unique):
    idx = _lookup.get((hkl_idx.unique_hkl[j, 0].item(),
                       hkl_idx.unique_hkl[j, 1].item(),
                       hkl_idx.unique_hkl[j, 2].item()))
    if idx is not None:
        I0_full[j] = dist_full.I0[idx]
I0_per_obs = I0_full[hkl_id]

# Scale from B/G correction
I_corr, log_G, B = fit_BG(core.I, hkl_id, cid, s_sq, n_unique, C)
scale_per_obs = torch.exp(log_G[cid.long()] + B[cid.long()] * (-2 * s_sq)).clamp(min=0.01)

# Background (total in peak region)
noise_model = PoissonNoiseModel()
bg_total = core.background * noise_model.n_peak  # bg_per_pixel × n_peak
n_photons = (core.I + bg_total).clamp(min=0)  # raw peak counts

print(f"  bg_total: median={bg_total.median():.1f}, n_ph: median={n_photons.median():.1f}")

# ══════════════════════════════════════════════════════════════
# JOINT POISSON + GENGAUSS MARGINAL REFINEMENT
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("JOINT POISSON+GENGAUSS MARGINAL REFINEMENT")
print("=" * 60)

from torchsx.merge._dstep import build_softplus_grid, _log_gg_norm

# Move to device
hkl_obs_d = core.hkl.to(device)
cid_d = cid.to(device)
hkl_id_d = hkl_id.to(device)
I0_d = I0_per_obs.to(device)
scale_d = scale_per_obs.to(device)
bg_d = bg_total.to(device)
n_ph_d = n_photons.to(device)
q_mag_d = q_mag.to(device)

# Per-obs alpha and beta from the EM merge (fixed)
alpha_full_arr = torch.full((n_unique,), dist_full.alpha.median().item())
beta_full_arr = torch.full((n_unique,), 1.5)  # default beta
for i in range(len(dist_hkl)):
    idx = _lookup.get((dist_hkl[i, 0].item(), dist_hkl[i, 1].item(), dist_hkl[i, 2].item()))
    if idx is not None:
        alpha_full_arr[idx] = dist_full.alpha[i]
        beta_full_arr[idx] = dist_full.beta[i]
alpha_per_obs = alpha_full_arr[hkl_id].to(device)
beta_per_obs = beta_full_arr[hkl_id].to(device)
# log_gg_norm computed per-step since alpha_eff is learnable

# Build softplus grid (I0-independent, pre-computed once)
N_GG_QUAD = 31  # GenGauss quadrature points (lighter than 51 for speed)
print(f"  Building softplus grid ({N_GG_QUAD} points)...")
with torch.no_grad():
    sig_lo = torch.full_like(n_ph_d, 0.5)  # conservative floor
    grid = build_softplus_grid(n_ph_d, bg_d, n_quad=N_GG_QUAD, signal_lo_override=sig_lo)
log_poi_jac = grid.log_poi + grid.log_jac_dt + grid.log_dt[:, None]

# Initialise model
exc_model = ExcitationError(C, UB, lambda_0).to(device)
profile = FreeFormProfile(n_points=N_R_CONTROL, r_max=R_MAX).to(device)
# Global orientation uncertainty: initialise to ~10 mdeg (in radians)
log_sigma_omega = nn.Parameter(
    torch.tensor(math.log(10e-3 * math.pi / 180), device=device)
)
# Learnable alpha scale: α_eff = α_EM × exp(log_alpha_scale)
# Starts at 1.0 (no change). Penalty encourages it to decrease.
log_alpha_scale = nn.Parameter(torch.tensor(0.0, device=device))

# Gauss-Hermite quadrature for orientation uncertainty
quad_t, quad_w = gauss_hermite_points(N_QUAD)
quad_t, quad_w = quad_t.to(device), quad_w.to(device)

# Work/free split
gen = torch.Generator(device=device).manual_seed(42)
free_mask = torch.rand(N, generator=gen, device=device) < FREE_FRACTION
work_mask = ~free_mask
n_work, n_free = work_mask.sum().item(), free_mask.sum().item()

# Use Adam (LBFGS closure is called multiple times → expensive with 6.5M × 31 grid)
optimizer = torch.optim.Adam([
    {"params": [exc_model.delta_omega_xy], "lr": LR_OMEGA},
    {"params": [exc_model.delta_lambda], "lr": 1e-5},
    {"params": profile.parameters(), "lr": LR_PROFILE},
    {"params": [log_sigma_omega], "lr": LR_SIGMA},
    {"params": [log_alpha_scale], "lr": LR_ALPHA},
])

print(f"Observations: {n_work:,} work / {n_free:,} free")
print(f"Parameters: {C*2} δω + {C} δλ + {N_R_CONTROL} profile + 1 σ_ω")
print(f"Restraints: λ_ω={RESTRAINT_OMEGA}, λ_λ={RESTRAINT_LAMBDA}")
print(f"Likelihood: Poisson × GenGauss marginal ({N_GG_QUAD} quadrature points)")
print(f"  med(α_EM)={alpha_per_obs.median():.3f}, med(β)={beta_per_obs.median():.3f}")
print(f"  α sharpening penalty: λ_α={LAMBDA_ALPHA}")

best_free_nll = float("inf")
best_step = 0
best_state = None
nll_work_hist, nll_free_hist = [], []
sigma_hist = []

for step in range(N_STEPS):
    optimizer.zero_grad()

    # Compute |r| and effective partiality (differentiable)
    abs_r = exc_model.abs_excitation_error(hkl_obs_d, cid_d)
    sigma_omega = torch.exp(log_sigma_omega)
    sigma_r = q_mag_d * sigma_omega
    p_eff = compute_p_eff(profile, abs_r, sigma_r, quad_t, quad_w)

    # Poisson+GenGauss marginal likelihood:
    # P(n_obs) = ∫ Poisson(n | signal(ε) + bg) × f_GG(ε; α_eff, β) dε
    # where signal(ε) = I0 × p_eff × scale × exp(ε)
    # and α_eff = α_EM × alpha_scale (learnable)
    alpha_scale = torch.exp(log_alpha_scale)
    alpha_eff = alpha_per_obs * alpha_scale

    log_signal_model = torch.log((I0_d * p_eff * scale_d).clamp(min=1e-30))
    eps = torch.log(grid.signal.clamp(min=1e-30)) - log_signal_model[:, None]  # (N, K)
    eps = eps.clamp(-5.0, 5.0)

    eps_abs = eps.abs().clamp(min=1e-6)
    log_gg = (-(eps_abs / alpha_eff[:, None]).pow(beta_per_obs[:, None])
              + _log_gg_norm(beta_per_obs)[:, None]
              - torch.log(alpha_eff[:, None]))  # GenGauss normalization includes 1/α
    log_integrand = log_gg + log_poi_jac
    log_marginal = torch.logsumexp(log_integrand, dim=1)  # (N,)

    # Work-set negative log-likelihood
    nll_work = -(log_marginal * work_mask).sum() / n_work

    # Restraints
    omega_pen = RESTRAINT_OMEGA * (exc_model.delta_omega_xy ** 2).mean()
    lambda_pen = RESTRAINT_LAMBDA * (exc_model.delta_lambda ** 2).mean()
    # Alpha sharpening penalty: encourage alpha_scale < 1
    alpha_pen = LAMBDA_ALPHA * alpha_scale

    loss = nll_work + omega_pen + lambda_pen + alpha_pen

    if torch.isnan(loss) or torch.isinf(loss):
        print(f"  step {step}: NaN/Inf loss detected!")
        print(f"    abs_r: nan={abs_r.isnan().sum()} min={abs_r.min():.6f}")
        print(f"    p_eff: nan={p_eff.isnan().sum()} min={p_eff.min():.6f} max={p_eff.max():.6f}")
        print(f"    eps: nan={eps.isnan().sum()} min={eps.min():.2f} max={eps.max():.2f}")
        print(f"    log_gg: nan={log_gg.isnan().sum()} min={log_gg.min():.2f}")
        print(f"    log_marginal: nan={log_marginal.isnan().sum()} min={log_marginal.min():.2f}")
        print(f"    σ_ω={torch.exp(log_sigma_omega).item():.6f}")
        break

    loss.backward()

    # Check for NaN gradients
    all_grads = []
    for p in list(exc_model.parameters()) + list(profile.parameters()) + [log_sigma_omega, log_alpha_scale]:
        if p.grad is not None:
            all_grads.append(p.grad)
    has_nan_grad = any(g.isnan().any() for g in all_grads)
    if has_nan_grad:
        # Print which parameter has NaN gradient
        for name, p in [("delta_omega_xy", exc_model.delta_omega_xy),
                        ("delta_lambda", exc_model.delta_lambda),
                        ("profile", profile.raw_increments),
                        ("log_sigma_omega", log_sigma_omega),
                        ("log_alpha_scale", log_alpha_scale)]:
            if p.grad is not None:
                n_nan = p.grad.isnan().sum().item()
                if n_nan > 0:
                    print(f"  NaN grad in {name}: {n_nan}/{p.grad.numel()}")
        print(f"  Skipping step {step} due to NaN gradients")
        optimizer.zero_grad()
        continue

    torch.nn.utils.clip_grad_norm_(all_grads, max_norm=10.0)
    optimizer.step()

    if step % LOG_EVERY == 0 or step == N_STEPS - 1:
        with torch.no_grad():
            nll_f = -(log_marginal * free_mask).sum().item() / n_free
            nll_w = nll_work.item()
            sigma_deg = math.degrees(sigma_omega.item()) * 1000
            dw_med = exc_model.delta_omega_xy.norm(dim=1).median().item() * 1e3
            omega_pen_val = RESTRAINT_OMEGA * (exc_model.delta_omega_xy ** 2).mean().item()
            p_at_gamma = torch.exp(profile.log_partiality(
                torch.tensor([0.003], device=device))).item()

        nll_work_hist.append(nll_w)
        nll_free_hist.append(nll_f)
        sigma_hist.append(sigma_deg)

        if nll_f < best_free_nll:
            best_free_nll = nll_f
            best_step = step
            best_state = {
                "exc": {k: v.detach().clone() for k, v in exc_model.state_dict().items()},
                "profile": {k: v.detach().clone() for k, v in profile.state_dict().items()},
                "log_sigma": log_sigma_omega.detach().clone(),
                "log_alpha_scale": log_alpha_scale.detach().clone(),
            }

        alpha_sc = torch.exp(log_alpha_scale).item()

        print(f"  step {step:>4}: NLL work={nll_w:.2f} free={nll_f:.2f}  "
              f"α_scale={alpha_sc:.3f}  σ_ω={sigma_deg:.1f} mdeg  "
              f"|δω|={dw_med:.2f} mdeg  p(γ)={p_at_gamma:.3f}")

        if PATIENCE > 0 and step - best_step > PATIENCE:
            print(f"  Early stop: free NLL peaked at step {best_step}")
            break

# Restore best
if best_state is not None:
    exc_model.load_state_dict(best_state["exc"])
    profile.load_state_dict(best_state["profile"])
    log_sigma_omega.data.copy_(best_state["log_sigma"])
    log_alpha_scale.data.copy_(best_state["log_alpha_scale"])
    print(f"  Restored best (step {best_step}, free NLL={best_free_nll:.2f})")

sigma_final = math.degrees(torch.exp(log_sigma_omega).item()) * 1000
alpha_scale_final = torch.exp(log_alpha_scale).item()
print(f"\n  Final σ_ω = {sigma_final:.1f} mdeg")
print(f"  Final α_scale = {alpha_scale_final:.4f} (α_eff = {alpha_scale_final * dist_full.alpha.median().item():.4f})")
print(f"  Final |δω| median = {exc_model.delta_omega_xy.detach().norm(dim=1).median().item()*1e3:.2f} mdeg")

# ══════════════════════════════════════════════════════════════
# Apply refined partiality to split-half merge
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SPLIT-HALF with refined partiality")
print("=" * 60)

# Compute partiality for full dataset
with torch.no_grad():
    abs_r_final = exc_model.abs_excitation_error(hkl_obs_d, cid_d)
    sigma_r_final = q_mag_d * torch.exp(log_sigma_omega)
    p_final = compute_p_eff(profile, abs_r_final, sigma_r_final, quad_t, quad_w).cpu()

# Normalise for use with fit_em's partiality parameter
p_normed = p_final / p_final.median()
p_normed = p_normed.clamp(min=0.1, max=5.0)
print(f"  Partiality: median={p_final.median():.3f}, p10={p_final.quantile(0.1):.3f}, "
      f"p90={p_final.quantile(0.9):.3f}")

# For each half: compute per-half partiality from the SAME exc_model
# Since split_random remaps crystal IDs, we need per-half |r|.
# BUT the exc_model uses full-dataset crystal indices.
# Solution: compute p for the full dataset in stream order,
# then split preserves the per-obs assignment.

# Approach: use fit_em's partiality param on full stream copies
dist_part = {}
for label, half in [("half1", h1), ("half2", h2)]:
    print(f"\n  {label}...")
    # Compute geometry for this half
    half_core = half._core
    half_UB = torch.stack([half_core.astar, half_core.bstar, half_core.cstar], dim=2)
    half_lam = 12398.42 / half_core.beam_energy[half_core.image_id.long()]
    half_exc = ExcitationError(half_core.astar.shape[0], half_UB, half_lam).to(device)
    with torch.no_grad():
        half_abs_r = half_exc.abs_excitation_error(
            half_core.hkl.to(device), half_core.crystal_id.to(device))
        half_RUB = half_exc._all_RUB()
        half_q = torch.einsum("nij,nj->ni",
                              half_RUB[half_core.crystal_id.long().to(device)],
                              half_core.hkl.float().to(device))
        half_q_mag = half_q.norm(dim=1)
        half_sigma_r = half_q_mag * torch.exp(log_sigma_omega)
        half_p = compute_p_eff(profile, half_abs_r, half_sigma_r, quad_t, quad_w).cpu()

    half_p_normed = half_p / half_p.median()
    half_p_normed = half_p_normed.clamp(min=0.1, max=5.0)
    print(f"  partiality: p10={half_p_normed.quantile(0.1):.3f}, "
          f"p90={half_p_normed.quantile(0.9):.3f}")

    d = fit_em(half, config=EM_CFG, corrections=[ScaleBCorrection()],
               device=device, partiality=half_p_normed)
    dist_part[label] = d
    print(f"  med(I₀)={d.I0.median():.1f}, med(α)={d.alpha.median():.4f}")

m_part = match_halves(
    dist_part["half1"].hkl, to_F(dist_part["half1"].I0),
    dist_part["half2"].hkl, to_F(dist_part["half2"].I0),
    CELL, d_min=D_MIN, d_max=D_MAX,
)
cc_part, r_part = cc_half(m_part), r_split(m_part)
part_alpha = 0.5 * (dist_part["half1"].alpha.median().item() +
                     dist_part["half2"].alpha.median().item())

# ══════════════════════════════════════════════════════════════
# Results
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print(f"COMPARISON ({D_MIN}–{D_MAX} Å)")
print("=" * 65)
print(f"  {'Method':>30s}  {'CC1/2':>7s}  {'CC*':>7s}  {'R_split':>8s}  {'N':>6s}  {'med(α)':>8s}")
print(f"  {'-'*67}")
for name, m, alpha in [
    ("EM Poisson (baseline)", m_base, base_alpha),
    ("EM + Poisson partiality", m_part, part_alpha),
]:
    cc = cc_half(m)
    print(f"  {name:>30s}  {cc:>7.4f}  {cc_star(cc):>7.4f}  {r_split(m):>8.4f}  "
          f"{m.F1.numel():>6d}  {alpha:>8.4f}")

delta_cc = cc_part - cc_base
delta_r = r_part - r_base
delta_alpha = part_alpha - base_alpha
print(f"\n  Delta: CC1/2={delta_cc:+.4f}, R_split={delta_r:+.4f}, alpha={delta_alpha:+.4f}")
print(f"  σ_ω = {sigma_final:.1f} mdeg")

# Per-shell
shells_base = shell_stats(m_base, n_shells=N_SHELLS)
shells_part = shell_stats(m_part, n_shells=N_SHELLS)
print(f"\n{'='*80}")
print(f"PER-SHELL CC1/2")
print(f"{'='*80}")
print(f"  {'d_max':>7s} {'d_min':>7s}  {'Baseline':>10s}  {'Poisson':>10s}  {'Delta':>8s}")
print(f"  {'-'*50}")
for sb, sp in zip(shells_base, shells_part):
    delta = sp.cc_half - sb.cc_half
    print(f"  {sb.d_max:>7.2f} {sb.d_min:>7.2f}  {sb.cc_half:>10.4f}  {sp.cc_half:>10.4f}  {delta:>+8.4f}")

# ══════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. NLL trajectory
ax = axes[0, 0]
steps = np.arange(len(nll_work_hist)) * LOG_EVERY
ax.plot(steps, nll_work_hist, "C0-", lw=2, label="Work")
ax.plot(steps, nll_free_hist, "C3--", lw=2, label="Free")
ax.axvline(best_step, color="gray", ls=":", lw=1)
ax.set_xlabel("Step"); ax.set_ylabel("NLL / obs")
ax.set_title("Poisson NLL"); ax.legend(); ax.grid(True, alpha=0.3)

# 2. Fitted profile
ax = axes[0, 1]
with torch.no_grad():
    r_plot = torch.linspace(0, R_MAX, 200, device=device)
    log_p_plot = profile.log_partiality(r_plot).cpu().numpy()
    # Also show the convolved profile at different resolutions
    for q_val, label in [(0.5, "d=2.0 Å"), (0.8, "d=1.25 Å"), (1.1, "d=0.9 Å")]:
        sr = q_val * torch.exp(log_sigma_omega).item()
        p_conv = compute_p_eff(
            profile,
            r_plot,
            torch.full_like(r_plot, sr),
            quad_t, quad_w,
        ).cpu().numpy()
        ax.plot(r_plot.cpu() * 1e3, p_conv, "--", lw=1.5, label=f"convolved ({label})")

ax.plot(r_plot.cpu() * 1e3, np.exp(log_p_plot), "k-", lw=2, label="relp profile")
ax.set_xlabel("|r| (×10⁻³ Å⁻¹)"); ax.set_ylabel("p(r)")
ax.set_title(f"Free-form profile (σ_ω={sigma_final:.1f} mdeg)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 3. CC1/2 comparison
ax = axes[1, 0]
for name, shells, color, marker in [
    ("Baseline", shells_base, "C0", "o"),
    ("+ Poisson part.", shells_part, "C3", "s"),
]:
    d_mid = [0.5 * (s.d_max + s.d_min) for s in shells]
    ax.plot(d_to_x(d_mid), [s.cc_half for s in shells], "-",
            color=color, marker=marker, ms=5, lw=1.8, label=name)
ax.set_ylabel("CC1/2"); ax.legend(loc="lower left")
ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
setup_res_axis(ax)

# 4. σ_ω trajectory
ax = axes[1, 1]
ax.plot(steps, sigma_hist, "C0-o", ms=4, lw=2)
ax.set_xlabel("Step"); ax.set_ylabel("σ_ω (mdeg)")
ax.set_title("Orientation Uncertainty"); ax.grid(True, alpha=0.3)

fig.suptitle("Joint Poisson Partiality Refinement", fontsize=13)
fig.tight_layout()
fig.savefig(OUTDIR / "refine_poisson.png", dpi=150)
print(f"\nSaved refine_poisson.png")
plt.close(fig)

# Save
torch.save({
    "cc_base": cc_base, "r_base": r_base, "base_alpha": base_alpha,
    "cc_part": cc_part, "r_part": r_part, "part_alpha": part_alpha,
    "sigma_omega_mdeg": sigma_final,
    "nll_work": nll_work_hist, "nll_free": nll_free_hist,
    "profile_state": profile.cpu().state_dict(),
}, OUTDIR / "refine_poisson_summary.pt")

print("\nDone.")
