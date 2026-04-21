#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -p gpu-day
#SBATCH -t 6:00:00
#SBATCH -o /das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/partiality/joint_refine.out

"""Unified joint refinement: ScaleB + partiality + orientations + α.

Single optimization with all corrections learnable jointly:
    - log_G, B per crystal (ScaleB)
    - Free-form profile (relp shape) + σ_ω (mosaicity)
    - δω, δλ per crystal (orientations)
    - α per resolution shell (sharpening penalty drives down)

Loss = -log P_marginal(n_obs | I0, scale, p, α, β) + λ_α × α + restraints

I0 is fixed from a quick initial EM merge (just for the merged intensity reference).
Per-observation free set monitors overfitting.
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
from torchsx.merge._dstep import build_softplus_grid, _log_gg_norm
from torchsx.partiality._excitation import ExcitationError

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
N_STEPS = 600
LR_OMEGA = 1e-4
LR_PROFILE = 1e-2
LR_SIGMA = 1e-3
LR_ALPHA = 1e-2
LR_SCALEB = 1e-3
LAMBDA_ALPHA = 0.05
RESTRAINT_OMEGA = 100.0
RESTRAINT_LAMBDA = 2e4
RESTRAINT_LOG_G = 1.0      # L2 on log G
RESTRAINT_B = 0.01         # L2 on B
FREE_FRACTION = 0.1
PATIENCE = 100
LOG_EVERY = 20
N_GG_QUAD = 31
N_R_CONTROL = 20
R_MAX = 0.008
N_ALPHA_SHELLS = 6         # per-resolution-shell alpha

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# ══════════════════════════════════════════════════════════════
# Free-form monotone profile (same as before)
# ══════════════════════════════════════════════════════════════

class FreeFormProfile(nn.Module):
    def __init__(self, n_points: int = 20, r_max: float = 0.008):
        super().__init__()
        self.register_buffer("r_control", torch.linspace(0, r_max, n_points + 1))
        self.raw_increments = nn.Parameter(torch.full((n_points,), -3.0))

    def log_p_at_control(self) -> Tensor:
        increments = F.softplus(self.raw_increments)
        cum = increments.cumsum(0)
        return torch.cat([torch.zeros(1, device=cum.device), -cum])

    def log_partiality(self, r: Tensor) -> Tensor:
        log_p = self.log_p_at_control()
        rp = self.r_control
        r_clamped = r.clamp(0, rp[-1])
        idx = torch.searchsorted(rp, r_clamped) - 1
        idx = idx.clamp(0, len(rp) - 2)
        dr = (rp[idx + 1] - rp[idx]).clamp(min=1e-10)
        t = ((r_clamped - rp[idx]) / dr).clamp(0, 1)
        return log_p[idx] * (1 - t) + log_p[idx + 1] * t


# ══════════════════════════════════════════════════════════════
# Learnable per-crystal ScaleB
# ══════════════════════════════════════════════════════════════

class ScaleBModule(nn.Module):
    """Per-crystal log_G and B as learnable parameters.

    Initialised from a closed-form B/G fit to give a reasonable starting point.
    """

    def __init__(self, n_crystals: int):
        super().__init__()
        self.log_G = nn.Parameter(torch.zeros(n_crystals))
        self.B = nn.Parameter(torch.zeros(n_crystals))

    def init_from(self, log_G: Tensor, B: Tensor) -> None:
        with torch.no_grad():
            self.log_G.copy_(log_G)
            self.B.copy_(B)

    def forward(self, crystal_id: Tensor, s_sq: Tensor) -> Tensor:
        """Return per-obs scale (linear, not log)."""
        log_scale = self.log_G[crystal_id] - 2.0 * self.B[crystal_id] * s_sq
        return torch.exp(log_scale.clamp(-10, 10))


# ══════════════════════════════════════════════════════════════
# Gauss-Hermite quadrature
# ══════════════════════════════════════════════════════════════

def gauss_hermite_points(n: int):
    from numpy.polynomial.hermite import hermgauss
    t_np, w_np = hermgauss(n)
    w_np = w_np / math.sqrt(math.pi)
    return torch.from_numpy(t_np).float(), torch.from_numpy(w_np).float()


def compute_p_eff(profile, abs_r, sigma_r, quad_t, quad_w):
    sqrt2 = math.sqrt(2.0)
    r_grid = abs_r[:, None] + sigma_r[:, None] * sqrt2 * quad_t[None, :]
    r_grid = r_grid.clamp(min=0)
    log_p_grid = profile.log_partiality(r_grid)
    log_w = torch.log(quad_w.clamp(min=1e-30))[None, :]
    log_p_eff = torch.logsumexp(log_p_grid + log_w, dim=1)
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
# Load data and initial EM
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

# Per-HKL d-spacing for shell assignment
from torchref.base.reciprocal.hkl import get_d_spacing
median_cell = core.cell.median(dim=0).values
d_per_hkl = get_d_spacing(hkl_idx.unique_hkl.float(), median_cell)
d_per_obs = d_per_hkl[hkl_id]

# Resolution shells for alpha (equal-count by inv_d²)
shell_edges_inv_d2 = torch.linspace(
    (1.0 / D_MAX) ** 2, (1.0 / D_MIN) ** 2, N_ALPHA_SHELLS + 1)
inv_d2 = 1.0 / d_per_obs.clamp(min=1e-3) ** 2
shell_id = torch.bucketize(inv_d2, shell_edges_inv_d2[1:-1]).clamp(0, N_ALPHA_SHELLS - 1)
print(f"Geometry: {n_unique} unique HKLs, q_mag range [{q_mag.min():.3f}, {q_mag.max():.3f}] Å⁻¹")
print(f"Alpha shells: {N_ALPHA_SHELLS}")

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

# ── Full-dataset EM for I0 ──
print("\n── Full-dataset EM for I0 reference ──")
dist_full = fit_em(s, config=EM_CFG, corrections=[ScaleBCorrection()], device=device)
print(f"  med(I₀)={dist_full.I0.median():.1f}, med(α)={dist_full.alpha.median():.4f}")

# Map I0 to per-obs (CPU dict lookup)
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

# Quick B/G init for ScaleB (closed form)
from torchsx.partiality._pairs import fit_BG
_, log_G_init, B_init = fit_BG(core.I, hkl_id, cid, s_sq, n_unique, C)

# Background and photons
noise_model = PoissonNoiseModel()
bg_total = core.background * noise_model.n_peak
n_photons = (core.I + bg_total).clamp(min=0)

print(f"  ScaleB init: med|log G|={log_G_init.abs().median():.3f}, med B={B_init.median():.1f}")
print(f"  bg_total: median={bg_total.median():.1f}, n_ph: median={n_photons.median():.1f}")

# ══════════════════════════════════════════════════════════════
# UNIFIED JOINT REFINEMENT
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("UNIFIED JOINT REFINEMENT")
print("=" * 60)

# Move to device
hkl_obs_d = core.hkl.to(device)
cid_d = cid.long().to(device)
hkl_id_d = hkl_id.to(device)
shell_id_d = shell_id.to(device)
I0_d = I0_per_obs.to(device)
s_sq_d = s_sq.to(device)
bg_d = bg_total.to(device)
n_ph_d = n_photons.to(device)
q_mag_d = q_mag.to(device)

# Per-obs beta from EM (we don't learn beta; it's a shape parameter)
beta_full_arr = torch.full((n_unique,), 1.5)
for i in range(len(dist_hkl)):
    idx = _lookup.get((dist_hkl[i, 0].item(), dist_hkl[i, 1].item(), dist_hkl[i, 2].item()))
    if idx is not None:
        beta_full_arr[idx] = dist_full.beta[i]
beta_per_obs = beta_full_arr[hkl_id].to(device)

# Build softplus grid (I0-independent, pre-computed once)
print(f"  Building softplus grid ({N_GG_QUAD} points)...")
with torch.no_grad():
    sig_lo = torch.full_like(n_ph_d, 0.5)
    grid = build_softplus_grid(n_ph_d, bg_d, n_quad=N_GG_QUAD, signal_lo_override=sig_lo)
log_poi_jac = grid.log_poi + grid.log_jac_dt + grid.log_dt[:, None]

# Modules
exc_model = ExcitationError(C, UB, lambda_0).to(device)
profile = FreeFormProfile(n_points=N_R_CONTROL, r_max=R_MAX).to(device)
scaleB = ScaleBModule(C).to(device)
scaleB.init_from(log_G_init.to(device), B_init.to(device))

# Learnable scalars
log_sigma_omega = nn.Parameter(
    torch.tensor(math.log(10e-3 * math.pi / 180), device=device)
)
# Per-shell alpha (initialised to baseline alpha ≈ 1.5)
log_alpha_shells = nn.Parameter(
    torch.full((N_ALPHA_SHELLS,), math.log(1.5), device=device)
)

# Gauss-Hermite quadrature for orientation uncertainty
N_OMEGA_QUAD = 11
quad_t, quad_w = gauss_hermite_points(N_OMEGA_QUAD)
quad_t, quad_w = quad_t.to(device), quad_w.to(device)

# Work/free split
gen = torch.Generator(device=device).manual_seed(42)
free_mask = torch.rand(N, generator=gen, device=device) < FREE_FRACTION
work_mask = ~free_mask
n_work, n_free = work_mask.sum().item(), free_mask.sum().item()

# Optimiser
all_params = (
    list(exc_model.parameters())
    + list(profile.parameters())
    + list(scaleB.parameters())
    + [log_sigma_omega, log_alpha_shells]
)

optimizer = torch.optim.Adam([
    {"params": [exc_model.delta_omega_xy], "lr": LR_OMEGA},
    {"params": [exc_model.delta_lambda], "lr": 1e-5},
    {"params": profile.parameters(), "lr": LR_PROFILE},
    {"params": [log_sigma_omega], "lr": LR_SIGMA},
    {"params": [log_alpha_shells], "lr": LR_ALPHA},
    {"params": [scaleB.log_G, scaleB.B], "lr": LR_SCALEB},
])

print(f"Observations: {n_work:,} work / {n_free:,} free")
print(f"Parameters:")
print(f"  ScaleB:   {2*C} (G, B per crystal)")
print(f"  Orient:   {C*2} δω + {C} δλ")
print(f"  Profile:  {N_R_CONTROL}")
print(f"  σ_ω:      1")
print(f"  α shells: {N_ALPHA_SHELLS}")
print(f"Restraints: λ_ω={RESTRAINT_OMEGA}, λ_λ={RESTRAINT_LAMBDA}, "
      f"λ_logG={RESTRAINT_LOG_G}, λ_B={RESTRAINT_B}")
print(f"α sharpening: λ_α={LAMBDA_ALPHA}")
print(f"Likelihood: Poisson × GenGauss marginal ({N_GG_QUAD} ε quadrature × "
      f"{N_OMEGA_QUAD} ω quadrature)")

best_free_nll = float("inf")
best_step = 0
best_state = None
nll_work_hist, nll_free_hist = [], []
alpha_hist = []

for step in range(N_STEPS):
    optimizer.zero_grad()

    # Compute |r| (differentiable through δω)
    abs_r = exc_model.abs_excitation_error(hkl_obs_d, cid_d)

    # Effective partiality (differentiable through profile and σ_ω)
    sigma_omega = torch.exp(log_sigma_omega)
    sigma_r = q_mag_d * sigma_omega
    p_eff = compute_p_eff(profile, abs_r, sigma_r, quad_t, quad_w)

    # Per-obs scale (differentiable through ScaleBModule)
    scale_per_obs = scaleB(cid_d, s_sq_d)

    # Per-obs alpha (per-shell, differentiable)
    alpha_per_obs = torch.exp(log_alpha_shells)[shell_id_d]

    # Poisson × GenGauss marginal
    log_signal_model = torch.log((I0_d * p_eff * scale_per_obs).clamp(min=1e-30))
    eps = torch.log(grid.signal.clamp(min=1e-30)) - log_signal_model[:, None]
    eps = eps.clamp(-5.0, 5.0)
    eps_abs = eps.abs().clamp(min=1e-6)

    log_gg = (-(eps_abs / alpha_per_obs[:, None]).pow(beta_per_obs[:, None])
              + _log_gg_norm(beta_per_obs)[:, None]
              - torch.log(alpha_per_obs[:, None]))
    log_integrand = log_gg + log_poi_jac
    log_marginal = torch.logsumexp(log_integrand, dim=1)

    nll_work = -(log_marginal * work_mask).sum() / n_work

    # Restraints
    omega_pen = RESTRAINT_OMEGA * (exc_model.delta_omega_xy ** 2).mean()
    lambda_pen = RESTRAINT_LAMBDA * (exc_model.delta_lambda ** 2).mean()
    logG_pen = RESTRAINT_LOG_G * (scaleB.log_G ** 2).mean()
    B_pen = RESTRAINT_B * (scaleB.B ** 2).mean()

    # α sharpening: penalise the mean alpha across shells
    alpha_mean = torch.exp(log_alpha_shells).mean()
    alpha_pen = LAMBDA_ALPHA * alpha_mean

    loss = nll_work + omega_pen + lambda_pen + logG_pen + B_pen + alpha_pen

    if torch.isnan(loss) or torch.isinf(loss):
        print(f"  step {step}: NaN/Inf loss — stopping")
        break

    loss.backward()

    has_nan = any(p.grad is not None and p.grad.isnan().any() for p in all_params)
    if has_nan:
        print(f"  step {step}: NaN gradient — skipping")
        optimizer.zero_grad()
        continue

    torch.nn.utils.clip_grad_norm_(all_params, max_norm=10.0)
    optimizer.step()

    # Center log_G to prevent global scale drift
    with torch.no_grad():
        scaleB.log_G -= scaleB.log_G.mean()

    if step % LOG_EVERY == 0 or step == N_STEPS - 1:
        with torch.no_grad():
            nll_w = nll_work.item()
            nll_f = -(log_marginal * free_mask).sum().item() / n_free
            sigma_deg = math.degrees(sigma_omega.item()) * 1000
            dw_med = exc_model.delta_omega_xy.norm(dim=1).median().item() * 1e3
            alpha_vals = torch.exp(log_alpha_shells).cpu().numpy()
            p_at_gamma = torch.exp(profile.log_partiality(
                torch.tensor([0.003], device=device))).item()

        nll_work_hist.append(nll_w)
        nll_free_hist.append(nll_f)
        alpha_hist.append(alpha_vals.copy())

        if nll_f < best_free_nll:
            best_free_nll = nll_f
            best_step = step
            best_state = {
                "exc": {k: v.detach().clone() for k, v in exc_model.state_dict().items()},
                "profile": {k: v.detach().clone() for k, v in profile.state_dict().items()},
                "scaleB": {k: v.detach().clone() for k, v in scaleB.state_dict().items()},
                "log_sigma": log_sigma_omega.detach().clone(),
                "log_alpha_shells": log_alpha_shells.detach().clone(),
            }

        print(f"  step {step:>4}: NLL work={nll_w:.3f} free={nll_f:.3f}  "
              f"α={alpha_vals.min():.2f}→{alpha_vals.max():.2f}  "
              f"σ_ω={sigma_deg:.1f} mdeg  |δω|={dw_med:.2f} mdeg  "
              f"p(γ)={p_at_gamma:.3f}")

        if PATIENCE > 0 and step - best_step > PATIENCE:
            print(f"  Early stop: free NLL peaked at step {best_step}")
            break

# Restore best
if best_state is not None:
    exc_model.load_state_dict(best_state["exc"])
    profile.load_state_dict(best_state["profile"])
    scaleB.load_state_dict(best_state["scaleB"])
    log_sigma_omega.data.copy_(best_state["log_sigma"])
    log_alpha_shells.data.copy_(best_state["log_alpha_shells"])
    print(f"  Restored best (step {best_step}, free NLL={best_free_nll:.3f})")

sigma_final = math.degrees(torch.exp(log_sigma_omega).item()) * 1000
alpha_final = torch.exp(log_alpha_shells).detach().cpu().numpy()
print(f"\n  Final σ_ω = {sigma_final:.1f} mdeg")
print(f"  Final α per shell: {alpha_final}")
print(f"  Final |δω| median = {exc_model.delta_omega_xy.detach().norm(dim=1).median().item()*1e3:.2f} mdeg")
print(f"  Final ScaleB: med|log G|={scaleB.log_G.detach().abs().median().item():.3f}, "
      f"med B={scaleB.B.detach().median().item():.2f}")

# ══════════════════════════════════════════════════════════════
# Apply learned partiality to split-half merge
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SPLIT-HALF with joint-refined partiality")
print("=" * 60)

# Compute full-dataset partiality (for diagnostics)
with torch.no_grad():
    abs_r_final = exc_model.abs_excitation_error(hkl_obs_d, cid_d)
    sigma_r_final = q_mag_d * torch.exp(log_sigma_omega)
    p_final = compute_p_eff(profile, abs_r_final, sigma_r_final, quad_t, quad_w).cpu()

print(f"  Full-dataset partiality: median={p_final.median():.3f}, "
      f"p10={p_final.quantile(0.1):.3f}, p90={p_final.quantile(0.9):.3f}")

# Per-half partiality (compute geometry independently)
dist_part = {}
for label, half in [("half1", h1), ("half2", h2)]:
    print(f"\n  {label}...")
    half_core = half._core
    half_UB = torch.stack([half_core.astar, half_core.bstar, half_core.cstar], dim=2)
    half_lam = 12398.42 / half_core.beam_energy[half_core.image_id.long()]
    half_exc = ExcitationError(half_core.astar.shape[0], half_UB, half_lam).to(device)

    with torch.no_grad():
        half_abs_r = half_exc.abs_excitation_error(
            half_core.hkl.to(device), half_core.crystal_id.long().to(device))
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
    ("EM + joint partiality", m_part, part_alpha),
]:
    cc = cc_half(m)
    print(f"  {name:>30s}  {cc:>7.4f}  {cc_star(cc):>7.4f}  {r_split(m):>8.4f}  "
          f"{m.F1.numel():>6d}  {alpha:>8.4f}")

print(f"\n  Delta: CC1/2={cc_part-cc_base:+.4f}, R_split={r_part-r_base:+.4f}, "
      f"alpha={part_alpha-base_alpha:+.4f}")
print(f"  Internal α range: {alpha_final.min():.3f}–{alpha_final.max():.3f}")
print(f"  σ_ω = {sigma_final:.1f} mdeg")

# Per-shell
shells_base = shell_stats(m_base, n_shells=N_SHELLS)
shells_part = shell_stats(m_part, n_shells=N_SHELLS)
print(f"\n{'='*80}\nPER-SHELL CC1/2\n{'='*80}")
print(f"  {'d_max':>7s} {'d_min':>7s}  {'Baseline':>10s}  {'Joint':>10s}  {'Delta':>8s}")
print(f"  {'-'*50}")
for sb, sp in zip(shells_base, shells_part):
    print(f"  {sb.d_max:>7.2f} {sb.d_min:>7.2f}  {sb.cc_half:>10.4f}  {sp.cc_half:>10.4f}  "
          f"{sp.cc_half-sb.cc_half:>+8.4f}")

# Plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
steps_arr = np.arange(len(nll_work_hist)) * LOG_EVERY
ax.plot(steps_arr, nll_work_hist, "C0-", lw=2, label="Work")
ax.plot(steps_arr, nll_free_hist, "C3--", lw=2, label="Free")
ax.axvline(best_step, color="gray", ls=":", lw=1)
ax.set_xlabel("Step"); ax.set_ylabel("NLL / obs")
ax.set_title("Marginal NLL"); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
alpha_arr = np.array(alpha_hist)  # (n_steps, n_shells)
shell_centers = 0.5 * (shell_edges_inv_d2[:-1] + shell_edges_inv_d2[1:])
shell_d = 1.0 / shell_centers.sqrt().numpy()
for sh in range(N_ALPHA_SHELLS):
    ax.plot(steps_arr, alpha_arr[:, sh], lw=1.5, label=f"d={shell_d[sh]:.1f} Å")
ax.set_xlabel("Step"); ax.set_ylabel("α per shell")
ax.set_title("Alpha trajectory"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
with torch.no_grad():
    r_plot = torch.linspace(0, R_MAX, 200, device=device)
    log_p_plot = profile.log_partiality(r_plot).cpu().numpy()
ax.plot(r_plot.cpu() * 1e3, np.exp(log_p_plot), "k-", lw=2)
ax.set_xlabel("|r| (×10⁻³ Å⁻¹)"); ax.set_ylabel("p(r)")
ax.set_title(f"Free-form profile (σ_ω={sigma_final:.1f} mdeg)")
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
for name, shells, color, marker in [
    ("Baseline", shells_base, "C0", "o"),
    ("+ Joint", shells_part, "C3", "s"),
]:
    d_mid = [0.5 * (s.d_max + s.d_min) for s in shells]
    ax.plot(d_to_x(d_mid), [s.cc_half for s in shells], "-",
            color=color, marker=marker, ms=5, lw=1.8, label=name)
ax.set_ylabel("CC1/2"); ax.legend(loc="lower left")
ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
setup_res_axis(ax)

fig.suptitle("Joint Refinement: ScaleB + Partiality + α", fontsize=13)
fig.tight_layout()
fig.savefig(OUTDIR / "joint_refine.png", dpi=150)
print(f"\nSaved joint_refine.png")
plt.close(fig)

torch.save({
    "cc_base": cc_base, "r_base": r_base, "base_alpha": base_alpha,
    "cc_part": cc_part, "r_part": r_part, "part_alpha": part_alpha,
    "alpha_final": alpha_final,
    "sigma_omega_mdeg": sigma_final,
    "nll_work": nll_work_hist, "nll_free": nll_free_hist,
}, OUTDIR / "joint_refine_summary.pt")

print("\nDone.")
