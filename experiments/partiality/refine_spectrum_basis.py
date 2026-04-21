#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -p gpu-day
#SBATCH -t 8:00:00
#SBATCH -o /das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/partiality/refine_spectrum_basis.out

"""Joint refinement with SASE-spectrum-basis partiality model.

Per-observation partiality model:

    p(|r_nom|; q, B_c, S_c) = ∫ S_c(δλ/λ) × G(|r_nom| - q·δλ/λ; γ_dis) d(δλ/λ)

where:
    S_c(δλ/λ) = mean(δλ/λ) + Σ_k w_c[k] × φ_k(δλ/λ)    (per-crystal latent w_c)
    γ_dis = √(γ_global² + (B_c/8π²)·s²)                (Wilson disorder)
    q = ||q_observation||                              (known per obs)
    G(·; γ) = relp profile (Gaussian)

The basis modes φ_k come from PCA of measured SASE spectra.
Per-crystal w_c are learnable, regularized toward 0 (mean spectrum).
γ_global is a single learnable parameter.
ScaleB (G, B per crystal) is also learnable.
α per resolution shell with sharpening penalty.

Joint optimization via Adam with Poisson+GenGauss marginal likelihood.
Per-observation free set with early stopping.
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
from torchsx.partiality._pairs import fit_BG

# ── Configuration ────────────────────────────────────────────
STREAM_DIR = Path("/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams")
OUTDIR = Path("/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/partiality/results")
BASIS_FILE = OUTDIR / "spectrum_basis.pt"

DELAYS = [-91, -136, -182, -227]
CELL = torch.tensor([14.97, 18.85, 18.89, 89.37, 84.94, 67.84], dtype=torch.float32)
SEED = 42
D_MIN, D_MAX = 0.9, 10.0
N_SHELLS = 20

EM_CFG = EMConfig(n_em_cycles=10, min_obs=5, alpha_prior_lambda=0.0)

# Refinement
N_STEPS = 600
LR_OMEGA = 1e-4
LR_W_C = 1e-2          # per-crystal spectrum coordinates
LR_GAMMA_GLOBAL = 1e-3
LR_SIGMA = 1e-3
LR_ALPHA = 1e-2
LR_SCALEB = 1e-3
LAMBDA_ALPHA = 0.05    # sharpening penalty
LAMBDA_W_C = 0.1       # L2 on per-crystal spectrum coords (regularize toward mean)
RESTRAINT_OMEGA = 100.0
RESTRAINT_LAMBDA = 2e4
RESTRAINT_LOG_G = 1.0
RESTRAINT_B = 0.01
FREE_FRACTION = 0.1
PATIENCE = 100
LOG_EVERY = 20
N_GG_QUAD = 21
N_LAMBDA_QUAD = 41     # quadrature for the spectrum integral
N_BASIS_KEEP = 3       # number of spectrum basis components to use
N_ALPHA_SHELLS = 6
CHUNK_SIZE = 200_000   # observations per chunk for memory

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


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
# Learnable per-crystal ScaleB
# ══════════════════════════════════════════════════════════════

class ScaleBModule(nn.Module):
    def __init__(self, n_crystals: int):
        super().__init__()
        self.log_G = nn.Parameter(torch.zeros(n_crystals))
        self.B = nn.Parameter(torch.zeros(n_crystals))

    def init_from(self, log_G: Tensor, B: Tensor) -> None:
        with torch.no_grad():
            self.log_G.copy_(log_G)
            self.B.copy_(B)

    def forward(self, crystal_id: Tensor, s_sq: Tensor) -> Tensor:
        log_scale = self.log_G[crystal_id] - 2.0 * self.B[crystal_id] * s_sq
        return torch.exp(log_scale.clamp(-10, 10))


# ══════════════════════════════════════════════════════════════
# Spectrum-basis partiality
# ══════════════════════════════════════════════════════════════

class SpectrumBasisPartiality(nn.Module):
    """Per-observation partiality from a SASE-spectrum basis.

    For observation j of crystal c with reciprocal vector q_j and resolution s_j:

        γ_dis(j) = √(γ_global² + (B_c/8π²) · s_j²)
        S_c(δλ/λ) = mean + Σ_k w_c[k] · φ_k
        p(|r_j|; q_j, γ_dis, c) = ∫ S_c(δλ/λ) · G(|r_j| - q_j·δλ/λ; γ_dis) d(δλ/λ)

    The integral is computed by quadrature on a fine δλ/λ grid.
    """

    def __init__(
        self,
        basis: Tensor,            # (K, N_λ) modes (functions of δλ/λ)
        mean: Tensor,             # (N_λ,) mean spectrum
        delta_lambda: Tensor,     # (N_λ,) δλ/λ axis
        n_crystals: int,
        gamma_global_init: float = 1e-3,
    ):
        super().__init__()
        self.register_buffer("basis", basis)
        self.register_buffer("mean", mean)
        self.register_buffer("delta_lambda", delta_lambda)
        self.K = basis.shape[0]
        # Integration weight: dλ
        d_dl = (delta_lambda[1] - delta_lambda[0]).item()
        self.register_buffer("d_dl", torch.tensor(d_dl))

        # Per-crystal latent coordinates (start at 0 = mean spectrum)
        self.w_c = nn.Parameter(torch.zeros(n_crystals, self.K))
        # Global relp width (size effect)
        self.log_gamma_global = nn.Parameter(
            torch.tensor(math.log(gamma_global_init))
        )

    def get_spectrum(self, crystal_id: Tensor) -> Tensor:
        """Per-crystal spectrum: (N_obs, N_λ)."""
        # Spectrum = mean + w · basis  →  (N_obs, N_λ)
        w = self.w_c[crystal_id]                           # (N, K)
        spec = self.mean[None, :] + w @ self.basis        # (N, N_λ)
        return spec

    def gamma_eff(
        self,
        B_per_obs: Tensor,
        s_sq: Tensor,
        q_mag: Tensor,
        sigma_omega: Tensor,
    ) -> Tensor:
        """Effective broadening per observation, combining all sources:

            σ_eff² = γ_global² + (B/(8π²))·s² + (q·σ_ω)²
                    ↑ size       ↑ disorder      ↑ orientation uncertainty

        Wilson disorder: <u²> = B/(8π²), σ²_relp = (B/(8π²))·s².
        Orientation uncertainty: σ_r = q × σ_ω.
        """
        gamma_global_sq = torch.exp(self.log_gamma_global) ** 2
        disorder_sq = (B_per_obs / (8 * math.pi ** 2) * s_sq).clamp(min=0)
        orient_sq = (q_mag * sigma_omega) ** 2
        return (gamma_global_sq + disorder_sq + orient_sq).clamp(min=1e-10).sqrt()

    def forward(
        self,
        abs_r: Tensor,        # (N,) nominal |r|
        q_mag: Tensor,        # (N,) reciprocal vector magnitude
        crystal_id: Tensor,   # (N,) crystal index
        B_per_obs: Tensor,    # (N,) B factor per observation (from ScaleB)
        s_sq: Tensor,         # (N,) resolution
        sigma_omega: Tensor,  # scalar — orientation uncertainty
        hkl_id: Tensor,       # (N,) HKL index for per-HKL normalization
        n_hkl: int,
        chunk_size: int = 200_000,
    ) -> Tensor:
        """Compute partiality per observation, normalized so mean(p)=1 per HKL.

        Since I0 from EM is fit assuming p=1 (i.e., I0 ≈ mean of observed
        intensities for that HKL), the partiality must satisfy mean(p)=1
        per HKL to leave the absolute scale unchanged.  The model then
        only learns the per-observation DEVIATION from the mean.

        Effective Gaussian width includes: relp size, disorder broadening,
        and orientation uncertainty (in quadrature).
        """
        N = abs_r.shape[0]
        gamma_eff_all = self.gamma_eff(B_per_obs, s_sq, q_mag, sigma_omega)  # (N,)

        # Compute raw log-partiality per observation, in chunks
        log_p_raw = torch.empty(N, device=abs_r.device, dtype=abs_r.dtype)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            sl = slice(start, end)

            spec = self.get_spectrum(crystal_id[sl])              # (chunk, N_λ)
            gamma_eff = gamma_eff_all[sl]
            q_chunk = q_mag[sl]
            dl = self.delta_lambda[None, :]                       # (1, N_λ)

            offset = abs_r[sl, None] - q_chunk[:, None] * dl
            log_kernel = -0.5 * (offset / gamma_eff[:, None]) ** 2
            kernel = torch.exp(log_kernel.clamp(min=-30, max=0))
            p_raw = (spec * kernel).sum(dim=1).clamp(min=1e-10)
            log_p_raw[sl] = torch.log(p_raw)

        # Per-HKL normalization: mean(log p) = 0 within each HKL
        # (geometric mean = 1, matches the log-space GenGauss noise model)
        sum_log_p = torch.zeros(n_hkl, device=log_p_raw.device).scatter_add_(
            0, hkl_id, log_p_raw)
        cnt = torch.zeros(n_hkl, device=log_p_raw.device).scatter_add_(
            0, hkl_id, torch.ones_like(log_p_raw))
        mean_log_p_per_hkl = sum_log_p / cnt.clamp(min=1)
        log_p_norm = log_p_raw - mean_log_p_per_hkl[hkl_id]

        return torch.exp(log_p_norm.clamp(min=-5, max=5))


# ══════════════════════════════════════════════════════════════
# Load data and run baseline EM
# ══════════════════════════════════════════════════════════════

print("── Loading streams ──")
streams = [SFXStream.load(STREAM_DIR / f"delay_{d}.sfx") for d in DELAYS]
s = SFXStream.concat(streams)
h1, h2 = s.split_random(n_parts=2, seed=SEED)
print(f"Total: {s.n_observations:,} obs, {s.n_crystals:,} crystals")

core = s._core
C = core.astar.shape[0]
N = core.hkl.shape[0]
UB, lambda_0, q_mag = compute_geometry(core)
hkl_idx = compute_hkl_index(core.hkl)
hkl_id = hkl_idx.hkl_id
n_unique = hkl_idx.n_unique
cid = core.crystal_id
s_sq = (q_mag / 2) ** 2

from torchref.base.reciprocal.hkl import get_d_spacing
median_cell = core.cell.median(dim=0).values
d_per_hkl = get_d_spacing(hkl_idx.unique_hkl.float(), median_cell)
d_per_obs = d_per_hkl[hkl_id]

# Resolution shells for alpha
shell_edges_inv_d2 = torch.linspace(
    (1.0 / D_MAX) ** 2, (1.0 / D_MIN) ** 2, N_ALPHA_SHELLS + 1)
inv_d2 = 1.0 / d_per_obs.clamp(min=1e-3) ** 2
shell_id = torch.bucketize(inv_d2, shell_edges_inv_d2[1:-1]).clamp(0, N_ALPHA_SHELLS - 1)
print(f"Geometry: {n_unique} unique HKLs")

# ── Baseline EM split-half ──
print("\n── BASELINE: EM Poisson ──")
dist_base = {}
for label, half in [("half1", h1), ("half2", h2)]:
    print(f"  {label}...")
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
print(f"  Baseline: CC1/2={cc_base:.4f}, R_split={r_base:.4f}, α={base_alpha:.4f}")

# ── Full-dataset EM for I0 reference ──
print("\n── Full-dataset EM ──")
dist_full = fit_em(s, config=EM_CFG, corrections=[ScaleBCorrection()], device=device)
print(f"  med(I₀)={dist_full.I0.median():.1f}, med(α)={dist_full.alpha.median():.4f}")

# Map I0 to per-obs
dist_hkl = dist_full.hkl.to(torch.int16)
I0_full = torch.full((n_unique,), dist_full.I0.median().item())
beta_full_arr = torch.full((n_unique,), 1.5)
_lookup = {}
for i in range(len(dist_hkl)):
    _lookup[(dist_hkl[i, 0].item(), dist_hkl[i, 1].item(), dist_hkl[i, 2].item())] = i
for j in range(n_unique):
    idx = _lookup.get((hkl_idx.unique_hkl[j, 0].item(),
                       hkl_idx.unique_hkl[j, 1].item(),
                       hkl_idx.unique_hkl[j, 2].item()))
    if idx is not None:
        I0_full[j] = dist_full.I0[idx]
        beta_full_arr[j] = dist_full.beta[i]
I0_per_obs = I0_full[hkl_id]
beta_per_obs = beta_full_arr[hkl_id]

# Quick B/G init
_, log_G_init, B_init = fit_BG(core.I, hkl_id, cid, s_sq, n_unique, C)

noise_model = PoissonNoiseModel()
bg_total = core.background * noise_model.n_peak
n_photons = (core.I + bg_total).clamp(min=0)

# ── Load spectrum basis ──
print(f"\n── Loading spectrum basis from {BASIS_FILE} ──")
basis_data = torch.load(BASIS_FILE, weights_only=True)
raw_basis_full = basis_data["raw_basis"][:N_BASIS_KEEP]      # (K, N_λ)
raw_mean_full = basis_data["raw_mean"]                        # (N_λ,)
delta_lambda_full = basis_data["delta_lambda_over_lambda"]   # (N_λ,)

# Downsample λ axis to 64 bins for memory
n_lambda_use = 64
idx = torch.linspace(0, len(delta_lambda_full) - 1, n_lambda_use).long()
raw_basis = raw_basis_full[:, idx]
raw_mean = raw_mean_full[idx]
delta_lambda_axis = delta_lambda_full[idx]
print(f"  Basis: {raw_basis.shape[0]} components × {raw_basis.shape[1]} λ bins (downsampled)")
print(f"  δλ/λ range: [{delta_lambda_axis.min():.5f}, {delta_lambda_axis.max():.5f}]")

# Normalise basis amplitude
basis_norm = raw_basis.norm(dim=1, keepdim=True).clamp(min=1e-10)
raw_basis = raw_basis / basis_norm

# ══════════════════════════════════════════════════════════════
# JOINT REFINEMENT
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("JOINT REFINEMENT WITH SPECTRUM BASIS")
print("=" * 60)

hkl_obs_d = core.hkl.to(device)
cid_d = cid.long().to(device)
hkl_id_d = hkl_id.to(device)
shell_id_d = shell_id.to(device)
I0_d = I0_per_obs.to(device)
s_sq_d = s_sq.to(device)
bg_d = bg_total.to(device)
n_ph_d = n_photons.to(device)
q_mag_d = q_mag.to(device)
beta_per_obs_d = beta_per_obs.to(device)

# Build softplus grid (I0-independent)
print(f"  Building softplus grid ({N_GG_QUAD} points)...")
with torch.no_grad():
    sig_lo = torch.full_like(n_ph_d, 0.5)
    grid = build_softplus_grid(n_ph_d, bg_d, n_quad=N_GG_QUAD, signal_lo_override=sig_lo)
log_poi_jac = grid.log_poi + grid.log_jac_dt + grid.log_dt[:, None]

# Modules
exc_model = ExcitationError(C, UB, lambda_0).to(device)
scaleB = ScaleBModule(C).to(device)
scaleB.init_from(log_G_init.to(device), B_init.to(device))
partiality = SpectrumBasisPartiality(
    basis=raw_basis.to(device),
    mean=raw_mean.to(device),
    delta_lambda=delta_lambda_axis.to(device),
    n_crystals=C,
    gamma_global_init=1e-3,
).to(device)

log_sigma_omega = nn.Parameter(
    torch.tensor(math.log(10e-3 * math.pi / 180), device=device)
)
log_alpha_shells = nn.Parameter(
    torch.full((N_ALPHA_SHELLS,), math.log(1.5), device=device)
)

# Work/free split
gen = torch.Generator(device=device).manual_seed(42)
free_mask = torch.rand(N, generator=gen, device=device) < FREE_FRACTION
work_mask = ~free_mask
n_work, n_free = work_mask.sum().item(), free_mask.sum().item()

all_params = (
    list(exc_model.parameters())
    + list(partiality.parameters())
    + list(scaleB.parameters())
    + [log_sigma_omega, log_alpha_shells]
)

optimizer = torch.optim.Adam([
    {"params": [exc_model.delta_omega_xy], "lr": LR_OMEGA},
    {"params": [exc_model.delta_lambda], "lr": 1e-5},
    {"params": [partiality.w_c], "lr": LR_W_C},
    {"params": [partiality.log_gamma_global], "lr": LR_GAMMA_GLOBAL},
    {"params": [log_sigma_omega], "lr": LR_SIGMA},
    {"params": [log_alpha_shells], "lr": LR_ALPHA},
    {"params": [scaleB.log_G, scaleB.B], "lr": LR_SCALEB},
])

print(f"Observations: {n_work:,} work / {n_free:,} free")
print(f"Parameters:")
print(f"  ScaleB:    {2*C}")
print(f"  Orient:    {C*2} δω + {C} δλ")
print(f"  w_c:       {C * partiality.K}")
print(f"  γ_global:  1")
print(f"  σ_ω:       1")
print(f"  α shells:  {N_ALPHA_SHELLS}")
print(f"Penalties: λ_α={LAMBDA_ALPHA}, λ_w={LAMBDA_W_C}")

best_free_nll = float("inf")
best_step = 0
best_state = None
nll_work_hist, nll_free_hist = [], []
alpha_hist, gamma_hist = [], []

for step in range(N_STEPS):
    optimizer.zero_grad()

    # Geometry
    abs_r = exc_model.abs_excitation_error(hkl_obs_d, cid_d)

    # ScaleB
    scale_per_obs = scaleB(cid_d, s_sq_d)
    B_per_obs = scaleB.B[cid_d]

    # Spectrum-basis partiality (mean=1 per HKL, includes σ_ω in width)
    sigma_omega = torch.exp(log_sigma_omega)
    p_eff = partiality(
        abs_r, q_mag_d, cid_d, B_per_obs, s_sq_d, sigma_omega,
        hkl_id_d, n_unique,
    )

    # Per-shell alpha
    alpha_per_obs = torch.exp(log_alpha_shells)[shell_id_d]

    # Poisson × GenGauss marginal
    log_signal_model = torch.log((I0_d * p_eff * scale_per_obs).clamp(min=1e-30))
    eps = torch.log(grid.signal.clamp(min=1e-30)) - log_signal_model[:, None]
    eps = eps.clamp(-5.0, 5.0)
    eps_abs = eps.abs().clamp(min=1e-6)

    log_gg = (-(eps_abs / alpha_per_obs[:, None]).pow(beta_per_obs_d[:, None])
              + _log_gg_norm(beta_per_obs_d)[:, None]
              - torch.log(alpha_per_obs[:, None]))
    log_integrand = log_gg + log_poi_jac
    log_marginal = torch.logsumexp(log_integrand, dim=1)

    nll_work = -(log_marginal * work_mask).sum() / n_work

    # Restraints
    omega_pen = RESTRAINT_OMEGA * (exc_model.delta_omega_xy ** 2).mean()
    lambda_pen = RESTRAINT_LAMBDA * (exc_model.delta_lambda ** 2).mean()
    logG_pen = RESTRAINT_LOG_G * (scaleB.log_G ** 2).mean()
    B_pen = RESTRAINT_B * (scaleB.B ** 2).mean()
    w_c_pen = LAMBDA_W_C * (partiality.w_c ** 2).mean()
    alpha_pen = LAMBDA_ALPHA * torch.exp(log_alpha_shells).mean()

    loss = nll_work + omega_pen + lambda_pen + logG_pen + B_pen + w_c_pen + alpha_pen

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

    # Center log_G
    with torch.no_grad():
        scaleB.log_G -= scaleB.log_G.mean()

    if step % LOG_EVERY == 0 or step == N_STEPS - 1:
        with torch.no_grad():
            nll_w = nll_work.item()
            nll_f = -(log_marginal * free_mask).sum().item() / n_free
            sigma_deg = math.degrees(torch.exp(log_sigma_omega).item()) * 1000
            dw_med = exc_model.delta_omega_xy.norm(dim=1).median().item() * 1e3
            alpha_vals = torch.exp(log_alpha_shells).cpu().numpy()
            gamma_global = torch.exp(partiality.log_gamma_global).item()
            w_c_norm = partiality.w_c.norm(dim=1).median().item()

        nll_work_hist.append(nll_w)
        nll_free_hist.append(nll_f)
        alpha_hist.append(alpha_vals.copy())
        gamma_hist.append(gamma_global)

        if nll_f < best_free_nll:
            best_free_nll = nll_f
            best_step = step
            best_state = {
                "exc": {k: v.detach().clone() for k, v in exc_model.state_dict().items()},
                "scaleB": {k: v.detach().clone() for k, v in scaleB.state_dict().items()},
                "partiality": {k: v.detach().clone() for k, v in partiality.state_dict().items()},
                "log_sigma": log_sigma_omega.detach().clone(),
                "log_alpha_shells": log_alpha_shells.detach().clone(),
            }

        print(f"  step {step:>4}: NLL work={nll_w:.3f} free={nll_f:.3f}  "
              f"α={alpha_vals.min():.2f}→{alpha_vals.max():.2f}  "
              f"γ_g={gamma_global*1e3:.2f}e-3  σ_ω={sigma_deg:.1f} mdeg  "
              f"|w_c|={w_c_norm:.3f}  |δω|={dw_med:.2f} mdeg")

        if PATIENCE > 0 and step - best_step > PATIENCE:
            print(f"  Early stop: free NLL peaked at step {best_step}")
            break

# Restore best
if best_state is not None:
    exc_model.load_state_dict(best_state["exc"])
    scaleB.load_state_dict(best_state["scaleB"])
    partiality.load_state_dict(best_state["partiality"])
    log_sigma_omega.data.copy_(best_state["log_sigma"])
    log_alpha_shells.data.copy_(best_state["log_alpha_shells"])
    print(f"  Restored best (step {best_step}, free NLL={best_free_nll:.3f})")

sigma_final = math.degrees(torch.exp(log_sigma_omega).item()) * 1000
gamma_final = torch.exp(partiality.log_gamma_global).item()
alpha_final = torch.exp(log_alpha_shells).detach().cpu().numpy()
print(f"\n  Final γ_global = {gamma_final*1e3:.3f}e-3 Å⁻¹")
print(f"  Final σ_ω = {sigma_final:.1f} mdeg")
print(f"  Final α per shell: {alpha_final}")
print(f"  Final |w_c| median = {partiality.w_c.detach().norm(dim=1).median().item():.3f}")

# ══════════════════════════════════════════════════════════════
# Apply to split-half merge
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SPLIT-HALF with spectrum-basis partiality")
print("=" * 60)

with torch.no_grad():
    abs_r_final = exc_model.abs_excitation_error(hkl_obs_d, cid_d)
    scale_final = scaleB(cid_d, s_sq_d)
    B_final = scaleB.B[cid_d]
    sigma_omega_final = torch.exp(log_sigma_omega)
    p_full = partiality(
        abs_r_final, q_mag_d, cid_d, B_final, s_sq_d, sigma_omega_final,
        hkl_id_d, n_unique,
    ).cpu()

    # Diagnostic: also compute partiality with w_c=0 (mean spectrum only)
    saved_wc = partiality.w_c.detach().clone()
    partiality.w_c.data.zero_()
    p_mean_only = partiality(
        abs_r_final, q_mag_d, cid_d, B_final, s_sq_d, sigma_omega_final,
        hkl_id_d, n_unique,
    ).cpu()
    partiality.w_c.data.copy_(saved_wc)

print(f"  Full partiality (w_c ≠ 0): median={p_full.median():.4f}, "
      f"p10={p_full.quantile(0.1):.4f}, p90={p_full.quantile(0.9):.4f}")
print(f"  Mean-only (w_c = 0):       median={p_mean_only.median():.4f}, "
      f"p10={p_mean_only.quantile(0.1):.4f}, p90={p_mean_only.quantile(0.9):.4f}")

# Reproduce the deterministic split to know which observations belong to each half.
# split_random does: assignment = torch.randint(2, (n_crystals,), generator=Generator().manual_seed(SEED))
# Then crystal mask per half = (assignment == h), obs mask = crystal_mask[crystal_id]
gen_split = torch.Generator()
gen_split.manual_seed(SEED)
assignment = torch.randint(2, (s.n_crystals,), generator=gen_split)
print(f"  Split: {(assignment==0).sum()} crystals in half1, {(assignment==1).sum()} in half2")

# Per-observation crystal assignment (in full-stream order)
crystal_assignment_per_obs = assignment[s._core.crystal_id.long()]

def renormalize_half(p_full_tensor, obs_mask, half_core):
    """Extract half's partialities and re-normalize so mean(log p)=0 per HKL."""
    half_part = p_full_tensor[obs_mask]
    half_hkl_idx_loc = compute_hkl_index(half_core.hkl)
    log_p_half = torch.log(half_part.clamp(min=1e-10))
    n_u = half_hkl_idx_loc.n_unique
    hid = half_hkl_idx_loc.hkl_id
    sum_lp = torch.zeros(n_u).scatter_add_(0, hid, log_p_half)
    cnt = torch.zeros(n_u).scatter_add_(0, hid, torch.ones_like(log_p_half))
    mean_lp = sum_lp / cnt.clamp(min=1)
    return torch.exp(log_p_half - mean_lp[hid])


# Three variants:
#   (a) no partiality (sanity check — should match baseline)
#   (b) mean spectrum only (w_c = 0)
#   (c) full partiality with learned w_c
variants = [
    ("no partiality", None),
    ("mean spec only", p_mean_only),
    ("full w_c", p_full),
]

results = {}
for variant_name, p_source in variants:
    print(f"\n── Variant: {variant_name} ──")
    dist_v = {}
    for label, half_idx, half in [("half1", 0, h1), ("half2", 1, h2)]:
        obs_mask = (crystal_assignment_per_obs == half_idx)

        if p_source is None:
            half_partiality = None
        else:
            half_partiality = renormalize_half(p_source, obs_mask, half._core)
            assert half_partiality.shape[0] == half._core.hkl.shape[0]

        d = fit_em(half, config=EM_CFG, corrections=[ScaleBCorrection()],
                   device=device, partiality=half_partiality)
        dist_v[label] = d
        if half_partiality is not None:
            print(f"  {label}: p10={half_partiality.quantile(0.1):.3f} "
                  f"p50={half_partiality.median():.3f} "
                  f"p90={half_partiality.quantile(0.9):.3f}  "
                  f"med(α)={d.alpha.median():.4f}")
        else:
            print(f"  {label}: med(α)={d.alpha.median():.4f}")

    m_v = match_halves(
        dist_v["half1"].hkl, to_F(dist_v["half1"].I0),
        dist_v["half2"].hkl, to_F(dist_v["half2"].I0),
        CELL, d_min=D_MIN, d_max=D_MAX,
    )
    results[variant_name] = (dist_v, m_v,
        0.5 * (dist_v["half1"].alpha.median().item() +
               dist_v["half2"].alpha.median().item()))

# Use the "full w_c" variant as the main comparison
dist_part, m_part, part_alpha = results["full w_c"]
cc_part, r_part = cc_half(m_part), r_split(m_part)

m_part = match_halves(
    dist_part["half1"].hkl, to_F(dist_part["half1"].I0),
    dist_part["half2"].hkl, to_F(dist_part["half2"].I0),
    CELL, d_min=D_MIN, d_max=D_MAX,
)
cc_part, r_part = cc_half(m_part), r_split(m_part)
part_alpha = 0.5 * (dist_part["half1"].alpha.median().item() +
                     dist_part["half2"].alpha.median().item())

# ── Results ──
print("\n" + "=" * 65)
print(f"COMPARISON ({D_MIN}–{D_MAX} Å)")
print("=" * 65)
print(f"  {'Method':>30s}  {'CC1/2':>7s}  {'CC*':>7s}  {'R_split':>8s}  {'N':>6s}  {'med(α)':>8s}")
print(f"  {'-'*67}")
print(f"  {'EM Poisson (baseline)':>30s}  {cc_base:>7.4f}  "
      f"{cc_star(cc_base):>7.4f}  {r_base:>8.4f}  {m_base.F1.numel():>6d}  {base_alpha:>8.4f}")
for vname, (d_v, m_v, a_v) in results.items():
    cc_v = cc_half(m_v)
    print(f"  {'+ ' + vname:>30s}  {cc_v:>7.4f}  {cc_star(cc_v):>7.4f}  "
          f"{r_split(m_v):>8.4f}  {m_v.F1.numel():>6d}  {a_v:>8.4f}")

print(f"\n  Delta: CC1/2={cc_part-cc_base:+.4f}, R_split={r_part-r_base:+.4f}, "
      f"alpha={part_alpha-base_alpha:+.4f}")
print(f"  Internal α range: {alpha_final.min():.3f}–{alpha_final.max():.3f}")
print(f"  γ_global = {gamma_final*1e3:.3f}e-3 Å⁻¹")

# Per-shell
shells_base = shell_stats(m_base, n_shells=N_SHELLS)
shells_part = shell_stats(m_part, n_shells=N_SHELLS)
print(f"\n{'='*80}\nPER-SHELL CC1/2\n{'='*80}")
print(f"  {'d_max':>7s} {'d_min':>7s}  {'Baseline':>10s}  {'Spectrum':>10s}  {'Delta':>8s}")
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
alpha_arr = np.array(alpha_hist)
shell_centers = 0.5 * (shell_edges_inv_d2[:-1] + shell_edges_inv_d2[1:])
shell_d_lbl = 1.0 / shell_centers.sqrt().numpy()
for sh in range(N_ALPHA_SHELLS):
    ax.plot(steps_arr, alpha_arr[:, sh], lw=1.5, label=f"d={shell_d_lbl[sh]:.1f} Å")
ax.set_xlabel("Step"); ax.set_ylabel("α per shell")
ax.set_title("Alpha per shell"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.hist(p_full.numpy(), bins=80, alpha=0.7)
ax.axvline(p_full.median().item(), color="red", ls="--", lw=1.5)
ax.set_xlabel("Partiality"); ax.set_ylabel("Count")
ax.set_title(f"Partiality distribution (median={p_full.median():.3f})")

ax = axes[1, 1]
for name, shells, color, marker in [
    ("Baseline", shells_base, "C0", "o"),
    ("+ Spectrum", shells_part, "C3", "s"),
]:
    d_mid = [0.5 * (s.d_max + s.d_min) for s in shells]
    ax.plot(d_to_x(d_mid), [s.cc_half for s in shells], "-",
            color=color, marker=marker, ms=5, lw=1.8, label=name)
ax.set_ylabel("CC1/2"); ax.legend(loc="lower left")
ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
setup_res_axis(ax)

fig.suptitle("Joint Refinement with SASE Spectrum Basis", fontsize=13)
fig.tight_layout()
fig.savefig(OUTDIR / "refine_spectrum_basis.png", dpi=150)
print(f"\nSaved refine_spectrum_basis.png")

torch.save({
    "cc_base": cc_base, "r_base": r_base, "base_alpha": base_alpha,
    "cc_part": cc_part, "r_part": r_part, "part_alpha": part_alpha,
    "alpha_final": alpha_final,
    "gamma_final": gamma_final,
    "sigma_omega_mdeg": sigma_final,
    "nll_work": nll_work_hist, "nll_free": nll_free_hist,
}, OUTDIR / "refine_spectrum_basis_summary.pt")

print("\nDone.")
