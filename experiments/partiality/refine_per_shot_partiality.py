#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -p gpu-day
#SBATCH -t 8:00:00
#SBATCH -o /das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/partiality/refine_per_shot_partiality.out

"""Joint refinement using *measured* per-crystal SASE spectra.

Same partiality model as refine_spectrum_basis.py, except the spectrum
S_c(δλ/λ) is FIXED to the measured spectrum recorded for crystal c (no
learnable basis coefficients w_c).  Free parameters collapse to:

    γ_global, σ_ω, ScaleB(G,B) per crystal, ExcitationError, α per shell.

Only crystals with a matched per-shot spectrum are used (built by
build_per_shot_spectra.py).
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
PER_SHOT_DIR = Path("/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/partiality/results/per_shot_spectra")
OUTDIR = Path("/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/partiality/results")
INDEX_FILE = PER_SHOT_DIR / "per_shot_index.pt"

CELL = torch.tensor([14.97, 18.85, 18.89, 89.37, 84.94, 67.84], dtype=torch.float32)
SEED = 42
D_MIN, D_MAX = 0.9, 10.0
N_SHELLS = 20

EM_CFG = EMConfig(
    n_em_cycles=4,         # later cycles diverge at 900x multiplicity
    min_obs=20,
    alpha_prior_lambda=0.1,
    n_shells=20,
    n_newton_final=4,      # cap LBFGS iter to avoid overshoot at high mult
    n_newton_start=3,
)
# Conservative EM for partiality variants — fewer cycles to avoid the
# half1-specific NaN we saw with the partiality variant in job 846914.
EM_CFG_PART = EMConfig(
    n_em_cycles=2,
    min_obs=20,
    alpha_prior_lambda=0.1,
    n_shells=20,
    n_newton_final=3,
    n_newton_start=3,
)
EM_DEVICE = "cpu"          # GPU OOMs in LBFGS I0 fit at 16M obs

# Refinement
N_STEPS = 600
LR_OMEGA = 1e-4
LR_GAMMA_GLOBAL = 1e-3
LR_SIGMA = 1e-3
LR_ALPHA = 1e-2
LR_SCALEB = 1e-3
LAMBDA_ALPHA = 0.05
RESTRAINT_OMEGA = 100.0
RESTRAINT_LAMBDA = 2e4
RESTRAINT_LOG_G = 1.0
RESTRAINT_B = 0.01
# Direct sharpness regularizer on the partiality: penalize the per-HKL variance
# of log p.  Without this, the model overfits SASE spike structure and sends
# log p to the [-5, 5] clamp boundaries (see job 843030: CC1/2 0.99 → 0.71).
LAMBDA_LOGP_VAR = 5.0
FREE_FRACTION = 0.1
PATIENCE = 100
LOG_EVERY = 20
N_GG_QUAD = 11           # halved from 21 — 16M obs * 11 quad pts is the GPU memory limit
N_LAMBDA_USE = 32        # downsample λ axis to keep memory bounded (was 64)
N_ALPHA_SHELLS = 6
CHUNK_SIZE = 100_000

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
# Per-shot spectrum partiality (no learnable spectrum)
# ══════════════════════════════════════════════════════════════

class PerShotSpectrumPartiality(nn.Module):
    """Partiality from a fixed measured per-crystal spectrum.

    For observation j of crystal c with reciprocal vector q_j and resolution s_j:

        γ_eff² = γ_global² + (B_c/(8π²))·s_j² + (q_j · σ_ω)²
        p(|r_j|; q_j, c) = ∫ S_c(δλ/λ) · G(|r_j| - q_j·δλ/λ; γ_eff) d(δλ/λ)

    The crystal spectrum S_c is loaded from disk and stored as a buffer
    (no learnable parameters per crystal).
    """

    def __init__(
        self,
        spectra: Tensor,           # (C, N_λ) per-crystal measured spectra (sum=1)
        delta_lambda: Tensor,      # (N_λ,) δλ/λ axis
        gamma_global_init: float = 1e-3,
    ):
        super().__init__()
        self.register_buffer("spectra", spectra)
        self.register_buffer("delta_lambda", delta_lambda)
        d_dl = (delta_lambda[1] - delta_lambda[0]).item()
        self.register_buffer("d_dl", torch.tensor(d_dl))
        self.log_gamma_global = nn.Parameter(
            torch.tensor(math.log(gamma_global_init))
        )

    def gamma_eff(self, B_per_obs, s_sq, q_mag, sigma_omega):
        gamma_global_sq = torch.exp(self.log_gamma_global) ** 2
        disorder_sq = (B_per_obs / (8 * math.pi ** 2) * s_sq).clamp(min=0)
        orient_sq = (q_mag * sigma_omega) ** 2
        return (gamma_global_sq + disorder_sq + orient_sq).clamp(min=1e-10).sqrt()

    def log_p_raw(
        self,
        abs_r: Tensor,
        q_mag: Tensor,
        crystal_id: Tensor,
        B_per_obs: Tensor,
        s_sq: Tensor,
        sigma_omega: Tensor,
    ) -> Tensor:
        """Per-observation raw log partiality (NO per-HKL normalization).

        Pure per-obs computation, safe to call on a slice.
        """
        gamma_eff = self.gamma_eff(B_per_obs, s_sq, q_mag, sigma_omega)
        spec = self.spectra[crystal_id]                          # (N, N_λ)
        dl = self.delta_lambda[None, :]                          # (1, N_λ)
        offset = abs_r[:, None] - q_mag[:, None] * dl             # (N, N_λ)
        log_kernel = -0.5 * (offset / gamma_eff[:, None]) ** 2
        kernel = torch.exp(log_kernel.clamp(min=-30, max=0))
        p_raw = (spec * kernel).sum(dim=1).clamp(min=1e-10)
        return torch.log(p_raw)

    @torch.no_grad()
    def forward(
        self,
        abs_r: Tensor,
        q_mag: Tensor,
        crystal_id: Tensor,
        B_per_obs: Tensor,
        s_sq: Tensor,
        sigma_omega: Tensor,
        hkl_id: Tensor,
        n_hkl: int,
        chunk_size: int = CHUNK_SIZE,
    ) -> Tensor:
        """Diagnostic forward (no grad): per-obs normalized partiality."""
        N = abs_r.shape[0]
        log_p_full = torch.empty(N, device=abs_r.device, dtype=abs_r.dtype)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            sl = slice(start, end)
            log_p_full[sl] = self.log_p_raw(
                abs_r[sl], q_mag[sl], crystal_id[sl],
                B_per_obs[sl], s_sq[sl], sigma_omega,
            )
        means = self.compute_hkl_log_means(
            abs_r, q_mag, crystal_id, B_per_obs, s_sq, sigma_omega,
            hkl_id, n_hkl, chunk_size=chunk_size,
        )
        log_p_norm = log_p_full - means[hkl_id]
        return torch.exp(log_p_norm.clamp(min=-5, max=5))

    @torch.no_grad()
    def compute_hkl_log_means(
        self,
        abs_r: Tensor,
        q_mag: Tensor,
        crystal_id: Tensor,
        B_per_obs: Tensor,
        s_sq: Tensor,
        sigma_omega: Tensor,
        hkl_id: Tensor,
        n_hkl: int,
        chunk_size: int = CHUNK_SIZE,
    ) -> Tensor:
        """Per-HKL mean of log p_raw, computed under no_grad."""
        N = abs_r.shape[0]
        sum_log_p = torch.zeros(n_hkl, device=abs_r.device, dtype=abs_r.dtype)
        cnt = torch.zeros(n_hkl, device=abs_r.device, dtype=abs_r.dtype)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            sl = slice(start, end)
            log_p = self.log_p_raw(
                abs_r[sl], q_mag[sl], crystal_id[sl],
                B_per_obs[sl], s_sq[sl], sigma_omega,
            )
            sum_log_p.scatter_add_(0, hkl_id[sl], log_p)
            cnt.scatter_add_(0, hkl_id[sl], torch.ones_like(log_p))
        return sum_log_p / cnt.clamp(min=1)


# ══════════════════════════════════════════════════════════════
# Load matched data
# ══════════════════════════════════════════════════════════════

print(f"── Loading per-shot index from {INDEX_FILE} ──")
manifest_data = torch.load(INDEX_FILE, weights_only=False)
manifest = manifest_data["manifest"]
e_axis_eV = manifest_data["e_axis_eV"].numpy()
print(f"  {len(manifest)} delays, total matched: {manifest_data['n_matched']:,}")

# Concatenate all matched streams + spectra (in delay order)
streams = []
spec_chunks = []
for entry in manifest:
    if entry["matched_sfx"] is None:
        continue
    s_d = SFXStream.load(entry["matched_sfx"])
    spec_d = torch.load(entry["spectra_pt"], weights_only=False)["spectra"]
    assert spec_d.shape[0] == s_d.n_crystals, (
        f"shape mismatch in {entry['matched_sfx']}: "
        f"{spec_d.shape[0]} spectra vs {s_d.n_crystals} crystals"
    )
    streams.append(s_d)
    spec_chunks.append(spec_d)

s = SFXStream.concat(streams)
spectra_per_crystal = torch.cat(spec_chunks, dim=0)             # (C_total, N_E)
assert spectra_per_crystal.shape[0] == s.n_crystals
print(f"  Concatenated: {s.n_observations:,} obs, {s.n_crystals:,} crystals")

h1, h2 = s.split_random(n_parts=2, seed=SEED)

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

# ── Build δλ/λ axis from energy axis ──
E0 = 12960.0
delta_lambda_full = torch.from_numpy(-(e_axis_eV - E0) / E0).float()  # signed
# Downsample
idx_down = torch.linspace(0, len(delta_lambda_full) - 1, N_LAMBDA_USE).long()
delta_lambda_axis = delta_lambda_full[idx_down]
spectra_down = spectra_per_crystal[:, idx_down]
# Re-normalise after downsampling
spectra_down = spectra_down / spectra_down.sum(dim=1, keepdim=True).clamp(min=1e-10)
print(f"  Spectra: {spectra_down.shape}, δλ/λ ∈ [{delta_lambda_axis.min():.5f}, "
      f"{delta_lambda_axis.max():.5f}]")

# ── Baseline EM split-half ──
print("\n── BASELINE: EM Poisson ──")
dist_base = {}
for label, half in [("half1", h1), ("half2", h2)]:
    print(f"  {label}...")
    d = fit_em(half, config=EM_CFG, corrections=[ScaleBCorrection()], device=EM_DEVICE)
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
dist_full = fit_em(s, config=EM_CFG, corrections=[ScaleBCorrection()], device=EM_DEVICE)
print(f"  med(I₀)={dist_full.I0.median():.1f}, med(α)={dist_full.alpha.median():.4f}")

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
        beta_full_arr[j] = dist_full.beta[idx]
I0_per_obs = I0_full[hkl_id]
beta_per_obs = beta_full_arr[hkl_id]

_, log_G_init, B_init = fit_BG(core.I, hkl_id, cid, s_sq, n_unique, C)

noise_model = PoissonNoiseModel()
bg_total = core.background * noise_model.n_peak
n_photons = (core.I + bg_total).clamp(min=0)

# ══════════════════════════════════════════════════════════════
# JOINT REFINEMENT
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("JOINT REFINEMENT WITH PER-SHOT SPECTRA")
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

print(f"  Building softplus grid ({N_GG_QUAD} points)...")
with torch.no_grad():
    sig_lo = torch.full_like(n_ph_d, 0.5)
    grid = build_softplus_grid(n_ph_d, bg_d, n_quad=N_GG_QUAD, signal_lo_override=sig_lo)
log_poi_jac = grid.log_poi + grid.log_jac_dt + grid.log_dt[:, None]

exc_model = ExcitationError(C, UB, lambda_0).to(device)
scaleB = ScaleBModule(C).to(device)
B_init_d = B_init.to(device)
scaleB.init_from(log_G_init.to(device), B_init_d)
print(f"  B init: median={B_init.median().item():.2f} Å², "
      f"p10={B_init.quantile(0.1).item():.2f}, p90={B_init.quantile(0.9).item():.2f}")
partiality = PerShotSpectrumPartiality(
    spectra=spectra_down.to(device),
    delta_lambda=delta_lambda_axis.to(device),
    gamma_global_init=1e-3,
).to(device)

log_sigma_omega = nn.Parameter(
    torch.tensor(math.log(10e-3 * math.pi / 180), device=device)
)
log_alpha_shells = nn.Parameter(
    torch.full((N_ALPHA_SHELLS,), math.log(1.5), device=device)
)

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
    {"params": [partiality.log_gamma_global], "lr": LR_GAMMA_GLOBAL},
    {"params": [log_sigma_omega], "lr": LR_SIGMA},
    {"params": [log_alpha_shells], "lr": LR_ALPHA},
    {"params": [scaleB.log_G, scaleB.B], "lr": LR_SCALEB},
])

print(f"Observations: {n_work:,} work / {n_free:,} free")
print(f"Parameters:")
print(f"  ScaleB:    {2*C}")
print(f"  Orient:    {C*2} δω + {C} δλ")
print(f"  γ_global:  1")
print(f"  σ_ω:       1")
print(f"  α shells:  {N_ALPHA_SHELLS}")

best_free_nll = float("inf")
best_step = 0
best_state = None
nll_work_hist, nll_free_hist = [], []
alpha_hist, gamma_hist, sigma_hist = [], [], []

def free_nll_chunked(chunk_starts):
    """Compute free-set NLL across all chunks (no grad)."""
    nll_sum = 0.0
    with torch.no_grad():
        sigma_omega = torch.exp(log_sigma_omega)
        means = partiality.compute_hkl_log_means(
            exc_model.abs_excitation_error(hkl_obs_d, cid_d),
            q_mag_d, cid_d, scaleB.B[cid_d], s_sq_d, sigma_omega,
            hkl_id_d, n_unique,
        )
        alpha_per_obs_full = torch.exp(log_alpha_shells)[shell_id_d]
        scale_full = scaleB(cid_d, s_sq_d)
        for start in chunk_starts:
            end = min(start + LOSS_CHUNK_SIZE, N)
            sl = slice(start, end)
            abs_r_c = exc_model.abs_excitation_error(hkl_obs_d[sl], cid_d[sl])
            log_p = partiality.log_p_raw(
                abs_r_c, q_mag_d[sl], cid_d[sl],
                scaleB.B[cid_d[sl]], s_sq_d[sl], sigma_omega,
            )
            log_p_norm = (log_p - means[hkl_id_d[sl]]).clamp(-5, 5)
            p_eff_c = torch.exp(log_p_norm)
            log_sig = torch.log(
                (I0_d[sl] * p_eff_c * scale_full[sl]).clamp(min=1e-30)
            )
            eps = torch.log(grid.signal[sl].clamp(min=1e-30)) - log_sig[:, None]
            eps = eps.clamp(-5.0, 5.0)
            eps_abs = eps.abs().clamp(min=1e-6)
            log_gg = (-(eps_abs / alpha_per_obs_full[sl, None]).pow(beta_per_obs_d[sl, None])
                      + _log_gg_norm(beta_per_obs_d[sl])[:, None]
                      - torch.log(alpha_per_obs_full[sl, None]))
            log_marg = torch.logsumexp(log_gg + log_poi_jac[sl], dim=1)
            nll_sum += -(log_marg * free_mask[sl]).sum().item()
    return nll_sum / n_free


# Pre-compute chunk starts
LOSS_CHUNK_SIZE = 1_000_000     # 1M obs per loss chunk → ~5–8 GB GPU mem peak
chunk_starts = list(range(0, N, LOSS_CHUNK_SIZE))
print(f"  Loss computed in {len(chunk_starts)} chunks of {LOSS_CHUNK_SIZE:,}")

for step in range(N_STEPS):
    optimizer.zero_grad()

    # Pre-compute per-HKL log-means under no_grad (constant for this step)
    with torch.no_grad():
        sigma_omega_const = torch.exp(log_sigma_omega)
        abs_r_full = exc_model.abs_excitation_error(hkl_obs_d, cid_d)
        B_full = scaleB.B[cid_d]
        means_const = partiality.compute_hkl_log_means(
            abs_r_full, q_mag_d, cid_d, B_full, s_sq_d, sigma_omega_const,
            hkl_id_d, n_unique,
        )
        del abs_r_full, B_full

    # Chunked grad-accumulating loss
    nll_work_acc = 0.0
    nll_free_acc = 0.0
    var_logp_acc = 0.0
    for start in chunk_starts:
        end = min(start + LOSS_CHUNK_SIZE, N)
        sl = slice(start, end)
        n_chunk = end - start

        # Recompute scalars inside the chunk so each chunk has its own graph
        sigma_omega = torch.exp(log_sigma_omega)
        log_alpha_shells_exp = torch.exp(log_alpha_shells)

        abs_r_c = exc_model.abs_excitation_error(hkl_obs_d[sl], cid_d[sl])
        scale_c = scaleB(cid_d[sl], s_sq_d[sl])
        B_c = scaleB.B[cid_d[sl]]
        log_p_c = partiality.log_p_raw(
            abs_r_c, q_mag_d[sl], cid_d[sl], B_c, s_sq_d[sl], sigma_omega,
        )
        # Note: do NOT clamp here so the variance penalty has gradient signal
        # at the tails. The clamp is only applied for the EM-merge call later.
        log_p_norm = log_p_c - means_const[hkl_id_d[sl]]
        log_p_norm_clamped = log_p_norm.clamp(-5, 5)
        p_eff_c = torch.exp(log_p_norm_clamped)

        alpha_c = log_alpha_shells_exp[shell_id_d[sl]]
        log_sig = torch.log((I0_d[sl] * p_eff_c * scale_c).clamp(min=1e-30))
        eps = torch.log(grid.signal[sl].clamp(min=1e-30)) - log_sig[:, None]
        eps = eps.clamp(-5.0, 5.0)
        eps_abs = eps.abs().clamp(min=1e-6)

        log_gg = (-(eps_abs / alpha_c[:, None]).pow(beta_per_obs_d[sl, None])
                  + _log_gg_norm(beta_per_obs_d[sl])[:, None]
                  - torch.log(alpha_c[:, None]))
        log_marg = torch.logsumexp(log_gg + log_poi_jac[sl], dim=1)

        nll_chunk_work = -(log_marg * work_mask[sl]).sum() / n_work

        # Sharpness regularizer: per-obs squared log p, summed over chunk and
        # divided by N (mean across all obs). Since per-HKL mean(log p) ≈ 0,
        # mean(log p²) ≈ multiplicity-weighted mean of Var(log p) per HKL.
        var_chunk = LAMBDA_LOGP_VAR * (log_p_norm ** 2).sum() / N

        loss_chunk = nll_chunk_work + var_chunk
        loss_chunk.backward()

        nll_work_acc += nll_chunk_work.item()
        var_logp_acc += var_chunk.item()
        with torch.no_grad():
            nll_free_acc += -(log_marg * free_mask[sl]).sum().item()

        del abs_r_c, scale_c, B_c, log_p_c, log_p_norm, log_p_norm_clamped, p_eff_c
        del alpha_c, log_sig, eps, eps_abs, log_gg, log_marg
        del nll_chunk_work, var_chunk, loss_chunk

    # Restraints (cheap, single backward call)
    omega_pen = RESTRAINT_OMEGA * (exc_model.delta_omega_xy ** 2).mean()
    lambda_pen = RESTRAINT_LAMBDA * (exc_model.delta_lambda ** 2).mean()
    logG_pen = RESTRAINT_LOG_G * (scaleB.log_G ** 2).mean()
    B_pen = RESTRAINT_B * (scaleB.B ** 2).mean()
    alpha_pen = LAMBDA_ALPHA * torch.exp(log_alpha_shells).mean()
    pen = omega_pen + lambda_pen + logG_pen + B_pen + alpha_pen
    pen.backward()

    if math.isnan(nll_work_acc) or math.isinf(nll_work_acc):
        print(f"  step {step}: NaN/Inf NLL — stopping")
        break

    has_nan = any(p.grad is not None and p.grad.isnan().any() for p in all_params)
    if has_nan:
        print(f"  step {step}: NaN gradient — skipping")
        optimizer.zero_grad()
        continue

    torch.nn.utils.clip_grad_norm_(all_params, max_norm=10.0)
    optimizer.step()

    with torch.no_grad():
        scaleB.log_G -= scaleB.log_G.mean()

    if step % LOG_EVERY == 0 or step == N_STEPS - 1:
        with torch.no_grad():
            nll_w = nll_work_acc
            nll_f = nll_free_acc / n_free
            sigma_deg = math.degrees(torch.exp(log_sigma_omega).item()) * 1000
            dw_med = exc_model.delta_omega_xy.norm(dim=1).median().item() * 1e3
            alpha_vals = torch.exp(log_alpha_shells).cpu().numpy()
            gamma_global = torch.exp(partiality.log_gamma_global).item()

        nll_work_hist.append(nll_w)
        nll_free_hist.append(nll_f)
        alpha_hist.append(alpha_vals.copy())
        gamma_hist.append(gamma_global)
        sigma_hist.append(sigma_deg)

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
              f"varP={var_logp_acc:.3f}  "
              f"α={alpha_vals.min():.2f}→{alpha_vals.max():.2f}  "
              f"γ_g={gamma_global*1e3:.2f}e-3  σ_ω={sigma_deg:.1f} mdeg  "
              f"|δω|={dw_med:.2f} mdeg")

        if PATIENCE > 0 and step - best_step > PATIENCE:
            print(f"  Early stop: free NLL peaked at step {best_step}")
            break

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

# ══════════════════════════════════════════════════════════════
# Apply to split-half merge
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SPLIT-HALF with per-shot spectrum partiality")
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

    # Control: partiality with PHYSICAL default params (no learning) and
    # ORIGINAL crystal orientations.  This isolates whether the spectrum
    # information itself helps, independent of the joint refinement.
    exc_default = ExcitationError(C, UB, lambda_0).to(device)
    abs_r_default = exc_default.abs_excitation_error(hkl_obs_d, cid_d)
    sigma_omega_default = torch.tensor(10e-3 * math.pi / 180, device=device)
    saved_log_gamma = partiality.log_gamma_global.detach().clone()
    partiality.log_gamma_global.data.copy_(torch.tensor(math.log(1e-3)))
    p_default = partiality(
        abs_r_default, q_mag_d, cid_d,
        torch.zeros_like(B_final),     # no disorder term
        s_sq_d, sigma_omega_default,
        hkl_id_d, n_unique,
    ).cpu()
    partiality.log_gamma_global.data.copy_(saved_log_gamma)

print(f"  Partiality (refined): median={p_full.median():.4f}, "
      f"p10={p_full.quantile(0.1):.4f}, p90={p_full.quantile(0.9):.4f}")
print(f"  Partiality (default): median={p_default.median():.4f}, "
      f"p10={p_default.quantile(0.1):.4f}, p90={p_default.quantile(0.9):.4f}")

# Reproduce split assignment
gen_split = torch.Generator()
gen_split.manual_seed(SEED)
assignment = torch.randint(2, (s.n_crystals,), generator=gen_split)
crystal_assignment_per_obs = assignment[s._core.crystal_id.long()]


def renormalize_half(p_full_tensor, obs_mask, half_core):
    half_part = p_full_tensor[obs_mask]
    half_hkl_idx_loc = compute_hkl_index(half_core.hkl)
    log_p_half = torch.log(half_part.clamp(min=1e-10))
    n_u = half_hkl_idx_loc.n_unique
    hid = half_hkl_idx_loc.hkl_id
    sum_lp = torch.zeros(n_u).scatter_add_(0, hid, log_p_half)
    cnt = torch.zeros(n_u).scatter_add_(0, hid, torch.ones_like(log_p_half))
    mean_lp = sum_lp / cnt.clamp(min=1)
    return torch.exp(log_p_half - mean_lp[hid])


variants = [
    ("no partiality", None),
    ("default params", p_default),
    ("per-shot spec", p_full),
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

        # Use conservative EM for partiality variants (avoids half1 NaN)
        cfg = EM_CFG if p_source is None else EM_CFG_PART
        d = fit_em(half, config=cfg, corrections=[ScaleBCorrection()],
                   device=EM_DEVICE, partiality=half_partiality)
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

dist_part, m_part, part_alpha = results["per-shot spec"]
cc_part, r_part = cc_half(m_part), r_split(m_part)

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
print(f"  {'d_max':>7s} {'d_min':>7s}  {'Baseline':>10s}  {'Per-shot':>10s}  {'Delta':>8s}")
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
    ("+ Per-shot", shells_part, "C3", "s"),
]:
    d_mid = [0.5 * (sh.d_max + sh.d_min) for sh in shells]
    ax.plot(d_to_x(d_mid), [sh.cc_half for sh in shells], "-",
            color=color, marker=marker, ms=5, lw=1.8, label=name)
ax.set_ylabel("CC1/2"); ax.legend(loc="lower left")
ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
setup_res_axis(ax)

fig.suptitle("Joint Refinement with Per-Shot Measured SASE Spectra", fontsize=13)
fig.tight_layout()
fig.savefig(OUTDIR / "refine_per_shot_partiality.png", dpi=150)
print(f"\nSaved refine_per_shot_partiality.png")

torch.save({
    "cc_base": cc_base, "r_base": r_base, "base_alpha": base_alpha,
    "cc_part": cc_part, "r_part": r_part, "part_alpha": part_alpha,
    "alpha_final": alpha_final,
    "gamma_final": gamma_final,
    "sigma_omega_mdeg": sigma_final,
    "nll_work": nll_work_hist, "nll_free": nll_free_hist,
    "n_crystals": s.n_crystals,
    "n_observations": s.n_observations,
}, OUTDIR / "refine_per_shot_partiality_summary.pt")

print("\nDone.")
