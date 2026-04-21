#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH -o /das/work/p17/p17490/Peter/torchsx/sfx_scan/angular_modulation.out
#SBATCH -c 16
#SBATCH -p day
#SBATCH -t 1:00:00

"""Empirical angular modulation analysis.

For each observation, compute the diffracted beam direction in the lab frame
and look for systematic residual patterns as a function of (2θ, φ).
Fit low-order spherical harmonics (global parameters only) and measure
variance reduction.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import sph_harm_y

from torchsx.stream import SFXStream
from torchsx.merge._distribution import fit_gengauss
from torchsx.merge._hkl import compute_hkl_index

# ── Configuration ──────────────────────────────────────────
STREAM_DIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/streams")
DELAYS = [-91, -136, -182, -227, -273]
OUTDIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/resolution_study")
FIT_KW = dict(min_obs=20, n_shells=20, n_smooth_iter=15, smooth_lambda=0.3)
N_EM = 10

# ── Load data ──────────────────────────────────────────────
print("Loading streams...")
streams = []
for d in DELAYS:
    path = STREAM_DIR / f"delay_{d}.stream"
    print(f"  {path.name} ...", end=" ", flush=True)
    st = SFXStream.from_stream(str(path), n_workers=16)
    print(f"{st.n_crystals} crystals, {st.n_observations} obs")
    streams.append(st)

s = SFXStream.concat(streams)
core = s._core
print(f"\nTotal: {core.cell.shape[0]} crystals, {core.I.shape[0]} observations")

# ── Step 1: GenGauss fit → μ_h ────────────────────────────
print("\nFitting GenGauss+EM...")
fit = fit_gengauss(s, n_em_cycles=N_EM, scale_crystals=True, **FIT_KW)
print(f"  {fit.hkl.shape[0]} HKLs, med(α)={fit.alpha.median():.4f}")

# ── Step 2: Map μ_h back to observations ──────────────────
print("\nMapping μ_h to observations...")
fit_lookup = {}
for i in range(fit.hkl.shape[0]):
    key = tuple(fit.hkl[i].tolist())
    fit_lookup[key] = i

n_obs = core.I.shape[0]
obs_fit_idx = torch.full((n_obs,), -1, dtype=torch.long)
for j in range(n_obs):
    key = tuple(core.hkl[j].tolist())
    if key in fit_lookup:
        obs_fit_idx[j] = fit_lookup[key]

valid = (obs_fit_idx >= 0) & (core.I > 0)
print(f"  {valid.sum()}/{n_obs} valid observations")

log_I = torch.log(core.I[valid])
mu_obs = fit.mu[obs_fit_idx[valid]]
residual = log_I - mu_obs
d_obs = fit.d_spacing[obs_fit_idx[valid]]

# ── Step 3: Compute scattering vector direction ───────────
print("\nComputing scattering vectors...")

UB = torch.stack([core.astar, core.bstar, core.cstar], dim=1)  # (C,3,3)
crystal_id_valid = core.crystal_id[valid].long()
hkl_valid = core.hkl[valid].float()

# q_lab = UB @ h for each observation
UB_obs = UB[crystal_id_valid]  # (N, 3, 3)
q_lab = torch.einsum("nij,nj->ni", UB_obs, hkl_valid)  # (N, 3)

# Diffracted beam direction: k_f = q + s0
beam_energy = core.beam_energy[core.image_id]
median_energy = beam_energy.median().item()
wavelength = 12398.42 / median_energy
inv_wl = 1.0 / wavelength
s0 = torch.tensor([0.0, 0.0, inv_wl])

k_f = q_lab + s0  # (N, 3) outgoing wavevector
k_f_norm = k_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
k_f_hat = k_f / k_f_norm  # unit vector

# Spherical coordinates
# θ = polar angle from beam axis (z), φ = azimuthal angle
cos_theta = k_f_hat[:, 2].clamp(-1, 1)
theta = torch.acos(cos_theta)  # 0 (forward) to π (backward)
phi = torch.atan2(k_f_hat[:, 1], k_f_hat[:, 0])  # -π to π

# 2θ scattering angle (= π - θ for backscattering convention, but for
# forward scattered: 2θ = θ since s0 is along z)
two_theta = theta

print(f"  2θ range: [{two_theta.min()*180/3.14159:.1f}°, {two_theta.max()*180/3.14159:.1f}°]")
print(f"  median 2θ: {two_theta.median()*180/3.14159:.1f}°")

# ── Step 4: Build spherical harmonic basis ────────────────
print("\nBuilding spherical harmonic basis...")

theta_np = theta.numpy()
phi_np = phi.numpy()
res_np = residual.numpy()

# Real spherical harmonics up to l_max
# Y_lm(θ, φ) for m ≥ 0: Re[Y_lm], for m < 0: Im[Y_l|m|]
L_MAX = 6
basis_columns = []
basis_labels = []

for l in range(L_MAX + 1):
    for m in range(-l, l + 1):
        # Complex → real spherical harmonics
        if m > 0:
            Y = np.sqrt(2) * sph_harm_y(l, m, theta_np, phi_np).real
        elif m == 0:
            Y = sph_harm_y(l, 0, theta_np, phi_np).real
        else:  # m < 0
            Y = np.sqrt(2) * sph_harm_y(l, -m, theta_np, phi_np).imag
        basis_columns.append(Y.astype(np.float64))
        basis_labels.append(f"Y_{l},{m}")

X = np.column_stack(basis_columns)  # (N, n_basis)
n_basis = X.shape[1]
print(f"  l_max={L_MAX}: {n_basis} basis functions, {len(res_np)} observations")

# ── Step 5: Fit global spherical harmonic model ───────────
print("\nFitting spherical harmonic model (OLS)...")

# Solve X @ c = residual via normal equations (X'X is tiny: n_basis × n_basis)
XtX = X.T @ X  # (n_basis, n_basis)
Xty = X.T @ res_np.astype(np.float64)  # (n_basis,)
c = np.linalg.solve(XtX, Xty)  # (n_basis,)

# Predicted modulation
pred = X @ c
residual_corrected = res_np - pred

# Variance reduction
var_before = np.var(res_np)
var_after = np.var(residual_corrected)
var_reduction = 1.0 - var_after / var_before
print(f"  Variance before: {var_before:.6f}")
print(f"  Variance after:  {var_after:.6f}")
print(f"  Variance reduction: {var_reduction*100:.3f}%")

# Coefficients
print(f"\n  Top coefficients by |c|:")
order = np.argsort(np.abs(c))[::-1]
for i in order[:15]:
    print(f"    {basis_labels[i]:>8s}: {c[i]:+.6f}")

# ── Step 6: Scan l_max to see diminishing returns ────────
print(f"\n── Variance reduction vs l_max ──")
print(f"{'l_max':>5s} {'n_params':>8s} {'var_red%':>10s} {'Δvar_red%':>10s}")

prev_vr = 0.0
for lmax_test in range(L_MAX + 1):
    n_params_test = (lmax_test + 1) ** 2
    X_test = X[:, :n_params_test]
    XtX_test = X_test.T @ X_test
    Xty_test = X_test.T @ res_np.astype(np.float64)
    c_test = np.linalg.solve(XtX_test, Xty_test)
    pred_test = X_test @ c_test
    vr = 1.0 - np.var(res_np - pred_test) / var_before
    delta = vr - prev_vr
    print(f"{lmax_test:5d} {n_params_test:8d} {vr*100:10.4f} {delta*100:+10.4f}")
    prev_vr = vr

# ── Step 7: Per-resolution-shell analysis ─────────────────
print(f"\n── Variance reduction per resolution shell ──")
d_edges = [10.0, 2.0, 1.5, 1.2, 1.0, 0.9]
d_np = d_obs.numpy()

print(f"{'Shell':>12s} {'var_before':>10s} {'var_after':>10s} {'var_red%':>10s}")
for i in range(len(d_edges) - 1):
    mask = (d_np <= d_edges[i]) & (d_np > d_edges[i + 1])
    if mask.sum() < 100:
        continue
    vb = np.var(res_np[mask])
    va = np.var(residual_corrected[mask])
    vr = 1.0 - va / vb
    print(f"{d_edges[i]:.1f}-{d_edges[i+1]:.1f} Å {vb:10.4f} {va:10.4f} {vr*100:10.3f}%")

# ── Step 8: Plots ─────────────────────────────────────────
OUTDIR.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# (a) 2D residual map in (2θ, φ) — before correction
ax = axes[0, 0]
# Bin into 2D grid
n_theta_bins, n_phi_bins = 50, 60
theta_edges = np.linspace(theta_np.min(), theta_np.max(), n_theta_bins + 1)
phi_edges = np.linspace(-np.pi, np.pi, n_phi_bins + 1)

grid_sum = np.zeros((n_theta_bins, n_phi_bins))
grid_cnt = np.zeros((n_theta_bins, n_phi_bins))
theta_idx = np.clip(np.digitize(theta_np, theta_edges) - 1, 0, n_theta_bins - 1)
phi_idx = np.clip(np.digitize(phi_np, phi_edges) - 1, 0, n_phi_bins - 1)
np.add.at(grid_sum, (theta_idx, phi_idx), res_np)
np.add.at(grid_cnt, (theta_idx, phi_idx), 1)
grid_mean = np.where(grid_cnt > 50, grid_sum / grid_cnt, np.nan)

im = ax.imshow(grid_mean.T, aspect="auto", origin="lower",
               extent=[np.degrees(theta_np.min()), np.degrees(theta_np.max()),
                       -180, 180],
               cmap="RdBu_r", vmin=-1.0, vmax=-0.4)
ax.set_xlabel("2θ (°)")
ax.set_ylabel("φ (°)")
ax.set_title("Mean residual (before)")
fig.colorbar(im, ax=ax, shrink=0.8)

# (b) Same after correction
ax = axes[0, 1]
grid_sum2 = np.zeros((n_theta_bins, n_phi_bins))
np.add.at(grid_sum2, (theta_idx, phi_idx), residual_corrected)
grid_mean2 = np.where(grid_cnt > 50, grid_sum2 / grid_cnt, np.nan)

im = ax.imshow(grid_mean2.T, aspect="auto", origin="lower",
               extent=[np.degrees(theta_np.min()), np.degrees(theta_np.max()),
                       -180, 180],
               cmap="RdBu_r", vmin=-1.0, vmax=-0.4)
ax.set_xlabel("2θ (°)")
ax.set_ylabel("φ (°)")
ax.set_title("Mean residual (after l_max=6)")
fig.colorbar(im, ax=ax, shrink=0.8)

# (c) Predicted modulation pattern
ax = axes[0, 2]
grid_pred = np.zeros((n_theta_bins, n_phi_bins))
np.add.at(grid_pred, (theta_idx, phi_idx), pred)
grid_pred_mean = np.where(grid_cnt > 50, grid_pred / grid_cnt, np.nan)
vmax_pred = np.nanmax(np.abs(grid_pred_mean - np.nanmedian(grid_pred_mean)))
im = ax.imshow(grid_pred_mean.T, aspect="auto", origin="lower",
               extent=[np.degrees(theta_np.min()), np.degrees(theta_np.max()),
                       -180, 180],
               cmap="RdBu_r",
               vmin=np.nanmedian(grid_pred_mean) - vmax_pred,
               vmax=np.nanmedian(grid_pred_mean) + vmax_pred)
ax.set_xlabel("2θ (°)")
ax.set_ylabel("φ (°)")
ax.set_title("Fitted SH modulation")
fig.colorbar(im, ax=ax, shrink=0.8)

# (d) Variance reduction vs l_max
ax = axes[1, 0]
lmax_vals = list(range(L_MAX + 1))
vr_vals = []
for lmax_test in lmax_vals:
    n_p = (lmax_test + 1) ** 2
    X_t = X[:, :n_p]
    c_t = np.linalg.solve(X_t.T @ X_t, X_t.T @ res_np.astype(np.float64))
    vr_vals.append((1.0 - np.var(res_np - X_t @ c_t) / var_before) * 100)
ax.bar(lmax_vals, vr_vals, color="steelblue")
ax.set_xlabel("l_max")
ax.set_ylabel("Variance reduction (%)")
ax.set_title("Diminishing returns with l_max")

# (e) Coefficient magnitudes
ax = axes[1, 1]
abs_c = np.abs(c)
ax.bar(range(n_basis), abs_c, width=0.8, color="coral")
# Mark l boundaries
cumulative = 0
for l in range(L_MAX + 1):
    n_m = 2 * l + 1
    ax.axvline(cumulative - 0.5, ls=":", color="gray", alpha=0.5)
    ax.text(cumulative + n_m / 2 - 0.5, abs_c.max() * 0.95, f"l={l}",
            ha="center", fontsize=7)
    cumulative += n_m
ax.set_xlabel("Basis function index")
ax.set_ylabel("|coefficient|")
ax.set_title("SH coefficient magnitudes")

# (f) Azimuthal dependence at fixed 2θ
ax = axes[1, 2]
# Pick a 2θ band (20-30°) and plot residual vs φ
for theta_lo, theta_hi, label in [(10, 20, "2θ=10-20°"),
                                   (20, 30, "2θ=20-30°"),
                                   (30, 45, "2θ=30-45°")]:
    t_mask = (np.degrees(theta_np) >= theta_lo) & (np.degrees(theta_np) < theta_hi)
    if t_mask.sum() < 1000:
        continue
    phi_bins_1d = np.linspace(-np.pi, np.pi, 37)
    phi_mid = (phi_bins_1d[:-1] + phi_bins_1d[1:]) / 2
    medians = []
    for k in range(36):
        pm = t_mask & (phi_np >= phi_bins_1d[k]) & (phi_np < phi_bins_1d[k + 1])
        if pm.sum() > 50:
            medians.append(np.median(res_np[pm]))
        else:
            medians.append(np.nan)
    ax.plot(np.degrees(phi_mid), medians, "o-", markersize=3, label=label)
ax.set_xlabel("φ (°)")
ax.set_ylabel("Median residual")
ax.set_title("Azimuthal modulation at fixed 2θ")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = OUTDIR / "angular_modulation.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved {out_path}")
