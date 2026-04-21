#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH -o /das/work/p17/p17490/Peter/torchsx/sfx_scan/partiality_empirical.out
#SBATCH -c 16
#SBATCH -p day
#SBATCH -t 1:00:00

"""Empirical partiality analysis: excitation error vs observed intensity.

Uses the existing EwaldPartiality model (no refinement, just geometry)
to compute excitation error per observation, then correlates with
the residual log(I_obs) - μ_h to reveal the partiality function.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from torchsx.stream import SFXStream
from torchsx.merge._distribution import fit_gengauss
from torchsx.merge._hkl import compute_hkl_index
from torchsx.scaling._partiality import EwaldPartiality

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

# ── Step 1: GenGauss+EM fit to get μ_h per HKL ──────────
print("\nFitting GenGauss+EM (method C: scale_crystals=True)...")
fit = fit_gengauss(s, n_em_cycles=N_EM, scale_crystals=True, **FIT_KW)
print(f"  {fit.hkl.shape[0]} HKLs fitted, med(α)={fit.alpha.median():.4f}")

# ── Step 2: Compute excitation error per observation ──────
print("\nComputing excitation errors...")

# Build UB matrices from astar/bstar/cstar
# UB = [astar | bstar | cstar]^T → (n_crystals, 3, 3)
UB = torch.stack([core.astar, core.bstar, core.cstar], dim=1)  # (C, 3, 3)

# Wavelength from beam energy (median)
# E = hc/λ → λ = hc/E; hc = 12398.42 eV·Å
beam_energy = core.beam_energy[core.image_id]  # per-crystal
median_energy = beam_energy.median().item()
wavelength = 12398.42 / median_energy
print(f"  Median beam energy: {median_energy:.1f} eV → λ = {wavelength:.4f} Å")

# Create EwaldPartiality (no refinement — just compute excitation error from geometry)
ewald = EwaldPartiality(
    n_crystals=core.cell.shape[0],
    UB=UB,
    wavelength=wavelength,
)

with torch.no_grad():
    r = ewald.excitation_error(core.hkl.float(), core.crystal_id)

print(f"  Excitation error: median |r| = {r.abs().median():.6f} Å⁻¹")
print(f"  r range: [{r.min():.4f}, {r.max():.4f}] Å⁻¹")
print(f"  % with |r| < 0.01: {(r.abs() < 0.01).float().mean()*100:.1f}%")
print(f"  % with |r| < 0.05: {(r.abs() < 0.05).float().mean()*100:.1f}%")

# ── Step 3: Map fitted μ_h back to observations ──────────
print("\nMapping μ_h to observations...")
from torchref.base.reciprocal.hkl import get_d_spacing

# Match observation HKLs to fitted HKLs
hkl_idx = compute_hkl_index(core.hkl)
fit_hkl_idx = compute_hkl_index(fit.hkl)

# Build lookup: fitted HKL tuple → index
fit_lookup = {}
for i in range(fit.hkl.shape[0]):
    key = tuple(fit.hkl[i].tolist())
    fit_lookup[key] = i

# Map each observation to its fitted HKL
n_obs = core.I.shape[0]
obs_fit_idx = torch.full((n_obs,), -1, dtype=torch.long)
for j in range(n_obs):
    key = tuple(core.hkl[j].tolist())
    if key in fit_lookup:
        obs_fit_idx[j] = fit_lookup[key]

has_fit = obs_fit_idx >= 0
pos_mask = core.I > 0
valid = has_fit & pos_mask
print(f"  {valid.sum()}/{n_obs} observations matched to fitted HKLs")

# Extract matched data
log_I = torch.log(core.I[valid])
mu_obs = fit.mu[obs_fit_idx[valid]]
r_valid = r[valid]
alpha_obs = fit.alpha[obs_fit_idx[valid]]
d_obs = fit.d_spacing[obs_fit_idx[valid]]
s_sq_obs = 1.0 / (4.0 * d_obs ** 2)

# Residual: log(I_obs) - μ_h ≈ log(partiality) + noise
residual = log_I - mu_obs

print(f"  Residual: median={residual.median():.4f}, std={residual.std():.4f}")

# ── Step 4: Bin by |r| and resolution ──────────────────────
print("\nBinning by excitation error and resolution...")

r_abs = r_valid.abs()

# Resolution shells
d_edges = torch.tensor([10.0, 2.0, 1.5, 1.2, 1.0, 0.9])
n_res = len(d_edges) - 1
res_labels = [f"{d_edges[i]:.1f}-{d_edges[i+1]:.1f} Å" for i in range(n_res)]

# |r| bins (finer at small |r| where partiality changes fastest)
r_edges = np.array([0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30])
n_rbins = len(r_edges) - 1

# Compute median residual in each (|r|, resolution) bin
print(f"\n{'|r| range':>15s}", end="")
for lab in res_labels:
    print(f"  {lab:>12s}", end="")
print(f"  {'All':>12s}")
print("-" * (15 + 14 * (n_res + 1)))

median_grid = np.full((n_rbins, n_res), np.nan)
count_grid = np.zeros((n_rbins, n_res), dtype=int)
median_all = np.full(n_rbins, np.nan)

r_abs_np = r_abs.numpy()
d_np = d_obs.numpy()
res_np = residual.numpy()

for i in range(n_rbins):
    r_mask = (r_abs_np >= r_edges[i]) & (r_abs_np < r_edges[i+1])
    r_label = f"[{r_edges[i]:.3f},{r_edges[i+1]:.3f})"

    if r_mask.sum() > 10:
        median_all[i] = np.median(res_np[r_mask])

    print(f"{r_label:>15s}", end="")
    for j in range(n_res):
        d_mask = (d_np <= d_edges[j].item()) & (d_np > d_edges[j+1].item())
        mask = r_mask & d_mask
        n = mask.sum()
        count_grid[i, j] = n
        if n > 50:
            med = np.median(res_np[mask])
            median_grid[i, j] = med
            print(f"  {med:+10.4f}({n:>0d})", end="")
        else:
            print(f"  {'—':>12s}", end="")

    if r_mask.sum() > 10:
        print(f"  {median_all[i]:+10.4f}", end="")
    print()

# ── Step 5: Fit simple Gaussian partiality model ──────────
# log(p) = -r²/(2σ²) → median_residual ≈ -r²/(2σ²) + const
# For each resolution shell, fit σ from the slope of residual vs r²
print(f"\n── Empirical relp width (σ_eff) per resolution shell ──")
print(f"{'Shell':>15s} {'σ_eff (Å⁻¹)':>12s} {'FWHM (Å⁻¹)':>12s} {'n_obs':>8s}")

sigma_per_shell = []
for j in range(n_res):
    d_mask = (d_np <= d_edges[j].item()) & (d_np > d_edges[j+1].item())
    r2 = r_abs_np[d_mask] ** 2
    res_shell = res_np[d_mask]

    # Robust fit: bin median residual vs r², fit line through origin
    r2_bins = np.array([r_edges[i]**2 for i in range(len(r_edges))])
    r2_mid = []
    med_r = []
    for i in range(n_rbins):
        rm = (r_abs_np[d_mask] >= r_edges[i]) & (r_abs_np[d_mask] < r_edges[i+1])
        if rm.sum() > 50:
            r2_mid.append((r_edges[i]**2 + r_edges[i+1]**2) / 2)
            med_r.append(np.median(res_shell[rm]))

    if len(r2_mid) >= 3:
        r2_mid = np.array(r2_mid)
        med_r = np.array(med_r)
        # Subtract the intercept (r=0 value) and fit slope
        intercept = med_r[0]
        delta = med_r - intercept
        # Weighted least squares: delta = -r²/(2σ²) → slope = -1/(2σ²)
        # slope = Σ(r2 * delta) / Σ(r2²)
        slope = np.sum(r2_mid * delta) / np.sum(r2_mid ** 2)
        if slope < -1e-6:
            sigma = np.sqrt(-1.0 / (2 * slope))
            fwhm = 2.355 * sigma
        else:
            sigma = float('inf')
            fwhm = float('inf')
    else:
        sigma = float('nan')
        fwhm = float('nan')

    sigma_per_shell.append(sigma)
    n_shell = d_mask.sum()
    print(f"{res_labels[j]:>15s} {sigma:12.5f} {fwhm:12.5f} {n_shell:8d}")


# ── Step 6: Plots ──────────────────────────────────────────
OUTDIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# (a) Histogram of |r|
ax = axes[0, 0]
ax.hist(r_abs_np, bins=100, range=(0, 0.3), density=True, alpha=0.7)
ax.set_xlabel("|r| (Å⁻¹)")
ax.set_ylabel("Density")
ax.set_title("Excitation error distribution")
ax.axvline(r_abs_np.mean(), ls="--", color="red", label=f"mean={r_abs_np.mean():.4f}")
ax.legend()

# (b) Median residual vs |r|, per resolution shell
ax = axes[0, 1]
r_mid = [(r_edges[i] + r_edges[i+1]) / 2 for i in range(n_rbins)]
for j in range(n_res):
    valid_pts = ~np.isnan(median_grid[:, j])
    if valid_pts.any():
        ax.plot(np.array(r_mid)[valid_pts], median_grid[valid_pts, j],
                "o-", label=res_labels[j], markersize=4)
ax.set_xlabel("|r| (Å⁻¹)")
ax.set_ylabel("Median residual (log I_obs − μ_h)")
ax.set_title("Partiality signal: residual vs |r|")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# (c) Same but vs r²
ax = axes[0, 2]
r2_mid = [(r_edges[i]**2 + r_edges[i+1]**2) / 2 for i in range(n_rbins)]
for j in range(n_res):
    valid_pts = ~np.isnan(median_grid[:, j])
    if valid_pts.any():
        ax.plot(np.array(r2_mid)[valid_pts], median_grid[valid_pts, j],
                "o-", label=res_labels[j], markersize=4)
ax.set_xlabel("|r|² (Å⁻²)")
ax.set_ylabel("Median residual")
ax.set_title("Partiality vs r² (should be linear for Gaussian)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# (d) σ_eff vs resolution
ax = axes[1, 0]
d_mid = [(d_edges[j] + d_edges[j+1]) / 2 for j in range(n_res)]
s_sq_mid = [1/(4*d**2) for d in d_mid]
valid_s = [i for i, s in enumerate(sigma_per_shell) if np.isfinite(s)]
if valid_s:
    ax.plot([s_sq_mid[i] for i in valid_s],
            [sigma_per_shell[i] for i in valid_s], "ko-")
ax.set_xlabel("s² = 1/(4d²) (Å⁻²)")
ax.set_ylabel("σ_eff (Å⁻¹)")
ax.set_title("Relp width vs resolution")
ax.grid(True, alpha=0.3)

# (e) 2D heatmap: median residual
ax = axes[1, 1]
im = ax.imshow(median_grid.T, aspect="auto", origin="lower",
               extent=[0, n_rbins, 0, n_res], cmap="RdBu_r", vmin=-1, vmax=0.2)
ax.set_xticks(np.arange(n_rbins) + 0.5)
ax.set_xticklabels([f"{r_edges[i]:.3f}" for i in range(n_rbins)], fontsize=6, rotation=45)
ax.set_yticks(np.arange(n_res) + 0.5)
ax.set_yticklabels(res_labels, fontsize=8)
ax.set_xlabel("|r| bin lower edge (Å⁻¹)")
ax.set_title("Median residual heatmap")
fig.colorbar(im, ax=ax)

# (f) Scatter: r vs residual (subsample)
ax = axes[1, 2]
idx = torch.randperm(len(r_valid))[:50000]
ax.scatter(r_abs_np[idx], res_np[idx], s=0.5, alpha=0.1, c="C0")
# Overlay median trend
valid_all = ~np.isnan(median_all)
if valid_all.any():
    ax.plot(np.array(r_mid)[valid_all], median_all[valid_all],
            "r-o", lw=2, markersize=4, label="median")
ax.set_xlabel("|r| (Å⁻¹)")
ax.set_ylabel("log(I_obs) − μ_h")
ax.set_title("Residual vs excitation error")
ax.set_ylim(-4, 4)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = OUTDIR / "partiality_empirical.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved {out_path}")
