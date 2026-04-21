#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH -o /das/work/p17/p17490/Peter/torchsx/sfx_scan/feature_importance.out
#SBATCH -c 16
#SBATCH -p day
#SBATCH -t 2:00:00

"""Fully empirical feature importance analysis.

Throw every available parameter from the stream into a regression
against the residual log(I_obs) - μ_h.  No physics assumptions —
just let the data tell us what explains variance.

Features:
  Per-observation: fs, ss, panel_id, peak, background, d_spacing, s²
  Per-crystal:     cell (a,b,c,α,β,γ), profile_rad, resolution,
                   astar/bstar/cstar (9 components)
  Per-image:       beam_energy, camera_length
  Derived:         2θ, φ, |q|, k_f components
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

# ── Step 2: Map μ_h to observations ──────────────────────
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
residual = (log_I - mu_obs).numpy()
d_obs = fit.d_spacing[obs_fit_idx[valid]]

# ── Step 3: Build feature matrix ──────────────────────────
print("\nBuilding feature matrix...")

cid = core.crystal_id[valid].long()
iid = core.image_id[cid]  # image per crystal

# Scattering geometry
UB = torch.stack([core.astar, core.bstar, core.cstar], dim=1)
hkl_f = core.hkl[valid].float()
q_lab = torch.einsum("nij,nj->ni", UB[cid], hkl_f)
median_energy = core.beam_energy[iid].median().item()
wavelength = 12398.42 / median_energy
inv_wl = 1.0 / wavelength
s0 = torch.tensor([0.0, 0.0, inv_wl])
k_f = q_lab + s0
k_f_norm = k_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
k_f_hat = k_f / k_f_norm
two_theta = torch.acos(k_f_hat[:, 2].clamp(-1, 1))
phi = torch.atan2(k_f_hat[:, 1], k_f_hat[:, 0])
q_mag = q_lab.norm(dim=-1)

# Collect all features
features = {}

# Per-observation (from stream)
features["fs"] = core.fs[valid].numpy()
features["ss"] = core.ss[valid].numpy()
features["panel_id"] = core.panel_id[valid].float().numpy()
features["peak"] = core.peak[valid].numpy()
features["background"] = core.background[valid].numpy()
features["d_spacing"] = d_obs.numpy()
features["s_sq"] = (1.0 / (4.0 * d_obs ** 2)).numpy()

# Per-crystal (broadcast to obs)
cell = core.cell[cid]  # (N, 6)
features["cell_a"] = cell[:, 0].numpy()
features["cell_b"] = cell[:, 1].numpy()
features["cell_c"] = cell[:, 2].numpy()
features["cell_alpha"] = cell[:, 3].numpy()
features["cell_beta"] = cell[:, 4].numpy()
features["cell_gamma"] = cell[:, 5].numpy()
features["profile_rad"] = core.profile_rad[cid].numpy()
features["resolution"] = core.resolution[cid].numpy()

# Orientation (9 components)
astar_obs = core.astar[cid]
bstar_obs = core.bstar[cid]
cstar_obs = core.cstar[cid]
for i, ax in enumerate(["x", "y", "z"]):
    features[f"astar_{ax}"] = astar_obs[:, i].numpy()
    features[f"bstar_{ax}"] = bstar_obs[:, i].numpy()
    features[f"cstar_{ax}"] = cstar_obs[:, i].numpy()

# Per-image
features["beam_energy"] = core.beam_energy[iid].numpy()
features["camera_length"] = core.camera_length[iid].numpy()

# Derived geometry
features["two_theta"] = two_theta.numpy()
features["phi"] = phi.numpy()
features["q_mag"] = q_mag.numpy()
features["kf_x"] = k_f_hat[:, 0].numpy()
features["kf_y"] = k_f_hat[:, 1].numpy()
features["kf_z"] = k_f_hat[:, 2].numpy()

# Clean up: remove constant features, replace NaN/Inf
feature_names = []
feature_cols = []
for name, vals in features.items():
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    if vals.std() < 1e-10:
        print(f"  Skipping constant feature: {name}")
        continue
    # Standardize
    vals_std = (vals - vals.mean()) / vals.std()
    feature_names.append(name)
    feature_cols.append(vals_std)

X = np.column_stack(feature_cols)
n_features = len(feature_names)
print(f"  {n_features} features, {X.shape[0]} observations")

# ── Step 4: Univariate correlations ──────────────────────
print(f"\n── Univariate correlation with residual ──")
print(f"{'Feature':>20s} {'Pearson r':>10s} {'r²%':>8s}")
print("-" * 42)

correlations = []
for i, name in enumerate(feature_names):
    r = np.corrcoef(X[:, i], residual)[0, 1]
    correlations.append((name, r, r**2 * 100))

correlations.sort(key=lambda x: abs(x[1]), reverse=True)
for name, r, r2 in correlations:
    print(f"{name:>20s} {r:+10.6f} {r2:8.4f}")

# ── Step 5: Multiple linear regression ───────────────────
print(f"\n── Multiple linear regression (all features) ──")
# OLS: X @ β = residual
XtX = X.T @ X
Xty = X.T @ residual
beta = np.linalg.solve(XtX + 1e-6 * np.eye(n_features), Xty)  # tiny ridge
pred_linear = X @ beta
var_before = np.var(residual)
var_after = np.var(residual - pred_linear)
r2_linear = 1.0 - var_after / var_before
print(f"  R² = {r2_linear*100:.4f}% ({n_features} features)")
print(f"  Variance: {var_before:.4f} → {var_after:.4f}")

# Feature importance: |beta| (standardized features → comparable)
print(f"\n  Top features by |β|:")
order = np.argsort(np.abs(beta))[::-1]
for i in order[:20]:
    print(f"    {feature_names[i]:>20s}: β={beta[i]:+.6f}")

# ── Step 6: Incremental R² (greedy forward selection) ────
print(f"\n── Greedy forward feature selection ──")
print(f"{'Step':>4s} {'Feature':>20s} {'R²%':>10s} {'ΔR²%':>10s}")

remaining = set(range(n_features))
selected = []
prev_r2 = 0.0

for step in range(min(20, n_features)):
    best_r2 = -1
    best_feat = -1
    for f in remaining:
        cols = selected + [f]
        X_sub = X[:, cols]
        XtX_sub = X_sub.T @ X_sub + 1e-8 * np.eye(len(cols))
        beta_sub = np.linalg.solve(XtX_sub, X_sub.T @ residual)
        r2 = 1.0 - np.var(residual - X_sub @ beta_sub) / var_before
        if r2 > best_r2:
            best_r2 = r2
            best_feat = f

    selected.append(best_feat)
    remaining.discard(best_feat)
    delta = best_r2 - prev_r2
    print(f"{step+1:4d} {feature_names[best_feat]:>20s} {best_r2*100:10.4f} {delta*100:+10.4f}")
    prev_r2 = best_r2

    if delta < 0.001:  # <0.001% improvement — stop
        print(f"  (stopping: marginal gain < 0.001%)")
        break

# ── Step 7: Gradient boosted trees (nonlinear) ───────────
print(f"\n── Gradient boosted trees (nonlinear feature importance) ──")
from sklearn.ensemble import HistGradientBoostingRegressor

# Subsample for speed
np.random.seed(42)
n_sub = min(500_000, len(residual))
idx = np.random.choice(len(residual), n_sub, replace=False)
X_sub = X[idx]
y_sub = residual[idx]

# Train/test split
n_train = int(0.8 * n_sub)
X_train, X_test = X_sub[:n_train], X_sub[n_train:]
y_train, y_test = y_sub[:n_train], y_sub[n_train:]

gbm = HistGradientBoostingRegressor(
    max_iter=200, max_depth=4, learning_rate=0.1,
    min_samples_leaf=100, random_state=42,
)
gbm.fit(X_train, y_train)

r2_train = 1.0 - np.var(y_train - gbm.predict(X_train)) / np.var(y_train)
r2_test = 1.0 - np.var(y_test - gbm.predict(X_test)) / np.var(y_test)
print(f"  R² train: {r2_train*100:.4f}%")
print(f"  R² test:  {r2_test*100:.4f}%")

# Permutation importance (test set)
from sklearn.inspection import permutation_importance
perm = permutation_importance(gbm, X_test, y_test, n_repeats=5,
                              random_state=42)
perm_order = np.argsort(perm.importances_mean)[::-1]
print(f"\n  Top features (permutation importance on test set):")
for i in perm_order[:20]:
    imp = perm.importances_mean[i]
    std = perm.importances_std[i]
    print(f"    {feature_names[i]:>20s}: {imp:.6f} ± {std:.6f}")

# ── Step 8: Plots ─────────────────────────────────────────
OUTDIR.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) Univariate |r| bar chart
ax = axes[0, 0]
top_n = 20
top_corr = correlations[:top_n]
ax.barh(range(top_n), [abs(c[1]) for c in top_corr], color="steelblue")
ax.set_yticks(range(top_n))
ax.set_yticklabels([c[0] for c in top_corr], fontsize=8)
ax.set_xlabel("|Pearson r|")
ax.set_title("Univariate correlation with residual")
ax.invert_yaxis()

# (b) Forward selection curve
ax = axes[0, 1]
# Recompute for plotting
r2_curve = []
sel = []
rem = set(range(n_features))
for step in range(min(20, n_features)):
    best_r2 = -1
    best_f = -1
    for f in rem:
        cols = sel + [f]
        X_s = X[:, cols]
        b = np.linalg.solve(X_s.T @ X_s + 1e-8 * np.eye(len(cols)),
                            X_s.T @ residual)
        r2 = 1.0 - np.var(residual - X_s @ b) / var_before
        if r2 > best_r2:
            best_r2 = r2
            best_f = f
    sel.append(best_f)
    rem.discard(best_f)
    r2_curve.append(best_r2 * 100)
ax.plot(range(1, len(r2_curve) + 1), r2_curve, "o-", color="coral")
ax.set_xlabel("Number of features")
ax.set_ylabel("R² (%)")
ax.set_title("Greedy forward selection")
ax.grid(True, alpha=0.3)

# (c) GBM permutation importance
ax = axes[1, 0]
top_perm = perm_order[:15]
ax.barh(range(len(top_perm)),
        [perm.importances_mean[i] for i in top_perm], color="coral",
        xerr=[perm.importances_std[i] for i in top_perm])
ax.set_yticks(range(len(top_perm)))
ax.set_yticklabels([feature_names[i] for i in top_perm], fontsize=8)
ax.set_xlabel("Permutation importance (R² drop)")
ax.set_title("GBM feature importance (test set)")
ax.invert_yaxis()

# (d) Residual vs top feature scatter
ax = axes[1, 1]
top_feat = correlations[0][0]
top_idx = feature_names.index(top_feat)
subsample = np.random.choice(len(residual), min(50000, len(residual)), replace=False)
ax.scatter(X[subsample, top_idx], residual[subsample], s=0.3, alpha=0.1)
ax.set_xlabel(f"{top_feat} (standardized)")
ax.set_ylabel("Residual")
ax.set_title(f"Residual vs top feature: {top_feat}")
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = OUTDIR / "feature_importance.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved {out_path}")
