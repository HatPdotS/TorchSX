#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH -o /das/work/p17/p17490/Peter/torchsx/sfx_scan/sigma_resolution_depndence.out
#SBATCH -c 36
#SBATCH -p day
#SBATCH -t 2:00:00



"""
Study resolution dependence of noise parameters in SFX data.

Model: I_obs = I_true · exp(ε) + η
  ε ~ N(0, σ_LN²)    multiplicative / scale noise
  η ~ N(0, σ_gauss²)  additive / measurement noise

For HKL h with true intensity I_h:
  Var(I_obs | h) = I_h² · [exp(σ_LN²)(exp(σ_LN²)-1)] + σ_gauss²
                 ≈ I_h² · σ_LN²  +  σ_gauss²      (small σ_LN)

Regressing Var_h on I_h² within resolution shells separates the two.

Extended 3-component model adds Poisson shot noise:
  Var_h = α · I_h²  +  β · I_h  +  γ
  α → multiplicative (σ_LN²), β → Poisson gain, γ → readout noise²
"""

from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from torchsx.stream import SFXStream
from torchsx.merge._hkl import compute_hkl_index
from torchref.base.reciprocal.hkl import get_d_spacing

# ── Configuration ──────────────────────────────────────────────
STREAM_DIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/streams")
DELAYS = [-91, -136, -182, -227, -273]
SPACEGROUP = "P-1"
OUTDIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/resolution_study")
N_SHELLS = 20
MIN_MULT = 10        # minimum observations per HKL
N_WORKERS = 16

# ── Plotting helpers: x = 1/d², labels in d (Å) ─────────────
D_PLOT_MIN, D_PLOT_MAX = 0.9, 10.0  # resolution range for plots

def d_to_x(d):
    """Convert d-spacing (Å) to plot coordinate 1/d²."""
    return 1.0 / np.asarray(d) ** 2

def setup_res_axis(ax, xlabel=True):
    """Set x-axis to 1/d² with resolution tick labels in Å."""
    tick_d = np.array([10, 5, 3, 2, 1.5, 1.2, 1.0, 0.9])
    tick_d = tick_d[(tick_d >= D_PLOT_MIN) & (tick_d <= D_PLOT_MAX)]
    ax.set_xticks(d_to_x(tick_d))
    ax.set_xticklabels([f"{d:g}" for d in tick_d])
    ax.set_xlim(d_to_x(D_PLOT_MAX), d_to_x(D_PLOT_MIN))
    if xlabel:
        ax.set_xlabel("Resolution (Å)")

# ── 1. Load & concatenate streams ─────────────────────────────
streams = []
for d in DELAYS:
    stem = STREAM_DIR / f"delay_{d}.stream"
    cache = stem.with_suffix(".sfx")
    print(f"Loading {stem.name} ...")
    s = SFXStream.from_stream(stem, spacegroup=SPACEGROUP,
                              cache=cache, n_workers=N_WORKERS)
    print(f"  {s.n_crystals} crystals, {s.n_observations} obs")
    streams.append(s)

s = SFXStream.concat(streams)
print(f"\nTotal: {s.n_crystals} crystals, {s.n_observations} observations")

# ── 2. Background outlier rejection ───────────────────────────
# Model: bg(obs) = scale_crystal × bg_curve(d)
# Spots with bg >> expected sit on ice/powder rings or chip edges.

import math

hkl_idx = compute_hkl_index(s._core.hkl)
hkl_id = hkl_idx.hkl_id
unique_hkl = hkl_idx.unique_hkl
n_unique = hkl_idx.n_unique

I_obs_raw = s._core.I
I_bg_raw = s._core.background   # per-pixel mean background from CrystFEL
crystal_id = s._core.crystal_id

# d-spacing per observation (via median cell)
median_cell = s._core.cell.median(dim=0).values
d_per_obs = get_d_spacing(s._core.hkl.float(), median_cell)

# (a) Global background profile: median bg in resolution bins
N_BG_BINS = 50
BG_THRESHOLD = 3.0               # reject if bg > threshold × expected
d_rank = d_per_obs.argsort().argsort()
bg_bin = (d_rank.float() / len(d_per_obs) * N_BG_BINS).long().clamp(max=N_BG_BINS - 1)

bg_curve = torch.zeros(N_BG_BINS)          # median bg per resolution bin
bg_curve_d = torch.zeros(N_BG_BINS)        # bin center d
for b in range(N_BG_BINS):
    m = bg_bin == b
    if m.sum() > 0:
        bg_curve[b] = I_bg_raw[m].median()
        bg_curve_d[b] = d_per_obs[m].median()

# Expected bg for each observation (from its resolution bin)
bg_expected_per_obs = bg_curve[bg_bin]

# (b) Per-crystal scale: median(bg_obs / bg_expected) per crystal
n_crystals = s.n_crystals
ratios_all = I_bg_raw / bg_expected_per_obs.clamp(min=1e-6)
_cryst_order = crystal_id.long().argsort()
_cryst_counts = torch.bincount(crystal_id.long(), minlength=n_crystals)
_cryst_groups = ratios_all[_cryst_order].split(_cryst_counts.tolist())
crystal_scale = torch.tensor(
    [g.median().item() if len(g) > 0 else 1.0 for g in _cryst_groups])

# (c) Normalized background residual
bg_model = crystal_scale[crystal_id.long()] * bg_expected_per_obs
bg_ratio = I_bg_raw / bg_model.clamp(min=1e-6)

# (d) Reject outliers
bg_good = bg_ratio < BG_THRESHOLD
n_rejected = (~bg_good).sum().item()
print(f"\nBackground outlier rejection (threshold={BG_THRESHOLD}×):")
print(f"  Rejected {n_rejected} / {len(bg_good)} observations "
      f"({100 * n_rejected / len(bg_good):.2f}%)")

# Apply mask to all observation-level tensors
I_obs = I_obs_raw[bg_good]
I_bg = I_bg_raw[bg_good]
hkl_id = hkl_id[bg_good]
crystal_id_filt = crystal_id[bg_good]
d_per_obs_filt = d_per_obs[bg_good]

print(f"  Remaining: {len(I_obs)} observations")

# Background diagnostic plot
OUTDIR.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (a) bg vs resolution — all observations, color by accepted/rejected
ax = axes[0, 0]
n_scatter = min(20000, len(I_bg_raw))
idx_sc = torch.randperm(len(I_bg_raw))[:n_scatter]
ax.scatter(d_to_x(d_per_obs[idx_sc].numpy()), I_bg_raw[idx_sc].numpy(),
           c=bg_good[idx_sc].numpy().astype(float), cmap="RdYlGn",
           s=1, alpha=0.2, vmin=0, vmax=1)
ax.plot(d_to_x(bg_curve_d.numpy()), bg_curve.numpy(), "k-", lw=2, label="median bg(d)")
ax.set_ylabel("Background (counts/px)")
ax.set_title("Background vs resolution (green=kept, red=rejected)")
ax.legend(fontsize=8)
setup_res_axis(ax)

# (b) Distribution of bg_ratio
ax = axes[0, 1]
ax.hist(bg_ratio.clamp(max=10).numpy(), bins=100, color="C0", edgecolor="none")
ax.axvline(BG_THRESHOLD, color="r", ls="--", label=f"threshold={BG_THRESHOLD}")
ax.set_xlabel("bg / expected")
ax.set_ylabel("Count")
ax.set_title("Background ratio distribution")
ax.set_yscale("log")
ax.legend(fontsize=8)

# (c) Fraction rejected vs resolution
n_bg_diag_bins = 20
d_rank_diag = d_per_obs.argsort().argsort()
diag_bin = (d_rank_diag.float() / len(d_per_obs) * n_bg_diag_bins).long().clamp(max=n_bg_diag_bins - 1)
frac_rej = torch.zeros(n_bg_diag_bins)
d_diag = torch.zeros(n_bg_diag_bins)
for b in range(n_bg_diag_bins):
    m = diag_bin == b
    if m.sum() > 0:
        frac_rej[b] = (~bg_good[m]).float().mean()
        d_diag[b] = d_per_obs[m].median()
ax = axes[1, 0]
ax.plot(d_to_x(d_diag.numpy()), frac_rej.numpy(), "o-", color="C3")
ax.set_ylabel("Fraction rejected")
ax.set_title("Rejection rate vs resolution")
setup_res_axis(ax)

# (d) Per-crystal background scale distribution
ax = axes[1, 1]
ax.hist(crystal_scale.numpy(), bins=100, color="C5", edgecolor="none")
ax.axvline(crystal_scale.median().item(), color="k", ls="--",
           label=f"median={crystal_scale.median():.2f}")
ax.set_xlabel("Crystal bg scale")
ax.set_ylabel("Count")
ax.set_title("Per-crystal background scale factor")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(str(OUTDIR / "background_filter.png"), dpi=150)
plt.close(fig)
print(f"Saved {OUTDIR / 'background_filter.png'}")

# ── Integration geometry (CrystFEL ring integration 7,9,11) ──
R_PEAK, R_GAP, R_BG = 7, 9, 11
n_peak = math.pi * R_PEAK ** 2             # ≈154 pixels
n_bg_px = math.pi * (R_BG ** 2 - R_GAP ** 2)  # ≈126 pixels

# Predicted Poisson variance per observation (background-subtracted):
#   σ² = (I_net + bg·n_peak) + (n_peak/n_bg)²·(bg·n_bg)
#      = I_net + bg·n_peak·(1 + n_peak/n_bg)
# A gain factor g absorbs ADU→photon conversion (fit below).
bg_contrib = I_bg * n_peak * (1.0 + n_peak / n_bg_px)
sigma_poisson_raw = (I_obs + bg_contrib).clamp(min=0).sqrt()  # √(total counts), gain=1

print(f"\nIntegration geometry: peak r={R_PEAK} ({n_peak:.0f} px), "
      f"bg annulus r={R_GAP}-{R_BG} ({n_bg_px:.0f} px)")
print(f"Background (after filter): median={I_bg.median():.2f}, "
      f"mean={I_bg.mean():.2f} counts/px")

# ── 3. Per-HKL statistics (on filtered observations) ─────────
# group observations by HKL
counts = torch.bincount(hkl_id, minlength=n_unique)
order = hkl_id.argsort()
groups_I = I_obs[order].split(counts.tolist())
# Per-obs σ_gauss² from background (Poisson model, gain=1 for now)
sigma2_per_obs = (I_obs + bg_contrib).clamp(min=0)   # σ²_poisson per obs
groups_sig2 = sigma2_per_obs[order].split(counts.tolist())

mean_I = torch.zeros(n_unique)
var_I = torch.zeros(n_unique)
var_signal = torch.zeros(n_unique)     # Var(I) with per-obs σ²_gauss subtracted
mult = torch.zeros(n_unique, dtype=torch.long)
frac_neg = torch.zeros(n_unique)
mean_sigG2 = torch.zeros(n_unique)     # <σ_gauss²> per HKL (from background)

for h, (gI, gS) in enumerate(zip(groups_I, groups_sig2)):
    n = len(gI)
    if n == 0:
        continue
    mult[h] = n
    mu = gI.mean()
    mean_I[h] = mu
    if n > 1:
        var_I[h] = gI.var(correction=1)
        # Per-obs noise subtraction: mean[(I_j - Ī)² - σ²_gauss,j]
        var_signal[h] = ((gI - mu) ** 2 - gS).mean().clamp(min=0)
    frac_neg[h] = (gI <= 0).float().mean()
    mean_sigG2[h] = gS.mean()

# ── 4. Resolution per unique HKL ──────────────────────────────
d_spacing = get_d_spacing(unique_hkl.float(), median_cell)   # Å

# ── 5. Filter ─────────────────────────────────────────────────
keep = (mult >= MIN_MULT) & (mean_I > 0)
d_f = d_spacing[keep]
mean_f = mean_I[keep]
var_f = var_I[keep]
var_sig_f = var_signal[keep]           # per-obs noise-subtracted signal variance
mult_f = mult[keep]
frac_neg_f = frac_neg[keep]
sigG2_f = mean_sigG2[keep]             # per-HKL <σ_gauss²> from background

print(f"{keep.sum().item()} / {n_unique} HKLs pass filter "
      f"(mult ≥ {MIN_MULT}, <I> > 0)")

# ── 6. Resolution shells (all unique HKLs) ───────────────────
rank_all = d_spacing.argsort().argsort()
hkl_shell_all = (rank_all.float() / n_unique * N_SHELLS).long().clamp(max=N_SHELLS - 1)

# Map each observation to its HKL's shell
obs_shell = hkl_shell_all[hkl_id]          # (N_obs,)
# Shell for filtered HKLs
hkl_shell_f = hkl_shell_all[keep]

# ── 7. Estimate σ_gauss from negative observations ───────────
# I_obs = I_true·exp(ε) + η, and I_true·exp(ε) > 0 always.
# Negative observations arise when η overwhelms signal.
# For weak reflections I_obs ≈ η, so RMS(I_neg) ≈ σ_gauss.
# (Slightly underestimates because |I_neg| = |η| - signal < |η|,
#  but for the many weak reflections it's a good direct estimator.)

shell_d = torch.zeros(N_SHELLS)
shell_meanI = torch.zeros(N_SHELLS)
shell_frac_neg = torch.zeros(N_SHELLS)
shell_n_hkl = torch.zeros(N_SHELLS, dtype=torch.long)
shell_n_neg = torch.zeros(N_SHELLS, dtype=torch.long)
shell_n_obs = torch.zeros(N_SHELLS, dtype=torch.long)
shell_sigG = torch.zeros(N_SHELLS)
shell_sigLN = torch.zeros(N_SHELLS)
shell_sigG_poisson = torch.zeros(N_SHELLS)   # predicted from background
shell_mean_bg = torch.zeros(N_SHELLS)        # mean bg per pixel

# Negative-tail outlier tracking
NEG_PCTILES = [50, 90, 95, 99]           # percentiles of |I_neg|
SIGMA_THRESHOLDS = [3, 5, 10]            # count obs beyond N·σ_gauss
shell_neg_pctiles = torch.zeros(N_SHELLS, len(NEG_PCTILES))   # |I_neg| percentiles
shell_neg_beyond = torch.zeros(N_SHELLS, len(SIGMA_THRESHOLDS), dtype=torch.long)
shell_neg_min = torch.zeros(N_SHELLS)    # most negative observation

for sid in range(N_SHELLS):
    # --- σ_gauss from ALL negative observations in this shell ---
    obs_mask = obs_shell == sid
    I_shell = I_obs[obs_mask]
    shell_n_obs[sid] = obs_mask.sum()
    neg = I_shell[I_shell <= 0]
    shell_n_neg[sid] = len(neg)

    if len(neg) > 10:
        shell_sigG[sid] = (neg ** 2).mean().sqrt()

        # Negative-tail statistics
        abs_neg = neg.abs()
        for i, p in enumerate(NEG_PCTILES):
            shell_neg_pctiles[sid, i] = torch.quantile(abs_neg.float(), p / 100.0)
        shell_neg_min[sid] = neg.min()  # most negative value

    # Predicted σ from background (Poisson model, gain=1)
    sig_poi_shell = sigma_poisson_raw[obs_mask]
    if len(sig_poi_shell) > 0:
        shell_sigG_poisson[sid] = sig_poi_shell.mean()
        shell_mean_bg[sid] = I_bg[obs_mask].mean()

    # Count observations beyond N·σ_gauss (all obs, not just negatives)
    if shell_sigG[sid] > 0:
        for j, thr in enumerate(SIGMA_THRESHOLDS):
            shell_neg_beyond[sid, j] = (I_shell < -thr * shell_sigG[sid]).sum()

    # --- Per-shell HKL stats (filtered) ---
    hm = hkl_shell_f == sid
    if hm.sum() < 5:
        continue

    Ih = mean_f[hm]
    Vs = var_sig_f[hm]                 # per-obs noise-subtracted signal variance

    shell_d[sid] = d_f[hm].mean()
    shell_meanI[sid] = Ih.mean()
    shell_frac_neg[sid] = frac_neg_f[hm].mean()
    shell_n_hkl[sid] = hm.sum()

    # --- σ_LN from per-obs noise-subtracted signal variance ---
    # var_signal_h = mean[(I_j - Ī)² - σ²_gauss,j]  ≈ Var(signal)
    # CV_mult² = var_signal / I_h²
    # σ_LN = √(log(1 + CV_mult²))
    cv2_mult = Vs / Ih ** 2
    sigLN2_per_hkl = torch.log1p(cv2_mult)
    shell_sigLN[sid] = sigLN2_per_hkl.median().clamp(min=0).sqrt()

# numpy views for plotting
ok = shell_d > 0
d_np = shell_d[ok].numpy()
mI_np = shell_meanI[ok].numpy()
sLN = shell_sigLN[ok].numpy()
sG = shell_sigG[ok].numpy()
sG_poi = shell_sigG_poisson[ok].numpy()
bg_np = shell_mean_bg[ok].numpy()

# shell x-coordinates
x_shell = d_to_x(d_np)

# Fit gain: σ_gauss(empirical) = g · σ_poisson(predicted)
gain_fit = np.sum(sG * sG_poi) / np.sum(sG_poi ** 2)
print(f"\nGain factor g (σ_gauss = g · σ_poisson): {gain_fit:.4f}")
print(f"  → g² = {gain_fit**2:.4f}  (ADU²/photon)")

# ── Per-HKL σ_LN and I0 correction ──────────────────────────
# Per-HKL σ_LN using per-observation noise-subtracted signal variance
# var_signal already has per-obs σ²_gauss subtracted (gain=1).
# Scale by g² if gain ≠ 1: var_signal_corrected = var_obs - g²·σ²_poisson
# Since var_signal = mean[(I-Ī)² - σ²_poi], and the true noise is g²·σ²_poi:
#   var_signal_corrected = mean[(I-Ī)²] - g²·mean[σ²_poi]
#                        = var_obs - g²·<σ²_poi>
#                        = var_signal + (1 - g²)·<σ²_poi>
var_sig_corrected = (var_sig_f + (1.0 - gain_fit ** 2) * sigG2_f).clamp(min=0)
cv2_mult_f = var_sig_corrected / mean_f ** 2
sigLN2_f = torch.log1p(cv2_mult_f)          # per-HKL σ_LN²
sigLN_f = sigLN2_f.clamp(min=0).sqrt()      # per-HKL σ_LN

# I0 = <I_obs> · exp(-σ_LN²/2)  (bias correction)
I0_f = mean_f * torch.exp(-sigLN2_f / 2)

# Correction factor by shell
bias_correction = torch.zeros(N_SHELLS)
for sid in range(N_SHELLS):
    hm = hkl_shell_f == sid
    if hm.sum() > 0:
        bias_correction[sid] = torch.exp(-sigLN2_f[hm].median() / 2)

print(f"\nPer-HKL σ_LN: median={sigLN_f.median():.3f}, "
      f"mean={sigLN_f.mean():.3f}, p90={sigLN_f.quantile(0.9):.3f}")
print(f"I0 bias correction: median factor={torch.exp(-sigLN2_f.median()/2):.4f}"
      f"  (I0/<I> at median σ_LN)")

# ── 7. Summary table ──────────────────────────────────────────
hdr = (f"{'shell':>5} {'d(Å)':>7} {'<I>':>10} {'<bg/px>':>8}"
       f" {'σ_G(neg)':>9} {'σ_G(poi)':>9} {'ratio':>6} {'σ_LN':>7}"
       f"  {'f_neg':>6} {'n_hkl':>6}")
print(f"\n{hdr}\n{'-' * len(hdr)}")
for sid in range(N_SHELLS):
    if shell_d[sid] <= 0:
        continue
    ratio = shell_sigG[sid] / shell_sigG_poisson[sid] if shell_sigG_poisson[sid] > 0 else 0
    print(f"{sid:>5d} {shell_d[sid]:>7.2f} {shell_meanI[sid]:>10.1f}"
          f" {shell_mean_bg[sid]:>8.2f}"
          f" {shell_sigG[sid]:>9.2f} {shell_sigG_poisson[sid]:>9.2f}"
          f" {ratio:>6.3f} {shell_sigLN[sid]:>7.4f}"
          f"  {shell_frac_neg[sid]:>6.3f} {shell_n_hkl[sid]:>6d}")

# ── 7b. Negative outlier table ─────────────────────────────────
pct_hdr = "  ".join(f"p{p:02d}" for p in NEG_PCTILES)
thr_hdr = "  ".join(f">{t}σ" for t in SIGMA_THRESHOLDS)
hdr2 = (f"{'shell':>5} {'d(Å)':>7} {'σ_G':>7} {'min':>9}"
        f"  {pct_hdr}  {thr_hdr}")
print(f"\nNegative-tail diagnostics (|I_neg| percentiles & counts beyond Nσ)")
print(f"{hdr2}\n{'-' * len(hdr2)}")
for sid in range(N_SHELLS):
    if shell_d[sid] <= 0:
        continue
    pct_vals = "  ".join(f"{shell_neg_pctiles[sid, i]:>4.1f}" for i in range(len(NEG_PCTILES)))
    thr_vals = "  ".join(f"{shell_neg_beyond[sid, j]:>4d}" for j in range(len(SIGMA_THRESHOLDS)))
    print(f"{sid:>5d} {shell_d[sid]:>7.2f} {shell_sigG[sid]:>7.2f} {shell_neg_min[sid]:>9.1f}"
          f"  {pct_vals}  {thr_vals}")

# ── 8. Parametric fits ────────────────────────────────────────


def model_LN_inv_d2(d, a, b):
    """σ_LN(d) = a + b / d²"""
    return a + b / d ** 2


def model_gauss_const(d, c):
    """σ_gauss(d) = c   (resolution-independent)"""
    return np.full_like(d, c)


def model_gauss_linear_d2(d, a, b):
    """σ_gauss(d) = a + b / d²"""
    return a + b / d ** 2


fit_labels = {}
try:
    popt_LN, _ = curve_fit(model_LN_inv_d2, d_np, sLN, p0=[0.3, 0.1])
    fit_labels["σ_LN"] = f"a + b/d²  (a={popt_LN[0]:.4f}, b={popt_LN[1]:.4f})"
    print(f"\nσ_LN(d) = {popt_LN[0]:.4f} + {popt_LN[1]:.4f} / d²")
except Exception as e:
    popt_LN = None
    print(f"\nσ_LN fit failed: {e}")

try:
    popt_G_const, _ = curve_fit(model_gauss_const, d_np, sG, p0=[10.0])
    print(f"σ_gauss (constant) = {popt_G_const[0]:.2f}")
except Exception:
    popt_G_const = None

try:
    popt_G_d2, _ = curve_fit(model_gauss_linear_d2, d_np, sG, p0=[5.0, 1.0])
    fit_labels["σ_gauss"] = (f"a + b/d²  (a={popt_G_d2[0]:.2f}, b={popt_G_d2[1]:.2f})")
    print(f"σ_gauss(d) = {popt_G_d2[0]:.2f} + {popt_G_d2[1]:.2f} / d²")
except Exception as e:
    popt_G_d2 = None
    print(f"σ_gauss fit failed: {e}")

# ── 9. Figure 1 — overview ────────────────────────────────────
OUTDIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# (a) σ_LN vs resolution
ax = axes[0, 0]
ax.plot(x_shell, sLN, "o-", color="C0")
if popt_LN is not None:
    d_fine = np.linspace(D_PLOT_MIN, D_PLOT_MAX, 200)
    ax.plot(d_to_x(d_fine), model_LN_inv_d2(d_fine, *popt_LN), "--", color="gray",
            label=fit_labels["σ_LN"])
    ax.legend(fontsize=8)
ax.set_ylabel("σ_LN")
ax.set_title("Lognormal width vs resolution")
setup_res_axis(ax)

# (b) σ_gauss: empirical (negatives) vs predicted (background)
ax = axes[0, 1]
ax.plot(x_shell, sG, "s-", color="C1", label="from negatives")
ax.plot(x_shell, gain_fit * sG_poi, "o--", color="C5",
        label=f"g·σ_poisson (g={gain_fit:.3f})")
ax.plot(x_shell, sG_poi, "x:", color="gray", alpha=0.5, label="σ_poisson (g=1)")
ax.legend(fontsize=8)
ax.set_ylabel("σ_gauss")
ax.set_title("Empirical vs Poisson prediction")
setup_res_axis(ax)

# (c) σ_gauss(neg) vs σ_poisson — direct scatter (no res axis)
ax = axes[0, 2]
ax.plot(sG_poi, sG, "o", color="C1")
lo, hi = 0, max(sG.max(), sG_poi.max()) * 1.1
ax.plot([lo, hi], [gain_fit * lo, gain_fit * hi], "--", color="gray",
        label=f"g={gain_fit:.3f}  (g²={gain_fit**2:.3f})")
ax.plot([lo, hi], [lo, hi], ":", color="k", alpha=0.3, label="g=1")
ax.set_xlabel("σ_poisson (from background)")
ax.set_ylabel("σ_gauss (from negatives)")
ax.set_title("Empirical vs predicted noise")
ax.legend(fontsize=8)
ax.set_aspect("equal")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

# (d) Wilson-like: log<I> vs resolution + per-HKL scatter
ax = axes[1, 0]
n_plot = min(8000, len(mean_f))
idx_w = torch.randperm(len(mean_f))[:n_plot]
pos_mask = mean_f[idx_w] > 0
ax.scatter(d_to_x(d_f[idx_w[pos_mask]].numpy()),
           mean_f[idx_w[pos_mask]].numpy(),
           s=1, alpha=0.1, color="C7")
ax.semilogy(x_shell, mI_np, "o-", color="C2", zorder=5)
ax.set_ylabel("<I>")
ax.set_title("Mean intensity vs resolution")
setup_res_axis(ax)

# (e) Fraction negative vs resolution
ax = axes[1, 1]
ax.plot(x_shell, shell_frac_neg[ok].numpy(), "o-", color="C3")
ax.set_ylabel("Fraction I ≤ 0")
ax.set_title("Negative observations vs resolution")
setup_res_axis(ax)

# (f) Per-HKL CV scatter
ax = axes[1, 2]
cv = (var_f ** 0.5) / mean_f
n_plot = min(5000, len(cv))
idx = torch.randperm(len(cv))[:n_plot]
ax.scatter(d_to_x(d_f[idx].numpy()), cv[idx].numpy(), s=1, alpha=0.3, color="C4")
ax.set_ylabel("CV = std / mean")
ax.set_title("Coefficient of variation vs resolution")
ax.set_ylim(0, min(cv.quantile(0.99).item() * 1.5, 10))
setup_res_axis(ax)

fig.tight_layout()
fig.savefig(str(OUTDIR / "noise_vs_resolution.png"), dpi=150)
plt.close(fig)
print(f"\nSaved {OUTDIR / 'noise_vs_resolution.png'}")

# ── 10. Figure 2 — σ_gauss / σ_LN ratio & diagnostic ────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# (a) σ_gauss / <I>  — relative noise floor
ax = axes[0]
ax.plot(x_shell, sG / mI_np, "o-", color="C1")
ax.set_ylabel("σ_gauss / <I>")
ax.set_title("Relative Gaussian noise")
setup_res_axis(ax)

# (b) predicted vs observed frac_neg
from scipy.stats import norm
pred_frac_neg = norm.cdf(-mI_np / sG)
ax = axes[1]
ax.plot(x_shell, shell_frac_neg[ok].numpy(), "o-", color="C3", label="observed")
ax.plot(x_shell, pred_frac_neg, "s--", color="gray", label="predicted Φ(-<I>/σ_G)")
ax.set_ylabel("Fraction I ≤ 0")
ax.set_title("Negative fraction: observed vs predicted")
ax.legend(fontsize=8)
setup_res_axis(ax)

# (c) variance decomposition: stacked contributions
ax = axes[2]
sig_LN_contrib = (sLN ** 2) * (mI_np ** 2)
sig_G_contrib = sG ** 2
total_var = sig_LN_contrib + sig_G_contrib
ax.bar(x_shell, sig_G_contrib / total_var, width=np.diff(np.append(x_shell, x_shell[-1]*1.1)) * 0.8,
       color="C1", label="σ_gauss²")
ax.bar(x_shell, sig_LN_contrib / total_var, width=np.diff(np.append(x_shell, x_shell[-1]*1.1)) * 0.8,
       bottom=sig_G_contrib / total_var, color="C0", label="σ_LN²·I²")
ax.set_ylabel("Fraction of total variance")
ax.set_title("Variance decomposition by shell")
ax.legend(fontsize=8)
setup_res_axis(ax)

fig.tight_layout()
fig.savefig(str(OUTDIR / "diagnostics.png"), dpi=150)
plt.close(fig)
print(f"Saved {OUTDIR / 'diagnostics.png'}")

# ── 11. Figure 3 — representative per-HKL histograms ─────────
# Show 4 shells: overlay data histogram with the fitted model
show_shells = np.linspace(0, N_SHELLS - 1, 4, dtype=int)
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

for ax, sid in zip(axes, show_shells):
    hm = hkl_shell_f == sid
    if hm.sum() < 5:
        continue
    Ih = mean_f[hm]
    Vh = var_f[hm]

    # scatter: Var vs I² with σ_gauss line
    ax.scatter(Ih.numpy() ** 2, Vh.numpy(), s=6, alpha=0.4)
    xmax = np.percentile(Ih.numpy() ** 2, 98)
    xfit = np.linspace(0, xmax, 100)
    sigLN2 = shell_sigLN[sid].item() ** 2
    sigG2 = shell_sigG[sid].item() ** 2
    ax.plot(xfit, sigLN2 * xfit + sigG2, "r-",
            label=f"σ_LN={shell_sigLN[sid]:.3f}\nσ_G={shell_sigG[sid]:.1f}")
    ax.axhline(sigG2, ls=":", color="C1", alpha=0.5, label="σ_gauss² floor")
    ax.set_xlim(0, xmax * 1.05)
    ax.set_xlabel("I²")
    ax.set_ylabel("Var(I_obs)")
    ax.set_title(f"Shell {sid}  d={shell_d[sid]:.2f} Å")
    ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig(str(OUTDIR / "shell_regressions.png"), dpi=150)
plt.close(fig)
print(f"Saved {OUTDIR / 'shell_regressions.png'}")

# ── 12. Figure 4 — negative outlier diagnostics ──────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (a) |I_neg| percentiles vs resolution, normalized by σ_gauss
ax = axes[0, 0]
for i, p in enumerate(NEG_PCTILES):
    vals = shell_neg_pctiles[ok, i].numpy() / sG
    ax.plot(x_shell, vals, "o-", label=f"p{p}", markersize=4)
ax.axhline(1.0, ls=":", color="k", alpha=0.3, label="1σ")
ax.axhline(3.0, ls=":", color="r", alpha=0.3, label="3σ")
ax.set_ylabel("|I_neg| / σ_gauss")
ax.set_title("Negative-tail percentiles (normalized)")
ax.legend(fontsize=7)
setup_res_axis(ax)

# (b) Fraction of ALL observations beyond Nσ thresholds
ax = axes[0, 1]
for j, thr in enumerate(SIGMA_THRESHOLDS):
    frac = shell_neg_beyond[ok, j].float().numpy() / shell_n_obs[ok].float().numpy()
    ax.plot(x_shell, frac, "o-", label=f"I < -{thr}σ_G", markersize=4)
    expected = norm.cdf(-thr)
    ax.axhline(expected, ls="--", color=f"C{j}", alpha=0.3)
ax.set_ylabel("Fraction of all obs")
ax.set_title("Outlier fractions vs Gaussian expectation (dashed)")
ax.set_yscale("log")
ax.legend(fontsize=7)
setup_res_axis(ax)

# (c) Most negative observation / σ_gauss vs resolution
ax = axes[1, 0]
worst = shell_neg_min[ok].numpy() / sG
ax.plot(x_shell, worst, "o-", color="C3")
ax.set_ylabel("min(I_obs) / σ_gauss")
ax.set_title("Most extreme negative (in σ units)")
setup_res_axis(ax)

# (d) Negative-tail histograms for 4 representative shells
ax = axes[1, 1]
show_shells_neg = np.linspace(0, N_SHELLS - 1, 4, dtype=int)
for sid in show_shells_neg:
    obs_mask = obs_shell == sid
    I_shell = I_obs[obs_mask]
    neg = I_shell[I_shell <= 0]
    if len(neg) < 20:
        continue
    # normalize by σ_gauss
    neg_norm = (neg / shell_sigG[sid]).numpy()
    ax.hist(neg_norm, bins=50, alpha=0.4, density=True,
            label=f"d={shell_d[sid]:.1f}Å")
# overlay expected half-Gaussian
x = np.linspace(-8, 0, 200)
ax.plot(x, 2 * norm.pdf(x), "k--", label="N(0,1) left tail")
ax.set_xlabel("I_neg / σ_gauss")
ax.set_ylabel("Density")
ax.set_title("Negative tail shape (normalized by σ_G)")
ax.legend(fontsize=7)
ax.set_xlim(-8, 0.5)

fig.tight_layout()
fig.savefig(str(OUTDIR / "negative_outliers.png"), dpi=150)
plt.close(fig)
print(f"Saved {OUTDIR / 'negative_outliers.png'}")

# ── 13. Figure 5 — per-HKL σ_LN and I0 correction ───────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (a) Per-HKL σ_LN scatter vs resolution
ax = axes[0, 0]
n_plot = min(5000, len(sigLN_f))
idx = torch.randperm(len(sigLN_f))[:n_plot]
ax.scatter(d_to_x(d_f[idx].numpy()), sigLN_f[idx].numpy(), s=1, alpha=0.3, color="C0")
ax.plot(x_shell, sLN, "o-", color="red", markersize=5, lw=2, label="shell median")
ax.set_ylabel("σ_LN (per HKL)")
ax.set_title("Lognormal width vs resolution (bg-corrected)")
ax.set_ylim(0, min(sigLN_f.quantile(0.99).item() * 1.2, 10))
ax.legend(fontsize=8)
setup_res_axis(ax)

# (b) I0 bias correction factor vs resolution
ax = axes[0, 1]
bias_np = bias_correction[ok].numpy()
ax.plot(x_shell, bias_np, "o-", color="C2")
ax.axhline(1.0, ls=":", color="k", alpha=0.3)
ax.set_ylabel("I0 / <I_obs>  =  exp(-σ_LN²/2)")
ax.set_title("Mean-intensity bias correction")
setup_res_axis(ax)

# (c) Wilson plot: per-HKL scatter + shell means + corrected I0
ax = axes[1, 0]
n_plot = min(8000, len(mean_f))
idx = torch.randperm(len(mean_f))[:n_plot]
ax.scatter(d_to_x(d_f[idx].numpy()), mean_f[idx].numpy(),
           s=1, alpha=0.1, color="C7", label="_nolegend_")
I0_shell = mI_np * bias_np
ax.semilogy(x_shell, mI_np, "o-", color="C2", label="<I_obs>  (biased)")
ax.semilogy(x_shell, I0_shell, "s-", color="C0", label="I0  (corrected)")
ax.set_ylabel("Intensity")
ax.set_title("Wilson plot: raw mean vs corrected I0")
ax.legend(fontsize=8)
setup_res_axis(ax)

# (d) Per-HKL scatter: I0 vs <I_obs>, colored by σ_LN
ax = axes[1, 1]
n_plot = min(5000, len(I0_f))
idx = torch.randperm(len(I0_f))[:n_plot]
sc = ax.scatter(mean_f[idx].numpy(), I0_f[idx].numpy(),
                c=sigLN_f[idx].numpy(), s=2, alpha=0.4,
                cmap="viridis", vmin=0, vmax=sigLN_f.quantile(0.95).item())
ax.plot([0, mean_f.max().item()], [0, mean_f.max().item()],
        "k:", alpha=0.3, label="I0 = <I>")
ax.set_xlabel("<I_obs>")
ax.set_ylabel("I0 = <I>·exp(-σ_LN²/2)")
ax.set_title("Bias correction per HKL")
ax.legend(fontsize=8)
plt.colorbar(sc, ax=ax, label="σ_LN")

fig.tight_layout()
fig.savefig(str(OUTDIR / "sigma_LN_and_I0.png"), dpi=150)
plt.close(fig)
print(f"Saved {OUTDIR / 'sigma_LN_and_I0.png'}")

# ══════════════════════════════════════════════════════════════
# 14. Physical model for σ_LN resolution dependence
# ══════════════════════════════════════════════════════════════
#
# Known beam (SwissFEL):
#   E = 12.96 keV → λ = 0.9569 Å, Δλ/λ = 0.4%, α = 0.7 mrad
# Crystals: ~5×5×5 µm, P-1
# CrystFEL: r_p = 0.04 nm⁻¹ = 0.004 Å⁻¹
#
# ── Ewald sphere width (KNOWN) ──
#   σ_bw(q)  = (Δλ/λ)·λ·q²/2         bandwidth (exact: Δξ = Δλ/(2d²))
#   σ_div(q) = α·q                    divergence
#   σ_ewald(q) = √(σ_bw² + σ_div²)
#
# ── Relp width: 2nd-order polynomial in q = 1/d (Williamson-Hall) ──
#   σ_relp(q) = σ_0 + σ_1·q + σ_2·q²
#     σ_0  = size/mosaicity (constant)
#     σ_1  = strain coefficient (Δd/d broadening, 1st order)
#     σ_2  = disorder coefficient (lattice defects, 2nd order)
#
# Crystal-to-crystal variation: σ_1 varies (strain distribution),
# σ_0 and σ_2 are common.
#
# ── Ewald sphere slice of a 3D Gaussian relp ──
# The Ewald sphere (2D surface) slices through the 3D relp at
# excitation error ξ.  Integrating the 2D transverse directions:
#   I_obs ∝ (1/σ_relp) · exp(-ξ²/(2 σ_total²))
#
# The 1/σ_relp prefactor: as the relp broadens, the PEAK HEIGHT
# drops (total scattering power is conserved in reciprocal space).
# This is critical: broader relps ≠ higher signal.
#
# Therefore:  ln(p) = -ln(σ_total) - ξ²/(2 σ_total²) + const
#
# Var[ln(p)] over (ξ ~ U(-r_p,r_p), ε_c ~ strain dist) by MC.

# ── Known parameters ──
ENERGY_KEV = 12.96
WAVELENGTH = 12398.4 / (ENERGY_KEV * 1000)   # Å
BANDWIDTH = 0.004                              # Δλ/λ
DIVERGENCE = 0.7e-3                            # rad
CRYSTAL_SIZE = 50000.0                         # Å (5 µm)
R_PROFILE = 0.004                              # Å⁻¹ (CrystFEL)

print(f"\n── Beam parameters ──")
print(f"  λ = {WAVELENGTH:.4f} Å,  Δλ/λ = {BANDWIDTH},  "
      f"α = {DIVERGENCE*1e3:.1f} mrad")
print(f"  Crystal size L = {CRYSTAL_SIZE/1e4:.0f} µm,  "
      f"1/L = {1/CRYSTAL_SIZE:.2e} Å⁻¹")
print(f"  r_p = {R_PROFILE} Å⁻¹")

profile_rad = s._core.profile_rad
print(f"  Stream profile_rad: median={profile_rad.median():.6f}, "
      f"range=[{profile_rad.min():.6f}, {profile_rad.max():.6f}]")

# ── Ewald sphere width ──
def sigma_ewald_q(q):
    """Effective Ewald sphere width at |q| (Å⁻¹)."""
    q = np.asarray(q, dtype=np.float64)
    s_bw = BANDWIDTH * WAVELENGTH * q ** 2 / 2.0
    s_div = DIVERGENCE * q
    return np.sqrt(s_bw ** 2 + s_div ** 2)

# ── Pre-generate MC samples (fixed seed) ──
# Reduced counts for fast curve_fit; plenty for smooth Var estimates.
MC_RNG = np.random.default_rng(42)
MC_N_CRYST = 2_000
MC_N_XI = 100
mc_z = MC_RNG.standard_normal(MC_N_CRYST)
mc_xi2 = MC_RNG.uniform(-R_PROFILE, R_PROFILE, MC_N_XI) ** 2  # pre-square

# Pre-compute ξ² moments for analytical fallback
_xi2_mean = R_PROFILE ** 2 / 3.0      # E[ξ²] for U(-r,r)
_xi4_mean = R_PROFILE ** 4 / 5.0      # E[ξ⁴]
_xi2_var = _xi4_mean - _xi2_mean ** 2  # Var[ξ²]

def var_lnp_batch(q_arr, sig0, sig1_mean, sig1_std, sig2):
    """Vectorized Var[ln(p)] for an array of q values at once.

    Uses analytical decomposition where possible, MC for strain variation.

    ln(p) = -ln(σ_total) - ξ²/(2σ_total²)
    Var[ln(p)] = E_ε[Var_ξ[ln(p)|ε]] + Var_ε[E_ξ[ln(p)|ε]]
               = E_ε[ξ²_var / (4σ_t⁴)] + Var_ε[-ln(σ_t) - ξ²_mean/(2σ_t²)]
    (Uses law of total variance over strain distribution.)
    """
    q_arr = np.asarray(q_arr, dtype=np.float64)
    N_q = len(q_arr)

    # Strain samples: (MC_N_CRYST,)
    if sig1_std > 1e-12:
        eps_samples = np.abs(sig1_mean + sig1_std * mc_z)
    else:
        eps_samples = np.full(MC_N_CRYST, max(sig1_mean, 0.0))

    # Ewald sphere: (N_q,)
    s_ew = sigma_ewald_q(q_arr)

    # Per-crystal, per-q relp width: (MC_N_CRYST, N_q)
    s_relp = (sig0
              + eps_samples[:, None] * q_arr[None, :]
              + sig2 * q_arr[None, :] ** 2)
    s_relp = np.maximum(s_relp, 1e-8)
    s_total = np.sqrt(s_relp ** 2 + s_ew[None, :] ** 2)  # (MC_N_CRYST, N_q)
    inv_st2 = 1.0 / (s_total ** 2)                        # (MC_N_CRYST, N_q)

    # E_ξ[ln(p) | ε] = -ln(σ_t) - E[ξ²]/(2σ_t²)
    mean_lnp = -np.log(s_total) - _xi2_mean * inv_st2 / 2.0

    # Var_ξ[ln(p) | ε] = Var[ξ²] / (4 σ_t⁴)  (only the ξ² term varies)
    var_lnp_given_eps = _xi2_var * inv_st2 ** 2 / 4.0

    # Law of total variance over strain (axis=0 = crystals)
    result = var_lnp_given_eps.mean(axis=0) + mean_lnp.var(axis=0)
    return result  # (N_q,)

def model_relp_poly(d_arr, sig0, sig1_mean, sig1_std, sig2, sig_scale_sq):
    """σ²_LN(d) = partiality_var(q=1/d) + σ²_scale"""
    q_arr = 1.0 / np.asarray(d_arr, dtype=np.float64)
    return var_lnp_batch(q_arr, sig0, sig1_mean, sig1_std, sig2) + sig_scale_sq

sLN2 = sLN ** 2

# ── Print Ewald sphere widths ──
print(f"\n── Ewald sphere widths ──")
print(f"  {'d(Å)':>5} {'q(Å⁻¹)':>8} {'σ_ewald':>10} {'σ_bw':>10}"
      f" {'σ_div':>10} {'r_p/σ_ew':>8}")
for d_ex in [0.9, 1.0, 1.5, 2.0, 3.0, 5.0]:
    q_ex = 1.0 / d_ex
    sew = float(sigma_ewald_q(q_ex))
    sbw = BANDWIDTH * WAVELENGTH * q_ex ** 2 / 2.0
    sdv = DIVERGENCE * q_ex
    print(f"  {d_ex:>5.1f} {q_ex:>8.4f} {sew:>10.6f} {sbw:>10.6f}"
          f" {sdv:>10.6f} {R_PROFILE/sew:>8.2f}")

# ══════════════════════════════════════════════════════════════
# 14b. Distribution-matching fit (KL divergence on full shape)
# ══════════════════════════════════════════════════════════════
# Instead of fitting σ²_LN (variance only), fit the physical model
# parameters by matching the full observed distribution of I/⟨I⟩
# per shell via binned KL divergence.
from scipy.optimize import minimize

# ── Build pooled per-shell empirical distributions ──
# (Need these before the GoF section, reuse in plotting)
N_HIST_BINS = 60
LOG_RANGE = (-6.0, 3.0)   # range of log(I/⟨I⟩) for histograms
bin_edges = np.linspace(*LOG_RANGE, N_HIST_BINS + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Pool normalized observations per shell (strong HKLs, SNR > 3)
SNR_THRESH = 3.0
mean_sigG_per_hkl = sigG2_f.sqrt()
snr_f = mean_f / mean_sigG_per_hkl.clamp(min=1e-6)

# Build groups for later use (GoF test also needs these)
counts_all = torch.bincount(hkl_id, minlength=n_unique)
order_all = hkl_id.argsort()
groups_gof = I_obs[order_all].split(counts_all.tolist())

shell_hist_data = {}   # sid → normalized histogram (density)
shell_pooled = {}      # sid → raw pooled array

for sid in range(N_SHELLS):
    hm = hkl_shell_f == sid
    snr_mask = hm & (snr_f > SNR_THRESH)
    if snr_mask.sum() < 10:
        continue

    hkl_indices = torch.where(keep)[0][snr_mask]
    pooled = []
    for h_idx in hkl_indices:
        h = h_idx.item()
        g = groups_gof[h]
        pos = g[g > 0]
        if len(pos) < 5:
            continue
        mu_h = pos.mean()
        if mu_h > 0:
            pooled.append((pos / mu_h).numpy())

    if len(pooled) < 5:
        continue
    pooled = np.concatenate(pooled)
    pooled = pooled[pooled > 0]
    shell_pooled[sid] = pooled

    # Binned histogram of log(I/⟨I⟩)
    log_pooled = np.log(pooled)
    hist, _ = np.histogram(log_pooled, bins=bin_edges, density=True)
    hist = hist / hist.sum()  # ensure normalized as probability
    hist = np.maximum(hist, 1e-10)  # avoid log(0)
    shell_hist_data[sid] = hist

print(f"\n── Distribution-matching fit ──")
print(f"  Shells with data: {len(shell_hist_data)}")
print(f"  Histogram bins: {N_HIST_BINS} over log(I/⟨I⟩) ∈ [{LOG_RANGE[0]}, {LOG_RANGE[1]}]")

# ── MC distribution generator (fully vectorized across shells) ──
MC_FIT_RNG = np.random.default_rng(99)
MC_FIT_N = 10_000   # reduced from 50k — plenty for histogram matching
mc_fit_z = MC_FIT_RNG.standard_normal(MC_FIT_N)
mc_fit_xi2 = MC_FIT_RNG.uniform(-R_PROFILE, R_PROFILE, MC_FIT_N) ** 2
mc_fit_ln_scale_unit = MC_FIT_RNG.standard_normal(MC_FIT_N)  # pre-generate

# Pre-compute shell q-values and observed histograms as arrays
_fit_sids = sorted(shell_hist_data.keys())
_fit_qq = np.array([1.0 / shell_d[s].item() for s in _fit_sids])
_fit_sew = sigma_ewald_q(_fit_qq)                          # (N_shells,)
_fit_hists = np.stack([shell_hist_data[s] for s in _fit_sids])  # (N_shells, N_bins)
_fit_log_hists = np.log(_fit_hists)                         # pre-compute log

def generate_lnp_distribution(q, sig0, sig1_mean, sig1_std, sig2, sig_scale):
    """Generate MC samples of log(partiality × scale) at resolution q.
    Used only for plotting (not in the fast KL loop)."""
    if sig1_std > 1e-12:
        eps = np.abs(sig1_mean + sig1_std * mc_fit_z)
    else:
        eps = np.full(MC_FIT_N, max(sig1_mean, 0.0))
    s_ew = float(sigma_ewald_q(q))
    s_relp = np.maximum(sig0 + eps * q + sig2 * q ** 2, 1e-8)
    s_total = np.sqrt(s_relp ** 2 + s_ew ** 2)
    ln_p = -np.log(s_total) - mc_fit_xi2 / (2.0 * s_total ** 2)
    ln_total = ln_p + sig_scale * mc_fit_ln_scale_unit
    ln_total -= ln_total.mean()
    return ln_total

def kl_divergence_total(params):
    """Total KL across all shells — fully vectorized."""
    sig0, sig1_mean, sig1_std, sig2, sig_scale = params

    # Strain samples: (MC_FIT_N,)
    if sig1_std > 1e-12:
        eps = np.abs(sig1_mean + sig1_std * mc_fit_z)
    else:
        eps = np.full(MC_FIT_N, max(sig1_mean, 0.0))

    total_kl = 0.0
    for i, (qq, sew) in enumerate(zip(_fit_qq, _fit_sew)):
        # Relp + total width: (MC_FIT_N,)
        s_relp = np.maximum(sig0 + eps * qq + sig2 * qq ** 2, 1e-8)
        s_total = np.sqrt(s_relp ** 2 + sew ** 2)

        # ln(I/⟨I⟩) samples
        ln_total = (-np.log(s_total)
                    - mc_fit_xi2 / (2.0 * s_total ** 2)
                    + sig_scale * mc_fit_ln_scale_unit)
        ln_total -= ln_total.mean()

        # Histogram via searchsorted (faster than np.histogram)
        idx = np.searchsorted(bin_edges, ln_total, side='right') - 1
        idx = np.clip(idx, 0, N_HIST_BINS - 1)
        hist_model = np.bincount(idx, minlength=N_HIST_BINS).astype(np.float64)
        hist_model /= hist_model.sum()
        np.maximum(hist_model, 1e-10, out=hist_model)

        # KL(data || model)
        total_kl += np.sum(_fit_hists[i] * (_fit_log_hists[i] - np.log(hist_model)))

    return total_kl / len(_fit_qq)

# ── Optimize ──
print(f"  Minimizing KL divergence ({len(_fit_sids)} shells, "
      f"{MC_FIT_N} MC samples)...")
from scipy.optimize import differential_evolution

bounds_de = [
    (1e-5, 0.02),    # sig0: mosaicity
    (1e-5, 0.02),    # sig1_mean: mean strain
    (1e-5, 0.02),    # sig1_std: strain spread
    (1e-6, 0.005),   # sig2: disorder
    (0.05, 3.0),     # sig_scale: log-scale spread
]

result_de = differential_evolution(
    kl_divergence_total, bounds_de,
    seed=42, maxiter=50, tol=1e-3, polish=True,
    popsize=10,       # smaller population (default 15)
)

popt_rp = result_de.x
s0, s1m, s1s, s2, s_scale = popt_rp
print(f"\n  ── Distribution-matched parameters ──")
print(f"  σ_0       = {s0:.6f} Å⁻¹  ({s0*10:.5f} nm⁻¹)  [size/mosaicity]")
print(f"  ε_mean    = {s1m:.6f} Å⁻¹  ({s1m*10:.5f} nm⁻¹)  [mean strain]")
print(f"  ε_std     = {s1s:.6f} Å⁻¹  ({s1s*10:.5f} nm⁻¹)  [strain spread]")
print(f"  σ_2       = {s2:.7f} Å⁻¹  ({s2*10:.6f} nm⁻¹)  [disorder]")
print(f"  σ_scale   = {s_scale:.4f}  [overall scale variation]")
print(f"  KL div    = {result_de.fun:.6f}")
print(f"  Converged: {result_de.success}")

# Show model σ²_LN vs data for comparison
print(f"\n  {'d(Å)':>5} {'q':>6} {'σ_relp':>10} {'σ_ewald':>10}"
      f" {'σ²_part':>8} {'σ_LN_mod':>8} {'σ_LN_obs':>8}")
for i, dd in enumerate(d_np):
    qq = 1.0 / dd
    sr = s0 + s1m * qq + s2 * qq ** 2
    sew = float(sigma_ewald_q(qq))
    vp = float(var_lnp_batch(np.array([qq]), s0, s1m, s1s, s2)[0])
    vt = vp + s_scale ** 2
    print(f"  {dd:>5.2f} {qq:>6.3f} {sr:>10.6f} {sew:>10.6f}"
          f" {vp:>8.4f} {np.sqrt(vt):>8.3f} {sLN[i]:>8.3f}")

# Simple empirical fit for comparison
try:
    popt_LN2, _ = curve_fit(model_LN_inv_d2, d_np, sLN, p0=[1.0, 0.5])
    print(f"\n  Empirical: σ_LN = {popt_LN2[0]:.3f} + {popt_LN2[1]:.3f}/d²")
except Exception:
    popt_LN2 = None

# ══════════════════════════════════════════════════════════════
# 15. Lognormal goodness-of-fit test
# ══════════════════════════════════════════════════════════════
# (groups_gof, snr_f, shell_pooled already computed in 14b above)
from scipy.stats import shapiro, anderson, kstest

MIN_MULT_GOF = 20

gof_mask = (mult_f >= MIN_MULT_GOF) & (snr_f > SNR_THRESH)
print(f"\n── Lognormal goodness-of-fit ──")
print(f"Testing {gof_mask.sum().item()} HKLs "
      f"(mult ≥ {MIN_MULT_GOF}, SNR > {SNR_THRESH})")

gof_hkl_ids = torch.where(keep)[0][gof_mask]
shapiro_pvals = []
shapiro_d = []
shapiro_shell = []
ad_stats = []

for idx_h in gof_hkl_ids:
    h = idx_h.item()
    g = groups_gof[h]
    pos = g[g > 0]
    if len(pos) < MIN_MULT_GOF:
        continue
    log_pos = torch.log(pos).numpy()

    # Shapiro-Wilk test (H0: normal)
    if len(log_pos) <= 5000:  # shapiro limit
        stat, pval = shapiro(log_pos)
        shapiro_pvals.append(pval)
        shapiro_d.append(d_spacing[h].item())
        shapiro_shell.append(hkl_shell_all[h].item())

    # Anderson-Darling
    result = anderson(log_pos, dist='norm')
    ad_stats.append(result.statistic)

shapiro_pvals = np.array(shapiro_pvals)
shapiro_d = np.array(shapiro_d)
ad_stats = np.array(ad_stats)

frac_reject_005 = (shapiro_pvals < 0.05).mean()
frac_reject_001 = (shapiro_pvals < 0.01).mean()
print(f"  Shapiro-Wilk: {frac_reject_005:.1%} reject at p<0.05, "
      f"{frac_reject_001:.1%} reject at p<0.01")
print(f"  (If lognormal is perfect, expect ~5% and ~1% false positives)")

# Per-shell rejection rate
shell_reject = np.zeros(N_SHELLS)
shell_gof_n = np.zeros(N_SHELLS)
for i, sid in enumerate(np.array([hkl_shell_all[h].item() for h in gof_hkl_ids
                                   if counts_all[h] >= MIN_MULT_GOF])):
    if i < len(shapiro_pvals):
        _sid = int(sid)
        shell_gof_n[_sid] += 1
        if shapiro_pvals[i] < 0.05:
            shell_reject[_sid] += 1

# ══════════════════════════════════════════════════════════════
# 15b. Distribution identification
# ══════════════════════════════════════════════════════════════
# (shell_pooled already built in section 14b)
from scipy.stats import (lognorm, gamma as gamma_dist, invgauss,
                         gennorm, skew, kurtosis)

print(f"\n── Distribution identification ──")

shell_skew = np.full(N_SHELLS, np.nan)
shell_kurt = np.full(N_SHELLS, np.nan)
shell_beta = np.full(N_SHELLS, np.nan)   # generalized Gaussian exponent
shell_gennorm_loc = np.full(N_SHELLS, np.nan)
shell_gennorm_scale = np.full(N_SHELLS, np.nan)
shell_loglik = {name: np.full(N_SHELLS, np.nan)
                for name in ["lognormal", "gamma", "inv_gaussian", "gen_gaussian"]}

for sid in range(N_SHELLS):
    pooled = shell_pooled.get(sid)
    if pooled is None or len(pooled) < 100:
        continue

    # Shape statistics of log(normalized obs)
    log_pooled = np.log(pooled)
    shell_skew[sid] = skew(log_pooled)
    shell_kurt[sid] = kurtosis(log_pooled)  # excess kurtosis

    # Fit competing distributions and compute log-likelihood
    try:
        p_ln = lognorm.fit(pooled, floc=0)
        shell_loglik["lognormal"][sid] = np.sum(lognorm.logpdf(pooled, *p_ln))
    except Exception:
        pass
    try:
        p_gm = gamma_dist.fit(pooled, floc=0)
        shell_loglik["gamma"][sid] = np.sum(gamma_dist.logpdf(pooled, *p_gm))
    except Exception:
        pass
    try:
        p_ig = invgauss.fit(pooled, floc=0)
        shell_loglik["inv_gaussian"][sid] = np.sum(invgauss.logpdf(pooled, *p_ig))
    except Exception:
        pass

    # Generalized Gaussian in log-space: f(y) ∝ exp(-|y-μ|^β / α^β)
    # β=2 → Gaussian (=lognormal), β>2 → sharper, β<2 → heavier tails
    try:
        p_gg = gennorm.fit(log_pooled)  # returns (beta, loc, scale)
        shell_beta[sid] = p_gg[0]
        shell_gennorm_loc[sid] = p_gg[1]
        shell_gennorm_scale[sid] = p_gg[2]
        shell_loglik["gen_gaussian"][sid] = np.sum(gennorm.logpdf(log_pooled, *p_gg))
    except Exception:
        pass

# Print summary
ok_shape = ~np.isnan(shell_skew)
print(f"\n  Log(I/⟨I⟩) shape statistics per shell:")
print(f"  {'shell':>5} {'d(Å)':>6} {'skew':>7} {'kurt':>7} {'β':>5}"
      f"  {'best_dist':>14} {'ΔlogL(GG-LN)':>13}")
for sid in range(N_SHELLS):
    if np.isnan(shell_skew[sid]):
        continue
    lls = {k: v[sid] for k, v in shell_loglik.items()}
    best = max(lls, key=lambda k: lls[k] if not np.isnan(lls[k]) else -np.inf)
    ll_ln = lls.get("lognormal", np.nan)
    ll_gg = lls.get("gen_gaussian", np.nan)
    dll = ll_gg - ll_ln if not (np.isnan(ll_gg) or np.isnan(ll_ln)) else np.nan
    print(f"  {sid:>5d} {shell_d[sid]:>6.2f} {shell_skew[sid]:>7.3f}"
          f" {shell_kurt[sid]:>7.3f} {shell_beta[sid]:>5.2f}"
          f"  {best:>14} {dll:>13.1f}")

# Summary
med_skew = np.nanmedian(shell_skew)
med_kurt = np.nanmedian(shell_kurt)
med_beta = np.nanmedian(shell_beta)
print(f"\n  Median skewness   = {med_skew:.3f}  (0 = symmetric)")
print(f"  Median ex. kurt.  = {med_kurt:.3f}  (0 = Gaussian, <0 = sharper)")
print(f"  Median β exponent = {med_beta:.2f}  (2 = Gaussian, >2 = sharper)")
if med_beta > 2.1:
    print(f"  → Super-Gaussian: sharper peak than lognormal (β={med_beta:.2f})")
elif med_beta < 1.9:
    print(f"  → Sub-Gaussian: broader peak than lognormal (β={med_beta:.2f})")
else:
    print(f"  → Approximately Gaussian in log-space (lognormal)")
if med_skew < -0.1:
    print(f"  → Negative skew: left tail heavier")
elif med_skew > 0.1:
    print(f"  → Positive skew: right tail heavier")

# ── Physical model prediction vs data ──
# Generate theoretical distribution using distribution-matched parameters
theory_shells = {}
if popt_rp is not None:
    s0, s1m, s1s, s2, s_scale = popt_rp
    print(f"\n  Generating theoretical distributions from fitted model...")
    for sid in sorted(shell_pooled.keys()):
        dd = shell_d[sid].item()
        if dd <= 0:
            continue
        qq = 1.0 / dd
        ln_samples = generate_lnp_distribution(qq, s0, s1m, s1s, s2, s_scale)
        theory_shells[sid] = ln_samples  # already log(I/⟨I⟩)

# ══════════════════════════════════════════════════════════════
# 16. Figure 6 — σ_LN model & lognormal GoF
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# (a) σ²_LN: data vs relp polynomial model
ax = axes[0, 0]
d_fine = np.linspace(D_PLOT_MIN, D_PLOT_MAX, 60)
q_fine = 1.0 / d_fine
ax.plot(x_shell, sLN2, "o", color="C0", zorder=5, label="data")
if popt_rp is not None:
    s0, s1m, s1s, s2, s_scale = popt_rp
    # model_relp_poly expects sig_scale_sq
    model_fine = model_relp_poly(d_fine, s0, s1m, s1s, s2, s_scale ** 2)
    part_fine = var_lnp_batch(1.0 / d_fine, s0, s1m, s1s, s2)
    ax.plot(d_to_x(d_fine), model_fine, "-", color="k", lw=2,
            label="physical model")
    ax.fill_between(d_to_x(d_fine), 0, part_fine, alpha=0.2, color="C2",
                    label="partiality (MC)")
    ax.fill_between(d_to_x(d_fine), part_fine, model_fine, alpha=0.2,
                    color="C1", label=f"scale (σ={s_scale:.2f})")
ax.set_ylabel("σ²_LN")
ax.set_title("σ²_LN decomposition")
ax.legend(fontsize=6)
setup_res_axis(ax)

# (b) σ_LN data vs models
ax = axes[0, 1]
ax.plot(x_shell, sLN, "o", color="C0", label="data")
if popt_rp is not None:
    ax.plot(d_to_x(d_fine), np.sqrt(model_fine), "-", color="k", lw=2,
            label="physical model")
    ax.plot(d_to_x(d_fine), np.sqrt(np.maximum(part_fine, 0)), ":", color="C2",
            label="partiality only")
if popt_LN2 is not None:
    ax.plot(d_to_x(d_fine), model_LN_inv_d2(d_fine, *popt_LN2),
            "--", color="gray", alpha=0.5, label="empirical a+b/d²")
ax.set_ylabel("σ_LN")
ax.set_title("σ_LN vs resolution")
ax.legend(fontsize=6)
setup_res_axis(ax)

# (c) Ewald sphere & relp widths vs resolution
ax = axes[0, 2]
s_ew = sigma_ewald_q(q_fine)
ax.plot(d_to_x(d_fine), s_ew * 1e3, "-", color="C1", label="σ_ewald (bw+div)")
s_bw = BANDWIDTH * WAVELENGTH * q_fine ** 2 / 2
ax.plot(d_to_x(d_fine), s_bw * 1e3, ":", color="C1", alpha=0.5,
        label="bandwidth")
s_dv = DIVERGENCE * q_fine
ax.plot(d_to_x(d_fine), s_dv * 1e3, "--", color="C1", alpha=0.5,
        label="divergence")
if popt_rp is not None:
    # Mean relp width: σ_0 + ε_mean·q + σ_2·q²
    s_rp_mean = s0 + s1m * q_fine + s2 * q_fine ** 2
    ax.plot(d_to_x(d_fine), s_rp_mean * 1e3, "-", color="C2",
            label="σ_relp (mean ε)")
    # ±1σ_ε band
    s_rp_hi = s0 + (s1m + s1s) * q_fine + s2 * q_fine ** 2
    s_rp_lo = np.maximum(s0 + max(s1m - s1s, 0) * q_fine + s2 * q_fine ** 2, 0)
    ax.fill_between(d_to_x(d_fine), s_rp_lo * 1e3, s_rp_hi * 1e3,
                    alpha=0.15, color="C2", label="±1σ_ε")
    # total (mean)
    s_total_mean = np.sqrt(s_rp_mean ** 2 + s_ew ** 2)
    ax.plot(d_to_x(d_fine), s_total_mean * 1e3, "-", color="C4",
            label="σ_total (mean)")
ax.axhline(R_PROFILE * 1e3, ls="-.", color="gray", alpha=0.5,
           label=f"r_p = {R_PROFILE*1e3:.1f}×10⁻³")
ax.set_ylabel("Width (×10⁻³ Å⁻¹)")
ax.set_title("Ewald sphere & relp widths vs q")
ax.legend(fontsize=5)
setup_res_axis(ax)

# (d) Shapiro-Wilk p-values vs resolution
ax = axes[1, 0]
ax.scatter(d_to_x(shapiro_d), shapiro_pvals, s=2, alpha=0.3, color="C4")
ax.axhline(0.05, ls="--", color="r", alpha=0.5, label="p=0.05")
ax.set_ylabel("Shapiro-Wilk p-value")
ax.set_title(f"Lognormal test ({frac_reject_005:.0%} reject at 5%)")
ax.set_yscale("log")
ax.legend(fontsize=8)
setup_res_axis(ax)

# (e) Rejection rate vs resolution
ax = axes[1, 1]
shell_ok_gof = shell_gof_n > 0
rej_rate = np.where(shell_ok_gof, shell_reject / shell_gof_n.clip(1), np.nan)
x_gof = d_to_x(shell_d[shell_ok_gof].numpy())
ax.plot(x_gof, rej_rate[shell_ok_gof], "o-", color="C3")
ax.axhline(0.05, ls="--", color="gray", alpha=0.5, label="expected 5%")
ax.set_ylabel("Fraction rejecting H0 (p<0.05)")
ax.set_title("Lognormal rejection rate vs resolution")
ax.legend(fontsize=8)
setup_res_axis(ax)

# (f) QQ plot: 4 representative strong HKLs
ax = axes[1, 2]
from scipy.stats import norm as norm_dist
# pick 4 HKLs with high multiplicity and SNR, spread across resolution
gof_snr = snr_f[gof_mask].numpy()
gof_d_arr = d_f[gof_mask].numpy()
gof_idx_sorted = np.argsort(gof_d_arr)
pick = [gof_idx_sorted[int(f * len(gof_idx_sorted))] for f in [0.1, 0.35, 0.65, 0.9]]
colors_qq = ["C0", "C1", "C2", "C3"]
for ci, pi in enumerate(pick):
    h = gof_hkl_ids[pi].item()
    g = groups_gof[h]
    pos = g[g > 0]
    if len(pos) < 10:
        continue
    log_pos = torch.log(pos).numpy()
    log_pos_sorted = np.sort(log_pos)
    n = len(log_pos_sorted)
    theoretical_q = norm_dist.ppf((np.arange(1, n + 1) - 0.5) / n)
    # standardize
    mu, sigma = log_pos_sorted.mean(), log_pos_sorted.std()
    ax.scatter(theoretical_q, (log_pos_sorted - mu) / sigma,
               s=3, alpha=0.5, color=colors_qq[ci],
               label=f"d={d_spacing[h]:.2f}Å n={len(pos)}")
ax.plot([-4, 4], [-4, 4], "k:", alpha=0.3)
ax.set_xlabel("Theoretical quantiles (normal)")
ax.set_ylabel("Standardized log(I_pos)")
ax.set_title("QQ plot: log(I) vs normal")
ax.legend(fontsize=6)
ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)

fig.tight_layout()
fig.savefig(str(OUTDIR / "lognormal_model_and_gof.png"), dpi=150)
plt.close(fig)
print(f"Saved {OUTDIR / 'lognormal_model_and_gof.png'}")

# ══════════════════════════════════════════════════════════════
# 17. Generalized Gaussian + Gaussian mixture (fit-then-smooth)
# ══════════════════════════════════════════════════════════════
# Stage 1: Fit each shell independently (PyTorch L-BFGS on 6 params)
# Stage 2: Smooth parameters across shells with weighted moving average
from scipy.stats import norm as norm_dist2

print(f"\n── GenGauss + Gauss mixture (PyTorch L-BFGS) ──")

# Shells with enough data
fit_sids = sorted([s for s in shell_pooled if len(shell_pooled[s]) >= 500])
N_fit = len(fit_sids)
fit_data = [np.log(shell_pooled[s]) for s in fit_sids]
print(f"  Fitting {N_fit} shells independently, then smoothing")


def _fit_shell_lbfgs(data_np):
    """Fit GenGauss + Gauss mixture to log(I/⟨I⟩) via L-BFGS.

    Returns (β, μ₁, α, w₁, μ₂, σ₂) or None on failure.
    """
    y = torch.tensor(data_np, dtype=torch.float64)
    mu_all, sig_all = float(y.mean()), float(y.std())

    # Unconstrained parameters (reparameterized for positivity / bounds)
    #   β = softplus(beta_unc) + 0.5   → β ∈ (0.5, ∞)
    #   α = exp(log_alpha)             → α > 0
    #   σ₂ = exp(log_sigma2)           → σ₂ > 0
    #   w₁ = sigmoid(w_logit)          → w₁ ∈ (0, 1)
    beta_unc = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    mu1 = torch.tensor(mu_all - 0.3, dtype=torch.float64, requires_grad=True)
    log_alpha = torch.tensor(np.log(max(sig_all * 0.8, 0.05)),
                             dtype=torch.float64, requires_grad=True)
    w_logit = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    mu2 = torch.tensor(mu_all + 1.0, dtype=torch.float64, requires_grad=True)
    log_sigma2 = torch.tensor(np.log(max(sig_all * 0.7, 0.05)),
                              dtype=torch.float64, requires_grad=True)

    params = [beta_unc, mu1, log_alpha, w_logit, mu2, log_sigma2]
    optimizer = torch.optim.LBFGS(params, lr=1.0, max_iter=100,
                                  line_search_fn="strong_wolfe",
                                  tolerance_grad=1e-7, tolerance_change=1e-9)

    def closure():
        optimizer.zero_grad()
        beta = torch.nn.functional.softplus(beta_unc) + 0.5
        alpha = torch.exp(log_alpha)
        sigma2 = torch.exp(log_sigma2)
        w1 = torch.sigmoid(w_logit)

        # GenGauss log-pdf: log(β/(2α Γ(1/β))) - |y-μ₁|^β / α^β
        z1 = torch.abs(y - mu1) / alpha.clamp(min=1e-8)
        log_pdf1 = (torch.log(beta) - torch.log(2.0 * alpha)
                    - torch.lgamma(1.0 / beta)
                    - z1.clamp(min=1e-8).pow(beta))

        # Gaussian log-pdf
        log_pdf2 = (-0.5 * np.log(2 * np.pi) - log_sigma2
                    - 0.5 * ((y - mu2) / sigma2.clamp(min=1e-8)) ** 2)

        # Mixture NLL via log-sum-exp
        log_mix = torch.logaddexp(
            torch.log(w1.clamp(min=1e-8)) + log_pdf1,
            torch.log((1 - w1).clamp(min=1e-8)) + log_pdf2,
        )
        loss = -log_mix.mean()
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except Exception:
        return None

    with torch.no_grad():
        beta_val = (torch.nn.functional.softplus(beta_unc) + 0.5).item()
        mu1_val, mu2_val = mu1.item(), mu2.item()
        alpha_val = torch.exp(log_alpha).item()
        sigma2_val = torch.exp(log_sigma2).item()
        w1_val = torch.sigmoid(w_logit).item()

    # Ensure μ₁ < μ₂ (swap components if needed)
    if mu1_val > mu2_val:
        beta_val, mu1_val, alpha_val, w1_val, mu2_val, sigma2_val = (
            beta_val, mu2_val, sigma2_val, 1 - w1_val, mu1_val, alpha_val)

    return np.array([beta_val, mu1_val, alpha_val, w1_val, mu2_val, sigma2_val])


# Stage 1: independent per-shell fits
raw_params = {}  # sid → (β, μ₁, α, w₁, μ₂, σ₂)
for i, sid in enumerate(fit_sids):
    result = _fit_shell_lbfgs(fit_data[i])
    if result is not None:
        raw_params[sid] = result
print(f"  Stage 1: {len(raw_params)}/{N_fit} shells converged")

# Stage 2: smooth with Gaussian kernel (σ=1.5 shells)
SMOOTH_SIGMA = 1.5  # in units of shell index
smooth_sids = sorted(raw_params.keys())
raw_arr = np.array([raw_params[s] for s in smooth_sids])  # (n, 6)
idx_arr = np.array([fit_sids.index(s) for s in smooth_sids])  # position index

mix_params = {}
# Don't smooth μ₁ and μ₂ (these track resolution-dependent intensity)
# Smooth: β (col 0), α (col 2), w₁ (col 3), σ₂ (col 5)
smooth_cols = [0, 2, 3, 5]
smoothed_arr = raw_arr.copy()
for j in smooth_cols:
    vals = raw_arr[:, j]
    smoothed = np.zeros_like(vals)
    for k in range(len(vals)):
        weights = np.exp(-0.5 * ((idx_arr - idx_arr[k]) / SMOOTH_SIGMA) ** 2)
        smoothed[k] = np.average(vals, weights=weights)
    smoothed_arr[:, j] = smoothed

for k, sid in enumerate(smooth_sids):
    mix_params[sid] = tuple(smoothed_arr[k])

# Print table
print(f"\n  {'shell':>5} {'d(Å)':>6} {'β':>5} {'w₁':>5} {'μ₁':>7} {'α':>6}"
      f" {'w₂':>5} {'μ₂':>7} {'σ₂':>6} {'Δμ':>6}")
for sid in fit_sids:
    beta, mu1, alpha, w1, mu2, sigma2 = mix_params[sid]
    dd = shell_d[sid].item()
    print(f"  {sid:>5d} {dd:>6.2f} {beta:>5.2f} {w1:>5.2f} {mu1:>7.3f}"
          f" {alpha:>6.3f} {1-w1:>5.2f} {mu2:>7.3f} {sigma2:>6.3f}"
          f" {mu2-mu1:>6.3f}")

# ── Figure 7: distributions with all model overlays ──
show_sids = sorted(shell_pooled.keys())
n_show = min(6, len(show_sids))
pick_sids = [show_sids[int(i * len(show_sids) / n_show)] for i in range(n_show)]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for ax, sid in zip(axes.flat, pick_sids):
    pooled = shell_pooled.get(sid)
    if pooled is None or len(pooled) < 100:
        continue

    dd = shell_d[sid].item()
    log_pooled = np.log(pooled)
    ax.hist(log_pooled, bins=80, density=True, alpha=0.4, color="C0",
            label="data")

    x_plot = np.linspace(np.percentile(log_pooled, 0.5),
                         np.percentile(log_pooled, 99.5), 200)

    # Single lognormal
    mu_fit, sig_fit = log_pooled.mean(), log_pooled.std()
    ax.plot(x_plot, norm_dist2.pdf(x_plot, mu_fit, sig_fit),
            "-", color="C1", lw=1, alpha=0.7, label="lognormal")

    # GenGauss + Gauss mixture
    if sid in mix_params:
        beta, mu1, alpha, w1, mu2, sigma2 = mix_params[sid]
        pdf1 = gennorm.pdf(x_plot, beta, loc=mu1, scale=alpha)
        pdf2 = norm_dist2.pdf(x_plot, loc=mu2, scale=sigma2)
        pdf_mix = w1 * pdf1 + (1 - w1) * pdf2
        ax.plot(x_plot, pdf_mix, "-", color="k", lw=2, label="GG+G mixture")
        ax.plot(x_plot, w1 * pdf1, "--", color="C5", lw=1,
                label=f"GenGauss (β={beta:.2f}, w={w1:.2f})")
        ax.plot(x_plot, (1 - w1) * pdf2, "--", color="C3", lw=1,
                label=f"Gauss (w={1-w1:.2f})")

    ax.set_xlabel("log(I / ⟨I⟩)")
    ax.set_ylabel("Density")
    ax.set_title(f"d = {dd:.2f} Å")
    ax.legend(fontsize=5)

for ax in axes.flat[n_show:]:
    ax.set_visible(False)

fig.suptitle("GenGauss(β≈1.5) + Gaussian mixture in log(I/⟨I⟩)", fontsize=13)
fig.tight_layout()
fig.savefig(str(OUTDIR / "distribution_comparison.png"), dpi=150)
plt.close(fig)
print(f"Saved {OUTDIR / 'distribution_comparison.png'}")

# ── Figure 7b: mixture model parameters vs resolution ──
fig, axes = plt.subplots(2, 3, figsize=(16, 8))

sids_mix = sorted(mix_params.keys())
d_mix = np.array([shell_d[s].item() for s in sids_mix])
beta_arr = np.array([mix_params[s][0] for s in sids_mix])
mu1_arr = np.array([mix_params[s][1] for s in sids_mix])
alpha_arr = np.array([mix_params[s][2] for s in sids_mix])
w1_arr = np.array([mix_params[s][3] for s in sids_mix])
mu2_arr = np.array([mix_params[s][4] for s in sids_mix])
sig2_arr = np.array([mix_params[s][5] for s in sids_mix])
dmu_arr = mu2_arr - mu1_arr

# (a) β exponent vs resolution
ax = axes[0, 0]
ax.plot(d_to_x(d_mix), beta_arr, "o-", color="C5", lw=2)
ax.axhline(2, ls=":", color="k", alpha=0.3, label="β=2 (Gaussian)")
ax.axhline(1, ls=":", color="r", alpha=0.3, label="β=1 (Laplace)")
ax.set_ylabel("β (shape exponent)")
ax.set_title("GenGauss exponent")
ax.legend(fontsize=8)
setup_res_axis(ax)

# (b) Component weights
ax = axes[0, 1]
ax.plot(d_to_x(d_mix), w1_arr, "o-", color="C5", label="GenGauss (main)")
ax.plot(d_to_x(d_mix), 1 - w1_arr, "s-", color="C3", label="Gauss (shoulder)")
ax.set_ylabel("Weight")
ax.set_title("Component weights")
ax.legend(fontsize=8)
setup_res_axis(ax)

# (c) Component means
ax = axes[0, 2]
ax.plot(d_to_x(d_mix), mu1_arr, "o-", color="C5", label="μ₁ (GenGauss)")
ax.plot(d_to_x(d_mix), mu2_arr, "s-", color="C3", label="μ₂ (Gauss)")
ax.set_ylabel("Mean of log(I/⟨I⟩)")
ax.set_title("Component means")
ax.legend(fontsize=8)
setup_res_axis(ax)

# (d) Component widths
ax = axes[1, 0]
ax.plot(d_to_x(d_mix), alpha_arr, "o-", color="C5", label="α (GenGauss)")
ax.plot(d_to_x(d_mix), sig2_arr, "s-", color="C3", label="σ₂ (Gauss)")
ax.set_ylabel("Scale parameter")
ax.set_title("Component widths")
ax.legend(fontsize=8)
setup_res_axis(ax)

# (e) Separation Δμ
ax = axes[1, 1]
ax.plot(d_to_x(d_mix), dmu_arr, "o-", color="C0")
ax.set_ylabel("Δμ = μ₂ − μ₁")
ax.set_title("Separation between components")
setup_res_axis(ax)

# (f) Intensity ratio of the two components
ax = axes[1, 2]
ax.plot(d_to_x(d_mix), np.exp(dmu_arr), "o-", color="C5")
ax.axhline(1, ls=":", color="k", alpha=0.3)
ax.set_ylabel("exp(Δμ) — intensity ratio")
ax.set_title("Stronger / weaker component ratio")
setup_res_axis(ax)

fig.suptitle("GMM component parameters vs resolution", fontsize=13)
fig.tight_layout()
fig.savefig(str(OUTDIR / "gmm_parameters.png"), dpi=150)
plt.close(fig)
print(f"Saved {OUTDIR / 'gmm_parameters.png'}")

# ── Figure 8: shape diagnostics vs resolution ──
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# (a) β exponent vs resolution
ax = axes[0, 0]
ok_b = ~np.isnan(shell_beta)
ax.plot(d_to_x(shell_d[ok_b].numpy()), shell_beta[ok_b], "o-", color="C5", lw=2)
ax.axhline(2, ls=":", color="k", alpha=0.3, label="β=2 (Gaussian/lognormal)")
ax.axhline(1, ls=":", color="r", alpha=0.3, label="β=1 (Laplace)")
ax.set_ylabel("β (shape exponent)")
ax.set_title("Generalized Gaussian exponent vs resolution")
ax.legend(fontsize=8)
setup_res_axis(ax)

# (b) Skewness vs resolution
ax = axes[0, 1]
ok_sk = ~np.isnan(shell_skew)
ax.plot(d_to_x(shell_d[ok_sk].numpy()), shell_skew[ok_sk], "o-", color="C0")
ax.axhline(0, ls=":", color="k", alpha=0.3)
ax.set_ylabel("Skewness of log(I/⟨I⟩)")
ax.set_title("Log-space skewness")
setup_res_axis(ax)

# (c) Excess kurtosis vs resolution
ax = axes[1, 0]
ax.plot(d_to_x(shell_d[ok_sk].numpy()), shell_kurt[ok_sk], "o-", color="C1")
ax.axhline(0, ls=":", color="k", alpha=0.3, label="Gaussian (kurt=0)")
ax.set_ylabel("Excess kurtosis")
ax.set_title("Log-space kurtosis")
ax.legend(fontsize=8)
setup_res_axis(ax)

# (d) Log-likelihood: gen_gaussian vs lognormal
ax = axes[1, 1]
dll_gg = shell_loglik["gen_gaussian"] - shell_loglik["lognormal"]
ok_ll = ~np.isnan(dll_gg)
ax.plot(d_to_x(shell_d[ok_ll].numpy()), dll_gg[ok_ll], "o-", color="C5",
        label="gen. Gaussian − lognormal")
dll_gm = shell_loglik["gamma"] - shell_loglik["lognormal"]
ok_gm = ~np.isnan(dll_gm)
ax.plot(d_to_x(shell_d[ok_gm].numpy()), dll_gm[ok_gm], "s-", color="C2",
        alpha=0.6, label="gamma − lognormal")
ax.axhline(0, ls=":", color="k", alpha=0.3)
ax.set_ylabel("ΔlogL vs lognormal")
ax.set_title("Likelihood improvement over lognormal")
ax.legend(fontsize=7)
setup_res_axis(ax)

fig.tight_layout()
fig.savefig(str(OUTDIR / "distribution_shape.png"), dpi=150)
plt.close(fig)
print(f"Saved {OUTDIR / 'distribution_shape.png'}")

# ── Figure 9: log(std) vs log(⟨I⟩) per HKL ──────────────────
# This separates the Poisson (slope ½) from multiplicative (slope 1)
# regimes. At low intensity, std ≈ σ_gauss ∝ √I (Poisson).
# At high intensity, std ≈ σ_LN · I (multiplicative).
fig, ax = plt.subplots(figsize=(8, 7))

pos_mask_plot = (mean_f > 0) & (var_f > 0)
logI = torch.log10(mean_f[pos_mask_plot]).numpy()
logS = torch.log10(var_f[pos_mask_plot].sqrt()).numpy()
d_plot = d_f[pos_mask_plot].numpy()

sc = ax.scatter(logI, logS, c=d_to_x(d_plot), s=1, alpha=0.15,
                cmap="viridis", rasterized=True)
plt.colorbar(sc, ax=ax, label="1/d² (resolution)")

# Overlay Poisson line: std = g · √(I + bg·n_peak·(1+n/n_bg))
# For large I: std ≈ g·√I → log(std) = ½ log(I) + log(g)
I_range = np.logspace(logI.min(), logI.max(), 200)
ax.plot(np.log10(I_range), 0.5 * np.log10(I_range) + np.log10(gain_fit),
        "r--", lw=2, label=f"Poisson: slope ½ (g={gain_fit:.3f})")

# Multiplicative line: std = σ_LN · I → log(std) = log(I) + log(σ_LN)
med_sLN = np.median(sLN)
ax.plot(np.log10(I_range), np.log10(I_range) + np.log10(med_sLN),
        "b--", lw=2, label=f"Multiplicative: slope 1 (σ_LN={med_sLN:.2f})")

# Combined: std = √(g²·I + σ_LN²·I²)
std_combined = np.sqrt(gain_fit ** 2 * I_range + med_sLN ** 2 * I_range ** 2)
ax.plot(np.log10(I_range), np.log10(std_combined),
        "k-", lw=2, label="combined")

ax.set_xlabel("log₁₀(⟨I⟩)")
ax.set_ylabel("log₁₀(std(I))")
ax.set_title("Noise regime: Poisson → multiplicative crossover")
ax.legend(fontsize=8)
ax.set_aspect("equal")
ax.set_xlim(logI.min() - 0.1, logI.max() + 0.1)
ax.set_ylim(logS.min() - 0.1, logS.max() + 0.1)

fig.tight_layout()
fig.savefig(str(OUTDIR / "log_std_vs_intensity.png"), dpi=150)
plt.close(fig)
print(f"Saved {OUTDIR / 'log_std_vs_intensity.png'}")

# ══════════════════════════════════════════════════════════════
# 20. Strain anisotropy diagnostic
# ══════════════════════════════════════════════════════════════
# If strain is anisotropic (1 soft + 2 stiff directions), then
# per-HKL σ_LN should depend on the direction of g in reciprocal space.
# HKLs along the soft axis → wider relps → higher σ_LN.
#
# Test: project each HKL's reciprocal vector onto the crystal axes
# (a*, b*, c*) and check if σ_LN correlates with direction.

print(f"\n── Strain anisotropy diagnostic ──")

# Per-HKL σ_LN (already computed: sigLN_f for filtered HKLs)
# Direction of each HKL: use fractional components h,k,l
# normalized by |g| to get the unit direction in reciprocal space.
# For triclinic P-1, h,k,l directly give the reciprocal-space direction
# relative to (a*, b*, c*).

hkl_filt = unique_hkl[keep].float()   # (N_filt, 3)
# Reciprocal-space vector g = h·a* + k·b* + l·c*
# Use median reciprocal vectors
med_astar = s._core.astar.median(dim=0).values  # (3,)
med_bstar = s._core.bstar.median(dim=0).values
med_cstar = s._core.cstar.median(dim=0).values
recip_mat = torch.stack([med_astar, med_bstar, med_cstar], dim=0)  # (3, 3)

# g = hkl @ recip_mat → (N_filt, 3) in Cartesian Å⁻¹
g_cart = hkl_filt @ recip_mat
g_mag = g_cart.norm(dim=1).clamp(min=1e-8)
g_hat = g_cart / g_mag[:, None]  # unit direction

# Directional cosines: projection onto a*, b*, c* directions
astar_hat = med_astar / med_astar.norm()
bstar_hat = med_bstar / med_bstar.norm()
cstar_hat = med_cstar / med_cstar.norm()

cos_a = (g_hat @ astar_hat).abs()  # |cos(angle with a*)|
cos_b = (g_hat @ bstar_hat).abs()
cos_c = (g_hat @ cstar_hat).abs()

# Only use HKLs with well-determined σ_LN
good_sLN = sigLN_f > 0.01
n_good = good_sLN.sum().item()
print(f"  {n_good} HKLs with σ_LN > 0.01")

# Correlation of σ_LN with directional cosines
for name, cos_proj in [("a*", cos_a), ("b*", cos_b), ("c*", cos_c)]:
    mask = good_sLN & (d_f > 0.95) & (d_f < 2.0)  # mid-resolution
    if mask.sum() > 100:
        r = np.corrcoef(cos_proj[mask].numpy(), sigLN_f[mask].numpy())[0, 1]
        print(f"  Corr(σ_LN, |cos({name})|) = {r:+.4f}  "
              f"(mid-res, {mask.sum()} HKLs)")

# ── Figure 10: anisotropy ──
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# (a-c) σ_LN vs |cos(angle with a*/b*/c*)| at mid-resolution
for i, (ax, name, cos_proj) in enumerate(zip(
        axes[0], ["a*", "b*", "c*"], [cos_a, cos_b, cos_c])):
    mask = good_sLN & (d_f > 0.95) & (d_f < 2.0)
    n_sc = min(5000, mask.sum().item())
    idx_sc = torch.where(mask)[0][torch.randperm(mask.sum())[:n_sc]]
    ax.scatter(cos_proj[idx_sc].numpy(), sigLN_f[idx_sc].numpy(),
               s=1, alpha=0.15, color=f"C{i}")
    # Binned trend
    n_dir_bins = 10
    for b in range(n_dir_bins):
        lo, hi = b / n_dir_bins, (b + 1) / n_dir_bins
        bm = mask & (cos_proj >= lo) & (cos_proj < hi)
        if bm.sum() > 10:
            ax.plot([(lo + hi) / 2], [sigLN_f[bm].median().item()],
                    "ko", markersize=6)
    ax.set_xlabel(f"|cos(angle with {name})|")
    ax.set_ylabel("σ_LN")
    ax.set_title(f"σ_LN vs {name} projection (0.95–2.0 Å)")
    ax.set_ylim(0, sigLN_f[good_sLN].quantile(0.99).item() * 1.1)

# (d) σ_LN in h,k,l space — ternary-ish projection
# Color each HKL by its σ_LN
ax = axes[1, 0]
mask = good_sLN & (d_f > 0.95) & (d_f < 2.0)
n_sc = min(5000, mask.sum().item())
idx_sc = torch.where(mask)[0][torch.randperm(mask.sum())[:n_sc]]
sc = ax.scatter(cos_a[idx_sc].numpy(), cos_b[idx_sc].numpy(),
                c=sigLN_f[idx_sc].numpy(), s=2, alpha=0.3,
                cmap="hot", vmin=sigLN_f[good_sLN].quantile(0.1).item(),
                vmax=sigLN_f[good_sLN].quantile(0.9).item())
ax.set_xlabel("|cos(a*)|")
ax.set_ylabel("|cos(b*)|")
ax.set_title("σ_LN colored by direction")
plt.colorbar(sc, ax=ax, label="σ_LN")

# (e) σ_LN vs resolution, split by dominant axis
ax = axes[1, 1]
dominant = torch.stack([cos_a, cos_b, cos_c], dim=1).argmax(dim=1)
for axis_id, name, color in [(0, "a* dom.", "C0"), (1, "b* dom.", "C1"),
                               (2, "c* dom.", "C2")]:
    m = good_sLN & (dominant == axis_id)
    if m.sum() < 50:
        continue
    # Bin by resolution
    d_ax = d_f[m]
    sLN_ax = sigLN_f[m]
    n_bins_ax = 15
    rank_ax = d_ax.argsort().argsort()
    bin_ax = (rank_ax.float() / m.sum() * n_bins_ax).long().clamp(max=n_bins_ax - 1)
    for b in range(n_bins_ax):
        bm = bin_ax == b
        if bm.sum() > 5:
            ax.plot(d_to_x(d_ax[bm].median().item()),
                    sLN_ax[bm].median().item(),
                    "o", color=color, markersize=4, alpha=0.8)
    ax.plot([], [], "o", color=color, label=name)  # for legend
ax.set_ylabel("σ_LN (median per bin)")
ax.set_title("σ_LN vs resolution by dominant axis")
ax.legend(fontsize=8)
setup_res_axis(ax)

# (f) Histogram of σ_LN split by dominant axis
ax = axes[1, 2]
mask_mid = good_sLN & (d_f > 0.95) & (d_f < 2.0)
for axis_id, name, color in [(0, "a* dom.", "C0"), (1, "b* dom.", "C1"),
                               (2, "c* dom.", "C2")]:
    m = mask_mid & (dominant == axis_id)
    if m.sum() > 20:
        ax.hist(sigLN_f[m].numpy(), bins=40, alpha=0.4, color=color,
                density=True, label=f"{name} (n={m.sum()})")
ax.set_xlabel("σ_LN")
ax.set_ylabel("Density")
ax.set_title("σ_LN distribution by dominant axis (0.95–2.0 Å)")
ax.legend(fontsize=7)

fig.suptitle("Strain anisotropy: σ_LN vs reciprocal-space direction", fontsize=13)
fig.tight_layout()
fig.savefig(str(OUTDIR / "strain_anisotropy.png"), dpi=150)
plt.close(fig)
print(f"Saved {OUTDIR / 'strain_anisotropy.png'}")

# ══════════════════════════════════════════════════════════════
# 21. Per-HKL generalized Gaussian fit (closed-form + Laplacian smooth)
# ══════════════════════════════════════════════════════════════
# Strategy (fast — no iterative optimizer):
#   1. Assign β per HKL from its resolution shell (from section 17)
#   2. Compute MLE μ and α per HKL in closed form given β
#   3. Laplacian-smooth α and β over the adjacency graph
#
# For GenGauss with known β:
#   MLE μ = weighted Lβ-median (iteratively reweighted, ~5 iterations)
#   MLE α^β = (β/n) Σ|y_j - μ|^β

print(f"\n══ Per-HKL GenGauss fit (closed-form + Laplacian smooth) ══")

MIN_OBS_PERHKL = 20
N_SMOOTH_ITER = 15     # Laplacian smoothing iterations
SMOOTH_LAMBDA = 0.3    # mixing: x_new = (1-λ)·x + λ·mean(neighbors)

# ── Build per-HKL observation index ──
pos_mask_fit = I_obs > 0
log_I_all = torch.log(I_obs[pos_mask_fit])
hkl_id_pos = hkl_id[pos_mask_fit]

counts_pos = torch.bincount(hkl_id_pos, minlength=n_unique)
valid_hkl_mask = counts_pos >= MIN_OBS_PERHKL
valid_indices = torch.where(valid_hkl_mask)[0]
n_valid = valid_indices.shape[0]
print(f"  {n_valid} HKLs with ≥ {MIN_OBS_PERHKL} positive observations")

# Remap to contiguous indices
remap = torch.full((n_unique,), -1, dtype=torch.long)
remap[valid_indices] = torch.arange(n_valid)
obs_new_hkl = remap[hkl_id_pos]
obs_keep = obs_new_hkl >= 0
log_I_fit = log_I_all[obs_keep]
hkl_fit_idx = obs_new_hkl[obs_keep]
n_obs_fit = log_I_fit.shape[0]
print(f"  {n_obs_fit} observations in fit")

# Sort by HKL for grouped access
sort_order = hkl_fit_idx.argsort()
log_I_sorted = log_I_fit[sort_order]
obs_counts = torch.bincount(hkl_fit_idx, minlength=n_valid)
offsets = torch.zeros(n_valid + 1, dtype=torch.long)
offsets[1:] = obs_counts.cumsum(0)

# ── Assign β per HKL from shell fit ──
hkl_valid = unique_hkl[valid_indices]
d_perhkl = get_d_spacing(hkl_valid.float(), median_cell)

# Map each valid HKL to its shell, then look up β from smooth shell fit
hkl_shell_valid = hkl_shell_all[valid_indices]
beta_perhkl = torch.full((n_valid,), 1.5)  # default
for sid in sorted(mix_params.keys()):
    m = hkl_shell_valid == sid
    if m.sum() > 0:
        beta_perhkl[m] = mix_params[sid][0]  # β from shell fit
print(f"  β assigned from shell fit: range [{beta_perhkl.min():.2f}, {beta_perhkl.max():.2f}]")

# ── Closed-form MLE for μ and α given β (vectorized) ──
# Use scatter operations to avoid Python loop over 50k HKLs.
# For GenGauss with known β:
#   MLE μ ≈ weighted median (for β<2 downweights outliers; β=2 → mean)
#   MLE α^β = (β/n) Σ|y - μ|^β
#
# Fast approach: use median for μ (exact for β=1, good approx for β~1.5),
# then compute α in closed form.

# Per-HKL median via grouped operation
mu_perhkl = torch.zeros(n_valid)
alpha_perhkl = torch.zeros(n_valid)

# Vectorized: compute median and α for each HKL using offsets
# (the loop is in Python but does minimal work per iteration — just tensor slicing)
beta_np = beta_perhkl.numpy()
for i in range(n_valid):
    s, e = offsets[i].item(), offsets[i + 1].item()
    vals = log_I_sorted[s:e]
    n_h = e - s
    beta_i = beta_np[i]

    # μ = median (fast, robust, exact for β=1)
    mu_i = vals.median().item()

    # One IRLS correction to shift from median toward Lβ-mean
    if abs(beta_i - 1.0) > 0.05:
        residual = (vals - mu_i).abs().clamp(min=1e-8)
        w = residual.pow(beta_i - 2)
        w = w / w.sum()
        mu_i = (w * vals).sum().item()

    # α^β = (β/n) Σ|y - μ|^β
    abs_r = (vals - mu_i).abs().clamp(min=1e-8)
    alpha_i = float(((beta_i / n_h) * abs_r.pow(beta_i).sum().item()) ** (1.0 / beta_i))

    mu_perhkl[i] = mu_i
    alpha_perhkl[i] = max(alpha_i, 0.01)

print(f"  Raw MLE: μ range [{mu_perhkl.min():.2f}, {mu_perhkl.max():.2f}], "
      f"α range [{alpha_perhkl.min():.3f}, {alpha_perhkl.max():.3f}]")

# ── Build adjacency graph ──
hkl_to_new = {}
for i in range(n_valid):
    key = (hkl_valid[i, 0].item(), hkl_valid[i, 1].item(), hkl_valid[i, 2].item())
    hkl_to_new[key] = i

# Build sparse adjacency as edge index tensors (for vectorized smoothing)
edge_src = []
edge_dst = []
for i in range(n_valid):
    h, k, l = hkl_valid[i, 0].item(), hkl_valid[i, 1].item(), hkl_valid[i, 2].item()
    for dh, dk, dl in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
        j = hkl_to_new.get((h+dh, k+dk, l+dl))
        if j is not None:
            edge_src.append(j)
            edge_dst.append(i)

edge_src_t = torch.tensor(edge_src, dtype=torch.long)
edge_dst_t = torch.tensor(edge_dst, dtype=torch.long)
# Count neighbors per node
n_neighbors = torch.bincount(edge_dst_t, minlength=n_valid).float().clamp(min=1)
print(f"  {len(edge_src)//2} adjacency edges "
      f"(mean {n_neighbors.mean():.1f} neighbors/HKL)")

# ── Laplacian smoothing on α and β (vectorized) ──
log_alpha_smooth = torch.log(alpha_perhkl.clone())
beta_smooth = beta_perhkl.clone()

for it in range(N_SMOOTH_ITER):
    # Scatter-add neighbor values, then divide by neighbor count
    nb_sum_la = torch.zeros(n_valid)
    nb_sum_la.scatter_add_(0, edge_dst_t, log_alpha_smooth[edge_src_t])
    nb_mean_la = nb_sum_la / n_neighbors

    nb_sum_b = torch.zeros(n_valid)
    nb_sum_b.scatter_add_(0, edge_dst_t, beta_smooth[edge_src_t])
    nb_mean_b = nb_sum_b / n_neighbors

    has_nb = n_neighbors > 0.5  # nodes with neighbors
    log_alpha_smooth[has_nb] = ((1 - SMOOTH_LAMBDA) * log_alpha_smooth[has_nb]
                                 + SMOOTH_LAMBDA * nb_mean_la[has_nb])
    beta_smooth[has_nb] = ((1 - SMOOTH_LAMBDA) * beta_smooth[has_nb]
                            + SMOOTH_LAMBDA * nb_mean_b[has_nb])

alpha_perhkl = torch.exp(log_alpha_smooth)
beta_perhkl = beta_smooth
I0_perhkl = torch.exp(mu_perhkl)

print(f"  After {N_SMOOTH_ITER} Laplacian iterations (λ={SMOOTH_LAMBDA}):")
print(f"    α: [{alpha_perhkl.min():.3f}, {alpha_perhkl.median():.3f}, {alpha_perhkl.max():.3f}]")
print(f"    β: [{beta_perhkl.min():.2f}, {beta_perhkl.median():.2f}, {beta_perhkl.max():.2f}]")

print(f"\n  Per-HKL fit summary:")
print(f"    I0:  median={I0_perhkl.median():.2f}, "
      f"range=[{I0_perhkl.min():.2f}, {I0_perhkl.max():.2f}]")
print(f"    α:   median={alpha_perhkl.median():.3f}, "
      f"range=[{alpha_perhkl.min():.3f}, {alpha_perhkl.max():.3f}]")
print(f"    β:   median={beta_perhkl.median():.3f}, "
      f"range=[{beta_perhkl.min():.3f}, {beta_perhkl.max():.3f}]")

# ── Figure 11: per-HKL fit parameters vs resolution ──
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

d_np = d_perhkl.numpy()
x_plot_hkl = d_to_x(d_np)

# (a) β vs resolution
ax = axes[0, 0]
n_sc = min(10000, n_valid)
idx_sc = torch.randperm(n_valid)[:n_sc]
ax.scatter(x_plot_hkl[idx_sc], beta_perhkl[idx_sc].numpy(),
           s=1, alpha=0.1, color="C5", rasterized=True)
# Binned median
n_res_bins = 30
d_rank_hkl = d_perhkl.argsort().argsort()
res_bin = (d_rank_hkl.float() / n_valid * n_res_bins).long().clamp(max=n_res_bins - 1)
for b in range(n_res_bins):
    bm = res_bin == b
    if bm.sum() > 10:
        ax.plot(d_to_x(d_perhkl[bm].median().item()),
                beta_perhkl[bm].median().item(), "ko", markersize=4)
ax.axhline(2, ls=":", color="k", alpha=0.3, label="β=2 (Gaussian)")
ax.axhline(1, ls=":", color="r", alpha=0.3, label="β=1 (Laplace)")
ax.set_ylabel("β")
ax.set_title("Shape exponent β vs resolution")
ax.legend(fontsize=7)
setup_res_axis(ax)

# (b) α vs resolution
ax = axes[0, 1]
ax.scatter(x_plot_hkl[idx_sc], alpha_perhkl[idx_sc].numpy(),
           s=1, alpha=0.1, color="C0", rasterized=True)
for b in range(n_res_bins):
    bm = res_bin == b
    if bm.sum() > 10:
        ax.plot(d_to_x(d_perhkl[bm].median().item()),
                alpha_perhkl[bm].median().item(), "ko", markersize=4)
ax.set_ylabel("α (scale)")
ax.set_title("Width α vs resolution")
setup_res_axis(ax)

# (c) I0 vs resolution (log scale)
ax = axes[0, 2]
ax.scatter(x_plot_hkl[idx_sc], torch.log10(I0_perhkl[idx_sc]).numpy(),
           s=1, alpha=0.1, color="C2", rasterized=True)
for b in range(n_res_bins):
    bm = res_bin == b
    if bm.sum() > 10:
        ax.plot(d_to_x(d_perhkl[bm].median().item()),
                torch.log10(I0_perhkl[bm].median()).item(), "ko", markersize=4)
ax.set_ylabel("log₁₀(I0)")
ax.set_title("Estimated I0 (GenGauss mode) vs resolution")
setup_res_axis(ax)

# (d) β anisotropy: colored by dominant axis
ax = axes[1, 0]
# Compute direction for valid HKLs
g_cart_v = hkl_valid.float() @ recip_mat
g_mag_v = g_cart_v.norm(dim=1).clamp(min=1e-8)
g_hat_v = g_cart_v / g_mag_v[:, None]
cos_a_v = (g_hat_v @ astar_hat).abs()
cos_b_v = (g_hat_v @ bstar_hat).abs()
cos_c_v = (g_hat_v @ cstar_hat).abs()
dom_v = torch.stack([cos_a_v, cos_b_v, cos_c_v], dim=1).argmax(dim=1)

for axis_id, name, color in [(0, "a*", "C0"), (1, "b*", "C1"), (2, "c*", "C2")]:
    m = dom_v == axis_id
    if m.sum() < 50:
        continue
    d_ax = d_perhkl[m]
    beta_ax = beta_perhkl[m]
    rank_ax = d_ax.argsort().argsort()
    bin_ax = (rank_ax.float() / m.sum() * n_res_bins).long().clamp(max=n_res_bins - 1)
    for b in range(n_res_bins):
        bm = bin_ax == b
        if bm.sum() > 5:
            ax.plot(d_to_x(d_ax[bm].median().item()),
                    beta_ax[bm].median().item(), "o", color=color, markersize=3, alpha=0.7)
    ax.plot([], [], "o", color=color, label=f"{name} dom.")
ax.axhline(2, ls=":", color="k", alpha=0.3)
ax.set_ylabel("β (median)")
ax.set_title("β anisotropy by dominant axis")
ax.legend(fontsize=7)
setup_res_axis(ax)

# (e) α anisotropy
ax = axes[1, 1]
for axis_id, name, color in [(0, "a*", "C0"), (1, "b*", "C1"), (2, "c*", "C2")]:
    m = dom_v == axis_id
    if m.sum() < 50:
        continue
    d_ax = d_perhkl[m]
    alpha_ax = alpha_perhkl[m]
    rank_ax = d_ax.argsort().argsort()
    bin_ax = (rank_ax.float() / m.sum() * n_res_bins).long().clamp(max=n_res_bins - 1)
    for b in range(n_res_bins):
        bm = bin_ax == b
        if bm.sum() > 5:
            ax.plot(d_to_x(d_ax[bm].median().item()),
                    alpha_ax[bm].median().item(), "o", color=color, markersize=3, alpha=0.7)
    ax.plot([], [], "o", color=color, label=f"{name} dom.")
ax.set_ylabel("α (median)")
ax.set_title("α anisotropy by dominant axis")
ax.legend(fontsize=7)
setup_res_axis(ax)

# (f) I0 comparison: per-HKL GenGauss mode vs arithmetic mean
ax = axes[1, 2]
# Map valid_indices back to the 'keep' filtered set for comparison
# valid_indices are indices into unique_hkl; mean_I is for all unique_hkl
mean_arith = mean_I[valid_indices]
pos_both = (mean_arith > 0) & (I0_perhkl > 0)
n_sc2 = min(10000, pos_both.sum().item())
idx_sc2 = torch.where(pos_both)[0][torch.randperm(pos_both.sum())[:n_sc2]]
ax.scatter(torch.log10(mean_arith[idx_sc2]).numpy(),
           torch.log10(I0_perhkl[idx_sc2]).numpy(),
           c=d_to_x(d_perhkl[idx_sc2].numpy()), s=1, alpha=0.15,
           cmap="viridis", rasterized=True)
lims = [min(torch.log10(mean_arith[pos_both]).min().item(),
            torch.log10(I0_perhkl[pos_both]).min().item()),
        max(torch.log10(mean_arith[pos_both]).max().item(),
            torch.log10(I0_perhkl[pos_both]).max().item())]
ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="1:1")
ax.set_xlabel("log₁₀(⟨I⟩ arithmetic mean)")
ax.set_ylabel("log₁₀(I0 GenGauss)")
ax.set_title("I0 (GenGauss mode) vs ⟨I⟩")
ax.legend(fontsize=7)
ax.set_aspect("equal")

fig.suptitle("Per-HKL generalized Gaussian fit (adjacency-restrained)", fontsize=13)
fig.tight_layout()
fig.savefig(str(OUTDIR / "perhkl_gengauss.png"), dpi=150)
plt.close(fig)
print(f"Saved {OUTDIR / 'perhkl_gengauss.png'}")
