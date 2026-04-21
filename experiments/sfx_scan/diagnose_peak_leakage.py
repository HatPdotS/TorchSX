#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH -o /das/work/p17/p17490/Peter/torchsx/sfx_scan/diagnose_peak_leakage.out
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 2:00:00
#SBATCH --mem=64G

"""Diagnose whether peak height causes trivial NLL improvement.

Tests:
1. Correlation between log(peak) and log(I_obs)
2. How well log(peak) - f(s²) predicts the optimal c_j = I_obs/I₀
3. Retrain WITHOUT peak/background → does improvement survive?
"""

import sys
import os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "/das/work/p17/p17490/Peter/torchsx")

from pathlib import Path
import math
import torch
import torch.nn as nn

from torchsx.stream import SFXStream
from torchsx.merge._distribution import fit_gengauss
from torchsx.merge._noise import PoissonNoiseModel
from torchsx.merge._neural_model import (
    CrystalTransformerCorrection,
    CorrectionFeatures,
    extract_features,
)

# ── Configuration ──
STREAM_DIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/streams")
DELAYS = [-91, -136, -182, -227, -273, -45]
SPACEGROUP = "P-1"
OUTDIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/neural_correction")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ══════════════════════════════════════════════════════════════
# 1. Load data (same as v2)
# ══════════════════════════════════════════════════════════════
print("── Loading streams ──")
streams = []
for d in DELAYS:
    p = STREAM_DIR / f"delay_{d}.stream"
    print(f"  {p.name} ...", end=" ", flush=True)
    st = SFXStream.from_stream(str(p), spacegroup=SPACEGROUP, n_workers=1)
    print(f"{st.n_crystals} crystals, {st.n_observations} obs")
    streams.append(st)

s = SFXStream.concat(streams)
core = s._core
print(f"\nTotal: {s.n_crystals} crystals, {s.n_observations} observations")

# ══════════════════════════════════════════════════════════════
# 2. Joint fit
# ══════════════════════════════════════════════════════════════
print("\n── Joint fit ──")
noise = PoissonNoiseModel(gain=1.0)
joint_fit = fit_gengauss(
    s, n_em_cycles=1, scale_crystals=False,
    poisson_noise=noise, mode="joint",
    min_obs=20, n_shells=20, n_smooth_iter=15, smooth_lambda=0.3,
    joint_device=DEVICE,
)

# ══════════════════════════════════════════════════════════════
# 3. Map observations to HKLs (same as v2)
# ══════════════════════════════════════════════════════════════
from torchsx.merge._hkl import compute_hkl_index

hkl_idx = compute_hkl_index(core.hkl)
def _hkl_hash(hkl):
    return hkl[:, 0].long() * 1000000 + hkl[:, 1].long() * 1000 + hkl[:, 2].long()

jf_hash = _hkl_hash(joint_fit.hkl)
raw_hash = _hkl_hash(hkl_idx.unique_hkl)
hash_to_jf = {h.item(): i for i, h in enumerate(jf_hash)}

valid_hkl_id = torch.full((core.n_observations,), -1, dtype=torch.long)
for raw_id in range(hkl_idx.n_unique):
    h = raw_hash[raw_id].item()
    if h in hash_to_jf:
        mask = hkl_idx.hkl_id == raw_id
        valid_hkl_id[mask] = hash_to_jf[h]

keep = valid_hkl_id >= 0
N = keep.sum().item()
hkl_id = valid_hkl_id[keep]

I_obs = core.I[keep]
peak = core.peak[keep]
background = core.background[keep]

features = extract_features(core, wavelength=0.9567)
s_sq = features.s_sq[keep]

I0 = joint_fit.I0

# ══════════════════════════════════════════════════════════════
# TEST 1: Correlation between log(peak) and log(I_obs)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TEST 1: Peak ↔ I_obs correlation")
print("="*60)

pos = (peak > 0) & (I_obs > 0)
lp = torch.log(peak[pos])
li = torch.log(I_obs[pos])

# Pearson correlation
lp_c = lp - lp.mean()
li_c = li - li.mean()
r = (lp_c * li_c).sum() / (lp_c.norm() * li_c.norm())
print(f"  Pearson r(log peak, log I_obs): {r:.4f}")

# ══════════════════════════════════════════════════════════════
# TEST 2: How predictable is c_j = I_obs/I₀ from peak + s²?
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TEST 2: Predictability of c_j from peak + resolution")
print("="*60)

# True c_j = I_obs / I₀
I0_per_obs = I0[hkl_id]
pos2 = (I0_per_obs > 0) & (I_obs > 0) & (peak > 0)
log_c_true = torch.log(I_obs[pos2] / I0_per_obs[pos2])
log_peak_pos = torch.log(peak[pos2])
s_sq_pos = s_sq[pos2]

# Variance of true log(c_j)
var_logc = log_c_true.var().item()
print(f"  var(log(c_j)):   {var_logc:.4f}  (std={var_logc**0.5:.3f})")
print(f"  This is the total observation-to-observation variation")

# Decompose: log(c_j) = log(I_obs) - log(I₀)
#                      ≈ log(peak) + noise - log(I₀)
# What fraction of var(log c_j) can peak explain?
lc = log_c_true - log_c_true.mean()
lpk = log_peak_pos - log_peak_pos.mean()
r_peak_c = (lc * lpk).sum() / (lc.norm() * lpk.norm())
print(f"  r(log peak, log c_j):  {r_peak_c:.4f}")
print(f"  R² (peak alone):       {r_peak_c.item()**2:.4f}")

# With resolution binning: predict E[log(I₀) | s²] then subtract
n_bins = 50
s_sq_sorted, sort_idx = s_sq_pos.sort()
log_I0_sorted = torch.log(I0_per_obs[pos2][sort_idx])
bin_size = len(s_sq_sorted) // n_bins
mean_logI0_per_bin = torch.zeros(n_bins)
for b in range(n_bins):
    start = b * bin_size
    end = (b + 1) * bin_size if b < n_bins - 1 else len(s_sq_sorted)
    mean_logI0_per_bin[b] = log_I0_sorted[start:end].mean()

# Simple prediction: log(c_j) ≈ log(peak) - E[log(I₀)|resolution]
# For each obs, find its resolution bin and use that prediction
bin_idx = (s_sq_pos / s_sq_pos.max() * (n_bins - 1)).long().clamp(0, n_bins - 1)
predicted_logI0 = mean_logI0_per_bin[bin_idx]
predicted_logc = log_peak_pos - predicted_logI0
residual = log_c_true - predicted_logc
r_explained = 1.0 - residual.var().item() / var_logc
print(f"\n  With log(peak) - E[log(I₀)|s²]:")
print(f"    R² explained:  {r_explained:.4f}")
print(f"    Residual std:  {residual.std():.3f}")

# What about without peak? Just resolution?
# E[log(c_j) | s²] should be ~0 if crystals are scaled
bin_idx2 = (s_sq_pos / s_sq_pos.max() * (n_bins - 1)).long().clamp(0, n_bins - 1)
log_c_sorted = log_c_true[sort_idx]
mean_logc_per_bin = torch.zeros(n_bins)
for b in range(n_bins):
    start = b * bin_size
    end = (b + 1) * bin_size if b < n_bins - 1 else len(s_sq_sorted)
    mean_logc_per_bin[b] = log_c_sorted[start:end].mean()
predicted_logc_resonly = mean_logc_per_bin[bin_idx2]
residual_resonly = log_c_true - predicted_logc_resonly
r_explained_resonly = 1.0 - residual_resonly.var().item() / var_logc
print(f"\n  With just E[log(c_j) | s²] (resolution-only):")
print(f"    R² explained:  {r_explained_resonly:.4f}")
print(f"    Residual std:  {residual_resonly.std():.3f}")

# ══════════════════════════════════════════════════════════════
# TEST 3: Decompose var(log c_j) into physical sources
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TEST 3: Variance decomposition of log(c_j)")
print("="*60)

# log(c_j) = log(I_obs/I₀) = log(I_obs) - log(I₀)
# This includes: true partiality + random ε + any scale variation
# The ε component has variance = α²Γ(3/β)/Γ(1/β) (GenGauss variance)
alpha_obs = joint_fit.alpha[hkl_id[pos2]]
beta_obs = joint_fit.beta[hkl_id[pos2]]
# GenGauss variance = α²·Γ(3/β)/Γ(1/β)
gg_var = alpha_obs**2 * torch.exp(torch.lgamma(3.0/beta_obs) - torch.lgamma(1.0/beta_obs))
mean_gg_var = gg_var.mean().item()
print(f"  Expected variance from ε (GenGauss):  {mean_gg_var:.4f}  (std={mean_gg_var**0.5:.3f})")
print(f"  Total var(log c_j):                   {var_logc:.4f}  (std={var_logc**0.5:.3f})")
excess = var_logc - mean_gg_var
print(f"  Excess (scale + structure factor):     {excess:.4f}  (std={max(0,excess)**0.5:.3f})")

# ══════════════════════════════════════════════════════════════
# TEST 4: Train model WITHOUT peak and background
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TEST 4: Train WITHOUT peak/background (ablation)")
print("="*60)

# We need a modified model that zeros out peak and background
# Simplest: create features with peak=0, background=0

crystal_id = core.crystal_id[keep].long()
sigma2_poi = noise.sigma2(core.I[keep], core.background[keep]).to(DEVICE)

I_obs_dev = I_obs.to(DEVICE)
hkl_id_dev = hkl_id.to(DEVICE)
sigma2_poi_dev = sigma2_poi

# Work/free mask (same seed as v2)
torch.manual_seed(42)
work_mask = (torch.rand(N) > 0.1).to(DEVICE)
free_mask = ~work_mask

# Sort by crystal
sort_idx_all = crystal_id.argsort(stable=True)
crystal_id_sorted = crystal_id[sort_idx_all].to(DEVICE)
hkl_id_sorted = hkl_id_dev[sort_idx_all.to(DEVICE)]
I_obs_sorted = I_obs_dev[sort_idx_all.to(DEVICE)]
sigma2_poi_sorted = sigma2_poi_dev[sort_idx_all.to(DEVICE)]
work_mask_sorted = work_mask[sort_idx_all.to(DEVICE)]
free_mask_sorted = free_mask[sort_idx_all.to(DEVICE)]

# Sort features
sort_cpu = sort_idx_all.cpu()
feat_all = extract_features(core, wavelength=0.9567)

# Create features with peak/bg ZEROED
feat_sorted_nopeak = CorrectionFeatures(
    peak=torch.zeros(N),  # ZEROED
    background=torch.zeros(N),  # ZEROED
    s_sq=feat_all.s_sq[keep][sort_cpu],
    excitation_error=feat_all.excitation_error[keep][sort_cpu],
    q_lab=feat_all.q_lab[keep][sort_cpu],
    fs=feat_all.fs[keep][sort_cpu],
    ss=feat_all.ss[keep][sort_cpu],
    panel_id=feat_all.panel_id[keep][sort_cpu],
    profile_rad=feat_all.profile_rad[keep][sort_cpu],
    crystal_id=crystal_id[sort_cpu],
)

# Crystal offsets
n_cryst = int(crystal_id_sorted.max().item()) + 1
obs_per_crystal = torch.zeros(n_cryst, dtype=torch.long, device=DEVICE)
obs_per_crystal.scatter_add_(0, crystal_id_sorted,
                              torch.ones(N, dtype=torch.long, device=DEVICE))
crystal_offsets = torch.zeros(n_cryst + 1, dtype=torch.long, device=DEVICE)
crystal_offsets[1:] = obs_per_crystal.cumsum(0)

# Frozen parameters on GPU
I0_dev = joint_fit.I0.to(DEVICE)
alpha_dev = joint_fit.alpha.to(DEVICE)
beta_dev = joint_fit.beta.to(DEVICE)

# ── NLL function (copied from v2 to avoid import issues) ──
N_QUAD = 201

def _sample_gengauss(alpha_: torch.Tensor, beta_: torch.Tensor, K: int) -> torch.Tensor:
    N_ = alpha_.shape[0]
    device_ = alpha_.device
    inv_beta = (1.0 / beta_).unsqueeze(1)
    gamma_dist = torch.distributions.Gamma(inv_beta.expand(N_, K), 1.0)
    g = gamma_dist.sample()
    abs_x = alpha_.unsqueeze(1) * g.pow(inv_beta)
    u = torch.rand(N_, K, device=device_)
    sign = torch.where(u > 0.5, torch.ones_like(u), -torch.ones_like(u))
    return sign * abs_x

def compute_nll(log_c, hkl_id_b, I_obs_b, sigma2_poi_b, mask,
                n_samples=N_QUAD, stochastic=True):
    c = torch.exp(log_c)
    I0_b = I0_dev[hkl_id_b]
    alpha_b = alpha_dev[hkl_id_b]
    beta_b = beta_dev[hkl_id_b]
    if not stochastic:
        torch.manual_seed(0)
    eps_samples = _sample_gengauss(alpha_b, beta_b, n_samples)
    predicted = (c * I0_b).unsqueeze(1) * torch.exp(eps_samples)
    residual = I_obs_b.unsqueeze(1) - predicted
    log_gauss = (-residual ** 2 / (2.0 * sigma2_poi_b.unsqueeze(1))
                 - 0.5 * torch.log(sigma2_poi_b.unsqueeze(1)))
    log_p = torch.logsumexp(log_gauss, dim=1) - math.log(n_samples)
    return -log_p[mask].mean()

# Compute baseline for comparison
print("\n  Computing baseline NLL...")
BATCH_CRYSTALS = 256
baseline_sum, baseline_n = 0.0, 0
with torch.no_grad():
    for ci_start in range(0, n_cryst, BATCH_CRYSTALS):
        ci_end = min(ci_start + BATCH_CRYSTALS, n_cryst)
        s_off = crystal_offsets[ci_start].item()
        e_off = crystal_offsets[ci_end].item()
        if e_off <= s_off:
            continue
        idx = slice(s_off, e_off)
        zero_c = torch.zeros(e_off - s_off, device=DEVICE)
        nll = compute_nll(zero_c, hkl_id_sorted[idx], I_obs_sorted[idx],
                          sigma2_poi_sorted[idx], work_mask_sorted[idx],
                          stochastic=False)
        n_w = work_mask_sorted[idx].sum().item()
        baseline_sum += nll.item() * n_w
        baseline_n += n_w
baseline = baseline_sum / max(baseline_n, 1)
print(f"  Baseline NLL: {baseline:.4f}")

# Train model without peak
print("\n  Training WITHOUT peak/background...")
model = CrystalTransformerCorrection(
    n_panels=32, panel_embed_dim=8,
    d_model=128, n_heads=4, n_layers=2, dropout=0.1,
).to(DEVICE)
print(f"  {model.n_params()} params")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

N_EPOCHS = 30
PATIENCE = 5
best_free = float('inf')
best_state = None
patience_counter = 0

for epoch in range(N_EPOCHS):
    model.train()
    train_loss_sum, train_n = 0.0, 0
    perm = torch.randperm(n_cryst, device=DEVICE)

    for bi in range(0, n_cryst, BATCH_CRYSTALS):
        batch_cids = perm[bi:bi + BATCH_CRYSTALS]
        batch_cids_sorted, _ = batch_cids.sort()

        obs_slices = []
        n_per_crystal = []
        for ci in batch_cids_sorted:
            s_off = crystal_offsets[ci].item()
            e_off = crystal_offsets[ci + 1].item()
            if e_off > s_off:
                obs_slices.append(torch.arange(s_off, e_off, device=DEVICE))
                n_per_crystal.append(e_off - s_off)

        if not obs_slices:
            continue
        obs_idx = torch.cat(obs_slices)
        n_per = torch.tensor(n_per_crystal, device=DEVICE)
        obs_cpu = obs_idx.cpu()

        batch_feat = CorrectionFeatures(
            peak=feat_sorted_nopeak.peak[obs_cpu].to(DEVICE),  # zeros
            background=feat_sorted_nopeak.background[obs_cpu].to(DEVICE),  # zeros
            s_sq=feat_sorted_nopeak.s_sq[obs_cpu].to(DEVICE),
            excitation_error=feat_sorted_nopeak.excitation_error[obs_cpu].to(DEVICE),
            q_lab=feat_sorted_nopeak.q_lab[obs_cpu].to(DEVICE),
            fs=feat_sorted_nopeak.fs[obs_cpu].to(DEVICE),
            ss=feat_sorted_nopeak.ss[obs_cpu].to(DEVICE),
            panel_id=feat_sorted_nopeak.panel_id[obs_cpu].to(DEVICE),
            profile_rad=feat_sorted_nopeak.profile_rad[obs_cpu].to(DEVICE),
            crystal_id=torch.repeat_interleave(
                torch.arange(len(n_per_crystal), device=DEVICE), n_per
            ),
        )

        alpha_obs = alpha_dev[hkl_id_sorted[obs_idx]]
        log_c = model(batch_feat, n_per, alpha_obs)
        nll = compute_nll(log_c, hkl_id_sorted[obs_idx], I_obs_sorted[obs_idx],
                          sigma2_poi_sorted[obs_idx], work_mask_sorted[obs_idx])

        optimizer.zero_grad()
        nll.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        n_w = work_mask_sorted[obs_idx].sum().item()
        train_loss_sum += nll.item() * n_w
        train_n += n_w

    scheduler.step()
    train_nll = train_loss_sum / max(train_n, 1)

    # Free-set evaluation
    model.eval()
    free_sum, free_n = 0.0, 0
    with torch.no_grad():
        for ci_start in range(0, n_cryst, BATCH_CRYSTALS):
            ci_end = min(ci_start + BATCH_CRYSTALS, n_cryst)
            s_off = crystal_offsets[ci_start].item()
            e_off = crystal_offsets[ci_end].item()
            if e_off <= s_off:
                continue

            obs_idx = torch.arange(s_off, e_off, device=DEVICE)
            obs_cpu = obs_idx.cpu()

            cids_chunk = crystal_id_sorted[s_off:e_off]
            unique_cids = cids_chunk.unique()
            remap = torch.zeros(n_cryst, dtype=torch.long, device=DEVICE)
            remap[unique_cids] = torch.arange(len(unique_cids), device=DEVICE)
            cid_local = remap[cids_chunk]
            n_per_eval = torch.zeros(len(unique_cids), dtype=torch.long, device=DEVICE)
            n_per_eval.scatter_add_(0, cid_local,
                                     torch.ones(e_off - s_off, dtype=torch.long, device=DEVICE))

            batch_feat = CorrectionFeatures(
                peak=feat_sorted_nopeak.peak[obs_cpu].to(DEVICE),
                background=feat_sorted_nopeak.background[obs_cpu].to(DEVICE),
                s_sq=feat_sorted_nopeak.s_sq[obs_cpu].to(DEVICE),
                excitation_error=feat_sorted_nopeak.excitation_error[obs_cpu].to(DEVICE),
                q_lab=feat_sorted_nopeak.q_lab[obs_cpu].to(DEVICE),
                fs=feat_sorted_nopeak.fs[obs_cpu].to(DEVICE),
                ss=feat_sorted_nopeak.ss[obs_cpu].to(DEVICE),
                panel_id=feat_sorted_nopeak.panel_id[obs_cpu].to(DEVICE),
                profile_rad=feat_sorted_nopeak.profile_rad[obs_cpu].to(DEVICE),
                crystal_id=cid_local,
            )

            alpha_obs = alpha_dev[hkl_id_sorted[obs_idx]]
            log_c = model(batch_feat, n_per_eval, alpha_obs)
            nll = compute_nll(log_c, hkl_id_sorted[obs_idx], I_obs_sorted[obs_idx],
                              sigma2_poi_sorted[obs_idx], free_mask_sorted[s_off:e_off],
                              stochastic=False)
            n_f = free_mask_sorted[s_off:e_off].sum().item()
            free_sum += nll.item() * n_f
            free_n += n_f

    free_nll = free_sum / max(free_n, 1)

    improved = free_nll < best_free
    if improved:
        best_free = free_nll
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    marker = " *" if improved else ""
    print(f"  Epoch {epoch+1:3d}/{N_EPOCHS}: train={train_nll:.4f}  "
          f"free={free_nll:.4f}{marker}", flush=True)

    if patience_counter >= PATIENCE:
        print(f"  Early stopping (best epoch: {epoch+1 - PATIENCE})")
        break

# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("COMPARISON:")
print(f"  Baseline NLL:              {baseline:.4f}")
print(f"  With peak (v2):            3.4069  (improvement: {baseline - 3.4069:.4f})")
print(f"  Without peak (ablation):   {best_free:.4f}  (improvement: {baseline - best_free:.4f})")
print(f"\n  Peak contribution:  {3.4069 - best_free:+.4f} NLL")
if best_free < baseline:
    print(f"  Non-peak improvement:  {baseline - best_free:.4f} ({(baseline-best_free)/baseline*100:.1f}%)")
    print(f"    → {torch.exp(torch.tensor(baseline - best_free)):.2f}× more probable")
else:
    print(f"  No improvement without peak → physics features alone have no signal")
print(f"{'='*60}")

# Save ablation model
if best_state is not None:
    model.load_state_dict(best_state)
    torch.save({
        "model_state": model.state_dict(),
        "baseline": baseline,
        "best_free": best_free,
        "ablation": "no_peak_no_bg",
    }, str(OUTDIR / "neural_v2_nopeak.pt"))
    print(f"Saved to {OUTDIR / 'neural_v2_nopeak.pt'}")
