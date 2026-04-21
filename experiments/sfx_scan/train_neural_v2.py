#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH -o /das/work/p17/p17490/Peter/torchsx/sfx_scan/train_neural_v2.out
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 4:00:00
#SBATCH --mem=64G

"""Neural correction v2: physics-based features (peak, excitation error, q_lab).

No ROIs, no HKL identifiers. The transformer sees each crystal's observations
with their peak heights, Ewald sphere offsets, scattering vectors, and
detector positions. It learns per-observation corrections that maximize
the quadrature NLL.

Features:
    peak, background, s², excitation_error, q_lab (3D),
    fs, ss, panel_id (embedded), profile_rad

Architecture: CrystalTransformerCorrection (276k params)
Loss: quadrature NLL with frozen (I₀, α, β) from joint fit
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
    compute_excitation_error,
)
# Monte Carlo NLL defined below — no quadrature import needed

# ── Configuration ──
STREAM_DIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/streams")
DELAYS = [-91, -136, -182, -227, -273, -45]
SPACEGROUP = "P-1"
OUTDIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/neural_correction")
OUTDIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")

# ══════════════════════════════════════════════════════════════
# 1. Load dark streams
# ══════════════════════════════════════════════════════════════
print("\n── Loading streams ──")
streams = []
for d in DELAYS:
    p = STREAM_DIR / f"delay_{d}.stream"
    print(f"  {p.name} ...", end=" ", flush=True)
    st = SFXStream.from_stream(str(p), spacegroup=SPACEGROUP, n_workers=1)
    print(f"{st.n_crystals} crystals, {st.n_observations} obs")
    streams.append(st)

s = SFXStream.concat(streams)
print(f"\nTotal: {s.n_crystals} crystals, {s.n_observations} observations")

# ══════════════════════════════════════════════════════════════
# 2. Joint fit WITHOUT corrections → frozen (I₀, α, β)
# ══════════════════════════════════════════════════════════════
print("\n── Joint fit (no corrections) ──")
noise = PoissonNoiseModel(gain=1.0)
joint_fit = fit_gengauss(
    s, n_em_cycles=1, scale_crystals=False,
    poisson_noise=noise, mode="joint",
    min_obs=20, n_shells=20, n_smooth_iter=15, smooth_lambda=0.3,
    joint_device=DEVICE,
)
print(f"  {joint_fit.hkl.shape[0]} HKLs, med(α)={joint_fit.alpha.median():.4f}")
print(f"  I₀: min={joint_fit.I0.min():.2f}, med={joint_fit.I0.median():.2f}, "
      f"max={joint_fit.I0.max():.2f}")

# ══════════════════════════════════════════════════════════════
# 3. Extract physics features
# ══════════════════════════════════════════════════════════════
print("\n── Extracting features ──")
core = s._core
features = extract_features(core, wavelength=0.9567)
print(f"  excitation_error: med={features.excitation_error.median():.4f} Å⁻¹")
print(f"  peak: med={features.peak.median():.1f}")
print(f"  q_lab: shape={features.q_lab.shape}")

# ══════════════════════════════════════════════════════════════
# 4. Map observations to valid HKLs
# ══════════════════════════════════════════════════════════════
print("\n── Mapping observations to HKLs ──")
from torchsx.merge._hkl import compute_hkl_index

hkl_idx = compute_hkl_index(core.hkl)
n_valid = joint_fit.hkl.shape[0]

# Build hash-based mapping
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
print(f"  {N}/{core.n_observations} observations mapped to {n_valid} valid HKLs")

# ══════════════════════════════════════════════════════════════
# 5. Prepare data for training
# ══════════════════════════════════════════════════════════════
print("\n── Preparing training data ──")

# Filter to valid observations
hkl_id = valid_hkl_id[keep].to(DEVICE)
crystal_id = core.crystal_id[keep].long()
I_obs = core.I[keep].to(DEVICE)
sigma2_poi = noise.sigma2(core.I[keep], core.background[keep]).to(DEVICE)

# Build CorrectionFeatures for valid obs (stays on CPU, batch-loaded)
feat_valid = CorrectionFeatures(
    peak=features.peak[keep],
    background=features.background[keep],
    s_sq=features.s_sq[keep],
    excitation_error=features.excitation_error[keep],
    q_lab=features.q_lab[keep],
    fs=features.fs[keep],
    ss=features.ss[keep],
    panel_id=features.panel_id[keep],
    profile_rad=features.profile_rad[keep],
    crystal_id=crystal_id,
)

# Work/free mask (10% free)
torch.manual_seed(42)
work_mask = (torch.rand(N) > 0.1).to(DEVICE)
free_mask = ~work_mask
n_work = work_mask.sum().item()
n_free = free_mask.sum().item()
print(f"  Work: {n_work}, Free: {n_free}")

# Sort by crystal for batching
sort_idx = crystal_id.argsort(stable=True)
crystal_id_sorted = crystal_id[sort_idx].to(DEVICE)
hkl_id_sorted = hkl_id[sort_idx.to(DEVICE)]
I_obs_sorted = I_obs[sort_idx.to(DEVICE)]
sigma2_poi_sorted = sigma2_poi[sort_idx.to(DEVICE)]
work_mask_sorted = work_mask[sort_idx.to(DEVICE)]
free_mask_sorted = free_mask[sort_idx.to(DEVICE)]

# Sort features (CPU tensors — use CPU sort index)
sort_cpu = sort_idx.cpu()
feat_sorted = CorrectionFeatures(
    peak=feat_valid.peak[sort_cpu],
    background=feat_valid.background[sort_cpu],
    s_sq=feat_valid.s_sq[sort_cpu],
    excitation_error=feat_valid.excitation_error[sort_cpu],
    q_lab=feat_valid.q_lab[sort_cpu],
    fs=feat_valid.fs[sort_cpu],
    ss=feat_valid.ss[sort_cpu],
    panel_id=feat_valid.panel_id[sort_cpu],
    profile_rad=feat_valid.profile_rad[sort_cpu],
    crystal_id=crystal_id[sort_cpu],
)

# Crystal offsets for batching
n_cryst = int(crystal_id_sorted.max().item()) + 1
obs_per_crystal = torch.zeros(n_cryst, dtype=torch.long, device=DEVICE)
obs_per_crystal.scatter_add_(0, crystal_id_sorted,
                              torch.ones(N, dtype=torch.long, device=DEVICE))
crystal_offsets = torch.zeros(n_cryst + 1, dtype=torch.long, device=DEVICE)
crystal_offsets[1:] = obs_per_crystal.cumsum(0)

# Frozen distribution parameters on GPU
I0 = joint_fit.I0.to(DEVICE)
alpha = joint_fit.alpha.to(DEVICE)
beta = joint_fit.beta.to(DEVICE)

# Quadrature grid
N_QUAD = 201  # ~200 points: Δε̃ ≈ 0.03, fine enough for strong reflections
# Quadrature grid created inside compute_nll (with jitter)

# Load cached ROI patches (CPU, memory-mapped)
ROI_PATH = OUTDIR / "rois.safetensors"
USE_PATCHES = ROI_PATH.exists()
all_patches = None
if USE_PATCHES:
    from torchsx.io._roi_store import ROIStore
    roi_store = ROIStore(str(ROI_PATH))
    raw_patches = roi_store["patches"]  # (N_total, 21, 21) memory-mapped
    # Max-normalize, filter to valid obs, sort by crystal (all on CPU)
    patch_max = raw_patches.amax(dim=(-1, -2), keepdim=True).clamp(min=1.0)
    all_patches_raw = raw_patches / patch_max
    all_patches = all_patches_raw[keep.nonzero(as_tuple=True)[0]]
    all_patches = all_patches[sort_cpu]
    PATCH_SIZE = 21
    print(f"  ROI patches: {all_patches.shape} (CPU, batch-loaded)")
else:
    PATCH_SIZE = 0
    print(f"  No ROI cache found — using scalar features only")

print(f"  {n_cryst} crystals, {N} observations")

# ══════════════════════════════════════════════════════════════
# 6. Quadrature NLL function
# ══════════════════════════════════════════════════════════════
def _sample_gengauss(alpha: torch.Tensor, beta: torch.Tensor, K: int) -> torch.Tensor:
    """Draw K samples from GenGauss(0, α, β) per observation.

    Uses inverse CDF: if U ~ Uniform(0,1), then
        X = α · sign(U - 0.5) · [gammaincinv(1/β, 2|U-0.5|)]^(1/β)

    Parameters
    ----------
    alpha : (N,) scale parameter
    beta : (N,) shape parameter
    K : number of samples

    Returns
    -------
    samples : (N, K) drawn from GenGauss(0, α, β)
    """
    N = alpha.shape[0]
    device = alpha.device
    # Uniform samples in (0, 1), avoid exact 0 and 0.5
    u = torch.rand(N, K, device=device).clamp(1e-7, 1 - 1e-7)
    # |U - 0.5| mapped through inverse incomplete gamma
    abs_u = (2 * (u - 0.5).abs()).clamp(min=1e-7)
    inv_beta = (1.0 / beta).unsqueeze(1)  # (N, 1)
    # gammaincinv(a, p): inverse of regularized lower incomplete gamma
    # torch doesn't have gammaincinv — use scipy-style approximation via
    # the relationship: if X ~ Gamma(1/β, 1), then |Y| = X^(1/β) ~ half-GenGauss
    # Sample Gamma(1/β, 1) via inverse CDF using torch.special.gammainc
    # Actually, use the fact that for GenGauss:
    #   |X/α|^β ~ Gamma(1/β, 1)
    # So: sample g ~ Gamma(1/β, 1), then |X| = α · g^(1/β)
    # Gamma sampling: use torch.distributions
    gamma_dist = torch.distributions.Gamma(inv_beta.expand(N, K), 1.0)
    g = gamma_dist.sample()  # (N, K)
    abs_x = alpha.unsqueeze(1) * g.pow(inv_beta)  # (N, K)
    # Random sign
    sign = torch.where(u > 0.5, torch.ones_like(u), -torch.ones_like(u))
    return sign * abs_x


def compute_nll(log_c, hkl_id_b, I_obs_b, sigma2_poi_b, mask,
                n_samples=N_QUAD, stochastic=True):
    """Monte Carlo NLL: marginal likelihood estimated by sampling from the prior.

    Draws K samples from GenGauss(0, α, β) per observation, evaluates
    the Gaussian likelihood at each sample, and averages.
    No grid, no dead zones, unbiased gradient estimator.

        p(I_obs | c·I₀, α, β, σ²_poi)
            ≈ (1/K) Σ_k N(I_obs; c·I₀·exp(ε_k), σ²_poi)
            where ε_k ~ GenGauss(0, α, β)

    Parameters
    ----------
    log_c : (N,) log correction factors
    hkl_id_b : (N,) HKL indices
    I_obs_b : (N,) observed intensities
    sigma2_poi_b : (N,) Poisson variance per observation
    mask : (N,) bool — which observations to include
    n_samples : number of Monte Carlo samples (default 201)
    stochastic : if True, draw new random samples (training);
                 if False, use fixed seed (evaluation)
    """
    c = torch.exp(log_c)                     # (N,)
    I0_b = I0[hkl_id_b]                      # (N,)
    alpha_b = alpha[hkl_id_b]                # (N,)
    beta_b = beta[hkl_id_b]                  # (N,)

    # Draw samples from GenGauss prior
    if not stochastic:
        torch.manual_seed(0)
    eps_samples = _sample_gengauss(alpha_b, beta_b, n_samples)  # (N, K)

    # Predicted intensity: c · I₀ · exp(ε_k)
    predicted = (c * I0_b).unsqueeze(1) * torch.exp(eps_samples)  # (N, K)

    # Gaussian log-likelihood: log N(I_obs; predicted, σ²_poi)
    residual = I_obs_b.unsqueeze(1) - predicted                    # (N, K)
    log_gauss = (-residual ** 2 / (2.0 * sigma2_poi_b.unsqueeze(1))
                 - 0.5 * torch.log(sigma2_poi_b.unsqueeze(1)))     # (N, K)

    # Monte Carlo estimate: log p ≈ logsumexp(log_gauss) - log(K)
    log_p = torch.logsumexp(log_gauss, dim=1) - math.log(n_samples)  # (N,)

    return -log_p[mask].mean()


# ══════════════════════════════════════════════════════════════
# 7. Baseline NLL (c=1)
# ══════════════════════════════════════════════════════════════
print("\n── Baseline NLL ──")
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
print(f"  Baseline NLL (c=1): {baseline:.4f}")

# ══════════════════════════════════════════════════════════════
# 8. Model + Training
# ══════════════════════════════════════════════════════════════
print("\n── Training CrystalTransformerCorrection ──")
model = CrystalTransformerCorrection(
    n_panels=32, panel_embed_dim=8,
    d_model=128, n_heads=4, n_layers=2, dropout=0.1,
    patch_size=PATCH_SIZE, patch_embed_dim=32,
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

    # Shuffle crystal order
    perm = torch.randperm(n_cryst, device=DEVICE)

    for bi in range(0, n_cryst, BATCH_CRYSTALS):
        batch_cids = perm[bi:bi + BATCH_CRYSTALS]
        batch_cids_sorted, _ = batch_cids.sort()

        # Gather observations for these crystals
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

        # Build features for this batch (load from CPU → GPU)
        batch_feat = CorrectionFeatures(
            peak=feat_sorted.peak[obs_cpu].to(DEVICE),
            background=feat_sorted.background[obs_cpu].to(DEVICE),
            s_sq=feat_sorted.s_sq[obs_cpu].to(DEVICE),
            excitation_error=feat_sorted.excitation_error[obs_cpu].to(DEVICE),
            q_lab=feat_sorted.q_lab[obs_cpu].to(DEVICE),
            fs=feat_sorted.fs[obs_cpu].to(DEVICE),
            ss=feat_sorted.ss[obs_cpu].to(DEVICE),
            panel_id=feat_sorted.panel_id[obs_cpu].to(DEVICE),
            profile_rad=feat_sorted.profile_rad[obs_cpu].to(DEVICE),
            crystal_id=torch.repeat_interleave(
                torch.arange(len(n_per_crystal), device=DEVICE), n_per
            ),
        )

        alpha_obs = alpha[hkl_id_sorted[obs_idx]]
        bp = all_patches[obs_cpu].to(DEVICE) if all_patches is not None else None
        log_c = model(batch_feat, n_per, alpha_obs, patches=bp)

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

            # Crystal grouping for this chunk
            cids_chunk = crystal_id_sorted[s_off:e_off]
            unique_cids = cids_chunk.unique()
            remap = torch.zeros(n_cryst, dtype=torch.long, device=DEVICE)
            remap[unique_cids] = torch.arange(len(unique_cids), device=DEVICE)
            cid_local = remap[cids_chunk]
            n_per_eval = torch.zeros(len(unique_cids), dtype=torch.long, device=DEVICE)
            n_per_eval.scatter_add_(0, cid_local,
                                     torch.ones(e_off - s_off, dtype=torch.long, device=DEVICE))

            batch_feat = CorrectionFeatures(
                peak=feat_sorted.peak[obs_cpu].to(DEVICE),
                background=feat_sorted.background[obs_cpu].to(DEVICE),
                s_sq=feat_sorted.s_sq[obs_cpu].to(DEVICE),
                excitation_error=feat_sorted.excitation_error[obs_cpu].to(DEVICE),
                q_lab=feat_sorted.q_lab[obs_cpu].to(DEVICE),
                fs=feat_sorted.fs[obs_cpu].to(DEVICE),
                ss=feat_sorted.ss[obs_cpu].to(DEVICE),
                panel_id=feat_sorted.panel_id[obs_cpu].to(DEVICE),
                profile_rad=feat_sorted.profile_rad[obs_cpu].to(DEVICE),
                crystal_id=cid_local,
            )

            alpha_obs = alpha[hkl_id_sorted[obs_idx]]
            bp_eval = all_patches[obs_cpu].to(DEVICE) if all_patches is not None else None
            log_c = model(batch_feat, n_per_eval, alpha_obs, patches=bp_eval)

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

# Restore best model
if best_state is not None:
    model.load_state_dict(best_state)

# ══════════════════════════════════════════════════════════════
# 9. Summary
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Results:")
print(f"  Baseline (c=1): {baseline:.4f}")
print(f"  Best free NLL:  {best_free:.4f}")
delta = baseline - best_free
if delta > 0:
    print(f"  ✓ Improvement: {delta:.4f} ({delta/baseline*100:.1f}%)")
    print(f"    Observations are {torch.exp(torch.tensor(delta)):.2f}× more probable")
else:
    print(f"  ✗ No improvement")
print(f"{'='*60}")

# Save
save_path = OUTDIR / "neural_v2.pt"
torch.save({
    "model_state": model.state_dict(),
    "baseline": baseline,
    "best_free": best_free,
}, str(save_path))
print(f"Saved to {save_path}")
