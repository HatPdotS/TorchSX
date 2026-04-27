"""Training loop for the neural correction network.

Trains the crystal-context transformer to predict per-observation correction
factors that maximize the quadrature NLL (same loss as the joint optimizer).

I₀ and α are frozen from the joint fit.  The net only learns correction
factors from binary ROI shapes and geometric metadata.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from ._dstep import _quadrature_nll, _quadrature_range
from ._neural_correction import (
    NeuralCorrectionModel,
    CrystalBatchData,
    prepare_crystal_batch,
)
from ._noise import PoissonNoiseModel


@dataclass
class TrainingConfig:
    """Configuration for neural correction training."""

    n_epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    n_crystals_per_batch: int = 512
    patience: int = 5           # early stopping patience (epochs)
    n_quad: int = 21            # quadrature points
    device: str = "cpu"
    # Model architecture
    d_model: int = 128          # transformer hidden dim
    n_layers: int = 3           # transformer layers
    n_heads: int = 4            # attention heads
    patch_embed_dim: int = 32   # CNN output dim
    dropout: float = 0.1


@dataclass
class TrainingLog:
    """Training history."""

    train_nll: list[float] = field(default_factory=list)
    free_nll: list[float] = field(default_factory=list)
    best_epoch: int = 0
    stopped_early: bool = False


def _compute_correction_nll(
    log_c: Tensor,
    batch: CrystalBatchData,
    I0: Tensor,
    alpha: Tensor,
    beta: Tensor,
    eps_tilde: Tensor,
    mask: Tensor,
) -> Tensor:
    """Quadrature NLL with neural correction applied.

    Parameters
    ----------
    log_c : (B,) predicted log correction factor
    batch : CrystalBatchData with I_obs, sigma2_poi, hkl_id
    I0, alpha, beta : (n_valid,) frozen distribution parameters
    eps_tilde : (K,) quadrature grid
    mask : (B,) bool — which observations to include in loss

    Returns
    -------
    scalar NLL
    """
    c = torch.exp(log_c)  # already bounded by ±4α via tanh in model

    # Per-observation distribution parameters
    hkl_id = batch.hkl_id
    I0_po = I0[hkl_id]          # (B,)
    alpha_po = alpha[hkl_id]    # (B,)
    beta_po = beta[hkl_id]      # (B,)

    # Corrected predicted intensity: c_j × I₀_h × exp(α_h × ε̃_k)
    # We modify I₀ by the correction factor
    I0_corrected = c * I0_po    # (B,)

    # Pre-compute Gaussian constants
    sigma2_poi = batch.sigma2_poi
    inv_2s2 = 0.5 / sigma2_poi.clamp(min=1e-6)
    log_norm_gauss = -0.5 * torch.log(2.0 * math.pi * sigma2_poi.clamp(min=1e-6))

    # GenGauss weights (alpha-independent in standardised coordinates)
    n_quad = eps_tilde.shape[0]
    d_eps = float(eps_tilde[1] - eps_tilde[0]) if n_quad > 1 else 1.0
    log_gg_norm = (
        torch.log(beta_po) - math.log(2.0)
        - torch.lgamma(1.0 / beta_po) + math.log(d_eps)
    )

    # Quadrature loop
    log_joints = []
    for k in range(n_quad):
        et = float(eps_tilde[k])
        s_k = torch.exp(alpha_po * et)
        predicted = I0_corrected * s_k
        residual = batch.I_obs - predicted
        log_gauss = -(residual ** 2) * inv_2s2 + log_norm_gauss
        log_gg = -(abs(et) ** beta_po) + log_gg_norm
        log_joints.append(log_gauss + log_gg)

    log_joints_stack = torch.stack(log_joints, dim=0)  # (K, B)
    log_p = torch.logsumexp(log_joints_stack, dim=0)   # (B,)

    # NLL on masked observations only
    return -log_p[mask].sum() / mask.sum().clamp(min=1).float()


def train_correction(
    stream,
    roi_store,
    joint_fit,
    noise_model: PoissonNoiseModel | None = None,
    config: TrainingConfig | None = None,
) -> tuple[NeuralCorrectionModel, TrainingLog]:
    """Train the neural correction network.

    Parameters
    ----------
    stream : SFXStream
        Full dataset (all crystals).
    roi_store : ROIStore
        Pre-excised ROI patches (21×21).
    joint_fit : HKLDistribution
        Frozen (I₀, α, β) from joint optimizer.
    noise_model : PoissonNoiseModel, optional
        For σ²_poi computation. Default: gain=1.
    config : TrainingConfig, optional

    Returns
    -------
    model : NeuralCorrectionModel
    log : TrainingLog
    """
    if config is None:
        config = TrainingConfig()
    if noise_model is None:
        noise_model = PoissonNoiseModel(gain=1.0)

    device = torch.device(config.device)
    core = stream._core

    # Frozen distribution parameters
    I0 = joint_fit.I0.to(device)
    alpha = joint_fit.alpha.to(device)
    beta = joint_fit.beta.to(device)

    # Quadrature grid (adaptive range for heavy-tailed β)
    R = _quadrature_range(float(beta.median()))
    n_quad_eff = max(config.n_quad, round(config.n_quad * R / 3.0))
    eps_tilde = torch.linspace(-R, R, n_quad_eff, device=device)

    # HKL index — remap raw indices to joint fit's valid HKL set
    from ..merge._hkl import compute_hkl_index
    hkl_index = compute_hkl_index(core.hkl)

    # Build mapping: raw_hkl_id → valid_index (or -1 if not in joint fit)
    # Match by comparing HKL triplets
    valid_hkl = joint_fit.hkl  # (n_valid, 3) int16
    raw_hkl = hkl_index.unique_hkl  # (n_unique, 3) int16
    n_valid = valid_hkl.shape[0]
    n_unique = raw_hkl.shape[0]

    # Build lookup: for each raw HKL, find its index in valid_hkl (or -1)
    raw_to_valid = torch.full((n_unique,), -1, dtype=torch.long)
    # Hash-based matching
    def _hkl_hash(hkl):
        return hkl[:, 0].long() * 1000000 + hkl[:, 1].long() * 1000 + hkl[:, 2].long()
    valid_hash = _hkl_hash(valid_hkl)
    raw_hash = _hkl_hash(raw_hkl)
    hash_to_valid = {h.item(): i for i, h in enumerate(valid_hash)}
    for j, h in enumerate(raw_hash):
        vi = hash_to_valid.get(h.item(), -1)
        if vi >= 0:
            raw_to_valid[j] = vi
    n_mapped = (raw_to_valid >= 0).sum().item()
    print(f"  HKL mapping: {n_mapped}/{n_unique} raw HKLs mapped to {n_valid} valid HKLs")

    # Per-observation valid mask (only observations of valid HKLs)
    obs_valid_hkl = raw_to_valid[hkl_index.hkl_id] >= 0  # (N_obs,)
    print(f"  Observations with valid HKL: {obs_valid_hkl.sum().item()}/{core.n_observations}")

    # ── Preload ALL data on GPU (4.8 GB on 40 GB A100) ──
    print("  Preloading data on GPU...", flush=True)

    # Remap HKL indices for all observations
    valid_hkl_id_all = raw_to_valid[hkl_index.hkl_id]  # (N_obs,) -1 if invalid
    keep = valid_hkl_id_all >= 0
    keep_idx = keep.nonzero(as_tuple=True)[0]
    N = keep_idx.shape[0]

    # Scale ROI patches: normalize by max pixel per patch (preserves shape, removes intensity)
    raw_patches = roi_store['patches'][keep_idx].to(dtype=torch.float32)
    patch_max = raw_patches.flatten(1).amax(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1.0)
    all_patches = raw_patches / patch_max  # (N, 21, 21) in [0, 1], stays on CPU
    print(f"  Patches: {all_patches.shape}, {all_patches.element_size() * all_patches.nelement() / 1e9:.1f} GB (CPU, batch-loaded to GPU)")

    # Scalar metadata
    from torchref.base.reciprocal.hkl import get_d_spacing
    median_cell = core.cell.median(dim=0).values
    hkl_float = core.hkl[keep_idx].float()
    d_all = get_d_spacing(hkl_float, median_cell)
    s_sq_all = 1.0 / (4.0 * d_all ** 2)

    cid_all = core.crystal_id[keep_idx].long()
    # metadata_all built after rank computation (needs I_obs on device)

    I_obs_all = core.I[keep_idx].to(device)
    sigma2_poi_all = noise_model.sigma2(core.I[keep_idx], core.background[keep_idx]).to(device)
    hkl_id_all = valid_hkl_id_all[keep_idx].to(device)
    crystal_id_all = cid_all.to(device)

    # Per-crystal rank: rank(I_obs within same crystal) / (n_obs_per_crystal - 1) → [0, 1]
    # Encodes "within this crystal, this reflection is the Nth strongest"
    # Does NOT leak per-HKL intensity information
    print("  Computing per-crystal rank (CPU)...", end=" ", flush=True)
    cid_cpu = cid_all.cpu()
    I_cpu = I_obs_all.cpu()
    n_cryst = int(cid_cpu.max().item()) + 1
    sort_idx = torch.argsort(cid_cpu * (I_cpu.max() - I_cpu.min() + 1) + I_cpu)
    sorted_cid = cid_cpu[sort_idx]
    obs_per_cryst = torch.zeros(n_cryst, dtype=torch.long)
    obs_per_cryst.scatter_add_(0, cid_cpu, torch.ones(N, dtype=torch.long))
    offsets = torch.zeros(n_cryst + 1, dtype=torch.long)
    offsets[1:] = obs_per_cryst.cumsum(0)
    group_offset = offsets[sorted_cid]
    within_pos = (torch.arange(N) - group_offset).float()
    n_per = obs_per_cryst[sorted_cid].float().clamp(min=2) - 1
    rank_sorted = (within_pos / n_per).clamp(0, 1)
    rank_all = torch.zeros(N)
    rank_all[sort_idx] = rank_sorted
    rank_all = rank_all.to(device)
    print(f"med={rank_all.median():.3f}")

    # Build metadata (including rank)
    metadata_all = torch.stack([
        hkl_float[:, 0].to(device), hkl_float[:, 1].to(device), hkl_float[:, 2].to(device),
        core.panel_id[keep_idx].float().to(device),
        core.fs[keep_idx].to(device),
        core.ss[keep_idx].to(device),
        s_sq_all.to(device),
        core.profile_rad[cid_all].to(device),
        rank_all,  # already on device
    ], dim=-1)  # (N, 9)

    # Work/free mask (10% free per HKL, same seed as joint optimizer)
    torch.manual_seed(42)
    work_mask_all = (torch.rand(core.n_observations) > 0.1)[keep_idx].to(device)
    free_mask_all = ~work_mask_all

    # Pre-sort by crystal for batching
    sort_by_crystal = crystal_id_all.argsort(stable=True)
    sort_cpu = sort_by_crystal.cpu()  # for indexing CPU tensors
    all_patches = all_patches[sort_cpu]
    metadata_all = metadata_all[sort_by_crystal]
    I_obs_all = I_obs_all[sort_by_crystal]
    sigma2_poi_all = sigma2_poi_all[sort_by_crystal]
    hkl_id_all = hkl_id_all[sort_by_crystal]
    crystal_id_all = crystal_id_all[sort_by_crystal]
    work_mask_all = work_mask_all[sort_by_crystal]
    free_mask_all = free_mask_all[sort_by_crystal]

    # Crystal boundaries (for batching)
    unique_crystals, counts_per_crystal = torch.unique_consecutive(
        crystal_id_all, return_counts=True
    )
    crystal_offsets = torch.zeros(unique_crystals.shape[0] + 1, dtype=torch.long, device=device)
    crystal_offsets[1:] = counts_per_crystal.cumsum(0)
    n_crystals_total = unique_crystals.shape[0]

    print(f"  {N:,} valid observations on GPU, {n_crystals_total:,} crystals")
    if device.type == "cuda":
        print(f"  GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB allocated")

    # ── Model ──
    n_metadata = 9  # h, k, l, panel_id, fs, ss, s², profile_rad, rank
    model = NeuralCorrectionModel(
        n_metadata=n_metadata,
        patch_embed_dim=config.patch_embed_dim,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    log = TrainingLog()
    best_free_nll = float("inf")
    patience_counter = 0

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Training neural correction: {n_params:,} params")
    print(f"  {N:,} observations, {core.n_crystals:,} crystals")
    print(f"  batch: {config.n_crystals_per_batch} crystals", flush=True)

    # ── Baseline NLL (no correction, log_c = 0 everywhere) ──
    model.eval()
    with torch.no_grad():
        baseline_nll_sum, baseline_n = 0.0, 0
        zero_log_c = torch.zeros(N, device=device)
        # Process in chunks to avoid OOM
        chunk = 100000
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            idx = slice(start, end)
            cbatch = CrystalBatchData(
                patches=all_patches[idx].to(device), metadata=metadata_all[idx],
                crystal_id=cid_all[idx], n_per_crystal=torch.tensor([end - start]),
                I_obs=I_obs_all[idx], sigma2_poi=sigma2_poi_all[idx],
                hkl_id=hkl_id_all[idx], work_mask=work_mask_all[idx],
                obs_idx=torch.arange(start, end, device=device),
            )
            loss = _compute_correction_nll(
                zero_log_c[idx], cbatch, I0, alpha, beta, eps_tilde,
                mask=work_mask_all[idx],
            )
            n_work = work_mask_all[idx].sum().item()
            baseline_nll_sum += loss.item() * n_work
            baseline_n += n_work
        baseline = baseline_nll_sum / max(baseline_n, 1)
        print(f"  Baseline NLL (c=1): {baseline:.4f}", flush=True)

    for epoch in range(config.n_epochs):
        t0 = time.time()
        # ── Training ──
        model.train()
        train_nll_sum = 0.0
        train_n = 0

        # Shuffle crystal order each epoch
        perm = torch.randperm(n_crystals_total, device=device)

        for batch_start in range(0, n_crystals_total, config.n_crystals_per_batch):
            batch_cryst = perm[batch_start:batch_start + config.n_crystals_per_batch]

            # Gather observations for these crystals
            obs_slices = []
            n_per_crystal = []
            for ci in batch_cryst:
                s, e = crystal_offsets[ci].item(), crystal_offsets[ci + 1].item()
                obs_slices.append(torch.arange(s, e, device=device))
                n_per_crystal.append(e - s)
            obs_idx = torch.cat(obs_slices)
            n_per_crystal_t = torch.tensor(n_per_crystal, device=device)

            # Batch data — patches from CPU, rest on GPU
            bp = all_patches[obs_idx.cpu()].to(device)
            md = metadata_all[obs_idx]
            I_obs_b = I_obs_all[obs_idx]
            s2p_b = sigma2_poi_all[obs_idx]
            hid_b = hkl_id_all[obs_idx]
            wm_b = work_mask_all[obs_idx]

            # Renumber crystal_id within batch (0, 0, ..., 1, 1, ..., ...)
            cid_batch = torch.repeat_interleave(
                torch.arange(len(n_per_crystal), device=device), n_per_crystal_t
            )

            optimizer.zero_grad()
            alpha_obs = alpha[hid_b]  # per-obs α from joint fit
            log_c = model(bp, md, cid_batch, n_per_crystal_t, alpha_obs)

            # Build batch data for NLL
            cbatch = CrystalBatchData(
                patches=bp, metadata=md,
                crystal_id=cid_batch, n_per_crystal=n_per_crystal_t,
                I_obs=I_obs_b, sigma2_poi=s2p_b,
                hkl_id=hid_b, work_mask=wm_b,
                obs_idx=obs_idx,
            )
            loss = _compute_correction_nll(
                log_c, cbatch, I0, alpha, beta, eps_tilde,
                mask=wm_b,
            )
            loss.backward()
            optimizer.step()

            train_nll_sum += loss.item() * wm_b.sum().item()
            train_n += wm_b.sum().item()

        avg_train = train_nll_sum / max(train_n, 1)
        log.train_nll.append(avg_train)

        # ── Free set evaluation (same GPU data, no data loading) ──
        model.eval()
        free_nll_sum = 0.0
        free_n = 0

        with torch.no_grad():
            for batch_start in range(0, n_crystals_total, config.n_crystals_per_batch):
                batch_cryst = torch.arange(
                    batch_start,
                    min(batch_start + config.n_crystals_per_batch, n_crystals_total),
                    device=device,
                )
                obs_slices = []
                n_per_crystal_eval = []
                for ci in batch_cryst:
                    s, e = crystal_offsets[ci].item(), crystal_offsets[ci + 1].item()
                    obs_slices.append(torch.arange(s, e, device=device))
                    n_per_crystal_eval.append(e - s)
                obs_idx = torch.cat(obs_slices)
                n_per_crystal_t = torch.tensor(n_per_crystal_eval, device=device)

                fm_b = free_mask_all[obs_idx]
                if fm_b.sum() == 0:
                    continue

                cid_batch = torch.repeat_interleave(
                    torch.arange(len(n_per_crystal_eval), device=device), n_per_crystal_t
                )
                alpha_obs = alpha[hkl_id_all[obs_idx]]
                bp_eval = all_patches[obs_idx.cpu()].to(device)
                log_c = model(bp_eval, metadata_all[obs_idx],
                              cid_batch, n_per_crystal_t, alpha_obs)

                cbatch = CrystalBatchData(
                    patches=bp_eval,
                    metadata=metadata_all[obs_idx],
                    crystal_id=cid_batch, n_per_crystal=n_per_crystal_t,
                    I_obs=I_obs_all[obs_idx], sigma2_poi=sigma2_poi_all[obs_idx],
                    hkl_id=hkl_id_all[obs_idx], work_mask=work_mask_all[obs_idx],
                    obs_idx=obs_idx,
                )
                loss = _compute_correction_nll(
                    log_c, cbatch, I0, alpha, beta, eps_tilde,
                    mask=fm_b,
                )
                free_nll_sum += loss.item() * fm_b.sum().item()
                free_n += fm_b.sum().item()

        avg_free = free_nll_sum / max(free_n, 1)
        log.free_nll.append(avg_free)
        dt = time.time() - t0

        print(
            f"  Epoch {epoch+1:3d}/{config.n_epochs}: "
            f"train={avg_train:.4f}  free={avg_free:.4f}  "
            f"({dt:.1f}s)"
            f"{'  *' if avg_free < best_free_nll else ''}", flush=True
        )

        # ── Early stopping ──
        if avg_free < best_free_nll:
            best_free_nll = avg_free
            log.best_epoch = epoch + 1
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"  Early stopping at epoch {epoch+1} "
                      f"(best: {log.best_epoch})")
                log.stopped_early = True
                break

    # Restore best model
    if hasattr(best_state, '__len__'):
        model.load_state_dict(best_state)

    return model, log


def compute_corrections(
    model: torch.nn.Module,
    stream,
    roi_store,
    joint_fit,
    noise_model: PoissonNoiseModel,
    config: TrainingConfig,
) -> torch.Tensor:
    """Run trained model on all observations, return log_c in stream order.

    Returns
    -------
    log_c : (N_obs,) tensor — log-space correction per observation.
        In the same order as stream._core.I (stream order).
    """
    from torchref.base.reciprocal.hkl import get_d_spacing

    device = config.device
    core = stream._core
    N = core.n_observations

    # Same HKL mapping as in train_correction
    from torchsx.deprecated.merge._hkl import compute_hkl_index
    hkl_idx = compute_hkl_index(core.hkl)
    valid_set = set()
    for i in range(joint_fit.hkl.shape[0]):
        h, k, l = joint_fit.hkl[i].tolist()
        valid_set.add((h, k, l))
    valid_hkl_id = torch.full((N,), -1, dtype=torch.long)
    for raw_id in range(hkl_idx.n_unique):
        h, k, l = hkl_idx.unique_hkl[raw_id].tolist()
        if (h, k, l) in valid_set:
            mask = hkl_idx.hkl_id == raw_id
            # Find the index in joint_fit
            for jf_id in range(joint_fit.hkl.shape[0]):
                if joint_fit.hkl[jf_id].tolist() == [h, k, l]:
                    valid_hkl_id[mask] = jf_id
                    break
    keep_idx = valid_hkl_id >= 0

    # Build metadata + patches
    median_cell = core.cell.median(dim=0).values.unsqueeze(0)
    hkl_float = core.hkl.float()
    d_all = get_d_spacing(hkl_float, median_cell)
    s_sq_all = 1.0 / (4.0 * d_all ** 2)

    # ROI patches
    all_patches = roi_store["patches"]

    # Build rank feature
    I_obs = core.I.to(device)
    hkl_id = valid_hkl_id.to(device)
    n_valid = joint_fit.hkl.shape[0]

    sort_idx = torch.argsort(hkl_id * 1e12 + I_obs)
    sorted_hkl = hkl_id[sort_idx]
    obs_per_hkl = torch.zeros(n_valid, device=device, dtype=torch.long)
    obs_per_hkl.scatter_add_(0, hkl_id[keep_idx].to(device),
                              torch.ones(keep_idx.sum(), device=device, dtype=torch.long))
    offsets = torch.zeros(n_valid + 1, device=device, dtype=torch.long)
    offsets[1:] = obs_per_hkl.cumsum(0)
    group_offset = offsets[sorted_hkl.clamp(min=0)]
    global_pos = torch.arange(N, device=device)
    within_pos = (global_pos - group_offset).float()
    n_per = obs_per_hkl[sorted_hkl.clamp(min=0)].float().clamp(min=2) - 1
    rank_sorted = (within_pos / n_per).clamp(0, 1)
    rank_all = torch.zeros(N, device=device)
    rank_all[sort_idx] = rank_sorted

    # Metadata
    metadata = torch.stack([
        hkl_float[:, 0].to(device), hkl_float[:, 1].to(device),
        hkl_float[:, 2].to(device),
        core.panel_id.float().to(device),
        core.fs.to(device), core.ss.to(device),
        s_sq_all.to(device),
        core.profile_rad[core.crystal_id.long()].to(device),
        rank_all,
    ], dim=-1)

    # α per observation for correction range
    alpha_per_obs = torch.zeros(N, device=device)
    alpha_per_obs[keep_idx] = joint_fit.alpha[valid_hkl_id[keep_idx]].to(device)

    # Run model in batches by crystal
    model.eval()
    log_c = torch.zeros(N, device=device)
    crystal_ids = core.crystal_id.long()

    n_cryst = crystal_ids.max().item() + 1
    batch_size = config.n_crystals_per_batch

    with torch.no_grad():
        for start in range(0, n_cryst, batch_size):
            end = min(start + batch_size, n_cryst)
            mask = (crystal_ids >= start) & (crystal_ids < end)
            if mask.sum() == 0:
                continue

            idx = mask.nonzero(as_tuple=True)[0]
            batch_patches = all_patches[idx].to(device)
            # Max-normalize
            pmax = batch_patches.amax(dim=(-1, -2), keepdim=True).clamp(min=1.0)
            batch_patches = batch_patches / pmax

            batch_meta = metadata[idx]
            batch_cid = crystal_ids[idx].to(device) - start
            batch_alpha = alpha_per_obs[idx]

            raw = model(batch_patches.unsqueeze(1), batch_meta, batch_cid)
            batch_log_c = torch.tanh(raw.squeeze(-1)) * 4.0 * batch_alpha
            log_c[idx] = batch_log_c

    return log_c.cpu()
