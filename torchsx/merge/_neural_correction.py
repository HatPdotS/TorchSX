"""Neural correction network for SFX variance reduction.

A crystal-context transformer that predicts per-observation multiplicative
correction factors from binary-thresholded ROI shapes and geometric metadata.

Architecture:
    Per observation: CNN(binary 21×21 → 32-dim) ∥ metadata(~10) → 42-dim token
    Per crystal: Transformer encoder (2 layers, 4 heads, d_model=64)
    Per observation output: Linear(64 → 1) → log(c_j), bounded via tanh

The correction factor c_j multiplies the predicted signal intensity:
    predicted_jk = c_j × I₀_h × exp(α_h × ε̃_k)

Poisson noise σ²_poi is unchanged (measured, not scaled).

Intensity obfuscation:
    - ROI patches are binary-thresholded (shape only, no photon counts)
    - I_obs is NOT passed as input
    - Identifying features (h,k,l, panel, position, resolution) ARE passed
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PatchEncoder(nn.Module):
    """CNN encoder for binary-thresholded ROI patches.

    Input: (B, 1, H, W) binary float tensor (0 or 1).
    Output: (B, embed_dim) patch embedding.
    """

    def __init__(self, patch_size: int = 21, embed_dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2),  # → (B, 16, 11, 11)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # → (B, 32, 6, 6)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),  # → (B, 32, 2, 2)
        )
        self.fc = nn.Linear(32 * 4, embed_dim)

    def forward(self, patches: Tensor) -> Tensor:
        """patches: (B, H, W) or (B, 1, H, W) binary float."""
        if patches.dim() == 3:
            patches = patches.unsqueeze(1)
        x = self.conv(patches)
        return self.fc(x.flatten(1))


class NeuralCorrectionModel(nn.Module):
    """Crystal-context transformer for per-observation correction factors.

    Each crystal's observations form a sequence. The transformer attends
    within the crystal to learn cross-observation patterns (detector
    position effects, systematic shifts).

    Parameters
    ----------
    n_metadata : int
        Number of scalar metadata features per observation.
    patch_embed_dim : int
        CNN output dimension for ROI patches.
    d_model : int
        Transformer hidden dimension.
    n_heads : int
        Attention heads.
    n_layers : int
        Transformer encoder layers.
    dropout : float
        Dropout rate in transformer and output head.
    max_log_c : float
        Bound on log(c) output via tanh scaling.
    """

    def __init__(
        self,
        n_metadata: int = 10,
        patch_embed_dim: int = 32,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_log_c: float = 2.0,
        patch_size: int = 21,
    ):
        super().__init__()
        self.max_log_c = max_log_c

        # Patch encoder
        self.patch_encoder = PatchEncoder(patch_size, patch_embed_dim)

        # Project concatenated features to d_model
        self.input_proj = nn.Linear(patch_embed_dim + n_metadata, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Output head: predict log(c)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        patches: Tensor,
        metadata: Tensor,
        crystal_id: Tensor,
        n_per_crystal: Tensor,
        alpha_per_obs: Tensor,
    ) -> Tensor:
        """Predict log(c_j) for each observation.

        Parameters
        ----------
        patches : (B, H, W) float — scaled ROI (normalized by max pixel)
        metadata : (B, n_metadata) float — scalar features
        crystal_id : (B,) long — crystal index per observation
        n_per_crystal : (n_crystals_in_batch,) long — obs count per crystal
        alpha_per_obs : (B,) float — per-HKL α for output scaling

        Returns
        -------
        log_c : (B,) — log correction factor, bounded to ±4α
        """
        B = patches.shape[0]
        n_crystals = n_per_crystal.shape[0]
        max_len = int(n_per_crystal.max().item())

        # Encode patches and concatenate metadata
        patch_emb = self.patch_encoder(patches)  # (B, patch_embed_dim)
        tokens = self.input_proj(
            torch.cat([patch_emb, metadata], dim=-1)
        )  # (B, d_model)

        # Pack into (n_crystals, max_len, d_model) with padding
        packed = torch.zeros(
            n_crystals, max_len, tokens.shape[-1],
            device=tokens.device, dtype=tokens.dtype,
        )
        # Build attention mask: True = ignore (padding)
        attn_mask = torch.ones(
            n_crystals, max_len, device=tokens.device, dtype=torch.bool,
        )

        # Fill packed tensor and mask
        offsets = torch.zeros(n_crystals + 1, dtype=torch.long,
                              device=n_per_crystal.device)
        offsets[1:] = n_per_crystal.cumsum(0)
        for c in range(n_crystals):
            n = int(n_per_crystal[c].item())
            start = int(offsets[c].item())
            packed[c, :n] = tokens[start:start + n]
            attn_mask[c, :n] = False  # real tokens are NOT masked

        # Transformer: attend within each crystal
        out = self.transformer(
            packed,
            src_key_padding_mask=attn_mask,
        )  # (n_crystals, max_len, d_model)

        # Unpack back to (B, d_model)
        unpacked = torch.zeros(B, out.shape[-1],
                               device=out.device, dtype=out.dtype)
        for c in range(n_crystals):
            n = int(n_per_crystal[c].item())
            start = int(offsets[c].item())
            unpacked[start:start + n] = out[c, :n]

        # Output: bounded log(c) in units of α (covers ±4σ of GenGauss)
        raw = self.output_head(unpacked).squeeze(-1)  # (B,)
        log_c = 4.0 * alpha_per_obs * torch.tanh(raw)

        return log_c


@dataclass
class CrystalBatchData:
    """Pre-processed batch ready for the neural correction model."""

    patches: Tensor              # (B, H, W) float scaled [0, 1]
    metadata: Tensor            # (B, n_features) float
    crystal_id: Tensor          # (B,) long
    n_per_crystal: Tensor       # (n_crystals,) long
    I_obs: Tensor               # (B,) float — for loss computation only
    sigma2_poi: Tensor          # (B,) float — Poisson variance
    hkl_id: Tensor              # (B,) long — index into merged HKL array
    work_mask: Tensor           # (B,) bool — True for training, False for free set
    obs_idx: Tensor             # (B,) long — global observation index

    def to(self, device: torch.device) -> CrystalBatchData:
        return CrystalBatchData(
            **{k: v.to(device) if isinstance(v, Tensor) else v
               for k, v in self.__dict__.items()}
        )


def prepare_crystal_batch(
    batch,  # ROIBatch from ROIBatchLoader
    core,   # _SFXStreamCore
    hkl_index,
    noise_model,
    work_mask_global: Tensor,
    raw_to_valid: Tensor | None = None,  # (n_unique,) mapping raw→valid HKL id, -1 if invalid
    threshold: float = 0.5,
) -> CrystalBatchData:
    """Convert ROIBatch to CrystalBatchData for neural correction.

    Thresholds patches to binary, computes metadata features,
    groups by crystal, computes per-crystal counts.
    """
    obs_idx = batch.obs_idx
    B = obs_idx.shape[0]

    # Binary threshold ROI patches
    binary_patches = (batch.patches > threshold).float()

    # Scalar metadata (NO intensity, NO background level)
    hkl = batch.hkl.float()  # (B, 3)
    panel = batch.panel_id.float().unsqueeze(-1)  # (B, 1)
    fs = batch.fs_center.unsqueeze(-1)  # (B, 1)
    ss = batch.ss_center.unsqueeze(-1)  # (B, 1)

    # Resolution: s² = 1/(4d²)
    # Compute from hkl and median cell
    from torchref.base.reciprocal.hkl import get_d_spacing
    median_cell = core.cell.median(dim=0).values
    d = get_d_spacing(batch.hkl.float(), median_cell)
    s_sq = (1.0 / (4.0 * d ** 2)).unsqueeze(-1)  # (B, 1)

    # Excitation error (if orientation available)
    # For now: use profile_radius as proxy for crystal quality
    cid = batch.crystal_id.long()
    prof_rad = core.profile_rad[cid].unsqueeze(-1)  # (B, 1)

    metadata = torch.cat([hkl, panel, fs, ss, s_sq, prof_rad], dim=-1)
    # → (B, 3+1+1+1+1+1) = (B, 8)

    # Group by crystal
    unique_crystals, inverse = torch.unique(cid, return_inverse=True)
    n_per_crystal = torch.bincount(inverse, minlength=unique_crystals.shape[0])

    # Sort by crystal for packing
    sort_order = inverse.argsort(stable=True)

    # Poisson variance
    I_raw = core.I[obs_idx]
    bg = core.background[obs_idx]
    sigma2_poi = noise_model.sigma2(I_raw, bg)

    # Work mask
    work = work_mask_global[obs_idx]

    # Remap HKL indices to valid set (from joint fit)
    raw_hkl_id = batch.hkl_id  # (B,) raw indices into hkl_index.unique_hkl
    if raw_to_valid is not None:
        valid_hkl_id = raw_to_valid[raw_hkl_id]  # (B,) remapped, -1 if invalid
        keep = valid_hkl_id >= 0
        if keep.sum() == 0:
            # Return empty batch
            return CrystalBatchData(
                binary_patches=binary_patches[:0],
                metadata=metadata[:0],
                crystal_id=inverse[:0],
                n_per_crystal=torch.zeros(0, dtype=torch.long),
                I_obs=I_raw[:0],
                sigma2_poi=sigma2_poi[:0],
                hkl_id=torch.zeros(0, dtype=torch.long),
                work_mask=work[:0],
                obs_idx=obs_idx[:0],
            )
        # Filter to valid observations
        binary_patches = binary_patches[keep]
        metadata = metadata[keep]
        I_raw = I_raw[keep]
        sigma2_poi = sigma2_poi[keep]
        work = work[keep]
        obs_idx = obs_idx[keep]
        valid_hkl_id = valid_hkl_id[keep]

        # Recompute crystal grouping after filtering
        cid_filtered = batch.crystal_id.long()[keep]
        unique_crystals, inverse = torch.unique(cid_filtered, return_inverse=True)
        n_per_crystal = torch.bincount(inverse, minlength=unique_crystals.shape[0])
        sort_order = inverse.argsort(stable=True)
    else:
        valid_hkl_id = raw_hkl_id
        sort_order = inverse.argsort(stable=True)

    return CrystalBatchData(
        binary_patches=binary_patches[sort_order],
        metadata=metadata[sort_order],
        crystal_id=inverse[sort_order],
        n_per_crystal=n_per_crystal,
        I_obs=I_raw[sort_order],
        sigma2_poi=sigma2_poi[sort_order],
        hkl_id=valid_hkl_id[sort_order],
        work_mask=work[sort_order],
        obs_idx=obs_idx[sort_order],
    )
