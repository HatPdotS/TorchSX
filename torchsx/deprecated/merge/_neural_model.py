"""Standalone neural correction model for SFX observations.

Predicts per-observation multiplicative correction factors from
geometric and measurement metadata. No HKL identifiers are used —
the model can only learn systematic corrections from physics
(excitation error, detector position, peak shape) and measurement
quality (peak height, background).

Features per observation:
    - peak: integrated peak counts (intensity proxy)
    - background: per-pixel background level
    - s²: resolution (1/4d²)
    - excitation_error: Ewald sphere offset (partiality predictor)
    - q_lab (3D): scattering vector in lab frame
    - fs, ss: detector position (sub-panel)
    - panel_id: detector panel (embedded)
    - profile_rad: mosaicity proxy

Architecture: Crystal-context transformer (observations within one
crystal attend to each other) or standalone MLP (per-observation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class CorrectionFeatures:
    """Per-observation features for neural correction.

    All tensors are (N,) except q_lab which is (N, 3).
    """
    peak: Tensor            # integrated peak counts
    background: Tensor      # per-pixel background
    s_sq: Tensor            # resolution 1/(4d²)
    excitation_error: Tensor  # Ewald sphere offset (Å⁻¹)
    q_lab: Tensor           # (N, 3) scattering vector in lab frame
    fs: Tensor              # fast-scan detector position
    ss: Tensor              # slow-scan detector position
    panel_id: Tensor        # (N,) long — panel index
    profile_rad: Tensor     # mosaicity proxy
    crystal_id: Tensor      # (N,) long — for crystal grouping


def compute_excitation_error(
    hkl: Tensor,
    astar: Tensor,
    bstar: Tensor,
    cstar: Tensor,
    crystal_id: Tensor,
    wavelength: float,
) -> tuple[Tensor, Tensor]:
    """Compute excitation error and lab-frame scattering vector per observation.

    Parameters
    ----------
    hkl : (N, 3) int — Miller indices
    astar, bstar, cstar : (N_cryst, 3) float — reciprocal basis vectors in lab frame
    crystal_id : (N,) long — crystal index per observation
    wavelength : float — beam wavelength in Å

    Returns
    -------
    excitation_error : (N,) float — signed distance from Ewald sphere (Å⁻¹)
    q_lab : (N, 3) float — scattering vector in lab frame
    """
    h = hkl[:, 0].float()
    k = hkl[:, 1].float()
    l = hkl[:, 2].float()
    cid = crystal_id.long()

    # q = h*astar + k*bstar + l*cstar  (lab frame)
    q_lab = (h.unsqueeze(1) * astar[cid]
             + k.unsqueeze(1) * bstar[cid]
             + l.unsqueeze(1) * cstar[cid])

    # Incoming beam wavevector: s0 = (0, 0, 1/λ)
    k0 = 1.0 / wavelength
    s0 = torch.zeros_like(q_lab)
    s0[:, 2] = k0

    # Diffracted wavevector: kf = q + s0
    kf = q_lab + s0

    # Excitation error: |kf| - k0
    excitation_error = kf.norm(dim=1) - k0

    return excitation_error, q_lab


def extract_features(core, wavelength: float = 0.9567) -> CorrectionFeatures:
    """Extract all features from an SFXStream core.

    Parameters
    ----------
    core : _SFXStreamCore
    wavelength : beam wavelength in Å (default: 12960 eV → 0.9567 Å)

    Returns
    -------
    CorrectionFeatures
    """
    from torchref.base.reciprocal.hkl import get_d_spacing

    cid = core.crystal_id.long()
    median_cell = core.cell.median(dim=0).values
    d = get_d_spacing(core.hkl.float(), median_cell)
    s_sq = 1.0 / (4.0 * d ** 2)

    exc_err, q_lab = compute_excitation_error(
        core.hkl, core.astar, core.bstar, core.cstar,
        core.crystal_id, wavelength,
    )

    return CorrectionFeatures(
        peak=core.peak,
        background=core.background,
        s_sq=s_sq,
        excitation_error=exc_err,
        q_lab=q_lab,
        fs=core.fs,
        ss=core.ss,
        panel_id=core.panel_id.long(),
        profile_rad=core.profile_rad[cid],
        crystal_id=cid,
    )


class PanelEmbedding(nn.Module):
    """Learned embedding for panel_id."""

    def __init__(self, n_panels: int = 32, embed_dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(n_panels, embed_dim)

    def forward(self, panel_id: Tensor) -> Tensor:
        return self.embed(panel_id)


class CorrectionMLP(nn.Module):
    """Per-observation MLP correction model.

    Input features (per observation):
        peak, background, s², excitation_error, q_lab (3D),
        fs, ss, profile_rad, panel_embed (8D)
        → 9 scalar + 3 vector + 8 embed = 20 dims

    Output: log(c_j) — log-space correction factor
    """

    def __init__(
        self,
        n_panels: int = 32,
        panel_embed_dim: int = 8,
        hidden_dims: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.panel_embed = PanelEmbedding(n_panels, panel_embed_dim)
        # Input: 7 scalar features + 3 q_lab components + panel_embed
        # peak, bg, s², exc_err, fs, ss, profile_rad (7) + q_lab (3) + panel (embed)
        n_input = 7 + 3 + panel_embed_dim

        layers = []
        prev = n_input
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

        # Initialize output near zero
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()

    def forward(self, features: CorrectionFeatures, alpha: Tensor | None = None) -> Tensor:
        """Predict log(c_j) per observation.

        Parameters
        ----------
        features : CorrectionFeatures with all fields on same device
        alpha : (N,) optional — per-obs α for bounding output range

        Returns
        -------
        log_c : (N,) log-space correction
        """
        # Normalize continuous features to ~unit scale
        peak_norm = torch.log1p(features.peak.clamp(min=0))
        bg_norm = torch.log1p(features.background.clamp(min=0))
        s_sq_norm = features.s_sq * 10.0  # scale to ~O(1)
        exc_norm = features.excitation_error  # already Å⁻¹, O(0.1)
        q_norm = features.q_lab * 0.1  # scale down from Å⁻¹
        fs_norm = features.fs / 1000.0  # pixel coords → O(1)
        ss_norm = features.ss / 1000.0
        prof_norm = features.profile_rad * 100.0  # nm⁻¹ → O(1)

        panel_emb = self.panel_embed(features.panel_id)

        x = torch.cat([
            peak_norm.unsqueeze(1),
            bg_norm.unsqueeze(1),
            s_sq_norm.unsqueeze(1),
            exc_norm.unsqueeze(1),
            q_norm,  # (N, 3)
            fs_norm.unsqueeze(1),
            ss_norm.unsqueeze(1),
            prof_norm.unsqueeze(1),
            panel_emb,  # (N, embed_dim)
        ], dim=1)

        raw = self.mlp(x).squeeze(1)  # (N,)

        # Bound to ±4α if provided
        if alpha is not None:
            return torch.tanh(raw) * 4.0 * alpha
        return raw

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class CrystalTransformerCorrection(nn.Module):
    """Crystal-context transformer correction model.

    Each crystal's observations form a sequence. Self-attention
    allows the model to learn crystal-level patterns (e.g., all
    observations on this crystal are systematically weak).

    Input: same features as CorrectionMLP, but grouped by crystal.
    Output: log(c_j) per observation.
    """

    def __init__(
        self,
        n_panels: int = 32,
        panel_embed_dim: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        patch_size: int = 0,
        patch_embed_dim: int = 32,
    ):
        super().__init__()
        self.panel_embed = PanelEmbedding(n_panels, panel_embed_dim)
        self.use_patches = patch_size > 0

        # Optional CNN for ROI patches (max-normalized)
        if self.use_patches:
            self.patch_encoder = nn.Sequential(
                nn.Conv2d(1, 16, 5, padding=2),
                nn.GELU(),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(2),
                nn.Flatten(),
                nn.Linear(32 * 4, patch_embed_dim),
                nn.GELU(),
            )
            n_patch = patch_embed_dim
        else:
            self.patch_encoder = None
            n_patch = 0

        # 7 scalar + 3 q_lab + panel_embed + optional patch_embed
        n_input = 7 + 3 + panel_embed_dim + n_patch

        self.input_proj = nn.Linear(n_input, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # Initialize output near zero
        with torch.no_grad():
            self.output_head[-1].weight.zero_()
            self.output_head[-1].bias.zero_()

    def forward(
        self,
        features: CorrectionFeatures,
        n_per_crystal: Tensor,
        alpha: Tensor | None = None,
        patches: Tensor | None = None,
    ) -> Tensor:
        """Predict log(c_j) with crystal context.

        Parameters
        ----------
        features : CorrectionFeatures — all observations in the batch
        n_per_crystal : (B,) — number of observations per crystal
        alpha : (N,) optional — per-obs α for bounding output
        patches : (N, H, W) optional — max-normalized ROI patches

        Returns
        -------
        log_c : (N,) log-space correction
        """
        # Build input features
        peak_norm = torch.log1p(features.peak.clamp(min=0))
        bg_norm = torch.log1p(features.background.clamp(min=0))
        s_sq_norm = features.s_sq * 10.0
        exc_norm = features.excitation_error
        q_norm = features.q_lab * 0.1
        fs_norm = features.fs / 1000.0
        ss_norm = features.ss / 1000.0
        prof_norm = features.profile_rad * 100.0
        panel_emb = self.panel_embed(features.panel_id)

        parts = [
            peak_norm.unsqueeze(1), bg_norm.unsqueeze(1),
            s_sq_norm.unsqueeze(1), exc_norm.unsqueeze(1),
            q_norm, fs_norm.unsqueeze(1), ss_norm.unsqueeze(1),
            prof_norm.unsqueeze(1), panel_emb,
        ]

        # Optional: ROI patch embedding
        if self.use_patches and patches is not None:
            patch_emb = self.patch_encoder(patches.unsqueeze(1))  # (N, patch_embed_dim)
            parts.append(patch_emb)

        x = torch.cat(parts, dim=1)
        x = self.input_proj(x)  # (N_total, d_model)

        # Pad into (B, max_len, d_model) for transformer
        B = n_per_crystal.shape[0]
        max_len = n_per_crystal.max().item()
        device = x.device

        padded = torch.zeros(B, max_len, x.shape[1], device=device)
        mask = torch.ones(B, max_len, dtype=torch.bool, device=device)  # True = masked

        offset = 0
        for i, n in enumerate(n_per_crystal):
            n = n.item()
            padded[i, :n] = x[offset:offset + n]
            mask[i, :n] = False  # unmask valid positions
            offset += n

        # Transformer with attention mask
        out = self.transformer(padded, src_key_padding_mask=mask)  # (B, max_len, d_model)

        # Unpad back to flat (N_total,)
        raw_list = []
        for i, n in enumerate(n_per_crystal):
            n = n.item()
            raw_list.append(self.output_head(out[i, :n]).squeeze(-1))
        raw = torch.cat(raw_list)  # (N_total,)

        if alpha is not None:
            return torch.tanh(raw) * 4.0 * alpha
        return raw

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
