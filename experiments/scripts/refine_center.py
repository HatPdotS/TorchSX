#!/usr/bin/env python
"""Differentiable detector geometry refinement from background.

Jointly optimises per-panel geometry (translation, tilt, gain) and a
piecewise-linear radial background profile R(q) so that the predicted
background matches the observed frame-summed counts.

Uses Poisson NLL with iterative 3-sigma outlier rejection.
Hierarchy: (1) gains only, (2) +translations, (3) +tilts, (4) joint.
"""

from __future__ import annotations

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchsx.geometry import Geometry
from sparse_analysis import (
    load_data,
    STREAM_PATH, H5_PATH,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FRAMES = 50
N_PROFILE_NODES = 100  # piecewise-linear nodes for R(q)
WAVELENGTH_A = 12398.0 / 12960.0  # Å  (photon_energy = 12960 eV)
POLARISATION_FRAC = 0.99  # horizontal polarisation


# ── Detector geometry model ─────────────────────────────────────────────────

class DetectorModel(nn.Module):
    """Differentiable detector geometry + radial background profile.

    R(q) is represented as softplus(raw_values) at fixed q-nodes,
    evaluated via vectorized linear interpolation.
    """

    def __init__(
        self,
        panel_corners: Tensor,   # (n_panels, 2) corner_x, corner_y in pixels
        panel_fs_vecs: Tensor,   # (n_panels, 2) fsx, fsy
        panel_ss_vecs: Tensor,   # (n_panels, 2) ssx, ssy
        panel_coffsets: Tensor,  # (n_panels,) coffset in meters
        pixel_dfs: Tensor,       # (n_good,) local fs coord
        pixel_dss: Tensor,       # (n_good,) local ss coord
        pixel_panel_id: Tensor,  # (n_good,) int, panel index
        clen: float,
        res: float,
        wavelength: float,
        n_profile_nodes: int = 100,
        q_min: float = 0.0,
        q_max: float = 8.0,
    ):
        super().__init__()
        n_panels = panel_corners.shape[0]

        # Fixed pixel-local coordinates
        self.register_buffer("pixel_dfs", pixel_dfs)
        self.register_buffer("pixel_dss", pixel_dss)
        self.register_buffer("pixel_panel_id", pixel_panel_id)

        # Fixed panel base geometry
        self.register_buffer("base_corner_x", panel_corners[:, 0])
        self.register_buffer("base_corner_y", panel_corners[:, 1])
        self.register_buffer("base_fsx", panel_fs_vecs[:, 0])
        self.register_buffer("base_fsy", panel_fs_vecs[:, 1])
        self.register_buffer("base_ssx", panel_ss_vecs[:, 0])
        self.register_buffer("base_ssy", panel_ss_vecs[:, 1])
        self.register_buffer("base_coffset", panel_coffsets)

        self.clen = clen
        self.res = res
        self.wavelength = wavelength

        # Learnable beam center offset (pixels)
        self.center_dx = nn.Parameter(torch.tensor(0.0))
        self.center_dy = nn.Parameter(torch.tensor(0.0))

        # Piecewise-linear radial profile: R(q) = softplus(raw[i]) lerped
        q_nodes = torch.linspace(q_min, q_max, n_profile_nodes)
        self.register_buffer("q_nodes", q_nodes)
        self.profile_raw = nn.Parameter(torch.zeros(n_profile_nodes))

    def lab_positions(self) -> tuple[Tensor, Tensor, Tensor]:
        """Lab-frame (x, y, z) in meters for each good pixel."""
        pid = self.pixel_panel_id.long()

        x_px = (self.base_corner_x[pid] - self.center_dx
                + self.pixel_dfs * self.base_fsx[pid]
                + self.pixel_dss * self.base_ssx[pid])
        y_px = (self.base_corner_y[pid] - self.center_dy
                + self.pixel_dfs * self.base_fsy[pid]
                + self.pixel_dss * self.base_ssy[pid])

        x = x_px / self.res
        y = y_px / self.res
        z = self.clen + self.base_coffset[pid]

        return x, y, z

    def compute_q(self, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        r_perp = torch.sqrt(x ** 2 + y ** 2)
        two_theta = torch.atan2(r_perp, z)
        return (4 * math.pi / self.wavelength) * torch.sin(two_theta / 2)

    def polarisation(self, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        r_perp = torch.sqrt(x ** 2 + y ** 2)
        two_theta = torch.atan2(r_perp, z)
        phi = torch.atan2(y, x)
        sin2_2th = torch.sin(two_theta) ** 2
        f = POLARISATION_FRAC
        return 1.0 - f * torch.cos(phi)**2 * sin2_2th - (1-f) * torch.sin(phi)**2 * sin2_2th

    def solid_angle(self, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        r2 = x**2 + y**2 + z**2
        cos_2th = z / torch.sqrt(r2)
        omega = cos_2th**3 / r2
        return omega / omega.mean()

    def eval_profile(self, q: Tensor) -> Tensor:
        """Vectorized piecewise-linear interpolation of R(q)."""
        q_nodes = self.q_nodes
        n = len(q_nodes)
        dq = q_nodes[1] - q_nodes[0]

        # Continuous index into the node array
        t = (q - q_nodes[0]) / dq
        t = t.clamp(0, n - 1 - 1e-6)
        idx = t.long()
        idx = idx.clamp(0, n - 2)
        frac = t - idx.float()

        vals = F.softplus(self.profile_raw)
        return vals[idx] * (1 - frac) + vals[idx + 1] * frac

    def forward(self) -> Tensor:
        x, y, z = self.lab_positions()
        q = self.compute_q(x, y, z)
        R_q = self.eval_profile(q)
        P = self.polarisation(x, y, z)
        omega = self.solid_angle(x, y, z)
        return P * R_q * omega


# ── Loss + outlier rejection ────────────────────────────────────────────────

def poisson_nll(K: Tensor, N_frames: int, mu: Tensor, mask: Tensor | None = None) -> Tensor:
    N_mu = N_frames * mu.clamp(min=1e-8)
    nll = N_mu - K * torch.log(N_mu)
    if mask is not None:
        nll = nll[mask]
    return nll.mean()


def update_outlier_mask(K: Tensor, N_frames: int, mu: Tensor, sigma_cut: float = 3.0) -> Tensor:
    N_mu = (N_frames * mu.clamp(min=1e-8)).detach()
    z = (K - N_mu) / torch.sqrt(N_mu)
    return z.abs() < sigma_cut


# ── Data loading ────────────────────────────────────────────────────────────

def prepare_data(geom: dict, n_frames: int):
    print(f"Loading {n_frames} frames...")
    t0 = time.time()
    frames, good = load_data(H5_PATH, n_frames)
    print(f"  loaded in {time.time()-t0:.1f}s")
    print(f"  good pixels: {good.sum():,}, mean counts: {frames[:, good].mean():.3f}")

    K = frames[:, good].sum(axis=0).astype(np.float64)
    K_gpu = torch.from_numpy(K).float().to(DEVICE)

    panels = geom.panels
    n_panels = geom.n_panels
    total_ss, total_fs = geom.detector_shape

    panel_id_map = np.full((total_ss, total_fs), -1, dtype=np.int32)
    dfs_map = np.zeros((total_ss, total_fs), dtype=np.float32)
    dss_map = np.zeros((total_ss, total_fs), dtype=np.float32)
    panel_corners = np.zeros((n_panels, 2), dtype=np.float32)
    panel_fs_vecs = np.zeros((n_panels, 2), dtype=np.float32)
    panel_ss_vecs = np.zeros((n_panels, 2), dtype=np.float32)
    panel_coffsets = np.zeros(n_panels, dtype=np.float32)

    for i, p in enumerate(panels):
        ss0, ss1 = p.min_ss, p.max_ss
        fs0, fs1 = p.min_fs, p.max_fs
        panel_id_map[ss0:ss1+1, fs0:fs1+1] = i
        dfs_grid, dss_grid = np.meshgrid(
            np.arange(fs1-fs0+1, dtype=np.float32),
            np.arange(ss1-ss0+1, dtype=np.float32),
        )
        dfs_map[ss0:ss1+1, fs0:fs1+1] = dfs_grid
        dss_map[ss0:ss1+1, fs0:fs1+1] = dss_grid
        panel_corners[i] = [p.corner_x, p.corner_y]
        panel_fs_vecs[i] = [p.fs_vec[0], p.fs_vec[1]]
        panel_ss_vecs[i] = [p.ss_vec[0], p.ss_vec[1]]
        panel_coffsets[i] = p.coffset

    return {
        "K": K_gpu,
        "pixel_dfs": torch.from_numpy(dfs_map[good]).float().to(DEVICE),
        "pixel_dss": torch.from_numpy(dss_map[good]).float().to(DEVICE),
        "pixel_panel_id": torch.from_numpy(panel_id_map[good]).long().to(DEVICE),
        "panel_corners": torch.from_numpy(panel_corners).to(DEVICE),
        "panel_fs_vecs": torch.from_numpy(panel_fs_vecs).to(DEVICE),
        "panel_ss_vecs": torch.from_numpy(panel_ss_vecs).to(DEVICE),
        "panel_coffsets": torch.from_numpy(panel_coffsets).to(DEVICE),
        "n_frames": n_frames,
        "good": good,
    }


# ── Refinement ──────────────────────────────────────────────────────────────

def freeze_all(model):
    for p in model.parameters():
        p.requires_grad_(False)

def unfreeze(params):
    for p in params:
        p.requires_grad_(True)


def run_stage(model, K, n_frames, stage_name, mask=None, n_iter=200, lr=0.01):
    print(f"\n--- Stage: {stage_name} ---")
    active = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in active)
    n_used = mask.sum().item() if mask is not None else K.shape[0]
    print(f"  {n_params} params, {n_used:,} pixels")

    optimizer = torch.optim.LBFGS(active, lr=lr, max_iter=20, line_search_fn="strong_wolfe")
    t0 = time.time()

    losses = []
    for step in range(n_iter):
        def closure():
            optimizer.zero_grad()
            mu = model()
            loss = poisson_nll(K, n_frames, mu, mask)
            loss.backward()
            return loss
        loss = optimizer.step(closure)
        losses.append(loss.item())
        if step % 20 == 0 or step == n_iter - 1:
            print(f"  step {step:4d}: loss = {loss.item():.6f}")
        if len(losses) > 2 and abs(losses[-1] - losses[-2]) < 1e-8:
            print(f"  converged at step {step}")
            break

    dt = time.time() - t0
    print(f"  {dt:.1f}s ({dt/len(losses)*1000:.0f}ms/step)")

    # Outlier rejection
    with torch.no_grad():
        mu = model()
        new_mask = update_outlier_mask(K, n_frames, mu, sigma_cut=3.0)
        if mask is not None:
            new_mask = new_mask & mask
        n_rejected = n_used - new_mask.sum().item()
        print(f"  outliers: {n_rejected:,} removed ({100*n_rejected/n_used:.2f}%)")

    return losses, new_mask


def main():
    print(f"=== Differentiable Geometry Refinement ({DEVICE}) ===\n")

    geom = Geometry.from_stream(STREAM_PATH)
    clen = geom.clen if geom.clen else 0.0939
    res = geom.res if geom.res else 13333.3
    print(f"  {geom.n_panels} panels, clen={clen:.4f} m, res={res:.1f} px/m")
    print(f"  wavelength={WAVELENGTH_A:.4f} A ({12398.0/WAVELENGTH_A:.0f} eV)")

    data = prepare_data(geom, N_FRAMES)

    # Determine q range
    with torch.no_grad():
        # Quick q range computation
        pid = data["pixel_panel_id"]
        cx = data["panel_corners"][:, 0][pid]
        cy = data["panel_corners"][:, 1][pid]
        fsx = data["panel_fs_vecs"][:, 0][pid]
        fsy = data["panel_fs_vecs"][:, 1][pid]
        ssx = data["panel_ss_vecs"][:, 0][pid]
        ssy = data["panel_ss_vecs"][:, 1][pid]
        x_px = cx + data["pixel_dfs"] * fsx + data["pixel_dss"] * ssx
        y_px = cy + data["pixel_dfs"] * fsy + data["pixel_dss"] * ssy
        x = x_px / res
        y = y_px / res
        z = clen + data["panel_coffsets"][pid]
        r_perp = torch.sqrt(x**2 + y**2)
        two_theta = torch.atan2(r_perp, z)
        q = (4 * math.pi / WAVELENGTH_A) * torch.sin(two_theta / 2)
        q_min, q_max = q.min().item() * 0.95, q.max().item() * 1.05
        print(f"\n  q range: {q.min():.3f} - {q.max():.3f} A^-1")

    model = DetectorModel(
        panel_corners=data["panel_corners"],
        panel_fs_vecs=data["panel_fs_vecs"],
        panel_ss_vecs=data["panel_ss_vecs"],
        panel_coffsets=data["panel_coffsets"],
        pixel_dfs=data["pixel_dfs"],
        pixel_dss=data["pixel_dss"],
        pixel_panel_id=data["pixel_panel_id"],
        clen=clen, res=res, wavelength=WAVELENGTH_A,
        n_profile_nodes=N_PROFILE_NODES,
        q_min=q_min, q_max=q_max,
    ).to(DEVICE)

    K = data["K"]
    n_frames = data["n_frames"]

    # Initialise profile from binned data
    with torch.no_grad():
        x, y, z = model.lab_positions()
        q = model.compute_q(x, y, z)
        P = model.polarisation(x, y, z)
        omega = model.solid_angle(x, y, z)
        R_crude = K / (n_frames * P * omega).clamp(min=1e-6)

        # Bin and average
        n_bins = N_PROFILE_NODES
        q_edges = torch.linspace(q_min, q_max, n_bins + 1, device=DEVICE)
        bin_idx = torch.bucketize(q, q_edges[1:]).clamp(0, n_bins - 1)
        R_binned = torch.zeros(n_bins, device=DEVICE)
        counts = torch.zeros(n_bins, device=DEVICE)
        R_binned.scatter_add_(0, bin_idx, R_crude)
        counts.scatter_add_(0, bin_idx, torch.ones_like(R_crude))
        valid = counts > 0
        R_binned[valid] /= counts[valid]
        R_binned[~valid] = R_binned[valid].mean()

        # inverse softplus: raw = log(exp(R) - 1)
        R_binned = R_binned.clamp(min=0.01)
        raw_init = torch.log(torch.exp(R_binned) - 1 + 1e-6)
        model.profile_raw.data.copy_(raw_init)

    mask = None

    # Refine beam center + radial profile
    freeze_all(model)
    unfreeze([model.center_dx, model.center_dy, model.profile_raw])
    losses1, mask = run_stage(model, K, n_frames, "Center + profile", mask=mask, n_iter=100)

    # Re-run with outlier mask
    losses2, mask = run_stage(model, K, n_frames, "Center + profile (clean)", mask=mask, n_iter=100)

    print(f"\n  Final inliers: {mask.sum().item():,} / {K.shape[0]:,}")

    # ── Results ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        dx = model.center_dx.item()
        dy = model.center_dy.item()
    res = geom.get("res", 13333.3)
    print(f"\n=== Results ===")
    print(f"  Center shift: dx={dx:.3f}, dy={dy:.3f} pixels")
    print(f"  In lab frame: dx={dx/res*1e6:.1f} um, dy={dy/res*1e6:.1f} um")

    # ── Plots ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    all_losses = losses1 + losses2
    axes[0].plot(all_losses)
    axes[0].axvline(len(losses1), color="r", linestyle="--", alpha=0.5, label="outlier reject")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Poisson NLL")
    axes[0].set_title("Loss convergence")
    axes[0].legend()

    with torch.no_grad():
        q_plot = torch.linspace(q_min, q_max, 1000, device=DEVICE)
        R_plot = model.eval_profile(q_plot)
    axes[1].plot(q_plot.cpu(), R_plot.cpu())
    axes[1].set_xlabel("q (A^-1)")
    axes[1].set_ylabel("R(q)")
    axes[1].set_title("Refined radial profile")

    with torch.no_grad():
        mu = model()
        residual = (K / n_frames - mu) / mu.clamp(min=1e-6)
        q_vals = model.compute_q(*model.lab_positions())

    n_bins = 200
    q_edges_plot = torch.linspace(q_min, q_max, n_bins + 1, device=DEVICE)
    bi = torch.bucketize(q_vals, q_edges_plot[1:]).clamp(0, n_bins - 1)
    res_b = torch.zeros(n_bins, device=DEVICE)
    cnt_b = torch.zeros(n_bins, device=DEVICE)
    res_b.scatter_add_(0, bi, residual)
    cnt_b.scatter_add_(0, bi, torch.ones_like(residual))
    v = cnt_b > 0
    res_b[v] /= cnt_b[v]
    bc = 0.5 * (q_edges_plot[:-1] + q_edges_plot[1:])

    axes[2].plot(bc[v].cpu(), res_b[v].cpu(), linewidth=0.5)
    axes[2].axhline(0, color="k", linewidth=0.5)
    axes[2].set_xlabel("q (A^-1)")
    axes[2].set_ylabel("(obs - pred) / pred")
    axes[2].set_title("Fractional residual vs q")

    fig.suptitle("Differentiable geometry refinement", fontsize=14)
    fig.tight_layout()
    fig.savefig("geometry_refinement.png", dpi=150)
    print("\nSaved geometry_refinement.png")


if __name__ == "__main__":
    main()
