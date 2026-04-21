#!/usr/bin/env python
"""Refine geometry for a single crystal against centroided spot positions.

Strategy: minimize residual between predicted lab positions (from UB @ hkl)
and observed lab positions (from stream fs/ss → pixel_to_lab), refining:
  - Global detector shift (dx, dy) in meters
  - Global detector distance offset (dz) in meters
  - Small UB rotation (3 Rodrigues angles)

Only uses strong spots (high peak intensity).
"""

import torch
import numpy as np
from torchsx.stream import SFXStream
from torchsx.forward.coords import RefinableGeometry

HC_EV_A = 12398.4

SFX_PATH = "merged_368.sfx"


def skew_matrix(angles: torch.Tensor) -> torch.Tensor:
    """Small-angle rotation matrix R ≈ I + [w]× (differentiable at zero)."""
    wx, wy, wz = angles[0], angles[1], angles[2]
    K = torch.stack([
        torch.stack([torch.zeros_like(wx), -wz, wy]),
        torch.stack([wz, torch.zeros_like(wx), -wx]),
        torch.stack([-wy, wx, torch.zeros_like(wx)]),
    ])
    return torch.eye(3, dtype=angles.dtype) + K


def predict_lab(UB, hkl, wavelength, clen):
    """Predict lab positions (meters) from UB, hkl, wavelength, clen."""
    h_lab = (UB @ hkl.T).T  # (N, 3) in Å⁻¹
    s_out = wavelength * h_lab
    s_out[:, 2] += 1.0
    s_out = s_out / s_out.norm(dim=-1, keepdim=True)
    t = clen / s_out[:, 2]
    x = t * s_out[:, 0]
    y = t * s_out[:, 1]
    return torch.stack([x, y], dim=-1)  # (N, 2) meters


def main():
    print("Loading .sfx...")
    stream = SFXStream.load(SFX_PATH, merge_friedel=False)
    core = stream._core
    geom = core.geometry
    print(f"  {core.n_crystals} crystals, {core.n_observations} obs")

    # Check units — print a few astar values
    print(f"\n  astar[0] = {core.astar[0]} (should be ~0.01-0.1 in Å⁻¹)")
    print(f"  astar norm = {core.astar[0].norm():.6f} Å⁻¹")

    # Find crystal with most reflections
    best_cid = core.n_refl.argmax().item()
    n = core.n_refl[best_cid].item()
    print(f"  Best crystal: {best_cid} with {n} reflections")

    # Try a few crystals and find one with small initial error
    print("\nScanning for crystal with good UB (small initial error)...")
    # Sample crystals with many reflections
    n_refl = core.n_refl.numpy()
    candidates = np.argsort(-n_refl)[:200]  # top 200 by n_refl

    best_err = 1e9
    for cid_try in candidates:
        cid_try = int(cid_try)
        mask = core.crystal_id == cid_try
        if mask.sum() < 50:
            continue
        hkl_t = core.hkl[mask].float()
        fs_t = core.fs[mask].float()
        ss_t = core.ss[mask].float()
        pid_t = core.panel_id[mask]

        img = core.image_id[cid_try].item()
        en = core.beam_energy[img].item()
        wl = HC_EV_A / en
        cl = core.camera_length[img].item() if core.camera_length is not None else 0
        if cl <= 0:
            cl = geom.clen

        UB_t = torch.stack([core.astar[cid_try], core.bstar[cid_try], core.cstar[cid_try]], dim=-1).float()
        with torch.no_grad():
            rg = RefinableGeometry(geom)
            rg._clen.fill_(cl)
            obs_xy = rg.pixel_to_lab(fs_t, ss_t, pid_t)[:, :2]
            pred_xy = predict_lab(UB_t, hkl_t, wl, cl)
            err = (pred_xy - obs_xy).norm(dim=-1) * geom.res
            med_err = err.median().item()

        if med_err < best_err:
            best_err = med_err
            best_cid = cid_try
            print(f"  cid={cid_try} n_refl={n_refl[cid_try]} median_err={med_err:.1f}px")
        if best_err < 20:
            break

    print(f"\n  Selected crystal {best_cid} (median_err={best_err:.1f}px, n_refl={n_refl[best_cid]})")

    obs_mask = core.crystal_id == best_cid
    hkl = core.hkl[obs_mask].float()
    fs = core.fs[obs_mask].float()
    ss = core.ss[obs_mask].float()
    panel_id = core.panel_id[obs_mask]
    I_obs = core.I[obs_mask].float()
    peak = core.peak[obs_mask].float()

    # Filter to strong spots by peak intensity (top quartile)
    peak_thresh = peak.quantile(0.5)
    strong = peak > peak_thresh
    print(f"  Strong spots (peak > {peak_thresh:.0f}): {strong.sum().item()}/{len(peak)}")

    hkl = hkl[strong]
    fs = fs[strong]
    ss = ss[strong]
    panel_id = panel_id[strong]
    I_obs = I_obs[strong]
    peak = peak[strong]
    N = len(hkl)

    # Get crystal parameters
    img_id = core.image_id[best_cid].item()
    energy_eV = core.beam_energy[img_id].item()
    wavelength = HC_EV_A / energy_eV
    clen_val = core.camera_length[img_id].item() if core.camera_length is not None else 0
    if clen_val <= 0:
        clen_val = geom.clen

    astar = core.astar[best_cid].float()
    bstar = core.bstar[best_cid].float()
    cstar = core.cstar[best_cid].float()
    UB0 = torch.stack([astar, bstar, cstar], dim=-1)  # (3, 3)

    # Observed lab positions from stream fs/ss
    ref_geom = RefinableGeometry(geom)
    ref_geom._clen.fill_(clen_val)
    with torch.no_grad():
        obs_xyz = ref_geom.pixel_to_lab(fs, ss, panel_id)  # (N, 3) meters
    obs_xy = obs_xyz[:, :2]  # (N, 2)

    # Remove outliers — only keep spots already close (< 15px)
    with torch.no_grad():
        pred_xy_init = predict_lab(UB0, hkl, wavelength, clen_val)
        err_init = (pred_xy_init - obs_xy).norm(dim=-1) * geom.res
    inlier = err_init < 10.0
    print(f"  Inliers (< 10px): {inlier.sum().item()}/{N}")

    hkl = hkl[inlier]
    fs = fs[inlier]
    ss = ss[inlier]
    panel_id = panel_id[inlier]
    obs_xy = obs_xy[inlier]
    N = len(hkl)

    # Initial prediction
    with torch.no_grad():
        pred_xy0 = predict_lab(UB0, hkl, wavelength, clen_val)
    err0 = (pred_xy0 - obs_xy).norm(dim=-1)
    err0_px = err0 * geom.res
    print(f"\n--- Before refinement ({N} spots) ---")
    print(f"  Median error: {err0_px.median():.2f} px")
    print(f"  Mean error:   {err0_px.mean():.2f} px")
    print(f"  Max error:    {err0_px.max():.2f} px")

    # Set up refinement parameters
    rot_angles = torch.zeros(3, requires_grad=True)  # Rodrigues rotation of UB
    det_shift = torch.zeros(2, requires_grad=True)    # detector x,y shift (meters)
    det_dz = torch.zeros(1, requires_grad=True)       # detector distance offset (meters)

    optimizer = torch.optim.Adam([
        {"params": [rot_angles], "lr": 1e-4},    # radians
        {"params": [det_shift], "lr": 1e-5},      # meters
        {"params": [det_dz], "lr": 1e-5},          # meters
    ])

    print(f"\nRefining {N} strong spots with Adam...")
    for step in range(3000):
        optimizer.zero_grad()
        R = skew_matrix(rot_angles)
        UB = R @ UB0
        clen_eff = clen_val + det_dz[0]
        pred_xy = predict_lab(UB, hkl, wavelength, clen_eff)
        pred_xy = pred_xy + det_shift.unsqueeze(0)
        residuals = (pred_xy - obs_xy).norm(dim=-1)
        loss = residuals.pow(2).mean()
        loss.backward()
        optimizer.step()

        if step % 300 == 0 or step == 2999:
            err_px = residuals.detach() * geom.res
            print(f"  step {step:4d}  loss={loss.item():.4e}  "
                  f"median={err_px.median():.2f}px  mean={err_px.mean():.2f}px  "
                  f"max={err_px.max():.2f}px  "
                  f"rot={rot_angles.detach().numpy()*1e3} mrad")

    # Final results
    with torch.no_grad():
        R = skew_matrix(rot_angles)
        UB_final = R @ UB0
        clen_eff = clen_val + det_dz[0].item()
        pred_xy_final = predict_lab(UB_final, hkl, wavelength, clen_eff) + det_shift.unsqueeze(0)
        err_final = (pred_xy_final - obs_xy).norm(dim=-1) * geom.res

    print(f"\n--- After refinement ---")
    print(f"  Median error: {err_final.median():.2f} px")
    print(f"  Mean error:   {err_final.mean():.2f} px")
    print(f"  Max error:    {err_final.max():.2f} px")
    print(f"\n  Detector shift: dx={det_shift[0].item()*1e6:.1f} µm, "
          f"dy={det_shift[1].item()*1e6:.1f} µm")
    print(f"  Detector dz:    {det_dz[0].item()*1e6:.1f} µm")
    print(f"  UB rotation:    {rot_angles.detach().numpy() * 1e3} mrad")

    # Per-panel statistics
    print(f"\n--- Per-panel residuals (after refinement) ---")
    for pi in range(geom.n_panels):
        mask = panel_id == pi
        if mask.sum() == 0:
            continue
        err_p = err_final[mask]
        print(f"  {geom.panels[pi].name:>4s}: n={mask.sum():3d}  "
              f"median={err_p.median():.2f}  mean={err_p.mean():.2f}  max={err_p.max():.2f} px")

    # Histogram of errors
    print(f"\n--- Error distribution (px) ---")
    for threshold in [1, 2, 3, 5, 10, 20, 50]:
        frac = (err_final < threshold).float().mean() * 100
        print(f"  < {threshold:2d} px: {frac:.1f}%")


if __name__ == "__main__":
    main()
