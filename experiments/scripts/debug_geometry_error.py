#!/usr/bin/env python
"""Diagnose geometry error pattern between our predictions and CrystFEL's.

For a single crystal, compare predicted vs CrystFEL positions for all HKLs
and look for systematic geometric patterns (rotation, scale, shear, etc.).
"""

import numpy as np
import torch

from torchsx import SFXStream
from torchsx.forward.coords import RefinableGeometry
from torchsx.scaling._orientation import CrystalOrientation

HC_EV_A = 12398.4


def main():
    print("Loading merged_368.sfx...")
    stream = SFXStream.load("merged_368.sfx", merge_friedel=False)
    core = stream._core
    geom = core.geometry

    # Pick crystal 700 (used in previous investigation)
    cid = 700
    img_id = core.image_id[cid].item()
    energy_eV = core.beam_energy[img_id].item()
    wavelength = HC_EV_A / energy_eV
    clen_stream = core.camera_length[img_id].item()
    D = clen_stream if clen_stream > 0 else geom.clen

    print(f"Crystal {cid}, image {img_id}")
    print(f"  wavelength={wavelength:.6f} Å, clen={D:.6f} m, geom.clen={geom.clen:.6f} m")
    print(f"  det_shift_x={core.det_shift_x[img_id].item():.4f} mm, "
          f"det_shift_y={core.det_shift_y[img_id].item():.4f} mm")

    # Get CrystFEL's positions
    obs_mask = core.crystal_id == cid
    cf_hkl = core.hkl[obs_mask].numpy()
    cf_fs = core.fs[obs_mask].numpy()
    cf_ss = core.ss[obs_mask].numpy()
    cf_panel = core.panel_id[obs_mask].numpy()
    n_obs = cf_hkl.shape[0]
    print(f"  {n_obs} reflections from CrystFEL")

    # Build UB matrix
    astar = core.astar[cid]
    bstar = core.bstar[cid]
    cstar = core.cstar[cid]
    UB = torch.stack([astar, bstar, cstar], dim=-1).unsqueeze(0)  # (1, 3, 3)

    orient = CrystalOrientation(UB)
    ref_geom = RefinableGeometry(geom)
    if D != geom.clen:
        ref_geom._clen.fill_(D)

    # Compute our predictions for each CrystFEL HKL
    hkl_t = torch.from_numpy(cf_hkl).float()
    h_lab = (UB[0] @ hkl_t.T).T  # (N, 3) in Å⁻¹

    with torch.no_grad():
        our_fs, our_ss, our_pid = ref_geom.recip_to_pixel(h_lab, wavelength)

    our_fs = our_fs.numpy()
    our_ss = our_ss.numpy()
    our_pid = our_pid.numpy()

    # Filter to spots where both land on detector
    on_det = our_pid >= 0
    print(f"  {on_det.sum()} spots on detector in our prediction")

    # Compute errors
    dfs = our_fs[on_det] - cf_fs[on_det]
    dss = our_ss[on_det] - cf_ss[on_det]
    err = np.sqrt(dfs**2 + dss**2)

    print(f"\nPosition errors (all {on_det.sum()} on-detector spots):")
    print(f"  median: {np.median(err):.2f} px")
    print(f"  mean:   {np.mean(err):.2f} px")
    print(f"  max:    {np.max(err):.2f} px")
    print(f"  <1px:   {(err < 1).sum()}/{len(err)}")
    print(f"  <3px:   {(err < 3).sum()}/{len(err)}")

    # Analyze error pattern by panel
    print("\nPer-panel errors:")
    panels_used = np.unique(cf_panel[on_det])
    panel_data = []
    for pid in sorted(panels_used):
        pmask = (cf_panel[on_det] == pid)
        if pmask.sum() == 0:
            continue
        p_err = err[pmask]
        p_dfs = dfs[pmask]
        p_dss = dss[pmask]
        p = geom.panels[pid]
        # Panel center in pixels from beam center
        cx = p.corner_x + p.n_fs/2 * p.fs_vec[0] + p.n_ss/2 * p.ss_vec[0]
        cy = p.corner_y + p.n_fs/2 * p.fs_vec[1] + p.n_ss/2 * p.ss_vec[1]
        dist_px = np.sqrt(cx**2 + cy**2)
        print(f"  {p.name:>4s}: n={pmask.sum():3d}  median={np.median(p_err):6.2f}px  "
              f"mean_dfs={np.mean(p_dfs):+7.2f}  mean_dss={np.mean(p_dss):+7.2f}  "
              f"dist={dist_px:.0f}px")
        panel_data.append((p.name, dist_px, np.median(p_err), np.mean(p_dfs), np.mean(p_dss)))

    # Check if det_shift explains the constant offset
    ds_x_mm = core.det_shift_x[img_id].item()
    ds_y_mm = core.det_shift_y[img_id].item()
    # Convert mm to pixels: shift in mm / (1000/res) = shift_mm * res / 1000
    ds_x_px = ds_x_mm * geom.res / 1000.0
    ds_y_px = ds_y_mm * geom.res / 1000.0
    print(f"\ndet_shift: x={ds_x_mm:.4f} mm ({ds_x_px:.2f} px), "
          f"y={ds_y_mm:.4f} mm ({ds_y_px:.2f} px)")

    # What if we apply det_shift as a corner offset to all panels?
    # In CrystFEL, det_shift shifts all panel corners
    # This means: all x_lab shift by ds_x_mm/1000 m, all y_lab shift by ds_y_mm/1000 m
    # In pixel units: corner_x += ds_x_px, corner_y += ds_y_px
    # But wait — CrystFEL's det_shift is the shift of the detector, so spots move
    # in the opposite direction? Or same direction?
    # CrystFEL: "predict_refine/det_shift x = ... y = ... mm"
    # From CrystFEL source: the shift is applied to the beam center position,
    # which means all pixel positions shift by +det_shift in (x,y)

    # Let me try both signs
    for sign, label in [(1, "same"), (-1, "opposite")]:
        shifted_dfs = dfs - sign * ds_x_px  # Approximate: det_shift affects x → fs mapping
        shifted_dss = dss - sign * ds_y_px
        shifted_err = np.sqrt(shifted_dfs**2 + shifted_dss**2)
        print(f"  After {label}-sign det_shift correction: "
              f"median={np.median(shifted_err):.2f} px")

    # Now let's look at the error as a function of lab position
    # Convert CrystFEL positions to lab frame
    cf_ss_t = torch.from_numpy(cf_ss[on_det].astype(np.float64))
    cf_fs_t = torch.from_numpy(cf_fs[on_det].astype(np.float64))
    xyz = geom.to_lab(cf_ss_t, cf_fs_t)  # (N, 3) meters
    x_m = xyz[:, 0].numpy()
    y_m = xyz[:, 1].numpy()
    z_m = xyz[:, 2].numpy()

    # 2theta
    r_xy = np.sqrt(x_m**2 + y_m**2)
    twotheta = np.arctan2(r_xy, z_m)

    # Check if error scales with 2theta
    corr_2theta = np.corrcoef(twotheta, err)[0, 1]
    print(f"\nCorrelation of error with 2θ: {corr_2theta:.4f}")

    # Check if error vector is radial (pointing away from beam center)
    # Radial direction in pixel space: (cf_fs - beam_fs, cf_ss - beam_ss) normalized
    # In lab frame it's cleaner
    r_hat_x = x_m / (r_xy + 1e-10)
    r_hat_y = y_m / (r_xy + 1e-10)

    # Error direction in lab frame
    # dfs and dss are in pixel units. We need to convert to lab frame.
    # For simplicity, use the fact that for most panels, fs ≈ x and ss ≈ y
    # But actually let's compute properly using pixel_to_lab

    # Our lab positions
    our_ss_t = torch.from_numpy(our_ss[on_det].astype(np.float64))
    our_fs_t = torch.from_numpy(our_fs[on_det].astype(np.float64))
    our_xyz = geom.to_lab(our_ss_t, our_fs_t)
    dx_m = our_xyz[:, 0].numpy() - x_m
    dy_m = our_xyz[:, 1].numpy() - y_m

    # Radial component of error
    d_radial = dx_m * r_hat_x + dy_m * r_hat_y
    d_tangential = -dx_m * r_hat_y + dy_m * r_hat_x

    print(f"\nError decomposition (lab frame, meters):")
    print(f"  Radial component:     mean={np.mean(d_radial)*1e6:+.1f} µm, "
          f"std={np.std(d_radial)*1e6:.1f} µm")
    print(f"  Tangential component: mean={np.mean(d_tangential)*1e6:+.1f} µm, "
          f"std={np.std(d_tangential)*1e6:.1f} µm")

    # Check if radial error scales with tan(2theta) — characteristic of wrong clen
    tan_2theta = np.tan(twotheta)
    corr_radial_tan2theta = np.corrcoef(tan_2theta, d_radial)[0, 1]
    print(f"  Correlation: radial error vs tan(2θ): {corr_radial_tan2theta:.4f}")

    # If error is purely radial and proportional to tan(2theta), it's a clen error
    # d_radial ≈ Δclen * tan(2θ) / cos(2θ) ≈ Δclen * sin(2θ) / cos²(2θ)
    # Actually for small Δclen: dx ≈ -Δclen * x/z, so d_radial ≈ -Δclen * r_xy / z
    # Let's fit: d_radial = alpha * r_xy/z
    A = (r_xy / z_m).reshape(-1, 1)
    alpha = np.linalg.lstsq(A, d_radial, rcond=None)[0][0]
    print(f"  Best-fit Δclen (radial): {alpha*1000:.4f} mm ({alpha:.6f} m)")
    print(f"  → adjusted clen would be: {D + alpha:.6f} m")

    residual = d_radial - alpha * r_xy / z_m
    print(f"  After clen correction: radial residual std={np.std(residual)*1e6:.1f} µm")

    # Also fit d_tangential = beta * r_xy/z (rotation around beam axis)
    beta = np.linalg.lstsq(A, d_tangential, rcond=None)[0][0]
    print(f"  Best-fit rotation component: {beta:.6f} m → "
          f"{np.degrees(np.arctan(beta/D))*3600:.1f} arcsec")

    # Full 2D fit: error = Δclen * ∂pos/∂clen + Δbeam_x * ∂pos/∂beam_x + ...
    # Let's try a general affine model:
    # (dx, dy) = M @ (x, y) + (tx, ty)
    # where M is a 2x2 matrix (captures rotation, scale, shear)
    xy = np.column_stack([x_m, y_m])  # (N, 2)
    dxy = np.column_stack([dx_m, dy_m])  # (N, 2)
    # Design matrix: [x, y, 1]
    A_full = np.column_stack([x_m, y_m, np.ones(len(x_m))])
    # Fit dx = a11*x + a12*y + tx
    params_x = np.linalg.lstsq(A_full, dx_m, rcond=None)[0]
    params_y = np.linalg.lstsq(A_full, dy_m, rcond=None)[0]

    print(f"\nAffine error model:")
    print(f"  dx = {params_x[0]:+.6f}*x + {params_x[1]:+.6f}*y + {params_x[2]*1e6:+.1f} µm")
    print(f"  dy = {params_y[0]:+.6f}*x + {params_y[1]:+.6f}*y + {params_y[2]*1e6:+.1f} µm")

    # The affine matrix M:
    M = np.array([[params_x[0], params_x[1]],
                  [params_y[0], params_y[1]]])
    print(f"  M = [[{M[0,0]:+.6f}, {M[0,1]:+.6f}],")
    print(f"       [{M[1,0]:+.6f}, {M[1,1]:+.6f}]]")

    # Decompose M: symmetric part = scale+shear, antisymmetric = rotation
    sym = (M + M.T) / 2
    antisym = (M - M.T) / 2
    print(f"  Symmetric (scale/shear): diag=({sym[0,0]:+.6f}, {sym[1,1]:+.6f}), "
          f"off-diag={sym[0,1]:+.6f}")
    print(f"  Antisymmetric (rotation): {antisym[0,1]:+.6f} "
          f"= {np.degrees(antisym[0,1])*3600:.1f} arcsec")

    # Residuals after affine correction
    pred_dx = A_full @ params_x
    pred_dy = A_full @ params_y
    res_dx = dx_m - pred_dx
    res_dy = dy_m - pred_dy
    res_err = np.sqrt(res_dx**2 + res_dy**2)
    print(f"\n  After affine correction:")
    print(f"    residual: median={np.median(res_err)*1e6:.1f} µm "
          f"({np.median(res_err)*geom.res:.2f} px)")

    # Convert to pixel error for comparison
    res_err_px = res_err * geom.res
    print(f"    <1px: {(res_err_px < 1).sum()}/{len(res_err_px)}")
    print(f"    <3px: {(res_err_px < 3).sum()}/{len(res_err_px)}")

    # Interpretation of the affine parameters:
    # If M ≈ [[s, 0], [0, s]] → isotropic scale error → clen error
    #   Δclen/clen ≈ s → Δclen ≈ s * D
    scale = (M[0,0] + M[1,1]) / 2
    print(f"\n  Effective scale factor: {scale:+.6f}")
    print(f"  → Δclen ≈ {scale * D * 1000:.4f} mm (D={D:.6f} m)")
    print(f"  → corrected clen ≈ {D * (1 + scale):.6f} m")


if __name__ == "__main__":
    main()
