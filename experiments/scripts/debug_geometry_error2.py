#!/usr/bin/env python
"""Test different clen values and analyze residual error patterns."""

import numpy as np
import torch

from torchsx import SFXStream
from torchsx.forward.coords import RefinableGeometry
from torchsx.scaling._orientation import CrystalOrientation

HC_EV_A = 12398.4


def predict_with_clen(UB, geom, clen, hkl_t, wavelength):
    """Predict spots with a given clen."""
    ref_geom = RefinableGeometry(geom)
    ref_geom._clen.fill_(clen)

    h_lab = (UB[0] @ hkl_t.T).T
    with torch.no_grad():
        fs, ss, pid = ref_geom.recip_to_pixel(h_lab, wavelength)
    return fs.numpy(), ss.numpy(), pid.numpy()


def main():
    print("Loading merged_368.sfx...")
    stream = SFXStream.load("merged_368.sfx", merge_friedel=False)
    core = stream._core
    geom = core.geometry

    cid = 700
    img_id = core.image_id[cid].item()
    energy_eV = core.beam_energy[img_id].item()
    wavelength = HC_EV_A / energy_eV

    obs_mask = core.crystal_id == cid
    cf_hkl = core.hkl[obs_mask].numpy()
    cf_fs = core.fs[obs_mask].numpy()
    cf_ss = core.ss[obs_mask].numpy()
    cf_panel = core.panel_id[obs_mask].numpy()

    astar = core.astar[cid]
    bstar = core.bstar[cid]
    cstar = core.cstar[cid]
    UB = torch.stack([astar, bstar, cstar], dim=-1).unsqueeze(0)
    hkl_t = torch.from_numpy(cf_hkl).float()

    # Scan clen
    print("\n=== clen scan ===")
    clens = np.arange(0.0920, 0.0960, 0.0001)
    best_clen = None
    best_median = 999
    for cl in clens:
        fs, ss, pid = predict_with_clen(UB, geom, cl, hkl_t, wavelength)
        on_det = pid >= 0
        if on_det.sum() == 0:
            continue
        err = np.sqrt((fs[on_det]-cf_fs[on_det])**2 + (ss[on_det]-cf_ss[on_det])**2)
        med = np.median(err)
        if med < best_median:
            best_median = med
            best_clen = cl
        if abs(cl - 0.0935) < 0.00005 or abs(cl - 0.0939) < 0.00005 or cl == best_clen:
            print(f"  clen={cl:.4f} m: median={med:.2f} px, <1px={np.sum(err<1)}, "
                  f"<3px={np.sum(err<3)}")

    print(f"\n  Best clen: {best_clen:.4f} m, median error: {best_median:.2f} px")

    # Now analyze the residual at best clen
    fs, ss, pid = predict_with_clen(UB, geom, best_clen, hkl_t, wavelength)
    on_det = pid >= 0
    dfs = fs[on_det] - cf_fs[on_det]
    dss = ss[on_det] - cf_ss[on_det]
    err = np.sqrt(dfs**2 + dss**2)

    print(f"\n=== Residual at clen={best_clen:.4f} m ===")
    print(f"  median: {np.median(err):.2f} px, mean: {np.mean(err):.2f} px, max: {np.max(err):.2f} px")
    print(f"  <1px: {(err<1).sum()}/{len(err)}, <3px: {(err<3).sum()}/{len(err)}")

    # Lab-frame error analysis
    cf_ss_t = torch.from_numpy(cf_ss[on_det].astype(np.float64))
    cf_fs_t = torch.from_numpy(cf_fs[on_det].astype(np.float64))
    xyz = geom.to_lab(cf_ss_t, cf_fs_t)
    x_m = xyz[:, 0].numpy()
    y_m = xyz[:, 1].numpy()
    z_m = xyz[:, 2].numpy()

    our_ss_t = torch.from_numpy(ss[on_det].astype(np.float64))
    our_fs_t = torch.from_numpy(fs[on_det].astype(np.float64))
    our_xyz = geom.to_lab(our_ss_t, our_fs_t)
    dx_m = our_xyz[:, 0].numpy() - x_m
    dy_m = our_xyz[:, 1].numpy() - y_m

    # Quadratic model: does error have a quadratic component?
    r_xy = np.sqrt(x_m**2 + y_m**2)
    r_hat_x = x_m / (r_xy + 1e-10)
    r_hat_y = y_m / (r_xy + 1e-10)
    d_radial = dx_m * r_hat_x + dy_m * r_hat_y
    d_tangential = -dx_m * r_hat_y + dy_m * r_hat_x

    # Fit radial error as polynomial in r_xy
    # d_radial = a0 + a1*r + a2*r^2
    A = np.column_stack([np.ones(len(r_xy)), r_xy, r_xy**2])
    params_r = np.linalg.lstsq(A, d_radial, rcond=None)[0]
    params_t = np.linalg.lstsq(A, d_tangential, rcond=None)[0]

    print(f"\n  Radial error polynomial (vs r_xy in m):")
    print(f"    d_radial = {params_r[0]*1e6:+.1f}µm + {params_r[1]*1e3:+.3f}mm/m*r + "
          f"{params_r[2]:+.3f}*r²")
    print(f"  Tangential error polynomial:")
    print(f"    d_tang = {params_t[0]*1e6:+.1f}µm + {params_t[1]*1e3:+.3f}mm/m*r + "
          f"{params_t[2]:+.3f}*r²")

    # Residual after quadratic fit
    pred_r = A @ params_r
    pred_t = A @ params_t
    res_r = d_radial - pred_r
    res_t = d_tangential - pred_t
    res_err_m = np.sqrt(res_r**2 + res_t**2)
    res_err_px = res_err_m * geom.res
    print(f"\n  After quadratic correction:")
    print(f"    median: {np.median(res_err_px):.2f} px")
    print(f"    <1px: {(res_err_px<1).sum()}/{len(res_err_px)}")
    print(f"    <3px: {(res_err_px<3).sum()}/{len(res_err_px)}")

    # Check per-panel after best clen
    print(f"\n=== Per-panel at clen={best_clen:.4f} ===")
    panels_used = np.unique(cf_panel[on_det])
    for pid_val in sorted(panels_used):
        pmask = cf_panel[on_det] == pid_val
        p_err = err[pmask]
        p = geom.panels[pid_val]
        cx = p.corner_x + p.n_fs/2 * p.fs_vec[0] + p.n_ss/2 * p.ss_vec[0]
        cy = p.corner_y + p.n_fs/2 * p.fs_vec[1] + p.n_ss/2 * p.ss_vec[1]
        dist = np.sqrt(cx**2 + cy**2)
        print(f"  {p.name:>4s}: n={pmask.sum():3d}  med={np.median(p_err):6.2f}px  dist={dist:.0f}px")

    # Now try scanning both clen and beam center shift
    print("\n=== clen + beam shift scan ===")
    best_combo = None
    best_combo_err = 999

    for cl_off in np.arange(-0.0005, 0.0006, 0.0001):
        cl = geom.clen + cl_off
        ref_geom = RefinableGeometry(geom)
        ref_geom._clen.fill_(cl)

        for dx_shift_px in np.arange(-10, 11, 2):
            for dy_shift_px in np.arange(-10, 11, 2):
                # Apply beam shift by adjusting all corners
                h_lab = (UB[0] @ hkl_t.T).T
                with torch.no_grad():
                    fs_p, ss_p, pid_p = ref_geom.recip_to_pixel(h_lab, wavelength)

                fs_p = fs_p.numpy()
                ss_p = ss_p.numpy()
                pid_p = pid_p.numpy()
                on = pid_p >= 0

                # Shift is applied to predictions
                fs_shifted = fs_p[on] + dx_shift_px
                ss_shifted = ss_p[on] + dy_shift_px

                e = np.sqrt((fs_shifted - cf_fs[on])**2 + (ss_shifted - cf_ss[on])**2)
                med = np.median(e)
                if med < best_combo_err:
                    best_combo_err = med
                    best_combo = (cl, dx_shift_px, dy_shift_px)

    cl_b, dx_b, dy_b = best_combo
    print(f"  Best: clen={cl_b:.4f}, shift=({dx_b:.0f}, {dy_b:.0f}) px, "
          f"median error={best_combo_err:.2f} px")

    # Fine scan around best
    for cl_off in np.arange(cl_b-0.0002, cl_b+0.00021, 0.00005):
        ref_geom = RefinableGeometry(geom)
        ref_geom._clen.fill_(cl_off)
        for dx_s in np.arange(dx_b-3, dx_b+4, 1):
            for dy_s in np.arange(dy_b-3, dy_b+4, 1):
                h_lab = (UB[0] @ hkl_t.T).T
                with torch.no_grad():
                    fs_p, ss_p, pid_p = ref_geom.recip_to_pixel(h_lab, wavelength)
                fs_p = fs_p.numpy()
                ss_p = ss_p.numpy()
                pid_p = pid_p.numpy()
                on = pid_p >= 0
                e = np.sqrt((fs_p[on]+dx_s-cf_fs[on])**2 + (ss_p[on]+dy_s-cf_ss[on])**2)
                med = np.median(e)
                if med < best_combo_err:
                    best_combo_err = med
                    best_combo = (cl_off, dx_s, dy_s)

    cl_b, dx_b, dy_b = best_combo
    print(f"  Fine: clen={cl_b:.5f}, shift=({dx_b:.0f}, {dy_b:.0f}) px, "
          f"median error={best_combo_err:.2f} px")

    # Show residual at the fine-tuned params
    ref_geom = RefinableGeometry(geom)
    ref_geom._clen.fill_(cl_b)
    h_lab = (UB[0] @ hkl_t.T).T
    with torch.no_grad():
        fs_p, ss_p, pid_p = ref_geom.recip_to_pixel(h_lab, wavelength)
    fs_p = fs_p.numpy() + dx_b
    ss_p = ss_p.numpy() + dy_b
    pid_p = pid_p.numpy()
    on = pid_p >= 0
    final_err = np.sqrt((fs_p[on]-cf_fs[on])**2 + (ss_p[on]-cf_ss[on])**2)
    print(f"\n  Final residual: median={np.median(final_err):.2f} px, "
          f"max={np.max(final_err):.2f} px")
    print(f"  <1px: {(final_err<1).sum()}/{len(final_err)}")
    print(f"  <3px: {(final_err<3).sum()}/{len(final_err)}")
    print(f"  <5px: {(final_err<5).sum()}/{len(final_err)}")


if __name__ == "__main__":
    main()
