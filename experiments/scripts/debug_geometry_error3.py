#!/usr/bin/env python
"""Test geometry with fresh stream parse and correct camera_length.

Key: use clen from average_camera_length, not geometry header.
"""

import numpy as np
import torch

from torchsx.io._parser import parse_stream
from torchsx.forward.coords import RefinableGeometry
from torchsx.scaling._orientation import CrystalOrientation

HC_EV_A = 12398.4


def main():
    print("Parsing merged_368.stream...")
    core = parse_stream("merged_368.stream", n_workers=1)
    geom = core.geometry
    print(f"  {core.n_crystals} crystals, {core.n_images} images")
    print(f"  geom.clen={geom.clen:.6f}, geom.res={geom.res:.1f}")

    cid = 700
    img_id = core.image_id[cid].item()
    energy_eV = core.beam_energy[img_id].item()
    wavelength = HC_EV_A / energy_eV
    clen_actual = core.camera_length[img_id].item()

    print(f"\nCrystal {cid}, image {img_id}:")
    print(f"  wavelength={wavelength:.6f} Å, camera_length={clen_actual:.6f} m")

    obs_mask = core.crystal_id == cid
    cf_hkl = core.hkl[obs_mask].numpy()
    cf_fs = core.fs[obs_mask].numpy()
    cf_ss = core.ss[obs_mask].numpy()
    cf_panel = core.panel_id[obs_mask].numpy()
    n_obs = cf_hkl.shape[0]

    astar = core.astar[cid]
    bstar = core.bstar[cid]
    cstar = core.cstar[cid]
    UB = torch.stack([astar, bstar, cstar], dim=-1).unsqueeze(0)
    hkl_t = torch.from_numpy(cf_hkl).float()

    # Test with both clen values
    for D, label in [(geom.clen, "geom.clen"), (clen_actual, "camera_length")]:
        print(f"\n=== Using {label} = {D:.6f} m ===")
        ref_geom = RefinableGeometry(geom)
        ref_geom._clen.fill_(D)

        h_lab = (UB[0] @ hkl_t.T).T
        with torch.no_grad():
            our_fs, our_ss, our_pid = ref_geom.recip_to_pixel(h_lab, wavelength)

        our_fs = our_fs.numpy()
        our_ss = our_ss.numpy()
        on_det = our_pid.numpy() >= 0

        dfs = our_fs[on_det] - cf_fs[on_det]
        dss = our_ss[on_det] - cf_ss[on_det]
        err = np.sqrt(dfs**2 + dss**2)

        print(f"  median: {np.median(err):.2f} px, max: {np.max(err):.2f} px")
        print(f"  <1px: {(err<1).sum()}/{len(err)}, <3px: {(err<3).sum()}/{len(err)}")

    # Now let's do a detailed manual check of specific reflections
    print("\n=== Manual check of specific reflections ===")
    D = clen_actual
    ref_geom = RefinableGeometry(geom)
    ref_geom._clen.fill_(D)

    h_lab_all = (UB[0] @ hkl_t.T).T

    # Sort by error to find worst and best cases
    with torch.no_grad():
        our_fs, our_ss, our_pid = ref_geom.recip_to_pixel(h_lab_all, wavelength)
    our_fs_np = our_fs.numpy()
    our_ss_np = our_ss.numpy()
    on_det = our_pid.numpy() >= 0
    errors = np.sqrt((our_fs_np - cf_fs)**2 + (our_ss_np - cf_ss)**2)

    # Check 5 worst and 5 best (on-detector)
    err_on = errors.copy()
    err_on[~on_det] = -1
    order = np.argsort(err_on)[::-1]

    print("\nWorst 5 spots:")
    for i in range(5):
        idx = order[i]
        h, k, l = cf_hkl[idx]
        h_lab_i = h_lab_all[idx].numpy()
        pid = cf_panel[idx]
        panel = geom.panels[pid]

        # Manual projection
        sx = wavelength * h_lab_i[0]
        sy = wavelength * h_lab_i[1]
        sz = wavelength * h_lab_i[2] + 1.0
        norm = np.sqrt(sx**2 + sy**2 + sz**2)
        sx, sy, sz = sx/norm, sy/norm, sz/norm

        z_panel = D + panel.coffset
        t = z_panel / sz
        x_m = t * sx
        y_m = t * sy

        # To pixel coords
        x_px = x_m * geom.res
        y_px = y_m * geom.res

        print(f"\n  [{h:3d},{k:3d},{l:3d}] panel={panel.name} err={errors[idx]:.2f}px")
        print(f"    h_lab = ({h_lab_i[0]:+.6f}, {h_lab_i[1]:+.6f}, {h_lab_i[2]:+.6f}) Å⁻¹")
        print(f"    |h_lab| = {np.linalg.norm(h_lab_i):.6f} Å⁻¹")
        print(f"    s_out = ({sx:.6f}, {sy:.6f}, {sz:.6f})")
        print(f"    x_m={x_m*1000:.3f}mm, y_m={y_m*1000:.3f}mm")
        print(f"    our:     fs={our_fs_np[idx]:.2f}, ss={our_ss_np[idx]:.2f}")
        print(f"    CrystFEL: fs={cf_fs[idx]:.2f}, ss={cf_ss[idx]:.2f}")

        # What h_lab would CrystFEL's position imply?
        # CrystFEL position → lab coords
        cf_ss_t = torch.tensor([cf_ss[idx]], dtype=torch.float64)
        cf_fs_t = torch.tensor([cf_fs[idx]], dtype=torch.float64)
        cf_pid_t = torch.tensor([pid], dtype=torch.long)
        cf_xyz = geom.to_lab(cf_ss_t, cf_fs_t, cf_pid_t)
        cf_x_m = cf_xyz[0, 0].item()
        cf_y_m = cf_xyz[0, 1].item()
        cf_z_m = cf_xyz[0, 2].item()

        # Reverse: lab → s_hat → h_lab
        cf_norm = np.sqrt(cf_x_m**2 + cf_y_m**2 + cf_z_m**2)
        cf_sx = cf_x_m / cf_norm
        cf_sy = cf_y_m / cf_norm
        cf_sz = cf_z_m / cf_norm
        # h_lab = (s_hat - z_hat) / lambda
        cf_hx = (cf_sx - 0) / wavelength
        cf_hy = (cf_sy - 0) / wavelength
        cf_hz = (cf_sz - 1) / wavelength

        print(f"    CrystFEL implied h_lab: ({cf_hx:+.6f}, {cf_hy:+.6f}, {cf_hz:+.6f}) Å⁻¹")
        dh = np.array([cf_hx - h_lab_i[0], cf_hy - h_lab_i[1], cf_hz - h_lab_i[2]])
        print(f"    Δh_lab: ({dh[0]:+.6f}, {dh[1]:+.6f}, {dh[2]:+.6f}) |Δh|={np.linalg.norm(dh):.6f}")

    print("\nBest 5 spots:")
    good_order = np.argsort(err_on)
    for i in range(5):
        idx = good_order[i]
        if err_on[idx] < 0:
            continue
        h, k, l = cf_hkl[idx]
        pid = cf_panel[idx]
        panel = geom.panels[pid]
        print(f"  [{h:3d},{k:3d},{l:3d}] panel={panel.name} err={errors[idx]:.2f}px "
              f"our=({our_fs_np[idx]:.2f},{our_ss_np[idx]:.2f}) "
              f"cf=({cf_fs[idx]:.2f},{cf_ss[idx]:.2f})")

    # Check: does the error pattern change if we use a few different crystals?
    print("\n=== Checking multiple crystals ===")
    for test_cid in [0, 100, 500, 1000, 5000, 10000]:
        if test_cid >= core.n_crystals:
            continue
        img = core.image_id[test_cid].item()
        e_eV = core.beam_energy[img].item()
        wl = HC_EV_A / e_eV
        cl = core.camera_length[img].item()

        mask = core.crystal_id == test_cid
        hkl = core.hkl[mask]
        fs = core.fs[mask].numpy()
        ss = core.ss[mask].numpy()

        ub = torch.stack([core.astar[test_cid], core.bstar[test_cid],
                         core.cstar[test_cid]], dim=-1).unsqueeze(0)
        h_lab = (ub[0] @ hkl.float().T).T

        ref = RefinableGeometry(geom)
        ref._clen.fill_(cl if cl > 0 else geom.clen)
        with torch.no_grad():
            p_fs, p_ss, p_pid = ref.recip_to_pixel(h_lab, wl)

        on = p_pid.numpy() >= 0
        if on.sum() == 0:
            continue
        e = np.sqrt((p_fs.numpy()[on]-fs[on])**2 + (p_ss.numpy()[on]-ss[on])**2)
        n_refl = core.n_refl[test_cid].item()
        print(f"  Crystal {test_cid:5d}: clen={cl:.6f} n_refl={n_refl:3d} "
              f"median_err={np.median(e):.2f}px <3px={np.sum(e<3)}/{len(e)}")


if __name__ == "__main__":
    main()
