#!/usr/bin/env python
"""Debug why screening_8.stream predictions are ~2600px off."""

import torch
import numpy as np
from torchsx.io._parser import parse_stream
from torchsx.forward.coords import RefinableGeometry

HC_EV_A = 12398.4

SCREENING = "/das/work/p17/p17489/Peter/2025-01-31_SwissFEL_SFX/per_run_detector_distance/work/run0080/screening_8/screening_8.stream"

def main():
    print("Parsing screening_8.stream...")
    core = parse_stream(SCREENING, n_workers=1)
    geom = core.geometry
    print(f"  {core.n_crystals} crystals, {core.n_images} images, {core.n_observations} obs")
    print(f"  geom.clen={geom.clen:.6f}, geom.res={geom.res:.1f}")
    print(f"  {geom.n_panels} panels")

    # Print all panel definitions
    print("\nPanel definitions:")
    for i, p in enumerate(geom.panels):
        print(f"  {i}: {p.name:>4s}  ss=[{p.min_ss},{p.max_ss}] fs=[{p.min_fs},{p.max_fs}]  "
              f"corner=({p.corner_x:.1f},{p.corner_y:.1f})  "
              f"fs_vec=({p.fs_vec[0]:+.6f},{p.fs_vec[1]:+.6f},{p.fs_vec[2]:+.6f})  "
              f"ss_vec=({p.ss_vec[0]:+.6f},{p.ss_vec[1]:+.6f},{p.ss_vec[2]:+.6f})  "
              f"coffset={p.coffset}")

    # Find a crystal with reflections
    cid = 0
    for c in range(core.n_crystals):
        if (core.crystal_id == c).sum() > 10:
            cid = c
            break
    print(f"\nUsing crystal {cid} (n_refl={core.n_refl[cid].item()})")
    img_id = core.image_id[cid].item()
    energy_eV = core.beam_energy[img_id].item()
    wavelength = HC_EV_A / energy_eV
    clen = core.camera_length[img_id].item() if core.camera_length is not None else 0
    if clen <= 0:
        clen = geom.clen

    print(f"\nCrystal {cid}, image {img_id}:")
    print(f"  energy={energy_eV:.1f} eV, wavelength={wavelength:.6f} Å, clen={clen:.6f} m")

    obs_mask = core.crystal_id == cid
    cf_hkl = core.hkl[obs_mask].numpy()
    cf_fs = core.fs[obs_mask].numpy()
    cf_ss = core.ss[obs_mask].numpy()
    cf_panel = core.panel_id[obs_mask].numpy()
    n_obs = cf_hkl.shape[0]
    print(f"  {n_obs} reflections")

    # Show first 5 reflections with their panel info
    print("\nFirst 10 reflections:")
    for i in range(min(10, n_obs)):
        h, k, l = cf_hkl[i]
        pid = cf_panel[i]
        p = geom.panels[pid]
        in_range = (p.min_fs <= cf_fs[i] <= p.max_fs) and (p.min_ss <= cf_ss[i] <= p.max_ss)
        print(f"  [{h:3d},{k:3d},{l:3d}] panel={p.name}(id={pid}) "
              f"fs={cf_fs[i]:.1f} ss={cf_ss[i]:.1f}  "
              f"panel_range: fs[{p.min_fs},{p.max_fs}] ss[{p.min_ss},{p.max_ss}]  "
              f"in_range={in_range}")

    # Build UB and predict
    astar = core.astar[cid]
    bstar = core.bstar[cid]
    cstar = core.cstar[cid]
    UB = torch.stack([astar, bstar, cstar], dim=-1)
    hkl_t = torch.from_numpy(cf_hkl).float()

    print(f"\nUB matrix:\n{UB}")
    print(f"astar = {astar}")
    print(f"bstar = {bstar}")
    print(f"cstar = {cstar}")

    h_lab = (UB @ hkl_t.T).T

    ref_geom = RefinableGeometry(geom)
    ref_geom._clen.fill_(clen)

    with torch.no_grad():
        our_fs, our_ss, our_pid = ref_geom.recip_to_pixel(h_lab, wavelength)

    our_fs = our_fs.numpy()
    our_ss = our_ss.numpy()
    our_pid = our_pid.numpy()

    print("\nFirst 10 predictions vs CrystFEL:")
    for i in range(min(10, n_obs)):
        h, k, l = cf_hkl[i]
        pid_cf = cf_panel[i]
        pid_us = our_pid[i]
        p_cf = geom.panels[pid_cf]
        p_us_name = geom.panels[pid_us].name if pid_us >= 0 else "OFF"
        err = np.sqrt((our_fs[i]-cf_fs[i])**2 + (our_ss[i]-cf_ss[i])**2) if pid_us >= 0 else float('nan')
        print(f"  [{h:3d},{k:3d},{l:3d}]  cf: fs={cf_fs[i]:7.1f} ss={cf_ss[i]:7.1f} {p_cf.name}  "
              f"us: fs={our_fs[i]:7.1f} ss={our_ss[i]:7.1f} {p_us_name}  err={err:.1f}px")

    # Manual projection for first reflection
    print("\n=== Manual projection of first reflection ===")
    i = 0
    h_lab_i = h_lab[i].numpy()
    print(f"  h_lab = ({h_lab_i[0]:+.6f}, {h_lab_i[1]:+.6f}, {h_lab_i[2]:+.6f}) Å⁻¹")

    # s_out = λ*h_lab + ẑ
    sx = wavelength * h_lab_i[0]
    sy = wavelength * h_lab_i[1]
    sz = wavelength * h_lab_i[2] + 1.0
    norm = np.sqrt(sx**2 + sy**2 + sz**2)
    sx, sy, sz = sx/norm, sy/norm, sz/norm
    print(f"  s_out = ({sx:.6f}, {sy:.6f}, {sz:.6f})")

    # For each panel, compute where the ray hits
    pid_cf = cf_panel[i]
    p = geom.panels[pid_cf]
    D = clen + p.coffset
    t = D / sz
    x_m = t * sx
    y_m = t * sy
    print(f"  Using panel {p.name}: D={D:.6f} m")
    print(f"  Lab position: ({x_m*1000:.3f}, {y_m*1000:.3f}) mm")

    # To pixel: x_px = x_m * res, y_px = y_m * res
    x_px = x_m * geom.res
    y_px = y_m * geom.res
    print(f"  Pixel from beam: ({x_px:.1f}, {y_px:.1f}) px")

    # Panel corner → (fs, ss) via Cramer's rule
    dx = x_px - p.corner_x
    dy = y_px - p.corner_y
    fsx, fsy, fsz = p.fs_vec
    ssx, ssy, ssz = p.ss_vec
    det = fsx * ssy - fsy * ssx
    fs_local = (dx * ssy - dy * ssx) / det
    ss_local = (fsx * dy - fsy * dx) / det
    fs_global = fs_local + p.min_fs
    ss_global = ss_local + p.min_ss
    print(f"  Panel {p.name}: corner=({p.corner_x:.1f},{p.corner_y:.1f}), dx=({dx:.1f},{dy:.1f})")
    print(f"  fs_local={fs_local:.1f}, ss_local={ss_local:.1f}")
    print(f"  fs_global={fs_global:.1f}, ss_global={ss_global:.1f}")
    print(f"  CrystFEL: fs={cf_fs[i]:.1f}, ss={cf_ss[i]:.1f}")

    # Also check: what does CrystFEL's position map to in lab frame?
    cf_ss_t = torch.tensor([cf_ss[i]], dtype=torch.float64)
    cf_fs_t = torch.tensor([cf_fs[i]], dtype=torch.float64)
    cf_pid_t = torch.tensor([pid_cf], dtype=torch.long)
    cf_xyz = geom.to_lab(cf_ss_t, cf_fs_t, cf_pid_t)
    print(f"\n  CrystFEL position in lab: ({cf_xyz[0,0].item()*1000:.3f}, {cf_xyz[0,1].item()*1000:.3f}, {cf_xyz[0,2].item()*1000:.3f}) mm")
    print(f"  Our predicted lab position: ({x_m*1000:.3f}, {y_m*1000:.3f}, {D*1000:.3f}) mm")

    # Check det_shift
    ds_x = core.det_shift_x[img_id].item()
    ds_y = core.det_shift_y[img_id].item()
    print(f"\n  det_shift: x={ds_x:.4f} mm, y={ds_y:.4f} mm")


if __name__ == "__main__":
    main()
