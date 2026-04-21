#!/usr/bin/env python
"""Debug single reflection to find source of ~1000px error."""
import torch
from torchsx.stream import SFXStream
from torchsx.forward.coords import RefinableGeometry

HC_EV_A = 12398.4

stream = SFXStream.load("merged_368.sfx", merge_friedel=False)
core = stream._core
geom = core.geometry

# Find crystal 0's first obs
for cid in range(core.n_crystals):
    mask = core.crystal_id == cid
    if mask.sum() > 50:
        break

print(f"Crystal {cid}, n_obs={mask.sum()}")
idx = mask.nonzero()[:5, 0]

for i in idx:
    h, k, l = core.hkl[i].tolist()
    fs_cf = core.fs[i].item()
    ss_cf = core.ss[i].item()
    pid = core.panel_id[i].item()
    p = geom.panels[pid]

    img = core.image_id[cid].item()
    en = core.beam_energy[img].item()
    cl = core.camera_length[img].item() if core.camera_length is not None else 0
    wl = HC_EV_A / en

    print(f"\n  [{h},{k},{l}] panel={p.name}(id={pid}) fs={fs_cf:.1f} ss={ss_cf:.1f}")
    print(f"  energy={en:.1f}eV  wl={wl:.6f}Å  clen={cl:.6f}m  geom.clen={geom.clen:.6f}m")

    # Observed lab position from pixel_to_lab
    rg = RefinableGeometry(geom)
    if cl > 0:
        rg._clen.fill_(cl)
    with torch.no_grad():
        obs_xyz = rg.pixel_to_lab(
            torch.tensor([fs_cf], dtype=torch.float32),
            torch.tensor([ss_cf], dtype=torch.float32),
            torch.tensor([pid], dtype=torch.long),
        )
    print(f"  obs_lab = ({obs_xyz[0,0].item()*1000:.3f}, {obs_xyz[0,1].item()*1000:.3f}, {obs_xyz[0,2].item()*1000:.3f}) mm")

    # Predicted position from UB
    UB = torch.stack([core.astar[cid], core.bstar[cid], core.cstar[cid]], dim=-1).float()
    hkl_t = core.hkl[i:i+1].float()
    h_lab = (UB @ hkl_t.T).T
    print(f"  h_lab = ({h_lab[0,0]:.6f}, {h_lab[0,1]:.6f}, {h_lab[0,2]:.6f}) Å⁻¹")

    # s_out
    s_out = wl * h_lab
    s_out[0, 2] += 1.0
    s_out = s_out / s_out.norm(dim=-1, keepdim=True)
    print(f"  s_out = ({s_out[0,0]:.6f}, {s_out[0,1]:.6f}, {s_out[0,2]:.6f})")

    clen_use = cl if cl > 0 else geom.clen
    t = clen_use / s_out[0, 2].item()
    x_pred = t * s_out[0, 0].item()
    y_pred = t * s_out[0, 1].item()
    print(f"  pred_lab = ({x_pred*1000:.3f}, {y_pred*1000:.3f}, {clen_use*1000:.3f}) mm")

    err_m = ((x_pred - obs_xyz[0, 0].item())**2 + (y_pred - obs_xyz[0, 1].item())**2)**0.5
    err_px = err_m * geom.res
    print(f"  error = {err_m*1000:.3f} mm = {err_px:.1f} px")

    # Also try with recip_to_pixel for comparison
    with torch.no_grad():
        rg2 = RefinableGeometry(geom)
        if cl > 0:
            rg2._clen.fill_(cl)
        pred_fs, pred_ss, pred_pid = rg2.recip_to_pixel(h_lab, wl)
    print(f"  recip_to_pixel: fs={pred_fs[0].item():.1f} ss={pred_ss[0].item():.1f} pid={pred_pid[0].item()}")
    print(f"  CrystFEL:       fs={fs_cf:.1f}          ss={ss_cf:.1f}          pid={pid}")
    if pred_pid[0].item() >= 0:
        err_px2 = ((pred_fs[0].item()-fs_cf)**2 + (pred_ss[0].item()-ss_cf)**2)**0.5
        print(f"  pixel error = {err_px2:.1f} px")
