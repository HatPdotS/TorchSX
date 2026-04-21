#!/usr/bin/env python
"""For each observed reflection, compute the relp radius required to excite it.

excitation_error = distance from Ewald sphere in reciprocal space.
This is the minimum relp radius needed for this reflection to appear.
"""

import torch
import numpy as np
from torchsx.stream import SFXStream
from torchsx.forward.coords import RefinableGeometry

HC_EV_A = 12398.4


def excitation_error(h_lab: torch.Tensor, wavelength: float) -> torch.Tensor:
    """Excitation error = |s_out| - 1/λ, where s_out = h_lab + s0."""
    s_out = wavelength * h_lab.clone()
    s_out[:, 2] += 1.0
    return (s_out.norm(dim=-1) - 1.0).abs() / wavelength  # Å⁻¹


def main():
    stream = SFXStream.load("merged_368.sfx", merge_friedel=False)
    core = stream._core
    geom = core.geometry

    # Sample many crystals for statistics
    np.random.seed(42)
    n_sample = 500
    sample_cids = np.random.choice(core.n_crystals, n_sample, replace=False)

    all_exc = []
    all_peak = []
    all_I = []
    all_d = []
    all_pos_err = []

    ref_geom = RefinableGeometry(geom)

    for i, cid in enumerate(sample_cids):
        cid = int(cid)
        mask = core.crystal_id == cid
        if mask.sum() < 10:
            continue

        img_id = core.image_id[cid].item()
        en = core.beam_energy[img_id].item()
        wl = HC_EV_A / en
        cl = core.camera_length[img_id].item()
        if cl <= 0:
            cl = geom.clen

        UB = torch.stack([core.astar[cid], core.bstar[cid], core.cstar[cid]], dim=-1).float()
        hkl = core.hkl[mask].float()
        h_lab = (UB @ hkl.T).T

        exc = excitation_error(h_lab, wl)
        d = 1.0 / h_lab.norm(dim=-1).clamp(min=1e-10)

        # Position error
        ref_geom._clen.fill_(cl)
        fs_obs = core.fs[mask].float()
        ss_obs = core.ss[mask].float()
        pid_obs = core.panel_id[mask]
        with torch.no_grad():
            pred_fs, pred_ss, pred_pid = ref_geom.recip_to_pixel(h_lab, wl)
        pos_err = torch.sqrt((pred_fs - fs_obs)**2 + (pred_ss - ss_obs)**2)

        all_exc.append(exc)
        all_peak.append(core.peak[mask].float())
        all_I.append(core.I[mask].float())
        all_d.append(d)
        all_pos_err.append(pos_err)

    exc = torch.cat(all_exc)
    peak = torch.cat(all_peak)
    I_obs = torch.cat(all_I)
    d = torch.cat(all_d)
    pos_err = torch.cat(all_pos_err)
    N = len(exc)

    profile_rad = core.profile_rad[0].item()  # same for all (fixed)
    print(f"Analyzed {N} reflections from {n_sample} crystals")
    print(f"profile_rad = {profile_rad:.4f} Å⁻¹")

    # Distribution of required relp
    print(f"\n--- Required relp radius to explain each observed spot ---")
    print(f"  min:    {exc.min():.6f} Å⁻¹  ({exc.min()/profile_rad:.2f}× profile_rad)")
    print(f"  25%:    {exc.quantile(0.25):.6f} Å⁻¹  ({exc.quantile(0.25)/profile_rad:.2f}×)")
    print(f"  median: {exc.median():.6f} Å⁻¹  ({exc.median()/profile_rad:.2f}×)")
    print(f"  75%:    {exc.quantile(0.75):.6f} Å⁻¹  ({exc.quantile(0.75)/profile_rad:.2f}×)")
    print(f"  90%:    {exc.quantile(0.90):.6f} Å⁻¹  ({exc.quantile(0.90)/profile_rad:.2f}×)")
    print(f"  95%:    {exc.quantile(0.95):.6f} Å⁻¹  ({exc.quantile(0.95)/profile_rad:.2f}×)")
    print(f"  99%:    {exc.quantile(0.99):.6f} Å⁻¹  ({exc.quantile(0.99)/profile_rad:.2f}×)")
    print(f"  max:    {exc.max():.6f} Å⁻¹  ({exc.max()/profile_rad:.2f}×)")

    # Cumulative: what fraction of spots are captured at each relp
    print(f"\n--- Fraction of observed spots captured at relp threshold ---")
    for mult in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        thresh = profile_rad * mult
        frac = (exc < thresh).float().mean() * 100
        print(f"  {mult:5.2f}× ({thresh:.4f} Å⁻¹): {frac:5.1f}%")

    # Binned: exc_error vs peak, I, d, pos_err
    print(f"\n--- Properties binned by excitation error ---")
    print(f"{'exc range (Å⁻¹)':>22s}  {'n':>6s}  {'peak':>6s}  {'I':>8s}  "
          f"{'d(Å)':>6s}  {'pos_err':>8s}")
    edges = [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.02]
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = (exc >= lo) & (exc < hi)
        if band.sum() == 0:
            continue
        print(f"  [{lo:.3f}, {hi:.3f})  {band.sum():6d}  "
              f"{peak[band].median():6.0f}  {I_obs[band].median():8.0f}  "
              f"{d[band].median():6.2f}  {pos_err[band].median():7.1f}px")


if __name__ == "__main__":
    main()
