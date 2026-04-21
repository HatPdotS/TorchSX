#!/usr/bin/env python
"""Predict spots on detector using refined UB and varying relp sizes.

For each hkl, compute excitation error (distance from Ewald sphere).
Spots with |excitation_error| < relp_radius are "predicted".
Compare predicted set vs observed set from stream.
"""

import torch
import numpy as np
from torchsx.stream import SFXStream
from torchsx.forward.coords import RefinableGeometry

HC_EV_A = 12398.4


def generate_hkl(d_min: float, cell_params: torch.Tensor) -> torch.Tensor:
    """Generate all hkl within resolution limit."""
    a, b, c, alpha, beta, gamma = cell_params.tolist()
    # Conservative: max index ~ a/d_min etc
    h_max = int(np.ceil(a / d_min)) + 1
    k_max = int(np.ceil(b / d_min)) + 1
    l_max = int(np.ceil(c / d_min)) + 1

    hh = torch.arange(-h_max, h_max + 1)
    kk = torch.arange(-k_max, k_max + 1)
    ll = torch.arange(-l_max, l_max + 1)
    h, k, l = torch.meshgrid(hh, kk, ll, indexing="ij")
    hkl = torch.stack([h.flatten(), k.flatten(), l.flatten()], dim=-1).float()
    # Remove (0,0,0)
    nonzero = hkl.abs().sum(dim=-1) > 0
    return hkl[nonzero]


def excitation_error(h_lab: torch.Tensor, wavelength: float) -> torch.Tensor:
    """Compute excitation error for each reciprocal vector.

    Excitation error = distance of h_lab from the Ewald sphere surface.
    η = |h_lab + s0|/|s0| - 1, where s0 = (0,0,1/λ)
    Or equivalently: η = (1/λ)|sqrt((λ*hx)² + (λ*hy)² + (1+λ*hz)²) - 1|
    """
    s_out = wavelength * h_lab.clone()
    s_out[:, 2] += 1.0
    r = s_out.norm(dim=-1)  # should be 1.0 on the Ewald sphere
    return (r - 1.0).abs() / wavelength  # in Å⁻¹


def main():
    print("Loading .sfx...")
    stream = SFXStream.load("merged_368.sfx", merge_friedel=False)
    core = stream._core
    geom = core.geometry

    # Use the same crystal as refine script (best one)
    cid = 25494
    img_id = core.image_id[cid].item()
    energy_eV = core.beam_energy[img_id].item()
    wavelength = HC_EV_A / energy_eV
    clen_val = core.camera_length[img_id].item()
    if clen_val <= 0:
        clen_val = geom.clen
    profile_rad = core.profile_rad[cid].item()
    d_min = core.resolution[cid].item()

    print(f"Crystal {cid}: energy={energy_eV:.0f}eV, λ={wavelength:.4f}Å, "
          f"clen={clen_val:.4f}m")
    print(f"  profile_rad={profile_rad:.6f} Å⁻¹, d_min={d_min:.2f} Å")

    # Apply refined UB rotation from previous run
    astar = core.astar[cid].float()
    bstar = core.bstar[cid].float()
    cstar = core.cstar[cid].float()
    UB0 = torch.stack([astar, bstar, cstar], dim=-1)

    # Apply small rotation from refinement
    rot = torch.tensor([2.3775e-3, -7.068e-4, 1.0885e-3])
    wx, wy, wz = rot[0], rot[1], rot[2]
    K = torch.tensor([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]])
    R = torch.eye(3) + K
    UB = R @ UB0

    # Observed spots
    obs_mask = core.crystal_id == cid
    obs_hkl = core.hkl[obs_mask]
    obs_fs = core.fs[obs_mask].float()
    obs_ss = core.ss[obs_mask].float()
    obs_pid = core.panel_id[obs_mask]
    obs_peak = core.peak[obs_mask].float()
    n_obs = obs_hkl.shape[0]

    # Check actual resolution of observed reflections
    obs_h_lab = (UB @ obs_hkl.float().T).T
    obs_d = 1.0 / obs_h_lab.norm(dim=-1).clamp(min=1e-10)
    print(f"  {n_obs} observed reflections in stream")
    print(f"  Observed d-spacing range: {obs_d.min():.2f} — {obs_d.max():.2f} Å")

    # Generate all possible hkl with generous resolution (use actual minimum d)
    cell = core.cell[cid]
    d_min_actual = max(obs_d.min().item() - 0.5, 1.0)  # give margin
    all_hkl = generate_hkl(d_min_actual, cell)
    print(f"  {all_hkl.shape[0]} candidate hkl within {d_min:.2f} Å resolution")

    # Compute h_lab and excitation error for all candidates
    h_lab_all = (UB @ all_hkl.T).T
    exc_err = excitation_error(h_lab_all, wavelength)

    # Filter by d-spacing — use actual observed range, not the reported d_min
    d_spacing = 1.0 / h_lab_all.norm(dim=-1).clamp(min=1e-10)
    in_res = d_spacing >= d_min_actual

    # Project onto detector
    ref_geom = RefinableGeometry(geom)
    ref_geom._clen.fill_(clen_val)
    with torch.no_grad():
        pred_fs, pred_ss, pred_pid = ref_geom.recip_to_pixel(h_lab_all, wavelength)
    on_det = pred_pid >= 0

    # Try different relp radii
    print(f"\n{'relp_rad (Å⁻¹)':>16s}  {'predicted':>9s}  {'matched':>7s}  "
          f"{'unmatched':>9s}  {'extra_obs':>9s}  {'match_rate':>10s}")
    print("-" * 75)

    # Build observed hkl set for matching
    obs_hkl_set = set()
    for i in range(n_obs):
        h, k, l = obs_hkl[i].tolist()
        obs_hkl_set.add((int(h), int(k), int(l)))

    for mult in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0]:
        relp = profile_rad * mult
        predicted = in_res & on_det & (exc_err < relp)
        n_pred = predicted.sum().item()

        # Which of the predicted hkl are also in observed?
        pred_hkl_set = set()
        for i in predicted.nonzero()[:, 0]:
            h, k, l = all_hkl[i].tolist()
            pred_hkl_set.add((int(h), int(k), int(l)))

        matched = pred_hkl_set & obs_hkl_set
        unmatched_pred = pred_hkl_set - obs_hkl_set  # predicted but not observed
        extra_obs = obs_hkl_set - pred_hkl_set       # observed but not predicted

        match_rate = len(matched) / n_obs * 100 if n_obs > 0 else 0
        print(f"  {relp:.6f} ({mult:4.1f}×)  {n_pred:9d}  {len(matched):7d}  "
              f"{len(unmatched_pred):9d}  {len(extra_obs):9d}  {match_rate:9.1f}%")

    # ── Investigate the ~39 unmatched observed spots ──
    # Use 2× profile_rad as threshold (86% match)
    relp = profile_rad * 2.0
    predicted = in_res & on_det & (exc_err < relp)
    pred_hkl_set = set()
    for i in predicted.nonzero()[:, 0]:
        h, k, l = all_hkl[i].tolist()
        pred_hkl_set.add((int(h), int(k), int(l)))

    obs_h_lab = (UB @ obs_hkl.float().T).T
    obs_exc = excitation_error(obs_h_lab, wavelength)
    obs_d = 1.0 / obs_h_lab.norm(dim=-1).clamp(min=1e-10)

    with torch.no_grad():
        obs_pred_fs, obs_pred_ss, obs_pred_pid = ref_geom.recip_to_pixel(obs_h_lab, wavelength)
    obs_err_px = torch.sqrt((obs_pred_fs - obs_fs)**2 + (obs_pred_ss - obs_ss)**2)

    # Which observed spots are NOT in predicted set?
    unmatched_idx = []
    matched_idx = []
    for i in range(n_obs):
        h, k, l = obs_hkl[i].tolist()
        key = (int(h), int(k), int(l))
        if key not in pred_hkl_set:
            unmatched_idx.append(i)
        else:
            matched_idx.append(i)
    unmatched_idx = torch.tensor(unmatched_idx)
    matched_idx = torch.tensor(matched_idx)

    print(f"\n{'='*75}")
    print(f"Investigating {len(unmatched_idx)} unmatched spots (not predicted at 2× relp)")
    print(f"{'='*75}")

    # 1. Basic stats comparison
    print(f"\n--- Matched ({len(matched_idx)}) vs Unmatched ({len(unmatched_idx)}) ---")
    for label, idx in [("Matched", matched_idx), ("Unmatched", unmatched_idx)]:
        exc = obs_exc[idx]
        d = obs_d[idx]
        pk = obs_peak[idx]
        err = obs_err_px[idx]
        I_vals = core.I[obs_mask][idx].float()
        print(f"  {label:>9s}:  exc_err median={exc.median():.4f} Å⁻¹  "
              f"d median={d.median():.2f} Å  peak median={pk.median():.0f}  "
              f"I median={I_vals.median():.0f}  pos_err median={err.median():.1f} px")

    # 2. List all unmatched spots
    print(f"\n--- All unmatched spots ---")
    print(f"{'hkl':>12s}  {'panel':>5s}  {'fs':>7s}  {'ss':>7s}  {'exc_err':>8s}  "
          f"{'d(Å)':>6s}  {'peak':>5s}  {'I':>8s}  {'pos_err':>8s}  {'pred_panel':>10s}")
    for i in unmatched_idx:
        h, k, l = obs_hkl[i].tolist()
        pid = obs_pid[i].item()
        pname = geom.panels[pid].name
        ppid = obs_pred_pid[i].item()
        ppname = geom.panels[ppid].name if ppid >= 0 else "OFF"
        I_val = core.I[obs_mask][i].item()
        print(f"  [{int(h):3d},{int(k):3d},{int(l):3d}]  {pname:>5s}  {obs_fs[i]:7.1f}  "
              f"{obs_ss[i]:7.1f}  {obs_exc[i]:.4f}  {obs_d[i]:6.2f}  "
              f"{obs_peak[i]:5.0f}  {I_val:8.0f}  {obs_err_px[i]:7.1f}px  {ppname:>10s}")

    # 3. Check if unmatched spots land on the same panels or are scattered
    print(f"\n--- Panel distribution ---")
    for label, idx in [("Matched", matched_idx), ("Unmatched", unmatched_idx)]:
        panels = obs_pid[idx].numpy()
        unique, counts = np.unique(panels, return_counts=True)
        dist = ", ".join(f"{geom.panels[p].name}:{c}" for p, c in zip(unique, counts))
        print(f"  {label}: {dist}")

    # 4. Check if unmatched hkl have systematic indices (e.g. forbidden reflections)
    print(f"\n--- hkl index analysis ---")
    um_hkl = obs_hkl[unmatched_idx].numpy()
    h_vals, k_vals, l_vals = um_hkl[:, 0], um_hkl[:, 1], um_hkl[:, 2]
    # Check parity (all-even, all-odd, mixed)
    print(f"  h range: [{h_vals.min()}, {h_vals.max()}]")
    print(f"  k range: [{k_vals.min()}, {k_vals.max()}]")
    print(f"  l range: [{l_vals.min()}, {l_vals.max()}]")
    # Are these high-index reflections?
    max_idx = np.abs(um_hkl).max(axis=1)
    print(f"  max(|h|,|k|,|l|) range: [{max_idx.min()}, {max_idx.max()}]")

    # 5. Is there a pattern? Check if unmatched reciprocal vectors cluster
    um_h_lab = obs_h_lab[unmatched_idx]
    ma_h_lab = obs_h_lab[matched_idx]
    print(f"\n--- Reciprocal space analysis ---")
    print(f"  Matched |h_lab| range:   [{ma_h_lab.norm(dim=-1).min():.4f}, "
          f"{ma_h_lab.norm(dim=-1).max():.4f}] Å⁻¹")
    print(f"  Unmatched |h_lab| range: [{um_h_lab.norm(dim=-1).min():.4f}, "
          f"{um_h_lab.norm(dim=-1).max():.4f}] Å⁻¹")

    # 6. Check if unmatched could be explained by a DIFFERENT crystal in same image
    # Look for other crystals in the same image
    other_cids = []
    for c in range(core.n_crystals):
        if c != cid and core.image_id[c].item() == img_id:
            other_cids.append(c)
    print(f"\n--- Other crystals in image {img_id}: {len(other_cids)} ---")

    if other_cids:
        for oc in other_cids[:3]:  # check up to 3
            oc_astar = core.astar[oc].float()
            oc_bstar = core.bstar[oc].float()
            oc_cstar = core.cstar[oc].float()
            oc_UB = torch.stack([oc_astar, oc_bstar, oc_cstar], dim=-1)
            oc_cell = core.cell[oc]
            print(f"\n  Crystal {oc}: cell=({oc_cell[0]:.2f},{oc_cell[1]:.2f},{oc_cell[2]:.2f}) Å")

            # Check if any unmatched spots can be predicted by this crystal
            oc_h_lab = (oc_UB @ obs_hkl[unmatched_idx].float().T).T
            oc_exc = excitation_error(oc_h_lab, wavelength)
            oc_close = oc_exc < 2 * profile_rad
            print(f"    Unmatched spots with exc_err < 2×relp: {oc_close.sum().item()}/{len(unmatched_idx)}")

    # 7. What fraction are these across ALL crystals?
    print(f"\n--- Unmatched fraction across crystals ---")
    sample_cids = np.random.choice(core.n_crystals, min(50, core.n_crystals), replace=False)
    unmatched_fracs = []
    for sc in sample_cids:
        sc = int(sc)
        sc_mask = core.crystal_id == sc
        if sc_mask.sum() < 20:
            continue
        sc_UB = torch.stack([core.astar[sc], core.bstar[sc], core.cstar[sc]], dim=-1).float()
        sc_hkl = core.hkl[sc_mask].float()
        sc_h_lab = (sc_UB @ sc_hkl.T).T
        sc_exc = excitation_error(sc_h_lab, wavelength)
        sc_pred = sc_exc < 2 * profile_rad
        # Use on-detector check
        with torch.no_grad():
            _, _, sc_pid = ref_geom.recip_to_pixel(sc_h_lab, wavelength)
        sc_on = sc_pid >= 0
        n_matched = (sc_pred & sc_on).sum().item()
        n_total = sc_mask.sum().item()
        unmatched_fracs.append(1.0 - n_matched / n_total)
    unmatched_fracs = np.array(unmatched_fracs)
    print(f"  Across {len(unmatched_fracs)} sampled crystals:")
    print(f"  Unmatched fraction: mean={unmatched_fracs.mean():.1%}, "
          f"median={np.median(unmatched_fracs):.1%}, "
          f"std={unmatched_fracs.std():.1%}")


if __name__ == "__main__":
    main()
