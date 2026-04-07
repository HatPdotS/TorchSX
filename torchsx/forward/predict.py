"""Reflection prediction, diagnostics, and observation/ROI update."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from torchsx._core import _SFXStreamCore
from torchsx.scaling._orientation import CrystalOrientation
from torchsx.forward.coords import RefinableGeometry


@dataclass
class PredictedReflections:
    """Predicted reflections for all crystals."""

    hkl: Tensor              # (N_pred, 3) int16
    crystal_id: Tensor        # (N_pred,) int32
    fs: Tensor                # (N_pred,) float32 — predicted detector position
    ss: Tensor                # (N_pred,) float32
    panel_id: Tensor          # (N_pred,) int64
    excitation_error: Tensor  # (N_pred,) float32


@dataclass
class PredictionDiagnostics:
    """Diagnostics comparing predicted vs observed reflection sets."""

    # Counts
    n_predicted: int
    n_observed: int
    n_matched: int
    n_predicted_only: int
    n_observed_only: int

    # Rates
    hit_rate: float       # precision = matched / predicted
    recovery_rate: float  # recall = matched / observed

    # Position accuracy for matched spots
    position_error_px: Tensor     # (N_matched,) pixel displacement
    median_position_error: float
    frac_within_1px: float
    frac_within_3px: float

    # Excitation errors
    excitation_error_matched: Tensor       # (N_matched,)
    excitation_error_predicted_only: Tensor  # (N_predicted_only,)

    # Per-crystal rates: (N_crystals, 3) — [n_matched, n_pred, n_obs] per crystal
    per_crystal_counts: Tensor

    # Resolution-shelled rates: (n_shells, 5) —
    #   [d_min, d_max, n_matched, n_pred, n_obs] per shell
    shell_stats: Tensor

    def summary(self) -> str:
        """Format a human-readable report."""
        lines = ["Prediction quality:"]
        lines.append(
            f"  Predicted: {self.n_predicted:,} | "
            f"Observed: {self.n_observed:,} | "
            f"Matched: {self.n_matched:,}"
        )
        lines.append(
            f"  Hit rate:      {self.hit_rate * 100:.1f}%  "
            f"(of predictions found in observations)"
        )
        lines.append(
            f"  Recovery rate: {self.recovery_rate * 100:.1f}%  "
            f"(of observations found in predictions)"
        )
        lines.append(
            f"  Position: median {self.median_position_error:.1f} px, "
            f"{self.frac_within_1px * 100:.0f}% < 1 px, "
            f"{self.frac_within_3px * 100:.0f}% < 3 px"
        )

        po_pct = (
            self.n_predicted_only / self.n_predicted * 100
            if self.n_predicted > 0 else 0.0
        )
        oo_pct = (
            self.n_observed_only / self.n_observed * 100
            if self.n_observed > 0 else 0.0
        )
        lines.append(
            f"  Observed-only: {self.n_observed_only:,} ({oo_pct:.0f}%) | "
            f"Predicted-only: {self.n_predicted_only:,} ({po_pct:.0f}%)"
        )

        # Resolution shells
        if self.shell_stats.shape[0] > 0:
            lines.append("  Resolution shells:")
            parts = []
            for i in range(self.shell_stats.shape[0]):
                d_lo = self.shell_stats[i, 0].item()
                d_hi = self.shell_stats[i, 1].item()
                n_m = int(self.shell_stats[i, 2].item())
                n_p = int(self.shell_stats[i, 3].item())
                n_o = int(self.shell_stats[i, 4].item())
                hit = n_m / n_p * 100 if n_p > 0 else 0.0
                rec = n_m / n_o * 100 if n_o > 0 else 0.0
                parts.append(f"{d_hi:.1f}-{d_lo:.1f}A: hit={hit:.0f}% rec={rec:.0f}%")
            lines.append("    " + " | ".join(parts))

        return "\n".join(lines)


def _build_key_set(crystal_id: Tensor, hkl: Tensor) -> dict[tuple, int]:
    """Build (crystal_id, h, k, l) -> index mapping."""
    keys = {}
    cid = crystal_id.long()
    h = hkl.long()
    for i in range(hkl.shape[0]):
        key = (cid[i].item(), h[i, 0].item(), h[i, 1].item(), h[i, 2].item())
        keys[key] = i
    return keys


def compare_predictions(
    predicted: PredictedReflections,
    core: _SFXStreamCore,
    cell: Tensor,
    n_shells: int = 10,
    friedel: bool = True,
) -> PredictionDiagnostics:
    """Compare predicted reflections against observed reflections in core.

    Parameters
    ----------
    predicted : PredictedReflections
    core : _SFXStreamCore — observed data
    cell : (N_crystals, 6) unit cell (a,b,c in A, alpha,beta,gamma in deg)
    n_shells : number of resolution shells for shell statistics
    friedel : bool — if True, treat (h,k,l) and (-h,-k,-l) as equivalent
        when matching predictions to observations
    """
    from torchref.base.reciprocal.hkl import get_d_spacing

    pred_keys = _build_key_set(predicted.crystal_id, predicted.hkl)
    obs_keys = _build_key_set(core.crystal_id, core.hkl)

    pred_set = set(pred_keys.keys())
    obs_set = set(obs_keys.keys())

    if friedel:
        # Also build Friedel-mate key sets
        pred_friedel = {(c, -h, -k, -l): i for (c, h, k, l), i in pred_keys.items()}
        obs_friedel = {(c, -h, -k, -l) for (c, h, k, l) in obs_set}

        # A prediction matches if it directly matches OR its Friedel mate matches
        matched_keys = set()
        for key in pred_set:
            if key in obs_set:
                matched_keys.add(key)
            elif key in obs_friedel:
                matched_keys.add(key)

        # An observation is matched if it or its Friedel mate is predicted
        matched_obs_keys = set()
        for key in obs_set:
            if key in pred_set:
                matched_obs_keys.add(key)
            elif key in set(pred_friedel.keys()):
                matched_obs_keys.add(key)

        predicted_only_keys = pred_set - matched_keys
        observed_only_keys = obs_set - matched_obs_keys
    else:
        matched_keys = pred_set & obs_set
        matched_obs_keys = matched_keys
        predicted_only_keys = pred_set - obs_set
        observed_only_keys = obs_set - pred_set

    n_predicted = len(pred_set)
    n_observed = len(obs_set)
    n_matched = len(matched_keys)
    n_matched_obs = len(matched_obs_keys)
    n_predicted_only = len(predicted_only_keys)
    n_observed_only = len(observed_only_keys)

    hit_rate = n_matched / n_predicted if n_predicted > 0 else 0.0
    recovery_rate = n_matched_obs / n_observed if n_observed > 0 else 0.0

    # Position accuracy for matched spots — use direct matches only
    # (Friedel mates land at different detector positions, so position
    # comparison only makes sense for direct HKL matches)
    direct_matched = pred_set & obs_set
    if len(direct_matched) > 0:
        matched_pred_idx = []
        matched_obs_idx = []
        for key in direct_matched:
            matched_pred_idx.append(pred_keys[key])
            matched_obs_idx.append(obs_keys[key])
        matched_pred_idx = torch.tensor(matched_pred_idx, dtype=torch.long)
        matched_obs_idx = torch.tensor(matched_obs_idx, dtype=torch.long)

        dfs = predicted.fs[matched_pred_idx] - core.fs[matched_obs_idx]
        dss = predicted.ss[matched_pred_idx] - core.ss[matched_obs_idx]
        position_error = (dfs ** 2 + dss ** 2).sqrt()
        median_pos_err = position_error.median().item()
        frac_1px = (position_error < 1.0).float().mean().item()
        frac_3px = (position_error < 3.0).float().mean().item()

        exc_matched = predicted.excitation_error[matched_pred_idx]
    else:
        position_error = torch.zeros(0)
        median_pos_err = float("nan")
        frac_1px = 0.0
        frac_3px = 0.0
        exc_matched = torch.zeros(0)

    # Excitation errors for predicted-only
    if n_predicted_only > 0:
        po_idx = torch.tensor(
            [pred_keys[k] for k in predicted_only_keys], dtype=torch.long,
        )
        exc_pred_only = predicted.excitation_error[po_idx]
    else:
        exc_pred_only = torch.zeros(0)

    # Per-crystal counts — use matched_obs_keys for recovery denominator
    n_crystals = core.n_crystals
    per_crystal = torch.zeros(n_crystals, 3)  # [matched_obs, pred, obs]
    for key in matched_obs_keys:
        per_crystal[key[0], 0] += 1
    for key in pred_set:
        per_crystal[key[0], 1] += 1
    for key in obs_set:
        per_crystal[key[0], 2] += 1

    # Resolution-shelled stats — vectorized via integer hashing
    if n_shells > 0 and n_predicted > 0 and n_observed > 0 and cell.shape[0] > 0:
        median_cell = cell.median(dim=0).values
        d_pred = get_d_spacing(predicted.hkl.float(), median_cell)
        d_obs = get_d_spacing(core.hkl.float(), median_cell)

        all_d = torch.cat([d_pred, d_obs])
        d_min_all = all_d.min().item()
        d_max_all = all_d.max().item()

        inv_d2_min = 1.0 / max(d_max_all, 1e-6) ** 2
        inv_d2_max = 1.0 / max(d_min_all, 1e-6) ** 2
        shell_edges = torch.linspace(inv_d2_min, inv_d2_max, n_shells + 1)

        # Assign each reflection to a shell via searchsorted
        inv_d2_pred = 1.0 / d_pred.clamp(min=1e-6) ** 2
        inv_d2_obs = 1.0 / d_obs.clamp(min=1e-6) ** 2
        pred_shell = torch.searchsorted(shell_edges, inv_d2_pred).clamp(1, n_shells) - 1
        obs_shell = torch.searchsorted(shell_edges, inv_d2_obs).clamp(1, n_shells) - 1

        # Hash (crystal_id, h, k, l) into int64 for fast set membership
        def _hash_keys(cid: Tensor, hkl: Tensor) -> Tensor:
            c = cid.long()
            h = hkl[:, 0].long() + 500
            k = hkl[:, 1].long() + 500
            l = hkl[:, 2].long() + 500
            return c * 1000_000_000 + h * 1000_000 + k * 1000 + l

        pred_hash = _hash_keys(predicted.crystal_id, predicted.hkl)
        obs_hash = _hash_keys(core.crystal_id, core.hkl)

        if friedel:
            pred_friedel_hash = _hash_keys(
                predicted.crystal_id, -predicted.hkl,
            )
            # Obs is matched if its hash or Friedel hash is in pred set
            pred_hash_set = set(pred_hash.tolist())
            pred_friedel_set = set(pred_friedel_hash.tolist())
            obs_is_matched = torch.tensor(
                [h in pred_hash_set or h in pred_friedel_set for h in obs_hash.tolist()],
                dtype=torch.bool,
            )
        else:
            pred_hash_set = set(pred_hash.tolist())
            obs_is_matched = torch.tensor(
                [h in pred_hash_set for h in obs_hash.tolist()],
                dtype=torch.bool,
            )

        shell_stats = torch.zeros(n_shells, 5)
        for s in range(n_shells):
            lo_inv = shell_edges[s].item()
            hi_inv = shell_edges[s + 1].item()
            d_hi_shell = 1.0 / max(lo_inv, 1e-12) ** 0.5
            d_lo_shell = 1.0 / max(hi_inv, 1e-12) ** 0.5
            shell_stats[s, 0] = d_lo_shell
            shell_stats[s, 1] = d_hi_shell

            n_pred_shell = (pred_shell == s).sum().item()
            obs_in = obs_shell == s
            n_obs_shell = obs_in.sum().item()
            n_matched_shell = (obs_in & obs_is_matched).sum().item()

            shell_stats[s, 2] = n_matched_shell
            shell_stats[s, 3] = n_pred_shell
            shell_stats[s, 4] = n_obs_shell
    else:
        shell_stats = torch.zeros(0, 5)

    return PredictionDiagnostics(
        n_predicted=n_predicted,
        n_observed=n_observed,
        n_matched=n_matched,
        n_predicted_only=n_predicted_only,
        n_observed_only=n_observed_only,
        hit_rate=hit_rate,
        recovery_rate=recovery_rate,
        position_error_px=position_error,
        median_position_error=median_pos_err,
        frac_within_1px=frac_1px,
        frac_within_3px=frac_3px,
        excitation_error_matched=exc_matched,
        excitation_error_predicted_only=exc_pred_only,
        per_crystal_counts=per_crystal,
        shell_stats=shell_stats,
    )


def predict_reflections(
    orientation: CrystalOrientation,
    geometry: RefinableGeometry,
    cell: Tensor,
    d_min: float,
    wavelength: float,
    sigma_cutoff: float = 3.0,
    relp_sigma: float | None = None,
) -> PredictedReflections:
    """Predict which reflections are observable given current orientations.

    For each crystal:
    1. Generate all possible HKL up to d_min
    2. Compute h_lab = RUB @ hkl
    3. Compute excitation error r = |h_lab + s0| - 1/lambda
    4. Keep reflections with |r| < sigma_cutoff * sigma_eff
    5. Project onto detector and keep those that land on a panel

    Parameters
    ----------
    orientation : CrystalOrientation
    geometry : RefinableGeometry
    cell : (N_crystals, 6) unit cell (a,b,c in A, alpha,beta,gamma in deg)
    d_min : float — resolution cutoff in Angstroms
    wavelength : float — beam wavelength in Angstroms
    sigma_cutoff : float — cutoff in units of relp sigma
    relp_sigma : float | None — effective relp width (A^-1). If None, uses 0.05.
    """
    from torchref.base.reciprocal.hkl import generate_possible_hkl

    sigma_eff = relp_sigma if relp_sigma is not None else 0.05
    inv_wl = 1.0 / wavelength
    s0 = torch.zeros(3, device=cell.device)
    s0[2] = inv_wl

    # Use median cell for HKL generation
    median_cell = cell.median(dim=0).values
    possible_hkl = generate_possible_hkl(median_cell, d_min)  # (N_hkl, 3)
    possible_hkl = possible_hkl.to(cell.device)

    RUB = orientation.RUB()  # (N_crystals, 3, 3)
    n_crystals = RUB.shape[0]
    n_hkl = possible_hkl.shape[0]

    hkl_list = []
    cid_list = []
    fs_list = []
    ss_list = []
    pid_list = []
    r_list = []

    h_float = possible_hkl.float()  # (N_hkl, 3)

    for c in range(n_crystals):
        # h_lab = RUB[c] @ hkl^T → (3, N_hkl) → (N_hkl, 3)
        h_lab = (RUB[c] @ h_float.T).T  # (N_hkl, 3)

        # Excitation error: r = |h_lab + s0| - 1/lambda
        q = h_lab + s0
        r = q.norm(dim=-1) - inv_wl  # (N_hkl,)

        # Keep within sigma cutoff
        keep = r.abs() < sigma_cutoff * sigma_eff
        if keep.sum() == 0:
            continue

        h_lab_keep = h_lab[keep]
        hkl_keep = possible_hkl[keep]
        r_keep = r[keep]

        # Project to detector
        with torch.no_grad():
            fs_pred, ss_pred, pid_pred = geometry.recip_to_pixel(h_lab_keep, wavelength)

        # Keep only reflections that land on detector
        on_det = pid_pred >= 0
        if on_det.sum() == 0:
            continue

        n_kept = on_det.sum().item()
        hkl_list.append(hkl_keep[on_det])
        cid_list.append(torch.full((n_kept,), c, dtype=torch.int32, device=cell.device))
        fs_list.append(fs_pred[on_det])
        ss_list.append(ss_pred[on_det])
        pid_list.append(pid_pred[on_det])
        r_list.append(r_keep[on_det])

    if not hkl_list:
        empty_f = torch.zeros(0, device=cell.device)
        empty_i16 = torch.zeros(0, 3, dtype=torch.int16, device=cell.device)
        empty_i32 = torch.zeros(0, dtype=torch.int32, device=cell.device)
        empty_i64 = torch.zeros(0, dtype=torch.long, device=cell.device)
        return PredictedReflections(
            hkl=empty_i16, crystal_id=empty_i32,
            fs=empty_f, ss=empty_f, panel_id=empty_i64,
            excitation_error=empty_f,
        )

    return PredictedReflections(
        hkl=torch.cat(hkl_list).to(torch.int16),
        crystal_id=torch.cat(cid_list),
        fs=torch.cat(fs_list),
        ss=torch.cat(ss_list),
        panel_id=torch.cat(pid_list),
        excitation_error=torch.cat(r_list),
    )


def update_observations(
    core: _SFXStreamCore,
    predicted: PredictedReflections,
    roi_store: "ROIStore | None" = None,
    threshold: float = 0.5,
    mode: str = "position_only",
    min_excitation_error: float | None = None,
    data_path: str | None = None,
    n_workers: int = 1,
) -> tuple[_SFXStreamCore, "ROIStore | None", PredictionDiagnostics]:
    """Compare predicted vs current observations and update core + ROI store.

    1. Match predicted <-> current by (crystal_id, hkl)
    2. Identify: kept, moved (position changed), new, removed
    3. Update core based on mode
    4. Re-excise ROIs for moved + new observations (if roi_store provided)

    Parameters
    ----------
    core : _SFXStreamCore
    predicted : PredictedReflections
    roi_store : ROIStore | None
    threshold : pixel displacement for re-excision
    mode : "position_only" | "expand" | "full"
        - position_only: only update fs/ss for moved spots
        - expand: also add predicted-only spots (with I=0, sigma=inf)
        - full: expand + remove observed-only spots
    min_excitation_error : float | None
        Gate for adding predicted-only spots. Only spots with
        |excitation_error| < min_excitation_error are added.
        If None, all predicted-only spots are added.
    data_path : HDF5 dataset path for ROI excision
    n_workers : parallel workers for ROI excision

    Returns
    -------
    (updated_core, updated_roi_store, diagnostics)
    """
    import dataclasses

    if mode not in ("position_only", "expand", "full"):
        raise ValueError(f"Unknown mode {mode!r}; use 'position_only', 'expand', or 'full'")

    # Build lookups
    pred_keys = _build_key_set(predicted.crystal_id, predicted.hkl)
    obs_keys = _build_key_set(core.crystal_id, core.hkl)

    pred_set = set(pred_keys.keys())
    obs_set = set(obs_keys.keys())

    # Classify observations
    kept_cur = []     # current obs indices: matched, position unchanged
    moved_cur = []    # current obs indices: matched, position changed
    moved_pred = []   # corresponding predicted indices
    new_pred = []     # predicted indices for brand new reflections
    observed_only = []  # current obs indices not in predictions

    for key in pred_set & obs_set:
        pi = pred_keys[key]
        ci = obs_keys[key]
        dfs = abs(core.fs[ci].item() - predicted.fs[pi].item())
        dss = abs(core.ss[ci].item() - predicted.ss[pi].item())
        if (dfs ** 2 + dss ** 2) ** 0.5 > threshold:
            moved_cur.append(ci)
            moved_pred.append(pi)
        else:
            kept_cur.append(ci)

    for key in pred_set - obs_set:
        pi = pred_keys[key]
        # Apply excitation error gate
        if min_excitation_error is not None:
            if abs(predicted.excitation_error[pi].item()) > min_excitation_error:
                continue
        new_pred.append(pi)

    for key in obs_set - pred_set:
        observed_only.append(obs_keys[key])

    # Compute diagnostics using compare_predictions
    diag = compare_predictions(predicted, core, core.cell)

    # --- mode: position_only ---
    new_fs = core.fs.clone()
    new_ss = core.ss.clone()
    for ci, pi in zip(moved_cur, moved_pred):
        new_fs[ci] = predicted.fs[pi]
        new_ss[ci] = predicted.ss[pi]

    if mode == "position_only":
        updated_core = dataclasses.replace(core, fs=new_fs, ss=new_ss)
        updated_roi = roi_store
        if roi_store is not None:
            updated_roi = roi_store.update(
                updated_core,
                threshold=threshold,
                data_path=data_path,
                n_workers=n_workers,
            )
        return updated_core, updated_roi, diag

    # --- mode: expand or full ---
    # Start with all current observations (with updated positions)
    if mode == "full":
        # Remove observed-only spots
        obs_only_set = set(observed_only)
        keep_mask = torch.ones(core.n_observations, dtype=torch.bool)
        for idx in obs_only_set:
            keep_mask[idx] = False

        # Apply position updates before masking
        new_fs_masked = new_fs[keep_mask]
        new_ss_masked = new_ss[keep_mask]

        # Gather kept observation tensors
        kept_hkl = core.hkl[keep_mask]
        kept_I = core.I[keep_mask]
        kept_sigma = core.sigma[keep_mask]
        kept_bg = core.background[keep_mask]
        kept_peak = core.peak[keep_mask]
        kept_panel = core.panel_id[keep_mask]
        kept_cid = core.crystal_id[keep_mask]
    else:
        # expand: keep all current observations
        kept_hkl = core.hkl
        kept_I = core.I
        kept_sigma = core.sigma
        kept_bg = core.background
        kept_peak = core.peak
        kept_panel = core.panel_id
        kept_cid = core.crystal_id
        new_fs_masked = new_fs
        new_ss_masked = new_ss

    # Add predicted-only spots
    n_new = len(new_pred)
    if n_new > 0:
        new_idx = torch.tensor(new_pred, dtype=torch.long)
        add_hkl = predicted.hkl[new_idx]
        add_cid = predicted.crystal_id[new_idx]
        add_fs = predicted.fs[new_idx]
        add_ss = predicted.ss[new_idx]
        add_panel = predicted.panel_id[new_idx].to(core.panel_id.dtype)
        add_I = torch.zeros(n_new, dtype=core.I.dtype)
        add_sigma = torch.full((n_new,), float("inf"), dtype=core.sigma.dtype)
        add_bg = torch.zeros(n_new, dtype=core.background.dtype)
        add_peak = torch.zeros(n_new, dtype=core.peak.dtype)

        final_hkl = torch.cat([kept_hkl, add_hkl])
        final_I = torch.cat([kept_I, add_I])
        final_sigma = torch.cat([kept_sigma, add_sigma])
        final_bg = torch.cat([kept_bg, add_bg])
        final_peak = torch.cat([kept_peak, add_peak])
        final_fs = torch.cat([new_fs_masked, add_fs])
        final_ss = torch.cat([new_ss_masked, add_ss])
        final_panel = torch.cat([kept_panel, add_panel])
        final_cid = torch.cat([kept_cid, add_cid])
    else:
        final_hkl = kept_hkl
        final_I = kept_I
        final_sigma = kept_sigma
        final_bg = kept_bg
        final_peak = kept_peak
        final_fs = new_fs_masked
        final_ss = new_ss_masked
        final_panel = kept_panel
        final_cid = kept_cid

    # Recompute n_refl per crystal
    new_n_refl = torch.zeros(core.n_crystals, dtype=core.n_refl.dtype)
    new_n_refl.scatter_add_(
        0, final_cid.long(),
        torch.ones(final_cid.shape[0], dtype=core.n_refl.dtype),
    )

    updated_core = dataclasses.replace(
        core,
        hkl=final_hkl,
        I=final_I,
        sigma=final_sigma,
        background=final_bg,
        peak=final_peak,
        fs=final_fs,
        ss=final_ss,
        panel_id=final_panel,
        crystal_id=final_cid,
        n_refl=new_n_refl,
    )

    # Re-excise ROIs if store provided
    updated_roi = roi_store
    if roi_store is not None:
        updated_roi = roi_store.update(
            updated_core,
            threshold=threshold,
            data_path=data_path,
            n_workers=n_workers,
        )

    return updated_core, updated_roi, diag
