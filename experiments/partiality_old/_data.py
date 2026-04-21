"""Data loading, B/G correction, and pair construction for rank refinement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
import numpy as np


@dataclass
class PairData:
    """Precomputed pair data for rank concordance refinement.

    Per-observation arrays (N_obs,):
    """
    hkl_obs: Tensor       # (N, 3) int16 — Miller indices
    crystal_id: Tensor    # (N,) int32
    hkl_id: Tensor        # (N,) int64 — unique HKL index
    I_corr: Tensor        # (N,) float32 — B/G-corrected intensity
    s_sq: Tensor          # (N,) float32 — (1/(2d))²

    # Per-crystal
    UB: Tensor            # (C, 3, 3) float32
    lambda_0: Tensor      # (C,) float32 — per-crystal wavelength (Å)

    # Pair index structure
    pair_i: Tensor        # (P,) int64 — obs index of first in pair
    pair_j: Tensor        # (P,) int64 — obs index of second in pair
    pair_sign: Tensor     # (P,) float32 — sign(I_i - I_j)
    pair_weight: Tensor   # (P,) float32

    n_crystals: int
    n_obs: int
    n_pairs: int

    def to(self, device):
        """Move all tensors to device."""
        fields = {}
        for k in ["hkl_obs", "crystal_id", "hkl_id", "I_corr", "s_sq",
                   "UB", "lambda_0", "pair_i", "pair_j", "pair_sign", "pair_weight"]:
            fields[k] = getattr(self, k).to(device)
        return PairData(**fields, n_crystals=self.n_crystals,
                        n_obs=self.n_obs, n_pairs=self.n_pairs)


def _concat_cores(cores):
    """Concatenate multiple _SFXStreamCore objects, offsetting IDs."""
    import dataclasses

    cryst_offset = 0
    img_offset = 0
    all_hkl, all_I, all_sigma, all_bg, all_peak = [], [], [], [], []
    all_fs, all_ss, all_pid, all_cid = [], [], [], []
    all_astar, all_bstar, all_cstar, all_cell = [], [], [], []
    all_profile_rad, all_resolution, all_n_refl = [], [], []
    all_image_id, all_crystal_in_image = [], []
    all_beam_energy, all_det_shift_x, all_det_shift_y, all_n_peaks = [], [], [], []

    for c in cores:
        all_hkl.append(c.hkl)
        all_I.append(c.I)
        all_sigma.append(c.sigma)
        all_bg.append(c.background)
        all_peak.append(c.peak)
        all_fs.append(c.fs)
        all_ss.append(c.ss)
        all_pid.append(c.panel_id)
        all_cid.append(c.crystal_id + cryst_offset)
        all_astar.append(c.astar)
        all_bstar.append(c.bstar)
        all_cstar.append(c.cstar)
        all_cell.append(c.cell)
        all_profile_rad.append(c.profile_rad)
        all_resolution.append(c.resolution)
        all_n_refl.append(c.n_refl)
        all_image_id.append(c.image_id + img_offset)
        all_crystal_in_image.append(c.crystal_in_image)
        all_beam_energy.append(c.beam_energy)
        all_det_shift_x.append(c.det_shift_x)
        all_det_shift_y.append(c.det_shift_y)
        all_n_peaks.append(c.n_peaks)
        cryst_offset += c.astar.shape[0]
        img_offset += c.beam_energy.shape[0]

    from torchsx._core import _SFXStreamCore
    return _SFXStreamCore(
        hkl=torch.cat(all_hkl),
        I=torch.cat(all_I),
        sigma=torch.cat(all_sigma),
        background=torch.cat(all_bg),
        peak=torch.cat(all_peak),
        fs=torch.cat(all_fs),
        ss=torch.cat(all_ss),
        panel_id=torch.cat(all_pid),
        crystal_id=torch.cat(all_cid),
        astar=torch.cat(all_astar),
        bstar=torch.cat(all_bstar),
        cstar=torch.cat(all_cstar),
        cell=torch.cat(all_cell),
        profile_rad=torch.cat(all_profile_rad),
        resolution=torch.cat(all_resolution),
        n_refl=torch.cat(all_n_refl),
        image_id=torch.cat(all_image_id),
        crystal_in_image=torch.cat(all_crystal_in_image),
        beam_energy=torch.cat(all_beam_energy),
        det_shift_x=torch.cat(all_det_shift_x),
        det_shift_y=torch.cat(all_det_shift_y),
        n_peaks=torch.cat(all_n_peaks),
        geometry=cores[0].geometry,
        metadata=cores[0].metadata,
        spacegroup=cores[0].spacegroup,
        source_path=cores[0].source_path,
    )


def load_and_prepare(
    sfx_paths: str | Path | list[str | Path],
    *,
    device: str = "cpu",
    I_threshold: float = 20.0,
    min_mult: int = 3,
    max_pairs_per_hkl: int = 500,
) -> PairData:
    """Load SFX data, apply B/G correction, build pair index.

    Parameters
    ----------
    sfx_paths : path(s) to .sfx file(s) — multiple are concatenated
    I_threshold : minimum corrected intensity for pair inclusion
    min_mult : minimum multiplicity per HKL to form pairs
    max_pairs_per_hkl : cap on pairs per HKL (random subsample)
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from torchsx.io._safetensors import load_sfx
    from torchsx.merge._hkl import compute_hkl_index

    if isinstance(sfx_paths, (str, Path)):
        sfx_paths = [sfx_paths]

    cores = []
    for p in sfx_paths:
        print(f"Loading {Path(p).name} ...", end=" ")
        c = load_sfx(str(p))
        print(f"{c.hkl.shape[0]:,} obs, {c.astar.shape[0]:,} crystals")
        cores.append(c)

    if len(cores) == 1:
        core = cores[0]
    else:
        core = _concat_cores(cores)

    N = core.hkl.shape[0]
    C = core.astar.shape[0]
    print(f"  Total: {N:,} obs, {C:,} crystals")

    # HKL indexing
    hkl_idx = compute_hkl_index(core.hkl)
    hkl_id = hkl_idx.hkl_id
    n_unique = hkl_idx.n_unique

    # Resolution
    from torchsx.merge._neural_model import compute_excitation_error
    wavelength = 12398.42 / core.beam_energy.median().item()
    _, q_lab = compute_excitation_error(
        core.hkl, core.astar, core.bstar, core.cstar,
        core.crystal_id, wavelength,
    )
    q_mag = q_lab.norm(dim=1)
    s_sq = (q_mag / 2) ** 2  # (sinθ/λ)²

    # UB matrices
    UB = torch.stack([core.astar, core.bstar, core.cstar], dim=2)  # (C, 3, 3)

    # Per-crystal wavelength from beam energy
    image_id = core.image_id if hasattr(core, "image_id") else torch.zeros(C, dtype=torch.int32)
    beam_E = core.beam_energy[image_id.long()]  # (C,) eV
    lambda_0 = 12398.42 / beam_E  # (C,) Å

    # ── Fit B & G ──
    print("Fitting B & G scale factors...")
    I_corr, log_G, B = _fit_BG(core.I, hkl_id, core.crystal_id, s_sq, n_unique, C)
    print(f"  median |log G| = {log_G.abs().median():.3f}, median B = {B.median():.3f}")

    # ── Build pairs ──
    print("Building pair index...")
    pair_i, pair_j, pair_sign, pair_weight = _build_pairs(
        I_corr, hkl_id, core.crystal_id, n_unique,
        I_threshold=I_threshold,
        min_mult=min_mult,
        max_pairs_per_hkl=max_pairs_per_hkl,
    )
    print(f"  {len(pair_i):,} pairs from {n_unique:,} HKLs")

    pd = PairData(
        hkl_obs=core.hkl,
        crystal_id=core.crystal_id,
        hkl_id=hkl_id,
        I_corr=I_corr,
        s_sq=s_sq,
        UB=UB,
        lambda_0=lambda_0,
        pair_i=pair_i,
        pair_j=pair_j,
        pair_sign=pair_sign,
        pair_weight=pair_weight,
        n_crystals=C,
        n_obs=N,
        n_pairs=len(pair_i),
    )
    return pd.to(device)


def _fit_BG(I_obs, hkl_id, crystal_id, s_sq, n_unique, n_crystals):
    """Fit per-crystal G and B via closed-form log-space OLS.

    I_obs = G_c × exp(-2 B_c s²) × I_true_h × noise
    log(I_obs) = log(G_c) - 2 B_c s² + log(I_true_h) + ε

    Iterates: estimate I_true from current corrections, re-fit G/B.
    """
    pos = I_obs > 0
    log_I = torch.zeros_like(I_obs)
    log_I[pos] = torch.log(I_obs[pos])

    # Initial merged I (mean of positives per HKL)
    I_sum = torch.zeros(n_unique)
    I_cnt = torch.zeros(n_unique)
    I_sum.scatter_add_(0, hkl_id[pos], I_obs[pos])
    I_cnt.scatter_add_(0, hkl_id[pos], torch.ones_like(I_obs[pos]))
    I_merged = I_sum / I_cnt.clamp(min=1)

    log_G = torch.zeros(n_crystals)
    B = torch.zeros(n_crystals)

    for iteration in range(3):
        # Residuals in log space
        log_I_merged = torch.log(I_merged.clamp(min=1e-6))
        resid = torch.zeros_like(I_obs)
        resid[pos] = log_I[pos] - log_I_merged[hkl_id[pos]]

        # Per-crystal 2×2 OLS: resid = a + b×(-2s²) where a=log_G, b=B
        cid = crystal_id.long()
        x = -2.0 * s_sq  # design matrix column for B

        # Accumulate per crystal: X'X and X'y (only positive obs)
        sum_1 = torch.zeros(n_crystals)
        sum_x = torch.zeros(n_crystals)
        sum_xx = torch.zeros(n_crystals)
        sum_r = torch.zeros(n_crystals)
        sum_rx = torch.zeros(n_crystals)

        sum_1.scatter_add_(0, cid[pos], torch.ones_like(I_obs[pos]))
        sum_x.scatter_add_(0, cid[pos], x[pos])
        sum_xx.scatter_add_(0, cid[pos], x[pos] ** 2)
        sum_r.scatter_add_(0, cid[pos], resid[pos])
        sum_rx.scatter_add_(0, cid[pos], resid[pos] * x[pos])

        # Ridge: λ = 0.01 × n_per_crystal
        ridge = 0.01 * sum_1.clamp(min=1)
        det = (sum_1 + ridge) * (sum_xx + ridge) - sum_x ** 2
        det = det.clamp(min=1e-10)

        log_G = ((sum_xx + ridge) * sum_r - sum_x * sum_rx) / det
        B = ((sum_1 + ridge) * sum_rx - sum_x * sum_r) / det

        # Zero-mean log_G
        log_G = log_G - log_G.mean()

        # Update I_merged with corrections applied
        correction = torch.exp(log_G[cid] + B[cid] * x)
        I_corrected = I_obs / correction.clamp(min=1e-6)
        I_sum.zero_()
        I_cnt.zero_()
        pos_c = I_corrected > 0
        I_sum.scatter_add_(0, hkl_id[pos_c], I_corrected[pos_c])
        I_cnt.scatter_add_(0, hkl_id[pos_c], torch.ones_like(I_corrected[pos_c]))
        I_merged = I_sum / I_cnt.clamp(min=1)

    I_corr = I_obs / torch.exp(log_G[cid] + B[cid] * x).clamp(min=1e-6)
    return I_corr, log_G, B


def _build_pairs(I_corr, hkl_id, crystal_id, n_unique, *,
                 I_threshold, min_mult, max_pairs_per_hkl):
    """Build (pair_i, pair_j, pair_sign, pair_weight) for all qualifying HKLs."""
    hkl_np = hkl_id.numpy()
    I_np = I_corr.numpy()
    cid_np = crystal_id.numpy()

    # Sort by HKL for contiguous slicing
    order = np.argsort(hkl_np)
    boundaries = np.searchsorted(hkl_np[order], np.arange(n_unique + 1))

    all_i, all_j, all_sign, all_weight = [], [], [], []
    rng = np.random.RandomState(42)

    for uid in range(n_unique):
        lo, hi = boundaries[uid], boundaries[uid + 1]
        if hi - lo < min_mult:
            continue

        idx = order[lo:hi]
        I_h = I_np[idx]

        # Filter to strong observations
        strong = I_h > I_threshold
        if strong.sum() < min_mult:
            continue
        idx = idx[strong]
        I_h = I_h[strong]
        n = len(idx)

        # Form all pairs (i < j in local indexing)
        ii, jj = np.triu_indices(n, k=1)
        if len(ii) > max_pairs_per_hkl:
            sel = rng.choice(len(ii), max_pairs_per_hkl, replace=False)
            ii, jj = ii[sel], jj[sel]

        obs_i = idx[ii]
        obs_j = idx[jj]
        sign_ij = np.sign(I_h[ii] - I_h[jj]).astype(np.float32)

        # Drop ties
        nonzero = sign_ij != 0
        obs_i = obs_i[nonzero]
        obs_j = obs_j[nonzero]
        sign_ij = sign_ij[nonzero]

        if len(obs_i) == 0:
            continue

        # Weight: min intensity of the pair (stronger pairs are more informative)
        w = np.minimum(I_h[ii[nonzero]], I_h[jj[nonzero]])
        w = np.sqrt(w.clip(min=1))  # sqrt to prevent strong-pair dominance

        all_i.append(obs_i)
        all_j.append(obs_j)
        all_sign.append(sign_ij)
        all_weight.append(w)

    pair_i = torch.from_numpy(np.concatenate(all_i)).long()
    pair_j = torch.from_numpy(np.concatenate(all_j)).long()
    pair_sign = torch.from_numpy(np.concatenate(all_sign))
    pair_weight = torch.from_numpy(np.concatenate(all_weight))

    return pair_i, pair_j, pair_sign, pair_weight
