"""Spectral models: per-crystal energy distribution for spectral integration."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

HC_EV_A = 12398.4  # hc in eV*A


class GaussianSpectrum(nn.Module):
    """Gaussian spectral model with per-crystal energy and bandwidth.

    Models the incident spectrum as a Gaussian in wavelength space,
    discretized on a fixed grid for numerical integration.

    Parameters
    ----------
    n_crystals : int
        Number of crystals.
    mean_energy_eV : float
        Mean photon energy in eV.
    sigma_eV : float
        Initial bandwidth sigma in eV.
    n_bins : int
        Number of spectral bins.
    n_sigma : float
        Half-width of the wavelength grid in units of sigma.
    """

    def __init__(
        self,
        n_crystals: int,
        mean_energy_eV: float,
        sigma_eV: float = 30.0,
        n_bins: int = 50,
        n_sigma: float = 3.0,
    ):
        super().__init__()
        self.log_energy = nn.Parameter(
            torch.full((n_crystals,), math.log(mean_energy_eV))
        )
        self.log_sigma_eV = nn.Parameter(torch.tensor(math.log(sigma_eV)))
        self._n_bins = n_bins
        self._n_sigma = n_sigma

        # Build wavelength grid centered on mean wavelength
        mean_wl = HC_EV_A / mean_energy_eV
        # sigma in wavelength: d(lambda)/dE * sigma_E ~ lambda^2 / hc * sigma_E
        sigma_wl = mean_wl ** 2 / HC_EV_A * sigma_eV
        wl_min = mean_wl - n_sigma * sigma_wl
        wl_max = mean_wl + n_sigma * sigma_wl
        self.register_buffer(
            "lambda_bins", torch.linspace(wl_min, wl_max, n_bins)
        )

    @property
    def n_bins(self) -> int:
        return self._n_bins

    def wavelengths(self) -> Tensor:
        """Return wavelength grid in Angstroms. Shape: (n_bins,)."""
        return self.lambda_bins

    def d_lambda(self) -> float:
        """Bin spacing in Angstroms."""
        return (self.lambda_bins[-1] - self.lambda_bins[0]).item() / max(self._n_bins - 1, 1)

    def spectrum(self, crystal_id: Tensor) -> Tensor:
        """Compute normalized spectral weights for given crystals.

        Parameters
        ----------
        crystal_id : (N,) int — crystal index per observation

        Returns
        -------
        weights : (N, n_bins) float — normalized Gaussian weights
        """
        energy = torch.exp(self.log_energy[crystal_id.long()])  # (N,) eV
        sigma_eV = torch.exp(self.log_sigma_eV)  # scalar

        # Convert energy grid to per-crystal wavelengths
        wl_center = HC_EV_A / energy  # (N,) A

        # sigma in wavelength space: approx lambda^2 / hc * sigma_E
        sigma_wl = wl_center ** 2 / HC_EV_A * sigma_eV  # (N,)

        # Evaluate Gaussian at each bin
        # lambda_bins: (n_bins,), wl_center: (N,), sigma_wl: (N,)
        diff = self.lambda_bins.unsqueeze(0) - wl_center.unsqueeze(1)  # (N, n_bins)
        log_w = -0.5 * (diff / sigma_wl.unsqueeze(1)) ** 2
        w = torch.exp(log_w)

        # Normalize so weights sum to 1 per observation
        return w / w.sum(dim=1, keepdim=True).clamp(min=1e-30)


# ---------------------------------------------------------------------------
# PCA spectral overlap model
# ---------------------------------------------------------------------------


@dataclass
class SpectralOverlapGrids:
    """Result of precompute_spectral_overlap: grids + per-shot coefficients.

    Attributes
    ----------
    mean_grid : (n_ds, n_vr, n_sr) — mean overlap profile on 3D grid
    mode_grids : (n_modes, n_ds, n_vr, n_sr) — PCA mode grids
    coeffs : (n_shots, n_modes) — per-shot PCA coefficients
    ds_r_range : (min, max) of Δs_r grid in Å⁻¹
    v_r_range : (min, max) of v_r grid in Å⁻²
    log_sr_range : (min, max) of log(σ_r) grid
    explained_variance : (n_modes,) — fraction of variance per mode
    """
    mean_grid: np.ndarray
    mode_grids: np.ndarray
    coeffs: np.ndarray
    ds_r_range: tuple[float, float]
    v_r_range: tuple[float, float]
    log_sr_range: tuple[float, float]
    explained_variance: np.ndarray

    def save(self, path: str) -> None:
        """Save grids to .npz file."""
        np.savez_compressed(
            path,
            mean_grid=self.mean_grid,
            mode_grids=self.mode_grids,
            coeffs=self.coeffs,
            ds_r_range=np.array(self.ds_r_range),
            v_r_range=np.array(self.v_r_range),
            log_sr_range=np.array(self.log_sr_range),
            explained_variance=self.explained_variance,
        )

    @classmethod
    def load(cls, path: str) -> SpectralOverlapGrids:
        """Load grids from .npz file."""
        d = np.load(path)
        return cls(
            mean_grid=d["mean_grid"],
            mode_grids=d["mode_grids"],
            coeffs=d["coeffs"],
            ds_r_range=tuple(d["ds_r_range"].tolist()),
            v_r_range=tuple(d["v_r_range"].tolist()),
            log_sr_range=tuple(d["log_sr_range"].tolist()),
            explained_variance=d["explained_variance"],
        )


def precompute_spectral_overlap(
    energy_eV: np.ndarray,
    spectra: np.ndarray,
    *,
    n_modes: int = 5,
    n_ds: int = 200,
    n_vr: int = 50,
    n_sr: int = 15,
    ds_r_range: tuple[float, float] = (-0.005, 0.005),
    v_r_range: tuple[float, float] = (0.01, 0.8),
    sigma_r_range: tuple[float, float] = (0.0001, 0.003),
    max_shots_pca: int = 2000,
    cache_path: str | None = None,
) -> SpectralOverlapGrids:
    """Precompute PCA-compressed spectral overlap grids from measured spectra.

    The spectral integral factorizes as:
        I(Δs_r, |Δs_t|) = O(Δs_r, v_r, σ_r) × exp(-|Δs_t|²/(2σ_t²))

    This function computes O on a 3D (Δs_r, v_r, σ_r) grid for each measured
    spectrum, then PCA-compresses across shots.

    Parameters
    ----------
    energy_eV : (n_bins,) — energy axis in eV (shared across all shots)
    spectra : (n_shots, n_bins) — measured spectral intensities (counts)
    n_modes : number of PCA modes to retain
    n_ds, n_vr, n_sr : grid sizes for Δs_r, v_r, σ_r axes
    ds_r_range : min/max of Δs_r grid in Å⁻¹
    v_r_range : min/max of v_r (= d*/λ, the radial sweep rate) in Å⁻²
    sigma_r_range : min/max of radial relp width σ_r in Å⁻¹
    max_shots_pca : max shots used for PCA basis (subsampled for speed).
        All shots still get coefficients projected onto the basis.
    cache_path : optional path to .npz file for caching. If the file exists,
        grids are loaded from cache (skipping computation). Otherwise, grids
        are computed and saved to this path.

    Returns
    -------
    SpectralOverlapGrids with mean, modes, and per-shot coefficients
    """
    import os
    if cache_path is not None and os.path.exists(cache_path):
        print(f"  Loading cached PCA grids from {cache_path}")
        return SpectralOverlapGrids.load(cache_path)

    import torch as _torch

    n_shots, n_bins = spectra.shape
    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"  PCA precomputation on GPU: {_torch.cuda.get_device_name()}")

    # Convert energy axis to wavelength (Å), increasing order
    wl_bins = HC_EV_A / energy_eV.astype(np.float64)
    if wl_bins[0] > wl_bins[-1]:
        wl_bins = wl_bins[::-1].copy()
        spectra = spectra[:, ::-1].copy()
    d_wl = float(np.abs(wl_bins[1] - wl_bins[0]))
    wl_center = float(np.median(wl_bins))

    # Normalize each spectrum to unit area in wavelength space
    area = spectra.astype(np.float64).sum(axis=1, keepdims=True) * d_wl
    w_norm_np = spectra.astype(np.float32) / np.clip(area, 1e-30, None).astype(np.float32)
    w_norm = _torch.from_numpy(w_norm_np).to(device)  # (n_shots, n_bins)
    wl_t = _torch.from_numpy(wl_bins.astype(np.float32)).to(device)

    # Build grids on device
    ds_r_grid = _torch.linspace(ds_r_range[0], ds_r_range[1], n_ds, device=device)
    v_r_vals = _torch.linspace(v_r_range[0], v_r_range[1], n_vr, device=device)
    log_sr_grid = np.linspace(
        np.log(sigma_r_range[0]), np.log(sigma_r_range[1]), n_sr
    )
    sr_vals = _torch.from_numpy(np.exp(log_sr_grid).astype(np.float32)).to(device)
    n_grid = n_ds * n_vr * n_sr

    # ── Build all kernels at once (vectorized on GPU) ─────────────────
    # kernel[v, s, b, d] = exp(-(ds_r[d] + v_r[v]*(wl[b]-wl_c))² / (2σ_r[s]²))
    s_off = v_r_vals[:, None] * (wl_t[None, :] - wl_center)  # (n_vr, n_bins)
    arg = ds_r_grid[None, None, None, :] + s_off[:, None, :, None]  # (n_vr, n_sr, n_bins, n_ds)
    inv_2s2 = 1.0 / (2.0 * sr_vals[None, :, None, None] ** 2)
    kernel = _torch.exp(-arg ** 2 * inv_2s2)  # (n_vr, n_sr, n_bins, n_ds)
    del arg, inv_2s2, s_off
    kernel = kernel.reshape(n_vr * n_sr, n_bins, n_ds)  # (G, n_bins, n_ds)

    def _compute_overlaps(w: _torch.Tensor) -> _torch.Tensor:
        """(n, n_bins) × (G, n_bins, n_ds) → (n, G*n_ds) via einsum on device."""
        out = _torch.einsum("nb, gbd -> ngd", w, kernel)
        return (d_wl * out).reshape(w.shape[0], -1)

    # Subsample for PCA basis
    rng = np.random.RandomState(42)
    if n_shots > max_shots_pca:
        pca_idx = rng.choice(n_shots, max_shots_pca, replace=False)
    else:
        pca_idx = np.arange(n_shots)
    w_pca = w_norm[pca_idx]

    # ── Pass 1: PCA basis from subsample ──────────────────────────────
    overlaps_pca = _compute_overlaps(w_pca)  # (n_pca, n_grid) on device
    grid_mean = overlaps_pca.mean(dim=0)
    centered_pca = overlaps_pca - grid_mean.unsqueeze(0)
    del overlaps_pca

    U_pca, S_vals, Vt = _torch.linalg.svd(centered_pca, full_matrices=False)
    total_var = (centered_pca ** 2).sum().item()
    S_vals = S_vals[:n_modes]
    Vt = Vt[:n_modes]
    explained = (S_vals ** 2 / total_var).cpu().numpy()
    del centered_pca, U_pca, S_vals

    # Save grid_mean and mode grids to CPU before cleanup
    grid_mean_np = grid_mean.cpu().numpy()
    mode_grids_flat = Vt.cpu().numpy()

    # ── Pass 2: project ALL shots onto PCA basis ──────────────────────
    coeffs = np.zeros((n_shots, n_modes), dtype=np.float32)
    Vt_T = Vt.T  # (n_grid, n_modes) on device

    chunk_size = 2000
    for start in range(0, n_shots, chunk_size):
        end = min(start + chunk_size, n_shots)
        overlap_chunk = _compute_overlaps(w_norm[start:end])
        centered_chunk = overlap_chunk - grid_mean.unsqueeze(0)
        coeffs[start:end] = (centered_chunk @ Vt_T).cpu().numpy()
        del overlap_chunk, centered_chunk

    del kernel, Vt, Vt_T, grid_mean, w_norm

    # Reshape: flat layout is (n_vr*n_sr) blocks of n_ds → (n_ds, n_vr, n_sr)
    grid_mean_np = grid_mean_np.reshape(n_vr, n_sr, n_ds).transpose(2, 0, 1).copy()
    mode_grids_np = (
        mode_grids_flat
        .reshape(n_modes, n_vr, n_sr, n_ds)
        .transpose(0, 3, 1, 2)
        .copy()
    )

    grids = SpectralOverlapGrids(
        mean_grid=grid_mean_np,
        mode_grids=mode_grids_np,
        coeffs=coeffs,
        ds_r_range=ds_r_range,
        v_r_range=v_r_range,
        log_sr_range=(log_sr_grid[0], log_sr_grid[-1]),
        explained_variance=explained,
    )

    if cache_path is not None:
        grids.save(cache_path)
        print(f"  Saved PCA grids to {cache_path}")

    return grids


class PCASpectrum(nn.Module):
    """PCA-based spectral overlap using measured FEL spectra.

    Precomputes the radial spectral overlap on a 3D grid (Δs_r, v_r, σ_r)
    using PCA compression of measured spectra. At runtime, the overlap is
    reconstructed via per-crystal coefficients and trilinear interpolation.

    The full spectral integral factorizes as:
        I = O(Δs_r, v_r, σ_r) × exp(-|Δs_t|²/(2σ_t²))

    where O is interpolated from the precomputed grid.

    Parameters
    ----------
    grids : SpectralOverlapGrids from precompute_spectral_overlap()
    n_crystals : number of crystals
    crystal_to_shot : (n_crystals,) int — maps crystal index to shot index
        in the spectra array used for precomputation
    """

    def __init__(
        self,
        grids: SpectralOverlapGrids,
        n_crystals: int,
        crystal_to_shot: np.ndarray | Tensor,
    ):
        super().__init__()

        # Store grids as buffers — these are frozen (precomputed)
        # grid_sample expects (N, C, D, H, W) input and (N, D_out, H_out, W_out, 3) grid
        # We pack mean + modes into the channel dimension: (1, 1+n_modes, n_ds, n_vr, n_sr)
        n_modes = grids.mode_grids.shape[0]
        all_grids = np.concatenate(
            [grids.mean_grid[None], grids.mode_grids], axis=0
        )  # (1+n_modes, n_ds, n_vr, n_sr)
        self.register_buffer(
            "_grids",
            torch.from_numpy(all_grids).float().unsqueeze(0),  # (1, 1+n_modes, n_ds, n_vr, n_sr)
        )

        # Grid coordinate ranges for normalization to [-1, 1]
        self.register_buffer("_ds_r_min", torch.tensor(grids.ds_r_range[0], dtype=torch.float32))
        self.register_buffer("_ds_r_max", torch.tensor(grids.ds_r_range[1], dtype=torch.float32))
        self.register_buffer("_v_r_min", torch.tensor(grids.v_r_range[0], dtype=torch.float32))
        self.register_buffer("_v_r_max", torch.tensor(grids.v_r_range[1], dtype=torch.float32))
        self.register_buffer("_log_sr_min", torch.tensor(grids.log_sr_range[0], dtype=torch.float32))
        self.register_buffer("_log_sr_max", torch.tensor(grids.log_sr_range[1], dtype=torch.float32))

        self._n_modes = n_modes

        # Per-crystal PCA coefficients — refineable
        if isinstance(crystal_to_shot, Tensor):
            crystal_to_shot = crystal_to_shot.numpy()
        init_coeffs = grids.coeffs[crystal_to_shot]  # (n_crystals, n_modes)
        self.coeffs = nn.Parameter(torch.from_numpy(init_coeffs).float())

        # Store explained variance for diagnostics
        self.register_buffer(
            "_explained_variance",
            torch.from_numpy(grids.explained_variance).float(),
        )

    @property
    def n_modes(self) -> int:
        return self._n_modes

    def _normalize_coords(
        self, ds_r: Tensor, v_r: Tensor, log_sigma_r: Tensor,
    ) -> Tensor:
        """Map physical coordinates to [-1, 1] for grid_sample.

        Parameters
        ----------
        ds_r : (...) — radial offset in Å⁻¹
        v_r : (...) — radial sweep rate in Å⁻²
        log_sigma_r : (...) — log of radial relp width

        Returns
        -------
        coords : (..., 3) — normalized coordinates (ds_r, v_r, log_sr) in [-1, 1]
        """
        x = 2 * (ds_r - self._ds_r_min) / (self._ds_r_max - self._ds_r_min) - 1
        y = 2 * (v_r - self._v_r_min) / (self._v_r_max - self._v_r_min) - 1
        z = 2 * (log_sigma_r - self._log_sr_min) / (self._log_sr_max - self._log_sr_min) - 1
        return torch.stack([z, y, x], dim=-1)  # grid_sample order: (z, y, x) = (W, H, D)

    def overlap(
        self,
        ds_r: Tensor,
        v_r: Tensor,
        log_sigma_r: Tensor,
        crystal_id: Tensor,
    ) -> Tensor:
        """Compute radial spectral overlap via grid interpolation.

        Parameters
        ----------
        ds_r : (B, ph, pw) — radial offset from relp center in Å⁻¹
        v_r : (B, ph, pw) — radial sweep rate d*/λ in Å⁻²
        log_sigma_r : (B,) — log of per-reflection radial relp width
        crystal_id : (B,) — crystal index per observation

        Returns
        -------
        overlap : (B, ph, pw) — radial spectral overlap value
        """
        B, ph, pw = ds_r.shape

        # Expand log_sigma_r to pixel dimensions
        log_sr = log_sigma_r[:, None, None].expand(B, ph, pw)

        # Normalize coordinates to [-1, 1]
        coords = self._normalize_coords(ds_r, v_r, log_sr)  # (B, ph, pw, 3)

        # grid_sample expects: input (N, C, D, H, W), grid (N, D_out, H_out, W_out, 3)
        # Our grids are (1, 1+n_modes, n_ds, n_vr, n_sr)
        # Query grid: (B, ph, pw, 3) → reshape to (1, B*ph, pw, 1, 3) or similar
        # Actually, for 3D grid_sample we need 5D input and grid.
        # Reshape query to (1, 1, B*ph*pw, 1, 3) — treat as single batch, depth=1
        N_pixels = B * ph * pw
        grid = coords.reshape(1, 1, N_pixels, 1, 3)  # (1, 1, N, 1, 3)

        # Expand grids for the query
        sampled = F.grid_sample(
            self._grids,  # (1, 1+n_modes, n_ds, n_vr, n_sr)
            grid,
            mode="bilinear",
            align_corners=True,
            padding_mode="border",
        )  # (1, 1+n_modes, 1, N, 1)

        sampled = sampled.squeeze(0).squeeze(-1).squeeze(1)  # (1+n_modes, N)

        # Separate mean and modes
        mean_val = sampled[0]  # (N,)
        mode_vals = sampled[1:]  # (n_modes, N)

        # Per-crystal coefficients
        c = self.coeffs[crystal_id.long()]  # (B, n_modes)

        # Reshape mode values: (n_modes, B, ph*pw)
        mode_vals = mode_vals.reshape(self._n_modes, B, ph * pw)
        c_exp = c.T.unsqueeze(-1)  # (n_modes, B, 1)

        # Weighted sum: mean + Σ c_k × mode_k
        overlap = mean_val.reshape(B, ph * pw) + (c_exp * mode_vals).sum(dim=0)

        return overlap.reshape(B, ph, pw)
