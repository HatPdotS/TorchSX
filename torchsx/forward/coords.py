"""RefinableGeometry: wraps Geometry with refineable per-panel corrections."""

from __future__ import annotations

import torch
from torch import nn, Tensor

from torchsx.geometry import Geometry


class RefinableGeometry(nn.Module):
    """Differentiable detector geometry with per-panel refineable corrections.

    Wraps an existing Geometry dataclass. All base parameters are registered
    as buffers (fixed); small corrections are nn.Parameters.

    Parameters
    ----------
    geometry : Geometry
        CrystFEL detector geometry.
    """

    def __init__(self, geometry: Geometry):
        super().__init__()
        n = geometry.n_panels
        self._n_panels = n

        # Global geometry constants
        self.register_buffer("_res", torch.tensor(geometry.res, dtype=torch.float32))
        self.register_buffer("_clen", torch.tensor(geometry.clen, dtype=torch.float32))

        # Per-panel base values (buffers)
        corner_x = torch.zeros(n)
        corner_y = torch.zeros(n)
        fs_vecs = torch.zeros(n, 3)
        ss_vecs = torch.zeros(n, 3)
        min_fs = torch.zeros(n)
        min_ss = torch.zeros(n)
        coffset = torch.zeros(n)

        n_fs = torch.zeros(n)
        n_ss = torch.zeros(n)

        for i, p in enumerate(geometry.panels):
            corner_x[i] = p.corner_x
            corner_y[i] = p.corner_y
            fs_vecs[i] = torch.tensor(p.fs_vec)
            ss_vecs[i] = torch.tensor(p.ss_vec)
            min_fs[i] = p.min_fs
            min_ss[i] = p.min_ss
            n_fs[i] = p.n_fs
            n_ss[i] = p.n_ss
            coffset[i] = p.coffset

        self.register_buffer("base_corner_x", corner_x)
        self.register_buffer("base_corner_y", corner_y)
        self.register_buffer("fs_vec", fs_vecs)
        self.register_buffer("ss_vec", ss_vecs)
        self.register_buffer("min_fs", min_fs)
        self.register_buffer("min_ss", min_ss)
        self.register_buffer("n_fs_panel", n_fs)
        self.register_buffer("n_ss_panel", n_ss)
        self.register_buffer("coffset", coffset)

        # Refineable per-panel corrections
        self.delta_corner_x = nn.Parameter(torch.zeros(n))
        self.delta_corner_y = nn.Parameter(torch.zeros(n))
        self.delta_coffset = nn.Parameter(torch.zeros(n))
        self.log_gain = nn.Parameter(torch.zeros(n))

    @property
    def n_panels(self) -> int:
        return self._n_panels

    def gain(self, panel_id: Tensor) -> Tensor:
        """Per-panel gain factor G = exp(log_gain)."""
        return torch.exp(self.log_gain[panel_id.long()])

    def pixel_to_lab(
        self, fs: Tensor, ss: Tensor, panel_id: Tensor,
    ) -> Tensor:
        """Convert pixel coordinates to lab-frame positions (meters).

        Parameters
        ----------
        fs, ss : (N,) float — pixel coordinates (global, including panel offset)
        panel_id : (N,) int — panel index

        Returns
        -------
        xyz : (N, 3) float — lab-frame position in meters
        """
        pid = panel_id.long()
        cx = self.base_corner_x[pid] + self.delta_corner_x[pid]
        cy = self.base_corner_y[pid] + self.delta_corner_y[pid]

        dfs = fs - self.min_fs[pid]
        dss = ss - self.min_ss[pid]

        fv = self.fs_vec[pid]  # (N, 3)
        sv = self.ss_vec[pid]  # (N, 3)

        # Lab position in pixel units
        x = cx + dfs * fv[:, 0] + dss * sv[:, 0]
        y = cy + dfs * fv[:, 1] + dss * sv[:, 1]
        z = dfs * fv[:, 2] + dss * sv[:, 2]

        # Convert pixels to meters
        res = self._res.clamp(min=1.0)
        xyz = torch.stack([x / res, y / res, z], dim=-1)
        xyz[:, 2] = xyz[:, 2] + self._clen + self.coffset[pid] + self.delta_coffset[pid]
        return xyz

    def pixel_to_s_hat(
        self, fs: Tensor, ss: Tensor, panel_id: Tensor,
    ) -> Tensor:
        """Compute unit scattering vector for each pixel (wavelength-independent).

        s_hat = normalize(pixel_lab_position)

        Returns
        -------
        s_hat : (N, 3) float — unit vectors from sample to pixel
        """
        xyz = self.pixel_to_lab(fs, ss, panel_id)
        return xyz / xyz.norm(dim=-1, keepdim=True)

    def pixel_to_recip(
        self,
        fs: Tensor,
        ss: Tensor,
        panel_id: Tensor,
        wavelength: Tensor | float,
    ) -> Tensor:
        """Convert pixel coordinates to reciprocal space.

        s_p(lambda) = (1/lambda) * (s_hat_out - s_hat_in)

        Parameters
        ----------
        fs, ss : (N,) or (N, ...) pixel coordinates
        panel_id : (N,) or (N, ...) panel indices
        wavelength : scalar or broadcastable tensor — in Angstroms

        Returns
        -------
        s_p : (..., 3) float — reciprocal-space coordinates in A^-1
        """
        s_hat = self.pixel_to_s_hat(fs, ss, panel_id)
        # Incident beam along +z
        s_hat_in = torch.zeros_like(s_hat)
        s_hat_in[..., 2] = 1.0
        if isinstance(wavelength, (int, float)):
            inv_wl = 1.0 / wavelength
        else:
            inv_wl = 1.0 / wavelength
        return inv_wl * (s_hat - s_hat_in)

    def recip_to_pixel(
        self,
        h_lab: Tensor,
        wavelength: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Project reciprocal-space vector onto detector (inverse mapping).

        Given h_lab in A^-1, compute the diffracted beam direction and
        find where it intersects the detector panels.

        Parameters
        ----------
        h_lab : (N, 3) float — reciprocal lattice points in lab frame
        wavelength : float — beam wavelength in Angstroms

        Returns
        -------
        fs : (N,) float — predicted fast-scan coordinate (-1 if off detector)
        ss : (N,) float — predicted slow-scan coordinate (-1 if off detector)
        panel_id : (N,) int — panel index (-1 if off detector)
        """
        # Diffracted beam direction: s_out = lambda * h_lab + s_hat_in
        s_hat_in = torch.zeros(3, device=h_lab.device)
        s_hat_in[2] = 1.0
        s_out = wavelength * h_lab + s_hat_in  # (N, 3)
        s_out = s_out / s_out.norm(dim=-1, keepdim=True)

        N = h_lab.shape[0]
        fs_out = torch.full((N,), -1.0, device=h_lab.device)
        ss_out = torch.full((N,), -1.0, device=h_lab.device)
        pid_out = torch.full((N,), -1, dtype=torch.long, device=h_lab.device)

        # For each panel, check if the ray intersects
        for pi in range(self._n_panels):
            z_panel = self._clen + self.coffset[pi] + self.delta_coffset[pi]
            # Intersect: t = z_panel / s_out_z
            t = z_panel / s_out[:, 2].clamp(min=1e-10)  # (N,)
            # Lab position at intersection
            x_lab = t * s_out[:, 0]  # meters
            y_lab = t * s_out[:, 1]

            res = self._res.clamp(min=1.0)
            # Convert to pixel units
            cx = self.base_corner_x[pi] + self.delta_corner_x[pi]
            cy = self.base_corner_y[pi] + self.delta_corner_y[pi]
            dx = x_lab * res - cx
            dy = y_lab * res - cy

            # Solve for dfs, dss: dx = dfs * fv_x + dss * sv_x, dy = dfs * fv_y + dss * sv_y
            fv = self.fs_vec[pi]  # (3,)
            sv = self.ss_vec[pi]
            det = fv[0] * sv[1] - fv[1] * sv[0]
            if det.abs() < 1e-10:
                continue
            dfs = (dx * sv[1] - dy * sv[0]) / det
            dss = (-dx * fv[1] + dy * fv[0]) / det

            fs_global = dfs + self.min_fs[pi]
            ss_global = dss + self.min_ss[pi]

            # Check panel bounds
            in_panel = (
                (dfs >= -0.5) & (dfs < self.n_fs_panel[pi] - 0.5)
                & (dss >= -0.5) & (dss < self.n_ss_panel[pi] - 0.5)
                & (s_out[:, 2] > 0)  # forward hemisphere
            )

            # Update outputs where this panel is the best match (and not yet assigned)
            unassigned = pid_out < 0
            mask = in_panel & unassigned
            fs_out[mask] = fs_global[mask]
            ss_out[mask] = ss_global[mask]
            pid_out[mask] = pi

        return fs_out, ss_out, pid_out
