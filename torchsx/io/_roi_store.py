"""ROIStore: memory-mapped ROI patches backed by safetensors."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from torchsx._core import _SFXStreamCore


class ROIStore:
    """Memory-mapped tensor dictionary of pixel patches around reflections.

    Backed by a single safetensors file on disk. Tensors are lazily
    memory-mapped — only pages that are accessed are read from disk.

    Stored tensors
    --------------
    patches : (N, ph, pw) float32 — raw pixel values
    valid : (N, ph, pw) uint8 — 1 = good pixel, 0 = bad/out-of-bounds
    center_fs : (N,) float32 — sub-pixel fs position within patch
    center_ss : (N,) float32 — sub-pixel ss position within patch
    fs_ref : (N,) float32 — fs coordinate at excision time
    ss_ref : (N,) float32 — ss coordinate at excision time

    Usage
    -----
    >>> rois = ROIStore.excise(core, "rois.safetensors", patch_size=21)
    >>> rois["patches"].shape
    torch.Size([1601995, 21, 21])
    >>> rois = rois.update(core, threshold=0.5)  # re-excise moved reflections
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._handle = safe_open(str(self._path), framework="pt", device="cpu")

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._handle.get_tensor(key)

    def keys(self) -> list[str]:
        return list(self._handle.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._handle.keys()

    def __len__(self) -> int:
        return self["patches"].shape[0]

    @property
    def patch_size(self) -> tuple[int, int]:
        shape = self["patches"].shape
        return (shape[1], shape[2])

    @property
    def path(self) -> Path:
        return self._path

    def close(self):
        """Release the memory-mapped file handle."""
        self._handle = None

    def __repr__(self) -> str:
        n = len(self)
        ph, pw = self.patch_size
        return f"ROIStore({self._path.name}, n={n}, patch={ph}x{pw})"

    @classmethod
    def concat(cls, stores: list[ROIStore], path: str | Path) -> ROIStore:
        """Concatenate multiple ROI stores into one.

        Parameters
        ----------
        stores : list of ROIStore
        path : output safetensors file path

        Returns
        -------
        ROIStore — new store containing all observations from all inputs
        """
        if not stores:
            raise ValueError("concat requires at least one store")
        if len(stores) == 1:
            import shutil
            shutil.copy2(str(stores[0]._path), str(path))
            return cls(path)

        keys = stores[0].keys()
        tensors = {}
        for key in keys:
            parts = [s[key] for s in stores]
            tensors[key] = torch.cat(parts, dim=0)

        path = Path(path)
        save_file(tensors, str(path))
        return cls(path)

    @classmethod
    def excise(
        cls,
        core: _SFXStreamCore,
        path: str | Path,
        patch_size: int | tuple[int, int] = 21,
        data_path: str | None = None,
        mask_path: str | None = None,
        mask_dataset: str | None = None,
        obs_mask: torch.Tensor | None = None,
        fill_value: float = 0.0,
        n_workers: int = 1,
    ) -> ROIStore:
        """Excise patches from HDF5 images and save to safetensors.

        Parameters
        ----------
        core : _SFXStreamCore with metadata (filenames, events)
        path : output safetensors file path
        patch_size : patch dimensions
        data_path : HDF5 dataset path (default: from geometry)
        mask_path : HDF5 bad-pixel mask file
        mask_dataset : dataset name in mask file
        obs_mask : (N_obs,) bool — excise only these observations
        fill_value : fill for out-of-bounds pixels
        n_workers : parallel workers for HDF5 I/O

        Returns
        -------
        ROIStore — memory-mapped handle to the saved patches
        """
        from torchsx.io._excise import excise_patches

        result = excise_patches(
            core,
            patch_size=patch_size,
            data_path=data_path,
            mask_path=mask_path,
            mask_dataset=mask_dataset,
            obs_mask=obs_mask,
            fill_value=fill_value,
            n_workers=n_workers,
        )

        # Gather reference positions for change detection
        if obs_mask is not None:
            indices = obs_mask.nonzero(as_tuple=True)[0]
            fs_ref = core.fs[indices]
            ss_ref = core.ss[indices]
        else:
            fs_ref = core.fs.clone()
            ss_ref = core.ss.clone()

        tensors = {
            "patches": result.patches,
            "valid": result.valid.to(torch.uint8),
            "center_fs": result.center_fs,
            "center_ss": result.center_ss,
            "fs_ref": fs_ref,
            "ss_ref": ss_ref,
        }

        path = Path(path)
        save_file(tensors, str(path))
        return cls(path)

    def update(
        self,
        core: _SFXStreamCore,
        threshold: float = 0.5,
        data_path: str | None = None,
        mask_path: str | None = None,
        mask_dataset: str | None = None,
        obs_mask: torch.Tensor | None = None,
        fill_value: float = 0.0,
        n_workers: int = 1,
    ) -> ROIStore:
        """Re-excise only observations whose positions moved beyond threshold.

        Compares current ``core.fs``/``core.ss`` against the stored reference
        positions. Only observations with displacement > ``threshold`` pixels
        are re-extracted from HDF5. The rest are carried over from the
        existing store.

        Parameters
        ----------
        core : _SFXStreamCore with (possibly updated) fs/ss coordinates
        threshold : pixel displacement threshold for re-excision
        data_path : HDF5 dataset path (default: from geometry)
        mask_path : HDF5 bad-pixel mask file
        mask_dataset : dataset name in mask file
        obs_mask : (N_obs,) bool — if the store was built with obs_mask,
            pass the same mask here so indices align
        fill_value : fill for out-of-bounds pixels
        n_workers : parallel workers

        Returns
        -------
        ROIStore — new store (same path, overwritten)
        """
        from torchsx.io._excise import excise_patches

        # Load stored reference positions
        fs_old = self["fs_ref"]
        ss_old = self["ss_ref"]
        n = fs_old.shape[0]

        # Current positions (matching the obs subset)
        if obs_mask is not None:
            indices = obs_mask.nonzero(as_tuple=True)[0]
            fs_new = core.fs[indices]
            ss_new = core.ss[indices]
        else:
            fs_new = core.fs
            ss_new = core.ss

        assert fs_new.shape[0] == n, (
            f"Position count mismatch: core has {fs_new.shape[0]}, "
            f"store has {n}"
        )

        # Find observations that moved
        displacement = torch.sqrt((fs_new - fs_old) ** 2 + (ss_new - ss_old) ** 2)
        changed = displacement > threshold
        n_changed = changed.sum().item()

        if n_changed == 0:
            return self

        # Load ALL existing data from the current store (we need to modify it)
        patches = self["patches"].clone()
        valid = self["valid"].clone()
        center_fs = self["center_fs"].clone()
        center_ss = self["center_ss"].clone()

        # Build obs_mask for the changed subset
        # Map local changed indices back to global observation indices
        if obs_mask is not None:
            global_indices = indices[changed]
        else:
            global_indices = changed.nonzero(as_tuple=True)[0]

        global_mask = torch.zeros(core.n_observations, dtype=torch.bool)
        global_mask[global_indices] = True

        # Re-excise only the changed observations
        result = excise_patches(
            core,
            patch_size=self.patch_size,
            data_path=data_path,
            mask_path=mask_path,
            mask_dataset=mask_dataset,
            obs_mask=global_mask,
            fill_value=fill_value,
            n_workers=n_workers,
        )

        # Scatter updated values back into full arrays
        local_changed = changed.nonzero(as_tuple=True)[0]
        patches[local_changed] = result.patches
        valid[local_changed] = result.valid.to(torch.uint8)
        center_fs[local_changed] = result.center_fs
        center_ss[local_changed] = result.center_ss

        # Save with updated reference positions
        tensors = {
            "patches": patches,
            "valid": valid,
            "center_fs": center_fs,
            "center_ss": center_ss,
            "fs_ref": fs_new,
            "ss_ref": ss_new,
        }

        # Close current handle before overwriting
        self.close()
        save_file(tensors, str(self._path))
        return ROIStore(self._path)
