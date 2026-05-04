"""SFXStream: public API for unmerged serial crystallography data."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from torchsx._core import _SFXStreamCore
from torchsx.geometry import Geometry
from torchsx.io._metadata import StreamMetadata


class SFXStream:
    """GPU-friendly container for unmerged SFX reflection data.

    Wraps stacked tensors at three levels (observations, crystals, images)
    with immutable filtering, resolution statistics, and .stream I/O.

    Examples
    --------
    >>> stream = SFXStream.from_stream("indexed.stream", spacegroup="P 21 21 21")
    >>> stream.n_crystals
    81121
    >>> filtered = stream.filter_resolution(d_min=2.0)
    >>> filtered.save("filtered.sfx")
    """

    def __init__(self, core: _SFXStreamCore):
        self._core = core

    # --- Construction ---

    @classmethod
    def from_stream(
        cls,
        path: str | Path,
        spacegroup: str = "P 1",
        cache: str | Path | None = None,
        n_workers: int = 1,
        merge_friedel: bool = True,
        max_crystals: int | None = None,
        seed: int = 0,
        sequential: bool = False,
    ) -> SFXStream:
        """Parse a CrystFEL .stream file.

        Parameters
        ----------
        path : .stream file path
        spacegroup : space group string
        cache : if provided, save/load .sfx cache at this path
        n_workers : parallel workers (default 1 for login node safety)
        merge_friedel : if True, canonicalize HKL to merge Friedel mates
        max_crystals : if set, parse a subset of chunks whose total crystal
            count is ≥ this value.  Uses a streaming two-pass strategy that
            never holds the whole file in memory — required for 50+ GB streams
            on memory-constrained nodes.
        seed : RNG seed for random subsampling (ignored if ``sequential`` or
            ``max_crystals`` is None).
        sequential : if True (and ``max_crystals`` is set), take the first
            chunks in file order instead of a random subset.  Stops scanning
            the file as soon as enough crystals are collected — no full-file
            indexing pass.  Use for fast testing on large streams.
        """
        from torchsx.io._parser import parse_stream
        from torchsx.io._safetensors import load_sfx, save_sfx

        if cache is not None:
            cache = Path(cache)
            json_sidecar = cache.with_suffix(cache.suffix + ".json")
            if cache.exists() and json_sidecar.exists():
                core = load_sfx(cache)
                stream = cls(core)
                if merge_friedel:
                    stream = stream.symmetrize()
                return stream

        core = parse_stream(
            path, spacegroup=spacegroup, n_workers=n_workers,
            max_crystals=max_crystals, seed=seed, sequential=sequential,
        )

        if cache is not None:
            save_sfx(core, cache)

        stream = cls(core)
        if merge_friedel:
            stream = stream.symmetrize()
        return stream

    @classmethod
    def load(cls, path: str | Path, merge_friedel: bool = True) -> SFXStream:
        """Load from .sfx (safetensors) format.

        Parameters
        ----------
        path : .sfx file path
        merge_friedel : if True, canonicalize HKL to merge Friedel mates
        """
        from torchsx.io._safetensors import load_sfx

        stream = cls(load_sfx(path))
        if merge_friedel:
            stream = stream.symmetrize()
        return stream

    def save(self, path: str | Path):
        """Save to .sfx (safetensors) format."""
        from torchsx.io._safetensors import save_sfx
        save_sfx(self._core, path)

    @classmethod
    def concat(cls, streams: Sequence[SFXStream]) -> SFXStream:
        """Concatenate multiple SFXStream objects into one.

        Remaps crystal_id and image_id so the three-level join structure
        (observations → crystals → images) remains consistent.

        Parameters
        ----------
        streams : sequence of SFXStream
            Two or more streams to concatenate. All must be on the same device.

        Returns
        -------
        SFXStream
            Combined stream with renumbered join keys.
        """
        if not streams:
            raise ValueError("concat requires at least one stream")
        if len(streams) == 1:
            return streams[0]

        cores = [s._core for s in streams]

        # Compute offsets for remapping join keys
        crystal_offsets = []
        image_offsets = []
        cum_crystals = 0
        cum_images = 0
        for c in cores:
            crystal_offsets.append(cum_crystals)
            image_offsets.append(cum_images)
            cum_crystals += c.n_crystals
            cum_images += c.n_images

        # --- Observation-level tensors (concat + remap crystal_id) ---
        hkl = torch.cat([c.hkl for c in cores], dim=0)
        I = torch.cat([c.I for c in cores])
        sigma = torch.cat([c.sigma for c in cores])
        background = torch.cat([c.background for c in cores])
        peak = torch.cat([c.peak for c in cores])
        fs = torch.cat([c.fs for c in cores])
        ss = torch.cat([c.ss for c in cores])
        panel_id = torch.cat([c.panel_id for c in cores])
        crystal_id = torch.cat([
            c.crystal_id + off for c, off in zip(cores, crystal_offsets)
        ])

        # --- Crystal-level tensors (concat + remap image_id) ---
        astar = torch.cat([c.astar for c in cores], dim=0)
        bstar = torch.cat([c.bstar for c in cores], dim=0)
        cstar = torch.cat([c.cstar for c in cores], dim=0)
        cell = torch.cat([c.cell for c in cores], dim=0)
        profile_rad = torch.cat([c.profile_rad for c in cores])
        resolution = torch.cat([c.resolution for c in cores])
        n_refl = torch.cat([c.n_refl for c in cores])
        image_id = torch.cat([
            c.image_id + off for c, off in zip(cores, image_offsets)
        ])
        crystal_in_image = torch.cat([c.crystal_in_image for c in cores])

        # --- Image-level tensors (simple concat) ---
        beam_energy = torch.cat([c.beam_energy for c in cores])
        det_shift_x = torch.cat([c.det_shift_x for c in cores])
        det_shift_y = torch.cat([c.det_shift_y for c in cores])
        n_peaks = torch.cat([c.n_peaks for c in cores])

        # --- Metadata: merge by concatenating per-level lists ---
        merged_meta = StreamMetadata()
        for c in cores:
            if not c.metadata:
                continue
            for top_key, sub_dict in c.metadata.items():
                if not isinstance(sub_dict, dict):
                    continue
                if top_key not in merged_meta:
                    merged_meta[top_key] = {}
                for field, values in sub_dict.items():
                    if field not in merged_meta[top_key]:
                        merged_meta[top_key][field] = []
                    if isinstance(values, list):
                        merged_meta[top_key][field].extend(values)
                    elif isinstance(values, torch.Tensor):
                        merged_meta[top_key][field].append(values)

        # Convert any accumulated tensor lists to single tensors
        for top_key, sub_dict in merged_meta.items():
            if isinstance(sub_dict, dict):
                for field, values in sub_dict.items():
                    if isinstance(values, list) and values and isinstance(values[0], torch.Tensor):
                        sub_dict[field] = torch.cat(values)

        # Geometry + spacegroup from first stream
        geometry = cores[0].geometry
        spacegroup = cores[0].spacegroup

        new_core = _SFXStreamCore(
            hkl=hkl, I=I, sigma=sigma, background=background,
            peak=peak, fs=fs, ss=ss, panel_id=panel_id,
            crystal_id=crystal_id,
            astar=astar, bstar=bstar, cstar=cstar, cell=cell,
            profile_rad=profile_rad, resolution=resolution,
            n_refl=n_refl, image_id=image_id,
            crystal_in_image=crystal_in_image,
            beam_energy=beam_energy, det_shift_x=det_shift_x,
            det_shift_y=det_shift_y, n_peaks=n_peaks,
            geometry=geometry, metadata=merged_meta,
            spacegroup=spacegroup,
        )
        return cls(new_core)

    def split_random(self, n_parts: int = 2, seed: int | None = None) -> list[SFXStream]:
        """Randomly split crystals into *n_parts* disjoint groups.

        Parameters
        ----------
        n_parts : number of groups (default 2 for split-half)
        seed : optional RNG seed for reproducibility

        Returns
        -------
        list of SFXStream, one per group
        """
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        assignment = torch.randint(n_parts, (self.n_crystals,), generator=gen)
        parts = []
        for i in range(n_parts):
            mask = assignment == i
            parts.append(self.filter_crystals(mask))
        return parts

    # --- Properties ---

    @property
    def n_crystals(self) -> int:
        return self._core.n_crystals

    @property
    def n_observations(self) -> int:
        return self._core.n_observations

    @property
    def n_images(self) -> int:
        return self._core.n_images

    @property
    def d_min(self) -> float:
        """Minimum d-spacing across all crystals (Å)."""
        return self._core.resolution.min().item()

    @property
    def geometry(self) -> Geometry | None:
        return self._core.geometry

    @property
    def metadata(self) -> StreamMetadata:
        return self._core.metadata

    @property
    def cell(self) -> torch.Tensor:
        """(N_crystals, 6) cell parameters in Å and degrees."""
        return self._core.cell

    @property
    def hkl(self) -> torch.Tensor:
        """(N_obs, 3) Miller indices."""
        return self._core.hkl

    @property
    def I(self) -> torch.Tensor:
        """(N_obs,) intensities."""
        return self._core.I

    @property
    def sigma(self) -> torch.Tensor:
        """(N_obs,) sigma values."""
        return self._core.sigma

    @property
    def crystal_id(self) -> torch.Tensor:
        """(N_obs,) crystal index for each observation."""
        return self._core.crystal_id

    @property
    def image_id(self) -> torch.Tensor:
        """(N_crystals,) image index for each crystal."""
        return self._core.image_id

    @property
    def crystal_in_image(self) -> torch.Tensor:
        """(N_crystals,) crystal index within its image (0, 1, 2... for multi-crystal)."""
        return self._core.crystal_in_image

    @property
    def filenames(self) -> list[str]:
        """Per-image filenames."""
        if self._core.metadata and "image" in self._core.metadata:
            return self._core.metadata["image"].get("filename", [])
        return []

    @property
    def events(self) -> list[str]:
        """Per-image event strings."""
        if self._core.metadata and "image" in self._core.metadata:
            return self._core.metadata["image"].get("event", [])
        return []

    def remap_filenames(self, pattern: str, replacement: str) -> SFXStream:
        """Return a new SFXStream with image filenames transformed by string substitution.

        Each filename has ``pattern`` replaced with ``replacement`` (via
        ``str.replace``). Use this to redirect HDF5 paths to a local copy.

        Example
        -------
        >>> local = s.remap_filenames(
        ...     "/sf/alvra/data/p21957/raw/",
        ...     "/das/work/.../h5files/",
        ... )
        """
        import dataclasses
        import re
        from copy import deepcopy

        fnames = self.filenames
        if not fnames:
            return self

        new_fnames = [re.sub(pattern, replacement, f) for f in fnames]

        new_meta = deepcopy(self._core.metadata)
        new_meta["image"]["filename"] = new_fnames
        new_core = dataclasses.replace(self._core, metadata=new_meta)
        return SFXStream(new_core)

    # --- External metadata ---

    def load_image_table(self, path: str | Path, sep: str = "\t") -> SFXStream:
        """Join an external metadata table to image-level data by (filename, event).

        The file should have columns: filename, event, and any extra columns
        (e.g. time, run_id, etc.). Extra columns are stored in metadata["image"].

        Parameters
        ----------
        path : path to TSV/CSV file with header row
        sep : column separator (default tab)

        Returns
        -------
        New SFXStream with extra metadata columns attached.

        Example file format::

            filename\tevent\ttime\trun_id
            /path/to/file.h5\t//948\t0.0\t122
            /path/to/file.h5\t//946\t1.5\t122
        """
        import csv
        from copy import deepcopy

        path = Path(path)
        # Read the table
        with open(path, "r") as f:
            reader = csv.DictReader(f, delimiter=sep)
            rows = list(reader)

        if not rows:
            return self

        # Build lookup: (filename, event) → row dict
        lookup = {}
        for row in rows:
            key = (row["filename"], row["event"])
            lookup[key] = row

        # Get current image filenames/events
        fnames = self.filenames
        evts = self.events
        n_images = self.n_images

        # Determine extra columns (everything except filename and event)
        extra_cols = [k for k in rows[0].keys() if k not in ("filename", "event")]

        # Build per-image arrays for extra columns
        extra_data: dict[str, list] = {col: [] for col in extra_cols}
        n_matched = 0

        for i in range(n_images):
            fn = fnames[i] if i < len(fnames) else ""
            ev = evts[i] if i < len(evts) else ""
            row = lookup.get((fn, ev))
            if row is not None:
                n_matched += 1
                for col in extra_cols:
                    val = row[col]
                    try:
                        extra_data[col].append(float(val))
                    except ValueError:
                        extra_data[col].append(val)
            else:
                for col in extra_cols:
                    extra_data[col].append(float("nan") if _is_numeric_list(extra_data[col]) else "")

        # Build new metadata
        new_meta = deepcopy(self._core.metadata)
        if "image" not in new_meta:
            new_meta["image"] = {}
        for col, values in extra_data.items():
            new_meta["image"][col] = values

        # Build new core with updated metadata
        import dataclasses
        new_core = dataclasses.replace(self._core, metadata=new_meta)
        return SFXStream(new_core)

    # --- Inspection ---

    def completeness(self, n_bins: int = 20) -> torch.Tensor:
        """Resolution-binned completeness (fraction of unique HKLs observed).

        Returns (n_bins,) tensor.
        """
        from torchsx.merging.indexing import completeness
        return completeness(self._core, n_bins)

    def multiplicity(self, n_bins: int = 20) -> torch.Tensor:
        """Resolution-binned mean multiplicity.

        Returns (n_bins,) tensor.
        """
        from torchsx.merging.indexing import multiplicity
        return multiplicity(self._core, n_bins)

    def cell_distribution(self) -> torch.Tensor:
        """Return (N_crystals, 6) cell parameters for pathology checks."""
        return self._core.cell

    def compute_hkl_index(self):
        """Compute contiguous HKL index mapping.

        Returns
        -------
        HKLIndex with unique_hkl (N_unique, 3), hkl_id (N_obs,), n_unique
        """
        from torchsx.merging.indexing import compute_hkl_index
        return compute_hkl_index(self._core.hkl)

    def merge(self):
        """Inverse-variance weighted merge of intensities by unique HKL.

        Returns
        -------
        dict with keys:
            hkl : (N_unique, 3) int16 — unique Miller indices
            I : (N_unique,) float32 — merged intensities
            sigma : (N_unique,) float32 — merged sigma
            multiplicity : (N_unique,) int32 — observations per reflection
        """
        from torchsx.merging.indexing import compute_hkl_index, merge_intensities
        idx = compute_hkl_index(self._core.hkl)
        I_m, sig_m, mult = merge_intensities(
            idx.hkl_id, self._core.I, self._core.sigma, idx.n_unique,
        )
        return {
            "hkl": idx.unique_hkl,
            "I": I_m,
            "sigma": sig_m,
            "multiplicity": mult,
        }

    def observations_by_hkl(self) -> dict[tuple[int, int, int], torch.Tensor]:
        """Group observed intensities by Miller index.

        Returns
        -------
        dict mapping (h, k, l) tuples to (N_i,) float32 tensors of intensities
        for that reflection.
        """
        from torchsx.merging.indexing import compute_hkl_index

        idx = compute_hkl_index(self._core.hkl)
        # Sort by hkl_id for efficient splitting
        order = idx.hkl_id.argsort()
        sorted_id = idx.hkl_id[order]
        sorted_I = self._core.I[order]

        # Find split points where hkl_id changes
        counts = torch.bincount(sorted_id, minlength=idx.n_unique)
        groups = sorted_I.split(counts.tolist())

        result: dict[tuple[int, int, int], torch.Tensor] = {}
        for i, group in enumerate(groups):
            if group.numel() == 0:
                continue
            h, k, l = idx.unique_hkl[i].tolist()
            result[(h, k, l)] = group
        return result

    # --- Filtering (all return new SFXStream) ---

    def filter_resolution(self, d_min: float = 0.0, d_max: float = 999.0) -> SFXStream:
        """Keep only observations within resolution range."""
        from torchsx.merging.indexing import filter_resolution
        return SFXStream(filter_resolution(self._core, d_min, d_max))

    def filter_crystals(self, mask: torch.Tensor) -> SFXStream:
        """Keep crystals where mask is True."""
        from torchsx.merging.indexing import filter_crystals
        return SFXStream(filter_crystals(self._core, mask))

    def filter_images(self, mask: torch.Tensor) -> SFXStream:
        """Keep images where mask is True."""
        from torchsx.merging.indexing import filter_images
        return SFXStream(filter_images(self._core, mask))

    def filter_outliers(self, n_sigma: float = 4.0) -> SFXStream:
        """Remove per-hkl outlier observations."""
        from torchsx.merging.indexing import filter_outliers
        return SFXStream(filter_outliers(self._core, n_sigma))

    def filter_cell(self, tolerance: float = 0.1) -> SFXStream:
        """Remove crystals with unit cell deviating from median by > tolerance."""
        from torchsx.merging.indexing import filter_cell
        return SFXStream(filter_cell(self._core, tolerance))

    # --- Patch extraction ---

    def excise_patches(
        self,
        patch_size: int | tuple[int, int] = 21,
        data_path: str | None = None,
        mask_path: str | None = None,
        mask_dataset: str | None = None,
        obs_mask: torch.Tensor | None = None,
        fill_value: float = 0.0,
        dtype: torch.dtype = torch.float32,
        n_workers: int = 1,
    ):
        """Extract pixel patches from raw HDF5 images around each reflection.

        Sorts observations by (file, event) for efficient sequential I/O.
        Each HDF5 file is opened once, each frame read once.

        Parameters
        ----------
        patch_size : int or (h, w) — patch dimensions (default 21x21)
        data_path : HDF5 dataset path (default: from geometry)
        mask_path : HDF5 file with bad-pixel mask
        mask_dataset : dataset name in mask file
        obs_mask : (N_obs,) bool — extract only these observations
        fill_value : fill for out-of-bounds pixels
        dtype : output dtype
        n_workers : parallel workers (1 file per worker; default 1)

        Returns
        -------
        ExcisedPatches
        """
        from torchsx.io._excise import excise_patches

        return excise_patches(
            self._core,
            patch_size=patch_size,
            data_path=data_path,
            mask_path=mask_path,
            mask_dataset=mask_dataset,
            obs_mask=obs_mask,
            fill_value=fill_value,
            dtype=dtype,
            n_workers=n_workers,
        )

    # --- ROI storage ---

    def excise_rois(
        self,
        path: str | Path,
        patch_size: int | tuple[int, int] = 21,
        data_path: str | None = None,
        mask_path: str | None = None,
        mask_dataset: str | None = None,
        obs_mask: torch.Tensor | None = None,
        fill_value: float = 0.0,
        n_workers: int = 1,
    ):
        """Excise pixel patches and save to a memory-mapped safetensors file.

        Parameters
        ----------
        path : output .safetensors file path
        patch_size : int or (h, w)
        data_path : HDF5 dataset path (default: from geometry)
        mask_path : HDF5 bad-pixel mask file
        mask_dataset : dataset name in mask file
        obs_mask : (N_obs,) bool — excise only these observations
        fill_value : fill for out-of-bounds pixels
        n_workers : parallel workers for HDF5 I/O

        Returns
        -------
        ROIStore — memory-mapped handle
        """
        from torchsx.io._roi_store import ROIStore

        return ROIStore.excise(
            self._core,
            path=path,
            patch_size=patch_size,
            data_path=data_path,
            mask_path=mask_path,
            mask_dataset=mask_dataset,
            obs_mask=obs_mask,
            fill_value=fill_value,
            n_workers=n_workers,
        )

    # --- Write-back ---

    def write_stream(self, path: str | Path, n_threads: int = 1):
        """Write to CrystFEL .stream format (reconstructed from tensors).

        The geometry / unit-cell sections are copied verbatim from
        ``core.source_path`` if it still exists, otherwise synthesized from
        ``core.geometry``.

        Parameters
        ----------
        path : output .stream file path
        n_threads : ≥1. Parallel-format / sequential-drain pipeline; memory
            peak stays bounded regardless of total output size. Effective
            only when the native writer is loaded.
        """
        from torchsx.io._writeback import write_stream_reconstruct
        write_stream_reconstruct(
            self._core, path,
            original_stream=self._core.source_path or None,
            n_threads=n_threads,
        )

    # --- Symmetry ---

    def symmetrize(self, spacegroup: str | None = None) -> SFXStream:
        """Remap HKL to canonical ASU representatives using space group symmetry.

        Parameters
        ----------
        spacegroup : space group string (e.g. "P -1"). If None, uses core.spacegroup.

        Returns
        -------
        New SFXStream with HKL replaced by canonical ASU indices.
        """
        import dataclasses

        from torchref.symmetry import canonicalize_hkl

        sg = spacegroup or self._core.spacegroup
        canonical_hkl, _, _, sort_indices = canonicalize_hkl(
            self._core.hkl.int(), sg, include_friedel=True,
        )
        # canonicalize_hkl returns sorted output; invert to restore original order
        inv = torch.empty_like(sort_indices)
        inv[sort_indices] = torch.arange(len(sort_indices))
        canonical_hkl = canonical_hkl[inv]

        new_core = dataclasses.replace(
            self._core,
            hkl=canonical_hkl.to(torch.int16),
            spacegroup=sg,
        )
        return SFXStream(new_core)

    # --- Device ---

    def to(self, device) -> SFXStream:
        """Move all tensors to device."""
        return SFXStream(self._core.to(device))

    def __repr__(self) -> str:
        return (
            f"SFXStream(n_crystals={self.n_crystals}, "
            f"n_observations={self.n_observations}, "
            f"n_images={self.n_images})"
        )

    def merge(self):
        """Merge intensities using the likelihood-random scale assumed model. Returns merged intensities and sigmas by unique HKL."""
        from torchsx.merging import Merger
        M = Merger(self)
        return M.merge()

    # --- Streamlined merging chain ---

    def make_loss(
        self,
        *,
        n_points: int = 513,
        S_min: float = 1e-3,
        S_max: float = 1e6,
        gain: float | str = "auto",
        min_obs: int = 5,
        scale: str = "b_per_crystal",
        smooth: float = 500.0,
        wilson_aniso: float = 0.0,
        wilson_aniso_U_ridge: float = 1.0,
        wilson_aniso_b_center: float = 1.0,
        obs_chunk_size: int = 0,
        device: str | torch.device = "cpu",
    ):
        """Build a :class:`LogSpaceLoss` from this stream.

        The returned object is the entry point of the streamlined chain:
        ``stream.make_loss(...).refine(...).posterior(...).integrate()``.
        See :class:`torchsx.merging.loss.LogSpaceLoss` for details.

        Parameters
        ----------
        n_points :
            Log-S quadrature grid size (default 513).
        S_min, S_max :
            Log-S grid bounds in absolute coordinates.
        gain :
            ADU/photon. ``"auto"`` (default) runs the bg-normalised
            truncated-Gaussian fit on negatives. Pass a float to pin.
        min_obs :
            Drop HKLs with fewer than this many positive observations.
        scale :
            Scale model name (``"b_per_crystal"`` or ``"panel"``).
        obs_chunk_size :
            Memory budget for the per-obs data target. ``0`` = no chunking.
        device :
            ``"cpu"`` or ``"cuda"``.
        """
        from torchsx.merging.loss import LogSpaceLoss
        return LogSpaceLoss.from_stream(
            self,
            n_points=n_points, S_min=S_min, S_max=S_max,
            gain=gain, min_obs=min_obs, scale=scale,
            smooth=smooth,
            wilson_aniso=wilson_aniso,
            wilson_aniso_U_ridge=wilson_aniso_U_ridge,
            wilson_aniso_b_center=wilson_aniso_b_center,
            obs_chunk_size=obs_chunk_size, device=device,
        )

def _is_numeric_list(lst: list) -> bool:
    """Check if all existing elements in a list are numeric."""
    return all(isinstance(x, (int, float)) for x in lst)
