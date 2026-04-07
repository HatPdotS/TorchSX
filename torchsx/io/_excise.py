"""Pixel patch extraction from raw HDF5 detector images."""

from __future__ import annotations

import multiprocessing as mp
import warnings
from dataclasses import dataclass

import numpy as np
import torch

from torchsx._core import _SFXStreamCore


@dataclass
class ExcisedPatches:
    """Pixel patches extracted around predicted reflection positions.

    Attributes
    ----------
    patches : (N, patch_h, patch_w) float32 — raw pixel values
    valid : (N, patch_h, patch_w) bool — True = good pixel
    center_fs : (N,) float32 — sub-pixel fs offset within patch
    center_ss : (N,) float32 — sub-pixel ss offset within patch
    obs_index : (N,) int64 — original observation indices
    patch_size : (patch_h, patch_w)
    """

    patches: torch.Tensor
    valid: torch.Tensor
    center_fs: torch.Tensor
    center_ss: torch.Tensor
    obs_index: torch.Tensor
    patch_size: tuple[int, int]


def _parse_event_id(event_str: str) -> int:
    """Extract integer event index from CrystFEL event string.

    ``"//948"`` → 948, ``"/948"`` → 948, ``"948"`` → 948.
    """
    s = event_str.strip().lstrip("/")
    return int(s)


def _build_file_event_index(
    core: _SFXStreamCore,
    obs_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
    """Map observations → (file, event) and build a sort order for sequential I/O.

    Returns
    -------
    obs_indices : (N,) int64 — selected observation indices
    sort_order : (N,) int64 — indices into obs_indices that sort by (file, event)
    obs_file_idx : (N,) int64 — file index per observation (sorted order)
    obs_event_idx : (N,) int64 — event index per observation (sorted order)
    unique_files : list[str] — unique filenames
    """
    meta = core.metadata
    filenames = meta["image"]["filename"]
    events = meta["image"]["event"]

    # obs → image_id via crystal_id → image_id
    image_ids = core.image_id[core.crystal_id]  # (N_obs,)

    # Select subset if mask provided
    if obs_mask is not None:
        obs_indices = obs_mask.nonzero(as_tuple=True)[0].to(torch.int64)
    else:
        obs_indices = torch.arange(core.n_observations, dtype=torch.int64)

    img_ids = image_ids[obs_indices].numpy()

    # Map to filenames and event indices
    obs_filenames = [filenames[i] for i in img_ids]
    obs_events = np.array([_parse_event_id(events[i]) for i in img_ids], dtype=np.int64)

    # Build unique file list and map to file indices
    unique_files = list(dict.fromkeys(obs_filenames))
    file_to_idx = {f: i for i, f in enumerate(unique_files)}
    obs_file_idx = np.array([file_to_idx[f] for f in obs_filenames], dtype=np.int64)

    # Sort by (file_idx, event_idx) for sequential I/O
    sort_keys = obs_file_idx.astype(np.int64) * (obs_events.max() + 1 if len(obs_events) > 0 else 1) + obs_events
    sort_order = np.argsort(sort_keys, kind="stable")

    return (
        obs_indices,
        torch.from_numpy(sort_order),
        torch.from_numpy(obs_file_idx[sort_order]),
        torch.from_numpy(obs_events[sort_order]),
        unique_files,
    )


def _clip_patch(
    center: int, half: int, array_dim: int,
) -> tuple[slice, slice]:
    """Compute boundary-safe source and destination slices for patch extraction.

    Returns (src_slice, dst_slice) where src_slice indexes into the frame
    and dst_slice indexes into the patch buffer.
    """
    src_lo = center - half
    src_hi = center + half + 1
    dst_lo = 0
    dst_hi = 2 * half + 1

    if src_lo < 0:
        dst_lo = -src_lo
        src_lo = 0
    if src_hi > array_dim:
        dst_hi -= src_hi - array_dim
        src_hi = array_dim

    return slice(src_lo, src_hi), slice(dst_lo, dst_hi)


def _extract_frame_patches(
    frame: np.ndarray,
    centers_ss: np.ndarray,
    centers_fs: np.ndarray,
    hh: int,
    hw: int,
    fill: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract multiple patches from a single frame.

    Parameters
    ----------
    frame : (H, W) array
    centers_ss, centers_fs : (M,) integer pixel centers
    hh, hw : half-height, half-width
    fill : fill value for out-of-bounds pixels

    Returns
    -------
    patches : (M, 2*hh+1, 2*hw+1) float32
    valid : (M, 2*hh+1, 2*hw+1) bool
    """
    m = len(centers_ss)
    ph, pw = 2 * hh + 1, 2 * hw + 1
    h_frame, w_frame = frame.shape

    patches = np.full((m, ph, pw), fill, dtype=np.float32)
    valid = np.zeros((m, ph, pw), dtype=bool)

    # Fast path: interior patches (no clipping needed)
    interior = (
        (centers_ss >= hh)
        & (centers_ss < h_frame - hh)
        & (centers_fs >= hw)
        & (centers_fs < w_frame - hw)
    )
    int_idx = np.where(interior)[0]
    if len(int_idx) > 0:
        ss_int = centers_ss[int_idx]
        fs_int = centers_fs[int_idx]
        for j, (s, f) in enumerate(zip(ss_int, fs_int)):
            patches[int_idx[j]] = frame[s - hh : s + hh + 1, f - hw : f + hw + 1]
        valid[int_idx] = True

    # Slow path: boundary patches
    boundary_idx = np.where(~interior)[0]
    for j in boundary_idx:
        ss_src, ss_dst = _clip_patch(centers_ss[j], hh, h_frame)
        fs_src, fs_dst = _clip_patch(centers_fs[j], hw, w_frame)
        if ss_src.start < ss_src.stop and fs_src.start < fs_src.stop:
            patches[j, ss_dst, fs_dst] = frame[ss_src, fs_src]
            valid[j, ss_dst, fs_dst] = True

    return patches, valid


def _load_mask(path: str, dataset: str) -> np.ndarray | None:
    """Load a detector mask from HDF5."""
    import h5py

    with h5py.File(path, "r") as f:
        return np.asarray(f[dataset], dtype=bool)


def _worker_process_file(
    filepath: str,
    data_path: str,
    event_groups: list[tuple[int, int, int]],
    ss_int: np.ndarray,
    fs_int: np.ndarray,
    hh: int, hw: int, ph: int, pw: int,
    fill_value: float,
    det_mask: np.ndarray | None,
    shm_patches_name: str,
    shm_valid_name: str,
    global_offset: int,
    local_n: int,
    total_n: int,
) -> list[str]:
    """Worker function that writes results directly into shared memory.

    Returns only warning messages (no large arrays through pipes).
    """
    import h5py
    from multiprocessing.shared_memory import SharedMemory

    try:
        import bitshuffle.h5  # noqa: F401
    except ImportError:
        pass

    # Attach to shared memory
    shm_p = SharedMemory(name=shm_patches_name, create=False)
    shm_v = SharedMemory(name=shm_valid_name, create=False)
    patches_all = np.ndarray((total_n, ph, pw), dtype=np.float32, buffer=shm_p.buf)
    valid_all = np.ndarray((total_n, ph, pw), dtype=np.uint8, buffer=shm_v.buf)

    # Our slice of the output
    out_patches = patches_all[global_offset : global_offset + local_n]
    out_valid = valid_all[global_offset : global_offset + local_n]

    warn_msgs: list[str] = []

    try:
        fh = h5py.File(filepath, "r")
    except (OSError, FileNotFoundError):
        warn_msgs.append(f"Cannot open HDF5 file: {filepath}")
        shm_p.close()
        shm_v.close()
        return warn_msgs

    try:
        dset = fh[data_path]
        n_dset = dset.shape[0]

        valid_groups = []
        for ev_id, ev_start, ev_end in event_groups:
            if 0 <= ev_id < n_dset:
                valid_groups.append((ev_id, ev_start, ev_end))
            else:
                warn_msgs.append(
                    f"Event {ev_id} out of range for {filepath} "
                    f"(dataset has {n_dset} frames)"
                )

        chunk_start = 0
        while chunk_start < len(valid_groups):
            chunk_end = chunk_start + 1
            while chunk_end < len(valid_groups):
                prev_ev = valid_groups[chunk_end - 1][0]
                cur_ev = valid_groups[chunk_end][0]
                if cur_ev != prev_ev + 1:
                    break
                chunk_end += 1

            first_ev = valid_groups[chunk_start][0]
            last_ev = valid_groups[chunk_end - 1][0]
            try:
                slab = np.asarray(dset[first_ev : last_ev + 1])
            except Exception:
                warn_msgs.append(
                    f"Cannot read events {first_ev}-{last_ev} from {filepath}"
                )
                chunk_start = chunk_end
                continue

            for ev_id, ev_start, ev_end in valid_groups[chunk_start:chunk_end]:
                frame = slab[ev_id - first_ev]
                sl = slice(ev_start, ev_end)
                p, v = _extract_frame_patches(
                    frame, ss_int[sl], fs_int[sl], hh, hw, fill_value,
                )
                out_patches[sl] = p
                out_valid[sl] = v.astype(np.uint8)

                if det_mask is not None:
                    for j in range(ev_start, ev_end):
                        ss_src, ss_dst = _clip_patch(
                            ss_int[j], hh, det_mask.shape[0],
                        )
                        fs_src, fs_dst = _clip_patch(
                            fs_int[j], hw, det_mask.shape[1],
                        )
                        if ss_src.start < ss_src.stop and fs_src.start < fs_src.stop:
                            mask_patch = det_mask[ss_src, fs_src]
                            out_valid[j, ss_dst, fs_dst] &= ~mask_patch

            del slab
            chunk_start = chunk_end
    finally:
        fh.close()

    shm_p.close()
    shm_v.close()
    return warn_msgs


def _process_file_serial(
    filepath: str,
    data_path: str,
    event_groups: list[tuple[int, int, int]],
    ss_int: np.ndarray,
    fs_int: np.ndarray,
    hh: int, hw: int, ph: int, pw: int,
    fill_value: float,
    det_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Process one file in-process (serial path). Returns (patches, valid, warnings)."""
    import h5py

    n_local = ss_int.shape[0]
    patches = np.full((n_local, ph, pw), fill_value, dtype=np.float32)
    valid = np.zeros((n_local, ph, pw), dtype=bool)
    warn_msgs: list[str] = []

    try:
        fh = h5py.File(filepath, "r")
    except (OSError, FileNotFoundError):
        warn_msgs.append(f"Cannot open HDF5 file: {filepath}")
        return patches, valid, warn_msgs

    try:
        dset = fh[data_path]
        n_dset = dset.shape[0]

        valid_groups = []
        for ev_id, ev_start, ev_end in event_groups:
            if 0 <= ev_id < n_dset:
                valid_groups.append((ev_id, ev_start, ev_end))
            else:
                warn_msgs.append(
                    f"Event {ev_id} out of range for {filepath} "
                    f"(dataset has {n_dset} frames)"
                )

        chunk_start = 0
        while chunk_start < len(valid_groups):
            chunk_end = chunk_start + 1
            while chunk_end < len(valid_groups):
                prev_ev = valid_groups[chunk_end - 1][0]
                cur_ev = valid_groups[chunk_end][0]
                if cur_ev != prev_ev + 1:
                    break
                chunk_end += 1

            first_ev = valid_groups[chunk_start][0]
            last_ev = valid_groups[chunk_end - 1][0]
            try:
                slab = np.asarray(dset[first_ev : last_ev + 1])
            except Exception:
                warn_msgs.append(
                    f"Cannot read events {first_ev}-{last_ev} from {filepath}"
                )
                chunk_start = chunk_end
                continue

            for ev_id, ev_start, ev_end in valid_groups[chunk_start:chunk_end]:
                frame = slab[ev_id - first_ev]
                sl = slice(ev_start, ev_end)
                p, v = _extract_frame_patches(
                    frame, ss_int[sl], fs_int[sl], hh, hw, fill_value,
                )
                patches[sl] = p
                valid[sl] = v

                if det_mask is not None:
                    for j in range(ev_start, ev_end):
                        ss_src, ss_dst = _clip_patch(
                            ss_int[j], hh, det_mask.shape[0],
                        )
                        fs_src, fs_dst = _clip_patch(
                            fs_int[j], hw, det_mask.shape[1],
                        )
                        if ss_src.start < ss_src.stop and fs_src.start < fs_src.stop:
                            mask_patch = det_mask[ss_src, fs_src]
                            valid[j, ss_dst, fs_dst] &= ~mask_patch

            del slab
            chunk_start = chunk_end
    finally:
        fh.close()

    return patches, valid, warn_msgs


def excise_patches(
    core: _SFXStreamCore,
    patch_size: int | tuple[int, int] = 21,
    data_path: str | None = None,
    mask_path: str | None = None,
    mask_dataset: str | None = None,
    obs_mask: torch.Tensor | None = None,
    fill_value: float = 0.0,
    dtype: torch.dtype = torch.float32,
    n_workers: int = 1,
) -> ExcisedPatches:
    """Extract pixel patches from raw HDF5 images around predicted reflection positions.

    Observations are sorted by (file, event) so each HDF5 file is opened once
    and each frame is read once, then all patches for that frame are extracted.

    Parameters
    ----------
    core : _SFXStreamCore with metadata containing image filenames and events
    patch_size : height, width of patch (or single int for square)
    data_path : HDF5 dataset path override (default: from geometry)
    mask_path : path to HDF5 file containing bad-pixel mask
    mask_dataset : dataset name within mask file
    obs_mask : (N_obs,) bool tensor — if provided, only extract these observations
    fill_value : value for out-of-bounds pixels
    dtype : output tensor dtype
    n_workers : number of parallel workers (default 1; set to number of files
        or CPU cores for parallel I/O on compute nodes)

    Returns
    -------
    ExcisedPatches
    """
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    ph, pw = patch_size
    hh, hw = ph // 2, pw // 2

    # Resolve HDF5 data path
    if data_path is None:
        if core.geometry is not None and hasattr(core.geometry, "data_path"):
            data_path = core.geometry.data_path
        else:
            data_path = "/data/data"

    # Build sorted file/event index
    obs_indices, sort_order, sorted_file_idx, sorted_event_idx, unique_files = (
        _build_file_event_index(core, obs_mask)
    )

    n = len(obs_indices)
    sorted_obs = obs_indices[sort_order]

    # Get pixel coordinates for sorted observations
    fs_all = core.fs[sorted_obs].numpy()
    ss_all = core.ss[sorted_obs].numpy()

    # Integer centers and sub-pixel offsets
    fs_int = np.round(fs_all).astype(np.int64)
    ss_int = np.round(ss_all).astype(np.int64)
    fs_frac = fs_all - fs_int
    ss_frac = ss_all - ss_int

    # Load detector mask if provided
    det_mask = None
    if mask_path is not None and mask_dataset is not None:
        det_mask = _load_mask(mask_path, mask_dataset)

    # Convert to numpy for grouping
    file_idx_np = sorted_file_idx.numpy()
    event_idx_np = sorted_event_idx.numpy()

    # Build per-file work items
    file_work = []  # (filepath, file_start, local_n, event_groups)
    pos = 0
    for fi, filepath in enumerate(unique_files):
        file_start = pos
        while pos < n and file_idx_np[pos] == fi:
            pos += 1
        file_end = pos
        if file_start == file_end:
            continue

        event_groups = []
        ev_pos = 0
        local_n = file_end - file_start
        local_events = event_idx_np[file_start:file_end]
        while ev_pos < local_n:
            ev_id = local_events[ev_pos]
            ev_start = ev_pos
            while ev_pos < local_n and local_events[ev_pos] == ev_id:
                ev_pos += 1
            event_groups.append((int(ev_id), ev_start, ev_pos))

        file_work.append((filepath, file_start, local_n, event_groups))

    n_workers = min(n_workers, len(file_work)) if file_work else 1

    if n_workers <= 1:
        # --- Serial path ---
        try:
            import bitshuffle.h5  # noqa: F401
        except ImportError:
            pass

        patches_np = np.full((n, ph, pw), fill_value, dtype=np.float32)
        valid_np = np.zeros((n, ph, pw), dtype=bool)

        for filepath, file_start, local_n, event_groups in file_work:
            sl = slice(file_start, file_start + local_n)
            p, v, warn_msgs = _process_file_serial(
                filepath, data_path, event_groups,
                ss_int[sl].copy(), fs_int[sl].copy(),
                hh, hw, ph, pw, fill_value, det_mask,
            )
            patches_np[sl] = p
            valid_np[sl] = v
            for msg in warn_msgs:
                warnings.warn(msg, stacklevel=2)
    else:
        # --- Parallel path: shared memory + forkserver ---
        from multiprocessing.shared_memory import SharedMemory

        # Allocate shared memory for output arrays
        patches_nbytes = n * ph * pw * np.dtype(np.float32).itemsize
        valid_nbytes = n * ph * pw * np.dtype(np.uint8).itemsize

        shm_patches = SharedMemory(create=True, size=patches_nbytes)
        shm_valid = SharedMemory(create=True, size=valid_nbytes)

        patches_np = np.ndarray(
            (n, ph, pw), dtype=np.float32, buffer=shm_patches.buf,
        )
        valid_np_u8 = np.ndarray(
            (n, ph, pw), dtype=np.uint8, buffer=shm_valid.buf,
        )
        patches_np[:] = fill_value
        valid_np_u8[:] = 0

        # Build worker args
        worker_args = []
        for filepath, file_start, local_n, event_groups in file_work:
            sl = slice(file_start, file_start + local_n)
            worker_args.append((
                filepath, data_path, event_groups,
                ss_int[sl].copy(), fs_int[sl].copy(),
                hh, hw, ph, pw, fill_value, det_mask,
                shm_patches.name, shm_valid.name,
                file_start, local_n, n,
            ))

        try:
            ctx = mp.get_context("forkserver")
            with ctx.Pool(n_workers) as pool:
                warn_lists = pool.starmap(_worker_process_file, worker_args)

            for warn_msgs in warn_lists:
                for msg in warn_msgs:
                    warnings.warn(msg, stacklevel=2)

            # Copy out of shared memory before cleanup
            patches_np = patches_np.copy()
            valid_np = valid_np_u8.astype(bool)
        finally:
            shm_patches.close()
            shm_patches.unlink()
            shm_valid.close()
            shm_valid.unlink()

    # Unsort back to original observation order
    unsort = torch.empty_like(sort_order)
    unsort[sort_order] = torch.arange(n)
    unsort_np = unsort.numpy()

    patches_t = torch.from_numpy(patches_np[unsort_np]).to(dtype)
    valid_t = torch.from_numpy(valid_np[unsort_np])
    center_fs = torch.from_numpy(fs_frac[unsort_np].astype(np.float32)) + hw
    center_ss = torch.from_numpy(ss_frac[unsort_np].astype(np.float32)) + hh

    return ExcisedPatches(
        patches=patches_t,
        valid=valid_t,
        center_fs=center_fs,
        center_ss=center_ss,
        obs_index=obs_indices[unsort] if obs_mask is not None else torch.arange(n, dtype=torch.int64),
        patch_size=patch_size,
    )
