"""Worker entry points: ProcessPoolExecutor trampolines + native/python dispatch.

``from torchsx.io import _native`` stays function-local in the native
trampolines so the extension loads once per worker via its cached ``.so``.
"""

from __future__ import annotations

from torchsx.io._parse.indexing import _count_byte_range
from torchsx.io._parse.records import (
    _allocate_tensors,
    _finalize,
    _line_iter,
    _parse_into_views,
)


def _parse_byte_range(
    path: str,
    start: int,
    end: int,
    panel_name_to_id: dict[str, int],
) -> dict:
    """Two-pass parse of ``[start, end)`` into tensors.

    Pass 1 counts; pass 2 writes directly into preallocated tensors through
    numpy views.
    """
    n_obs_alloc, n_crys, n_imgs = _count_byte_range(path, start, end)
    tensors = _allocate_tensors(n_obs_alloc, n_crys, n_imgs)
    views = {k: t.numpy() for k, t in tensors.items()}

    filenames: list[str] = []
    events: list[str] = []
    with open(path, "rb", buffering=1 << 20) as f:
        f.seek(start)
        it = _line_iter(f, end)
        n_cry_got, n_img_got, n_obs_got = _parse_into_views(
            it, panel_name_to_id, views, filenames, events
        )

    # Pass 1 counts structural markers; pass 2 counts what it could actually
    # parse. ``_finalize`` trims if any dimension over-allocated (e.g. a
    # malformed chunk with no ``Image filename:``).
    return _finalize(
        tensors, n_obs_got, n_obs_alloc, n_cry_got, n_crys, n_img_got, n_imgs,
        filenames, events,
    )


def _parse_byte_range_args(args: tuple) -> dict:
    """Trampoline for ProcessPoolExecutor (pickles a plain tuple)."""
    return _parse_byte_range(*args)


def _parse_into_shared(
    path: str,
    start: int,
    end: int,
    panel_name_to_id: dict[str, int],
    shared_tensors: dict[str, torch.Tensor],
    obs_offset: int,
    crys_offset: int,
    img_offset: int,
) -> tuple[int, int, int, list[str], list[str]]:
    """Worker: parse ``[start, end)`` and write directly into shared tensors.

    The shared tensors are pre-allocated at the *global* final size by the
    main process and placed in shared memory via ``torch.Tensor.share_memory_``.
    We get ``numpy`` views of them (which share storage) and pass those to
    the same ``_parse_into_views`` used by the serial path, seeded with the
    global offsets so writes land in this worker's slice.
    """
    views = {k: t.numpy() for k, t in shared_tensors.items()}
    filenames: list[str] = []
    events: list[str] = []
    with open(path, "rb", buffering=1 << 20) as f:
        f.seek(start)
        it = _line_iter(f, end)
        crys_end, img_end, obs_end = _parse_into_views(
            it, panel_name_to_id, views, filenames, events,
            crystal_idx=crys_offset, image_idx=img_offset, n_obs=obs_offset,
        )
    # Report segment-local counts so the main process can validate.
    return (
        obs_end - obs_offset,
        crys_end - crys_offset,
        img_end - img_offset,
        filenames,
        events,
    )


def _parse_into_shared_args(args: tuple) -> tuple:
    return _parse_into_shared(*args)


def _count_byte_range_args(args: tuple) -> tuple[int, int, int]:
    """Trampoline for the parallel count phase."""
    path, start, end = args
    return _count_byte_range(path, start, end)


def _native_count_args(args: tuple) -> tuple[int, int, int]:
    from torchsx.io import _native
    mod = _native.load()
    if mod is None:
        return _count_byte_range(*args)
    path, start, end = args
    return tuple(mod.count_byte_range(path, int(start), int(end)))


def _native_parse_args(args: tuple) -> tuple:
    from torchsx.io import _native
    mod = _native.load()
    path, s, e, panel_name_to_id, tensors, obs_off, crys_off, img_off = args
    if mod is None:
        return _parse_into_shared(
            path, s, e, panel_name_to_id, tensors, obs_off, crys_off, img_off
        )
    res = mod.parse_into_shared(
        path, int(s), int(e), panel_name_to_id,
        tensors["hkl"], tensors["I"], tensors["sigma"], tensors["peak"],
        tensors["background"], tensors["fs"], tensors["ss"], tensors["panel_id"],
        tensors["crystal_id"],
        tensors["cell"], tensors["astar"], tensors["bstar"], tensors["cstar"],
        tensors["profile_rad"], tensors["resolution"], tensors["n_refl"],
        tensors["image_id"], tensors["crystal_in_image"],
        tensors["beam_energy"], tensors["det_shift_x"], tensors["det_shift_y"],
        tensors["camera_length"], tensors["n_peaks"],
        int(obs_off), int(crys_off), int(img_off),
    )
    return tuple(res)
