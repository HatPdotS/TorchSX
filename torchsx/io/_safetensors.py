""".sfx (safetensors) I/O for SFX data."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from torchsx._core import _SFXStreamCore
from torchsx.geometry import Geometry
from torchsx.io._metadata import StreamMetadata


_TENSOR_FIELDS = [
    "hkl", "I", "sigma", "background", "peak", "fs", "ss",
    "panel_id", "crystal_id",
    "astar", "bstar", "cstar", "cell", "profile_rad", "resolution",
    "n_refl", "image_id", "crystal_in_image",
    "beam_energy", "det_shift_x", "det_shift_y", "n_peaks", "camera_length",
]


def save_sfx(core: _SFXStreamCore, path: str | Path):
    """Save _SFXStreamCore as .sfx (safetensors + JSON sidecar)."""
    from safetensors.torch import save_file

    path = Path(path)

    # Collect tensors
    tensors = {}
    for fname in _TENSOR_FIELDS:
        t = getattr(core, fname)
        if t is None:
            continue  # optional field
        # safetensors doesn't support int16 — upcast
        if t.dtype == torch.int16:
            t = t.to(torch.int32)
        tensors[fname] = t.contiguous().cpu()

    # Save metadata tensors if any
    meta_tensors = {}
    meta_schema = {}
    if core.metadata:
        meta_tensors, meta_schema = core.metadata.to_tensors()
        for k, v in meta_tensors.items():
            if v.dtype == torch.int16:
                v = v.to(torch.int32)
            tensors[f"_meta/{k}"] = v.contiguous().cpu()

    save_file(tensors, str(path))

    # JSON sidecar
    sidecar = {
        "spacegroup": core.spacegroup,
        "source_path": core.source_path,
        "metadata_schema": meta_schema,
    }
    if core.geometry:
        sidecar["geometry"] = core.geometry.to_dict()

    json_path = path.with_suffix(path.suffix + ".json")
    with open(json_path, "w") as f:
        json.dump(sidecar, f, indent=2)


def load_sfx(path: str | Path) -> _SFXStreamCore:
    """Load _SFXStreamCore from .sfx file."""
    from safetensors.torch import load_file

    path = Path(path)
    all_tensors = load_file(str(path))

    json_path = path.with_suffix(path.suffix + ".json")
    with open(json_path) as f:
        sidecar = json.load(f)

    # Extract core tensors
    kwargs = {}
    for fname in _TENSOR_FIELDS:
        if fname not in all_tensors:
            continue  # optional field not present in older .sfx files
        t = all_tensors[fname]
        # Restore original dtypes
        if fname in ("hkl", "panel_id"):
            t = t.to(torch.int16)
        elif fname in ("crystal_id", "n_refl", "image_id", "crystal_in_image", "n_peaks"):
            t = t.to(torch.int32)
        kwargs[fname] = t

    # Synthesize camera_length if missing (older .sfx files) — default to 0.0
    if "camera_length" not in kwargs and "beam_energy" in kwargs:
        kwargs["camera_length"] = torch.zeros_like(kwargs["beam_energy"])

    # Synthesize crystal_in_image if missing (older .sfx files)
    if "crystal_in_image" not in kwargs and "image_id" in kwargs:
        image_id = kwargs["image_id"]
        cii = torch.zeros_like(image_id)
        # Count crystals per image_id in order
        seen: dict[int, int] = {}
        for i in range(len(image_id)):
            iid = image_id[i].item()
            cii[i] = seen.get(iid, 0)
            seen[iid] = seen.get(iid, 0) + 1
        kwargs["crystal_in_image"] = cii

    # Reconstruct metadata
    meta_tensors = {}
    for k, v in all_tensors.items():
        if k.startswith("_meta/"):
            meta_tensors[k[6:]] = v  # strip "_meta/" prefix

    meta_schema = sidecar.get("metadata_schema", {})
    metadata = StreamMetadata.from_tensors(meta_tensors, meta_schema) if meta_schema else StreamMetadata()

    # Reconstruct geometry
    geometry = None
    if "geometry" in sidecar:
        geometry = Geometry.from_dict(sidecar["geometry"])

    kwargs["geometry"] = geometry
    kwargs["metadata"] = metadata
    kwargs["spacegroup"] = sidecar.get("spacegroup", "P 1")
    kwargs["source_path"] = sidecar.get("source_path", "")

    return _SFXStreamCore(**kwargs)
