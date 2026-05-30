""".sfx (safetensors) I/O for SFX data — column-driven, multi-level.

Layout (format ``columnar-v2``): each level column is one safetensors entry
keyed ``"<level>/<column>"`` (``obs/hkl``, ``crystal/cell``, ``image/beam_energy``,
``obs/shoebox``, …). The JSON sidecar carries the per-level column schema, the
join registry, spacegroup/source_path, geometry, and (transitionally) the
StreamMetadata schema. Older flat-key ``.sfx`` files are still read via a legacy
fallback so existing caches keep loading.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from torchsx._core import BaseStream, JoinSpec, _SFXStreamCore
from torchsx.geometry import Geometry
from torchsx.io._metadata import StreamMetadata
from torchsx.levels import Level


# Legacy flat-field layout (for reading pre-columnar .sfx files).
_LEGACY_TENSOR_FIELDS = [
    "hkl", "I", "sigma", "background", "peak", "fs", "ss",
    "panel_id", "crystal_id",
    "astar", "bstar", "cstar", "cell", "profile_rad", "resolution",
    "n_refl", "image_id", "crystal_in_image",
    "beam_energy", "det_shift_x", "det_shift_y", "n_peaks", "camera_length",
]


def save_sfx(core: BaseStream, path: str | Path):
    """Save a stream core as .sfx (safetensors + JSON sidecar), columnar v2."""
    from safetensors.torch import save_file

    path = Path(path)

    tensors: dict[str, torch.Tensor] = {}
    level_schema: dict[str, dict] = {}
    for name, level in core.levels.items():
        lv_tensors, lv_schema = level.state_dict(prefix=name)
        tensors.update(lv_tensors)
        level_schema[name] = lv_schema

    # Transitional: per-level metadata still lives in StreamMetadata.
    meta_schema = {}
    if core.metadata:
        meta_tensors, meta_schema = core.metadata.to_tensors()
        for k, v in meta_tensors.items():
            if v.dtype == torch.int16:
                v = v.to(torch.int32)
            tensors[f"_meta/{k}"] = v.contiguous().cpu()

    save_file(tensors, str(path))

    sidecar = {
        "format": "columnar-v2",
        "levels": level_schema,
        "joins": [[j.child_level, j.key_column, j.parent_level] for j in core.joins],
        "spacegroup": core.spacegroup,
        "source_path": core.source_path,
        "metadata_schema": meta_schema,
    }
    if core.geometry:
        sidecar["geometry"] = core.geometry.to_dict()

    json_path = path.with_suffix(path.suffix + ".json")
    with open(json_path, "w") as f:
        json.dump(sidecar, f, indent=2)


def load_sfx(path: str | Path) -> BaseStream:
    """Load a stream core from .sfx (columnar v2, or legacy flat-key fallback)."""
    from safetensors.torch import load_file

    path = Path(path)
    all_tensors = load_file(str(path))

    json_path = path.with_suffix(path.suffix + ".json")
    with open(json_path) as f:
        sidecar = json.load(f)

    geometry = Geometry.from_dict(sidecar["geometry"]) if "geometry" in sidecar else None

    meta_tensors = {k[6:]: v for k, v in all_tensors.items() if k.startswith("_meta/")}
    meta_schema = sidecar.get("metadata_schema", {})
    metadata = (
        StreamMetadata.from_tensors(meta_tensors, meta_schema)
        if meta_schema else StreamMetadata()
    )

    if sidecar.get("format") == "columnar-v2":
        levels = {
            name: Level.from_state(name, all_tensors, lv_schema)
            for name, lv_schema in sidecar["levels"].items()
        }
        joins = [JoinSpec(*j) for j in sidecar["joins"]]
        return BaseStream._make(
            levels, joins, geometry, metadata,
            sidecar.get("spacegroup", "P 1"), sidecar.get("source_path", ""),
        )

    return _load_legacy(all_tensors, sidecar, geometry, metadata)


def _load_legacy(all_tensors, sidecar, geometry, metadata) -> _SFXStreamCore:
    """Read a pre-columnar flat-key .sfx file into an SFXStreamCore."""
    kwargs = {}
    for fname in _LEGACY_TENSOR_FIELDS:
        if fname not in all_tensors:
            continue
        t = all_tensors[fname]
        if fname in ("hkl", "panel_id"):
            t = t.to(torch.int16)
        elif fname in ("crystal_id", "n_refl", "image_id", "crystal_in_image", "n_peaks"):
            t = t.to(torch.int32)
        kwargs[fname] = t

    if "camera_length" not in kwargs and "beam_energy" in kwargs:
        kwargs["camera_length"] = torch.zeros_like(kwargs["beam_energy"])

    if "crystal_in_image" not in kwargs and "image_id" in kwargs:
        image_id = kwargs["image_id"]
        cii = torch.zeros_like(image_id)
        seen: dict[int, int] = {}
        for i in range(len(image_id)):
            iid = image_id[i].item()
            cii[i] = seen.get(iid, 0)
            seen[iid] = seen.get(iid, 0) + 1
        kwargs["crystal_in_image"] = cii

    kwargs["geometry"] = geometry
    kwargs["metadata"] = metadata
    kwargs["spacegroup"] = sidecar.get("spacegroup", "P 1")
    kwargs["source_path"] = sidecar.get("source_path", "")
    return _SFXStreamCore(**kwargs)
