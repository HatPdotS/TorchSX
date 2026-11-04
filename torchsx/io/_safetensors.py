""".sfx (safetensors) I/O for SFX data — column-driven, multi-level.

Layout (format ``columnar-v2``): each level column is one safetensors entry
keyed ``"<level>/<column>"`` (``obs/hkl``, ``crystal/cell``, ``image/beam_energy``,
``obs/shoebox``, …). The JSON sidecar carries the per-level column schema, the
join registry, spacegroup/source_path, geometry, and (transitionally) the
StreamMetadata schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from torchsx._core import BaseStream, JoinSpec, absolutize_image_filenames
from torchsx.geometry import Geometry
from torchsx.io._metadata import StreamMetadata
from torchsx.levels import Level


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

    # Detector good-pixel mask (True=good), embedded so the .sfx is self-contained.
    detector_mask = getattr(core, "detector_mask", None)
    if detector_mask is not None:
        tensors["_detector_mask"] = detector_mask.to(torch.bool).contiguous().cpu()

    # Beam-fixed background pattern f(p) (σ units, (P,514,fs)) used by the Bayes
    # hit-finder; embedded alongside the mask for reproducibility.
    fixed_pattern = getattr(core, "fixed_pattern", None)
    if fixed_pattern is not None:
        tensors["_fixed_pattern"] = fixed_pattern.to(torch.float32).contiguous().cpu()

    # Retained source identities may share storage with their working columns.
    # safetensors requires distinct storage; copy aliases only at serialization.
    seen = set()
    for key, tensor in tensors.items():
        storage = tensor.untyped_storage().data_ptr()
        if storage in seen:
            tensors[key] = tensor.clone()
        seen.add(storage)
    save_file(tensors, str(path))

    sidecar = {
        "format": "columnar-v2",
        "levels": level_schema,
        "joins": [[j.child_level, j.key_column, j.parent_level] for j in core.joins],
        "spacegroup": core.spacegroup,
        "source_path": core.source_path,
        "friedel_merged": core.friedel_merged,
        "metadata_schema": meta_schema,
        "has_detector_mask": detector_mask is not None,
        "has_fixed_pattern": fixed_pattern is not None,
    }
    if core.geometry:
        sidecar["geometry"] = core.geometry.to_dict()

    json_path = path.with_suffix(path.suffix + ".json")
    with open(json_path, "w") as f:
        json.dump(sidecar, f, indent=2)


def load_sfx(path: str | Path) -> BaseStream:
    """Load a stream core from .sfx (columnar v2)."""
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
    # Normalise per-image source paths to absolute (covers streams written with
    # relative paths); source_path itself is normalised in BaseStream.__init__.
    absolutize_image_filenames(metadata)

    if sidecar.get("format") != "columnar-v2":
        raise ValueError(f"{path} is not a columnar-v2 .sfx file; convert the source again")
    levels = {
        name: Level.from_state(name, all_tensors, lv_schema)
        for name, lv_schema in sidecar["levels"].items()
    }
    joins = [JoinSpec(*j) for j in sidecar["joins"]]
    obj = BaseStream._make(
        levels, joins, geometry, metadata,
        sidecar.get("spacegroup", "P 1"), sidecar.get("source_path", ""),
        friedel_merged=sidecar.get("friedel_merged", True),
    )
    if "_detector_mask" in all_tensors:
        obj.detector_mask = all_tensors["_detector_mask"].to(torch.bool)
    if "_fixed_pattern" in all_tensors:
        obj.fixed_pattern = all_tensors["_fixed_pattern"].to(torch.float32)
    return obj
