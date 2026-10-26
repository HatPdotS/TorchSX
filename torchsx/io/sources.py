"""Small source dispatcher. Adapters return the existing stream representation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from torchsx.measurement import MEASUREMENT_KEY, MeasurementSpec


@dataclass(frozen=True)
class SourceInput:
    """Named files for one datasource (e.g. DIALS experiments + reflections)."""
    kind: str
    files: Mapping[str, str | Path]

    def __post_init__(self):
        object.__setattr__(self, "kind", self.kind.lower())
        object.__setattr__(self, "files", MappingProxyType({k: Path(v) for k, v in self.files.items()}))


def source_input(source: str | Path | SourceInput) -> SourceInput:
    if isinstance(source, SourceInput):
        return source
    path = Path(source)
    if path.suffix == ".stream":
        return SourceInput("crystfel", {"stream": path})
    if path.suffix == ".sfx":
        return SourceInput("sfx", {"cache": path})
    if path.suffix == ".duckdb":
        return SourceInput("probixi", {"database": path})
    raise ValueError(f"Cannot infer datasource from {path}; use SourceInput(kind, files).")


def _path(source: SourceInput) -> Path:
    if source.kind == "dials":
        raise NotImplementedError("The dials measurement adapter is not implemented yet.")
    name = {"crystfel": "stream", "sfx": "cache", "probixi": "database"}.get(source.kind)
    if name is None:
        raise ValueError(f"Unknown datasource kind {source.kind!r}")
    if set(source.files) != {name}:
        raise ValueError(f"{source.kind} requires exactly files={{'{name}': path}}")
    return source.files[name]


def read_source(source, **kwargs):
    from torchsx.stream import SFXStream
    source = source_input(source)
    path = _path(source)
    clipped = kwargs.pop("clipped", False)  # probixi only: keep clipped disks
    if source.kind == "crystfel":
        return SFXStream.from_stream(path, **kwargs)
    # Parser-only controls have no meaning for an already parsed .sfx, but the
    # symmetry is not one of them: it decides how the cached indices are reduced.
    merge_friedel = kwargs.pop("merge_friedel", True)
    spacegroup = kwargs.pop("spacegroup", None)
    for key in ("cache", "n_workers", "max_crystals", "seed", "sequential"):
        kwargs.pop(key, None)
    if kwargs:
        raise TypeError(f"Unexpected read options: {', '.join(kwargs)}")
    if source.kind == "probixi":
        return SFXStream.from_duckdb(path, spacegroup=spacegroup, merge_friedel=merge_friedel,
                                     clipped=clipped)
    return SFXStream.load(path, merge_friedel=merge_friedel, spacegroup=spacegroup)


def inspect_source(source) -> MeasurementSpec | None:
    """Read only a stream header or .sfx JSON sidecar; never reflection tensors."""
    import json
    source = source_input(source)
    path = _path(source)
    if source.kind == "crystfel":
        from torchsx.io._crystfel import read_measurement
        return read_measurement(path)
    if source.kind == "probixi":
        from torchsx.io._duckdb import read_measurement
        return read_measurement(path)
    with open(path.with_suffix(path.suffix + ".json")) as f:
        sidecar = json.load(f)
    entry = sidecar.get("metadata_schema", {}).get(f"{MEASUREMENT_KEY}/record")
    return MeasurementSpec.from_dict(entry["value"]) if entry else None
