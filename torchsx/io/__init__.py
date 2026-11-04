"""I/O submodule: stream parsing, safetensors, geometry, metadata, write-back."""

from torchsx.geometry import Geometry, Panel
from torchsx.io._metadata import StreamMetadata
from torchsx.io.times import (
    TimesAssignmentReport,
    assign_timepoints_to_images,
    load_split_labels,
    load_times_file,
    unique_finite_timepoints,
)


def __getattr__(name):
    if name == "parse_stream":
        from torchsx.io._parser import parse_stream
        return parse_stream
    if name in ("load_sfx", "save_sfx"):
        from torchsx.io import _safetensors
        return getattr(_safetensors, name)
    if name == "write_stream_reconstruct":
        from torchsx.io import _writeback
        return _writeback.write_stream_reconstruct
    if name == "load_duckdb":
        from torchsx.io._duckdb import load_duckdb
        return load_duckdb
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
