"""I/O submodule: stream parsing, safetensors, geometry, metadata, write-back."""

from torchsx.geometry import Geometry, Panel
from torchsx.io._metadata import StreamMetadata


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
    if name == "FrameStore":
        from torchsx.io._frame_store import FrameStore
        return FrameStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
