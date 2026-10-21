"""TorchSX: amortized scaling and merging of serial crystallography data.

The public names are imported on first use, so the command-line entry points
parse their arguments (and answer ``--help``) without importing torch.
"""
import importlib

__version__ = "0.1.0"

_EXPORTS = {
    "SFXStream": "torchsx.stream",
    "Geometry": "torchsx.geometry",
    "StreamMetadata": "torchsx.io",
    "SourceInput": "torchsx.io.sources",
    "MeasurementSpec": "torchsx.measurement",
    "compute_hkl_index": "torchsx.merging.hkl",
    "merge_intensities": "torchsx.merging.hkl",
    "HKLIndex": "torchsx.merging.hkl",
}

__all__ = [*_EXPORTS, "__version__"]


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in _EXPORTS:
        value = getattr(importlib.import_module(_EXPORTS[name]), name)
        globals()[name] = value
        return value
    try:
        return importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as exc:
        if exc.name != f"{__name__}.{name}":
            raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
