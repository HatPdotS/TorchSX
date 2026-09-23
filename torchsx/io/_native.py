"""The native ``.stream`` reader/writer: the ``torchsx_native`` Rust module.

It is a separate wheel built from ``native/`` in this repository (see
``native/README.md``). ``load()`` returns the module, or ``None`` when it is not
installed or ``TORCHSX_NO_NATIVE_PARSER`` is set; callers then use the
pure-Python parser and writer. A module built from a different source revision
than this checkout expects raises instead, so a stale build is never used.
"""
from __future__ import annotations

import os
import warnings

PROTOCOL = 1

_MOD = None
_RESOLVED = False


def load():
    """Return ``torchsx_native``, or ``None`` if unavailable or disabled."""
    global _MOD, _RESOLVED
    if not _RESOLVED:
        _MOD = _import()
        _RESOLVED = True
    return _MOD


def _import():
    if os.environ.get("TORCHSX_NO_NATIVE_PARSER"):
        return None
    try:
        import torchsx_native
    except ImportError:
        warnings.warn(
            "torchsx_native is not installed; .stream files are read and written "
            "with the slower pure-Python code. Build it with "
            "`maturin build --release` in native/ and pip-install the wheel.",
            stacklevel=3,
        )
        return None
    if getattr(torchsx_native, "PROTOCOL", None) != PROTOCOL:
        raise ImportError(
            f"torchsx_native at {torchsx_native.__file__} speaks protocol "
            f"{getattr(torchsx_native, 'PROTOCOL', None)}, this torchsx expects {PROTOCOL}; "
            "rebuild and reinstall it from native/."
        )
    return torchsx_native
