"""The native ``.stream`` reader/writer: the ``torchsx_native`` Rust module.

It is a separate wheel built from ``native/`` in this repository (see
``native/README.md``) and a torchsx dependency. ``load()`` returns the module;
a missing module, or one built from a different source revision than this
checkout expects, raises, so a stale build is never used.
"""
from __future__ import annotations

PROTOCOL = 1

_MOD = None


def load():
    """Return ``torchsx_native``."""
    global _MOD
    if _MOD is None:
        _MOD = _import()
    return _MOD


def _import():
    try:
        import torchsx_native
    except ImportError as exc:
        raise ImportError(
            "torchsx_native is not installed; torchsx reads and writes .stream "
            "files with it. Install the torchsx-native package (a torchsx "
            "dependency) or build it from native/."
        ) from exc
    if getattr(torchsx_native, "PROTOCOL", None) != PROTOCOL:
        raise ImportError(
            f"torchsx_native at {torchsx_native.__file__} speaks protocol "
            f"{getattr(torchsx_native, 'PROTOCOL', None)}, this torchsx expects {PROTOCOL}; "
            "rebuild and reinstall it from native/."
        )
    return torchsx_native
