"""The probixi release and database schema this TorchSX is paired with.

probixi holds the counterpart in ``probixi._compat``: ``SCHEMA_VERSION``, the
layout of the DuckDB run it writes, and ``TORCHSX_VERSION``, the first TorchSX
release that reads that layout.  Neither package depends on the other.  Each
imports the other's constants only where the other is installed and compares
them with its own.
"""

from __future__ import annotations

from functools import lru_cache

__all__ = ["PROBIXI_VERSION", "SUPPORTED_DB_SCHEMA", "installed_probixi", "probixi_mismatch",
           "version_tuple"]

#: newest probixi database schema the DuckDB reader understands; it reads older ones too
SUPPORTED_DB_SCHEMA = 2
#: first probixi release that writes that schema
PROBIXI_VERSION = "0.11.0"


def version_tuple(version) -> tuple[int, ...]:
    """Leading numeric fields of a version string: ``"0.10.3.dev1"`` gives ``(0, 10, 3)``."""
    out = []
    for field in str(version or "").split("."):
        digits = ""
        for ch in field:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


@lru_cache(maxsize=1)
def installed_probixi() -> dict | None:
    """What the importable probixi declares, or ``None`` where probixi is absent.

    ``version`` is ``probixi.__version__`` (the installed metadata for a probixi
    from before it carried one, which can belong to another checkout),
    ``schema`` the database schema it writes and ``torchsx`` the TorchSX
    release it asks for (``None`` before probixi named one).
    """
    try:
        import probixi  # optional, never a dependency
    except Exception:  # noqa: BLE001
        return None
    schema = torchsx_version = None
    try:
        from probixi import _compat as theirs
        schema, torchsx_version = int(theirs.SCHEMA_VERSION), str(theirs.TORCHSX_VERSION)
    except Exception:  # noqa: BLE001 - a probixi from before the pairing
        try:
            from probixi.io.db import SCHEMA_VERSION as schema
        except Exception:  # noqa: BLE001
            return None
    try:
        release = str(probixi.__version__)
    except Exception:  # noqa: BLE001 - a probixi without __version__
        try:
            from importlib.metadata import version
            release = version("probixi")
        except Exception:  # noqa: BLE001
            release = None
    return {"version": release, "schema": int(schema), "torchsx": torchsx_version}


def probixi_mismatch() -> str | None:
    """Why the importable probixi and this TorchSX do not pair, or ``None``.

    ``None`` also where probixi is absent, and for an older probixi: the reader
    takes every schema up to ``SUPPORTED_DB_SCHEMA``.
    """
    from torchsx import __version__

    found = installed_probixi()
    if found is None:
        return None
    name = f"probixi {found['version']}" if found["version"] else "probixi"
    if found["schema"] > SUPPORTED_DB_SCHEMA:
        upgrade = f"Upgrade TorchSX to {found['torchsx']} or later." if found["torchsx"] \
            else "Upgrade TorchSX."
        return (f"the installed {name} writes database schema {found['schema']}; TorchSX "
                f"{__version__} reads schema {SUPPORTED_DB_SCHEMA} and older (probixi "
                f"{PROBIXI_VERSION}). {upgrade}")
    return None
