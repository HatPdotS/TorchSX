"""Lazy-load a JIT-compiled C++ extension for fast .stream parsing.

``load()`` compiles ``_parser_native.cpp`` on first call and caches it under
``~/.cache/torch_extensions``. Subsequent calls reuse the cached .so.

Environment probing (done once, by the first worker to reach ``_try_load``):
- If ``TORCHSX_CXX`` / ``TORCHSX_PY_INCLUDE`` / ``TORCHSX_GCC_LIB`` are set,
  they're used as-is.
- On PSI compute nodes the Pmodules layout
  (``/opt/psi/Programming/{gcc,Python}/…``) is auto-detected, since RHEL 8's
  system GCC is too old for torch headers and ``python3.11-devel`` isn't
  installed.
- Otherwise the build falls through to system defaults.

If compilation fails for any reason, ``load()`` returns ``None`` and the
parser falls back to pure-Python.
"""
from __future__ import annotations

import os
import sysconfig
import threading
import warnings
from pathlib import Path

_LOCK = threading.Lock()
_MOD = None  # resolved once; either the extension module or None
_RESOLVED = False


def load():
    """Return the compiled extension module, or None if unavailable."""
    global _MOD, _RESOLVED
    if _RESOLVED:
        return _MOD
    with _LOCK:
        if _RESOLVED:
            return _MOD
        _MOD = _try_load()
        _RESOLVED = True
        return _MOD


# ---------------------------------------------------------------------------
# Environment probing
# ---------------------------------------------------------------------------


_PSI_GCC_CANDIDATES = (
    "/opt/psi/Programming/gcc/14.3.0",
    "/opt/psi/Programming/gcc/14.2.0",
    "/opt/psi/Programming/gcc/13.3.0",
    "/opt/psi/Programming/gcc/13.2.0",
    "/opt/psi/Programming/gcc/12.3.0",
    "/opt/psi/Programming/gcc/11.4.0",
)
_PSI_PY_CANDIDATES = (
    "/opt/psi/Programming/Python/3.13.2",
    "/opt/psi/Programming/Python/3.12.9",
    "/opt/psi/Programming/Python/3.11.11",
    "/opt/psi/Programming/Python/3.10.16",
)


def _probe_env():
    """Return (cxx, cc, py_include, gcc_lib_dir).

    Any slot may be None if the caller shouldn't override.
    """
    cxx = os.environ.get("TORCHSX_CXX")
    cc = os.environ.get("TORCHSX_CC")
    py_include = os.environ.get("TORCHSX_PY_INCLUDE")
    gcc_lib = os.environ.get("TORCHSX_GCC_LIB")

    if not cxx:
        for root in _PSI_GCC_CANDIDATES:
            cand_cxx = f"{root}/bin/g++"
            if os.path.isfile(cand_cxx) and os.access(cand_cxx, os.X_OK):
                cxx = cand_cxx
                cc = cc or f"{root}/bin/gcc"
                gcc_lib = gcc_lib or f"{root}/lib64"
                break

    if not py_include:
        # Prefer the running interpreter's include dir if it has Python.h.
        cand = sysconfig.get_paths().get("include")
        if cand and os.path.exists(os.path.join(cand, "Python.h")):
            py_include = cand
        else:
            ver = f"python{sysconfig.get_python_version()}"
            for root in _PSI_PY_CANDIDATES:
                psi_cand = f"{root}/include/{ver}"
                if os.path.exists(os.path.join(psi_cand, "Python.h")):
                    py_include = psi_cand
                    break

    return cxx, cc, py_include, gcc_lib


def _ensure_ninja_on_path():
    """Make the ``ninja`` executable reachable by torch.

    The ``ninja`` pip package ships the binary inside its package directory
    and exposes it as ``ninja.BIN_DIR``, but that path isn't on PATH by
    default in spawned / sbatch environments (``shutil.which('ninja')``
    returns None). torch checks for the binary, not the Python package, so
    we have to prepend it ourselves.
    """
    try:
        import ninja as _ninja_pkg
    except Exception:
        return
    bin_dir = getattr(_ninja_pkg, "BIN_DIR", None)
    if bin_dir and os.path.isdir(bin_dir):
        cur = os.environ.get("PATH", "")
        if bin_dir not in cur.split(os.pathsep):
            os.environ["PATH"] = bin_dir + os.pathsep + cur


_EXT_NAME = "torchsx_parser_native"


def _cached_so_path() -> str | None:
    """Return the expected path of the JIT-built .so, or None.

    We replicate torch's own cache layout so workers can ``importlib`` the
    .so directly and skip ``torch.utils.cpp_extension.load()``'s ninja
    rebuild-check (which serializes 16 workers hitting a shared file lock
    and is the single biggest multi-worker overhead).
    """
    try:
        from torch.utils.cpp_extension import _get_build_directory
    except Exception:
        return None
    try:
        build_dir = _get_build_directory(_EXT_NAME, verbose=False)
    except Exception:
        return None
    so_path = os.path.join(build_dir, f"{_EXT_NAME}.so")
    return so_path if os.path.exists(so_path) else None


def _direct_import(so_path: str):
    """Import the .so directly via importlib — no torch, no ninja."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(_EXT_NAME, so_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _try_load():
    if os.environ.get("TORCHSX_NO_NATIVE_PARSER"):
        return None

    # Fast path: if the .so has already been compiled (e.g. main built it
    # before spawning workers), skip torch's rebuild machinery entirely.
    so_path = _cached_so_path()
    if so_path is not None:
        try:
            return _direct_import(so_path)
        except Exception as e:
            warnings.warn(f"direct import of cached native .so failed: {e}")

    _ensure_ninja_on_path()
    try:
        from torch.utils.cpp_extension import load as torch_load
    except Exception as e:  # pragma: no cover
        warnings.warn(f"torch.utils.cpp_extension unavailable: {e}")
        return None

    src = Path(__file__).parent / "_parser_native.cpp"
    if not src.exists():
        warnings.warn(f"native parser source not found: {src}")
        return None

    cxx, cc, py_include, gcc_lib = _probe_env()
    if cxx:
        os.environ["CXX"] = cxx
    if cc:
        os.environ["CC"] = cc

    extra_cflags = ["-O3", "-std=c++17"]
    extra_include_paths: list[str] = []
    if py_include:
        extra_include_paths.append(py_include)

    # Static-link libstdc++/libgcc. On RHEL 8 the system
    # ``/lib64/libstdc++.so.6`` is already loaded by torch and lacks newer
    # CXXABI symbols; the dynamic linker will prefer the already-loaded copy
    # over anything we set via RPATH. Static linking sidesteps the conflict
    # entirely — our .so carries its own libstdc++ symbols.
    extra_ldflags: list[str] = ["-static-libstdc++", "-static-libgcc"]

    try:
        return torch_load(
            name=_EXT_NAME,
            sources=[str(src)],
            extra_cflags=extra_cflags,
            extra_include_paths=extra_include_paths,
            extra_ldflags=extra_ldflags,
            verbose=False,
        )
    except Exception as e:
        warnings.warn(
            f"native parser compile failed; falling back to Python: {e}"
        )
        return None
