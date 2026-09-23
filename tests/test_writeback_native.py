"""Invariants for the native CrystFEL stream writer.

The native ``torchsx_native.write_stream`` must:

1. **Byte-exact vs. Python** — the file it produces must be byte-identical to
   what the pure-Python writer produces from the same ``_SFXStreamCore``.
   Caught by a single ``read_bytes() == read_bytes()`` comparison.
2. **Round-trip** — write → re-parse → recover the same tensors. All integer
   tensors must round-trip exactly. Float tensors are allowed up to a few
   ULP drift in fields stored at the format's documented precision (notably
   ``astar/bstar/cstar`` at ``%+.7f``); this drift is intrinsic to the
   ``.stream`` format and is also exhibited by the pure-Python writer.

Both invariants are checked on a real stream so the test exercises non-trivial
crystal/observation/image counts.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from torchsx.io import _native
from torchsx.io._writeback import (
    _write_stream_reconstruct_native,
    _write_stream_reconstruct_python,
)
from torchsx.stream import SFXStream


@pytest.fixture(scope="module")
def native():
    mod = _native.load()
    if mod is None:
        pytest.skip("native module not installed")
    return mod


@pytest.fixture(scope="module")
def small_stream(real_stream: Path) -> SFXStream:
    return SFXStream.from_stream(
        str(real_stream), spacegroup="P 1", merge_friedel=False,
        max_crystals=200, sequential=True,
    )


@pytest.mark.slow
def test_byte_exact_vs_python(small_stream: SFXStream, native, tmp_path: Path):
    """Native and Python writers must produce byte-identical files."""
    py_path = tmp_path / "py.stream"
    nat_path = tmp_path / "nat.stream"
    src = small_stream._core.source_path

    _write_stream_reconstruct_python(small_stream._core, py_path, original_stream=src)
    _write_stream_reconstruct_native(small_stream._core, nat_path, src, native)

    py_bytes = py_path.read_bytes()
    nat_bytes = nat_path.read_bytes()
    assert len(py_bytes) == len(nat_bytes), (
        f"size mismatch: py={len(py_bytes)} nat={len(nat_bytes)}"
    )
    assert py_bytes == nat_bytes, "native writer output differs from Python"


@pytest.mark.slow
@pytest.mark.parametrize("n_threads", [1, 2, 4, 8])
def test_cross_thread_invariance(small_stream: SFXStream, native, tmp_path: Path, n_threads: int):
    """Output bytes are independent of n_threads (parallel-format / serial-drain)."""
    src = small_stream._core.source_path
    ref_path = tmp_path / "ref.stream"
    out_path = tmp_path / f"nat_t{n_threads}.stream"

    _write_stream_reconstruct_native(small_stream._core, ref_path, src, native, n_threads=1)
    _write_stream_reconstruct_native(small_stream._core, out_path, src, native, n_threads=n_threads)

    ref = ref_path.read_bytes()
    got = out_path.read_bytes()
    assert ref == got, f"n_threads={n_threads} differs from serial reference"


@pytest.mark.slow
def test_round_trip(small_stream: SFXStream, native, tmp_path: Path):
    """write (native) → re-parse → tensors recover exactly (ints) or near-exactly (floats)."""
    out = tmp_path / "round.stream"
    src = small_stream._core.source_path
    _write_stream_reconstruct_native(small_stream._core, out, src, native)

    s_re = SFXStream.from_stream(str(out), spacegroup="P 1", merge_friedel=False)
    c_in, c_out = small_stream._core, s_re._core

    assert c_in.n_crystals == c_out.n_crystals
    assert c_in.n_observations == c_out.n_observations
    assert c_in.n_images == c_out.n_images

    # Integer tensors: must round-trip bit-exactly.
    for name in ("hkl", "panel_id", "crystal_id", "image_id", "n_refl", "n_peaks"):
        a = getattr(c_in, name)
        b = getattr(c_out, name)
        assert torch.equal(a, b), f"{name} differs after round-trip"

    # Float tensors stored at .2f / .3f / .5f / .6f recover bit-exactly. The
    # %.7f reciprocal vectors can drift by ≤1 ULP on a small fraction of
    # entries — this is intrinsic to the format precision and matches the
    # Python writer's behavior.
    for name in ("I", "sigma", "peak", "background", "fs", "ss",
                 "cell", "profile_rad", "resolution", "beam_energy",
                 "det_shift_x", "det_shift_y"):
        a = getattr(c_in, name)
        b = getattr(c_out, name)
        assert torch.equal(a, b), f"{name} differs after round-trip"

    # %.7f fields: tolerate ULP-scale drift on a few entries.
    for name in ("astar", "bstar", "cstar"):
        a = getattr(c_in, name).float()
        b = getattr(c_out, name).float()
        diff = (a - b).abs()
        max_err = diff.max().item()
        assert max_err < 1e-7, (
            f"{name} round-trip drift too large: max_abs_err={max_err:.3e}"
        )
