"""The native parser and writer against their pure-Python twins.

- Parse: on well-formed and malformed streams, ``torchsx_native.parse_chunks``
  and the Python parser return bit-identical columns, the same filenames and
  events and the same exclusion report, whatever the thread count.
- Write: ``torchsx_native.write_stream`` is byte-identical to the Python writer,
  including on values that stress number formatting (exact ties, ±0, ±inf,
  NaN, subnormals, huge values).
"""

from __future__ import annotations

import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from torchsx._core import _SFXStreamCore
from torchsx.geometry import Geometry
from torchsx.io import _native
from torchsx.io._metadata import StreamMetadata
from torchsx.io._parser import (
    _find_first_chunk_offset,
    _index_chunks_until,
    _parse_byte_range,
    _parse_chunks_python,
    _parse_stream,
)
from torchsx.io._writeback import (
    _write_stream_reconstruct_native,
    _write_stream_reconstruct_python,
)
from torchsx.stream import SFXStream

BEGIN = "----- Begin chunk -----\n"


@pytest.fixture(scope="module")
def native():
    mod = _native.load()
    if mod is None:
        pytest.skip("native module not installed")
    return mod


@pytest.fixture
def base_text(tmp_path, sample_core) -> str:
    """A small well-formed stream: 2 images, 2 crystals (3 + 2 reflections)."""
    path = tmp_path / "base.stream"
    metadata = StreamMetadata(image={"filename": ["/raw/a.h5", "/raw/b.h5"], "event": ["//5", "//8"]})
    core = sample_core.replace(metadata=metadata, friedel_merged=False)
    _write_stream_reconstruct_python(core, path)
    return path.read_text()


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _parse_all(native, path: Path, n_threads=(1, 2, 4)):
    """Parse ``path`` every way; assert all agree; return one result."""
    path = str(path)
    geometry = Geometry.from_stream(path)
    rows = np.asarray(_index_chunks_until(path, None), dtype=np.int64).reshape(-1, 4)
    assert _same(native.index_chunks(path, 1), rows)

    tensors, filenames, events, report = _parse_chunks_python(path, rows, geometry.panel_name_to_id)
    py = {k: t.numpy() for k, t in tensors.items()}
    report = {k: tuple(v) for k, v in report.items()}

    if len(rows):
        full, f_fn, f_ev, f_rep = _parse_byte_range(
            path, _find_first_chunk_offset(path), Path(path).stat().st_size, geometry.panel_name_to_id
        )
        assert all(_same(full[k].numpy(), py[k]) for k in py)
        assert (f_fn, f_ev, {k: tuple(v) for k, v in f_rep.items()}) == (filenames, events, report)

    for n in n_threads:
        cols, n_fn, n_ev, n_rep = native.parse_chunks(path, rows, geometry.panel_names, n)
        assert set(cols) == set(py)
        for k in py:
            assert _same(cols[k], py[k]), (k, n)
        assert (n_fn, n_ev) == (filenames, events)
        assert dict(n_rep) == report, n
    return py, filenames, events, report


def _counts(cols) -> tuple[int, int, int]:
    return len(cols["I"]), len(cols["n_refl"]), len(cols["beam_energy"])


def _chunks(text: str) -> list[str]:
    head, *chunks = text.split(BEGIN)
    return [head] + chunks


def _join(parts: list[str]) -> str:
    return parts[0] + "".join(BEGIN + c for c in parts[1:])


def _edit_chunk(text: str, i: int, fn) -> str:
    parts = _chunks(text)
    parts[i + 1] = fn(parts[i + 1])
    return _join(parts)


def _edit_reflection(chunk: str, j: int, fn) -> str:
    lines = chunk.split("\n")
    first = next(n for n, line in enumerate(lines) if line.lstrip().startswith("h    k    l")) + 1
    lines[first + j] = fn(lines[first + j])
    return "\n".join(lines)


def _replace_line(chunk: str, prefix: str, new: str) -> str:
    lines = chunk.split("\n")
    n = next(n for n, line in enumerate(lines) if line.startswith(prefix))
    lines[n] = new
    return "\n".join(lines)


def _drop_lines(chunk: str, prefixes: tuple[str, ...]) -> str:
    return "\n".join(line for line in chunk.split("\n") if not line.startswith(prefixes))


def _field(line: str, k: int, value: str) -> str:
    parts = line.split()
    parts[k] = value
    return " ".join(parts)


MUTATIONS = {
    "reflection_short": (
        lambda t: _edit_chunk(t, 0, lambda c: _edit_reflection(c, 0, lambda l: " ".join(l.split()[:5]))),
        {"reflection_short": 1}, (4, 2, 2)),
    "reflection_value": (
        lambda t: _edit_chunk(t, 0, lambda c: _edit_reflection(c, 1, lambda l: _field(l, 3, "abc"))),
        {"reflection_value": 1}, (4, 2, 2)),
    "reflection_hkl_range": (
        lambda t: _edit_chunk(t, 1, lambda c: _edit_reflection(c, 0, lambda l: _field(l, 0, "40000"))),
        {"reflection_hkl_range": 1}, (4, 2, 2)),
    "crystal_cell": (
        lambda t: _edit_chunk(t, 0, lambda c: _replace_line(c, "Cell parameters", "Cell parameters 1.0 2.0 nm, 90 deg")),
        {"crystal_cell": 1}, (2, 1, 2)),
    "crystal_vector": (
        lambda t: _edit_chunk(t, 1, lambda c: _replace_line(c, "bstar", "bstar = 0.1 0.2 nm^-1")),
        {"crystal_vector": 1}, (3, 1, 2)),
    "crystal_cut_by_eof": (
        lambda t: t[: t.rindex("End of reflections")],
        {"crystal_truncated": 1}, (3, 1, 2)),
    "crystal_cut_by_marker": (
        lambda t: _edit_chunk(t, 0, lambda c: c[: c.index("End of reflections")]),
        {"crystal_truncated": 1}, (2, 1, 2)),
    "crystal_cut_in_header": (
        lambda t: _edit_chunk(t, 1, lambda c: c[: c.index("cstar")]),
        {"crystal_truncated": 1}, (3, 1, 2)),
    "image_truncated": (
        lambda t: t + BEGIN + "Image filename: /raw/late.h5\n",
        {"image_truncated": 1}, (5, 2, 2)),
    "chunk_no_image": (
        lambda t: _join(_chunks(t)[:2] + ["hit = 0\n----- End chunk -----\n"] + _chunks(t)[2:]),
        {"chunk_no_image": 1}, (5, 2, 2)),
    "key_value": (
        lambda t: _edit_chunk(
            _edit_chunk(t, 0, lambda c: _replace_line(
                _replace_line(c, "photon_energy_eV", "photon_energy_eV = twelve"),
                "profile_radius", "profile_radius = wide nm^-1")),
            1, lambda c: _replace_line(c, "num_peaks", "num_peaks = 1e20")),
        {"key_value": 3}, (5, 2, 2)),
    "det_shift": (
        lambda t: _edit_chunk(t, 1, lambda c: _replace_line(
            c, "predict_refine/det_shift", "predict_refine/det_shift x = abc y = 0.1 mm")),
        {"det_shift": 1}, (5, 2, 2)),
    "no_peak_section": (
        lambda t: _edit_chunk(t, 0, lambda c: _drop_lines(c, ("Peaks from peak search", "  fs/px", "End of peak list"))),
        {}, (5, 2, 2)),
    "no_end_of_peak_list": (
        lambda t: _edit_chunk(t, 1, lambda c: _drop_lines(c, ("End of peak list",))),
        {}, (5, 2, 2)),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_malformed_rules(name, native, base_text, tmp_path):
    mutate, expected_report, expected_counts = MUTATIONS[name]
    path = tmp_path / f"{name}.stream"
    path.write_text(mutate(base_text))
    cols, filenames, _, report = _parse_all(native, path)
    assert {k: v[0] for k, v in report.items()} == expected_report
    assert _counts(cols) == expected_counts
    assert len(filenames) == expected_counts[2]
    if name == "key_value":
        assert cols["beam_energy"][0] == 0.0 and cols["n_peaks"][1] == 0
        assert cols["profile_rad"][0] == np.float32(0.04 / 10.0)
    if name == "det_shift":
        assert cols["det_shift_x"][1] == 0.0


@pytest.mark.parametrize("name", ["crystal_cut_by_marker", "key_value"])
def test_one_warning_per_parse(name, base_text, tmp_path):
    path = tmp_path / "bad.stream"
    path.write_text(MUTATIONS[name][0](base_text))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SFXStream.read(path, spacegroup="P 1")
    hits = [w for w in caught if "parsed with exclusions" in str(w.message)]
    assert len(hits) == 1
    assert next(iter(MUTATIONS[name][1])).replace("_", " ") in str(hits[0].message)


def test_clean_parse_does_not_warn(base_text, tmp_path):
    path = tmp_path / "good.stream"
    path.write_text(base_text)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SFXStream.read(path, spacegroup="P 1")
    assert not [w for w in caught if "parsed with exclusions" in str(w.message)]


def test_line_ending_and_buffer_edge_cases(native, base_text, tmp_path):
    ref, *_ = _parse_all(native, _write(tmp_path / "ref.stream", base_text))
    variants = {
        "crlf": base_text.replace("\n", "\r\n"),
        "no_final_newline": base_text.rstrip("\n"),
        "long_line": _edit_chunk(base_text, 0, lambda c: c.replace("hit = 1\n", "hit = 1\njunk = " + "x" * 300_000 + "\n")),
    }
    for name, text in variants.items():
        cols, _, _, report = _parse_all(native, _write(tmp_path / f"{name}.stream", text))
        assert report == {}, name
        assert all(_same(cols[k], ref[k]) for k in ref), name


def test_invalid_utf8_filename(native, base_text, tmp_path):
    path = tmp_path / "latin1.stream"
    path.write_bytes(base_text.replace("/raw/a.h5", "/raw/caf\x00.h5").encode().replace(b"\x00", b"\xe9"))
    _, filenames, _, _ = _parse_all(native, path)
    assert filenames[0] == "/raw/caf�.h5"


def test_many_chunks_mixed_faults_thread_invariant(native, base_text, tmp_path):
    """Faults of every kind across 60 chunks: all thread counts agree with Python."""
    parts = _chunks(base_text)
    mutated = []
    for name, (mutate, _, _) in sorted(MUTATIONS.items()) * 2:
        if name in ("image_truncated", "crystal_cut_by_eof"):
            continue  # these only make sense at the end of a file
        mutated.extend(_chunks(mutate(base_text))[1:])
        mutated.extend(parts[1:])
    text = _join([parts[0]] + mutated)
    _parse_all(native, _write(tmp_path / "mixed.stream", text), n_threads=(1, 2, 3, 8))


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# Real data: all parse modes
# ---------------------------------------------------------------------------


def _core_bytes(core) -> dict[str, bytes]:
    return {f"{name}.{col}": level[col].numpy().tobytes() for name, level in core.levels.items() for col in level}


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["head", "sampled"])
def test_real_stream_modes_match_python(mode, real_stream, native, monkeypatch):
    kwargs = {"max_crystals": 3000, "sequential": mode == "head", "seed": 7}
    fast = [_parse_stream(str(real_stream), n_workers=n, **kwargs) for n in (1, 4)]
    monkeypatch.setattr(_native, "load", lambda: None)
    slow = _parse_stream(str(real_stream), **kwargs)
    for core in fast:
        assert _core_bytes(core) == _core_bytes(slow)
        assert core.metadata["image"] == slow.metadata["image"]


@pytest.mark.slow
def test_real_stream_full_thread_invariant(real_stream, native):
    a = _parse_stream(str(real_stream), n_workers=1)
    b = _parse_stream(str(real_stream), n_workers=8)
    assert _core_bytes(a) == _core_bytes(b)
    assert a.metadata["image"] == b.metadata["image"]


# ---------------------------------------------------------------------------
# Writer: number formatting
# ---------------------------------------------------------------------------


def _adversarial_floats(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Exact ties at p decimals are the odd multiples of 2**-(p+1).
    ties = [(2 * j + 1) / 2.0 ** (p + 1) * s for p in range(1, 8) for j in range(0, 400, 7) for s in (1, -1)]
    special = [0.0, -0.0, np.inf, -np.inf, np.nan, -np.nan, 1e-45, -1e-45, 1.2e-38,
               3.4e38, -3.4e38, 1e30, 0.005, -0.005, 0.045, 0.0005, 12960.0, 999.0]
    rand = rng.standard_normal(3000) * 10.0 ** rng.uniform(-4, 7, 3000)
    vals = np.concatenate([np.array(ties + special), rand]).astype(np.float32)
    return rng.permutation(np.resize(vals, n))


def test_writer_formatting_sweep(native, sample_geometry, tmp_path):
    n_img, per = 400, 12
    n_obs = n_img * per
    f = lambda seed, *shape: torch.from_numpy(_adversarial_floats(int(np.prod(shape)), seed).reshape(shape))
    rng = np.random.default_rng(3)
    hkl = rng.integers(-32768, 32768, (n_obs, 3)).astype(np.int16)
    hkl[:4] = [[-32768, 32767, 0], [0, 0, 0], [-5, 5, -1], [1, -1, 100]]
    core = _SFXStreamCore(
        hkl=torch.from_numpy(hkl),
        I=f(1, n_obs), sigma=f(2, n_obs), background=f(3, n_obs), peak=f(4, n_obs),
        fs=f(5, n_obs), ss=f(6, n_obs),
        panel_id=torch.from_numpy(rng.integers(0, 2, n_obs).astype(np.int16)),
        crystal_id=torch.arange(n_obs, dtype=torch.int32) // per,
        astar=f(7, n_img, 3), bstar=f(8, n_img, 3), cstar=f(9, n_img, 3),
        cell=f(10, n_img, 6) * 10, profile_rad=f(11, n_img), resolution=f(12, n_img),
        n_refl=torch.full((n_img,), per, dtype=torch.int32),
        image_id=torch.arange(n_img, dtype=torch.int32),
        crystal_in_image=torch.zeros(n_img, dtype=torch.int32),
        beam_energy=f(13, n_img), det_shift_x=f(14, n_img), det_shift_y=f(15, n_img),
        n_peaks=torch.from_numpy(rng.integers(-2**31, 2**31, n_img).astype(np.int32)),
        camera_length=f(16, n_img),
        geometry=sample_geometry,
        metadata=StreamMetadata(image={"filename": [f"/raw/é{i}.h5" for i in range(n_img)],
                                       "event": [f"//{i}" for i in range(n_img)]}),
        spacegroup="P 1",
        source_path="none.stream",
    )
    py, nat = tmp_path / "py.stream", tmp_path / "nat.stream"
    _write_stream_reconstruct_python(core, py)
    for n in (1, 4):
        _write_stream_reconstruct_native(core, nat, None, native, n_threads=n)
        a, b = py.read_bytes(), nat.read_bytes()
        if a != b:
            la, lb = a.split(b"\n"), b.split(b"\n")
            bad = next(i for i, (x, y) in enumerate(zip(la, lb)) if x != y)
            pytest.fail(f"line {bad}: python {la[bad]!r} != native {lb[bad]!r}")


def test_writer_rejects_inconsistent_layout(native, sample_core, tmp_path):
    bad = sample_core.replace(n_refl=torch.tensor([3, 5], dtype=torch.int32))
    with pytest.raises(ValueError, match="n_refl"):
        _write_stream_reconstruct_native(bad, tmp_path / "x.stream", None, native)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_loader_refuses_other_protocol(monkeypatch):
    fake = types.ModuleType("torchsx_native")
    fake.PROTOCOL = _native.PROTOCOL + 1
    fake.__file__ = "/stale/torchsx_native.so"
    monkeypatch.setitem(sys.modules, "torchsx_native", fake)
    monkeypatch.delenv("TORCHSX_NO_NATIVE_PARSER", raising=False)
    with pytest.raises(ImportError, match="protocol"):
        _native._import()


def test_loader_missing_module_warns(monkeypatch):
    monkeypatch.setitem(sys.modules, "torchsx_native", None)
    monkeypatch.delenv("TORCHSX_NO_NATIVE_PARSER", raising=False)
    with pytest.warns(UserWarning, match="not installed"):
        assert _native._import() is None
    monkeypatch.setenv("TORCHSX_NO_NATIVE_PARSER", "1")
    assert _native._import() is None
