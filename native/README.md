# torchsx-native

The CrystFEL `.stream` indexer, parser and writer for torchsx, in Rust
([PyO3](https://pyo3.rs) and [maturin](https://www.maturin.rs)). It is a
separate wheel so torchsx itself stays pure Python. Arrays cross the boundary
as numpy arrays, and the module does not link torch.

| Function | Does |
|----------|------|
| `index_chunks(path, n_threads=1, max_crystals=None)` | `(n, 4)` int64 rows `(start, end, n_crystals, n_obs_upper)` per chunk; `max_crystals` stops early (head mode) |
| `parse_chunks(path, chunks, panel_names, n_threads=1)` | Parses the given rows into `(columns, filenames, events, report)`; the columns use the `_allocate_tensors` names and dtypes |
| `write_stream(path, arrays, filenames, events, panel_names, lattice_type, centering, unique_axis, header, n_threads=1, images_per_batch=1024)` | Writes a stream byte-identical to the Python writer |
| `PROTOCOL` | Interface version; `torchsx/io/_native.py` refuses a build whose value differs |

Every call releases the GIL and runs on its own pool of `n_threads`. The
output does not depend on the thread count.

## Behaviour

- On well-formed streams the parse is bit-identical to the pure-Python parser
  (`torchsx/io/_parse/records.py`). Both follow the same rules, and
  `tests/test_native_parse.py` compares them.
- Malformed content never raises. It is excluded at the smallest unit that
  keeps the rest valid, and `report` counts it by reason, with the byte offset
  of the first occurrence. `records.py` lists the rules.
- Python's `str.split()`, `str.strip()`, `int()` and `float()` are reproduced,
  including PEP 515 underscores and `inf`/`nan`. Unicode decimal digits other
  than ASCII, which Python's `int()` also accepts, are not numbers here.

## Build

```bash
maturin build --release          # → target/wheels/torchsx_native-*-cp310-abi3-*.whl
pip install target/wheels/*.whl  # or, from the repo root: pip install ./native
cargo test --release             # Rust unit tests (needs libpython on LD_LIBRARY_PATH)
```

The wheel uses the stable ABI (`abi3`, Python ≥ 3.10), so one build serves every
Python version on a platform. A wheel built on a host carries that host's glibc
floor (for example `manylinux_2_34` on SLES 15 SP6). For older systems, build
with `maturin build --release --zig --compatibility manylinux_2_28`.

Bump `PROTOCOL` in `src/lib.rs` and in `torchsx/io/_native.py` together
whenever a signature or a returned layout changes.

## Release

torchsx-native is released from this repository together with torchsx
(`.github/workflows/publish.yml`, run by publishing a GitHub release `v<torchsx
version>`, or by hand against TestPyPI). The version is `Cargo.toml`'s. It changes
only when the native build changes, and torchsx pins its minor version
(`torchsx-native==0.1.*`):

- Rust change, same interface: bump the patch in `Cargo.toml`.
- Interface change: bump `PROTOCOL` (both files), the minor in `Cargo.toml`
  and the pin in the root `pyproject.toml`.

The workflow checks all of this (`.github/scripts/check_release.py`). It builds
abi3 wheels for manylinux_2_28 x86_64/aarch64, macOS universal2 and Windows x64,
smoke-tests each (`tests/smoke.py`) on its platform, and uploads torchsx-native
before torchsx.
