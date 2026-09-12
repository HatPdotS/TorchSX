# torchsx.io

I/O for serial crystallography data. Three formats, three jobs: DuckDB and
`.stream` are ingest, `.sfx` is the working format.

## Modules

| Module | Description |
|--------|-------------|
| `sources.py` | `SourceInput`, explicit dispatch and header-only measurement inspection |
| `_crystfel.py` | Interpret recorded CrystFEL commands and gain declarations |
| `_duckdb.py` | Read a probixi DuckDB run into a stream core (the key/unit mapping is documented there) |
| `_parser.py` | Parse CrystFEL `.stream` files into stacked tensors — a facade over `_parse/` |
| `_parse/` | The parser proper: `indexing` (byte scanning) and `records` (per-record parse) are leaves, `workers` builds on both, `orchestrator` drives all three |
| `_native.py` | Optional C extension for chunk-offset scanning, JIT-built via ninja |
| `_safetensors.py` | Save/load `.sfx` (safetensors + JSON sidecar), columnar v2 |
| `_metadata.py` | `StreamMetadata` — per-level metadata with tensor round-trip |
| `_writeback.py` | Write refined data back to `.stream` |
| `times.py` | Timepoint assignment and split labels |
| `_excise.py` | Extract pixel patches (ROIs) around reflections from HDF5 |
| `_roi_store.py` | `ROIStore` — memory-mapped patch storage in safetensors |

## Key functions

```python
from torchsx import SFXStream

# Ingest: from probixi, or from CrystFEL
stream = SFXStream.from_duckdb("run.duckdb", spacegroup="P 43 21 2")
stream = SFXStream.read("indexed.stream", spacegroup="P 43 21 2")
spec = stream.measurement  # torchsx.measurement.MeasurementSpec

# Work: the .sfx cache round-trip
stream.save("data.sfx")
stream = SFXStream.load("data.sfx")
```

## Notes

- Cell edges are stored in Angstrom internally; `.stream` gives nm and the
  parser converts, and probixi's `cell_a_A` is already Angstrom. Its
  `astar_x/y/z` are nm⁻¹ despite the unsuffixed name — see `_duckdb.py`.
- Join keys (`crystal_id`, `image_id`) are dense row positions, rewritten on
  every filter. A probixi `frame_id` is kept verbatim in
  `metadata["crystal"]["frame_id"]` for write-back.
- `.sfx` = safetensors binary + `.sfx.json` sidecar carrying the column schema
  and join registry, Friedel state, and the versioned measurement record under
  `metadata["_torchsx_measurement"]["record"]`. Legacy files have no spec.
- Adapters describe stored values; `prepare_observations` converts them once.
  `resolve_measurement` applies explicit > checkpoint > source precedence.
  The supported run has one ring recipe; effective masked apertures are unknown.
- `observed_hkl` and `source_lattice_index` are ordinary level columns, so the
  existing tensor transforms preserve them without special filtering code.
  Per-image `source_file`, `filename` and `event` identify source observations.
- Probixi and DIALS are reserved dispatcher kinds. Probixi's existing direct
  reader stays usable; no DIALS parser or dependency is added in this phase.
- Imports are lazy (`__getattr__`) so heavy or optional modules (duckdb, h5py,
  the native extension) are not loaded until used.
