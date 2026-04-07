# torchsx.io

I/O for serial crystallography data.

## Modules

| Module | Description |
|--------|-------------|
| `_parser.py` | Parse CrystFEL `.stream` files into stacked tensors |
| `_safetensors.py` | Save/load `.sfx` format (safetensors + JSON sidecar) |
| `_writeback.py` | Write refined data back to `.stream` format |
| `_geometry.py` | Geometry I/O helpers |
| `_metadata.py` | `StreamMetadata` — per-image/per-crystal metadata with tensor round-trip |
| `_frame_store.py` | `FrameStore` — memory-mapped frame I/O from HDF5 |
| `_roi_store.py` | `ROIStore` — ROI data storage in safetensors |
| `_excise.py` | Extract pixel patches (ROIs) around reflections from HDF5 |
| `_frame_extract.py` | Frame extraction utilities |

## Key Functions

```python
from torchsx.io import parse_stream, load_sfx, save_sfx

# Parse a CrystFEL .stream file
core = parse_stream("data.stream", spacegroup="P 1")

# Save/load fast binary format
save_sfx(core, "data.sfx")
core = load_sfx("data.sfx")
```

## Notes

- Cell parameters stored in Angstrom internally, converted nm/Angstrom on I/O
- `.sfx` = safetensors binary + `.sfx.json` sidecar for metadata
- Imports are lazy (`__getattr__`) to avoid loading heavy modules until needed
