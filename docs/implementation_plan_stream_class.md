# TorchRef SFX Stream Module — Implementation Plan

## Overview

The SFX stream module provides a high-performance, differentiable data structure for
unmerged serial crystallography data. The design priority is clean public API and
efficient internal representation — all index arithmetic is hidden inside the class.

---

## Module Structure

```
torchref/sfx/
├── _core.py          # _SFXStreamCore dataclass — raw stacked tensors
├── _io.py            # Parser, memmap writer, safetensors serialization
├── _symmetry.py      # ASU reduction, hkl_id linearization, centric flags
├── _scale.py         # Scale + B-factor refinement
├── _merge.py         # Weighted merge, CC½, R-merge, Wilson stats
├── _filter.py        # Resolution, outlier, crystal filtering
└── stream.py         # SFXStream — public API
```

---

## Data Model

### Two-level structure: Observations → Crystals → Images

All parameters are **stacked tensors** — no Python objects per crystal.
`crystal_id` is the only index needed to join levels.
Observation tensors are **sorted by image_id** at load time, permanently.

```python
@dataclass
class _SFXStreamCore:

    # === OBSERVATIONS (N_obs,) — sorted by image_id ===
    hkl:          Tensor  # (N_obs, 3)   int16   — ASU-reduced
    I:            Tensor  # (N_obs,)     float32
    sigma:        Tensor  # (N_obs,)     float32
    background:   Tensor  # (N_obs,)     float32
    peak:         Tensor  # (N_obs,)     float32 — peak value from indexer
    fs:           Tensor  # (N_obs,)     float32 — detector fast coord
    ss:           Tensor  # (N_obs,)     float32 — detector slow coord
    panel_id:     Tensor  # (N_obs,)     int16   — encoded panel name
    crystal_id:   Tensor  # (N_obs,)     int32   — join key to crystals
    hkl_id:       Tensor  # (N_obs,)     int64   — linearized ASU index, cached

    # === CRYSTALS (N_crystals,) ===
    astar:        Tensor  # (N_crystals, 3)
    bstar:        Tensor  # (N_crystals, 3)
    cstar:        Tensor  # (N_crystals, 3)
    cell:         Tensor  # (N_crystals, 6)  a,b,c,α,β,γ
    scale:        Tensor  # (N_crystals,)    → nn.Parameter during refinement
    B_iso:        Tensor  # (N_crystals,)    → nn.Parameter during refinement
    profile_rad:  Tensor  # (N_crystals,)
    resolution:   Tensor  # (N_crystals,)    reported d_min per crystal
    n_refl:       Tensor  # (N_crystals,)    int32
    image_id:     Tensor  # (N_crystals,)    int32  → join key to images

    # === IMAGES (N_images,) ===
    beam_energy:  Tensor  # (N_images,)
    det_shift_x:  Tensor  # (N_images,)
    det_shift_y:  Tensor  # (N_images,)
    n_peaks:      Tensor  # (N_images,)   from peak search

    # === METADATA (not tensors) ===
    spacegroup:   gemmi.SpaceGroup
    panel_names:  list[str]          # panel_id → name lookup
    filenames:    list[str]          # image_id → HDF5 path lookup
    events:       list[str]          # image_id → event string (EuXFEL etc.)
    d_min:        float
    source_path:  str                # original .stream path
    crystfel_version: str
```

### Key invariants enforced at construction:
- `crystal_id` is always a contiguous integer range `[0, N_crystals)`
- `hkl_id` is always consistent with `hkl` — recomputed atomically if hkl changes
- Observation tensors are always sorted by `image_id` via `crystal_id`
- Filtering always returns a new `SFXStream` — no mutation of observation tensors

---

## I/O Architecture

### Storage Format

Two-tier: slow canonical format (.stream) and fast native format (.sfx).

```
dataset.sfx          ← safetensors: all tensors, memory-mappable, zero-copy
dataset.sfx.json     ← spacegroup, filenames, events, panel_names, provenance
```

First load parses the stream and writes the native cache automatically.
All subsequent loads use the fast path — target: **< 1 second**.

### Parallel Parser — 3 Steps

**Step 1 — Crystal offset scan** (single thread, fast byte scan)
```
mmap entire file → scan for b"--- Begin crystal" → np.array of byte offsets
```
Typical speed: ~2 seconds for 10 GB.

**Step 2 — Sample-based allocation estimate**
```
Parse first 100 crystals → measure avg_refs_per_crystal, std
N_obs_alloc  = N_crystals * avg_refs * 1.1
N_crys_alloc = N_crystals * 1.1        # always exact from scan, but pad anyway
Warn if avg_refs < 5 or CV > 0.5
```

**Step 3 — Parallel parse + direct memmap write**
```
Divide crystal_offsets evenly across N workers
Pre-allocate memmaps on disk (N_obs_alloc, N_crys_alloc)
Each worker:
  - receives: path, my_crystal_offsets[], obs_start, cry_start
  - opens memmaps in "r+" mode
  - parses each crystal → writes directly into pre-assigned slice
  - returns: (obs_end, cry_end)   ← only IPC payload
Trim memmaps to actual size using worker reports
Sort all observation tensors by image_id
Build hkl_id via ASU reduction + linearization
Write final .sfx + .sfx.json
```

No temporary arrays, no inter-worker communication, no pickle overhead.

### Write-back to .stream

Two modes:

**patch** (default): mmap original stream, seek to each crystal's reflection block,
overwrite values in-place. Preserves all header text and annotations exactly.

**reconstruct**: generate valid CrystFEL stream from scratch from tensors.
Loses free-text comments, preserves all data.

---

## Public API

### Construction
```python
# First load — parses stream, writes .sfx cache, returns SFXStream
stream = SFXStream.from_stream(
    "indexed.stream",
    spacegroup = "P 21 21 21",
    cache      = "dataset.sfx",   # written automatically; loaded on next call
    n_workers  = 32,
)

# Fast load from native format
stream = SFXStream.load("dataset.sfx")

# Save native
stream.save("dataset.sfx")
```

### Inspection
```python
stream.n_crystals       # int
stream.n_observations   # int
stream.n_images         # int
stream.d_min            # float
stream.completeness(n_bins=20)   # (n_bins,) Tensor
stream.multiplicity(n_bins=20)   # (n_bins,) Tensor
stream.cell_distribution()       # (N_crystals, 6) — for pathology checks
```

### Filtering — always returns new SFXStream, never mutates
```python
stream.filter_resolution(d_min=1.5, d_max=50.0)
stream.filter_crystals(mask)          # boolean (N_crystals,) Tensor
stream.filter_outliers(n_sigma=4.0)   # per-reflection, per hkl_id group
stream.filter_images(mask)            # boolean (N_images,) Tensor
```
All filters re-index `crystal_id` and `hkl_id` atomically.

### Write-back
```python
stream.write_stream("refined.stream", mode="patch")
stream.write_stream("refined.stream", mode="reconstruct")
stream.merge().save_mtz("merged.mtz")          # via gemmi
stream.merge().to_reflection_data()            # → TorchRef ReflectionData
```

---

## Core Internal Operations


### Image range iterator — free because observations are sorted
```python
def _image_ranges(self):
    # torch.unique_consecutive on image_id[crystal_id] 
    # yields (img_id, obs_start, obs_end) — no sorting needed
```

---

## Build Phases

### Phase 1 — Core + I/O (foundation)
1. `_SFXStreamCore` dataclass with validation and invariant checks
2. `safetensors` save/load — fast path working from day one
3. `_find_crystal_offsets` byte scanner
4. `_estimate_obs_per_crystal` from first 100 crystals
5. Single-process `_parse_crystal` — correctness before parallelism
6. Parallel parse + memmap write with worker reporting
7. Image-order sort on load
8. `SFXStream.from_stream()`, `load()`, `save()`

### Phase 2 — Symmetry + Indexing
9. ASU reduction via gemmi → `hkl` reduction
10. `hkl_id` linearization (computed once, cached)
11. `centric`, `epsilon` (multiplicity) per reflection
12. `filter_resolution()`, `filter_crystals()` with re-indexing

### Phase 3 — Merging + Metrics
13. `I_corrected()` (identity scale/B initially)
14. `scatter_add` weighted merge → `MergedData`
15. `cc_half()`, `r_merge()`, `r_meas()`
16. `wilson_plot()`
17. `filter_outliers()`

### Phase 4 — Scale Refinement
18. Promote `scale`, `B_iso` to `nn.Parameter`
19. `refine_scales()` with LBFGS
20. Per-shell diagnostics and convergence reporting

### Phase 5 — Integration + Write-back
21. `MergedData.to_reflection_data()` → TorchRef pipeline
22. `MergedData.save_mtz()` via gemmi
23. `write_stream(mode="reconstruct")`
24. `write_stream(mode="patch")`

### Phase 6 — TR-SFX Extension
25. `time_id` at crystal level (crystals belong to time points)
26. Two-level scatter merge: per-time-point `MergedData` list
27. `SFXStream.from_streams({"t0": ..., "t1": ...})` multi-dataset constructor

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch` | Tensors, autograd, scatter ops |
| `safetensors` | Fast memory-mappable tensor serialization |
| `gemmi` | Space group ops, MTZ I/O, ASU reduction |
| `numpy` | Memmap, parser intermediate arrays |
| `h5py` | HDF5 image reading (Phase 5+) |
| `multiprocessing` / `concurrent.futures` | Parallel parsing |

---

## Memory Budget (reference)

At 1e6 crystals × 1000 refs/crystal:

| Array | dtype | Size |
|---|---|---|
| hkl (3 values) | int16 | 6 GB |
| I | float32 | 4 GB |
| sigma | float32 | 4 GB |
| crystal_id | int32 | 4 GB |
| fs, ss | float32 | 8 GB |
| panel_id | int16 | 2 GB |
| background | float32 | 4 GB |
| **Total on disk** | | **~32 GB** |
| **Merge subset** (hkl+I+sigma+cid) | | **~18 GB** |

All stored as memmaps — RAM footprint is only what the active computation touches.