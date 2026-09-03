"""Load a probixi DuckDB run into an :class:`~torchsx._core._SFXStreamCore`.

probixi (``probixi.io.db``) writes one relational database per run: run-level
``geometry``/``panels``/``cell``, one ``frames`` row per file-event, and the
integrated ``reflections`` keyed by ``frame_id``. This is the ingest boundary —
the database is the archive, the stacked-tensor core is the working form.

Two things make the conversion more than a column rename:

**Keys.** ``frame_id`` is a stable content hash (sha1 of ``filename//event``);
``crystal_id`` is a *dense row position* into the crystal level, rewritten on
every filter (see ``_core._SFXStreamCore.filter``). The dense index is assigned
here, ordered by ``(filename, event)`` — never by ``frame_id`` and never by the
database's natural order, so that a reload of the same database yields the same
positions and any split label or checkpoint keyed on them stays valid.

**Units.** The database is mixed, and one column lies by omission:

===========================  ==================  ====================
DuckDB column                unit                torchsx column/unit
===========================  ==================  ====================
``cell_a_A`` … ``_gamma_deg``  Å, degrees        ``cell`` — same
``astar_x`` …                  nm⁻¹ *(unmarked)* ``astar`` — Å⁻¹
``profile_radius_nm_inv``      nm⁻¹              ``profile_rad`` — Å⁻¹
``diffraction_limit_nm_inv``   nm⁻¹              ``resolution`` — d_min in Å
===========================  ==================  ====================

``astar_x/y/z`` carry no unit suffix but ``probixi.io.db._frame_row`` multiplies
by ``A_INV_TO_NM_INV``, so they are nm⁻¹ like the rest.

Because probixi keys ``frames`` on one row per file-event, a run holds at most
one crystal per image; the image level is therefore 1:1 with the crystal level
and ``crystal_in_image`` is all zeros. ``beam_energy`` and ``camera_length``
come from the run-level ``geometry`` table (constant across images), and
``det_shift_x/y`` are absent from the schema and emitted as zeros — see
``CRYSTAL_FEAT_COLUMNS`` in ``torchsx.merging.observations``, which treats a
constant feature column as uninformative rather than failing.
"""

from __future__ import annotations

from pathlib import Path

import torch

from torchsx._core import _SFXStreamCore
from torchsx.geometry import Geometry
from torchsx.io._metadata import StreamMetadata

#: 1 Å⁻¹ = 10 nm⁻¹ (``probixi.io.writer.A_INV_TO_NM_INV``).
_NM_INV_PER_A_INV = 10.0

#: ``resolution`` fallback (Å) when a frame has no diffraction limit, matching
#: the stream parser's sentinel in ``io/_parse/records.py``.
_NO_RESOLUTION = 999.0

# Per-crystal columns, in the order the reciprocal/cell tensors are built.
_RECIP_COLS = (
    "astar_x", "astar_y", "astar_z",
    "bstar_x", "bstar_y", "bstar_z",
    "cstar_x", "cstar_y", "cstar_z",
)
_CELL_COLS = (
    "cell_a_A", "cell_b_A", "cell_c_A",
    "cell_alpha_deg", "cell_beta_deg", "cell_gamma_deg",
)


def _scalar_row(conn, table: str) -> dict:
    """First row of a single-row run-level table as a dict (``{}`` if empty)."""
    cur = conn.execute(f"SELECT * FROM {table}")  # noqa: S608 - fixed table names
    row = cur.fetchone()
    if row is None:
        return {}
    return dict(zip([d[0] for d in cur.description], row))


def load_duckdb(
    path: str | Path,
    spacegroup: str = "P 1",
    geometry: Geometry | None = None,
    load_geom_file: bool = True,
) -> _SFXStreamCore:
    """Build a stream core from a probixi DuckDB run.

    Only frames with ``indexed = TRUE`` become crystals; probixi records
    unindexed file-events as rows with null statistics and no reflections.

    Parameters
    ----------
    path : the ``.duckdb`` written by ``probixi.io.db.DuckDBOffloader``.
    spacegroup : space group of the run. probixi's ``cell`` table carries a
        lattice type and centering but not a space group, so it cannot be
        recovered from the database — pass it explicitly or accept ``P 1``.
    geometry : an already-built :class:`~torchsx.geometry.Geometry`. Takes
        precedence over ``load_geom_file``.
    load_geom_file : when no ``geometry`` is given, try to load the
        ``geometry.geometry_file`` path recorded in the database. The
        ``geometry`` table alone cannot reconstruct one (it has no per-panel
        corner or axis vectors), so this is the only route to real geometry;
        it is skipped without error when the path no longer exists.

    Returns
    -------
    _SFXStreamCore
        Wrap with ``SFXStream(core)``, or call
        :meth:`~torchsx.stream.SFXStream.from_duckdb`.
    """
    import duckdb

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such DuckDB run: {path}")

    conn = duckdb.connect(str(path), read_only=True)
    try:
        geom_row = _scalar_row(conn, "geometry")
        panel_names = [r[0] for r in conn.execute("SELECT name FROM panels").fetchall()]

        # Dense crystal_id, ordered by (filename, event): stable across reloads.
        # DuckDB assigns it and drives the reflection join, so neither the id
        # map nor the sort ever reaches Python.
        conn.execute(
            """
            CREATE TEMP VIEW _ordered AS
            SELECT frame_id,
                   row_number() OVER (ORDER BY filename, event) - 1 AS cid
            FROM frames
            WHERE indexed
            """
        )
        crys = conn.execute(
            f"""
            SELECT o.cid, f.frame_id, f.filename, f.event,
                   {', '.join('f.' + c for c in _RECIP_COLS)},
                   {', '.join('f.' + c for c in _CELL_COLS)},
                   f.profile_radius_nm_inv, f.diffraction_limit_nm_inv,
                   f.num_reflections, f.n_peaks
            FROM frames f JOIN _ordered o USING (frame_id)
            ORDER BY o.cid
            """
        ).fetchnumpy()

        refl = conn.execute(
            """
            SELECT r.h, r.k, r.l, r.intensity, r.sigma, r.peak, r.background,
                   r.fs, r.ss, r.panel, o.cid
            FROM reflections r JOIN _ordered o USING (frame_id)
            ORDER BY o.cid
            """
        ).fetchnumpy()
    finally:
        conn.close()

    n_crystals = len(crys["cid"])
    if n_crystals == 0:
        raise ValueError(f"{path}: no indexed frames (nothing to load)")

    def _c(name: str, fill: float = 0.0) -> torch.Tensor:
        """Crystal column → float32 tensor with nulls filled."""
        col = crys[name]
        if hasattr(col, "filled"):  # numpy masked array (nullable DOUBLE)
            col = col.filled(fill)
        return torch.as_tensor(col.astype("float64"), dtype=torch.float32)

    def _o(name: str, dtype=torch.float32, fill: float = 0.0) -> torch.Tensor:
        col = refl[name]
        if hasattr(col, "filled"):
            col = col.filled(fill)
        return torch.as_tensor(col.astype("float64"), dtype=dtype)

    # --- crystal level ------------------------------------------------
    recip = torch.stack([_c(c) for c in _RECIP_COLS], dim=1) / _NM_INV_PER_A_INV
    astar, bstar, cstar = recip[:, 0:3], recip[:, 3:6], recip[:, 6:9]
    cell = torch.stack([_c(c) for c in _CELL_COLS], dim=1)  # already Å / degrees

    profile_rad = _c("profile_radius_nm_inv") / _NM_INV_PER_A_INV
    limit_nm_inv = _c("diffraction_limit_nm_inv")
    resolution = torch.where(
        limit_nm_inv > 0,
        _NM_INV_PER_A_INV / limit_nm_inv.clamp(min=1e-12),
        torch.full_like(limit_nm_inv, _NO_RESOLUTION),
    )

    # One crystal per frame in probixi's schema, so image ≡ crystal, 1:1.
    image_id = torch.arange(n_crystals, dtype=torch.int32)

    # --- observation level --------------------------------------------
    hkl = torch.stack([_o("h"), _o("k"), _o("l")], dim=1).to(torch.int16)
    panel_lookup = {name: i for i, name in enumerate(panel_names)}
    panel_col = refl["panel"]
    panel_id = torch.tensor(
        [panel_lookup.get(p, -1) for p in panel_col], dtype=torch.int16
    )

    metadata = StreamMetadata()
    metadata["image"] = {
        "filename": [str(f) for f in crys["filename"]],
        "event": [int(e) for e in crys["event"]],
    }
    # frame_id is kept verbatim rather than recomputed from (filename, event):
    # it is the key any write-back to probixi must use, and .sfx rewrites
    # filenames to absolute paths (``absolutize_image_filenames``), which would
    # change the sha1 it is derived from.
    metadata["crystal"] = {
        "panel_names": panel_names,
        "frame_id": [str(f) for f in crys["frame_id"]],
    }

    if geometry is None and load_geom_file:
        geom_path = geom_row.get("geometry_file")
        if geom_path and Path(geom_path).exists():
            geometry = Geometry.from_geom_file(geom_path)

    photon_eV = float(geom_row.get("photon_energy_eV") or 0.0)
    clen = float(geom_row.get("clen") or 0.0)

    return _SFXStreamCore(
        hkl=hkl,
        I=_o("intensity"),
        sigma=_o("sigma"),
        background=_o("background"),
        peak=_o("peak"),
        fs=_o("fs"),
        ss=_o("ss"),
        panel_id=panel_id,
        crystal_id=_o("cid", dtype=torch.int32),
        astar=astar,
        bstar=bstar,
        cstar=cstar,
        cell=cell,
        profile_rad=profile_rad,
        resolution=resolution,
        n_refl=_c("num_reflections").to(torch.int32),
        image_id=image_id,
        crystal_in_image=torch.zeros(n_crystals, dtype=torch.int32),
        # Run-level in probixi's schema: constant across images.
        beam_energy=torch.full((n_crystals,), photon_eV, dtype=torch.float32),
        # Not in the schema; a constant column reads as uninformative. Allocated
        # separately -- aliasing two columns onto one storage breaks .sfx saving.
        det_shift_x=torch.zeros(n_crystals, dtype=torch.float32),
        det_shift_y=torch.zeros(n_crystals, dtype=torch.float32),
        n_peaks=_c("n_peaks").to(torch.int32),
        camera_length=torch.full((n_crystals,), clen, dtype=torch.float32),
        geometry=geometry,
        metadata=metadata,
        spacegroup=spacegroup,
        source_path=str(path),
    )
