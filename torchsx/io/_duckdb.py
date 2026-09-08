"""Load a probixi DuckDB run into an :class:`~torchsx._core._SFXStreamCore`.

Probixi v1 stores one crystal per indexed ``frames`` row. Version 2 separates
``frames`` and ``crystals`` and keys reflections by ``crystal_id``. Both map to
the existing observation → crystal → image tensor levels.

Database IDs are preserved as metadata. Dense tensor indices are assigned in
``(filename, event, lattice_index)`` order, independently of insertion order.
Unindexed images are omitted, as in the legacy reader.

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

Beam energy and camera length come from run-level geometry. Detector shifts
are unavailable and are emitted as zero-valued image columns.
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
    "astar_x",
    "astar_y",
    "astar_z",
    "bstar_x",
    "bstar_y",
    "bstar_z",
    "cstar_x",
    "cstar_y",
    "cstar_z",
)
_CELL_COLS = (
    "cell_a_A",
    "cell_b_A",
    "cell_c_A",
    "cell_alpha_deg",
    "cell_beta_deg",
    "cell_gamma_deg",
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

    Read legacy and version-2 databases, omitting unindexed images. Multiple
    crystals from one image retain a shared image index.

    Parameters
    ----------
    path : the ``.duckdb`` written by ``probixi.io.db.DuckDBOffloader``.
    spacegroup : space group of the run. probixi's ``cell`` table carries a
        lattice type and centering but not a space group, so it cannot be
        recovered from the database — pass it explicitly or accept ``P 1``.
    geometry : an already-built :class:`~torchsx.geometry.Geometry`. Takes
        precedence over ``load_geom_file``.
    load_geom_file : when no ``geometry`` is given, parse the geometry text
        embedded in ``geometry.geometry_file``. Legacy filename entries are
        also supported; missing files leave the geometry unset.

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

        tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
        versions = (
            conn.execute("SELECT version FROM schema_version").fetchall()
            if "schema_version" in tables
            else [(1,)]
        )
        if versions not in ([(1,)], [(2,)]):
            raise ValueError(f"{path}: unsupported Probixi schema version {versions}")
        v2 = versions == [(2,)]
        source = "crystals" if v2 else "frames"
        key = "crystal_id" if v2 else "frame_id"
        lattice = "c.lattice_index" if v2 else "0"
        conn.execute(f"""
            CREATE TEMP VIEW _images AS
            SELECT f.*, row_number() OVER (ORDER BY filename, event) - 1 AS iid
            FROM frames f WHERE f.indexed AND EXISTS (
                SELECT 1 FROM {source} c WHERE c.frame_id = f.frame_id AND c.indexed
            )
        """)
        images = conn.execute("SELECT * FROM _images ORDER BY iid").fetchnumpy()
        conn.execute(f"""
            CREATE TEMP VIEW _ordered AS
            SELECT c.*, i.iid, {lattice} AS lattice,
                   row_number() OVER (
                       ORDER BY i.iid, {lattice}, c.{key}
                   ) - 1 AS cid
            FROM {source} c JOIN _images i USING (frame_id) WHERE c.indexed
        """)
        crys = conn.execute("SELECT * FROM _ordered ORDER BY cid").fetchnumpy()
        refl = conn.execute(f"""
            SELECT r.h, r.k, r.l, r.intensity, r.sigma, r.peak, r.background,
                   r.fs, r.ss, r.panel, o.cid
            FROM reflections r JOIN _ordered o
              ON r.{key} = o.{key} AND r.frame_id = o.frame_id
            ORDER BY o.cid, r.h, r.k, r.l, r.panel, r.fs, r.ss,
                     r.intensity, r.sigma, r.peak, r.background
        """).fetchnumpy()
    finally:
        conn.close()

    n_crystals = len(crys["cid"])
    if n_crystals == 0:
        raise ValueError(f"{path}: no indexed frames (nothing to load)")

    def _tensor(table, name, dtype=torch.float32, fill=0):
        col = table[name]
        if hasattr(col, "filled"):
            col = col.filled(fill)
        return torch.as_tensor(col, dtype=dtype)

    def _c(name, dtype=torch.float32):
        return _tensor(crys, name, dtype)

    def _o(name, dtype=torch.float32):
        return _tensor(refl, name, dtype)

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

    image_id = _c("iid", torch.int32)
    n_images = len(images["iid"])

    # --- observation level --------------------------------------------
    hkl = torch.stack([_o(c, torch.int16) for c in ("h", "k", "l")], dim=1)
    panel_lookup = {name: i for i, name in enumerate(panel_names)}
    panel_col = refl["panel"]
    panel_id = torch.tensor(
        [panel_lookup.get(p, -1) for p in panel_col], dtype=torch.int16
    )

    metadata = StreamMetadata()
    metadata["image"] = {
        "filename": [str(f) for f in images["filename"]],
        "event": [int(e) for e in images["event"]],
        "frame_id": [str(f) for f in images["frame_id"]],
        "frame_index": [int(i) for i in images["frame_index"]],
    }
    # frame_id is kept verbatim rather than recomputed from (filename, event):
    # it is the key any write-back to probixi must use, and .sfx rewrites
    # filenames to absolute paths (``absolutize_image_filenames``), which would
    # change the sha1 it is derived from.
    metadata["crystal"] = {
        "panel_names": panel_names,
        "frame_id": [str(f) for f in crys["frame_id"]],
    }

    if v2:
        metadata["crystal"]["crystal_id"] = [str(c) for c in crys["crystal_id"]]

    if geometry is None and load_geom_file:
        geom = geom_row.get("geometry_file")
        if geom and ("\n" in geom or "=" in geom):
            geometry = Geometry._parse_geom_lines(geom.splitlines())
        elif geom and Path(geom).is_file():
            geometry = Geometry.from_geom_file(geom)

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
        n_refl=_c("num_reflections", torch.int32),
        image_id=image_id,
        crystal_in_image=_c("lattice", torch.int32),
        # Run-level in probixi's schema: constant across images.
        beam_energy=torch.full((n_images,), photon_eV, dtype=torch.float32),
        # Not in the schema; a constant column reads as uninformative. Allocated
        # separately -- aliasing two columns onto one storage breaks .sfx saving.
        det_shift_x=torch.zeros(n_images, dtype=torch.float32),
        det_shift_y=torch.zeros(n_images, dtype=torch.float32),
        n_peaks=_tensor(images, "n_peaks", torch.int32),
        camera_length=torch.full((n_images,), clen, dtype=torch.float32),
        geometry=geometry,
        metadata=metadata,
        spacegroup=spacegroup,
        source_path=str(path),
    )
