"""Load a probixi DuckDB run into an :class:`~torchsx._core._SFXStreamCore`.

One ``frames`` row per image and one ``crystals`` row per lattice (older
databases keep the crystal on ``frames``) map to the observation → crystal →
image tensor levels.

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
``peak_resolution_nm_inv``     nm⁻¹              ``peak_resolution`` — Å
===========================  ==================  ====================

``astar_x/y/z`` carry no unit suffix but ``probixi.io.db._frame_row`` multiplies
by ``A_INV_TO_NM_INV``, so they are nm⁻¹ like the rest.

probixi's diagnostics become extra crystal, image and observation columns;
the ``integration`` and ``calibration`` tables give the measurement spec.

Beam energy is each frame's ``photon_energy_eV`` where recorded, else the
run-level geometry's, as is the camera length. Detector shifts are
unavailable and are emitted as zero-valued image columns.
"""

from __future__ import annotations

from pathlib import Path

import torch

from torchsx._core import _SFXStreamCore
from torchsx.geometry import Geometry
from torchsx.io._metadata import StreamMetadata
from torchsx.measurement import MEASUREMENT_KEY, MeasurementSpec

#: 1 Å⁻¹ = 10 nm⁻¹ (``probixi.io.writer.A_INV_TO_NM_INV``).
_NM_INV_PER_A_INV = 10.0

#: ``resolution`` fallback (Å) when a frame has no diffraction limit, matching
#: the stream parser's sentinel in ``io/_parse/records.py``.
_NO_RESOLUTION = 999.0
#: newest probixi database schema this reader understands
SUPPORTED_DB_SCHEMA = 1

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
# probixi diagnostics: (DuckDB column, torchsx column, dtype, NULL fill).
_CRYSTAL_EXTRA = (
    ("n_indexed", "n_indexed", torch.float32, 0.0),
    ("rmsd", "rmsd", torch.float32, 0.0),
    ("mosaicity_deg", "mosaicity", torch.float32, 0.0),
    ("enrichment", "enrichment", torch.float32, 1.0),
    ("n_bright", "n_bright", torch.float32, 0.0),
    ("enrich_p", "enrich_p", torch.float32, 1.0),
)
_IMAGE_EXTRA = (
    ("scale", "scale", torch.float32, 1.0),
    ("scale_sigma", "scale_sigma", torch.float32, 0.0),
)
_OBS_EXTRA = (
    ("n_pixels", "n_pixels", torch.float32, 0.0),
    ("background_model", "background_model", torch.float32, float("nan")),
    ("background_model_var", "background_model_var", torch.float32, float("nan")),
)


def _scalar_row(conn, table: str) -> dict:
    """First row of a single-row run-level table as a dict (``{}`` if empty)."""
    cur = conn.execute(f"SELECT * FROM {table}")  # noqa: S608 - fixed table names
    row = cur.fetchone()
    if row is None:
        return {}
    return dict(zip([d[0] for d in cur.description], row))


def _columns(conn, table: str) -> set[str]:
    return {r[0] for r in conn.execute(f"DESCRIBE {table}").fetchall()}  # noqa: S608


def _measurement(conn, path: Path) -> MeasurementSpec:
    tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
    integ = _scalar_row(conn, "integration") if "integration" in tables else {}
    geom = _scalar_row(conn, "geometry")
    radii = None
    if all(integ.get(k) is not None for k in ("r_peak", "r_gap", "r_bg")):
        radii = (float(integ["r_peak"]), float(integ["r_gap"]), float(integ["r_bg"]))
    gain = integ.get("adu_per_photon")
    if gain is None:
        gain = geom.get("adu_per_photon")
    declarations = () if gain is None else (("global", "adu_per_photon", float(gain)),)
    n_peak = n_bg = None
    if radii is not None:
        from torchsx.merging.hkl.noise import _crystfel_pixel_counts
        n_peak, n_bg = _crystfel_pixel_counts(*radii)
    provenance = [("producer", "probixi")]
    if integ.get("aperture"):
        provenance.append(("aperture", str(integ["aperture"])))
    frame_cols = _columns(conn, "frames") if "frames" in tables else set()
    if "photon_energy_eV" in frame_cols:
        lo, hi = conn.execute("SELECT min(photon_energy_eV), max(photon_energy_eV) FROM frames "
                              "WHERE indexed AND photon_energy_eV > 0").fetchone()
        if lo and hi and hi / lo > 1.0001:  # frames processed at their own wavelengths
            provenance.append(("photon_energy", "per-frame"))
    if "crystals" in tables:
        # 10th percentile of the crystals' diffraction limits
        p10 = conn.execute("SELECT quantile_cont(10.0 / diffraction_limit_nm_inv, 0.1) FROM crystals "
                           "WHERE diffraction_limit_nm_inv > 0").fetchone()[0]
        if p10:
            provenance.append(("diffraction_limit_p10_A", f"{float(p10):.3f}"))
    cal = _scalar_row(conn, "calibration") if "calibration" in tables else {}
    if cal:
        if (cal.get("schema_version") or 0) > SUPPORTED_DB_SCHEMA:
            raise ValueError(f"{path}: probixi database schema {cal['schema_version']} is newer "
                             f"than this TorchSX supports ({SUPPORTED_DB_SCHEMA})")
        for k in ("probixi_version", "schema_version", "flux_variance", "mf_threshold", "tail_ratio"):
            if cal.get(k) is not None:
                provenance.append((k, str(cal[k])))
        if cal.get("gain") is not None:
            gain = cal["gain"]
            declarations = (("calibration", "adu_per_photon", float(gain)),)
    unresolved = [k for k, v in (("radii", radii), ("gain", gain)) if v is None]
    return MeasurementSpec(
        producer="probixi",
        producer_version=str(cal["probixi_version"]) if cal.get("probixi_version") else None,
        integration_method="rings",
        intensity_unit="adu", intensity_definition="background_subtracted_sum",
        background_unit="adu/pixel", background_definition="local_mean_per_pixel",
        uncertainty_definition="supplied_sigma_of_intensity",
        gain_declarations=declarations, radii=radii,
        nominal_n_peak=n_peak, nominal_n_bg=n_bg,
        pixel_count_definition="recorded_per_observation" if radii else "unknown",
        source_files=(str(path.resolve()),), provenance=tuple(provenance),
        unresolved=tuple(unresolved),
    )


def read_measurement(path: str | Path) -> MeasurementSpec:
    """Measurement spec of a probixi run, read from its run-level tables."""
    import duckdb

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such DuckDB run: {path}")
    conn = duckdb.connect(str(path), read_only=True)
    try:
        return _measurement(conn, path)
    finally:
        conn.close()


def load_duckdb(
    path: str | Path,
    spacegroup: str = "P 1",
    geometry: Geometry | None = None,
    load_geom_file: bool = True,
    clipped: bool = False,
) -> _SFXStreamCore:
    """Build a stream core from a probixi DuckDB run.

    Read legacy and multi-lattice databases, omitting unindexed images. Multiple
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
    clipped : keep disks clipped by the mask, the frame edge or a neighbour;
        dropped by default, as probixi's stream writer does.

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
        spec = _measurement(conn, path)

        tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
        multi = "crystals" in tables
        source = "crystals" if multi else "frames"
        key = "crystal_id" if multi else "frame_id"
        lattice = "c.lattice_index" if multi else "0"
        refl_cols = _columns(conn, "reflections")
        obs_extra = [e for e in _OBS_EXTRA if e[0] in refl_cols]
        flagged = ""
        if not clipped and {"n_masked", "n_overlap"} <= refl_cols:
            flagged = ("WHERE coalesce(r.n_masked, 0) = 0 "
                       "AND coalesce(r.n_overlap, 0) = 0")
        conn.execute(f"""
            CREATE TEMP VIEW _images AS
            SELECT f.*, row_number() OVER (ORDER BY filename, event) - 1 AS iid
            FROM frames f WHERE f.indexed AND EXISTS (
                SELECT 1 FROM {source} c WHERE c.frame_id = f.frame_id
            )
        """)
        images = conn.execute("SELECT * FROM _images ORDER BY iid").fetchnumpy()
        conn.execute(f"""
            CREATE TEMP VIEW _ordered AS
            SELECT c.*, i.iid, {lattice} AS lattice,
                   row_number() OVER (
                       ORDER BY i.iid, {lattice}, c.{key}
                   ) - 1 AS cid
            FROM {source} c JOIN _images i USING (frame_id)
        """)
        crys = conn.execute("SELECT * FROM _ordered ORDER BY cid").fetchnumpy()
        extra_sql = "".join(f", r.{c}" for c, *_ in obs_extra)
        refl = conn.execute(f"""
            SELECT r.h, r.k, r.l, r.intensity, r.sigma, r.peak, r.background,
                   r.fs, r.ss, r.panel, o.cid{extra_sql}
            FROM reflections r JOIN _ordered o
              ON r.{key} = o.{key} AND r.frame_id = o.frame_id
            {flagged}
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

    def _d_spacing(limit_nm_inv, none=_NO_RESOLUTION):
        return torch.where(
            limit_nm_inv > 0,
            _NM_INV_PER_A_INV / limit_nm_inv.clamp(min=1e-12),
            torch.full_like(limit_nm_inv, none),
        )

    resolution = _d_spacing(_c("diffraction_limit_nm_inv"))

    image_id = _c("iid", torch.int32)
    n_images = len(images["iid"])

    # --- observation level --------------------------------------------
    hkl = torch.stack([_o(c, torch.int16) for c in ("h", "k", "l")], dim=1)
    panel_lookup = {name: i for i, name in enumerate(panel_names)}
    panel_col = refl["panel"]
    panel_id = torch.tensor(
        [panel_lookup.get(p, -1) for p in panel_col], dtype=torch.int16
    )
    crystal_id = _o("cid", dtype=torch.int32)
    # reflections actually loaded, as the stream parser counts them
    n_refl = torch.bincount(crystal_id.long(), minlength=n_crystals).to(torch.int32)

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
    if multi:
        metadata["crystal"]["crystal_id"] = [str(c) for c in crys["crystal_id"]]
    metadata[MEASUREMENT_KEY] = {"record": spec.to_dict()}

    if geometry is None and load_geom_file:
        geom = geom_row.get("geometry_file")
        if geom and ("\n" in geom or "=" in geom):
            geometry = Geometry._parse_geom_lines(geom.splitlines())
        elif geom and Path(geom).is_file():
            geometry = Geometry.from_geom_file(geom)

    photon_eV = float(geom_row.get("photon_energy_eV") or 0.0)
    clen = float(geom_row.get("clen") or 0.0)
    # per-frame photon energy where recorded, else the run's
    beam_energy = torch.full((n_images,), photon_eV, dtype=torch.float32)
    if "photon_energy_eV" in images:
        per_frame = _tensor(images, "photon_energy_eV", torch.float32, float("nan"))
        beam_energy = torch.where(torch.isfinite(per_frame) & (per_frame > 0), per_frame, beam_energy)

    core = _SFXStreamCore(
        hkl=hkl,
        I=_o("intensity"),
        sigma=_o("sigma"),
        background=_o("background"),
        peak=_o("peak"),
        fs=_o("fs"),
        ss=_o("ss"),
        panel_id=panel_id,
        crystal_id=crystal_id,
        astar=astar,
        bstar=bstar,
        cstar=cstar,
        cell=cell,
        profile_rad=profile_rad,
        resolution=resolution,
        n_refl=n_refl,
        image_id=image_id,
        crystal_in_image=_c("lattice", torch.int32),
        beam_energy=beam_energy,
        # Not in the schema; a constant column reads as uninformative. Allocated
        # separately -- aliasing two columns onto one storage breaks .sfx saving.
        det_shift_x=torch.zeros(n_images, dtype=torch.float32),
        det_shift_y=torch.zeros(n_images, dtype=torch.float32),
        n_peaks=_tensor(images, "n_peaks", torch.int32),
        camera_length=torch.full((n_images,), clen, dtype=torch.float32),
        geometry=geometry,
        metadata=metadata,
        friedel_merged=False,
        spacegroup=spacegroup,
        source_path=str(path),
    )
    # the run's diagnostics, where the database has them
    core = core.with_columns("crystal", **{
        name: _tensor(crys, col, dtype, fill)
        for col, name, dtype, fill in _CRYSTAL_EXTRA if col in crys
    })
    image_extra = {
        name: _tensor(images, col, dtype, fill)
        for col, name, dtype, fill in _IMAGE_EXTRA if col in images
    }
    if "peak_resolution_nm_inv" in images:
        # NaN rather than the 999 Å sentinel: a feature, not a cut
        image_extra["peak_resolution"] = _d_spacing(_tensor(images, "peak_resolution_nm_inv"),
                                                    none=float("nan"))
    core = core.with_columns("image", **image_extra)
    core = core.with_columns("obs", **{
        name: _tensor(refl, col, dtype, fill) for col, name, dtype, fill in obs_extra
    })
    return core
