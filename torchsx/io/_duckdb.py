"""Load a probixi DuckDB run into an :class:`~torchsx._core.SFXStreamCore`.

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

from dataclasses import replace
from pathlib import Path

import duckdb
import torch

from torchsx._compat import PROBIXI_VERSION, SUPPORTED_DB_SCHEMA, probixi_mismatch
from torchsx._core import SFXStreamCore
from torchsx.geometry import Geometry
from torchsx.io._metadata import StreamMetadata
from torchsx.measurement import MEASUREMENT_KEY, MeasurementSpec

#: 1 Å⁻¹ = 10 nm⁻¹ (``probixi.io.writer.A_INV_TO_NM_INV``).
_NM_INV_PER_A_INV = 10.0

#: ``resolution`` fallback (Å) when a frame has no diffraction limit, matching
#: the stream parser's sentinel.
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


#: relative spread of the per-rank gains above which the loader warns
RANK_GAIN_TOLERANCE = 0.02


def _calibration_rows(conn, frame_cols: set[str]) -> list[dict]:
    """Every ``calibration`` row, rank 0 first, each with ``n_frames``: the
    indexed frames its rank processed (one row and no ``rank``: every frame)."""
    cols = _columns(conn, "calibration")
    if "rank" in cols and "rank" in frame_cols:
        cur = conn.execute("""
            SELECT c.*, (SELECT count(*) FROM frames f
                         WHERE f.indexed AND f.rank IS NOT DISTINCT FROM c.rank) AS n_frames
            FROM calibration c ORDER BY c.rank NULLS FIRST""")
    else:
        cur = conn.execute("SELECT *, NULL AS n_frames FROM calibration")
    names = [d[0] for d in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


def _rank_gain(cals: list[dict], path: Path) -> tuple[float, int, float] | None:
    """``(gain, ranks, relative spread)``: the ranks' gains weighted by their
    indexed frames.  A multi-GPU probixi run calibrates each rank on its own
    frames; one detector should give one gain, so a spread above
    ``RANK_GAIN_TOLERANCE`` warns rather than silently picking rank 0."""
    rows = [r for r in cals if r.get("gain") is not None and float(r["gain"]) > 0]
    if not rows:
        return None
    gains = [float(r["gain"]) for r in rows]
    if len(rows) == 1:
        return gains[0], 1, 0.0
    weights = [float(r["n_frames"] or 0) for r in rows]
    if sum(weights) <= 0:
        weights = [1.0] * len(rows)
    gain = sum(g * w for g, w in zip(gains, weights)) / sum(weights)
    spread = (max(gains) - min(gains)) / gain
    if spread > RANK_GAIN_TOLERANCE:
        import warnings
        per_rank = ", ".join(f"rank {r.get('rank')}: {g:.4g}" for r, g in zip(rows, gains))
        warnings.warn(f"{path}: per-rank calibrated gains differ by {100 * spread:.1f} % "
                      f"({per_rank}); using their frame-weighted mean {gain:.4g}",
                      stacklevel=3)
    return gain, len(rows), spread


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
    if integ.get("bg_annulus_pixels"):  # pixels behind background_model's variance
        provenance.append(("bg_annulus_pixels", f"{float(integ['bg_annulus_pixels']):g}"))
    if integ.get("bg_model_var_scale") is not None:  # probixi >= 0.10.3 rescaled the variance
        provenance.append(("bg_model_var_scale", f"{float(integ['bg_model_var_scale']):.6g}"))
        if integ.get("bg_model_var_dmin_A") is not None:  # the fit's resolution cut
            provenance.append(("bg_model_var_dmin_A", f"{float(integ['bg_model_var_dmin_A']):.4g}"))
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
    cals = _calibration_rows(conn, frame_cols) if "calibration" in tables else []
    cal = cals[0] if cals else {}
    for row in cals:
        if (row.get("schema_version") or 0) > SUPPORTED_DB_SCHEMA:
            writer = f" (probixi {row['probixi_version']})" if row.get("probixi_version") else ""
            raise ValueError(f"{path}: probixi database schema {row['schema_version']}{writer} is "
                             f"newer than this TorchSX supports (schema {SUPPORTED_DB_SCHEMA}, the "
                             f"layout of probixi {PROBIXI_VERSION}); upgrade TorchSX")
    mismatch = probixi_mismatch()
    if mismatch:
        import warnings

        warnings.warn(mismatch, stacklevel=3)
    if cal:
        for k in ("probixi_version", "schema_version", "flux_variance", "mf_threshold", "tail_ratio"):
            if cal.get(k) is not None:
                provenance.append((k, str(cal[k])))
        rank_gain = _rank_gain(cals, path)
        if rank_gain is not None:
            gain, n_ranks, spread = rank_gain
            declarations = (("calibration", "adu_per_photon", float(gain)),)
            if n_ranks > 1:
                provenance.append(("calibration_ranks", str(n_ranks)))
                provenance.append(("gain_rank_spread", f"{spread:.4g}"))
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


#: probixi's default for ``--enrich-alpha``, used when the database records no alpha
DEFAULT_ENRICH_ALPHA = 1e-3
#: under ``auto``, the largest ``enrich_p`` of a second or later lattice of a frame
#: (probixi ``--max-lattices`` above one). Such a lattice is indexed on the peaks the
#: earlier ones left over. Read at the first lattice's alpha, the further lattices of
#: weakly diffracting crystals made the merge worse; at this bound the strong ones
#: still add their signal.
EXTRA_LATTICE_ENRICH_ALPHA = 1e-10


def parse_enrich_alpha(value) -> float | str | None:
    """``"auto"``, ``None`` (``"off"``) or a probability in (0, 1].

    ``auto`` takes the alpha the database records and falls back to
    :data:`DEFAULT_ENRICH_ALPHA`.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "auto":
            return "auto"
        if text in ("off", "none"):
            return None
        try:
            value = float(text)
        except ValueError:
            raise ValueError(f"enrich_alpha must be auto, off or a probability, got {value!r}") from None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"enrich_alpha must be auto, off or a probability, got {value!r}")
    alpha = float(value)
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"enrich_alpha must lie in (0, 1], got {alpha:g}")
    return alpha


def _enrichment_gate(conn, enrich_alpha="auto") -> dict:
    """The enrichment gate on an open probixi database.

    Returns ``predicate`` (SQL on ``crystals``, or None), ``alpha``,
    ``alpha_extra`` (the bound on further lattices of a frame, or None),
    ``origin`` and the crystal counts ``n_total`` and ``n_kept`` before any
    other filter.
    """
    alpha = parse_enrich_alpha(enrich_alpha)
    tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
    gate = {"predicate": None, "alpha": None, "alpha_extra": None, "origin": "off",
            "n_total": None, "n_kept": None}
    if "crystals" not in tables:
        gate["origin"] = "no crystals table"
        return gate
    n_total = int(conn.execute("SELECT count(*) FROM crystals").fetchone()[0])
    gate["n_total"] = gate["n_kept"] = n_total
    if alpha is None:
        return gate
    crystal_cols = _columns(conn, "crystals")
    if "enrich_p" not in crystal_cols or not conn.execute(
            "SELECT count(enrich_p) FROM crystals").fetchone()[0]:
        gate["origin"] = "no enrich_p recorded"
        return gate
    origin, extra = "requested", None
    if alpha == "auto":
        recorded = None
        if "integration" in tables and "bg_model_var_alpha" in _columns(conn, "integration"):
            recorded = _scalar_row(conn, "integration").get("bg_model_var_alpha")
        if recorded is not None and 0.0 < float(recorded) <= 1.0:
            alpha, origin = float(recorded), "database, integration.bg_model_var_alpha"
        else:
            alpha, origin = DEFAULT_ENRICH_ALPHA, "torchsx default"
        # further lattices of a frame: a given alpha gates every lattice alike
        if alpha > EXTRA_LATTICE_ENRICH_ALPHA and "lattice_index" in crystal_cols and conn.execute(
                "SELECT count(*) FROM crystals WHERE lattice_index > 0").fetchone()[0]:
            extra = EXTRA_LATTICE_ENRICH_ALPHA
    predicate = f"enrich_p <= {alpha!r}"
    if extra is not None:
        predicate += f" AND (coalesce(lattice_index, 0) = 0 OR enrich_p <= {extra!r})"
    n_kept = int(conn.execute(f"SELECT count(*) FROM crystals WHERE {predicate}").fetchone()[0])
    gate.update(predicate=predicate, alpha=alpha, alpha_extra=extra, origin=origin, n_kept=n_kept)
    return gate


def read_enrichment_gate(path: str | Path, enrich_alpha="auto") -> dict:
    """The enrichment gate :func:`load_duckdb` applies to a probixi run.

    probixi scores every lattice with ``enrich_p``, the probability that its
    predicted spots land on signal as often as they do by chance. A database
    keeps every lattice. The loader drops the lattices with
    ``enrich_p > alpha``. Under ``"auto"`` a second or later lattice of a
    frame must also reach :data:`EXTRA_LATTICE_ENRICH_ALPHA`.

    Parameters
    ----------
    path : the probixi database.
    enrich_alpha : ``"auto"`` (the alpha the database records in
        ``integration.bg_model_var_alpha``, else :data:`DEFAULT_ENRICH_ALPHA`,
        and :data:`EXTRA_LATTICE_ENRICH_ALPHA` for further lattices of a frame),
        a probability in (0, 1] for every lattice, or None / ``"off"`` for no gate.

    Returns
    -------
    dict
        ``predicate`` (SQL on ``crystals``, None without a gate), ``alpha``,
        ``alpha_extra``, ``origin``, ``n_total`` and ``n_kept``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such DuckDB run: {path}")
    conn = duckdb.connect(str(path), read_only=True)
    try:
        return _enrichment_gate(conn, enrich_alpha)
    finally:
        conn.close()


def read_measurement(path: str | Path) -> MeasurementSpec:
    """Measurement spec of a probixi run, read from its run-level tables."""
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
    crystals_where: str | None = None,
    enrich_alpha: float | str | None = "auto",
) -> SFXStreamCore:
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
    crystals_where : SQL predicate on the columns of probixi's ``crystals``
        table, for example ``enrich_p <= 0.05``. Only the crystals it accepts
        are loaded, and an image left without a crystal is omitted. It applies
        on top of the enrichment gate.
    enrich_alpha : the enrichment gate, see :func:`read_enrichment_gate`.
        ``"auto"`` (default) drops the lattices whose ``enrich_p`` exceeds the
        alpha the database records, else :data:`DEFAULT_ENRICH_ALPHA`, and the
        further lattices of a frame above :data:`EXTRA_LATTICE_ENRICH_ALPHA`.
        None or ``"off"`` loads every lattice.

    Returns
    -------
    SFXStreamCore
        Wrap with ``SFXStream(core)``, or call
        :meth:`~torchsx.stream.SFXStream.from_duckdb`.
    """
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
        gate = _enrichment_gate(conn, enrich_alpha)["predicate"]
        # The filter in force goes into the provenance: a stream saved from this
        # read (.sfx) then says which crystals it holds.
        spec = replace(spec, provenance=spec.provenance + (
            ("enrich_gate", gate or "off"),
            *((("crystals_where", str(crystals_where)),) if crystals_where else ())))
        if crystals_where:
            if not multi:
                raise ValueError(f"{path}: crystals_where needs a database with a crystals table")
            if ";" in crystals_where:
                raise ValueError("crystals_where takes one SQL predicate, without ';'")
        accepted = [f"({p})" for p in (gate, crystals_where) if p]
        if accepted:
            source = f"(SELECT * FROM crystals WHERE {' AND '.join(accepted)})"
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
        raise ValueError(f"{path}: no indexed frames (nothing to load)"
                         + (f" with crystals_where {crystals_where!r}" if crystals_where else "")
                         + (f" under the enrichment gate {gate} (enrich_alpha='off' loads "
                            f"every lattice)" if gate else ""))

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

    core = SFXStreamCore(
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
