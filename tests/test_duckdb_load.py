"""``SFXStream.from_duckdb`` — probixi's relational run → stacked tensors.

The fixture builds a database against probixi's own DDL (copied verbatim from
``probixi/io/db.py``; ``test_schema_matches_probixi`` re-checks it against the
installed source when there is one), so the mapping under test is the real one.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

duckdb = pytest.importorskip("duckdb")

from torchsx.io._duckdb import load_duckdb  # noqa: E402
from torchsx.stream import SFXStream  # noqa: E402

# Verbatim from probixi.io.db._SCHEMA (v0.3.1).
_SCHEMA = """
CREATE TABLE geometry (
    beam_center_row  DOUBLE,
    beam_center_col  DOUBLE,
    clen             DOUBLE,
    pixel_size       DOUBLE,
    wavelength       DOUBLE,
    photon_energy_eV DOUBLE,
    adu_per_photon   DOUBLE,
    n_panels         INTEGER,
    geometry_file    VARCHAR
);

CREATE TABLE panels (
    name   VARCHAR,
    min_fs INTEGER,
    max_fs INTEGER,
    min_ss INTEGER,
    max_ss INTEGER
);

CREATE TABLE cell (
    a_A          DOUBLE,
    b_A          DOUBLE,
    c_A          DOUBLE,
    alpha_deg    DOUBLE,
    beta_deg     DOUBLE,
    gamma_deg    DOUBLE,
    volume_A3    DOUBLE,
    lattice_type VARCHAR,
    centering    VARCHAR,
    unique_axis  VARCHAR
);

CREATE TABLE frames (
    frame_id                 VARCHAR PRIMARY KEY,
    frame_index              BIGINT,
    filename                 VARCHAR,
    event                    BIGINT,
    indexed                  BOOLEAN,
    serial                   BIGINT,
    n_peaks                  INTEGER,
    n_indexed                INTEGER,
    rmsd                     DOUBLE,
    scale                    DOUBLE,
    scale_sigma              DOUBLE,
    mosaicity_deg            DOUBLE,
    profile_radius_nm_inv    DOUBLE,
    enrichment               DOUBLE,
    n_bright                 INTEGER,
    enrich_p                 DOUBLE,
    diffraction_limit_nm_inv DOUBLE,
    peak_resolution_nm_inv   DOUBLE,
    num_reflections          INTEGER,
    cell_a_A                 DOUBLE,
    cell_b_A                 DOUBLE,
    cell_c_A                 DOUBLE,
    cell_alpha_deg           DOUBLE,
    cell_beta_deg            DOUBLE,
    cell_gamma_deg           DOUBLE,
    astar_x                  DOUBLE,
    astar_y                  DOUBLE,
    astar_z                  DOUBLE,
    bstar_x                  DOUBLE,
    bstar_y                  DOUBLE,
    bstar_z                  DOUBLE,
    cstar_x                  DOUBLE,
    cstar_y                  DOUBLE,
    cstar_z                  DOUBLE
);

CREATE TABLE reflections (
    frame_id          VARCHAR,
    h                 INTEGER,
    k                 INTEGER,
    l                 INTEGER,
    intensity         DOUBLE,
    sigma             DOUBLE,
    peak              DOUBLE,
    background        DOUBLE,
    fs                DOUBLE,
    ss                DOUBLE,
    panel             VARCHAR,
    resolution_nm_inv DOUBLE
);

CREATE TABLE peaks (
    frame_id          VARCHAR,
    fs                DOUBLE,
    ss                DOUBLE,
    intensity         DOUBLE,
    resolution_nm_inv DOUBLE,
    panel             VARCHAR
);
"""

PHOTON_EV = 11330.0
CLEN = 0.0938
PANELS = ["p0", "p1"]


def _frame_id(filename: str, event: int) -> str:
    """probixi.io.db.frame_id."""
    return hashlib.sha1(f"{filename}//{event}".encode("utf-8")).hexdigest()[:16]


# (filename, event, indexed, n_peaks, n_refl, profile_nm_inv, limit_nm_inv, astar_x_nm_inv)
# Inserted out of (filename, event) order, and the two indexed events are chosen
# so their frame_id hashes sort the *other* way round — otherwise
# test_crystal_order_is_filename_event_not_frame_id would pass either way.
_FRAMES = [
    ("run_b.h5", 8, True, 140, 3, 0.40, 3.0, 1.5),
    ("run_a.h5", 1, True, 90, 2, 0.20, 2.5, 0.5),
    ("run_a.h5", 3, False, 4, 0, None, None, None),  # unindexed: no reflections
]


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "run.duckdb"
    conn = duckdb.connect(str(path))
    conn.execute(_SCHEMA)
    conn.execute(
        "INSERT INTO geometry VALUES (512.0, 512.0, ?, 7.5e-5, 1.094, ?, 1.0, 2, ?)",
        [CLEN, PHOTON_EV, str(tmp_path / "absent.geom")],
    )
    for i, name in enumerate(PANELS):
        conn.execute(
            "INSERT INTO panels VALUES (?, 0, 1023, ?, ?)",
            [name, i * 512, (i + 1) * 512 - 1],
        )
    conn.execute(
        "INSERT INTO cell VALUES (79.0, 79.0, 38.0, 90.0, 90.0, 90.0, 237158.0, "
        "'tetragonal', 'P', 'c')"
    )

    for serial, (fn, ev, indexed, n_peaks, n_refl, prof, limit, ax) in enumerate(
        _FRAMES
    ):
        fid = _frame_id(fn, ev)
        if not indexed:
            conn.execute(
                "INSERT INTO frames (frame_id, frame_index, filename, event, indexed, "
                "serial, n_peaks, n_indexed) VALUES (?, ?, ?, ?, FALSE, ?, ?, 0)",
                [fid, serial, fn, ev, serial, n_peaks],
            )
            continue
        conn.execute(
            "INSERT INTO frames VALUES (?, ?, ?, ?, TRUE, ?, ?, ?, 0.0004, 1.0, 0.05, "
            "0.1, ?, 2.0, 30, 0.01, ?, 3.2, ?, "
            "79.0, 79.0, 38.0, 90.0, 90.0, 90.0, "
            "?, 0.0, 0.0, 0.0, ?, 0.0, 0.0, 0.0, ?)",
            [
                fid,
                serial,
                fn,
                ev,
                serial,
                n_peaks,
                n_refl,
                prof,
                limit,
                n_refl,
                ax,
                ax * 2,
                ax * 4,
            ],
        )
        for j in range(n_refl):
            conn.execute(
                "INSERT INTO reflections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    fid,
                    j + 1,
                    -j,
                    2 * j,
                    100.0 + j,
                    10.0 + j,
                    120.0 + j,
                    5.0,
                    10.0 * j,
                    20.0 * j,
                    PANELS[j % 2],
                    2.0,
                ],
            )
    conn.close()
    return path


def _indexed_sorted():
    """The indexed frames in the order load_duckdb must assign."""
    return sorted([f for f in _FRAMES if f[2]], key=lambda f: (f[0], f[1]))


def test_levels_and_counts(db_path):
    core = load_duckdb(db_path, spacegroup="P 43 21 2")
    expected = _indexed_sorted()
    assert core.levels["crystal"].n == len(expected) == 2
    assert (
        core.levels["image"].n == 2
    ), "image level is 1:1 with crystal in probixi's schema"
    assert core.levels["obs"].n == sum(f[4] for f in expected) == 5
    assert core.spacegroup == "P 43 21 2"


def test_unindexed_frames_are_dropped(db_path):
    core = load_duckdb(db_path)
    filenames = core.metadata["image"]["filename"]
    events = core.metadata["image"]["event"]
    assert ("run_a.h5", 3) not in list(zip(filenames, events))


def test_crystal_order_is_filename_event_not_frame_id(db_path):
    """The dense index must follow (filename, event), never the id hash."""
    core = load_duckdb(db_path)
    expected = _indexed_sorted()
    got = list(zip(core.metadata["image"]["filename"], core.metadata["image"]["event"]))
    assert got == [(f[0], f[1]) for f in expected]

    # Guard the guard: this fixture must actually discriminate the two orders,
    # or the assertion above proves nothing.
    by_hash = sorted(expected, key=lambda f: _frame_id(f[0], f[1]))
    assert (
        by_hash != expected
    ), "fixture no longer distinguishes hash order from (filename, event)"


def test_reload_is_deterministic(db_path):
    a = load_duckdb(db_path)
    b = load_duckdb(db_path)
    assert torch.equal(a.crystal_id, b.crystal_id)
    assert torch.equal(a.astar, b.astar)
    assert a.metadata["image"]["filename"] == b.metadata["image"]["filename"]


def test_crystal_id_is_dense_and_groups_observations(db_path):
    core = load_duckdb(db_path)
    cid = core.crystal_id
    assert cid.dtype == torch.int32
    assert int(cid.min()) == 0
    assert int(cid.max()) == core.levels["crystal"].n - 1
    # every crystal is referenced, and reflections arrive grouped
    assert torch.equal(
        cid.unique(), torch.arange(core.levels["crystal"].n, dtype=torch.int32)
    )
    assert torch.equal(cid, cid.sort().values)
    counts = torch.bincount(cid.long())
    assert counts.tolist() == [f[4] for f in _indexed_sorted()]


def test_reciprocal_vectors_converted_nm_inv_to_a_inv(db_path):
    core = load_duckdb(db_path)
    expected = _indexed_sorted()
    for i, f in enumerate(expected):
        ax_nm_inv = f[7]
        assert core.astar[i, 0].item() == pytest.approx(ax_nm_inv / 10.0)
        assert core.bstar[i, 1].item() == pytest.approx(ax_nm_inv * 2 / 10.0)
        assert core.cstar[i, 2].item() == pytest.approx(ax_nm_inv * 4 / 10.0)


def test_cell_is_passed_through_in_angstrom(db_path):
    core = load_duckdb(db_path)
    for row in core.cell:
        assert row[:3].tolist() == pytest.approx([79.0, 79.0, 38.0])
        assert row[3:].tolist() == pytest.approx([90.0, 90.0, 90.0])


def test_profile_radius_and_resolution_units(db_path):
    core = load_duckdb(db_path)
    for i, f in enumerate(_indexed_sorted()):
        prof_nm_inv, limit_nm_inv = f[5], f[6]
        assert core.profile_rad[i].item() == pytest.approx(prof_nm_inv / 10.0)
        # resolution is d_min in Å, from a reciprocal limit in nm^-1
        assert core.resolution[i].item() == pytest.approx(10.0 / limit_nm_inv)


def test_panel_names_resolve_to_indices(db_path):
    core = load_duckdb(db_path)
    assert core.metadata["crystal"]["panel_names"] == PANELS
    assert core.panel_id.dtype == torch.int16
    assert int(core.panel_id.min()) >= 0, "every panel string matched a panels row"
    assert set(core.panel_id.tolist()) <= set(range(len(PANELS)))


def test_run_level_beam_metadata_is_constant(db_path):
    core = load_duckdb(db_path)
    assert torch.allclose(
        core.beam_energy, torch.full_like(core.beam_energy, PHOTON_EV)
    )
    assert torch.allclose(core.camera_length, torch.full_like(core.camera_length, CLEN))
    # absent from probixi's schema -> constant, which the scale encoder ignores
    assert torch.count_nonzero(core.det_shift_x) == 0
    assert torch.count_nonzero(core.det_shift_y) == 0


def test_no_columns_share_storage(db_path):
    """Two columns aliasing one storage silently breaks .sfx saving."""
    core = load_duckdb(db_path)
    seen: dict[int, str] = {}
    for level_name, level in core.levels.items():
        for col, tensor in level.columns.items():
            ptr = tensor.data_ptr()
            if tensor.numel() == 0:
                continue
            assert (
                ptr not in seen
            ), f"{level_name}/{col} shares storage with {seen[ptr]}"
            seen[ptr] = f"{level_name}/{col}"


def test_sfx_round_trip(db_path, tmp_path):
    """A DuckDB-sourced stream must survive the working-format round-trip."""
    pytest.importorskip("safetensors")
    stream = SFXStream.from_duckdb(db_path, spacegroup="P 43 21 2", merge_friedel=False)
    out = tmp_path / "run.sfx"
    stream.save(out)
    back = SFXStream.load(out, merge_friedel=False)

    assert back.n_crystals == stream.n_crystals
    assert back.n_observations == stream.n_observations
    a, b = stream._core, back._core
    for col in ("hkl", "crystal_id", "panel_id", "image_id"):
        assert torch.equal(getattr(b, col), getattr(a, col)), col
    for col in (
        "I",
        "sigma",
        "astar",
        "bstar",
        "cstar",
        "cell",
        "profile_rad",
        "resolution",
        "beam_energy",
    ):
        assert torch.allclose(getattr(b, col), getattr(a, col)), col
    # .sfx absolutizes image filenames, so compare basenames — and rely on
    # frame_id, not the filename, as the durable key back to probixi.
    assert [Path(f).name for f in b.metadata["image"]["filename"]] == [
        Path(f).name for f in a.metadata["image"]["filename"]
    ]
    assert b.metadata["image"]["event"] == a.metadata["image"]["event"]
    assert b.metadata["crystal"]["panel_names"] == a.metadata["crystal"]["panel_names"]
    assert b.metadata["crystal"]["frame_id"] == a.metadata["crystal"]["frame_id"]


def test_frame_id_is_carried_verbatim(db_path):
    """The write-back key must survive as data, not be re-derived from filenames."""
    core = load_duckdb(db_path)
    got = core.metadata["crystal"]["frame_id"]
    expected = [_frame_id(f[0], f[1]) for f in _indexed_sorted()]
    assert got == expected
    assert len(set(got)) == len(got), "frame_id must be unique per crystal"


@pytest.mark.parametrize("source", ["db_path", "v2_path"])
def test_feeds_the_merging_stack(source, request):
    """The point of the loader: merging accepts it without a stream file."""
    from torchsx.merging import MergeConfig, prepare_observations

    stream = SFXStream.from_duckdb(
        request.getfixturevalue(source), spacegroup="P 43 21 2"
    )
    prepared = prepare_observations(stream, MergeConfig(gain="trust_input"))
    assert prepared is not None

    from torchsx.merging.hkl import compute_hkl_index, merge_intensities

    idx = compute_hkl_index(stream.hkl)
    I_m, sig_m, mult = merge_intensities(
        idx.hkl_id, stream.I, stream.sigma, idx.n_unique
    )
    assert I_m.numel() == idx.n_unique
    assert int(mult.sum()) == stream.n_observations
    assert torch.isfinite(I_m).all() and torch.isfinite(sig_m).all()


def test_crystal_feature_table_shape_and_dead_columns(db_path):
    """probixi supplies 9 of 12 encoder features; 3 are structurally constant."""
    from torchsx.merging.observations import (
        CRYSTAL_FEAT_COLUMNS,
        _crystal_feature_table,
    )

    stream = SFXStream.from_duckdb(db_path, spacegroup="P 43 21 2")
    tbl = _crystal_feature_table(stream._core, stream.n_crystals)
    assert tbl.shape == (stream.n_crystals, len(CRYSTAL_FEAT_COLUMNS))

    dead = {
        CRYSTAL_FEAT_COLUMNS[i]
        for i in range(tbl.shape[1])
        if float(tbl[:, i].std()) == 0.0
    }
    # These three cannot vary: absent from probixi's schema or run-level in it.
    assert {"beam_energy", "det_shift_x", "det_shift_y"} <= dead
    # These must survive the mapping as live, varying features.
    assert dead.isdisjoint({"log_n_refl"})


def test_hkl_dtype_and_values(db_path):
    core = load_duckdb(db_path)
    assert core.hkl.dtype == torch.int16
    assert core.hkl.shape == (5, 3)


def test_missing_geom_file_is_not_fatal(db_path):
    """The recorded geometry_file is gone; loading still succeeds."""
    core = load_duckdb(db_path)
    assert core.geometry is None


def test_from_duckdb_wraps_and_symmetrizes(db_path):
    stream = SFXStream.from_duckdb(db_path, spacegroup="P 43 21 2")
    assert isinstance(stream, SFXStream)
    assert stream.n_crystals == 2
    assert stream._core.friedel_merged is True

    unmerged = SFXStream.from_duckdb(
        db_path, spacegroup="P 43 21 2", merge_friedel=False
    )
    assert unmerged._core.friedel_merged is False


def test_no_indexed_frames_raises(tmp_path):
    path = tmp_path / "empty.duckdb"
    conn = duckdb.connect(str(path))
    conn.execute(_SCHEMA)
    conn.execute("INSERT INTO geometry VALUES (0,0,0.1,7.5e-5,1.0,10000.0,1.0,0,'')")
    conn.close()
    with pytest.raises(ValueError, match="no indexed frames"):
        load_duckdb(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_duckdb(tmp_path / "nope.duckdb")


def test_schema_matches_probixi():
    """Our copied DDL must still match probixi's, when its source is reachable."""
    import re

    here = Path(__file__).resolve()
    rel = Path("probixi") / "probixi" / "io" / "db.py"
    # tests/ sits at <checkout>/tests, but also at <checkout>/.claude/worktrees/<n>/tests.
    src = next((p for a in here.parents if (p := a / rel).exists()), None)
    if src is None:
        try:
            import probixi.io.db as _db  # installed instead of alongside

            src = Path(_db.__file__)
        except ImportError:
            pytest.skip("probixi source neither alongside torchsx nor installed")
    m = re.search(r'_SCHEMA = """(.*?)"""', src.read_text(), re.S)
    assert m, "could not find _SCHEMA in probixi/io/db.py"

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    assert norm(m.group(1)) == norm(
        _SCHEMA
    ), "probixi's DuckDB schema changed; update _SCHEMA and the _duckdb.py mapping"


@pytest.fixture
def v2_path(db_path):
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("CREATE TABLE schema_version AS SELECT 2 AS version")
        conn.execute("""
            CREATE TABLE crystals AS
            SELECT frame_id || ':0' AS crystal_id, 0 AS lattice_index, *
            FROM frames WHERE indexed;
            ALTER TABLE reflections ADD COLUMN crystal_id VARCHAR;
            UPDATE reflections SET crystal_id = frame_id || ':0';
            INSERT INTO crystals
            SELECT * REPLACE (frame_id || ':1' AS crystal_id, 1 AS lattice_index,
                              7 AS n_peaks, 2.0 AS astar_x)
            FROM crystals WHERE filename = 'run_a.h5';
            INSERT INTO reflections
            SELECT * REPLACE (frame_id || ':1' AS crystal_id, -12.0 AS intensity)
            FROM reflections WHERE crystal_id IN (
                SELECT frame_id || ':0' FROM crystals WHERE lattice_index = 1
            );
            CREATE TABLE images AS
            SELECT frame_id, frame_index, filename, event, indexed, serial,
                   n_peaks, scale, scale_sigma, num_reflections FROM frames;
            DROP TABLE frames;
            ALTER TABLE images RENAME TO frames;
        """)
    return db_path


def test_v2_levels_and_values(v2_path):
    core = load_duckdb(v2_path)
    assert (core.n_images, core.n_crystals, core.n_observations) == (2, 3, 7)
    assert core.image_id.tolist() == [0, 0, 1]
    assert core.crystal_in_image.tolist() == [0, 1, 0]
    assert core.crystal_id.tolist() == [0, 0, 1, 1, 2, 2, 2]
    assert core.n_refl.tolist() == [2, 2, 3]
    assert core.n_peaks.tolist() == [90, 140]  # From frames, not peeled crystals.
    assert core.astar[:, 0].tolist() == pytest.approx([0.05, 0.2, 0.15])
    assert core.I.tolist() == [100, 101, -12, -12, 100, 101, 102]
    assert core.sigma.tolist() == [10, 11, 10, 11, 10, 11, 12]
    assert core.metadata["image"]["frame_index"] == [1, 0]
    ids = core.metadata["crystal"]["crystal_id"]
    assert len(set(ids)) == 3
    assert ids[0].rsplit(":", 1)[0] == ids[1].rsplit(":", 1)[0]


def test_v2_filter_concat_and_save(v2_path, tmp_path):
    from torchsx.merging.hkl.filter import filter_crystals

    core = load_duckdb(v2_path)
    selected = SFXStream(filter_crystals(core, torch.tensor([False, True, True])))
    assert selected.image_id.tolist() == [0, 1]
    assert selected._core.crystal_in_image.tolist() == [1, 0]
    assert (
        selected._core.metadata["crystal"]["crystal_id"]
        == core.metadata["crystal"]["crystal_id"][1:]
    )
    combined = SFXStream.concat([SFXStream(core), selected])
    assert combined.image_id.tolist() == [0, 0, 1, 2, 3]
    assert combined.crystal_id.tolist() == [0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4]
    out = tmp_path / "v2.sfx"
    combined.save(out)
    back = SFXStream.load(out, merge_friedel=False)
    assert torch.equal(back.image_id, combined.image_id)
    assert torch.equal(back.crystal_id, combined.crystal_id)
    for level in ("image", "crystal"):
        assert (
            back._core.metadata[level]["frame_id"]
            == combined._core.metadata[level]["frame_id"]
        )
    assert (
        back._core.metadata["crystal"]["crystal_id"]
        == combined._core.metadata["crystal"]["crystal_id"]
    )


@pytest.mark.parametrize("version", [[], [3], [2, 2]])
def test_unsupported_schema(db_path, version):
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("CREATE TABLE schema_version (version INTEGER)")
        for v in version:
            conn.execute("INSERT INTO schema_version VALUES (?)", [v])
    with pytest.raises(ValueError, match="unsupported Probixi schema version"):
        load_duckdb(db_path)


@pytest.mark.parametrize("embedded", [False, True])
def test_geometry_text_and_legacy_path(db_path, tmp_path, embedded, sample_geometry):
    text = """photon_energy = 11330
clen = 0.0938
res = 13333.3333
p0/min_fs = 0
p0/max_fs = 1023
p0/min_ss = 0
p0/max_ss = 511
p0/fs = +1x
p0/ss = +1y
p0/corner_x = -512
p0/corner_y = -256
"""
    geom = tmp_path / "detector.geom"
    geom.write_text(text)
    with duckdb.connect(str(db_path)) as conn:
        conn.execute(
            "UPDATE geometry SET geometry_file = ?", [text if embedded else str(geom)]
        )
    core = load_duckdb(db_path)
    assert core.geometry.panels[0].corner_x == -512
    assert core.geometry.photon_energy_eV == 11330
    assert load_duckdb(db_path, load_geom_file=False).geometry is None
    assert load_duckdb(db_path, geometry=sample_geometry).geometry is sample_geometry
