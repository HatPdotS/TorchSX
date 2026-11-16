"""``SFXStream.from_duckdb`` — probixi's relational run → stacked tensors.

The fixture builds a database against probixi's own DDL (copied verbatim from
``probixi/io/db.py``; ``test_schema_matches_probixi`` re-checks it against the
installed source when there is one), so the mapping under test is the real one.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import duckdb
import pytest
import torch

from torchsx.io._duckdb import (DEFAULT_ENRICH_ALPHA, load_duckdb, parse_enrich_alpha,
                                read_enrichment_gate, read_measurement)
from torchsx.io.sources import inspect_source, read_source
from torchsx.stream import SFXStream

# Verbatim from probixi.io.db._SCHEMA (v0.10.3, schema 1).
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

CREATE TABLE calibration (
    probixi_version        VARCHAR,
    schema_version         INTEGER,
    n_seed_frames          INTEGER,
    gain                   DOUBLE,
    read_var               DOUBLE,
    flux_variance          BOOLEAN,
    background_fluctuation DOUBLE,
    mf_threshold           DOUBLE,
    target_noise_peaks     DOUBLE,
    tail_ratio             DOUBLE,
    n_quiet_frames         INTEGER,
    kappa                  DOUBLE,
    prior_peak             DOUBLE,
    var_scale              DOUBLE,
    aperture               VARCHAR,
    clen                   DOUBLE,
    cell_a_A               DOUBLE,
    cell_b_A               DOUBLE,
    cell_c_A               DOUBLE,
    cell_alpha_deg         DOUBLE,
    cell_beta_deg          DOUBLE,
    cell_gamma_deg         DOUBLE,
    rank                   INTEGER
);

CREATE TABLE integration (
    r_peak            DOUBLE,
    r_gap             DOUBLE,
    r_bg              DOUBLE,
    adu_per_photon    DOUBLE,
    bg_annulus_pixels DOUBLE,
    aperture          VARCHAR,
    bg_model_var_scale DOUBLE,
    bg_model_var_dmin_A DOUBLE,
    bg_model_var_ref VARCHAR,
    bg_model_var_alpha DOUBLE,
    bg_model_var_n_ref BIGINT
);

CREATE TABLE frames (
    frame_id                 VARCHAR PRIMARY KEY,
    frame_index              BIGINT,
    filename                 VARCHAR,
    event                    BIGINT,
    indexed                  BOOLEAN,
    serial                   BIGINT,
    n_peaks                  INTEGER,
    peak_resolution_nm_inv   DOUBLE,
    scale                    DOUBLE,
    scale_sigma              DOUBLE,
    num_reflections          INTEGER,
    adu_per_photon           DOUBLE,
    event_name               VARCHAR,
    photon_energy_eV         DOUBLE,
    rank                     INTEGER
);

CREATE TABLE crystals (
    crystal_id               VARCHAR PRIMARY KEY,
    frame_id                 VARCHAR,
    lattice_index            INTEGER,
    n_indexed                INTEGER,
    rmsd                     DOUBLE,
    mosaicity_deg            DOUBLE,
    profile_radius_nm_inv    DOUBLE,
    enrichment               DOUBLE,
    n_bright                 INTEGER,
    enrich_p                 DOUBLE,
    diffraction_limit_nm_inv DOUBLE,
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
    crystal_id        VARCHAR,
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
    resolution_nm_inv DOUBLE,
    n_pixels          INTEGER,
    background_model  DOUBLE,
    background_model_var DOUBLE,
    n_pixels_total    INTEGER,
    n_masked          INTEGER,
    n_overlap         INTEGER
);

CREATE TABLE peaks (
    frame_id           VARCHAR,
    fs                 DOUBLE,
    ss                 DOUBLE,
    intensity          DOUBLE,
    resolution_nm_inv  DOUBLE,
    panel              VARCHAR,
    n_pixels           DOUBLE,
    photons            DOUBLE,
    background_photons DOUBLE,
    n_masked           INTEGER
);
"""

PHOTON_EV = 11330.0
CLEN = 0.0938
PANELS = ["p0", "p1"]
RADII = (3.0, 4.0, 7.36)
GAIN = 1.25


def _frame_id(filename: str, event: int) -> str:
    """probixi.io.db.frame_id."""
    return hashlib.sha1(f"{filename}//{event}".encode("utf-8")).hexdigest()[:16]


# (filename, event, indexed, n_peaks, peak_resolution_nm_inv, scale, scale_sigma)
# Inserted out of (filename, event) order, and the two indexed events are chosen
# so their frame_id hashes sort the *other* way round — otherwise
# test_crystal_order_is_filename_event_not_frame_id would pass either way.
_FRAMES = [
    ("run_b.h5", 8, True, 140, 3.5, 1.2, 0.05),
    ("run_a.h5", 1, True, 90, 2.8, 0.8, 0.04),
    ("run_a.h5", 3, False, 4, None, None, None),  # unindexed: no reflections
]
# (filename, event, lattice, n_refl, profile, limit, astar_x, n_indexed, rmsd, mosaicity, enrichment, n_bright, enrich_p)
_CRYSTALS = [
    ("run_b.h5", 8, 0, 3, 0.40, 3.0, 1.5, 60, 0.0004, 0.10, 8.0, 40, 1e-12),
    ("run_b.h5", 8, 1, 2, 0.30, 2.0, 2.0, 20, 0.0009, 0.20, 3.0, 9, 1e-3),
    ("run_a.h5", 1, 0, 2, 0.20, 2.5, 0.5, 30, 0.0006, 0.15, 5.0, 20, 1e-6),
]


def _disk(filename: str, lattice: int, j: int) -> tuple[int, int, int, int]:
    # (n_pixels, n_pixels_total, n_masked, n_overlap): one disk overlapped, one masked
    if (filename, lattice, j) == ("run_b.h5", 1, 1):
        return 6, 9, 0, 3
    if (filename, lattice, j) == ("run_a.h5", 0, 0):
        return 7, 9, 2, 0
    return 9, 9, 0, 0


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
    conn.execute("INSERT INTO integration VALUES (?, ?, ?, ?, 280.0, 'snr', NULL, NULL, NULL, NULL, NULL)",
                 [*RADII, GAIN])
    for serial, (fn, ev, indexed, n_peaks, pres, scale, ssig) in enumerate(_FRAMES):
        n_refl = sum(c[3] for c in _CRYSTALS if c[:2] == (fn, ev)) or None
        conn.execute(
            "INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [_frame_id(fn, ev), serial, fn, ev, indexed, serial, n_peaks, pres,
             scale, ssig, n_refl, GAIN, f"//{ev}", PHOTON_EV,  # the run energy per frame
             0 if fn == "run_a.h5" else 1],  # rank: run_a on 0, run_b on 1
        )
    for fn, ev, lat, n_refl, prof, limit, ax, n_idx, rmsd, mos, enr, nb, ep in _CRYSTALS:
        fid = _frame_id(fn, ev)
        cid = f"{fid}:{lat}"
        conn.execute(
            "INSERT INTO crystals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
            "79.0, 79.0, 38.0, 90.0, 90.0, 90.0, ?, 0.0, 0.0, 0.0, ?, 0.0, 0.0, 0.0, ?)",
            [cid, fid, lat, n_idx, rmsd, mos, prof, enr, nb, ep, limit, n_refl,
             ax, ax * 2, ax * 4],
        )
        for j in range(n_refl):
            npix, ntot, nmask, nover = _disk(fn, lat, j)
            conn.execute(
                "INSERT INTO reflections VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [cid, fid, j + 1, -j, 2 * j, 100.0 + j, 10.0 + j, 120.0 + j, 5.0,
                 10.0 * j, 20.0 * j, PANELS[j % 2], 2.0, npix, 40.0, 4.0,
                 ntot, nmask, nover],
            )
    conn.close()
    return path


def _ordered():
    """The crystals in the order load_duckdb must assign."""
    return sorted(_CRYSTALS, key=lambda c: (c[0], c[1], c[2]))


def test_levels_and_counts(db_path):
    core = load_duckdb(db_path, spacegroup="P 43 21 2")
    assert (core.n_images, core.n_crystals, core.n_observations) == (2, 3, 5)
    assert core.image_id.tolist() == [0, 1, 1]
    assert core.crystal_in_image.tolist() == [0, 0, 1]
    assert core.n_refl.tolist() == [1, 3, 1], "clipped disks are dropped, as the stream does"
    assert core.spacegroup == "P 43 21 2"
    full = load_duckdb(db_path, clipped=True)
    assert full.n_observations == 7
    assert full.n_refl.tolist() == [2, 3, 2]


def test_unindexed_frames_are_dropped(db_path):
    core = load_duckdb(db_path)
    filenames = core.metadata["image"]["filename"]
    events = core.metadata["image"]["event"]
    assert ("run_a.h5", 3) not in list(zip(filenames, events))


def test_crystal_order_is_filename_event_not_frame_id(db_path):
    """The dense index must follow (filename, event), never the id hash."""
    core = load_duckdb(db_path)
    expected = sorted({(f[0], f[1]) for f in _FRAMES if f[2]})
    got = list(zip(core.metadata["image"]["filename"], core.metadata["image"]["event"]))
    assert got == expected

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
    core = load_duckdb(db_path, clipped=True)
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
    assert counts.tolist() == [c[3] for c in _ordered()]


def test_reciprocal_vectors_converted_nm_inv_to_a_inv(db_path):
    core = load_duckdb(db_path)
    for i, c in enumerate(_ordered()):
        ax_nm_inv = c[6]
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
    for i, c in enumerate(_ordered()):
        prof_nm_inv, limit_nm_inv = c[4], c[5]
        assert core.profile_rad[i].item() == pytest.approx(prof_nm_inv / 10.0)
        # resolution is d_min in Å, from a reciprocal limit in nm^-1
        assert core.resolution[i].item() == pytest.approx(10.0 / limit_nm_inv)


def test_run_diagnostics_ride_along_as_columns(db_path):
    core = load_duckdb(db_path, clipped=True)
    crys = core.levels["crystal"]
    for i, c in enumerate(_ordered()):
        n_idx, rmsd, mos, enr, nb, ep = c[7:]
        assert crys["n_indexed"][i].item() == n_idx
        assert crys["rmsd"][i].item() == pytest.approx(rmsd)
        assert crys["mosaicity"][i].item() == pytest.approx(mos)
        assert crys["enrichment"][i].item() == pytest.approx(enr)
        assert crys["n_bright"][i].item() == nb
        assert crys["enrich_p"][i].item() == pytest.approx(ep, rel=1e-5)
    img = core.levels["image"]
    assert img["scale"].tolist() == pytest.approx([0.8, 1.2])
    assert img["scale_sigma"].tolist() == pytest.approx([0.04, 0.05])
    assert img["peak_resolution"].tolist() == pytest.approx([10 / 2.8, 10 / 3.5])
    obs = core.levels["obs"]
    assert obs["n_pixels"].tolist() == [7, 9, 9, 9, 9, 9, 6]
    assert obs["background_model"].tolist() == [40.0] * 7
    assert obs["background_model_var"].tolist() == [4.0] * 7
    # the clip counts only select the disks; they are not loaded
    assert {"n_pixels_total", "n_masked", "n_overlap"}.isdisjoint(obs.columns)
    assert load_duckdb(db_path).levels["obs"]["n_pixels"].tolist() == [9] * 5


def test_measurement_spec_from_integration_table(db_path):
    spec = read_measurement(db_path)
    assert spec.producer == "probixi"
    assert spec.radii == RADII
    assert spec.gain_declarations == (("global", "adu_per_photon", GAIN),)
    assert spec.nominal_n_peak and spec.nominal_n_bg
    assert spec.unresolved == ()
    assert inspect_source(db_path) == spec
    stream = SFXStream.from_duckdb(db_path, spacegroup="P 43 21 2")
    assert stream.measurement == spec


def test_read_source_dispatches_duckdb(db_path):
    stream = read_source(db_path, spacegroup="P 43 21 2", clipped=True)
    assert isinstance(stream, SFXStream) and stream.n_observations == 7
    assert SFXStream.read(db_path, spacegroup="P 43 21 2").n_observations == 5


def test_panel_names_resolve_to_indices(db_path):
    core = load_duckdb(db_path)
    assert core.metadata["crystal"]["panel_names"] == PANELS
    assert core.panel_id.dtype == torch.int16
    assert int(core.panel_id.min()) >= 0, "every panel string matched a panels row"
    assert set(core.panel_id.tolist()) <= set(range(len(PANELS)))


def test_run_level_beam_metadata_is_constant(db_path):
    with duckdb.connect(str(db_path)) as conn:  # a per-frame energy, as SACLA files give
        conn.execute("UPDATE frames SET photon_energy_eV = ? + event", [PHOTON_EV])
    core = load_duckdb(db_path)
    # the per-frame photon energy of each image (events 1 and 8), not the run's
    assert core.beam_energy.tolist() == [PHOTON_EV + 1, PHOTON_EV + 8]
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("UPDATE frames SET photon_energy_eV = NULL")
    core = load_duckdb(db_path)
    assert torch.allclose(core.beam_energy, torch.full_like(core.beam_energy, PHOTON_EV))
    assert torch.allclose(core.camera_length, torch.full_like(core.camera_length, CLEN))
    # absent from probixi's schema -> constant, which the scale encoder ignores
    assert torch.count_nonzero(core.det_shift_x) == 0
    assert torch.count_nonzero(core.det_shift_y) == 0


def test_no_columns_share_storage(db_path):
    """Two columns aliasing one storage silently breaks .sfx saving."""
    core = load_duckdb(db_path, clipped=True)
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
    stream = SFXStream.from_duckdb(db_path, spacegroup="P 43 21 2", merge_friedel=False,
                                   clipped=True)
    out = tmp_path / "run.sfx"
    stream.save(out)
    back = SFXStream.load(out, merge_friedel=False)

    assert back.n_crystals == stream.n_crystals
    assert back.n_observations == stream.n_observations
    a, b = stream._core, back._core
    for col in ("hkl", "crystal_id", "panel_id", "image_id"):
        assert torch.equal(getattr(b, col), getattr(a, col)), col
    for col in (
        "I", "sigma", "astar", "bstar", "cstar", "cell", "profile_rad", "resolution",
        "beam_energy", "n_pixels", "enrichment", "scale",
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
    assert back.measurement == stream.measurement


def test_frame_id_is_carried_verbatim(db_path):
    """The write-back key must survive as data, not be re-derived from filenames."""
    core = load_duckdb(db_path)
    got = core.metadata["crystal"]["frame_id"]
    assert got == [_frame_id(c[0], c[1]) for c in _ordered()]
    ids = core.metadata["crystal"]["crystal_id"]
    assert len(set(ids)) == 3
    assert ids[1].rsplit(":", 1)[0] == ids[2].rsplit(":", 1)[0]


def test_feeds_the_merging_stack(db_path):
    """The point of the loader: merging accepts it without a stream file."""
    from torchsx.merging import MergeConfig, prepare_observations

    stream = SFXStream.from_duckdb(db_path, spacegroup="P 43 21 2", clipped=True)
    nominal = prepare_observations(stream, MergeConfig(gain="trust_input"))
    recorded = prepare_observations(stream, MergeConfig(gain="trust_input", disk_size="recorded"))
    assert nominal.obs_info.n_pix.unique().numel() == 1
    assert sorted(recorded.obs_info.n_pix.tolist()) == [6, 7, 9, 9, 9, 9, 9]
    for prep in (nominal, recorded):  # background 5 ADU/px over n_pix at GAIN ADU/photon
        torch.testing.assert_close(prep.bg, 5.0 / GAIN * prep.obs_info.n_pix)
    # the model background replaces the ring's estimate but not the measured count
    model = prepare_observations(
        stream, MergeConfig(gain="trust_input", disk_size="recorded", background="model"))
    torch.testing.assert_close(model.bg, torch.full_like(model.bg, 40.0 / GAIN))
    torch.testing.assert_close(model.I_obs + model.bg, recorded.I_obs + recorded.bg)

    from torchsx.merging.hkl import compute_hkl_index, merge_intensities

    idx = compute_hkl_index(stream.hkl)
    I_m, sig_m, mult = merge_intensities(
        idx.hkl_id, stream.I, stream.sigma, idx.n_unique
    )
    assert I_m.numel() == idx.n_unique
    assert int(mult.sum()) == stream.n_observations
    assert torch.isfinite(I_m).all() and torch.isfinite(sig_m).all()


def test_crystal_feature_table_shape_and_dead_columns(db_path):
    """probixi supplies every encoder feature but the two structurally constant ones."""
    with duckdb.connect(str(db_path)) as conn:  # a per-frame energy, as SACLA files give
        conn.execute("UPDATE frames SET photon_energy_eV = ? + event", [PHOTON_EV])
    from torchsx.merging.observations import (
        CRYSTAL_FEAT_COLUMNS,
        PROBIXI_CRYSTAL_FEATURES,
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
    # These two cannot vary: absent from probixi's schema.
    assert {"det_shift_x", "det_shift_y"} <= dead
    # These must survive the mapping as live, varying features.
    assert dead.isdisjoint({"log_n_refl", "beam_energy", *PROBIXI_CRYSTAL_FEATURES})
    col = CRYSTAL_FEAT_COLUMNS.index
    torch.testing.assert_close(tbl[:, col("log_scale")], torch.tensor([0.8, 1.2, 1.2]).log())
    torch.testing.assert_close(tbl[:, col("log_enrich_p")], torch.tensor([1e-6, 1e-12, 1e-3]).log10())
    torch.testing.assert_close(tbl[:, col("log_n_indexed")], torch.log1p(torch.tensor([30.0, 60.0, 20.0])))
    torch.testing.assert_close(tbl[:, col("peak_resolution")], torch.tensor([10 / 2.8, 10 / 3.5, 10 / 3.5]))


def test_probixi_features_reach_the_encoder(db_path):
    from torchsx.merging import MergeConfig, prepare_observations
    from torchsx.merging.observations import (
        CRYSTAL_FEAT_COLUMNS, DEFAULT_CRYSTAL_FEATURES, PROBIXI_CRYSTAL_FEATURES,
    )
    from torchsx.merging.scale import AmortizedScale
    from torchsx.nn.obs_features import DISK_FEATURE, SELF_FEATURES

    stream = SFXStream.from_duckdb(db_path, spacegroup="P 43 21 2", clipped=True)
    prep = prepare_observations(stream, MergeConfig(gain="trust_input", disk_size="recorded"))
    scale = AmortizedScale.from_prep(
        prep, d_model=16,
        self_features=SELF_FEATURES + (DISK_FEATURE,),
        crystal_features=DEFAULT_CRYSTAL_FEATURES + PROBIXI_CRYSTAL_FEATURES,
    )
    feats = scale.features
    assert feats.n_crystal == len(CRYSTAL_FEAT_COLUMNS)
    assert not feats._const[feats._stat_names.index(DISK_FEATURE)], "recorded counts vary"
    x_self, _, x_crys = feats.build(prep.obs_info)
    assert x_crys.shape == (3, len(CRYSTAL_FEAT_COLUMNS))
    assert x_self[:, feats.self_features.index(DISK_FEATURE)].std() > 0
    # the default tiers are unchanged: no disk column, the twelve stream columns
    plain = AmortizedScale.from_prep(prep, d_model=16).features
    assert DISK_FEATURE not in plain.self_features
    assert plain.crystal_features == DEFAULT_CRYSTAL_FEATURES


def test_cli_switches_on_what_the_database_records(db_path, tmp_path):
    from torchsx.cli import _pipeline as P
    from torchsx.cli.train import _build_parser
    from torchsx.merging.observations import PROBIXI_CRYSTAL_FEATURES
    from torchsx.nn.obs_features import DISK_FEATURE

    args = _build_parser().parse_args([
        str(db_path), "--spacegroup", "P 43 21 2", "--dmin", "1.0", "--gain", "trust_input",
        "--device", "cpu", "-o", str(tmp_path / "run")])
    bundle = P.load_prep(args, verbose=False)
    assert bundle.recipe["disk_size"] == "recorded"
    assert bundle.recipe["background"] == "model"
    assert bundle.recipe["per_image_k0"] is False, "every frame at the run's energy"
    assert bundle.recipe["measurement"]["source"]["producer"] == "probixi"
    prep = bundle.prep
    assert prep.n_obs_total == 7
    rows = torch.arange(prep.n_obs_total)
    oi = prep.obs_info.gather(rows) if hasattr(prep.obs_info, "gather") else prep.batch(rows, "cpu")
    torch.testing.assert_close(oi.bg, torch.full_like(oi.bg, 40.0 / GAIN))
    assert sorted(oi.n_pix.tolist()) == [6, 7, 9, 9, 9, 9, 9]
    # Var[bg] / bg in photons from the recorded columns: 4 / (GAIN * 40)
    assert bundle.bg_var_factor == pytest.approx(4.0 / (GAIN * 40.0))
    host, params, scale, obj, S = P.build_stack(bundle, args, verbose=False)
    assert DISK_FEATURE in scale.features.self_features
    assert set(PROBIXI_CRYSTAL_FEATURES) <= set(scale.features.crystal_features)
    assert obj.likelihood.bg_var_factor == pytest.approx(4.0 / (GAIN * 40.0))
    assert obj.likelihood.bg_mean_offset == 0.0
    # a merge rebuilds the tiers its checkpoint recorded, not the defaults
    from tests.test_cli_pipeline import _trainer_from_stack
    tr = _trainer_from_stack(host, params, scale, obj, S)
    try:
        ck = torch.load(tr.save_best(tmp_path / "model.pt"), weights_only=False)
    finally:
        tr.close()
    tiers = {k: list(getattr(scale.features, k))
             for k in ("self_features", "context_features", "crystal_features")}
    assert {k: ck["scale"][k] for k in tiers} == tiers
    _, _, again, _, _ = P.build_stack(bundle, args, ckpt=ck, verbose=False)
    assert {k: list(getattr(again.features, k)) for k in tiers} == tiers
    prep.close()


def test_sentinels_become_missing_values(db_path):
    from torchsx.merging.observations import CRYSTAL_FEAT_COLUMNS, _crystal_feature_table

    with duckdb.connect(str(db_path)) as conn:
        conn.execute("UPDATE frames SET peak_resolution_nm_inv = NULL WHERE event = 1")
    core = load_duckdb(db_path, spacegroup="P 43 21 2", clipped=True)
    assert core.levels["image"]["peak_resolution"].tolist() == pytest.approx(
        [float("nan"), 10 / 3.5], nan_ok=True)
    assert core.resolution.tolist() == pytest.approx([10 / 2.5, 10 / 3.0, 10 / 2.0]), "the cut keeps its sentinel"
    tbl = _crystal_feature_table(core, core.n_crystals)
    col = CRYSTAL_FEAT_COLUMNS.index("peak_resolution")
    assert tbl[:, col].tolist() == pytest.approx([10 / 3.5] * 3)
    assert torch.isfinite(tbl).all()


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
    assert stream.n_crystals == 3
    assert stream._core.friedel_merged is True

    unmerged = SFXStream.from_duckdb(
        db_path, spacegroup="P 43 21 2", merge_friedel=False
    )
    assert unmerged._core.friedel_merged is False


def test_filter_concat_and_save(db_path, tmp_path):
    from torchsx.merging.hkl.filter import filter_crystals

    core = load_duckdb(db_path)
    selected = SFXStream(filter_crystals(core, torch.tensor([False, True, True])))
    assert selected.image_id.tolist() == [0, 0]
    assert selected._core.crystal_in_image.tolist() == [0, 1]
    assert (
        selected._core.metadata["crystal"]["crystal_id"]
        == core.metadata["crystal"]["crystal_id"][1:]
    )
    combined = SFXStream.concat([SFXStream(core), selected])
    assert combined.image_id.tolist() == [0, 1, 1, 2, 2]
    assert combined.crystal_id.tolist() == [0, 1, 1, 1, 2, 3, 3, 3, 4]
    out = tmp_path / "v.sfx"
    combined.save(out)
    back = SFXStream.load(out, merge_friedel=False)
    assert torch.equal(back.image_id, combined.image_id)
    assert torch.equal(back.crystal_id, combined.crystal_id)
    for level in ("image", "crystal"):
        assert (
            back._core.metadata[level]["frame_id"]
            == combined._core.metadata[level]["frame_id"]
        )


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


def test_calibration_provenance_and_schema_gate(db_path):
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("INSERT INTO calibration VALUES ('0.10.0', 1, 100, 1.5, 0.0, false, 1.0, 5.9, "
                     "6.0, 10.2, 0, 2.3, 0.03, 0.96, 'snr', 0.1, 79.0, 79.0, 38.0, 90.0, 90.0, 90.0, 0)")
    spec = read_measurement(db_path)
    assert spec.producer_version == "0.10.0"
    assert {("probixi_version", "0.10.0"), ("schema_version", "1"), ("flux_variance", "False"),
            ("mf_threshold", "5.9"), ("tail_ratio", "10.2")} <= set(spec.provenance)
    assert spec.gain_declarations == (("calibration", "adu_per_photon", 1.5),)
    assert load_duckdb(db_path, spacegroup="P 43 21 2").n_observations == 5
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("UPDATE calibration SET schema_version = 99")
    with pytest.raises(ValueError, match="newer"):
        read_measurement(db_path)
    with pytest.raises(ValueError, match="newer"):
        load_duckdb(db_path)


def test_varying_frame_energies_switch_on_per_image_k0(db_path, tmp_path):
    from torchsx.cli import _pipeline as P
    from torchsx.cli.train import _build_parser
    from torchsx.merging import MergeConfig, prepare_observations

    base = [str(db_path), "--spacegroup", "P 43 21 2", "--dmin", "1.0", "--gain", "trust_input", "--device", "cpu"]
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("UPDATE frames SET photon_energy_eV = 10000.0")
    same = P.load_prep(_build_parser().parse_args(base + ["-o", str(tmp_path / "a")]), verbose=False)
    assert same.recipe["per_image_k0"] is False
    assert ("photon_energy", "per-frame") not in read_measurement(db_path).provenance
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("UPDATE frames SET photon_energy_eV = 10500.0 WHERE event = 8")
    varied = P.load_prep(_build_parser().parse_args(base + ["-o", str(tmp_path / "b")]), verbose=False)
    assert varied.recipe["per_image_k0"] is True
    assert ("photon_energy", "per-frame") in read_measurement(db_path).provenance
    # the store holds the offsets at each image's own wavelength
    stream = SFXStream.from_duckdb(db_path, spacegroup="P 43 21 2", clipped=True)
    median = prepare_observations(stream, MergeConfig(gain="trust_input")).obs_info.ewald_offset
    own = prepare_observations(stream, MergeConfig(gain="trust_input", per_image_k0=True)).obs_info.ewald_offset
    stored = varied.prep.batch(torch.arange(varied.prep.n_obs_total), "cpu").ewald_offset
    torch.testing.assert_close(stored.sort().values, own.sort().values)
    assert not torch.allclose(stored.sort().values, median.sort().values)
    same.prep.close()
    varied.prep.close()


def test_dmin_defaults_to_the_crystals_diffraction_limits(db_path, tmp_path):
    from torchsx.cli import _pipeline as P
    from torchsx.cli.train import _build_parser

    with duckdb.connect(str(db_path)) as conn:
        conn.execute("UPDATE crystals SET diffraction_limit_nm_inv = 10.0 / 1.5")  # 1.5 A everywhere
    spec = read_measurement(db_path)
    assert dict(spec.provenance)["diffraction_limit_p10_A"] == "1.500"
    args = _build_parser().parse_args([str(db_path), "--spacegroup", "P 43 21 2", "--gain", "trust_input",
                                       "--device", "cpu", "-o", str(tmp_path / "run")])
    bundle = P.load_prep(args, verbose=False)
    assert bundle.recipe["d_min"] == pytest.approx(1.5)


def test_cache_survives_a_provenance_only_change(db_path, tmp_path, monkeypatch):
    from dataclasses import replace

    import torchsx.io._duckdb as D
    from torchsx.cli import _pipeline as P
    from torchsx.cli.train import _build_parser

    base = [str(db_path), "--spacegroup", "P 43 21 2", "--dmin", "1.0", "--gain", "trust_input",
            "--device", "cpu", "--cache", str(tmp_path / "store")]
    P.load_prep(_build_parser().parse_args(base + ["-o", str(tmp_path / "a")]), verbose=False)
    original = D._measurement
    monkeypatch.setattr(D, "_measurement", lambda conn, path: replace(
        original(conn, path), provenance=original(conn, path).provenance + (("new_entry", "1"),)))
    bundle = P.load_prep(_build_parser().parse_args(base + ["-o", str(tmp_path / "b")]), verbose=False)
    assert bundle.prep.n_obs_total > 0


_CAL = ("INSERT INTO calibration VALUES ('0.10.0', 1, 100, ?, 0.0, false, 1.0, 5.9, 6.0, 10.2, 0, "
        "2.3, 0.03, 0.96, 'snr', 0.1, 79.0, 79.0, 38.0, 90.0, 90.0, 90.0, ?)")


def test_per_rank_calibration_gain(db_path):
    """A multi-GPU run writes one calibration row per rank; the gain is their
    frame-weighted mean (rank 1 indexed one frame, rank 0 one), not rank 0's."""
    with duckdb.connect(str(db_path)) as conn:
        conn.execute(_CAL, [1.50, 0])
        conn.execute(_CAL, [1.52, 1])
    spec = read_measurement(db_path)
    (src, key, gain), = spec.gain_declarations
    assert (src, key) == ("calibration", "adu_per_photon")
    assert gain == pytest.approx(1.51)
    prov = dict(spec.provenance)
    assert prov["calibration_ranks"] == "2"
    assert float(prov["gain_rank_spread"]) == pytest.approx(0.02 / 1.51, rel=1e-3)
    # a spread beyond tolerance warns but still loads
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("UPDATE calibration SET gain = 2.0 WHERE rank = 1")
    with pytest.warns(UserWarning, match="per-rank calibrated gains"):
        spec = read_measurement(db_path)
    assert spec.gain_declarations[0][2] == pytest.approx(1.75)
    # every rank's schema is checked, not only the first row's
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("UPDATE calibration SET schema_version = 99 WHERE rank = 1")
    with pytest.raises(ValueError, match="newer"):
        read_measurement(db_path)


def test_model_background_sigma(db_path):
    """With the model background the intensity carries the model's own sigma,
    sigma^2 = V_model (1 + n_bg / n) + g max(I_model, 0) in ADU^2, and the count
    I + n b keeps the disk the source summed even with nominal disks."""
    from torchsx.merging import MergeConfig, prepare_observations

    assert ("bg_annulus_pixels", "280") in read_measurement(db_path).provenance
    stream = SFXStream.from_duckdb(db_path, spacegroup="P 43 21 2", clipped=True)
    core = stream._core
    n = core.levels["obs"]["n_pixels"].float()
    I_model = core.I + n * core.background - 40.0  # ADU
    want = (4.0 * (1 + 280.0 / n) + GAIN * I_model.clamp(min=0)).sqrt() / GAIN  # photons
    annulus = prepare_observations(stream, MergeConfig(gain="trust_input", disk_size="recorded"))
    for disks in ("recorded", "nominal"):
        model = prepare_observations(
            stream, MergeConfig(gain="trust_input", disk_size=disks, background="model"))
        # same observation order: obs_info.sigma follows I_obs
        got = dict(zip((model.I_obs * GAIN).round(decimals=3).tolist(), model.obs_info.sigma.tolist()))
        for i, s in zip((I_model).round(decimals=3).tolist(), want.tolist()):
            assert got[i] == pytest.approx(s, rel=1e-5)
    torch.testing.assert_close(annulus.obs_info.sigma.sort().values,
                               (core.sigma / GAIN).sort().values)


def test_cli_background_override(db_path, tmp_path):
    """--background annulus restores the ring estimate on a probixi database;
    the store's model sigma matches the in-memory preparation."""
    from torchsx.cli import _pipeline as P
    from torchsx.cli.train import _build_parser
    from torchsx.merging import MergeConfig, prepare_observations

    base = [str(db_path), "--spacegroup", "P 43 21 2", "--dmin", "1.0", "--gain", "trust_input",
            "--device", "cpu"]
    ann = P.load_prep(_build_parser().parse_args(base + ["--background", "annulus",
                                                         "-o", str(tmp_path / "a")]), verbose=False)
    mod = P.load_prep(_build_parser().parse_args(base + ["-o", str(tmp_path / "m")]), verbose=False)
    assert ann.recipe["background"] == "annulus" and ann.recipe["disk_size"] == "recorded"
    assert mod.recipe["background"] == "model"
    rows = torch.arange(mod.prep.n_obs_total)
    a, m = ann.prep.batch(rows, "cpu"), mod.prep.batch(rows, "cpu")
    torch.testing.assert_close(a.bg, 5.0 / GAIN * a.n_pix)
    torch.testing.assert_close(m.bg, torch.full_like(m.bg, 40.0 / GAIN))
    # recorded disks keep the clipped ones, in the store as in memory
    stream = SFXStream.from_duckdb(db_path, spacegroup="P 43 21 2", clipped=True)
    ref = prepare_observations(stream, MergeConfig(gain="trust_input", disk_size="recorded",
                                                   background="model"))
    torch.testing.assert_close(m.sigma.sort().values, ref.obs_info.sigma.sort().values)
    assert not torch.allclose(a.sigma.sort().values, m.sigma.sort().values)



def test_gain_calibration_sees_the_recorded_disks(db_path, tmp_path, monkeypatch):
    """The gain calibration whitens each observation with its own disk: in
    memory and from the prepared store it receives the recorded pixel counts,
    and none when the disks are taken as nominal."""
    from torchsx.cli import _pipeline as P
    from torchsx.cli.train import _build_parser
    from torchsx.merging import MergeConfig, prepare_observations
    from torchsx.merging.hkl.noise import PoissonNoiseModel

    seen = []

    def calibrate(self, I, background, ewald, **kw):
        seen.append(kw.get("n_pix"))
        self.gain_calibration_info = {"valid": True, "method": "fixture"}
        return GAIN, 0.0, 0.0

    monkeypatch.setattr(PoissonNoiseModel, "calibrate_sigma2_of_bg", calibrate)
    stream = SFXStream.from_duckdb(db_path, spacegroup="P 43 21 2", clipped=True)
    prepare_observations(stream, MergeConfig(gain="auto", disk_size="recorded"))
    prepare_observations(stream, MergeConfig(gain="auto"))
    assert sorted(seen[0].tolist()) == [6, 7, 9, 9, 9, 9, 9] and seen[1] is None
    seen.clear()
    P.load_prep(_build_parser().parse_args(
        [str(db_path), "--spacegroup", "P 43 21 2", "--dmin", "1.0", "--gain", "auto",
         "--cache", str(tmp_path / "c"), "-o", str(tmp_path / "o")]), verbose=False)
    assert seen and seen[0] is not None and set(seen[0].tolist()) <= {6.0, 7.0, 9.0}


def test_fit_model_sigma_scale_recovers_a_known_factor():
    """Pure background past the limit: the robust fit returns the factor by which
    the recorded variance overstates the scatter, and ignores a few signal rows."""
    from torchsx.merging.observations import fit_model_sigma_scale, sigma_scale_shells

    from torchsx.merging.observations import robust_sd

    g = torch.Generator().manual_seed(0)
    n, k_true = 200_000, 0.4
    # background-dominated (the Poisson term max(I, 0) is ~1 % of the variance): k itself
    var_bg = 5e3 + 1e4 * torch.rand(n, generator=g)            # photons^2, as recorded
    I = torch.randn(n, generator=g) * (k_true * var_bg).sqrt()   # true scatter: k var_bg
    I[:500] += 1e6                                               # leftover signal / outliers
    k, m = fit_model_sigma_scale(I, var_bg)
    assert m == n
    assert k == pytest.approx(k_true, rel=0.03)
    shells = sigma_scale_shells(I, var_bg, torch.linspace(2.0, 4.0, n), k)
    assert len(shells) == 8
    assert all(s1 == pytest.approx(1.0, abs=0.05) and s0 == pytest.approx(k_true ** 0.5, abs=0.05)
               for _, _, _, s0, s1 in shells)
    # few photons: k is what makes I / sqrt(k var + max(I, 0)) unit-spread, below k_true
    small = var_bg / 100.0
    I_small = torch.randn(n, generator=g) * (k_true * small).sqrt()
    k_small, _ = fit_model_sigma_scale(I_small, small)
    assert robust_sd(I_small / (k_small * small + I_small.clamp(min=0)).sqrt()) == pytest.approx(1.0, abs=0.01)
    assert 0.25 < k_small < k_true
    assert fit_model_sigma_scale(I[:10], var_bg[:10]) == (None, 10)


def test_model_sigma_scale_policies(db_path, tmp_path):
    """A fixed k scales the background term only, identically in memory and in
    the store; off is k = 1; the fit is recorded in the recipe."""
    from torchsx.cli import _pipeline as P
    from torchsx.cli.train import _build_parser
    from torchsx.merging import MergeConfig, prepare_observations

    stream = SFXStream.from_duckdb(db_path, spacegroup="P 43 21 2", clipped=True)
    core = stream._core
    n = core.levels["obs"]["n_pixels"].float()
    I_model = core.I + n * core.background - 40.0
    for k in (1.0, 0.25):
        want = (k * 4.0 * (1 + 280.0 / n) + GAIN * I_model.clamp(min=0)).sqrt() / GAIN
        prep = prepare_observations(stream, MergeConfig(gain="trust_input", disk_size="recorded",
                                                        background="model", model_sigma_scale=k))
        torch.testing.assert_close(prep.obs_info.sigma.sort().values, want.sort().values)
    off = prepare_observations(stream, MergeConfig(gain="trust_input", disk_size="recorded",
                                                   background="model", model_sigma_scale="off"))
    auto = prepare_observations(stream, MergeConfig(gain="trust_input", disk_size="recorded",
                                                    background="model"))
    # seven rows, none past its crystal's limit: auto leaves k at 1
    torch.testing.assert_close(auto.obs_info.sigma, off.obs_info.sigma)

    base = [str(db_path), "--spacegroup", "P 43 21 2", "--dmin", "1.0", "--gain", "trust_input",
            "--device", "cpu"]
    fixed = P.load_prep(_build_parser().parse_args(base + ["--model-sigma-scale", "0.25",
                                                           "-o", str(tmp_path / "k")]), verbose=False)
    assert fixed.recipe["model_sigma_scale"] == 0.25
    assert fixed.recipe["model_sigma_scale_fit"] == {"policy": "fixed", "k": 0.25}
    rows = torch.arange(fixed.prep.n_obs_total)
    ref = prepare_observations(stream, MergeConfig(gain="trust_input", disk_size="recorded",
                                                   background="model", model_sigma_scale=0.25))
    torch.testing.assert_close(fixed.prep.batch(rows, "cpu").sigma.sort().values,
                               ref.obs_info.sigma.sort().values)
    auto = P.load_prep(_build_parser().parse_args(base + ["-o", str(tmp_path / "a")]), verbose=False)
    assert auto.recipe["model_sigma_scale"] == "auto"
    assert auto.recipe["model_sigma_scale_fit"]["fitted"] is False
    with pytest.raises(SystemExit, match="model-sigma-scale"):
        P.load_prep(_build_parser().parse_args(base + ["--model-sigma-scale", "-1",
                                                       "-o", str(tmp_path / "bad")]), verbose=False)


def test_model_sigma_scale_fit_in_the_store(db_path, tmp_path, monkeypatch):
    """auto in the store: the reference rows are those past their crystal's
    limit, and the fitted k reaches every model sigma."""
    from torchsx.cli import _pipeline as P
    from torchsx.cli.train import _build_parser
    from torchsx.merging import observations as O

    # every crystal's limit just above the data's d, so every row is a reference row
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("UPDATE crystals SET diffraction_limit_nm_inv = 0.02")
    monkeypatch.setattr(O, "SIGMA_SCALE_MIN_OBS", 3)
    base = [str(db_path), "--spacegroup", "P 43 21 2", "--dmin", "1.0", "--gain", "trust_input",
            "--device", "cpu"]
    auto = P.load_prep(_build_parser().parse_args(base + ["-o", str(tmp_path / "a")]), verbose=False)
    fit = auto.recipe["model_sigma_scale_fit"]
    assert fit["policy"] == "auto" and fit["fitted"] is True and fit["n_ref"] == 7
    k = fit["k"]
    fixed = P.load_prep(_build_parser().parse_args(base + ["--model-sigma-scale", str(k),
                                                           "-o", str(tmp_path / "k")]), verbose=False)
    rows = torch.arange(auto.prep.n_obs_total)
    torch.testing.assert_close(auto.prep.batch(rows, "cpu").sigma, fixed.prep.batch(rows, "cpu").sigma)


def test_model_sigma_scale_defers_to_probixi(db_path, tmp_path, monkeypatch):
    """A database whose variance probixi already rescaled (0.10.3) keeps k = 1
    under auto, in memory and in the store; a fixed k still applies."""
    from torchsx.cli import _pipeline as P
    from torchsx.cli.train import _build_parser
    from torchsx.merging import MergeConfig, prepare_observations
    from torchsx.merging import observations as O

    with duckdb.connect(str(db_path)) as conn:
        conn.execute("UPDATE integration SET bg_model_var_scale = 0.5")
        conn.execute("UPDATE crystals SET diffraction_limit_nm_inv = 0.02")  # all rows past
    monkeypatch.setattr(O, "SIGMA_SCALE_MIN_OBS", 3)
    assert ("bg_model_var_scale", "0.5") in read_measurement(db_path).provenance
    stream = SFXStream.from_duckdb(db_path, spacegroup="P 43 21 2", clipped=True)
    cfg = dict(gain="trust_input", disk_size="recorded", background="model")
    auto = prepare_observations(stream, MergeConfig(**cfg))
    off = prepare_observations(stream, MergeConfig(**cfg, model_sigma_scale="off"))
    torch.testing.assert_close(auto.obs_info.sigma, off.obs_info.sigma)
    base = [str(db_path), "--spacegroup", "P 43 21 2", "--dmin", "1.0", "--gain", "trust_input",
            "--device", "cpu"]
    bundle = P.load_prep(_build_parser().parse_args(base + ["-o", str(tmp_path / "a")]), verbose=False)
    assert bundle.recipe["model_sigma_scale_fit"] == {"policy": "source", "k": 1.0,
                                                      "source_scales": [0.5]}
    fixed = P.load_prep(_build_parser().parse_args(base + ["--model-sigma-scale", "0.25",
                                                           "-o", str(tmp_path / "k")]), verbose=False)
    assert fixed.recipe["model_sigma_scale_fit"] == {"policy": "fixed", "k": 0.25}


def test_crystals_where_keeps_only_the_accepted_crystals(db_path):
    core = load_duckdb(db_path, clipped=True, crystals_where="enrich_p <= 1e-5")
    assert (core.n_images, core.n_crystals, core.n_observations) == (2, 2, 5)
    assert core.n_refl.tolist() == [2, 3]
    one = load_duckdb(db_path, clipped=True, crystals_where="num_reflections = 3")
    assert (one.n_images, one.n_crystals, one.n_observations) == (1, 1, 3)
    stream = read_source(db_path, spacegroup="P 43 21 2", clipped=True,
                         crystals_where="enrich_p <= 1e-5")
    assert stream._core.n_crystals == 2
    with pytest.raises(ValueError, match="nothing to load"):
        load_duckdb(db_path, crystals_where="enrich_p < 0")
    with pytest.raises(ValueError, match="one SQL predicate"):
        load_duckdb(db_path, crystals_where="true; DROP TABLE crystals")


def _chance_lattice(db_path, enrich_p=0.5):
    """Give the second lattice of run_b a chance-level enrichment."""
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("UPDATE crystals SET enrich_p = ? WHERE lattice_index = 1", [enrich_p])


def test_enrichment_gate_drops_chance_lattices_by_default(db_path):
    _chance_lattice(db_path)
    gate = read_enrichment_gate(db_path)
    assert gate == {"predicate": f"enrich_p <= {DEFAULT_ENRICH_ALPHA!r}", "alpha": DEFAULT_ENRICH_ALPHA,
                    "origin": "torchsx default", "n_total": 3, "n_kept": 2}
    core = load_duckdb(db_path, clipped=True)
    assert (core.n_images, core.n_crystals, core.n_observations) == (2, 2, 5)
    stream = SFXStream.from_duckdb(db_path, spacegroup="P 43 21 2", clipped=True)
    assert stream._core.n_crystals == 2
    for off in (None, "off"):
        assert load_duckdb(db_path, clipped=True, enrich_alpha=off).n_crystals == 3
        assert read_enrichment_gate(db_path, off)["predicate"] is None
    assert read_source(db_path, spacegroup="P 43 21 2", clipped=True, enrich_alpha="off")._core.n_crystals == 3
    # a number replaces the default; crystals_where applies on top of the gate
    assert load_duckdb(db_path, clipped=True, enrich_alpha=1e-9).n_crystals == 1
    assert load_duckdb(db_path, clipped=True, enrich_alpha=0.9).n_crystals == 3
    assert load_duckdb(db_path, clipped=True, crystals_where="num_reflections = 2").n_crystals == 1
    with pytest.raises(ValueError, match="enrichment gate"):
        load_duckdb(db_path, enrich_alpha=1e-20)


def test_enrichment_gate_takes_the_alpha_the_database_records(db_path):
    _chance_lattice(db_path, 0.01)
    assert load_duckdb(db_path, clipped=True).n_crystals == 2          # 0.01 > the default 1e-3
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("UPDATE integration SET bg_model_var_alpha = 0.05")
    gate = read_enrichment_gate(db_path)
    assert (gate["alpha"], gate["origin"], gate["n_kept"]) == (
        0.05, "database, integration.bg_model_var_alpha", 3)
    assert load_duckdb(db_path, clipped=True).n_crystals == 3
    assert load_duckdb(db_path, clipped=True, enrich_alpha=1e-3).n_crystals == 2


def test_enrichment_gate_needs_recorded_enrichment(db_path):
    """A database without enrich_p values loads every lattice."""
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("UPDATE crystals SET enrich_p = NULL")
    gate = read_enrichment_gate(db_path)
    assert (gate["predicate"], gate["origin"], gate["n_kept"]) == (None, "no enrich_p recorded", 3)
    assert load_duckdb(db_path, clipped=True).n_crystals == 3


def test_parse_enrich_alpha():
    assert parse_enrich_alpha("auto") == "auto" and parse_enrich_alpha(" AUTO ") == "auto"
    assert parse_enrich_alpha(None) is None and parse_enrich_alpha("off") is None
    assert parse_enrich_alpha("1e-3") == 1e-3 and parse_enrich_alpha(0.05) == 0.05
    for bad in ("0", "1.5", -1.0, "strict", True):
        with pytest.raises(ValueError, match="enrich_alpha"):
            parse_enrich_alpha(bad)


def test_cli_enrich_alpha_reaches_the_store_and_the_recipe(db_path, tmp_path):
    from torchsx.cli import _pipeline as P
    from torchsx.cli.train import _build_parser

    _chance_lattice(db_path)
    base = [str(db_path), "--spacegroup", "P 43 21 2", "--dmin", "1.0", "--gain", "trust_input",
            "--device", "cpu"]
    load = lambda *extra: P.load_prep(_build_parser().parse_args(  # noqa: E731
        base + list(extra) + ["-o", str(tmp_path / ("run" + "".join(extra)))]), verbose=False)
    gated = load()
    assert gated.prep.obs_info.n_crystals == 2
    assert gated.recipe["enrich_alpha"] == "auto"
    assert gated.recipe["enrich_gate"] == [f"enrich_p <= {DEFAULT_ENRICH_ALPHA!r}"]
    every = load("--enrich-alpha", "off")
    assert every.prep.obs_info.n_crystals == 3
    assert (every.recipe["enrich_alpha"], every.recipe["enrich_gate"]) == ("off", ["off"])
    strict = load("--enrich-alpha", "1e-9")
    assert strict.prep.obs_info.n_crystals == 1 and strict.recipe["enrich_alpha"] == 1e-9
    with pytest.raises(SystemExit):
        _build_parser().parse_args(base + ["--enrich-alpha", "2", "-o", str(tmp_path / "bad")])

