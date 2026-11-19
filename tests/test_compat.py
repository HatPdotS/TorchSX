"""The probixi pairing: version and schema constants, and the guarded import."""

from __future__ import annotations

import sys
import types

import duckdb
import pytest

import torchsx
from torchsx import _compat
from torchsx.io._duckdb import read_measurement
from tests.test_duckdb_load import db_path  # noqa: F401 - fixture


def _fake_probixi(monkeypatch, schema, torchsx_version="9.9.9", paired=True):
    """Put a stand-in probixi on the import path: paired (with ``_compat``) or older."""
    pkg = types.ModuleType("probixi")
    pkg.__path__ = []
    if paired:
        pkg.__version__ = "9.0.0"
    monkeypatch.setitem(sys.modules, "probixi", pkg)
    if paired:
        compat = types.ModuleType("probixi._compat")
        compat.SCHEMA_VERSION, compat.TORCHSX_VERSION = schema, torchsx_version
        monkeypatch.setitem(sys.modules, "probixi._compat", compat)
    else:
        monkeypatch.setitem(sys.modules, "probixi._compat", None)
        io = types.ModuleType("probixi.io")
        io.__path__ = []
        db = types.ModuleType("probixi.io.db")
        db.SCHEMA_VERSION = schema
        monkeypatch.setitem(sys.modules, "probixi.io", io)
        monkeypatch.setitem(sys.modules, "probixi.io.db", db)
    _compat.installed_probixi.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_lookup():
    _compat.installed_probixi.cache_clear()
    yield
    _compat.installed_probixi.cache_clear()


def test_constants_are_exported_without_torch_imports():
    assert torchsx.SUPPORTED_DB_SCHEMA == _compat.SUPPORTED_DB_SCHEMA == 2
    assert torchsx.PROBIXI_VERSION == _compat.PROBIXI_VERSION
    assert _compat.version_tuple(_compat.PROBIXI_VERSION) >= (0, 10, 3)
    assert _compat.version_tuple("0.10.3.dev1") == (0, 10, 3)
    assert _compat.version_tuple(None) == ()


def test_no_probixi_and_older_probixi_are_not_a_mismatch(monkeypatch):
    monkeypatch.setitem(sys.modules, "probixi", None)
    assert _compat.installed_probixi() is None and _compat.probixi_mismatch() is None
    _fake_probixi(monkeypatch, schema=1, paired=False)
    assert _compat.installed_probixi()["schema"] == 1
    assert _compat.installed_probixi()["torchsx"] is None
    assert _compat.probixi_mismatch() is None
    _fake_probixi(monkeypatch, schema=_compat.SUPPORTED_DB_SCHEMA)
    assert _compat.probixi_mismatch() is None


def test_newer_probixi_names_the_torchsx_it_asks_for(monkeypatch, db_path):  # noqa: F811
    _fake_probixi(monkeypatch, schema=_compat.SUPPORTED_DB_SCHEMA + 1, torchsx_version="9.9.9")
    message = _compat.probixi_mismatch()
    assert f"probixi 9.0.0 writes database schema {_compat.SUPPORTED_DB_SCHEMA + 1}" in message
    assert "Upgrade TorchSX to 9.9.9 or later." in message
    # the reader says so once it opens a database, and still reads one it understands
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("INSERT INTO calibration VALUES ('0.10.3', 2, 100, 1.5, 0.0, false, 1.0, 5.9, "
                     "6.0, 10.2, 0, 2.3, 0.03, 0.96, 'snr', 0.1, 79.0, 79.0, 38.0, 90.0, 90.0, 90.0, 0)")
    with pytest.warns(UserWarning, match="Upgrade TorchSX to 9.9.9 or later"):
        spec = read_measurement(db_path)
    assert ("schema_version", "2") in set(spec.provenance)


def test_installed_probixi_agrees_with_this_torchsx():
    """Where a paired probixi is installed, both packages name the same schema."""
    theirs = pytest.importorskip("probixi._compat")
    if theirs.SCHEMA_VERSION < _compat.SUPPORTED_DB_SCHEMA:
        pytest.skip(f"the installed probixi writes schema {theirs.SCHEMA_VERSION}")
    assert theirs.SCHEMA_VERSION == _compat.SUPPORTED_DB_SCHEMA
    assert _compat.probixi_mismatch() is None
    found = _compat.installed_probixi()
    assert _compat.version_tuple(found["version"]) >= _compat.version_tuple(_compat.PROBIXI_VERSION)
