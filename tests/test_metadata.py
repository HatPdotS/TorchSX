"""Tests for StreamMetadata."""

import tempfile
from pathlib import Path

import torch

from torchsx.io._metadata import StreamMetadata


def test_round_trip_numeric(sample_metadata):
    """Numeric fields survive to_tensors/from_tensors round-trip."""
    tensors, schema = sample_metadata.to_tensors()
    meta2 = StreamMetadata.from_tensors(tensors, schema)

    for a, b in zip(meta2["crystal"]["scale"], sample_metadata["crystal"]["scale"]):
        assert abs(a - b) < 1e-6
    for i in range(2):
        for j in range(6):
            assert abs(meta2["crystal"]["cell"][i][j] - sample_metadata["crystal"]["cell"][i][j]) < 1e-5


def test_round_trip_strings(sample_metadata):
    """String fields survive round-trip via string table encoding."""
    tensors, schema = sample_metadata.to_tensors()
    meta2 = StreamMetadata.from_tensors(tensors, schema)

    assert meta2["image"]["filename"] == sample_metadata["image"]["filename"]


def test_round_trip_integers(sample_metadata):
    """Integer fields survive round-trip."""
    tensors, schema = sample_metadata.to_tensors()
    meta2 = StreamMetadata.from_tensors(tensors, schema)

    assert meta2["image"]["event_id"] == sample_metadata["image"]["event_id"]


def test_string_table_deduplication():
    """String table should deduplicate strings."""
    meta = StreamMetadata()
    meta["test"] = {"names": ["a", "b", "a", "c", "b"]}
    tensors, schema = meta.to_tensors()

    assert "test/names" in tensors
    assert len(schema["test/names"]["strings"]) == 3  # unique: a, b, c
    assert tensors["test/names"].shape[0] == 5  # 5 entries


def test_save_load(sample_metadata, tmp_path):
    """Save/load round-trip."""
    path = tmp_path / "meta"
    sample_metadata.save(path)

    loaded = StreamMetadata.load(path)
    assert loaded["image"]["filename"] == sample_metadata["image"]["filename"]
    for a, b in zip(loaded["crystal"]["scale"], sample_metadata["crystal"]["scale"]):
        assert abs(a - b) < 1e-6


def test_empty_metadata():
    """Empty metadata should work."""
    meta = StreamMetadata()
    tensors, schema = meta.to_tensors()
    assert len(tensors) == 0
    assert len(schema) == 0

    meta2 = StreamMetadata.from_tensors(tensors, schema)
    assert len(meta2) == 0


def test_to_device():
    """to() should move tensors."""
    meta = StreamMetadata()
    meta["test"] = {"values": [1.0, 2.0, 3.0]}
    # Just ensure it doesn't crash on CPU→CPU
    meta2 = meta.to("cpu")
    assert meta2["test"]["values"] == [1.0, 2.0, 3.0]
