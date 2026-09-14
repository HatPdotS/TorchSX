"""Integration tests for SFXStream public API."""

import pytest
import torch

from torchsx.stream import SFXStream


def test_sfxstream_from_core(sample_core):
    """Construct SFXStream from core."""
    stream = SFXStream(sample_core)
    assert stream.n_crystals == 2
    assert stream.n_observations == 5
    assert stream.n_images == 2
    assert repr(stream).startswith("SFXStream(")


def test_sfxstream_save_load(sample_core, tmp_path):
    """Save and load round-trip."""
    stream = SFXStream(sample_core)
    path = tmp_path / "test.sfx"
    stream.save(path)

    loaded = SFXStream.load(path)
    assert loaded.n_crystals == stream.n_crystals
    assert loaded.n_observations == stream.n_observations
    assert torch.equal(loaded.hkl, stream.hkl)


def test_sfxstream_filter_crystals(sample_core):
    """Filter crystals via SFXStream API."""
    stream = SFXStream(sample_core)
    mask = torch.tensor([True, False])
    filtered = stream.filter_crystals(mask)
    assert filtered.n_crystals == 1
    assert filtered.n_observations == 3


def test_sfxstream_filter_cell(sample_core):
    """Cell filter via SFXStream API."""
    stream = SFXStream(sample_core)
    filtered = stream.filter_cell(tolerance=0.1)
    assert filtered.n_crystals == stream.n_crystals


def test_sfxstream_write_stream(sample_core, tmp_path):
    """Write stream via SFXStream API."""
    stream = SFXStream(sample_core)
    out_path = tmp_path / "output.stream"
    stream.write_stream(out_path)

    text = out_path.read_text()
    assert "CrystFEL stream format 2.3" in text
    assert "Begin crystal" in text


def test_sfxstream_to_device(sample_core):
    """Device transfer via SFXStream API."""
    stream = SFXStream(sample_core)
    stream2 = stream.to("cpu")
    assert stream2.n_observations == stream.n_observations


@pytest.mark.slow
def test_sfxstream_cache_round_trip(tmp_path):
    """Parse real stream, cache as .sfx, reload and compare."""
    from tests.conftest import STREAM_PATH

    cache_path = tmp_path / "cached.sfx"
    stream1 = SFXStream.from_stream(STREAM_PATH, spacegroup="P 1", cache=cache_path)

    # Load from cache
    stream2 = SFXStream.from_stream(STREAM_PATH, spacegroup="P 1", cache=cache_path)

    assert stream2.n_crystals == stream1.n_crystals
    assert stream2.n_observations == stream1.n_observations
    assert torch.equal(stream2.hkl, stream1.hkl)


def test_observations_by_hkl(sample_core):
    """Group intensities by Miller index."""
    stream = SFXStream(sample_core)
    obs = stream.observations_by_hkl()

    # sample_core has 5 obs: hkl = [[1,0,0],[0,1,0],[1,1,0],[2,0,0],[0,2,0]]
    # All unique → 5 keys, each with 1 observation
    assert len(obs) == 5
    assert (1, 0, 0) in obs
    assert obs[(1, 0, 0)].shape == (1,)
    assert obs[(1, 0, 0)].item() == pytest.approx(10.0)
    assert obs[(2, 0, 0)].item() == pytest.approx(40.0)

    # Total observations should equal n_observations
    total = sum(v.numel() for v in obs.values())
    assert total == sample_core.n_observations


# ---------------------------------------------------------------------------
# concat
# ---------------------------------------------------------------------------


def test_concat_two(sample_core):
    """Concat two identical streams doubles everything."""
    s = SFXStream(sample_core)
    joined = SFXStream.concat([s, s])

    assert joined.n_observations == 2 * s.n_observations
    assert joined.n_crystals == 2 * s.n_crystals
    assert joined.n_images == 2 * s.n_images

    # crystal_id for second half should be offset
    cid = joined.crystal_id
    assert cid[:s.n_observations].max() < s.n_crystals
    assert cid[s.n_observations:].min() >= s.n_crystals

    # image_id for second half should be offset
    iid = joined.image_id
    assert iid[:s.n_crystals].max() < s.n_images
    assert iid[s.n_crystals:].min() >= s.n_images

    # Intensities are just concatenated
    assert torch.equal(joined.I[:s.n_observations], s.I)
    assert torch.equal(joined.I[s.n_observations:], s.I)


def test_concat_single(sample_core):
    """Concat of one stream returns it unchanged."""
    s = SFXStream(sample_core)
    assert SFXStream.concat([s]) is s


def test_concat_empty_raises():
    """Concat of empty list raises ValueError."""
    with pytest.raises(ValueError):
        SFXStream.concat([])


def test_concat_three(sample_core):
    """Concat three streams."""
    s = SFXStream(sample_core)
    joined = SFXStream.concat([s, s, s])

    assert joined.n_observations == 3 * s.n_observations
    assert joined.n_crystals == 3 * s.n_crystals
    assert joined.n_images == 3 * s.n_images

    # Verify crystal_id continuity: max of chunk i < min of chunk i+1
    n = s.n_observations
    for i in range(2):
        chunk_max = joined.crystal_id[i * n : (i + 1) * n].max()
        chunk_next_min = joined.crystal_id[(i + 1) * n : (i + 2) * n].min()
        assert chunk_max < chunk_next_min


def test_concat_preserves_data(sample_core):
    """Cell params, hkl, and sigma are faithfully concatenated."""
    s = SFXStream(sample_core)
    joined = SFXStream.concat([s, s])

    assert torch.equal(joined.cell[:s.n_crystals], s.cell)
    assert torch.equal(joined.cell[s.n_crystals:], s.cell)
    assert torch.equal(joined.sigma[:s.n_observations], s.sigma)
    assert torch.equal(joined.hkl[:s.n_observations], s.hkl)


def test_concat_carries_extra_column(sample_core):
    """An arbitrary obs column (shoebox) rides along through concat."""
    n = sample_core.n_observations
    core = sample_core.with_columns("obs", shoebox=torch.arange(n * 4.0).reshape(n, 2, 2))
    s = SFXStream(core)
    joined = SFXStream.concat([s, s])
    sb = joined._core.levels["obs"]["shoebox"]
    assert sb.shape == (2 * n, 2, 2)
    assert torch.equal(sb[:n], core.levels["obs"]["shoebox"])
    assert type(joined._core).__name__ == "ShoeboxStreamCore"


# ---------------------------------------------------------------------------
# split_random
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# symmetrize
# ---------------------------------------------------------------------------


def test_symmetrize_friedel_mates(sample_core):
    """Friedel mates collapse to same canonical HKL under P -1."""
    # Insert both (1,0,0) and (-1,0,0) as observations
    hkl = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [2, 0, 0], [0, 2, 0]], dtype=torch.int16)
    core = sample_core.replace(hkl=hkl)
    stream = SFXStream(core)

    sym = stream.symmetrize("P -1")

    # Friedel mates should map to the same canonical index
    assert torch.equal(sym.hkl[0], sym.hkl[1])
    # Observation count unchanged
    assert sym.n_observations == stream.n_observations


def test_symmetrize_preserves_count(sample_core):
    """Symmetrize does not change observation count."""
    stream = SFXStream(sample_core)
    sym = stream.symmetrize("P -1")
    assert sym.n_observations == stream.n_observations
    assert sym.n_crystals == stream.n_crystals


def test_symmetrize_default_spacegroup(sample_core):
    """Uses core.spacegroup when no argument given."""
    core = sample_core.replace(spacegroup="P -1")
    stream = SFXStream(core)
    sym = stream.symmetrize()
    assert sym._core.spacegroup == "P -1"
    # Should still canonicalize
    assert sym.n_observations == stream.n_observations


def test_load_merge_friedel(sample_core, tmp_path):
    """merge_friedel picks the merging group: the Laue class, or the point group."""
    hkl = torch.tensor([[1, 0, 1], [-1, 0, -1], [0, 1, 1], [2, 0, 1], [0, 2, 1]],
                       dtype=torch.int16)
    core = sample_core.replace(hkl=hkl, spacegroup="P 63", friedel_merged=False)
    stream = SFXStream(core)
    path = tmp_path / "test.sfx"
    stream.save(path)

    loaded = SFXStream.load(path, merge_friedel=True)
    assert torch.equal(loaded.hkl[0], loaded.hkl[1])          # mates share one index
    assert loaded.friedel_merged

    split = SFXStream.load(path, merge_friedel=False)
    assert not split.friedel_merged
    # The mates are the two branches of one ASU index, told apart by its sign.
    assert torch.equal(split.hkl[0], -split.hkl[1])
    assert torch.equal(split.hkl[0].abs(), loaded.hkl[0])


def test_symmetrize_separates_only_acentric_mates(sample_core):
    """A centric reflection is its own mate, so it must stay on one branch."""
    # In P 63 the l = 0 zone is centric; l != 0 is not.
    hkl = torch.tensor([[1, 0, 0], [-1, 0, 0], [1, 0, 1], [-1, 0, -1], [0, 2, 1]],
                       dtype=torch.int16)
    stream = SFXStream(sample_core.replace(hkl=hkl, spacegroup="P 63",
                                           friedel_merged=False))
    split = stream.symmetrize(separate_friedel=True)
    assert torch.equal(split.hkl[0], split.hkl[1])            # centric: still merged
    assert torch.equal(split.hkl[2], -split.hkl[3])           # acentric: split
    # Re-merging the signed column reproduces the Laue-class merge.
    assert torch.equal(split.symmetrize().hkl, stream.symmetrize().hkl)


def test_symmetrize_cannot_split_a_merged_stream(sample_core):
    """Without the observed index the two mates are indistinguishable."""
    merged = SFXStream(sample_core.replace(spacegroup="P 63", friedel_merged=True))
    with pytest.raises(ValueError, match="Cannot separate Friedel mates"):
        merged.symmetrize(separate_friedel=True)


def test_split_random_two_halves(sample_core):
    """split_random(2) produces two disjoint subsets covering all crystals."""
    s = SFXStream(sample_core)
    # Concat several copies to get enough crystals for meaningful split
    big = SFXStream.concat([s] * 10)
    parts = big.split_random(n_parts=2, seed=42)

    assert len(parts) == 2
    assert parts[0].n_crystals + parts[1].n_crystals == big.n_crystals
    assert parts[0].n_observations + parts[1].n_observations == big.n_observations


def test_split_random_reproducible(sample_core):
    """Same seed gives same split."""
    s = SFXStream(sample_core)
    big = SFXStream.concat([s] * 10)
    a1, a2 = big.split_random(n_parts=2, seed=123)
    b1, b2 = big.split_random(n_parts=2, seed=123)

    assert a1.n_crystals == b1.n_crystals
    assert a2.n_crystals == b2.n_crystals


def test_split_random_three_parts(sample_core):
    """split_random(3) produces three groups."""
    s = SFXStream(sample_core)
    big = SFXStream.concat([s] * 10)
    parts = big.split_random(n_parts=3, seed=0)

    assert len(parts) == 3
    assert sum(p.n_crystals for p in parts) == big.n_crystals
