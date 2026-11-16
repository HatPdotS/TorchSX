"""Resolution-binned completeness and multiplicity (``torchsx.merging.hkl.stats``)."""

import pytest
import torch

from torchsx import SFXStream
from torchsx._core import SFXStreamCore

CELL = [40.0, 50.0, 60.0, 90.0, 90.0, 90.0]
SPACEGROUP = "P 21 21 21"


def _core(hkl, spacegroup=SPACEGROUP):
    n = hkl.shape[0]
    return SFXStreamCore(
        hkl=hkl.to(torch.int16),
        I=torch.ones(n),
        sigma=torch.ones(n),
        background=torch.zeros(n),
        peak=torch.ones(n),
        fs=torch.zeros(n),
        ss=torch.zeros(n),
        panel_id=torch.zeros(n, dtype=torch.int16),
        crystal_id=torch.zeros(n, dtype=torch.int32),
        astar=torch.zeros(1, 3),
        bstar=torch.zeros(1, 3),
        cstar=torch.zeros(1, 3),
        cell=torch.tensor([CELL]),
        profile_rad=torch.zeros(1),
        resolution=torch.ones(1),
        n_refl=torch.tensor([n], dtype=torch.int32),
        image_id=torch.zeros(1, dtype=torch.int32),
        crystal_in_image=torch.zeros(1, dtype=torch.int32),
        beam_energy=torch.ones(1),
        det_shift_x=torch.zeros(1),
        det_shift_y=torch.zeros(1),
        n_peaks=torch.ones(1, dtype=torch.int32),
        spacegroup=spacegroup,
    )


@pytest.fixture(scope="module")
def asu():
    """Every unique, not absent reflection of the cell to 3 A, as ASU indices."""
    from torchref.base.reciprocal.hkl import generate_possible_hkl
    from torchref.symmetry import SpaceGroup

    group = SpaceGroup(SPACEGROUP)
    hkl = generate_possible_hkl(torch.tensor(CELL, dtype=torch.float64), 3.0)
    hkl = hkl[~group.is_absent(hkl)]
    hkl = group.canonicalize_hkl(hkl.int(), include_friedel=True)[0]
    return torch.unique(hkl, dim=0)


def test_complete_data_seen_twice_through_equivalent_indices(asu):
    # Each reflection once at its ASU index and once at a symmetry mate
    # (2-fold about b, then Friedel): both must land on one reflection.
    mate = asu * torch.tensor([1, -1, 1])
    stream = SFXStream(_core(torch.cat([asu, mate])))
    torch.testing.assert_close(stream.completeness(n_bins=8), torch.ones(8))
    torch.testing.assert_close(stream.multiplicity(n_bins=8), torch.full((8,), 2.0))


def test_half_the_reflections_give_about_half_completeness(asu):
    kept = asu[torch.arange(len(asu)) % 2 == 0]
    c = SFXStream(_core(kept)).completeness(n_bins=5)
    assert c.shape == (5,)
    assert ((c > 0.4) & (c < 0.6)).all(), c
    torch.testing.assert_close(SFXStream(_core(kept)).multiplicity(n_bins=5), torch.ones(5))


def test_an_absent_index_does_not_count_as_observed(asu):
    absent = torch.tensor([[0, 0, 1], [0, 0, -3]])  # 0 0 l, l odd: absent in P 21 21 21
    c = SFXStream(_core(torch.cat([asu, absent]))).completeness(n_bins=6)
    torch.testing.assert_close(c, torch.ones(6))


def test_p1_stream_with_friedel_mates(sample_core):
    # sample_core is P 1: Friedel mates are the only equivalents.
    stream = SFXStream(sample_core)
    c, m = stream.completeness(n_bins=3), stream.multiplicity(n_bins=3)
    assert c.shape == m.shape == (3,)
    assert ((c >= 0) & (c <= 1)).all() and (m >= 0).all()
    assert m.max() == 1.0  # five distinct reflections, one observation each
