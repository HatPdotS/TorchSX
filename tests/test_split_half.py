"""Tests for split-half comparison metrics.

Exercises the live ``torchsx.merging.hkl.stats`` implementation (the
deprecated ``merge._stats`` copy is retained for frozen ``deprecated``
subpackages; see ``torchsx.deprecated``).
"""

import math

import torch
import pytest

from torchsx.merging.hkl.stats import (
    MatchedHalves,
    cc_half,
    cc_star,
    r_split,
    match_halves,
    shell_stats,
    format_shell_table,
    write_shell_csv,
)


@pytest.fixture
def perfect_halves():
    """Perfectly correlated halves (F1 == F2)."""
    F = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    d = torch.tensor([5.0, 4.0, 3.0, 2.5, 2.0])
    hkl = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 0], [0, 0, 1]], dtype=torch.int16)
    return MatchedHalves(F1=F, F2=F.clone(), d_spacing=d, hkl=hkl)


@pytest.fixture
def noisy_halves():
    """Half-correlated halves."""
    torch.manual_seed(0)
    F1 = torch.rand(100) * 100
    F2 = F1 + torch.randn(100) * 10
    d = torch.linspace(2.0, 10.0, 100)
    hkl = torch.zeros(100, 3, dtype=torch.int16)
    return MatchedHalves(F1=F1, F2=F2, d_spacing=d, hkl=hkl)


def test_cc_half_perfect(perfect_halves):
    cc = cc_half(perfect_halves)
    assert cc == pytest.approx(1.0, abs=1e-6)


def test_cc_half_noisy(noisy_halves):
    cc = cc_half(noisy_halves)
    assert 0.5 < cc < 1.0


def test_cc_star_perfect():
    assert cc_star(1.0) == pytest.approx(1.0, abs=1e-6)


def test_cc_star_negative():
    assert cc_star(-0.5) == 0.0


def test_cc_star_formula():
    val = 0.8
    expected = math.sqrt(2.0 * val / (1.0 + val))
    assert cc_star(val) == pytest.approx(expected, abs=1e-10)


def test_r_split_perfect(perfect_halves):
    r = r_split(perfect_halves)
    assert r == pytest.approx(0.0, abs=1e-6)


def test_r_split_noisy(noisy_halves):
    r = r_split(noisy_halves)
    assert 0 < r < 1.0


def test_r_split_empty():
    empty = MatchedHalves(
        F1=torch.zeros(0), F2=torch.zeros(0),
        d_spacing=torch.zeros(0), hkl=torch.zeros(0, 3, dtype=torch.int16),
    )
    assert r_split(empty) == 0.0


def test_match_halves_identical():
    """Matching identical HKL sets returns all reflections."""
    cell = torch.tensor([14.97, 18.85, 18.89, 89.37, 84.94, 67.84])
    hkl = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=torch.int16)
    F1 = torch.tensor([10.0, 20.0, 30.0])
    F2 = torch.tensor([11.0, 21.0, 31.0])

    matched = match_halves(hkl, F1, hkl, F2, cell)
    assert matched.F1.numel() == 3
    assert matched.F2.numel() == 3
    assert matched.d_spacing.numel() == 3


def test_match_halves_partial():
    """Only matching HKLs are returned."""
    cell = torch.tensor([14.97, 18.85, 18.89, 89.37, 84.94, 67.84])
    hkl1 = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=torch.int16)
    hkl2 = torch.tensor([[1, 0, 0], [2, 0, 0]], dtype=torch.int16)
    F1 = torch.tensor([10.0, 20.0, 30.0])
    F2 = torch.tensor([11.0, 41.0])

    matched = match_halves(hkl1, F1, hkl2, F2, cell)
    assert matched.F1.numel() == 1
    assert matched.F1[0].item() == pytest.approx(10.0)
    assert matched.F2[0].item() == pytest.approx(11.0)


def test_match_halves_resolution_filter():
    """d_min and d_max filter matched reflections by resolution."""
    cell = torch.tensor([14.97, 18.85, 18.89, 89.37, 84.94, 67.84])
    # (1,0,0) has large d, (5,5,5) has small d
    hkl = torch.tensor([[1, 0, 0], [2, 0, 0], [5, 5, 5]], dtype=torch.int16)
    F = torch.tensor([10.0, 20.0, 30.0])

    # No filter — all 3 matched
    matched_all = match_halves(hkl, F, hkl, F, cell)
    assert matched_all.F1.numel() == 3

    # Compute d-spacings to pick sensible cutoffs
    d = matched_all.d_spacing
    d_mid = d.median().item()

    # d_min filter (high-res cutoff) — exclude the smallest d
    matched_dmin = match_halves(hkl, F, hkl, F, cell, d_min=d_mid)
    assert matched_dmin.F1.numel() < 3
    assert (matched_dmin.d_spacing >= d_mid).all()

    # d_max filter (low-res cutoff) — exclude the largest d
    matched_dmax = match_halves(hkl, F, hkl, F, cell, d_max=d_mid)
    assert matched_dmax.F1.numel() < 3
    assert (matched_dmax.d_spacing <= d_mid).all()


def test_shell_stats_noisy(noisy_halves):
    shells = shell_stats(noisy_halves, n_shells=5)
    assert len(shells) > 0
    for s in shells:
        assert s.n_matched > 0
        assert s.d_min <= s.d_max


def test_write_shell_csv(noisy_halves, tmp_path):
    """write_shell_csv produces a valid CSV with header and overall row."""
    shells = shell_stats(noisy_halves, n_shells=5)
    path = tmp_path / "shells.csv"
    write_shell_csv(shells, path, overall_cc=0.9, overall_r=0.1)

    text = path.read_text()
    lines = text.strip().split("\n")
    assert lines[0] == "d_max,d_min,n_matched,cc_half,cc_star,r_split"
    # One line per shell + header + overall
    assert len(lines) == len(shells) + 2
    # Overall row has empty d_max/d_min
    assert lines[-1].startswith(",,")


def test_write_shell_csv_no_overall(noisy_halves, tmp_path):
    """write_shell_csv without overall row."""
    shells = shell_stats(noisy_halves, n_shells=5)
    path = tmp_path / "shells.csv"
    write_shell_csv(shells, path)

    lines = path.read_text().strip().split("\n")
    assert len(lines) == len(shells) + 1  # header + shells, no overall


def test_format_shell_table(noisy_halves):
    shells = shell_stats(noisy_halves, n_shells=5)
    table = format_shell_table(shells, overall_cc=0.9, overall_r=0.1)
    assert "CC1/2" in table
    assert "R_split" in table
    assert "Overall" in table
