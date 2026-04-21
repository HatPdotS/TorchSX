"""Tests for sparse pixel-level refinement: matching, forward, loss."""

import torch
import pytest

from torchsx.forward._pixel_matching import (
    PixelReflectionPairs,
    SparseForwardResult,
    match_pixels_to_reflections,
)
from torchsx.forward._sparse_pixel_loss import SparsePixelLoss, SparsePixelResult


# ── Matching tests ─────────────────────────────────────────────────────

class TestMatchPixelsToReflections:
    def test_basic_matching(self):
        """Pixels within search_radius are matched, others are not."""
        refl_ss = torch.tensor([100.0, 200.0])
        refl_fs = torch.tensor([100.0, 200.0])
        refl_frame = torch.tensor([0, 0])

        pixel_ss = torch.tensor([102.0, 198.0, 500.0])
        pixel_fs = torch.tensor([101.0, 199.0, 500.0])
        pixel_frame = torch.tensor([0, 0, 0])
        pixel_counts = torch.tensor([5, 3, 1], dtype=torch.int32)
        pixel_bg = torch.tensor([1.0, 1.0, 1.0])

        refl_idx, pixel_idx, pair_ptr = match_pixels_to_reflections(
            refl_ss, refl_fs, refl_frame,
            pixel_ss, pixel_fs, pixel_frame,
            pixel_counts, pixel_bg,
            search_radius=10.0,
        )

        assert refl_idx.shape[0] == 2
        assert pixel_idx.shape[0] == 2
        assert pair_ptr.shape[0] == 3
        assert pair_ptr[-1].item() == 2

    def test_no_cross_frame_matching(self):
        """Reflections and pixels on different frames should not match."""
        refl_ss = torch.tensor([100.0])
        refl_fs = torch.tensor([100.0])
        refl_frame = torch.tensor([0])

        pixel_ss = torch.tensor([101.0])
        pixel_fs = torch.tensor([101.0])
        pixel_frame = torch.tensor([1])  # different frame
        pixel_counts = torch.tensor([5], dtype=torch.int32)
        pixel_bg = torch.tensor([1.0])

        refl_idx, pixel_idx, pair_ptr = match_pixels_to_reflections(
            refl_ss, refl_fs, refl_frame,
            pixel_ss, pixel_fs, pixel_frame,
            pixel_counts, pixel_bg,
            search_radius=10.0,
        )

        assert refl_idx.shape[0] == 0

    def test_multiple_pixels_per_reflection(self):
        """A reflection can match multiple nearby pixels."""
        refl_ss = torch.tensor([100.0])
        refl_fs = torch.tensor([100.0])
        refl_frame = torch.tensor([0])

        pixel_ss = torch.tensor([99.0, 100.0, 101.0, 100.0, 100.0])
        pixel_fs = torch.tensor([100.0, 99.0, 100.0, 101.0, 100.0])
        pixel_frame = torch.tensor([0, 0, 0, 0, 0])
        pixel_counts = torch.tensor([5, 3, 4, 2, 6], dtype=torch.int32)
        pixel_bg = torch.ones(5)

        refl_idx, pixel_idx, pair_ptr = match_pixels_to_reflections(
            refl_ss, refl_fs, refl_frame,
            pixel_ss, pixel_fs, pixel_frame,
            pixel_counts, pixel_bg,
            search_radius=5.0,
        )

        assert refl_idx.shape[0] == 5
        assert (refl_idx == 0).all()
        assert pair_ptr[0].item() == 0
        assert pair_ptr[1].item() == 5

    def test_empty_inputs(self):
        """Empty input arrays should return empty outputs."""
        refl_idx, pixel_idx, pair_ptr = match_pixels_to_reflections(
            torch.empty(0), torch.empty(0),
            torch.empty(0, dtype=torch.long),
            torch.empty(0), torch.empty(0),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.int32), torch.empty(0),
            search_radius=10.0,
        )
        assert refl_idx.shape[0] == 0
        assert pixel_idx.shape[0] == 0
        assert pair_ptr.shape[0] == 1  # 0 reflections + 1

    def test_csr_consistency(self):
        """CSR pointers should be consistent with pair counts."""
        refl_ss = torch.tensor([100.0, 200.0, 300.0])
        refl_fs = torch.tensor([100.0, 200.0, 300.0])
        refl_frame = torch.tensor([0, 0, 0])

        pixel_ss = torch.tensor([101.0, 99.0, 201.0])
        pixel_fs = torch.tensor([101.0, 99.0, 201.0])
        pixel_frame = torch.tensor([0, 0, 0])
        pixel_counts = torch.tensor([5, 3, 4], dtype=torch.int32)
        pixel_bg = torch.ones(3)

        refl_idx, pixel_idx, pair_ptr = match_pixels_to_reflections(
            refl_ss, refl_fs, refl_frame,
            pixel_ss, pixel_fs, pixel_frame,
            pixel_counts, pixel_bg,
            search_radius=5.0,
        )

        assert pair_ptr.shape[0] == 4  # 3 reflections + 1
        assert pair_ptr[0].item() == 0
        assert pair_ptr[-1].item() == refl_idx.shape[0]

        # Pairs should be sorted by refl_idx
        assert (refl_idx[:-1] <= refl_idx[1:]).all()

    def test_multi_frame(self):
        """Matching works correctly across multiple frames."""
        refl_ss = torch.tensor([100.0, 200.0])
        refl_fs = torch.tensor([100.0, 200.0])
        refl_frame = torch.tensor([0, 1])

        pixel_ss = torch.tensor([101.0, 201.0])
        pixel_fs = torch.tensor([101.0, 201.0])
        pixel_frame = torch.tensor([0, 1])
        pixel_counts = torch.tensor([5, 3], dtype=torch.int32)
        pixel_bg = torch.ones(2)

        refl_idx, pixel_idx, pair_ptr = match_pixels_to_reflections(
            refl_ss, refl_fs, refl_frame,
            pixel_ss, pixel_fs, pixel_frame,
            pixel_counts, pixel_bg,
            search_radius=5.0,
        )

        assert refl_idx.shape[0] == 2  # one pair per frame

    def test_cross_panel_matching(self):
        """Pixels near panel boundaries should match reflections on adjacent panels."""
        # Reflection at ss=513 (panel 0 edge), pixel at ss=515 (panel 1 start)
        # Should match since we use distance only, no panel constraint
        refl_ss = torch.tensor([513.0])
        refl_fs = torch.tensor([100.0])
        refl_frame = torch.tensor([0])

        pixel_ss = torch.tensor([515.0])
        pixel_fs = torch.tensor([101.0])
        pixel_frame = torch.tensor([0])
        pixel_counts = torch.tensor([5], dtype=torch.int32)
        pixel_bg = torch.tensor([1.0])

        refl_idx, pixel_idx, pair_ptr = match_pixels_to_reflections(
            refl_ss, refl_fs, refl_frame,
            pixel_ss, pixel_fs, pixel_frame,
            pixel_counts, pixel_bg,
            search_radius=5.0,
        )

        assert refl_idx.shape[0] == 1  # should match across panels


# ── PixelReflectionPairs tests ─────────────────────────────────────────

class TestPixelReflectionPairs:
    def _make_simple_pairs(self) -> PixelReflectionPairs:
        """Create a simple PixelReflectionPairs for testing."""
        return PixelReflectionPairs(
            crystal_id=torch.tensor([0, 1], dtype=torch.int32),
            hkl_id=torch.tensor([0, 1], dtype=torch.int64),
            hkl=torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.int16),
            panel_id=torch.tensor([0, 0], dtype=torch.int16),
            ss_pred=torch.tensor([100.0, 200.0]),
            fs_pred=torch.tensor([100.0, 200.0]),
            pixel_ss=torch.tensor([101.0, 199.0, 100.0]),
            pixel_fs=torch.tensor([101.0, 199.0, 100.0]),
            pixel_panel=torch.tensor([0, 0, 0], dtype=torch.int16),
            pixel_counts=torch.tensor([5, 3, 4], dtype=torch.int32),
            pixel_bg=torch.tensor([1.0, 1.0, 1.0]),
            refl_idx=torch.tensor([0, 0, 1], dtype=torch.int64),
            pixel_idx=torch.tensor([2, 0, 1], dtype=torch.int64),
            pair_ptr=torch.tensor([0, 2, 3], dtype=torch.int64),
        )

    def test_properties(self):
        pairs = self._make_simple_pairs()
        assert pairs.n_refl == 2
        assert pairs.n_pixels == 3
        assert pairs.n_pairs == 3

    def test_to_device(self):
        pairs = self._make_simple_pairs()
        pairs2 = pairs.to(torch.device("cpu"))
        assert pairs2.n_refl == 2


# ── SparseForwardResult tests ─────────────────────────────────────────

class TestSparseForwardResult:
    def test_creation(self):
        result = SparseForwardResult(
            n_signal_pairs=torch.tensor([0.1, 0.2, 0.3]),
            n_pred_pixels=torch.tensor([1.1, 1.2, 1.3]),
            F_sq=torch.tensor([10.0, 20.0]),
            scale=torch.tensor([1.0, 1.0]),
        )
        assert result.n_signal_pairs.shape == (3,)
        assert result.n_pred_pixels.shape == (3,)


# ── Loss function tests ───────────────────────────────────────────────

class TestSparsePixelLoss:
    def _make_result_and_pairs(self):
        """Create matching SparseForwardResult and PixelReflectionPairs."""
        N_pixels = 10
        N_refl = 2
        N_pairs = 15

        pairs = PixelReflectionPairs(
            crystal_id=torch.zeros(N_refl, dtype=torch.int32),
            hkl_id=torch.arange(N_refl, dtype=torch.int64),
            hkl=torch.zeros(N_refl, 3, dtype=torch.int16),
            panel_id=torch.zeros(N_refl, dtype=torch.int16),
            ss_pred=torch.zeros(N_refl),
            fs_pred=torch.zeros(N_refl),
            pixel_ss=torch.zeros(N_pixels),
            pixel_fs=torch.zeros(N_pixels),
            pixel_panel=torch.zeros(N_pixels, dtype=torch.int16),
            pixel_counts=torch.randint(0, 10, (N_pixels,), dtype=torch.int32),
            pixel_bg=torch.ones(N_pixels),
            refl_idx=torch.randint(0, N_refl, (N_pairs,), dtype=torch.int64),
            pixel_idx=torch.randint(0, N_pixels, (N_pairs,), dtype=torch.int64),
            pair_ptr=torch.tensor([0, 8, 15], dtype=torch.int64),
        )

        result = SparseForwardResult(
            n_signal_pairs=torch.rand(N_pairs) * 0.1,
            n_pred_pixels=torch.rand(N_pixels) + 1.0,
            F_sq=torch.ones(N_refl),
            scale=torch.ones(N_refl),
        )

        return result, pairs

    def test_loss_runs(self):
        result, pairs = self._make_result_and_pairs()
        loss_fn = SparsePixelLoss()
        lr = loss_fn(result, pairs)
        assert isinstance(lr, SparsePixelResult)
        assert lr.loss.shape == ()
        assert lr.n_pixels == 10
        assert lr.n_pairs == 15

    def test_loss_with_validation(self):
        result, pairs = self._make_result_and_pairs()
        loss_fn = SparsePixelLoss(val_frac=0.2)
        lr = loss_fn(result, pairs)
        assert lr.val_loss is not None
        assert lr.n_val > 0

    def test_loss_gradient(self):
        """Loss should produce gradients for n_pred_pixels."""
        result, pairs = self._make_result_and_pairs()
        result.n_pred_pixels.requires_grad_(True)
        loss_fn = SparsePixelLoss()
        lr = loss_fn(result, pairs)
        lr.loss.backward()
        assert result.n_pred_pixels.grad is not None
        assert not torch.isnan(result.n_pred_pixels.grad).any()

    def test_lower_loss_for_better_prediction(self):
        """Loss should be lower when prediction matches observation."""
        N_pixels = 100
        counts = torch.tensor([5] * N_pixels, dtype=torch.int32)

        pairs = PixelReflectionPairs(
            crystal_id=torch.zeros(1, dtype=torch.int32),
            hkl_id=torch.zeros(1, dtype=torch.int64),
            hkl=torch.zeros(1, 3, dtype=torch.int16),
            panel_id=torch.zeros(1, dtype=torch.int16),
            ss_pred=torch.zeros(1),
            fs_pred=torch.zeros(1),
            pixel_ss=torch.zeros(N_pixels),
            pixel_fs=torch.zeros(N_pixels),
            pixel_panel=torch.zeros(N_pixels, dtype=torch.int16),
            pixel_counts=counts,
            pixel_bg=torch.ones(N_pixels),
            refl_idx=torch.zeros(N_pixels, dtype=torch.int64),
            pixel_idx=torch.arange(N_pixels, dtype=torch.int64),
            pair_ptr=torch.tensor([0, N_pixels], dtype=torch.int64),
        )

        loss_fn = SparsePixelLoss()

        # Good prediction: lambda = 5
        result_good = SparseForwardResult(
            n_signal_pairs=torch.zeros(N_pixels),
            n_pred_pixels=torch.full((N_pixels,), 5.0),
            F_sq=torch.ones(1),
            scale=torch.ones(1),
        )

        # Bad prediction: lambda = 20
        result_bad = SparseForwardResult(
            n_signal_pairs=torch.zeros(N_pixels),
            n_pred_pixels=torch.full((N_pixels,), 20.0),
            F_sq=torch.ones(1),
            scale=torch.ones(1),
        )

        loss_good = loss_fn(result_good, pairs).loss.item()
        loss_bad = loss_fn(result_bad, pairs).loss.item()

        assert loss_good < loss_bad


# ── Scatter-add aggregation test ───────────────────────────────────────

class TestScatterAdd:
    def test_multiple_reflections_per_pixel(self):
        """When multiple reflections contribute to one pixel, signals add."""
        N_pixels = 1
        N_refl = 3
        N_pairs = 3

        signal_per_pair = torch.tensor([1.0, 2.0, 3.0])
        pixel_idx = torch.tensor([0, 0, 0], dtype=torch.int64)

        n_signal_per_pixel = torch.zeros(N_pixels)
        n_signal_per_pixel.scatter_add_(0, pixel_idx, signal_per_pair)

        assert n_signal_per_pixel[0].item() == pytest.approx(6.0)


# ── PixelIndexMapper reverse lookup test ───────────────────────────────

class TestPixelIndexMapperReverse:
    def test_flat_to_ss_fs(self):
        from torchsx.forward._sparse_data import PixelIndexMapper

        # 4×4 mask with some bad pixels
        mask = torch.ones(4, 4, dtype=torch.bool)
        mask[0, 0] = False  # pixel (0,0) is bad
        mask[2, 3] = False  # pixel (2,3) is bad

        mapper = PixelIndexMapper(mask)
        assert mapper.n_good == 14

        # Flat index 0 → first good pixel (0,1)
        ss, fs = mapper.flat_to_ss_fs(torch.tensor([0]))
        assert ss[0].item() == 0
        assert fs[0].item() == 1

        # Round-trip: grid_to_flat → flat_to_ss_fs should recover coordinates
        ss_grid = torch.tensor([[[1, 1], [2, 2]]], dtype=torch.float32)
        fs_grid = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.float32)
        flat_idx, on_good = mapper.grid_to_flat(ss_grid, fs_grid)

        # All these pixels are good
        assert on_good.all()

        # Reverse lookup
        flat_vals = flat_idx[on_good]
        ss_back, fs_back = mapper.flat_to_ss_fs(flat_vals)

        expected_ss = torch.tensor([1, 1, 2, 2], dtype=torch.int32)
        expected_fs = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
        assert (ss_back == expected_ss).all()
        assert (fs_back == expected_fs).all()
