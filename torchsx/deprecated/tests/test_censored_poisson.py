"""Tests for CensoredPoissonLoss and supporting utilities."""

from __future__ import annotations

import torch
import pytest
from torch import Tensor

from torchsx.deprecated.forward._censored_poisson import (
    CensoredPoissonLoss,
    CensoredPoissonResult,
    compute_censoring_threshold,
)
from torchsx.deprecated.forward._sparse_data import PixelIndexMapper


# ──────────────────────── compute_censoring_threshold ────────────────────────


class TestComputeCensoringThreshold:
    def test_monotone_in_background(self):
        """Higher background → higher threshold."""
        bg = torch.tensor([1.0, 10.0, 100.0, 1000.0])
        t = compute_censoring_threshold(bg, z_threshold=4.0)
        assert (t[1:] >= t[:-1]).all()

    def test_monotone_in_z(self):
        """Higher z_threshold → higher threshold."""
        bg = torch.full((5,), 50.0)
        t_low = compute_censoring_threshold(bg, z_threshold=3.0)
        t_high = compute_censoring_threshold(bg, z_threshold=5.0)
        assert (t_high >= t_low).all()

    def test_minimum_clamp(self):
        """Threshold is at least 1."""
        bg = torch.tensor([0.0, 0.001, 0.01])
        t = compute_censoring_threshold(bg, z_threshold=4.0)
        assert (t >= 1.0).all()

    def test_matches_anscombe(self):
        """Verify threshold inverts the Anscombe transform correctly."""
        bg = torch.tensor([50.0])
        z_thr = 4.0
        t = compute_censoring_threshold(bg, z_thr)
        # At threshold k=t: z = (sqrt(k + 3/8) - sqrt(bg + 3/8)) / 0.5
        k = t.item()
        z_at_t = (torch.sqrt(torch.tensor(k + 0.375))
                  - torch.sqrt(bg + 0.375)) / 0.5
        # z_at_t should be >= z_threshold (we used ceil)
        assert z_at_t.item() >= z_thr - 0.01
        # But one less than threshold should be below
        z_below = (torch.sqrt(torch.tensor(k - 1 + 0.375))
                   - torch.sqrt(bg + 0.375)) / 0.5
        assert z_below.item() < z_thr + 0.01


# ──────────────────────── CensoredPoissonLoss ────────────────────────────────


def _make_batch(B=4, ph=11, pw=11, *, signal_frac=0.1, bg_level=50.0):
    """Create synthetic test tensors for CensoredPoissonLoss."""
    background = torch.full((B, ph, pw), bg_level)
    n_signal = torch.zeros(B, ph, pw)
    gain = torch.ones(B, 1, 1)
    valid = torch.ones(B, ph, pw, dtype=torch.bool)

    # Place some predicted signal in center pixels
    center = ph // 2
    n_signal[:, center-1:center+2, center-1:center+2] = 100.0

    # Sparse set: center pixels have high counts
    is_sparse = torch.zeros(B, ph, pw, dtype=torch.bool)
    is_sparse[:, center-1:center+2, center-1:center+2] = True

    sparse_counts = torch.zeros(B, ph, pw, dtype=torch.int32)
    sparse_counts[:, center-1:center+2, center-1:center+2] = 150

    threshold = compute_censoring_threshold(background, z_threshold=4.0)

    return n_signal, background, is_sparse, sparse_counts, threshold, valid, gain


class TestCensoredPoissonLoss:
    def test_returns_result(self):
        """Basic smoke test: loss returns CensoredPoissonResult."""
        args = _make_batch()
        loss_fn = CensoredPoissonLoss()
        result = loss_fn(*args)
        assert isinstance(result, CensoredPoissonResult)
        assert result.loss.shape == ()
        assert result.n_term1 > 0

    def test_term1_activates_for_sparse_predicted(self):
        """Term 1 fires when sparse pixels overlap with predicted signal."""
        n_signal, bg, is_sparse, counts, threshold, valid, gain = _make_batch()
        loss_fn = CensoredPoissonLoss()
        result = loss_fn(n_signal, bg, is_sparse, counts, threshold, valid, gain)
        assert result.n_term1 > 0
        # Poisson NLL can be negative when k > λ, just check it's finite
        assert torch.isfinite(result.term1)

    def test_term2_activates_for_censored_predicted(self):
        """Term 2 fires when censored pixels have predicted signal."""
        n_signal, bg, is_sparse, counts, threshold, valid, gain = _make_batch()
        # Make signal extend beyond sparse set
        n_signal[:, :, :] = 50.0  # Signal everywhere
        loss_fn = CensoredPoissonLoss()
        result = loss_fn(n_signal, bg, is_sparse, counts, threshold, valid, gain)
        assert result.n_term2 > 0
        assert result.term2.item() > 0

    def test_weak_signal_gets_gradient(self):
        """Sparse pixels with zero predicted signal still get gradient."""
        scale = torch.tensor(0.001, requires_grad=True)
        n_signal_base, bg, is_sparse, counts, threshold, valid, gain = _make_batch()
        # Very weak signal — previously would have been gated out
        n_signal = n_signal_base.detach() * scale
        loss_fn = CensoredPoissonLoss()
        result = loss_fn(n_signal, bg, is_sparse, counts, threshold, valid, gain)
        assert result.n_term1 > 0
        result.loss.backward()
        assert scale.grad is not None
        assert scale.grad.item() != 0.0

    def test_no_sparse_zero_term1(self):
        """No sparse pixels → term1 is zero."""
        n_signal, bg, is_sparse, counts, threshold, valid, gain = _make_batch()
        is_sparse = torch.zeros_like(is_sparse)
        loss_fn = CensoredPoissonLoss()
        result = loss_fn(n_signal, bg, is_sparse, counts, threshold, valid, gain)
        assert result.n_term1 == 0

    def test_valid_mask_excludes_pixels(self):
        """Invalid pixels are excluded from all terms."""
        n_signal, bg, is_sparse, counts, threshold, valid, gain = _make_batch()
        n_signal[:, :, :] = 50.0
        valid_full = valid.clone()
        loss_fn = CensoredPoissonLoss()
        result_full = loss_fn(n_signal, bg, is_sparse, counts, threshold, valid_full, gain)

        # Mask out half the pixels
        valid_half = valid.clone()
        valid_half[:, :, :5] = False
        result_half = loss_fn(n_signal, bg, is_sparse, counts, threshold, valid_half, gain)

        assert result_half.n_term1 + result_half.n_term2 < \
               result_full.n_term1 + result_full.n_term2

    def test_gradient_through_scale(self):
        """Gradient flows from loss through a simple scale parameter."""
        scale = torch.tensor(2.0, requires_grad=True)
        n_signal_base, bg, is_sparse, counts, threshold, valid, gain = _make_batch()
        # Use scale != 1 so k != λ (at k==λ the Poisson NLL gradient is zero)
        n_signal = n_signal_base.detach() * scale

        loss_fn = CensoredPoissonLoss()
        result = loss_fn(n_signal, bg, is_sparse, counts, threshold, valid, gain)
        result.loss.backward()

        assert scale.grad is not None
        assert scale.grad.item() != 0.0

    def test_gradient_term2_through_scale(self):
        """Gradient from term2 (censored penalty) flows through scale."""
        scale = torch.tensor(1.0, requires_grad=True)
        n_signal_base = torch.zeros(2, 11, 11)
        # Signal everywhere → censored pixels get term2
        n_signal_base[:, :, :] = 50.0
        n_signal = n_signal_base * scale

        bg = torch.full((2, 11, 11), 50.0)
        is_sparse = torch.zeros(2, 11, 11, dtype=torch.bool)
        # Only a few sparse pixels
        is_sparse[:, 5, 5] = True
        counts = torch.zeros(2, 11, 11, dtype=torch.int32)
        counts[:, 5, 5] = 150
        threshold = compute_censoring_threshold(bg, 4.0)
        valid = torch.ones(2, 11, 11, dtype=torch.bool)
        gain = torch.ones(2, 1, 1)

        loss_fn = CensoredPoissonLoss()
        result = loss_fn(n_signal, bg, is_sparse, counts, threshold, valid, gain)
        assert result.n_term2 > 0
        result.loss.backward()
        assert scale.grad is not None

    def test_poisson_cdf_matches_scipy(self):
        """Verify gammaincc matches scipy.stats.poisson.cdf for a range of values."""
        pytest.importorskip("scipy")
        from scipy.stats import poisson

        lam_vals = torch.tensor([1.0, 5.0, 10.0, 50.0, 100.0])
        t_vals = torch.tensor([0.0, 3.0, 8.0, 45.0, 90.0])

        for lam, t in zip(lam_vals, t_vals):
            # PyTorch: gammaincc(t+1, lam) = Poisson CDF P(X <= t | lam)
            cdf_torch = torch.special.gammaincc(t + 1.0, lam).item()
            cdf_scipy = poisson.cdf(int(t.item()), lam.item())
            assert abs(cdf_torch - cdf_scipy) < 1e-6, \
                f"CDF mismatch at lam={lam.item()}, t={t.item()}: " \
                f"torch={cdf_torch:.8f} vs scipy={cdf_scipy:.8f}"

    def test_term2_penalizes_hallucinated_signal(self):
        """Higher predicted signal on censored pixels → higher term2 loss."""
        bg = torch.full((2, 11, 11), 50.0)
        is_sparse = torch.zeros(2, 11, 11, dtype=torch.bool)
        counts = torch.zeros(2, 11, 11, dtype=torch.int32)
        threshold = compute_censoring_threshold(bg, 4.0)
        valid = torch.ones(2, 11, 11, dtype=torch.bool)
        gain = torch.ones(2, 1, 1)

        loss_fn = CensoredPoissonLoss()

        # Low signal
        n_low = torch.full((2, 11, 11), 10.0)
        r_low = loss_fn(n_low, bg, is_sparse, counts, threshold, valid, gain)

        # High signal
        n_high = torch.full((2, 11, 11), 200.0)
        r_high = loss_fn(n_high, bg, is_sparse, counts, threshold, valid, gain)

        assert r_high.term2.item() > r_low.term2.item()

    def test_gain_scales_rate(self):
        """Changing gain affects the loss."""
        n_signal, bg, is_sparse, counts, threshold, valid, _ = _make_batch()
        loss_fn = CensoredPoissonLoss()

        gain_1 = torch.ones(4, 1, 1)
        gain_2 = torch.full((4, 1, 1), 2.0)

        r1 = loss_fn(n_signal, bg, is_sparse, counts, threshold, valid, gain_1)
        r2 = loss_fn(n_signal, bg, is_sparse, counts, threshold, valid, gain_2)

        assert r1.loss.item() != r2.loss.item()


# ──────────────────────── PixelIndexMapper ───────────────────────────────────


class TestPixelIndexMapper:
    def test_good_pixels_mapped(self):
        """Good pixels get sequential flat indices."""
        mask = torch.zeros(100, 200, dtype=torch.bool)
        mask[10:20, 50:60] = True  # 10x10 = 100 good pixels
        mapper = PixelIndexMapper(mask, torch.device("cpu"))
        assert mapper.n_good == 100

        ss = torch.tensor([[[15.0]]])
        fs = torch.tensor([[[55.0]]])
        flat_idx, on_good = mapper.grid_to_flat(ss, fs)
        assert on_good.item() is True
        assert flat_idx.item() >= 0

    def test_bad_pixels_masked(self):
        """Bad pixels return on_good=False."""
        mask = torch.zeros(100, 200, dtype=torch.bool)
        mask[10:20, 50:60] = True
        mapper = PixelIndexMapper(mask, torch.device("cpu"))

        # Query a bad pixel
        ss = torch.tensor([[[5.0]]])
        fs = torch.tensor([[[5.0]]])
        flat_idx, on_good = mapper.grid_to_flat(ss, fs)
        assert on_good.item() is False

    def test_out_of_bounds(self):
        """Out-of-bounds pixels return on_good=False."""
        mask = torch.ones(50, 50, dtype=torch.bool)
        mapper = PixelIndexMapper(mask, torch.device("cpu"))

        ss = torch.tensor([[[-1.0, 55.0]]])
        fs = torch.tensor([[[0.0, 0.0]]])
        _, on_good = mapper.grid_to_flat(ss, fs)
        assert not on_good[0, 0, 0].item()
        assert not on_good[0, 0, 1].item()

    def test_batch_grid(self):
        """Works with (B, ph, pw) shaped grids."""
        mask = torch.ones(100, 100, dtype=torch.bool)
        mapper = PixelIndexMapper(mask, torch.device("cpu"))

        B, ph, pw = 3, 11, 11
        ss = torch.randint(0, 100, (B, ph, pw)).float()
        fs = torch.randint(0, 100, (B, ph, pw)).float()
        flat_idx, on_good = mapper.grid_to_flat(ss, fs)
        assert flat_idx.shape == (B, ph, pw)
        assert on_good.shape == (B, ph, pw)
        assert on_good.all()

    def test_sequential_indexing(self):
        """Flat indices are sequential within the good region."""
        mask = torch.zeros(10, 10, dtype=torch.bool)
        mask[0, 0] = True  # index 0
        mask[0, 1] = True  # index 1
        mask[1, 0] = True  # index 2
        mapper = PixelIndexMapper(mask, torch.device("cpu"))
        assert mapper.n_good == 3

        ss = torch.tensor([[[0.0, 0.0, 1.0]]])
        fs = torch.tensor([[[0.0, 1.0, 0.0]]])
        flat_idx, on_good = mapper.grid_to_flat(ss, fs)
        assert on_good.all()
        assert flat_idx[0, 0, 0].item() == 0
        assert flat_idx[0, 0, 1].item() == 1
        assert flat_idx[0, 0, 2].item() == 2


# ──────────────────────── Gradient checks ────────────────────────────────────


class TestGradientChecks:
    @pytest.mark.slow
    def test_gradcheck_term1(self):
        """torch.autograd.gradcheck on term1 through a scale parameter."""
        def func(scale):
            n_signal = torch.tensor([[[100.0]]]) * scale
            bg = torch.tensor([[[50.0]]])
            is_sparse = torch.ones(1, 1, 1, dtype=torch.bool)
            counts = torch.tensor([[[150]]], dtype=torch.int32)
            threshold = torch.tensor([[[80.0]]])
            valid = torch.ones(1, 1, 1, dtype=torch.bool)
            gain = torch.ones(1, 1, 1)
            loss_fn = CensoredPoissonLoss()
            result = loss_fn(n_signal, bg, is_sparse, counts, threshold, valid, gain)
            return result.term1

        scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        # Use double precision for gradcheck
        torch.autograd.gradcheck(func, (scale,), eps=1e-6, atol=1e-4)

    @pytest.mark.slow
    def test_gradcheck_term2(self):
        """torch.autograd.gradcheck on term2 through a scale parameter."""
        def func(scale):
            n_signal = torch.tensor([[[100.0]]]).double() * scale
            bg = torch.tensor([[[50.0]]]).double()
            is_sparse = torch.zeros(1, 1, 1, dtype=torch.bool)  # censored
            counts = torch.zeros(1, 1, 1, dtype=torch.int32)
            threshold = torch.tensor([[[80.0]]]).double()
            valid = torch.ones(1, 1, 1, dtype=torch.bool)
            gain = torch.ones(1, 1, 1).double()
            loss_fn = CensoredPoissonLoss()
            result = loss_fn(n_signal, bg, is_sparse, counts, threshold, valid, gain)
            return result.term2

        scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(func, (scale,), eps=1e-6, atol=1e-4)
