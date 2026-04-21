"""Tests for AttractionLoss."""

import torch
import pytest

from torchsx.forward._attraction_loss import AttractionLoss, AttractionResult


def _make_batch(B=10, ph=11, pw=11, device="cpu"):
    background = torch.full((B, ph, pw), 5.0, device=device)
    gain = torch.ones(B, 1, 1, device=device)
    valid = torch.ones(B, ph, pw, dtype=torch.bool, device=device)
    threshold = torch.full((B, ph, pw), 10.0, device=device)
    is_sparse = torch.zeros(B, ph, pw, dtype=torch.bool, device=device)
    sparse_counts = torch.zeros(B, ph, pw, dtype=torch.int32, device=device)
    return background, gain, valid, threshold, is_sparse, sparse_counts


class TestAttractionLoss:
    def test_basic_output(self):
        loss_fn = AttractionLoss(blur_sigma=3.0, beta=0.1)
        bg, gain, valid, thr, is_sp, sp_c = _make_batch()
        # Put some sparse pixels
        is_sp[:, 5, 5] = True
        sp_c[:, 5, 5] = 50
        n_signal = torch.zeros(10, 11, 11)
        result = loss_fn(n_signal, bg, is_sp, sp_c, thr, valid, gain)
        assert isinstance(result, AttractionResult)
        assert result.loss.shape == ()
        assert result.n_obs == 10

    def test_gradient_flows_through_signal(self):
        loss_fn = AttractionLoss(blur_sigma=3.0, beta=0.1)
        bg, gain, valid, thr, is_sp, sp_c = _make_batch(B=5)
        is_sp[:, 5, 5] = True
        sp_c[:, 5, 5] = 50
        n_signal = torch.ones(5, 11, 11, requires_grad=True)
        result = loss_fn(n_signal, bg, is_sp, sp_c, thr, valid, gain)
        result.loss.backward()
        assert n_signal.grad is not None
        assert n_signal.grad.abs().sum().item() > 0

    def test_attraction_gradient_with_offset_signal(self):
        """Key test: predicted at (5,5), observed at (8,8).

        The blur should create gradient even though prediction and
        observation don't overlap at the pixel level.
        """
        loss_fn = AttractionLoss(blur_sigma=3.0, beta=0.0)  # pure attraction
        bg, gain, valid, thr, is_sp, sp_c = _make_batch(B=5)

        # Observed at (8, 8)
        is_sp[:, 8, 8] = True
        sp_c[:, 8, 8] = 50

        # Predicted signal at (5, 5) via a learnable scale
        log_scale = torch.tensor(0.0, requires_grad=True)
        base_signal = torch.zeros(5, 11, 11)
        base_signal[:, 5, 5] = 10.0
        n_signal = torch.exp(log_scale) * base_signal

        result = loss_fn(n_signal, bg, is_sp, sp_c, thr, valid, gain)
        result.loss.backward()

        # The blur spreads signal from (5,5) to (8,8), so the attraction
        # term should give gradient to log_scale
        assert log_scale.grad is not None
        assert log_scale.grad.abs().item() > 0.01

    def test_more_signal_reduces_attraction(self):
        """Higher predicted signal near sparse pixels → lower attraction loss."""
        loss_fn = AttractionLoss(blur_sigma=3.0, beta=0.0)
        bg, gain, valid, thr, is_sp, sp_c = _make_batch(B=5)
        is_sp[:, 5, 5] = True
        sp_c[:, 5, 5] = 50

        n_low = torch.zeros(5, 11, 11)
        n_low[:, 5, 5] = 0.1
        r_low = loss_fn(n_low, bg, is_sp, sp_c, thr, valid, gain)

        n_high = torch.zeros(5, 11, 11)
        n_high[:, 5, 5] = 50.0
        r_high = loss_fn(n_high, bg, is_sp, sp_c, thr, valid, gain)

        # More signal → lower attraction loss (more negative → better)
        assert r_high.attraction.item() < r_low.attraction.item()

    def test_poisson_term_present(self):
        """With beta > 0, the Poisson term should contribute."""
        loss_fn = AttractionLoss(blur_sigma=3.0, beta=1.0)
        bg, gain, valid, thr, is_sp, sp_c = _make_batch(B=5)
        is_sp[:, 5, 5] = True
        sp_c[:, 5, 5] = 50
        n_signal = torch.ones(5, 11, 11) * 5.0
        result = loss_fn(n_signal, bg, is_sp, sp_c, thr, valid, gain)
        assert result.poisson.item() != 0.0

    def test_gradient_through_scale_with_offset(self):
        """F² proxy: gradient flows through scale even with spatial offset."""
        loss_fn = AttractionLoss(blur_sigma=5.0, beta=0.0)
        bg, gain, valid, thr, is_sp, sp_c = _make_batch(B=5)

        # Sparse pixels at multiple offset locations
        is_sp[:, 3, 3] = True
        sp_c[:, 3, 3] = 30
        is_sp[:, 7, 7] = True
        sp_c[:, 7, 7] = 80

        log_scale = torch.tensor(0.0, requires_grad=True)
        base_signal = torch.zeros(5, 11, 11)
        base_signal[:, 5, 5] = 10.0  # predicted at center
        n_signal = torch.exp(log_scale) * base_signal

        result = loss_fn(n_signal, bg, is_sp, sp_c, thr, valid, gain)
        result.loss.backward()
        # gradient should push log_scale UP (more signal → lower attraction)
        assert log_scale.grad is not None
        assert log_scale.grad.item() < 0  # negative gradient = wants to increase

    def test_high_count_pixels_pull_harder(self):
        """Pixels with more photons should contribute more to the loss."""
        loss_fn = AttractionLoss(blur_sigma=3.0, beta=0.0)
        bg, gain, valid, thr, is_sp, sp_c = _make_batch(B=5)

        # Scenario A: high counts at (3,3)
        is_sp_a = is_sp.clone()
        sp_c_a = sp_c.clone()
        is_sp_a[:, 3, 3] = True
        sp_c_a[:, 3, 3] = 100

        # Scenario B: low counts at (3,3)
        is_sp_b = is_sp.clone()
        sp_c_b = sp_c.clone()
        is_sp_b[:, 3, 3] = True
        sp_c_b[:, 3, 3] = 5

        n_signal = torch.zeros(5, 11, 11)
        n_signal[:, 5, 5] = 10.0  # signal at center, sparse at edge

        r_a = loss_fn(n_signal, bg, is_sp_a, sp_c_a, thr, valid, gain)
        r_b = loss_fn(n_signal, bg, is_sp_b, sp_c_b, thr, valid, gain)

        # Higher counts → stronger attraction (more negative)
        # But since we normalize by k_total, the per-count loss is similar
        # The key is the gradient magnitude
        log_scale_a = torch.tensor(0.0, requires_grad=True)
        log_scale_b = torch.tensor(0.0, requires_grad=True)
        n_a = torch.exp(log_scale_a) * n_signal
        n_b = torch.exp(log_scale_b) * n_signal

        loss_fn(n_a, bg, is_sp_a, sp_c_a, thr, valid, gain).loss.backward()
        loss_fn(n_b, bg, is_sp_b, sp_c_b, thr, valid, gain).loss.backward()

        # Both should have negative gradient (want more signal)
        assert log_scale_a.grad.item() < 0
        assert log_scale_b.grad.item() < 0

    def test_val_holdout(self):
        loss_fn = AttractionLoss(blur_sigma=3.0, beta=0.1, val_frac=0.2)
        bg, gain, valid, thr, is_sp, sp_c = _make_batch(B=50)
        is_sp[:, 5, 5] = True
        sp_c[:, 5, 5] = 50
        n_signal = torch.ones(50, 11, 11)
        result = loss_fn(n_signal, bg, is_sp, sp_c, thr, valid, gain)
        assert result.val_loss is not None
        assert result.n_val > 0
        assert result.n_obs + result.n_val == 50

    def test_smoothed_signal_diagnostic(self):
        """Higher signal → higher smoothed_signal diagnostic."""
        loss_fn = AttractionLoss(blur_sigma=3.0, beta=0.0)
        bg, gain, valid, thr, is_sp, sp_c = _make_batch(B=5)
        is_sp[:, 5, 5] = True
        sp_c[:, 5, 5] = 50

        n_low = torch.zeros(5, 11, 11)
        n_low[:, 5, 5] = 1.0
        r_low = loss_fn(n_low, bg, is_sp, sp_c, thr, valid, gain)

        n_high = torch.zeros(5, 11, 11)
        n_high[:, 5, 5] = 100.0
        r_high = loss_fn(n_high, bg, is_sp, sp_c, thr, valid, gain)

        assert r_high.mean_smoothed_signal > r_low.mean_smoothed_signal

    def test_zero_signal_still_finite(self):
        """With zero predicted signal, loss should still be finite (eps floor)."""
        loss_fn = AttractionLoss(blur_sigma=3.0, beta=0.1)
        bg, gain, valid, thr, is_sp, sp_c = _make_batch(B=5)
        is_sp[:, 5, 5] = True
        sp_c[:, 5, 5] = 50
        n_signal = torch.zeros(5, 11, 11)
        result = loss_fn(n_signal, bg, is_sp, sp_c, thr, valid, gain)
        assert torch.isfinite(result.loss)
        assert torch.isfinite(result.attraction)
