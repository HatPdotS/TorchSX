"""Tests for SoftAssignmentLoss."""

import torch
import pytest

from torchsx.deprecated.forward._soft_assignment import SoftAssignmentLoss, SoftAssignmentResult


def _make_batch(B=10, ph=11, pw=11, device="cpu"):
    background = torch.full((B, ph, pw), 5.0, device=device)
    gain = torch.ones(B, 1, 1, device=device)
    valid = torch.ones(B, ph, pw, dtype=torch.bool, device=device)
    threshold = torch.full((B, ph, pw), 10.0, device=device)
    is_sparse = torch.zeros(B, ph, pw, dtype=torch.bool, device=device)
    sparse_counts = torch.zeros(B, ph, pw, dtype=torch.int32, device=device)
    return background, gain, valid, threshold, is_sparse, sparse_counts


class TestSoftAssignment:
    def test_basic_output(self):
        loss_fn = SoftAssignmentLoss(sigma=3.0)
        bg, gain, valid, thr, is_sp, sp_c = _make_batch()
        n_signal = torch.zeros(10, 11, 11)
        result = loss_fn(n_signal, bg, is_sp, sp_c, thr, valid, gain)
        assert isinstance(result, SoftAssignmentResult)
        assert result.loss.shape == ()
        assert result.n_obs == 10

    def test_gradient_flows_through_signal(self):
        loss_fn = SoftAssignmentLoss(sigma=3.0)
        bg, gain, valid, thr, is_sp, sp_c = _make_batch(B=5)
        n_signal = torch.zeros(5, 11, 11, requires_grad=True)
        is_sp[:, 5, 5] = True
        sp_c[:, 5, 5] = 50
        result = loss_fn(n_signal, bg, is_sp, sp_c, thr, valid, gain)
        result.loss.backward()
        assert n_signal.grad is not None
        assert n_signal.grad.abs().sum().item() > 0

    def test_offset_signal_gets_gradient(self):
        """Predicted at center, observed offset — gradient at sparse pixel."""
        loss_fn = SoftAssignmentLoss(sigma=3.0)
        bg, gain, valid, thr, is_sp, sp_c = _make_batch(B=5)

        # Observed at (7, 7), model predicts at center (5, 5)
        is_sp[:, 7, 7] = True
        sp_c[:, 7, 7] = 50
        n_signal = torch.zeros(5, 11, 11, requires_grad=True)
        with torch.no_grad():
            n_signal[:, 5, 5] = 5.0

        result = loss_fn(n_signal, bg, is_sp, sp_c, thr, valid, gain)
        result.loss.backward()

        # Gradient should exist at the sparse pixel location
        sparse_grad = n_signal.grad[:, 7, 7].abs().mean().item()
        assert sparse_grad > 0

    def test_r_floor_ensures_gradient_with_zero_signal(self):
        """Even with zero predicted signal, r_floor ensures some gradient."""
        loss_fn = SoftAssignmentLoss(sigma=3.0, r_floor=0.05)
        bg, gain, valid, thr, is_sp, sp_c = _make_batch(B=5)
        is_sp[:, 5, 5] = True
        sp_c[:, 5, 5] = 50

        n_signal = torch.zeros(5, 11, 11, requires_grad=True)
        result = loss_fn(n_signal, bg, is_sp, sp_c, thr, valid, gain)
        result.loss.backward()

        # Should have gradient even though n_signal is zero everywhere
        total_grad = n_signal.grad.abs().sum().item()
        assert total_grad > 0

    def test_gradient_through_scale(self):
        """F² proxy: gradient flows through a scale parameter."""
        loss_fn = SoftAssignmentLoss(sigma=3.0)
        bg, gain, valid, thr, is_sp, sp_c = _make_batch(B=5)
        is_sp[:, 5, 5] = True
        sp_c[:, 5, 5] = 50

        log_scale = torch.tensor(0.0, requires_grad=True)
        base_signal = torch.zeros(5, 11, 11)
        base_signal[:, 5, 5] = 1.0
        n_signal = torch.exp(log_scale) * base_signal

        result = loss_fn(n_signal, bg, is_sp, sp_c, thr, valid, gain)
        result.loss.backward()
        assert log_scale.grad is not None
        assert log_scale.grad.abs().item() > 0.01

    def test_center_weighted_more_than_edge(self):
        """Distance weighting: center pixels contribute more."""
        loss_fn = SoftAssignmentLoss(sigma=2.0, r_floor=1.0)  # high r_floor to isolate distance effect
        bg, gain, valid, thr, is_sp, sp_c = _make_batch(B=5)

        # Identical sparse pixels at center vs edge
        n_signal = torch.ones(5, 11, 11) * 5.0

        # Test 1: sparse at center
        is_sp1 = is_sp.clone()
        is_sp1[:, 5, 5] = True
        sp_c1 = sp_c.clone()
        sp_c1[:, 5, 5] = 100
        r1 = loss_fn(n_signal, bg, is_sp1, sp_c1, thr, valid, gain)

        # Test 2: sparse at edge
        is_sp2 = is_sp.clone()
        is_sp2[:, 0, 0] = True
        sp_c2 = sp_c.clone()
        sp_c2[:, 0, 0] = 100
        r2 = loss_fn(n_signal, bg, is_sp2, sp_c2, thr, valid, gain)

        # Center sparse pixel should have higher weight
        assert r1.mean_weight > r2.mean_weight

    def test_val_holdout(self):
        loss_fn = SoftAssignmentLoss(sigma=3.0, val_frac=0.2)
        bg, gain, valid, thr, is_sp, sp_c = _make_batch(B=50)
        n_signal = torch.zeros(50, 11, 11)
        result = loss_fn(n_signal, bg, is_sp, sp_c, thr, valid, gain)
        assert result.val_loss is not None
        assert result.n_val > 0
        assert result.n_obs + result.n_val == 50

    def test_set_sigma_annealing(self):
        """Changing sigma invalidates cache and changes weights."""
        loss_fn = SoftAssignmentLoss(sigma=5.0)
        bg, gain, valid, thr, is_sp, sp_c = _make_batch(B=5)
        n_signal = torch.ones(5, 11, 11)
        is_sp[:, 0, 0] = True
        sp_c[:, 0, 0] = 50

        r_wide = loss_fn(n_signal, bg, is_sp, sp_c, thr, valid, gain)

        loss_fn.set_sigma(1.0)
        r_narrow = loss_fn(n_signal, bg, is_sp, sp_c, thr, valid, gain)

        # With narrow sigma, edge pixel weight should be much lower
        assert r_narrow.mean_weight < r_wide.mean_weight

    def test_responsibility_diagnostic(self):
        """Higher signal → higher responsibility."""
        loss_fn = SoftAssignmentLoss(sigma=3.0)
        bg, gain, valid, thr, is_sp, sp_c = _make_batch(B=5)
        is_sp[:, 5, 5] = True
        sp_c[:, 5, 5] = 50

        # Low signal
        n_low = torch.zeros(5, 11, 11)
        n_low[:, 5, 5] = 0.1
        r_low = loss_fn(n_low, bg, is_sp, sp_c, thr, valid, gain)

        # High signal
        n_high = torch.zeros(5, 11, 11)
        n_high[:, 5, 5] = 50.0
        r_high = loss_fn(n_high, bg, is_sp, sp_c, thr, valid, gain)

        assert r_high.mean_responsibility > r_low.mean_responsibility
