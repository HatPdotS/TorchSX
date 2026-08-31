"""Tests for ProfileFittedLoss."""

import torch
import pytest

from torchsx.deprecated.forward._profile_fitted import ProfileFittedLoss, ProfileFittedResult


def _make_batch(B=10, ph=11, pw=11, device="cpu"):
    """Create synthetic sparse batch tensors."""
    background = torch.full((B, ph, pw), 5.0, device=device)
    gain = torch.ones(B, 1, 1, device=device)
    valid = torch.ones(B, ph, pw, dtype=torch.bool, device=device)
    threshold = torch.full((B, ph, pw), 10.0, device=device)
    # No sparse pixels by default (all censored)
    is_sparse = torch.zeros(B, ph, pw, dtype=torch.bool, device=device)
    sparse_counts = torch.zeros(B, ph, pw, dtype=torch.int32, device=device)
    return background, gain, valid, threshold, is_sparse, sparse_counts


class TestProfileFittedLoss:
    def test_basic_output_shape(self):
        loss_fn = ProfileFittedLoss(sigma_kernel=1.5)
        bg, gain, valid, thr, is_sp, sp_c = _make_batch()
        n_signal = torch.zeros(10, 11, 11)

        result = loss_fn(n_signal, bg, is_sp, sp_c, thr, valid, gain)

        assert isinstance(result, ProfileFittedResult)
        assert result.loss.shape == ()
        assert result.n_obs == 10
        assert result.val_loss is None

    def test_zero_signal_zero_residual(self):
        """No signal, no sparse pixels → residual = 0, loss ≈ 0."""
        loss_fn = ProfileFittedLoss(sigma_kernel=1.5)
        bg, gain, valid, thr, is_sp, sp_c = _make_batch()
        n_signal = torch.zeros(10, 11, 11)

        result = loss_fn(n_signal, bg, is_sp, sp_c, thr, valid, gain)
        assert result.loss.item() < 1e-10

    def test_signal_with_matching_sparse(self):
        """When sparse counts match predicted signal + bg, loss is small."""
        loss_fn = ProfileFittedLoss(sigma_kernel=1.5)
        bg, gain, valid, thr, is_sp, sp_c = _make_batch(B=5)
        n_signal = torch.zeros(5, 11, 11)
        # Put signal at center, sparse pixels match
        n_signal[:, 5, 5] = 100.0
        is_sp[:, 4:7, 4:7] = True
        sp_c[:, 4:7, 4:7] = (bg[:, 4:7, 4:7] + n_signal[:, 4:7, 4:7]).int()

        result = loss_fn(n_signal, bg, is_sp, sp_c, thr, valid, gain)
        # Loss should be small since observed ≈ predicted
        assert result.loss.item() < 1.0

    def test_signal_mismatch_high_loss(self):
        """When predicted signal doesn't match observed, loss is high."""
        loss_fn = ProfileFittedLoss(sigma_kernel=1.5)
        bg, gain, valid, thr, is_sp, sp_c = _make_batch(B=5)
        n_signal = torch.zeros(5, 11, 11)
        # Observed bright spot at center, but model predicts nothing
        is_sp[:, 4:7, 4:7] = True
        sp_c[:, 5, 5] = 100  # bright
        sp_c[:, 4:7, 4:7] = torch.where(
            sp_c[:, 4:7, 4:7] > 0, sp_c[:, 4:7, 4:7],
            bg[:, 4:7, 4:7].int(),
        )

        result = loss_fn(n_signal, bg, is_sp, sp_c, thr, valid, gain)
        assert result.loss.item() > 10.0

    def test_gradient_flows_to_signal(self):
        """Gradient flows through to n_signal."""
        loss_fn = ProfileFittedLoss(sigma_kernel=1.5)
        bg, gain, valid, thr, is_sp, sp_c = _make_batch(B=5)

        n_signal = torch.zeros(5, 11, 11, requires_grad=True)
        is_sp[:, 5, 5] = True
        sp_c[:, 5, 5] = 50

        result = loss_fn(n_signal, bg, is_sp, sp_c, thr, valid, gain)
        result.loss.backward()

        assert n_signal.grad is not None
        # Center pixel should have strongest gradient
        center_grad = n_signal.grad[:, 5, 5].abs().mean().item()
        edge_grad = n_signal.grad[:, 0, 0].abs().mean().item()
        assert center_grad > edge_grad

    def test_gradient_through_scale_parameter(self):
        """F² gradient via a simple scale parameter."""
        loss_fn = ProfileFittedLoss(sigma_kernel=1.5)
        bg, gain, valid, thr, is_sp, sp_c = _make_batch(B=5)

        log_scale = torch.tensor(0.0, requires_grad=True)
        base_signal = torch.zeros(5, 11, 11)
        base_signal[:, 5, 5] = 1.0

        is_sp[:, 4:7, 4:7] = True
        sp_c[:, 5, 5] = 50

        n_signal = torch.exp(log_scale) * base_signal
        result = loss_fn(n_signal, bg, is_sp, sp_c, thr, valid, gain)
        result.loss.backward()

        assert log_scale.grad is not None
        assert log_scale.grad.abs().item() > 0.01

    def test_val_holdout(self):
        """Validation holdout produces val_loss."""
        loss_fn = ProfileFittedLoss(sigma_kernel=1.5, val_frac=0.2)
        bg, gain, valid, thr, is_sp, sp_c = _make_batch(B=50)
        n_signal = torch.zeros(50, 11, 11)

        result = loss_fn(n_signal, bg, is_sp, sp_c, thr, valid, gain)

        assert result.val_loss is not None
        assert result.n_val > 0
        assert result.n_obs + result.n_val == 50

    def test_snr_diagnostic(self):
        """SNR increases with stronger signal."""
        loss_fn = ProfileFittedLoss(sigma_kernel=1.5)
        bg, gain, valid, thr, is_sp, sp_c = _make_batch(B=10)
        is_sp[:, 5, 5] = True

        # Weak signal
        sp_c_weak = sp_c.clone()
        sp_c_weak[:, 5, 5] = 10
        r_weak = loss_fn(torch.zeros(10, 11, 11), bg, is_sp, sp_c_weak,
                         thr, valid, gain)

        # Strong signal
        sp_c_strong = sp_c.clone()
        sp_c_strong[:, 5, 5] = 100
        r_strong = loss_fn(torch.zeros(10, 11, 11), bg, is_sp, sp_c_strong,
                           thr, valid, gain)

        assert r_strong.mean_snr > r_weak.mean_snr

    def test_kernel_cached(self):
        """Kernel is cached across calls."""
        loss_fn = ProfileFittedLoss(sigma_kernel=1.5)
        bg, gain, valid, thr, is_sp, sp_c = _make_batch()
        n_signal = torch.zeros(10, 11, 11)

        loss_fn(n_signal, bg, is_sp, sp_c, thr, valid, gain)
        k1 = loss_fn._kernel

        loss_fn(n_signal, bg, is_sp, sp_c, thr, valid, gain)
        k2 = loss_fn._kernel

        assert k1 is k2
