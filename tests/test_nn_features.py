"""Tests for the multi-system reflection dataloader feature head.

Checks the structural feature contract (target fields cannot enter the scale
features), token/flat alignment, multi-system disjoint HKL blocks, the Wilson
prior wiring, and the 21×21 patch-channel plumbing.
"""

from __future__ import annotations

import pytest
import torch

from torchsx._core import _SFXStreamCore
from torchsx.stream import SFXStream
from torchsx.nn import ReflectionDataset, ScaleConsistencyNLL
from torchsx.nn.features import DEFAULT_SCALE_FEATURES, TARGET_FIELDS


def _synth_stream(*, n_cry=12, obsper=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    n_obs = n_cry * obsper
    hkl = torch.randint(-3, 4, (n_obs, 3), generator=g).to(torch.int16)
    I = torch.rand(n_obs, generator=g) * 100 + 1
    cid = torch.arange(n_cry).repeat_interleave(obsper).to(torch.int32)
    astar = torch.zeros(n_cry, 3); astar[:, 0] = 0.067
    bstar = torch.zeros(n_cry, 3); bstar[:, 1] = 0.053
    cstar = torch.zeros(n_cry, 3); cstar[:, 2] = 0.053
    core = _SFXStreamCore(
        hkl=hkl, I=I.float(), sigma=(I * 0.1).float(),
        background=torch.full((n_obs,), 0.5), peak=(I + 0.5).float(),
        fs=torch.rand(n_obs, generator=g) * 1000,
        ss=torch.rand(n_obs, generator=g) * 1000,
        panel_id=torch.randint(0, 2, (n_obs,), generator=g).to(torch.int16),
        crystal_id=cid, astar=astar, bstar=bstar, cstar=cstar,
        cell=torch.tensor([[14.97, 18.85, 18.89, 89.37, 84.94, 67.84]] * n_cry),
        profile_rad=torch.full((n_cry,), 0.004),
        resolution=torch.full((n_cry,), 2.57),
        n_refl=torch.full((n_cry,), obsper, dtype=torch.int32),
        image_id=torch.arange(n_cry, dtype=torch.int32),
        crystal_in_image=torch.zeros(n_cry, dtype=torch.int32),
        beam_energy=torch.full((n_cry,), 12960.0),
        det_shift_x=torch.zeros(n_cry), det_shift_y=torch.zeros(n_cry),
        n_peaks=torch.full((n_cry,), obsper, dtype=torch.int32),
        spacegroup="P 1", source_path="t",
    )
    return SFXStream(core)


def test_target_fields_rejected_from_scale_features():
    for tf in TARGET_FIELDS:
        with pytest.raises(ValueError):
            ReflectionDataset(_synth_stream(), scale_features=("s_sq", tf))


def test_default_features_exclude_targets():
    assert not (set(DEFAULT_SCALE_FEATURES) & set(TARGET_FIELDS))


def test_batch_shapes_and_mask():
    ds = ReflectionDataset(_synth_stream(n_cry=12, obsper=8), groups_per_batch=5)
    F = len(ds.scale_feature_names)
    seen = 0
    for b in ds:
        B, L = b.scale_features.shape[:2]
        assert b.scale_features.shape == (B, L, F)
        assert b.padding_mask.shape == (B, L)
        # mask True exactly where a group is shorter than L
        assert int((~b.padding_mask).sum()) == int(b.n_per_group.sum())
        # valid token rows are finite
        assert torch.isfinite(b.scale_features[~b.padding_mask]).all()
        seen += B
    assert seen == 12  # all crystals covered


def test_flat_token_alignment():
    """unpack(padded) == flat tensors in batch order."""
    ds = ReflectionDataset(_synth_stream(), groups_per_batch=4)
    for b in ds:
        # scale_features unpacked must match the per-obs feature rows
        flat_feat = b.unpack(b.scale_features)
        assert flat_feat.shape[0] == b.I_obs.shape[0]
        # the padded target I_obs, unpacked, must equal the flat I_obs
        assert torch.allclose(b.unpack(b.target["I_obs"]), b.I_obs)
        assert b.hkl_id.shape == b.I_obs.shape == b.crystal_id.shape


def test_multisystem_disjoint_hkl_blocks():
    ds = ReflectionDataset([_synth_stream(seed=0), _synth_stream(seed=1)],
                           groups_per_batch=6)
    n0 = ds.blocks[0].n_unique
    assert ds.n_hkl == n0 + ds.blocks[1].n_unique
    sys_hids = {0: set(), 1: set()}
    for b in ds:
        for s in (0, 1):
            sel = b.system_id == s
            if sel.any():
                sys_hids[s].update(b.hkl_id[sel].tolist())
    # system 0 uses [0, n0); system 1 uses [n0, n_hkl) — disjoint.
    assert max(sys_hids[0]) < n0
    assert min(sys_hids[1]) >= n0


def test_wilson_prior_and_mu_init():
    ds = ReflectionDataset([_synth_stream(seed=2), _synth_stream(seed=3)])
    mu = ds.mu_init()
    assert mu.shape == (ds.n_hkl,) and torch.isfinite(mu).all()
    prior = ds.wilson_prior()
    nlp = prior(mu)
    assert nlp.shape == (ds.n_hkl,) and torch.isfinite(nlp).all()


def test_loss_consumes_loader_batches():
    """End-to-end: a batch's flats feed the loss and produce a finite gradient."""
    ds = ReflectionDataset(_synth_stream(), groups_per_batch=6)
    m = ScaleConsistencyNLL(ds.n_hkl, mu_init=ds.mu_init(), prior=ds.wilson_prior(),
                            n_points=512, S_min=1e-2, S_max=1e5)
    b = next(iter(ds))
    M = b.I_obs.shape[0]
    lsm = torch.zeros(M, requires_grad=True)
    lss = torch.full((M,), -1.0)
    loss = m(lsm, lss, b.I_obs, b.sigma_det, b.hkl_id)
    loss.backward()
    assert torch.isfinite(loss)
    assert m.log_I0.grad is not None and torch.isfinite(m.log_I0.grad).all()


def test_patch_channel_plumbing():
    """Injected patches are packed to (B, L, h, w) aligned with the mask."""
    ds = ReflectionDataset(_synth_stream(n_cry=10, obsper=6), groups_per_batch=4)
    for blk in ds.blocks:
        blk.patches = torch.rand(blk.I_obs.shape[0], 5, 5)
    for b in ds:
        B, L = b.scale_features.shape[:2]
        assert b.patches.shape == (B, L, 5, 5)
        # PAD token patches are zero (pack zero-fills)
        assert (b.patches[b.padding_mask] == 0).all()


def test_include_hkl_requires_multisystem():
    with pytest.raises(ValueError):
        ReflectionDataset(_synth_stream(), include_hkl_in_features=True)
    ds = ReflectionDataset([_synth_stream(seed=0), _synth_stream(seed=1)],
                           include_hkl_in_features=True)
    assert ds.scale_feature_names[-3:] == ("h", "k", "l")
