"""The Ewald offset of a canonicalised (ASU) Miller index must be the offset of the reflection that was observed."""
import math
from types import SimpleNamespace

import pytest
import torch

from torchsx.merging.observations import _ewald_offset


def _core(spacegroup, n=4000, seed=0):
    """Random orientations, reflections placed ON the Ewald sphere, indices canonicalised to the ASU."""
    from torchref.symmetry import SpaceGroup
    g = torch.Generator().manual_seed(seed)
    energy = 12000.0; k0 = energy / 12398.4
    n_xtal = 40
    # reciprocal basis rows a*, b*, c* for a monoclinic-ish cell, randomly rotated per crystal
    B0 = torch.diag(torch.tensor([1 / 60.0, 1 / 90.0, 1 / 120.0]))
    rots = []
    for _ in range(n_xtal):
        q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g))
        rots.append(q * torch.sign(torch.det(q)))
    Rm = torch.stack(rots)
    Bc = B0[None] @ Rm                                             # (n_xtal, 3, 3)
    cid = torch.randint(0, n_xtal, (n,), generator=g)
    hkl = torch.randint(-12, 13, (n, 3), generator=g).float()
    hkl = hkl[(hkl != 0).any(dim=1)]; cid = cid[: hkl.shape[0]]
    q = torch.einsum("mi,mij->mj", hkl, Bc[cid])
    # rotate each crystal so that its reflections sit near the sphere: pick the sign giving the smaller offset and
    # then record the true offset from the un-canonicalised index (it is whatever geometry gives; the test is invariance)
    qz = torch.zeros_like(q); qz[:, 2] = k0
    off_raw = (q + qz).norm(dim=1) - k0
    off_flip = (qz - q).norm(dim=1) - k0
    true = torch.where(off_flip.abs() < off_raw.abs(), off_flip, off_raw)
    sg = SpaceGroup(spacegroup)
    canon, _, _, order = sg.canonicalize_hkl(hkl.int(), include_friedel=True)
    inv = torch.empty_like(order); inv[order] = torch.arange(order.numel()); canon = canon[inv]
    core = SimpleNamespace(hkl=canon, crystal_id=cid, astar=Bc[:, 0], bstar=Bc[:, 1], cstar=Bc[:, 2],
                           beam_energy=torch.full((n_xtal,), energy), friedel_merged=True, spacegroup=spacegroup)
    raw = SimpleNamespace(hkl=hkl.int(), crystal_id=cid, astar=Bc[:, 0], bstar=Bc[:, 1], cstar=Bc[:, 2],
                          beam_energy=torch.full((n_xtal,), energy), friedel_merged=True, spacegroup=spacegroup)
    return core, raw, true


@pytest.mark.parametrize("spacegroup", ["P 1", "P 1 21 1", "P 63", "P 21 21 21"])
def test_canonicalised_indices_recover_the_observed_offset(spacegroup):
    core, raw, true = _core(spacegroup)
    got = _ewald_offset(core)
    ref = _ewald_offset(raw)
    # the raw index with only the Friedel choice is the reference; the canonicalised index must give the same
    torch.testing.assert_close(got, ref, atol=1e-6, rtol=1e-5)
    # and it must be at most the smallest |offset| over the images, i.e. never worse than the raw index
    assert (got.abs() <= ref.abs() + 1e-6).all()
    # the pre-fix behaviour: canonical indices without image search are far off for non-P1 groups
    if spacegroup != "P 1":
        plain = SimpleNamespace(**{**vars(core), "spacegroup": "P 1"})
        naive = _ewald_offset(plain)
        assert (naive.abs() > got.abs() + 1e-3).float().mean() > 0.3


def test_unsymmetrised_stream_uses_the_observed_index_only():
    core, raw, _ = _core("P 63")
    raw.friedel_merged = False
    off = _ewald_offset(raw)
    qz_k0 = 12000.0 / 12398.4
    # identity only: recompute directly
    B = torch.stack([raw.astar, raw.bstar, raw.cstar], dim=1)[raw.crystal_id]
    q = torch.einsum("mi,mij->mj", raw.hkl.float(), B); qz = torch.zeros_like(q); qz[:, 2] = qz_k0
    torch.testing.assert_close(off, (q + qz).norm(dim=1) - qz_k0, atol=1e-6, rtol=1e-5)
