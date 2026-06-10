"""Tests for maximum-likelihood lattice refinement (refine.py)."""

import math

import torch

from torchsx.datareduction.refine import (
    RefineResult,
    crystals_from_stream,
    lattice_log_likelihood,
    refine_geometry,
    refine_lattice,
)


def _rand_rotation(seed):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g))
    return q @ torch.diag(torch.sign(torch.diag(q)))      # proper rotation


def _make_crystal(seed=0, cell=(79.0, 79.0, 38.0), noise=5e-4, n_noise=30):
    """q-vectors from a known rotated cell + off-lattice noise spots."""
    torch.manual_seed(seed)
    B0 = torch.diag(torch.tensor(cell))                   # basis rows a,b,c
    R = _rand_rotation(seed)
    basis_true = (R @ B0.T).T                             # rotate real-space vectors
    UB = torch.linalg.inv(basis_true)                     # cols a*,b*,c*
    hk = torch.tensor([[h, k, l] for h in range(-5, 6) for k in range(-5, 6)
                       for l in range(-8, 9)], dtype=torch.float32)
    q = hk @ UB.T
    q = q[(q.norm(dim=1) < 0.35) & (q.norm(dim=1) > 0.02)]
    q = q + noise * torch.randn_like(q)
    n_lat = q.shape[0]
    # off-lattice noise
    qn = (torch.rand(n_noise, 3) - 0.5) * 0.5
    q_all = torch.cat([q, qn])
    is_lat = torch.cat([torch.ones(n_lat, dtype=torch.bool),
                        torch.zeros(n_noise, dtype=torch.bool)])
    return q_all, is_lat, basis_true


def _perturb(basis, deg=1.5, scale=0.005, seed=1):
    g = torch.Generator().manual_seed(seed)
    axis = torch.randn(3, generator=g)
    axis = axis / axis.norm() * math.radians(deg)
    K = torch.tensor([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
    R = torch.linalg.matrix_exp(K)
    return (R @ basis.T).T * (1.0 + scale)


def test_refine_recovers_orientation_and_cell():
    q, is_lat, basis_true = _make_crystal(seed=2)
    w = torch.ones(q.shape[0])
    seed = _perturb(basis_true, deg=1.5, scale=0.006)
    res = refine_lattice(q, w, seed, sigma_d=1.5e-3, pi=0.5, refine_cell=True)
    assert isinstance(res, RefineResult)
    cl = res.cell_lengths
    assert float(cl[0]) == __import__("pytest").approx(79.0, abs=1.5)
    assert float(cl[2]) == __import__("pytest").approx(38.0, abs=1.5)
    # lattice spots get high responsibility, noise spots low
    assert float(res.responsibilities[is_lat].median()) > 0.9
    assert float(res.responsibilities[~is_lat].median()) < 0.2
    assert res.n_inlier >= int(is_lat.sum()) * 0.8


def test_refine_improves_loglik_over_seed():
    q, _, basis_true = _make_crystal(seed=3)
    w = torch.ones(q.shape[0])
    seed = _perturb(basis_true, deg=1.5, scale=0.006, seed=4)
    # loglik at the seed
    UB0 = torch.linalg.inv(seed.to(torch.float64))
    qd = q.to(torch.float64)
    hkl0 = torch.round(torch.linalg.solve(UB0, qd.T).T)
    ll0, _, _ = lattice_log_likelihood(qd, w.to(torch.float64), UB0, hkl0,
                                       sigma_d=1.5e-3, pi=0.5, u_out=1.0)
    res = refine_lattice(q, w, seed, sigma_d=1.5e-3, pi=0.5)
    assert res.loglik > float(ll0)


def test_geometry_forward_differentiable():
    """q = to_reciprocal(...; dclen, dbeam) must backprop to the geometry params."""
    from torchsx.geometry import Geometry, Panel
    panel = Panel(name="p0", min_fs=0, max_fs=399, min_ss=0, max_ss=513,
                  corner_x=-200.0, corner_y=-256.0,
                  fs_vec=(1.0, 0.0, 0.0), ss_vec=(0.0, 1.0, 0.0), coffset=0.0)
    geom = Geometry(panels=[panel], photon_energy_eV=12960.0, clen=0.1, res=10000.0)
    ss = torch.linspace(20, 480, 50, dtype=torch.float64)
    fs = torch.linspace(20, 380, 50, dtype=torch.float64)
    dclen = torch.zeros((), dtype=torch.float64, requires_grad=True)
    dbeam = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    q = geom.to_reciprocal(ss, fs, dclen=dclen, dbeam=dbeam)
    q.pow(2).sum().backward()
    assert dclen.grad is not None and torch.isfinite(dclen.grad).all()
    assert dbeam.grad is not None and bool((dbeam.grad.abs() > 0).any())


def _build_two_crystal_stream():
    from torchsx.datareduction.peakfinder import PROB
    from torchsx.datareduction.indexing import IndexResult
    from torchsx.datareduction.to_stream import crystal_to_chunk
    from torchsx.datareduction.spotfinding import FS, INTENSITY, PANEL, SS
    from torchsx.geometry import Geometry, Panel
    from torchsx.io._stream_builder import StreamBuilder

    panel = Panel(name="p0", min_fs=0, max_fs=399, min_ss=0, max_ss=513,
                  corner_x=-200.0, corner_y=-256.0,
                  fs_vec=(1.0, 0.0, 0.0), ss_vec=(0.0, 1.0, 0.0), coffset=0.0)
    geom = Geometry(panels=[panel], photon_energy_eV=12960.0, clen=0.1, res=10000.0)
    builder = StreamBuilder.initialize(geometry=geom, spacegroup="P 43 21 2",
                                       async_writer=False)
    bases = []
    for ci, n in enumerate((12, 16)):
        b = torch.tensor([[79., 0, 0], [0, 79., 0], [0, 0, 38.]]) * (1.0 + 0.01 * ci)
        bases.append(b)
        sp = torch.zeros(n, 5)
        sp[:, SS] = torch.linspace(20, 480, n)
        sp[:, FS] = torch.linspace(20, 380, n)
        sp[:, INTENSITY] = torch.linspace(50, 500, n)
        sp[:, PROB] = torch.linspace(0.6, 1.0, n)
        res = IndexResult(success=True, basis=b,
                          inlier_mask=torch.ones(n, dtype=torch.bool),
                          hkl=torch.randint(-5, 6, (n, 3)), n_indexed=n, n_spots=n,
                          max_error=1e-3)
        builder.add(crystal_to_chunk(sp, res, geom, frame_key=("f.h5", ci)))
    from torchsx.stream import SFXStream
    return SFXStream(builder.finalize()), geom, bases


def test_stream_roundtrip_reconstructs_basis_and_weight():
    import pytest
    stream, geom, bases = _build_two_crystal_stream()
    crystals = crystals_from_stream(stream, geom)
    assert len(crystals) == 2
    for c, b in zip(crystals, bases):
        # basis0 reconstructed from astar/bstar/cstar matches the input basis
        assert torch.allclose(c["basis0"].to(torch.float32), b, atol=1e-3)
        assert c["w"].min() >= 0.6 - 1e-6 and c["w"].max() <= 1.0 + 1e-6
        assert c["ss"].numel() == c["w"].numel()


def test_refine_geometry_runs_no_false_drift():
    # crystals built at the nominal geometry → nominal should already be ~optimal,
    # so refine_geometry must not introduce a large spurious clen/beam shift.
    stream, geom, _ = _build_two_crystal_stream()
    crystals = crystals_from_stream(stream, geom)
    out = refine_geometry(crystals, geom, sigma_d=2e-3, n_rounds=2, max_iter=20)
    assert out["loglik"] >= out["loglik0"] - 1e-3       # never worsens
    assert abs(out["dclen"]) < 5e-3                      # < 5 mm drift
    assert max(abs(x) for x in out["dbeam"]) < 5e-3


def test_orientation_only_mode():
    q, is_lat, basis_true = _make_crystal(seed=5)
    w = torch.ones(q.shape[0])
    # perturb orientation only (no cell scale change), refine orientation only
    seed = _perturb(basis_true, deg=1.2, scale=0.0, seed=6)
    res = refine_lattice(q, w, seed, sigma_d=1.5e-3, pi=0.5, refine_cell=False)
    assert float(res.responsibilities[is_lat].median()) > 0.9
    cl = res.cell_lengths
    assert float(cl[0]) == __import__("pytest").approx(79.0, abs=0.5)  # cell unchanged
