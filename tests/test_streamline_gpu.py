"""Focused tests for the GPU-residency / indexer-reuse streamlining.

Cover the three structural changes that have observable, device-independent
invariants:

1. ``_get_indexer`` builds the TORO indexer once and reuses the same instance
   across calls (no per-frame 50k-lattice rebuild).
2. ``EmpiricalNull`` keeps ``z_sorted`` on its working device and ``.to`` migrates
   it; ``sf`` is device-consistent.
3. ``Geometry`` caches its panel tensors per device so ``to_lab`` reuses them.

CUDA-specific equivalence is asserted only when a GPU is present.
"""

from __future__ import annotations

import pytest
import torch

from torchsx.geometry import Geometry, Panel


def _geom():
    panel = Panel(name="p0", min_fs=0, max_fs=399, min_ss=0, max_ss=513,
                  corner_x=-200.0, corner_y=-256.0,
                  fs_vec=(1.0, 0.0, 0.0), ss_vec=(0.0, 1.0, 0.0), coffset=0.0)
    return Geometry(panels=[panel], photon_energy_eV=12960.0, clen=0.1, res=10000.0)


# ── Item 1: indexer reuse ────────────────────────────────────────────────────
def _require_toro():
    """Put the TORO clone on sys.path (via _load_toro) and skip if unavailable."""
    from torchsx.datareduction.indexing import _load_toro
    try:
        _load_toro()
    except ImportError as e:
        pytest.skip(f"TORO clone unavailable: {e}")


def test_get_indexer_reuses_instance():
    _require_toro()
    from torchsx.datareduction.indexing import _get_indexer

    a = _get_indexer(False, "cpu", 2000, 5, 0.0012, 0.00075, 6)
    b = _get_indexer(False, "cpu", 2000, 5, 0.0012, 0.00075, 6)
    assert a is b                                   # cache hit, no rebuild
    # weighted is a distinct instance (separate key)
    w = _get_indexer(True, "cpu", 2000, 5, 0.0012, 0.00075, 6)
    assert w is not a
    # the frame-independent sphere lattice buffer is shared across the reuse
    assert a.unite_sphere_lattice.shape[0] == 2000


def test_index_spots_uses_cached_indexer(monkeypatch):
    """Two index_spots calls hit the cached factory rather than rebuilding."""
    _require_toro()
    from torchsx.datareduction import indexing

    calls = {"n": 0}
    orig = indexing._get_indexer.__wrapped__   # underlying (uncached) function

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    # wrap with a fresh cache so the counter sees only this test's calls
    import functools
    monkeypatch.setattr(indexing, "_get_indexer", functools.lru_cache(maxsize=None)(counting))

    torch.manual_seed(0)
    q = torch.randn(20, 3) * 0.1
    for _ in range(3):
        indexing.index_spots(q, device="cpu", min_num_spots=5)
    assert calls["n"] == 1                          # built once, reused twice


# ── Item 3: EmpiricalNull on device ──────────────────────────────────────────
def test_empirical_null_to_and_sf_consistent():
    from torchsx.datareduction.noise_model import EmpiricalNull

    z_sorted = torch.linspace(-2.0, 6.0, 1000)
    null = EmpiricalNull(z_sorted=z_sorted, n_lmax_per_frame=500.0)
    z = torch.tensor([-1.0, 0.0, 3.0, 5.5, 10.0])
    sf_cpu = null.sf(z)

    # .to('cpu') yields an EmpiricalNull with identical survival values
    null_cpu = null.to("cpu")
    assert isinstance(null_cpu, EmpiricalNull)
    assert null_cpu.z_sorted.device == torch.device("cpu")
    assert torch.equal(null_cpu.sf(z), sf_cpu)
    # survival function is monotone non-increasing and in [0, 1]
    assert bool((sf_cpu[:-1] >= sf_cpu[1:]).all())
    assert float(sf_cpu.min()) >= 0.0 and float(sf_cpu.max()) <= 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_empirical_null_sf_cuda_matches_cpu():
    from torchsx.datareduction.noise_model import EmpiricalNull

    z_sorted = torch.linspace(-2.0, 6.0, 1000)
    null = EmpiricalNull(z_sorted=z_sorted, n_lmax_per_frame=500.0)
    z = torch.tensor([-1.0, 0.0, 3.0, 5.5, 10.0])
    null_cuda = null.to("cuda")
    assert null_cuda.z_sorted.is_cuda
    out = null_cuda.sf(z.cuda())
    assert out.is_cuda
    assert torch.allclose(out.cpu(), null.sf(z))


# ── Item 5: per-device geometry cache ────────────────────────────────────────
def test_geometry_device_cache_reuses_tensors():
    geom = _geom()
    p0 = geom._panel_params_on("cpu")
    p1 = geom._panel_params_on("cpu")
    assert p0 is p1                                 # same cached dict, no re-.to()
    assert p0["fs_vecs"] is p1["fs_vecs"]
    m0 = geom._panel_id_map_on("cpu")
    assert geom._panel_id_map_on("cpu") is m0
    # to_lab works through the cache and matches a direct CPU transform
    ss = torch.tensor([10.0, 200.0, 500.0])
    fs = torch.tensor([5.0, 100.0, 380.0])
    xyz = geom.to_lab(ss, fs)
    assert xyz.shape == (3, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_geometry_to_lab_cuda_matches_cpu():
    geom = _geom()
    ss = torch.tensor([10.0, 200.0, 500.0])
    fs = torch.tensor([5.0, 100.0, 380.0])
    cpu = geom.to_lab(ss, fs)
    cuda = geom.to_lab(ss.cuda(), fs.cuda())
    assert cuda.is_cuda
    assert torch.allclose(cuda.cpu(), cpu, atol=1e-5)
    # device tensors are cached, not rebuilt
    assert geom._panel_params_on("cuda")["fs_vecs"] is geom._panel_params_on("cuda")["fs_vecs"]
