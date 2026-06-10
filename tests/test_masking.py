"""Tests for automatic detector masking (beamstop + module edges + external masks)."""

import numpy as np
import torch

from torchsx.datareduction.masking import (
    build_detector_mask,
    load_external_mask,
    load_mask,
    save_mask,
)
from torchsx.datareduction.peakfinder import TorchPeakFinderParams
from torchsx.geometry import Geometry, Panel


def _geom(n_panels=1, ss=48, fs=48):
    """Geometry with ``n_panels`` stacked along slow-scan; beam center near (0,0)
    of the first panel so radius grows outward across the stack."""
    panels = []
    for p in range(n_panels):
        panels.append(Panel(
            name=f"m{p}", min_fs=0, max_fs=fs - 1,
            min_ss=p * ss, max_ss=(p + 1) * ss - 1,
            corner_x=-fs / 2, corner_y=-ss / 2 + p * ss,
            fs_vec=(1.0, 0.0, 0.0), ss_vec=(0.0, 1.0, 0.0), coffset=0.0))
    return Geometry(panels=panels, photon_energy_eV=12960.0, clen=0.1, res=10000.0)


class _FakeReader:
    def __init__(self, frames, mask):
        self._f = frames          # (n, P, ss, fs) float
        self._m = mask            # (P, ss, fs) bool

    @property
    def n_frames(self): return self._f.shape[0]
    @property
    def n_panels(self): return self._f.shape[1]
    def panels(self, i): return self._f[i]
    def mask_panels(self): return self._m


# ───────────────────────────── edge guard band ──────────────────────────────

def test_edge_dilate_grows_bad_region_per_panel():
    P, ss, fs = 2, 40, 40
    mask = torch.ones(P, ss, fs, dtype=torch.bool)
    mask[0, 20, 20] = False                       # one bad pixel in panel 0
    reader = _FakeReader(torch.ones(1, P, ss, fs) * 10.0, mask)
    good = build_detector_mask(reader, _geom(P, ss, fs), [0],
                               edge_dilate=1, module_border=0,
                               beamstop=False, persistent=False, verbose=False)
    # the bad pixel's 3x3 neighborhood is now bad
    assert not good[0, 19:22, 19:22].any()
    # panel 1 untouched (no bleed across panels)
    assert good[1].all()


def test_module_border_zeros_outer_rows_cols():
    P, ss, fs = 1, 40, 40
    reader = _FakeReader(torch.ones(1, P, ss, fs) * 10.0,
                         torch.ones(P, ss, fs, dtype=torch.bool))
    good = build_detector_mask(reader, _geom(P, ss, fs), [0],
                               edge_dilate=0, module_border=2,
                               beamstop=False, persistent=False, verbose=False)
    assert not good[0, :2, :].any() and not good[0, -2:, :].any()
    assert not good[0, :, :2].any() and not good[0, :, -2:].any()
    assert good[0, 2:-2, 2:-2].all()              # interior preserved


# ─────────────────────────────── beamstop ───────────────────────────────────

def test_beamstop_disk_and_persistent_bright_masked():
    P, ss, fs = 1, 80, 80
    geom = _geom(P, ss, fs)
    r = geom.pixel_radius_px.reshape(P, ss, fs)
    central = r < 15                                  # central beamstop disk
    n = 8
    torch.manual_seed(0)
    frames = 10.0 + 2.0 * torch.randn(n, P, ss, fs)  # noisy background (realistic shell σ)
    frames[:, central] = 0.0                          # suppressed beamstop core (all frames)
    frames[:, 0, 60, 60] = 5000.0                    # persistent bright pixel (anomaly high)
    reader = _FakeReader(frames, torch.ones(P, ss, fs, dtype=torch.bool))
    good = build_detector_mask(reader, geom, list(range(n)),
                               edge_dilate=0, module_border=0,
                               beamstop=True, persistent=False,
                               shadow_frac=0.3, anomaly_nsigma=6.0, verbose=False)
    assert not good[r < 13].any()                     # beamstop core disk masked
    assert not good[0, 60, 60]                        # persistent bright masked
    assert good[0, 40, 20]                            # quiet background pixel stays good


def test_persistent_peak_masks_repeating_spot_only():
    P, ss, fs = 1, 80, 80
    geom = _geom(P, ss, fs)
    n = 6
    frames = torch.full((n, P, ss, fs), 5.0)      # below adc_thresh background
    # a repeating 3x3 bright blob at the same place every frame → persistent
    frames[:, 0, 60:63, 60:63] = 2000.0
    # a one-frame bright blob elsewhere → transient
    frames[0, 0, 60:63, 15:18] = 2000.0
    reader = _FakeReader(frames, torch.ones(P, ss, fs, dtype=torch.bool))
    params = TorchPeakFinderParams(adc_thresh=150.0, min_snr=5.0)
    good = build_detector_mask(reader, geom, list(range(n)),
                               edge_dilate=0, module_border=0,
                               beamstop=False, persistent=True,
                               persist_frac=0.5, finder_params=params, verbose=False)
    assert not good[0, 61, 61]                     # repeating spot center masked
    assert good[0, 61, 16]                          # transient spot preserved


# ───────────────────────────── external mask I/O ────────────────────────────

def test_load_external_mask_tensor_and_assembled():
    P, ss, fs = 2, 40, 40
    m = torch.ones(P, ss, fs, dtype=torch.bool)
    m[1, 5, 5] = False
    out = load_external_mask(m, n_panels=P, det_fs=fs, det_ss=ss)
    assert out.shape == (P, ss, fs) and not out[1, 5, 5]
    # assembled (P*ss, fs) form reshapes back
    out2 = load_external_mask(m.reshape(P * ss, fs), n_panels=P, det_fs=fs, det_ss=ss)
    assert torch.equal(out, out2)


def test_load_external_mask_auto_inverts_bad_flag():
    P, ss, fs = 1, 20, 20
    bad = torch.zeros(P, ss, fs, dtype=torch.bool)  # True = bad, and they're the minority
    bad[0, :2, :] = True                             # 10% flagged bad → True-frac < 0.5
    out = load_external_mask(bad, n_panels=P, det_fs=fs, det_ss=ss, semantics="auto")
    # True-frac < 0.5 → interpreted as bad-flag → inverted so True = good
    assert out[0, 19, 0] and not out[0, 0, 0]


def test_load_external_mask_shape_mismatch_raises():
    try:
        load_external_mask(torch.ones(3, 7, dtype=torch.bool), n_panels=2, det_fs=40, det_ss=40)
    except ValueError:
        return
    raise AssertionError("expected ValueError on shape mismatch")


def test_external_mask_npy_and_h5_roundtrip(tmp_path):
    P, ss, fs = 1, 16, 16
    m = np.ones((P, ss, fs), dtype=bool)
    m[0, 3, 3] = False
    npy = tmp_path / "ext.npy"
    np.save(npy, m)
    out_npy = load_external_mask(str(npy), n_panels=P, det_fs=fs, det_ss=ss)
    assert not out_npy[0, 3, 3]

    h5p = tmp_path / "ext.mask.h5"
    save_mask(torch.from_numpy(m), h5p, attrs={"edge_dilate": 1})
    out_h5 = load_mask(h5p)
    assert torch.equal(out_h5, torch.from_numpy(m))
    out_ext = load_external_mask(str(h5p), n_panels=P, det_fs=fs, det_ss=ss)
    assert torch.equal(out_ext, torch.from_numpy(m))


def test_detector_mask_embedded_in_sfx_roundtrip(tmp_path):
    """A detector_mask set on a core is embedded in the .sfx and reloads (True=good)."""
    from torchsx._core import BaseStream, _sx_joins_with_peaks
    from torchsx.io._safetensors import load_sfx, save_sfx
    from torchsx.levels import Level

    P, ss, fs = 2, 40, 40
    image = Level({"beam_energy": torch.tensor([12000.0, 12000.0])})
    core = BaseStream._make({"image": image}, _sx_joins_with_peaks(),
                            None, None, "P 1", "")
    assert core.detector_mask is None
    mask = torch.ones(P, ss, fs, dtype=torch.bool)
    mask[0, 7, 7] = False
    core.detector_mask = mask

    out = tmp_path / "withmask.sfx"
    save_sfx(core, out)
    reloaded = load_sfx(out)
    assert reloaded.detector_mask is not None
    assert reloaded.detector_mask.shape == (P, ss, fs)
    assert reloaded.detector_mask.dtype == torch.bool
    assert torch.equal(reloaded.detector_mask, mask)

    # the public SFXStream.load path (replace/symmetrize rebuild the core) must
    # still surface the mask — regression guard for the attribute being dropped.
    from torchsx.stream import SFXStream
    s = SFXStream.load(out, merge_friedel=False)
    assert s.detector_mask is not None and torch.equal(s.detector_mask, mask)

    # a core without a mask stays None (and doesn't write the tensor)
    core2 = BaseStream._make({"image": image}, _sx_joins_with_peaks(),
                             None, None, "P 1", "")
    save_sfx(core2, tmp_path / "nomask.sfx")
    assert load_sfx(tmp_path / "nomask.sfx").detector_mask is None


def test_external_mask_anded_in_build():
    P, ss, fs = 1, 40, 40
    reader = _FakeReader(torch.ones(1, P, ss, fs) * 10.0,
                         torch.ones(P, ss, fs, dtype=torch.bool))
    ext = torch.ones(P, ss, fs, dtype=torch.bool)
    ext[0, 10, 10] = False
    good = build_detector_mask(reader, _geom(P, ss, fs), [0], external=ext,
                               edge_dilate=0, module_border=0,
                               beamstop=False, persistent=False, verbose=False)
    assert not good[0, 10, 10]                     # external bad pixel folded in
