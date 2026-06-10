"""Automatic detector masking for the hit-finder.

The detector's intrinsic ``meta/pixel_mask`` (read by :meth:`JungfrauReader.mask_panels`)
already flags dead pixels and the inter-ASIC gap rows/columns, but it does **not**
cover two structural artifact sources that produce false hits:

* the **beamstop** — a deep central shadow plus a bright parasitic rim, and
* **module/ASIC edges** — charge-sharing pixels at panel borders and around gaps.

:func:`build_detector_mask` builds a combined good-pixel mask once per run on top of
that base mask:

1. **edge guard band** — dilate the existing bad/gap region by ``edge_dilate`` px
   (per panel, so modules never bleed into each other) and zero the outer
   ``module_border`` rows/cols of every module;
2. **beamstop** — union of (a) a *shadow floor* (pixels whose multi-frame mean
   collapses to a small fraction of the typical level — the beamstop core), (b) a
   *bright/shell anomaly* (pixels whose mean departs from the smooth radial
   background by > ``anomaly_nsigma`` σ — the parasitic rim and hot pixels), and
   (c) a *persistent-peak* pass (pixels that fire as a peak in > ``persist_frac`` of
   the sampled frames — real Bragg spots are random per shot, artifacts repeat).

Semantics are **True = good** throughout, matching the rest of the pipeline. The
result can be saved to / loaded from a sidecar h5 (:func:`save_mask`/:func:`load_mask`),
and an externally-supplied mask (tensor or h5/npy/pt file) can be folded in or used
wholesale via :func:`load_external_mask`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ._schema import FS, PANEL, SS
from .frames import JF_PANEL_SS
from .peakfinder import TorchPeakFinderParams, _radial_stats, find_spots_torch

_EPS = 1e-6


# ──────────────────────────────── helpers ───────────────────────────────────

def _dilate_bad(good: torch.Tensor, iters: int) -> torch.Tensor:
    """Grow the *bad* region of a ``(P, ss, fs)`` good-mask by ``iters`` px.

    A 3×3 max-pool on the bad map, applied per panel plane (zero padding, so panels
    never bleed into each other). Returns the eroded good-mask.
    """
    if iters <= 0:
        return good
    P, n_ss, n_fs = good.shape
    bad = (~good).to(torch.float32).view(P, 1, n_ss, n_fs)
    for _ in range(iters):
        bad = F.max_pool2d(bad, kernel_size=3, stride=1, padding=1)
    return good & (bad.view(P, n_ss, n_fs) < 0.5)


def _frame_mean(reader, frame_indices, device) -> torch.Tensor:
    """Per-pixel mean over ``frame_indices`` as a ``(P, 514, fs)`` float32 tensor."""
    from .noise_model import _load_panels  # GPU bslz4 decode w/ CPU fallback

    idx = [int(i) for i in frame_indices]
    acc = None
    for i in idx:
        fr = _load_panels(reader, i, device).to(torch.float64)
        acc = fr if acc is None else acc + fr
    return (acc / max(1, len(idx))).to(torch.float32)


# ───────────────────────────── external mask I/O ────────────────────────────

def load_external_mask(
    src,
    *,
    n_panels: int,
    det_fs: int,
    det_ss: int = JF_PANEL_SS,
    dataset: str | None = None,
    semantics: str = "auto",
) -> torch.Tensor:
    """Load an externally-supplied detector mask, normalized to ``(P, 514, fs)`` bool.

    Parameters
    ----------
    src : torch.Tensor | np.ndarray | str | Path
        A mask array (already ``(P, 514, fs)`` or assembled ``(P·514, fs)``), or a
        path to a ``.h5``/``.hdf5``, ``.npy``, or ``.pt``/``.pth`` file.
    n_panels, det_fs : int
        Expected panel count and fast-scan width (for shape validation/reshape).
    dataset : str, optional
        For an ``.h5`` source: the dataset to read. Defaults to trying ``good``,
        then ``meta/pixel_mask``, then the single 2-D/3-D dataset in the file.
    semantics : {"auto", "good", "bad"}
        Whether ``True`` means good (kept) or bad (masked). ``"auto"`` infers from the
        True-fraction (a detector good-mask is mostly True) and is logged.

    Returns
    -------
    torch.Tensor — ``(n_panels, 514, det_fs)`` bool, **True = good**.
    """
    if isinstance(src, torch.Tensor):
        arr = src.detach().cpu().numpy()
    elif isinstance(src, np.ndarray):
        arr = src
    else:
        path = Path(src)
        suf = path.suffix.lower()
        if suf in (".h5", ".hdf5"):
            import h5py

            with h5py.File(str(path), "r") as f:
                key = dataset
                if key is None:
                    for cand in ("good", "meta/pixel_mask", "mask", "pixel_mask"):
                        if cand in f:
                            key = cand
                            break
                if key is None:  # fall back to the single array dataset
                    arrs = []
                    f.visititems(lambda _n, o: arrs.append(o.name)
                                 if isinstance(o, h5py.Dataset) and o.ndim in (2, 3)
                                 else None)
                    if len(arrs) != 1:
                        raise ValueError(
                            f"{path}: specify dataset= (candidates: {arrs})")
                    key = arrs[0]
                arr = np.asarray(f[key][:])
        elif suf == ".npy":
            arr = np.load(str(path))
        elif suf in (".pt", ".pth"):
            arr = torch.load(str(path), map_location="cpu")
            arr = arr.detach().cpu().numpy() if isinstance(arr, torch.Tensor) else np.asarray(arr)
        else:
            raise ValueError(f"unsupported mask file type: {path.suffix}")

    arr = np.asarray(arr)
    # Normalize to a (P, ss, fs) boolean array.
    if arr.ndim == 3 and arr.shape == (n_panels, det_ss, det_fs):
        pass
    elif arr.ndim == 2 and arr.shape == (n_panels * det_ss, det_fs):
        arr = arr.reshape(n_panels, det_ss, det_fs)
    else:
        raise ValueError(
            f"external mask shape {arr.shape} is not ({n_panels},{det_ss},{det_fs}) "
            f"or ({n_panels * det_ss},{det_fs})")

    good = arr.astype(bool)
    true_frac = float(good.mean()) if good.size else 1.0
    if semantics == "bad" or (semantics == "auto" and true_frac < 0.5):
        # mostly-False → it's a bad-flag mask; invert so True = good
        good = ~good
        print(f"[mask] external mask: interpreted True=bad (True frac {true_frac:.3f}); inverted")
    return torch.from_numpy(np.ascontiguousarray(good))


# ───────────────────────────── sidecar I/O ──────────────────────────────────

def save_mask(mask: torch.Tensor, path: str | Path, *, attrs: dict | None = None) -> None:
    """Save a ``(P, 514, fs)`` bool good-mask to an h5 sidecar (dataset ``good``)."""
    import h5py

    m = mask.detach().cpu().to(torch.bool).numpy()
    with h5py.File(str(path), "w") as f:
        d = f.create_dataset("good", data=m, compression="gzip")
        d.attrs["semantics"] = "True=good"
        for k, v in (attrs or {}).items():
            try:
                f.attrs[k] = v
            except (TypeError, ValueError):
                f.attrs[k] = str(v)


def load_mask(path: str | Path) -> torch.Tensor:
    """Load a good-mask saved by :func:`save_mask` → ``(P, 514, fs)`` bool tensor."""
    import h5py

    with h5py.File(str(path), "r") as f:
        return torch.from_numpy(np.asarray(f["good"][:], dtype=bool))


def default_mask_path(source_path: str | Path) -> str:
    """Sidecar path ``<stem>.mask.h5`` next to a stream/source path."""
    p = Path(source_path)
    return str(p.with_suffix("").with_suffix(".mask.h5")) if p.suffix else str(p) + ".mask.h5"


# ──────────────────────────── mask construction ─────────────────────────────

def build_detector_mask(
    reader,
    geom,
    frame_indices,
    *,
    external=None,
    edge_dilate: int = 1,
    module_border: int = 1,
    beamstop: bool = True,
    shadow_frac: float = 0.30,
    anomaly_nsigma: float = 6.0,
    persistent: bool = True,
    persist_frac: float = 0.30,
    finder_params: TorchPeakFinderParams | None = None,
    device: str = "cpu",
    verbose: bool = True,
) -> torch.Tensor:
    """Build a combined good-pixel mask ``(P, 514, fs)`` (True = good).

    See the module docstring for the stages. ``frame_indices`` is the sample used for
    the data-driven beamstop/persistent-peak passes (reuse the calibration sample).
    No calibration is required, so this is a clean precondition for the whole run.
    """
    dev = torch.device(device)
    good = reader.mask_panels().to(torch.bool).to(dev)
    P, n_ss, n_fs = good.shape
    if external is not None:
        ext = load_external_mask(external, n_panels=P, det_fs=n_fs, det_ss=n_ss).to(dev)
        good = good & ext

    def _log(stage, before):
        removed = int((before & ~good).sum())
        if verbose:
            print(f"[mask] {stage:18s} removed {removed:8d} px "
                  f"({100.0 * removed / good.numel():.3f}%), good now {float(good.float().mean()) * 100:.2f}%")

    n_start = good.clone()

    # ── 1. edge guard band ───────────────────────────────────────────────────
    before = good.clone()
    good = _dilate_bad(good, edge_dilate)
    if module_border > 0:
        b = module_border
        good[:, :b, :] = False
        good[:, n_ss - b:, :] = False
        good[:, :, :b] = False
        good[:, :, n_fs - b:] = False
    _log("edge guard band", before)

    if beamstop or persistent:
        r = geom.pixel_radius_px.reshape(P, n_ss, n_fs).to(torch.float32).to(dev)
        mean = _frame_mean(reader, frame_indices, dev)  # (P,ss,fs) float32

    # ── 2. beamstop: central shadow disk + shell anomaly ─────────────────────
    if beamstop:
        # (a) beamstop core — the *contiguous central* region whose radial background
        # is suppressed well below the detector's normal level. Walk shells outward
        # from the beam center and mask everything inside the recovery radius as a
        # disk. (A global low-count floor is wrong here: the assembled detector has
        # large dead high-radius corners that are low for reasons unrelated to the
        # beamstop — using a central-contiguous walk isolates the actual beamstop.)
        before = good.clone()
        rb = r.reshape(-1).round().long().clamp(min=0)
        gflat = good.reshape(-1).to(torch.float32)
        nbin = int(rb.max().item()) + 1 if rb.numel() else 1
        cnt = torch.zeros(nbin, device=dev).scatter_add_(0, rb, gflat)
        ssum = torch.zeros(nbin, device=dev).scatter_add_(0, rb, mean.reshape(-1) * gflat)
        shell_mean = ssum / cnt.clamp(min=1.0)
        # reference background = peak shell-mean over well-populated shells (the
        # illuminated mid-radius band; robust to both central suppression and the
        # far-corner falloff). Threshold a fraction of it.
        well = cnt >= 200
        ref = float(shell_mean[well].max()) if bool(well.any()) else float(shell_mean.max())
        thr = shadow_frac * ref
        r_bs = 0.0
        for b in range(nbin):
            if cnt[b] < 1:                       # empty shell (gap) — keep walking
                continue
            if shell_mean[b] < thr:
                r_bs = float(b)
            else:
                break
        if r_bs > 0:
            good = good & ~(r <= r_bs)
        if verbose:
            print(f"[mask]   beamstop disk r<{r_bs:.0f} px (thr {thr:.2f} = "
                  f"{shadow_frac:.0%} of bg {ref:.1f})")
        _log("beamstop core", before)

        # (b) shell anomaly — departure from the smooth radial background of the mean
        # image (both bright rim and any residual shadow). Shell stats reject positive
        # outliers internally, so the spread is a clean per-radius noise estimate.
        before = good.clone()
        mean_bg, sigma_bg, _thr = _radial_stats(
            mean.reshape(-1), good.reshape(-1), r.reshape(-1),
            n_iters=5, min_snr=anomaly_nsigma, adc_thresh=0.0, bin_width=1.0)
        mean_bg = mean_bg.view(P, n_ss, n_fs)
        sigma_bg = sigma_bg.view(P, n_ss, n_fs).clamp(min=_EPS)
        anom = good & ((mean - mean_bg).abs() > anomaly_nsigma * sigma_bg)
        good = good & ~anom
        _log("beamstop anomaly", before)

    # ── 3. persistent-peak pass ──────────────────────────────────────────────
    if persistent:
        before = good.clone()
        params = finder_params or TorchPeakFinderParams()
        n_frames = 0
        hits = torch.zeros(P * n_ss * n_fs, dtype=torch.float32, device=dev)
        base = reader.mask_panels().to(torch.bool).to(dev)  # base mask for the finder
        for i in frame_indices:
            from .noise_model import _load_panels
            panels = _load_panels(reader, int(i), dev)
            spots = find_spots_torch(panels, base, r, params)
            n_frames += 1
            if spots.numel() == 0:
                continue
            panel = spots[:, PANEL].long().clamp(0, P - 1)
            loc_ss = (spots[:, SS] - panel.to(torch.float32) * n_ss).round().long().clamp(0, n_ss - 1)
            loc_fs = spots[:, FS].round().long().clamp(0, n_fs - 1)
            flat = (panel * n_ss + loc_ss) * n_fs + loc_fs
            hits.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
        hitmap = hits.view(P, n_ss, n_fs)
        persist = hitmap > (persist_frac * max(1, n_frames))
        persist = ~_dilate_bad(~persist, 1)  # dilate the persistent-bad region by 1 px
        good = good & ~persist
        if verbose:
            print(f"[mask]   persistent-peak: >{persist_frac:.0%} of {n_frames} frames")
        _log("persistent peak", before)

    _log("TOTAL added", n_start)
    return good
