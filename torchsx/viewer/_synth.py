"""Synthetic dataset for the viewer — a JF-format ``.h5`` + a matching ``.sfx``.

Exercises the full viewer with no GPU and no real data: writes a small JUNGFRAU
HDF5 cube with planted Gaussian spots, and a torchsx ``SFXStream`` whose indexed
reflections sit exactly on those spots (one crystal per image, embedded geometry,
filenames pointing at the synthetic ``.h5``). Then::

    python -m torchsx.viewer._synth --out /tmp/synth
    torchsx-viewer --stream /tmp/synth.sfx     # geometry travels in the stream

The matched-filter overlay still computes live on CPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEFAULT_GEOM = "raw_data/16M_v1_distance_93.8.geom"


def _write_h5(path: Path, frames_u16, mask_good):
    """Write a JF06T32V07-layout HDF5 (assembled cube + pixel mask)."""
    import h5py

    H, W = frames_u16.shape[1], frames_u16.shape[2]
    with h5py.File(path, "w") as f:
        grp = f.create_group("data/JF06T32V07")
        grp.create_dataset("data", data=frames_u16.astype(np.int32))
        grp.create_dataset("meta/pixel_mask", data=mask_good.astype(np.uint8))


def generate(out: str | Path, n_frames: int = 4, geom_path: str = DEFAULT_GEOM,
             seed: int = 0):
    """Write ``<out>.h5`` and ``<out>.sfx``; returns the .sfx path."""
    import torch

    from torchsx._core import SFXStreamCore
    from torchsx.geometry import Geometry
    from torchsx.io._metadata import StreamMetadata
    from torchsx.stream import SFXStream

    out = Path(out)
    h5_path = out.with_suffix(".h5").resolve()
    sfx_path = out.with_suffix(".sfx")
    rng = np.random.default_rng(seed)

    geom = Geometry.from_geom_file(geom_path)
    P = geom.n_panels
    nss, nfs = 514, geom.detector_shape[1]
    Htot = P * nss

    # ── per-frame: background + planted spots; collect obs at spot centers ──
    frames = np.empty((n_frames, Htot, nfs), dtype=np.uint16)
    half = 5
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    obs = {k: [] for k in ("hkl", "I", "fs", "ss", "panel")}
    crystal_id = []
    n_peaks = []

    for i in range(n_frames):
        cube = rng.poisson(2.0, size=(P, nss, nfs)).astype(np.float32)
        m = int(rng.integers(40, 120))
        for _ in range(m):
            p = int(rng.integers(0, P))
            s = int(rng.integers(half, nss - half))
            f = int(rng.integers(half, nfs - half))
            amp = float(rng.uniform(200, 4000))
            cube[p, s - half:s + half + 1, f - half:f + half + 1] += \
                amp * np.exp(-((yy / 2.0) ** 2 + (xx / 1.0) ** 2))
            obs["panel"].append(p)
            obs["ss"].append(p * nss + s)          # assembled slow-scan
            obs["fs"].append(f)
            obs["I"].append(amp)
            obs["hkl"].append(rng.integers(-20, 21, size=3))
            crystal_id.append(i)                    # one crystal per image
        n_peaks.append(m)
        frames[i] = np.clip(cube, 0, 65535).astype(np.uint16).reshape(Htot, nfs)

    _write_h5(h5_path, frames, np.ones((Htot, nfs), dtype=np.uint8))

    # ── build the stream core ───────────────────────────────────────────────
    n_obs = len(crystal_id)
    t = lambda a, dt: torch.tensor(np.asarray(a), dtype=dt)
    cell6 = torch.tensor([79.1, 79.1, 38.0, 90.0, 90.0, 90.0], dtype=torch.float32)
    core = SFXStreamCore(
        hkl=t(obs["hkl"], torch.int16).reshape(n_obs, 3) if n_obs else torch.zeros(0, 3, dtype=torch.int16),
        I=t(obs["I"], torch.float32),
        sigma=torch.sqrt(t(obs["I"], torch.float32).clamp(min=1.0)),
        background=torch.full((n_obs,), 2.0),
        peak=t(obs["I"], torch.float32),
        fs=t(obs["fs"], torch.float32),
        ss=t(obs["ss"], torch.float32),
        panel_id=t(obs["panel"], torch.int16),
        crystal_id=t(crystal_id, torch.int32),
        astar=torch.tensor([[1 / 79.1, 0, 0]] * n_frames, dtype=torch.float32),
        bstar=torch.tensor([[0, 1 / 79.1, 0]] * n_frames, dtype=torch.float32),
        cstar=torch.tensor([[0, 0, 1 / 38.0]] * n_frames, dtype=torch.float32),
        cell=cell6.repeat(n_frames, 1),
        profile_rad=torch.full((n_frames,), 0.001),
        resolution=torch.full((n_frames,), 2.0),
        n_refl=t(n_peaks, torch.int32),
        image_id=torch.arange(n_frames, dtype=torch.int32),
        crystal_in_image=torch.zeros(n_frames, dtype=torch.int32),
        beam_energy=torch.full((n_frames,), float(geom.photon_energy_eV)),
        det_shift_x=torch.zeros(n_frames),
        det_shift_y=torch.zeros(n_frames),
        n_peaks=t(n_peaks, torch.int32),
        camera_length=torch.full((n_frames,), float(geom.clen) * 1000.0),
        geometry=geom,
        metadata=_metadata(StreamMetadata, str(h5_path), n_frames),
        spacegroup="P 43 21 2",
    )
    # Embedded good-pixel mask (True=good), mirroring real streams. Plant a small
    # masked block so the viewer's masking path (black-out + bracketed values) is
    # exercised; the .h5 pixel_mask above stays all-good (separate concern).
    mask = torch.ones((P, nss, nfs), dtype=torch.bool)
    mask[0, 10:30, 10:30] = False
    core.detector_mask = mask
    SFXStream(core).save(sfx_path)
    return sfx_path


def _metadata(StreamMetadata, h5_path: str, n: int):
    meta = StreamMetadata()
    meta["image"] = {"filename": [h5_path] * n, "event": [f"//{i}" for i in range(n)]}
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output basename (writes <out>.h5 and <out>.sfx)")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--geom", default=DEFAULT_GEOM)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sfx = generate(args.out, n_frames=args.frames, geom_path=args.geom, seed=args.seed)
    print(f"wrote {sfx} and {Path(args.out).with_suffix('.h5')}")


if __name__ == "__main__":
    main()
