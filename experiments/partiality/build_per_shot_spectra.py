#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -p day
#SBATCH -t 12:00:00
#SBATCH -o /das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/partiality/build_per_shot_spectra.out

"""Build a per-crystal SASE spectrum lookup.

For every indexed crystal in the delay_*.stream files, look up its pulse_id
in the corresponding JF06 image file, then check whether a measured spectrum
exists in the BS spectrometer file for that run/acq.  Crystals with a matched
spectrum are kept; the spectrum is resampled onto a common energy axis and
stored in order alongside a matched .sfx subset of the stream.

Output (under experiments/partiality/results/per_shot_spectra/):
    delay_<d>_matched.sfx          – stream subset (matched crystals only)
    spectra_delay_<d>.pt           – tensor (M_d, N_E) in same crystal order
    per_shot_index.pt              – combined matched-crystal manifest

Spectra are normalised to sum=1 over the common energy grid before saving.
"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

from torchsx.stream import SFXStream

STREAM_DIR = Path("/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams")
OUTDIR = Path("/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/partiality/results/per_shot_spectra")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Common energy axis (must match build_spectrum_basis.py)
E0 = 12960.0
N_ENERGY_BINS = 256
DELTA_E_RANGE = 80.0
E_COMMON = np.linspace(E0 - DELTA_E_RANGE / 2, E0 + DELTA_E_RANGE / 2, N_ENERGY_BINS)

# BS spectrometer keys
BS_X = "SARFE10-PSSS059:SPECTRUM_X/data"
BS_Y = "SARFE10-PSSS059:SPECTRUM_Y/data"
BS_PID = "SARFE10-PSSS059:SPECTRUM_X/pulse_id"

JF_PID = "data/JF06T32V07/pulse_id"


# ── helpers ──────────────────────────────────────────────────────────────────

_event_re = re.compile(r"//(\d+)")


def parse_event(ev_str: str) -> int:
    m = _event_re.search(ev_str)
    if m is None:
        raise ValueError(f"unparseable event {ev_str!r}")
    return int(m.group(1))


def run_dir_for_image(image_path: str) -> str:
    """`/.../raw/run0122-.../data/acq0007.JF06T32V07.h5` → `/.../raw/run0122-.../data`."""
    return os.path.dirname(image_path)


def bs_path_for_image(image_path: str) -> str:
    """Map a JF image filename to the matching BSDATA file in the same run."""
    base = os.path.basename(image_path)
    acq = base.split(".")[0]                            # e.g. acq0007
    return os.path.join(run_dir_for_image(image_path), f"{acq}.BSDATA.h5")


def load_bs_lookup(bs_path: str) -> dict[int, np.ndarray] | None:
    """Return {pulse_id: resampled spectrum (N_ENERGY_BINS,)} or None on failure."""
    try:
        with h5py.File(bs_path, "r") as h:
            x = h[BS_X][:]                              # (N, 2560) energy axis
            y = h[BS_Y][:]                              # (N, 2560) intensity
            pid = h[BS_PID][:].flatten()
    except Exception as e:
        print(f"  [bs] skip {bs_path}: {e}", flush=True)
        return None

    # Filter to non-empty spectra
    valid = y.sum(axis=1) > 0
    if not valid.any():
        return None

    out: dict[int, np.ndarray] = {}
    for i in np.where(valid)[0]:
        spec = np.interp(E_COMMON, x[i], y[i], left=0.0, right=0.0)
        s = float(spec.sum())
        if s <= 0:
            continue
        out[int(pid[i])] = (spec / s).astype(np.float32)
    return out


def load_jf_pulse_ids(image_path: str) -> np.ndarray | None:
    try:
        with h5py.File(image_path, "r") as h:
            return h[JF_PID][:].flatten()
    except Exception as e:
        print(f"  [jf] skip {image_path}: {e}", flush=True)
        return None


def process_one_stream(stream_path: Path) -> dict | None:
    """Match every crystal in `stream_path` to a per-shot spectrum.

    Returns a dict with keys: stream, n_total, n_matched, matched_sfx, spectra_pt.
    """
    delay_label = stream_path.stem  # e.g. delay_-91
    cache_path = OUTDIR / f"{delay_label}_full.sfx"

    if cache_path.exists():
        s = SFXStream.load(cache_path)
    else:
        s = SFXStream.from_stream(stream_path, cache=cache_path)

    n_crystals = s.n_crystals
    if n_crystals == 0:
        return None

    image_id = s._core.image_id.numpy()                 # (n_crystals,)
    filenames = s.filenames                             # list per image
    events = s.events                                   # list per image

    # Group crystals by image
    crystals_per_image: dict[int, list[int]] = defaultdict(list)
    for c in range(n_crystals):
        crystals_per_image[int(image_id[c])].append(c)

    # Cache BS lookups by acq path so each BS file is read once
    bs_cache: dict[str, dict[int, np.ndarray] | None] = {}

    crystal_keep_mask = np.zeros(n_crystals, dtype=bool)
    crystal_spectrum = np.zeros((n_crystals, N_ENERGY_BINS), dtype=np.float32)

    n_images_processed = 0
    n_images_skipped = 0
    for img_idx, crystal_list in crystals_per_image.items():
        img_path = filenames[img_idx]
        bs_path = bs_path_for_image(img_path)

        if bs_path not in bs_cache:
            bs_cache[bs_path] = load_bs_lookup(bs_path)
        bs_lookup = bs_cache[bs_path]
        if bs_lookup is None or len(bs_lookup) == 0:
            n_images_skipped += 1
            continue

        pids = load_jf_pulse_ids(img_path)
        if pids is None:
            n_images_skipped += 1
            continue

        n_images_processed += 1
        ev = parse_event(events[img_idx])
        if not (0 <= ev < len(pids)):
            continue
        pid = int(pids[ev])
        spec = bs_lookup.get(pid)
        if spec is None:
            continue
        for c in crystal_list:
            crystal_keep_mask[c] = True
            crystal_spectrum[c] = spec

    n_matched = int(crystal_keep_mask.sum())
    print(
        f"  {delay_label}: {n_matched}/{n_crystals} crystals matched"
        f"  ({n_images_processed} images read, {n_images_skipped} skipped)",
        flush=True,
    )

    if n_matched == 0:
        return {
            "stream": str(stream_path),
            "n_total": n_crystals,
            "n_matched": 0,
            "matched_sfx": None,
            "spectra_pt": None,
        }

    # Filter and save subset
    keep_t = torch.from_numpy(crystal_keep_mask)
    s_matched = s.filter_crystals(keep_t)
    matched_path = OUTDIR / f"{delay_label}_matched.sfx"
    s_matched.save(matched_path)

    # Save spectra in the same crystal order as the filtered subset.
    # filter_crystals preserves crystal order — confirm and slice.
    matched_spec = crystal_spectrum[crystal_keep_mask]
    spectra_path = OUTDIR / f"spectra_{delay_label}.pt"
    torch.save(
        {
            "spectra": torch.from_numpy(matched_spec),    # (M, N_E), sum=1
            "e_axis_eV": torch.from_numpy(E_COMMON.astype(np.float32)),
            "delay_label": delay_label,
            "n_matched": int(matched_spec.shape[0]),
        },
        spectra_path,
    )

    return {
        "stream": str(stream_path),
        "n_total": int(n_crystals),
        "n_matched": int(matched_spec.shape[0]),
        "matched_sfx": str(matched_path),
        "spectra_pt": str(spectra_path),
    }


def main() -> None:
    stream_files = sorted(STREAM_DIR.glob("delay_*.stream"))
    print(f"Found {len(stream_files)} stream files in {STREAM_DIR}", flush=True)

    manifest: list[dict] = []
    n_tot = n_match = 0
    for sp in stream_files:
        try:
            entry = process_one_stream(sp)
        except Exception as e:
            print(f"  ! {sp.name}: {e}", flush=True)
            continue
        if entry is None:
            continue
        manifest.append(entry)
        n_tot += entry["n_total"]
        n_match += entry["n_matched"]

    torch.save(
        {
            "manifest": manifest,
            "e_axis_eV": torch.from_numpy(E_COMMON.astype(np.float32)),
            "n_total": n_tot,
            "n_matched": n_match,
        },
        OUTDIR / "per_shot_index.pt",
    )

    print()
    print("=" * 60, flush=True)
    print(f"TOTAL matched: {n_match}/{n_tot} crystals "
          f"({100 * n_match / max(n_tot, 1):.1f}%)", flush=True)
    print(f"Saved manifest to {OUTDIR / 'per_shot_index.pt'}", flush=True)


if __name__ == "__main__":
    main()
