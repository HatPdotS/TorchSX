"""Out-of-core preparation: inputs -> a :class:`~torchsx.merging.store.PreparedStore`.

The result is what reading each input with :meth:`SFXStream.read`, cutting it
with :func:`~torchsx.merging.hkl.filter.filter_resolution`, concatenating the
inputs and calling :func:`~torchsx.merging.observations.prepare_observations`
gives -- the same observations with the same values, crystal-major instead of
sorted by reflection -- but no step holds every observation in memory.

1. Each input is parsed in blocks of whole chunks and symmetrized.  Its
   observations go to two temporary files, the canonical Miller indices and the
   rest of each raw record.  Crystal and image tables stay in memory, with
   per-reflection counts of the observations: all, valid, positive, and
   invalid by reason.
2. Each input's resolution cut, at that input's median cell, is a property of
   the reflection, so the counts give the reflection index and the observation
   filter's totals directly.  Which crystals survive the cut takes one pass
   over the Miller indices alone.
3. With the crystal layout fixed (:meth:`StoreBuilder.set_layout`), the
   wavelength and the gain are global quantities of the tables; a calibrated
   gain takes one more pass for its sample.
4. A last pass writes the prepared records crystal by crystal, with the
   per-reflection sums of the unity merge and the plain mean.

The integer work of steps 1-2 (reflection counts, the cut) runs on the GPU
when there is one; every floating-point value is computed on the host, as
:func:`prepare_observations` computes it.
``TORCHSX_PREP_TIMING`` prints the time of each step.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from torchsx.merging.config import MergeConfig
from torchsx.merging.hkl import PoissonNoiseModel
from torchsx.merging.store import RECORD, RECORD_WIDTH, PreparedStore, RecordFile, StoreWriter

#: One parsed observation before preparation, less its canonical Miller index
#: (kept in a file of its own, the one column the resolution cut reads).
#: ``observed`` is the measured index; unrecorded ``n_pix`` is 0, ``bg_model*`` NaN.  44 bytes.
RAW = np.dtype([
    ("I", "<f4"), ("sigma", "<f4"), ("background", "<f4"), ("peak", "<f4"),
    ("fs", "<f4"), ("ss", "<f4"), ("observed", "<i2", (3,)), ("panel_id", "<i2"),
    ("n_pix", "<f4"), ("bg_model", "<f4"), ("bg_model_var", "<f4"),
])
HKL_BYTES = 6

#: Observations parsed and processed per block; bounds the builder's memory.
BLOCK_OBS = 4_000_000

_CRYSTAL = ("cell", "astar", "bstar", "cstar", "profile_rad", "resolution", "n_refl", "image_id")
_IMAGE = ("beam_energy", "det_shift_x", "det_shift_y", "n_peaks")
#: probixi's optional diagnostics, with the neutral fill for other sources.
_CRYSTAL_OPT = {"n_indexed": 0.0, "rmsd": 0.0, "mosaicity": 0.0, "enrichment": 1.0,
                "n_bright": 0.0, "enrich_p": 1.0}
_IMAGE_OPT = {"scale": 1.0, "scale_sigma": 0.0, "peak_resolution": 0.0}
#: Per-reflection observation counts collected while parsing.
_COUNTS = ("all", "kept", "positive", "I", "background", "sigma")


#: Marks a directory this module created for staging; only such a directory
#: (or an empty one) is ever deleted to make room for a new build.
STAGING_MARKER = ".torchsx-staging"


def staging_paths(out: str | Path) -> tuple[Path, Path]:
    """The staging directories of a store built at ``out``: the parsed inputs
    (``OUT.raw``) and the store being written (``OUT.partial``)."""
    out = Path(out)
    return out.with_name(out.name + ".raw"), out.with_name(out.name + ".partial")


def check_staging(path: str | Path) -> None:
    """Raise ``FileExistsError`` unless ``path`` is absent, an empty directory,
    or a staging directory left by an earlier build."""
    path = Path(path)
    if not path.exists() or (path / STAGING_MARKER).is_file():
        return
    if path.is_dir() and not any(path.iterdir()):
        return
    raise FileExistsError(f"{path} exists and was not created by TorchSX; move it away "
                          "or prepare into another directory")


def _fresh_staging(path: Path) -> Path:
    """``path`` as a new, marked staging directory, replacing only an earlier one."""
    check_staging(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    (path / STAGING_MARKER).touch()
    return path


def hkl_key(hkl: Tensor) -> Tensor:
    """One int64 per Miller index, ordered as ``(h, k, l)`` lexicographically."""
    h = hkl.to(torch.int64) + 32768
    return (h[:, 0] << 32) | (h[:, 1] << 16) | h[:, 2]


def key_hkl(key: Tensor) -> Tensor:
    return torch.stack([(key >> 32) - 32768, ((key >> 16) & 0xFFFF) - 32768,
                        (key & 0xFFFF) - 32768], dim=1).to(torch.int16)


def _field(records: np.ndarray, name: str) -> Tensor:
    """One field of packed records as a (strided, uncopied) tensor."""
    return torch.from_numpy(records[name])


class _Clock:
    """Wall time per step, printed with ``TORCHSX_PREP_TIMING``."""

    def __init__(self):
        self.on = bool(os.environ.get("TORCHSX_PREP_TIMING"))
        self.t = time.perf_counter()
        self.acc: dict[str, float] = {}

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.acc[name] = self.acc.get(name, 0.0) + now - self.t
        self.t = now

    def report(self, title: str) -> None:
        if self.on:
            for name, s in self.acc.items():
                print(f"  [prep] {title}: {name:<24s} {s:8.1f} s", flush=True)
        self.acc = {}
        self.t = time.perf_counter()


class _KeyCounts:
    """Sorted reflection keys of one input and the :data:`_COUNTS` of each."""

    def __init__(self, device="cpu"):
        self.keys = torch.zeros(0, dtype=torch.int64, device=device)
        self.counts = torch.zeros(0, len(_COUNTS), dtype=torch.int64, device=device)

    def to(self, device) -> "_KeyCounts":
        self.keys, self.counts = self.keys.to(device), self.counts.to(device)
        return self

    def index(self, key: Tensor) -> Tensor:
        """Position of each key, adding the keys not seen before."""
        if len(self.keys):
            idx = torch.searchsorted(self.keys, key).clamp(max=len(self.keys) - 1)
            if bool((self.keys[idx] == key).all()):
                return idx
        keys = torch.unique(torch.cat([self.keys, key]), sorted=True)
        counts = torch.zeros(len(keys), len(_COUNTS), dtype=torch.int64, device=keys.device)
        counts[torch.searchsorted(keys, self.keys)] = self.counts
        self.keys, self.counts = keys, counts
        return torch.searchsorted(self.keys, key)

    def add(self, idx: Tensor, masks: dict[str, Tensor | None]) -> None:
        self.counts += _count(idx, masks, len(self.keys))


def _count(idx: Tensor, masks: dict[str, Tensor | None], n_keys: int) -> Tensor:
    """``(n_keys, len(_COUNTS))``: per key, the rows in each mask (None: every row)."""
    out = torch.empty(n_keys, len(_COUNTS), dtype=torch.int64, device=idx.device)
    for j, name in enumerate(_COUNTS):
        w = masks[name]
        out[:, j] = torch.bincount(idx, weights=None if w is None else w.to(torch.float64),
                                   minlength=n_keys).to(torch.int64)
    return out


def _validity(I: Tensor, background: Tensor, sigma: Tensor | None):
    """Per-reason invalid masks and the valid mask of :func:`prepare_observations`."""
    bad = {"I": ~torch.isfinite(I), "background": ~torch.isfinite(background)}
    if sigma is not None:
        bad["sigma"] = ~torch.isfinite(sigma) | (sigma <= 0)
    valid = torch.ones_like(bad["I"])
    for b in bad.values():
        valid &= ~b
    return bad, valid


@dataclass
class _Input:
    name: str
    path: Path
    crystals: tuple[int, int]            # raw crystal range
    images: tuple[int, int]              # raw image range
    refl: _KeyCounts = None              # every crystal of the input
    extra: Tensor = None                 # counts of the crystals prepared once more
    in_range: Tensor = None              # per key: within the resolution cut
    to_valid: Tensor = None              # per key: prepared reflection index, or -1
    on_device: tuple = None              # (keys, to_valid) on the builder's device
    n_crystals: int = 0                  # after the resolution cut
    n_obs: int = 0                       # after the resolution cut


@dataclass
class _Tables:
    cols: dict = field(default_factory=dict)

    def add(self, name, value):
        self.cols.setdefault(name, []).append(value)

    def cat(self, name):
        return torch.cat(self.cols[name]) if self.cols.get(name) else torch.zeros(0)


class _Skeleton:
    """Crystal and image tables of the prepared crystal list, for the
    crystal-level steps of :func:`prepare_observations`."""

    def __init__(self, crystal: dict, image: dict, geometry):
        self.crystal, self.image, self.geometry = crystal, image, geometry

    def __getattr__(self, name):
        for level in (self.__dict__["crystal"], self.__dict__["image"]):
            if name in level:
                return level[name]
        raise AttributeError(name)

    def broadcast(self, column, from_level, to_level="obs"):
        if (from_level, to_level) != ("image", "crystal"):
            raise ValueError("only image -> crystal")
        return self.image[column][self.crystal["image_id"].long()]


class StoreBuilder:
    """Builds a :class:`PreparedStore` in ``workdir`` without holding the observations.

    ``add`` every input, ``filter_resolution``, ``set_layout``, ``finish``.
    """

    def __init__(self, workdir: str | Path, *, spacegroup: str | None, merge_friedel: bool = True,
                 n_workers: int = 1, block_obs: int | None = None, verbose: bool = True,
                 disk_size: str = "nominal", background: str = "annulus"):
        self.workdir = _fresh_staging(Path(workdir))
        self.spacegroup = spacegroup
        self.merge_friedel = bool(merge_friedel)
        if disk_size not in ("nominal", "recorded"):
            raise ValueError(f"disk_size must be 'nominal' or 'recorded', got {disk_size!r}")
        #: see :attr:`MergeConfig.disk_size`; ``recorded`` also keeps clipped disks
        self.disk_size = disk_size
        if background not in ("annulus", "model"):
            raise ValueError(f"background must be 'annulus' or 'model', got {background!r}")
        #: see :attr:`MergeConfig.background`
        self.background = background
        #: Var[bg]/bg of a modelled background, set by finish
        self.bg_var_factor: float | None = None
        self.n_workers = max(1, int(n_workers))
        self.block_obs = int(BLOCK_OBS if block_obs is None else block_obs)
        self.verbose = verbose
        self.inputs: list[_Input] = []
        self._out_hkl = open(self.workdir / "hkl.bin", "wb", buffering=1 << 24)
        self._out_raw = open(self.workdir / "raw.bin", "wb", buffering=1 << 24)
        self._hkl: RecordFile | None = None
        self._raw: RecordFile | None = None
        self.n_raw = 0
        self._crys = _Tables()
        self._img = _Tables()
        self.filenames: list[str] = []
        self.events: list[str] = []
        self._has_names = True
        self.geometry = None
        self._specs = []
        self.has_sigma: bool | None = None
        #: where the integer steps run
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clock = _Clock()

    # ── pass 1: parse ───────────────────────────────────────────────────

    def _blocks(self, path: Path):
        from torchsx.stream import SFXStream

        separate = not self.merge_friedel
        if path.suffix == ".stream":
            from torchsx.io._parser import parse_stream_blocks
            if self.spacegroup is None:
                import warnings
                warnings.warn(
                    f"{path}: spacegroup not set, falling back to 'P 1'. Centric "
                    f"detection and symmetry equivalents will be incorrect for any "
                    f"non-triclinic crystal. Pass spacegroup='P 63' (or similar) explicitly.",
                    stacklevel=3,
                )
                self.spacegroup = "P 1"
            def canonical(core):
                # on the parse thread, while the caller writes the previous block
                sym = SFXStream(core).symmetrize(self.spacegroup, separate_friedel=separate)._core
                return core, sym.hkl

            for core, hkl in parse_stream_blocks(path, spacegroup=self.spacegroup,
                                                 n_workers=self.n_workers,
                                                 block_obs=self.block_obs, transform=canonical):
                self.clock.lap("parse + symmetrize (waited for)")
                yield core, hkl.to(self.device)
            return
        # A cache written by torchsx is read whole, then cut into blocks of crystals.
        from torchsx.merging.hkl.filter import filter_crystals
        s = SFXStream.read(path, spacegroup=self.spacegroup, merge_friedel=self.merge_friedel,
                           clipped=self.disk_size == "recorded")._core
        self.spacegroup = self.spacegroup or s.spacegroup
        if bool((s.crystal_id[1:] < s.crystal_id[:-1]).any()):
            raise ValueError(f"{path}: observations are not ordered by crystal")
        ends = torch.bincount(s.crystal_id.long(), minlength=s.n_crystals).cumsum(0)
        c0 = 0
        while c0 < s.n_crystals:
            base = int(ends[c0 - 1]) if c0 else 0
            c1 = max(c0 + 1, int(torch.searchsorted(ends, base + self.block_obs, right=True)))
            mask = torch.zeros(s.n_crystals, dtype=torch.bool)
            mask[c0:c1] = True
            block = filter_crystals(s, mask)
            yield block, block.hkl.to(self.device)
            c0 = c1

    def add(self, name: str, path: str | Path) -> None:
        """Parse one input into the raw files (pass 1).

        Each block arrives as the parsed core and its canonical Miller indices
        (on :attr:`device`); the counts are made where the indices are.
        """
        from torchsx.measurement import MeasurementSpec

        path = Path(path)
        c0, i0 = self._n_crystals(), len(self.filenames)
        refl, spec, first = None, None, True
        for core, canonical in self._blocks(path):
            cid = core.crystal_id.long()
            if bool((cid[1:] < cid[:-1]).any()):
                raise ValueError(f"{path}: observations are not ordered by crystal")
            if self.geometry is None:
                self.geometry = core.geometry
            if first:
                spec = MeasurementSpec.from_metadata(core.metadata)
                first = False
            sigma = getattr(core, "sigma", None)
            if self.has_sigma is None:
                self.has_sigma = sigma is not None
            elif self.has_sigma != (sigma is not None):
                raise ValueError(f"{path}: sigma is present in some inputs only")
            observed = core.levels["obs"].get("observed_hkl")
            if observed is None:
                raise ValueError(f"{path}: the measured Miller indices were not retained")
            n = core.n_observations
            hkl = canonical.to("cpu", torch.int16).contiguous()
            raw = torch.empty((n, RAW.itemsize // 4), dtype=torch.float32)
            for j, f in enumerate(("I", "sigma", "background", "peak", "fs", "ss")):
                v = sigma if f == "sigma" else getattr(core, f)
                raw[:, j] = 0.0 if v is None else v
            tail = raw[:, 6:8].view(torch.int16)
            tail[:, :3] = observed
            tail[:, 3] = core.panel_id
            n_pix = core.levels["obs"].get("n_pixels")
            raw[:, 8] = 0.0 if n_pix is None else n_pix.float()
            for j, f in enumerate(("background_model", "background_model_var"), start=9):
                v = core.levels["obs"].get(f)
                raw[:, j] = float("nan") if v is None else v.float()
            self.clock.lap("extract")
            self._out_hkl.write(memoryview(hkl.numpy()).cast("B"))
            self._out_raw.write(memoryview(raw.numpy()).cast("B"))
            self.n_raw += n
            self.clock.lap("write raw")

            dev = canonical.device
            I = core.I.to(dev)
            bad, valid = _validity(I, core.background.to(dev),
                                   None if sigma is None else sigma.to(dev))
            if refl is None:
                refl = _KeyCounts(dev)
            idx = refl.index(hkl_key(canonical))
            refl.add(idx, {"all": None, "kept": valid, "positive": valid & (I > 0),
                           "I": bad["I"], "background": bad["background"],
                           "sigma": bad.get("sigma", torch.zeros_like(valid))})
            self._crys.add("counts", torch.bincount(cid, minlength=core.n_crystals))
            for c in _CRYSTAL:
                v = getattr(core, c)
                if c == "image_id":
                    v = v.long() + len(self.filenames)
                self._crys.add(c, v)
            for c in _IMAGE:
                self._img.add(c, getattr(core, c))
            for level, tbl, opt in (("crystal", self._crys, _CRYSTAL_OPT),
                                    ("image", self._img, _IMAGE_OPT)):
                for c, fill in opt.items():
                    v = core.levels[level].get(c)
                    tbl.add(c, torch.full((core.levels[level].n,), fill) if v is None else v.float())
            meta = core.metadata.get("image", {}) if core.metadata else {}
            self._has_names &= bool(meta.get("filename")) and bool(meta.get("event"))
            self.filenames += list(meta.get("filename") or [""] * core.n_images)
            self.events += list(meta.get("event") or [""] * core.n_images)
            self.clock.lap("count")
        if first:
            raise ValueError(f"{path}: no observations")
        self._specs.append(spec)
        self.inputs.append(_Input(name, path, (c0, self._n_crystals()), (i0, len(self.filenames)),
                                  refl=refl.to("cpu")))
        self.clock.report(f"parse {path.name}")

    def _n_crystals(self) -> int:
        return int(sum(t.numel() for t in self._crys.cols.get("counts", [])))

    def _close_raw(self) -> None:
        if self._raw is None:
            self._out_hkl.close()
            self._out_raw.close()
            self._hkl = RecordFile(self.workdir / "hkl.bin", HKL_BYTES, self.n_raw)
            self._raw = RecordFile(self.workdir / "raw.bin", RAW.itemsize, self.n_raw)
            self.counts = self._crys.cat("counts")
            self.offsets = torch.zeros(len(self.counts) + 1, dtype=torch.int64)
            self.offsets[1:] = self.counts.cumsum(0)
            self.crystal = {c: self._crys.cat(c) for c in (*_CRYSTAL, *_CRYSTAL_OPT)}
            self.image = {c: self._img.cat(c) for c in (*_IMAGE, *_IMAGE_OPT)}
            self._crys = self._img = None
            self.input_of_crystal = torch.zeros(len(self.counts), dtype=torch.int64)
            for k, inp in enumerate(self.inputs):
                self.input_of_crystal[inp.crystals[0]:inp.crystals[1]] = k

    # ── reading the raw files ───────────────────────────────────────────

    def _crystal_blocks(self, crystals: Tensor, raw: bool = True):
        """``(crystals, hkl, raw)`` for consecutive groups of raw crystal ids, in
        the given order, each group from one input and about ``block_obs`` rows.
        ``raw`` is None when only the Miller indices are asked for."""
        counts = self.counts[crystals]
        inp = self.input_of_crystal[crystals]
        a, n = 0, len(crystals)
        cum = counts.cumsum(0)
        while a < n:
            base = int(cum[a - 1]) if a else 0
            b = min(n, max(a + 1, int(torch.searchsorted(cum, base + self.block_obs, right=True))))
            other = torch.nonzero(inp[a:b] != inp[a], as_tuple=True)[0]
            if len(other):
                b = a + int(other[0])
            sel = crystals[a:b]
            starts = self.offsets[sel].numpy()
            ends = self.offsets[sel + 1].numpy()
            m = int((ends - starts).sum())
            hkl = np.empty((m, 3), dtype=np.int16)
            self._hkl.read_ranges(starts, ends, memoryview(hkl).cast("B"))
            rec = None
            if raw:
                rec = np.empty(m, dtype=RAW)
                self._raw.read_ranges(starts, ends, memoryview(rec).cast("B"))
            self.clock.lap("read raw")
            yield sel, torch.from_numpy(hkl), rec
            a = b

    def _local(self, sel: Tensor) -> Tensor:
        """Row -> position of its crystal in ``sel``."""
        return torch.repeat_interleave(torch.arange(len(sel)), self.counts[sel])

    # ── pass 2: resolution cut ──────────────────────────────────────────

    def filter_resolution(self, d_min: float, d_max: float = 999.0) -> None:
        """Each input cut at ``[d_min, d_max]`` on its own median cell."""
        from torchref.base.reciprocal.hkl import get_d_spacing

        self._close_raw()
        self.d_min, self.d_max = float(d_min), float(d_max)
        self.survives = torch.zeros(len(self.counts), dtype=torch.bool)
        for inp in self.inputs:
            c0, c1 = inp.crystals
            median = self.crystal["cell"][c0:c1].median(dim=0).values
            d = get_d_spacing(key_hkl(inp.refl.keys).float(), median)
            inp.in_range = (d >= self.d_min) & (d <= self.d_max)
            inp.n_obs = int(inp.refl.counts[inp.in_range, 0].sum())
            inp.extra = torch.zeros_like(inp.refl.counts)
            keys, in_range = inp.refl.keys.to(self.device), inp.in_range.to(self.device)
            for sel, hkl, _ in self._crystal_blocks(torch.arange(c0, c1), raw=False):
                hit = in_range[torch.searchsorted(keys, hkl_key(hkl.to(self.device)))]
                local = self._local(sel).to(self.device)
                self.survives[sel] = (torch.bincount(local[hit], minlength=len(sel)) > 0).cpu()
                self.clock.lap("cut")
            inp.n_crystals = int(self.survives[c0:c1].sum())
        self.primary = torch.nonzero(self.survives, as_tuple=True)[0]
        image_hit = torch.zeros(len(self.filenames), dtype=torch.bool)
        image_hit[self.crystal["image_id"][self.primary]] = True
        self.primary_images = torch.nonzero(image_hit, as_tuple=True)[0]
        self.clock.report("resolution cut")

    # ── the prepared crystal list ───────────────────────────────────────

    @property
    def surviving_image_id(self) -> Tensor:
        """Per surviving crystal, its image among the surviving images."""
        remap = torch.full((len(self.filenames),), -1, dtype=torch.int64)
        remap[self.primary_images] = torch.arange(len(self.primary_images))
        return remap[self.crystal["image_id"][self.primary]]

    @property
    def surviving_filenames(self) -> list[str]:
        """Per surviving image; empty when an input carries no image names."""
        if not self._has_names:
            return []
        return [self.filenames[i] for i in self.primary_images.tolist()]

    @property
    def surviving_events(self) -> list[str]:
        if not self._has_names:
            return []
        return [self.events[i] for i in self.primary_images.tolist()]

    def set_layout(self, extra: list[Tensor] | None = None) -> None:
        """The prepared crystals: every surviving crystal, then once more those
        of each mask in ``extra`` (one bool per surviving crystal), as
        concatenating the stream with its ``filter_crystals`` copies would."""
        logical = [self.primary]
        images = [self.primary_images]
        for mask in extra or []:
            sel = self.primary[mask]
            logical.append(sel)
            images.append(torch.unique(self.crystal["image_id"][sel]))
            for blk, hkl, raw in self._crystal_blocks(sel):
                inp = self.inputs[int(self.input_of_crystal[blk[0]])]
                bad, valid = _validity(_field(raw, "I"), _field(raw, "background"),
                                       _field(raw, "sigma") if self.has_sigma else None)
                idx = torch.searchsorted(inp.refl.keys, hkl_key(hkl))
                masks = {"all": None, "kept": valid, "positive": valid & (_field(raw, "I") > 0),
                         "I": bad["I"], "background": bad["background"],
                         "sigma": bad.get("sigma", torch.zeros_like(valid))}
                inp.extra += _count(idx, masks, len(inp.refl.keys))
        self.logical = torch.cat(logical)
        self.logical_images = torch.cat(images)
        # image index of each prepared crystal within the prepared image list
        img_pos, n_img = [], 0
        for crys, imgs in zip(logical, images):
            remap = torch.full((len(self.filenames),), -1, dtype=torch.int64)
            remap[imgs] = torch.arange(len(imgs)) + n_img
            img_pos.append(remap[self.crystal["image_id"][crys]])
            n_img += len(imgs)
        self.logical_image_id = torch.cat(img_pos)
        self.clock.report("layout")

    def skeleton(self) -> _Skeleton:
        crystal = {c: v[self.logical] for c, v in self.crystal.items() if c != "image_id"}
        crystal["image_id"] = self.logical_image_id
        image = {c: v[self.logical_images] for c, v in self.image.items()}
        return _Skeleton(crystal, image, self.geometry)

    # ── passes 3-4: calibration sample, prepared records ────────────────

    def _logical_blocks(self):
        """``(first logical crystal, raw crystal ids, input, hkl, raw records)``."""
        a = 0
        for sel, hkl, raw in self._crystal_blocks(self.logical):
            yield a, sel, self.inputs[int(self.input_of_crystal[sel[0]])], hkl, raw
            a += len(sel)

    def _prepared_rows(self, inp: _Input, sel: Tensor, hkl: Tensor, raw: np.ndarray, n_valid: int):
        """``(rows, local, h)`` of the prepared observations of a block, crystal
        by crystal and by reflection within a crystal (stable): the raw row, its
        crystal's position in ``sel`` and its prepared reflection index."""
        dev = self.device
        if inp.on_device is None:
            inp.on_device = (inp.refl.keys.to(dev), inp.to_valid.to(dev))
        keys, to_valid = inp.on_device
        h = to_valid[torch.searchsorted(keys, hkl_key(hkl.to(dev)))]
        _, valid = _validity(_field(raw, "I").to(dev), _field(raw, "background").to(dev),
                             _field(raw, "sigma").to(dev) if self.has_sigma else None)
        rows = torch.nonzero((h >= 0) & valid, as_tuple=True)[0]
        local = self._local(sel).to(dev)[rows]
        h = h[rows]
        order = torch.sort(local * n_valid + h, stable=True).indices
        return rows[order].cpu(), local[order].cpu(), h[order].cpu()

    def _k0_rows(self, rc: Tensor, k0: float, per_image: bool):
        if not per_image:
            return k0
        energy = self.image["beam_energy"][self.crystal["image_id"][rc]].float()
        return 1.0 / (12398.4 / energy.clamp(min=1.0))

    def _calibration_sample(self, k0: float, per_image: bool = False):
        from torchsx.merging.hkl.noise import calibration_candidates
        from torchsx.merging.observations import ewald_rows

        parts = []
        for _, sel, inp, hkl, raw in self._logical_blocks():
            # the calibration sees every valid observation within the cut
            h = inp.in_range[torch.searchsorted(inp.refl.keys, hkl_key(hkl))]
            _, valid = _validity(_field(raw, "I"), _field(raw, "background"),
                                 _field(raw, "sigma") if self.has_sigma else None)
            rows = torch.nonzero(h & valid, as_tuple=True)[0]
            rc = sel[self._local(sel)[rows]]
            off, _ = ewald_rows(torch.from_numpy(raw["observed"])[rows].float(),
                                self.crystal["astar"][rc], self.crystal["bstar"][rc],
                                self.crystal["cstar"][rc], self._k0_rows(rc, k0, per_image))
            I = _field(raw, "I")[rows]
            background = _field(raw, "background")[rows]
            take = calibration_candidates(I, background, off)
            parts.append((I[take], background[take], off[take]))
            self.clock.lap("calibration sample")
        return tuple(torch.cat([p[i] for p in parts]) for i in range(3))

    def finish(self, out: str | Path, *, config: MergeConfig, noise: PoissonNoiseModel,
               resolved_measurement=None, delay_bin_idx: Tensor | None = None,
               unique_timepoints: Tensor | None = None, meta=None) -> PreparedStore:
        """Write the store to ``out`` and return it opened.  The raw files are removed.

        ``meta`` (a dict, or a callable returning one once the gain is resolved)
        is recorded in the store's ``meta.json``.
        """
        from torchsx.measurement import combine_measurements, validate_measurement
        from torchsx.merging.observations import (
            _crystal_feature_table, centric_flags, ewald_k0, ewald_rows, fw_shell_ids,
            resolve_gain, unit_rows,
        )
        from torchsx.nn.merge_target import add_plain_sums, add_unity_sums
        from torchsx.nn.obs_features import ObsFeatureSpec
        from torchref.base.reciprocal.hkl import get_d_spacing

        out = Path(out)
        if out.exists():
            raise FileExistsError(f"{out} exists")
        spec = combine_measurements(self._specs)
        if resolved_measurement is not None:
            validate_measurement(spec, explicit_rings=dict(resolved_measurement.provenance)["radii"] != "source")

        # ── reflections: per-input counts within the cut, joined ──
        keys, counts = [], []
        for inp in self.inputs:
            c = (inp.refl.counts + inp.extra)[inp.in_range]
            keys.append(inp.refl.keys[inp.in_range])
            counts.append(c)
        keys, inv = torch.unique(torch.cat(keys), sorted=True, return_inverse=True)
        total = torch.zeros(len(keys), len(_COUNTS), dtype=torch.int64)
        total.index_add_(0, inv, torch.cat(counts))
        col = {name: total[:, j] for j, name in enumerate(_COUNTS)}
        n_in_range = int(col["all"].sum())
        n_rejected = n_in_range - int(col["kept"].sum())
        if n_rejected:
            reasons = ", ".join(f"{k}={int(col[k].sum()):,}" for k in ("I", "background", "sigma")
                                if int(col[k].sum()))
            print(f"  observation filter: removed {n_rejected:,}/{n_in_range:,} "
                  f"invalid measurements ({reasons}; reason counts may overlap)", flush=True)
        seen = col["kept"] > 0
        if not bool(seen.any()):
            raise ValueError("No valid observations remain after measurement filtering")
        keys, kept_count, pos = keys[seen], col["kept"][seen], col["positive"][seen]

        sk = self.skeleton()
        n_crystals = len(self.logical)
        k0 = ewald_k0(sk.beam_energy)
        gain = resolve_gain(sk, spec, resolved_measurement, config, noise,
                            lambda: self._calibration_sample(k0, config.per_image_k0))
        self.clock.report("gain")

        unique_hkl = key_hkl(keys)
        median_cell = sk.cell.median(dim=0).values
        d_all = get_d_spacing(unique_hkl.float(), median_cell)
        valid = pos >= config.min_obs
        n_valid = int(valid.sum())
        if n_valid == 0:
            raise ValueError("No HKLs with sufficient positive observations")
        remap = torch.full((len(keys),), -1, dtype=torch.int64)
        remap[valid] = torch.arange(n_valid)
        hkl_valid = unique_hkl[valid]
        d_valid = d_all[valid]
        for inp in self.inputs:
            at = torch.searchsorted(keys, inp.refl.keys).clamp(max=len(keys) - 1)
            found = (keys[at] == inp.refl.keys) & inp.in_range
            inp.to_valid = torch.where(found, remap[at], torch.full_like(at, -1))

        # ── prepared records ──
        tmp = _fresh_staging(staging_paths(out)[1])
        writer = StoreWriter(tmp)
        n_obs_total = int(kept_count[valid].sum())
        if n_obs_total > ObsFeatureSpec.STAT_SAMPLE:
            writer.draw_sample(n_obs_total, ObsFeatureSpec.STAT_SAMPLE, 0)
        crystal_counts = torch.zeros(n_crystals, dtype=torch.int64)
        sums = {k: torch.zeros(n_valid, dtype=torch.float32) for k in ("wI", "w", "I", "n")}
        obs_counts = torch.zeros(n_valid, dtype=torch.int64)
        n_panels = 0
        var_sum = model_sum = 0.0
        for a, sel, inp, hkl, raw in self._logical_blocks():
            rows, local, h = self._prepared_rows(inp, sel, hkl, raw, n_valid)
            self.clock.lap("select + order")
            rc = sel[local]

            I_obs = _field(raw, "I")[rows] / gain
            n_pix = torch.full((len(rows),), float(noise.n_peak))
            if self.disk_size == "recorded":
                recorded = _field(raw, "n_pix")[rows]
                n_pix = torch.where(recorded > 0, recorded, n_pix)
            bg = _field(raw, "background")[rows] / gain * n_pix
            if self.background == "model":
                # keep the measured count I + bg; move only its background estimate
                model_adu = _field(raw, "bg_model")[rows]
                model = model_adu / gain
                use = torch.isfinite(model)
                I_obs = torch.where(use, I_obs + bg - model, I_obs)
                bg = torch.where(use, model, bg)
                var_sum += float(_field(raw, "bg_model_var")[rows][use].double().sum())
                model_sum += float(model_adu[use].double().sum())
            if self.has_sigma:
                sigma = _field(raw, "sigma")[rows] / gain
            else:
                sigma = ((I_obs + bg).clamp(min=0) + bg.clamp(min=0) + 1.0).sqrt()
            panel = _field(raw, "panel_id")[rows]
            off, q = ewald_rows(torch.from_numpy(raw["observed"])[rows].float(),
                                self.crystal["astar"][rc], self.crystal["bstar"][rc],
                                self.crystal["cstar"][rc],
                                self._k0_rows(rc, k0, config.per_image_k0))
            rec = torch.empty((len(rows), RECORD_WIDTH), dtype=torch.float32)
            ints = rec.view(torch.int32)
            rec[:, 0], rec[:, 1] = I_obs, bg
            rec[:, 2] = sigma if self.has_sigma else 0.0
            ints[:, 3], ints[:, 4] = h.to(torch.int32), panel.to(torch.int32)
            for j, f in enumerate(("fs", "ss", "peak", "background"), start=5):
                rec[:, j] = _field(raw, f)[rows]
            rec[:, 9] = off
            rec[:, 10:13] = unit_rows(q)
            rec[:, 13] = n_pix
            self.clock.lap("records")
            writer.write(rec.numpy().view(RECORD).reshape(-1))
            self.clock.lap("write store")

            add_unity_sums(sums["wI"], sums["w"], I_obs, sigma, h)
            add_plain_sums(sums["I"], sums["n"], I_obs, h)
            obs_counts += torch.bincount(h, minlength=n_valid)
            crystal_counts[a:a + len(sel)] += torch.bincount(local, minlength=len(sel))
            if len(rows):
                n_panels = max(n_panels, int(panel.max()) + 1)
            self.clock.lap("sums")
        if writer.n_obs != n_obs_total:
            raise RuntimeError(f"wrote {writer.n_obs} observations, counted {n_obs_total}")
        if model_sum > 0:
            self.bg_var_factor = var_sum / (gain * model_sum)  # Var[bg]/bg in photons

        offsets = torch.zeros(n_crystals + 1, dtype=torch.int64)
        offsets[1:] = crystal_counts.cumsum(0)
        tables = {
            "crystal_offsets": offsets,
            "crystal_feat": _crystal_feature_table(sk, n_crystals),
            "crystal_res": sk.resolution.float(),
            "crystal_beam_energy": sk.broadcast("beam_energy", "image", "crystal").float(),
            "hkl_valid": hkl_valid,
            "d_spacing": d_valid,
            "median_cell": median_cell,
            "obs_counts": obs_counts,
            "shell_id_fw": fw_shell_ids(d_valid, config.fw_n_shells),
            "is_centric": centric_flags(self.spacegroup, hkl_valid).cpu(),
            "sum_wI": sums["wI"], "sum_w": sums["w"], "sum_I": sums["I"], "count": sums["n"],
        }
        if delay_bin_idx is not None:
            tables["delay_bin_idx"] = delay_bin_idx
            tables["unique_timepoints"] = unique_timepoints
        meta = meta() if callable(meta) else meta
        writer.finish(tables, dict(meta or {}, n_crystals=n_crystals, n_valid=n_valid,
                                   n_panels=n_panels, has_sigma=bool(self.has_sigma),
                                   gain=gain, spacegroup=self.spacegroup))
        self.clock.lap("finish")
        self.clock.report("prepare")
        self.close()
        # renamed only once complete, so an interrupted build is never read as a store
        (tmp / STAGING_MARKER).unlink()
        tmp.rename(out)
        return PreparedStore(out)

    def close(self) -> None:
        for f in (self._hkl, self._raw):
            if f is not None:
                f.close()
        for f in (self._out_hkl, self._out_raw):
            if not f.closed:
                f.close()
        shutil.rmtree(self.workdir, ignore_errors=True)
