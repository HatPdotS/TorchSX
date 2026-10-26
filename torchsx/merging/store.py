"""Prepared observations on disk.

A store holds what :func:`~torchsx.merging.observations.prepare_observations`
returns, with the per-observation columns in one file of fixed-width records
instead of host tensors.  Host memory then scales with the crystal and
reflection counts, not with the observation count.

Rows are crystal-major: a crystal's observations are contiguous and sorted by
reflection index, and crystals follow each other in their prepared order.  A
training batch is a set of whole crystals, so it is one read per crystal; a
readout pass over crystals in order is a sequential read of the file.

Directory layout::

    meta.json    format, sizes, record layout, provenance
    obs.bin      n_obs records of :data:`RECORD`
    tables.pt    per-crystal and per-reflection tensors
    sample.npy   records of a fixed random row sample (see ``row_sample``)

Every record field is four bytes, so a batch of records is one contiguous
``(n, RECORD_WIDTH)`` float32/int32 block: it crosses the bus as one copy and
is split into columns on the compute device.  Batches are read into ordinary
host memory of their exact size; a page-locked pool would round every buffer
up to a power of two and hold it.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from torchsx.merging.observations import ObsInfo

STORE_FORMAT = "torchsx-prepared-store"
STORE_VERSION = 1

#: One observation.  ``I_obs``, ``bg`` and ``sigma`` are in photons (divided by
#: the detector gain; ``bg`` summed over the ``n_pix`` pixels of the disk),
#: ``background`` and ``peak`` are the stream's per-pixel ADU values, ``q`` the
#: lab-frame unit scattering vector.  Every field is four bytes wide.
RECORD = np.dtype([
    ("I_obs", "<f4"), ("bg", "<f4"), ("sigma", "<f4"),
    ("hkl_idx", "<i4"), ("panel_id", "<i4"),
    ("fs", "<f4"), ("ss", "<f4"), ("peak", "<f4"), ("background", "<f4"),
    ("ewald_offset", "<f4"), ("qx", "<f4"), ("qy", "<f4"), ("qz", "<f4"),
    ("n_pix", "<f4"),
])
RECORD_WIDTH = len(RECORD.names)
_COL = {name: i for i, name in enumerate(RECORD.names)}
assert RECORD.itemsize == 4 * RECORD_WIDTH

#: Concurrent reads per batch.  The file system serves many small reads in
#: parallel far faster than one after another.
READ_THREADS = 16
#: Batches read ahead at most (read-ahead of a training step plus that of an
#: evaluation pass running between steps).
MAX_PENDING = 3

_POOLS: dict[str, ThreadPoolExecutor] = {}
_POOL_LOCK = threading.Lock()


def _pool(kind: str = "io") -> ThreadPoolExecutor:
    """``io`` runs single range reads; ``ahead`` runs whole batch or chunk reads,
    which wait on ``io`` tasks -- separate pools, so neither can starve the other."""
    with _POOL_LOCK:
        pool = _POOLS.get(kind)
        if pool is None:
            workers = READ_THREADS if kind == "io" else MAX_PENDING + 1
            pool = _POOLS[kind] = ThreadPoolExecutor(max_workers=workers,
                                                     thread_name_prefix=f"torchsx-{kind}")
        return pool


def _pread_into(fd: int, buf: memoryview, offset: int) -> None:
    """Fill ``buf`` from ``offset``; network file systems may return short reads."""
    while len(buf):
        n = os.preadv(fd, [buf], offset)
        if n <= 0:
            raise EOFError(f"short read at byte {offset}")
        buf = buf[n:]
        offset += n


class RecordFile:
    """Read-only access to a file of fixed-width records by row ranges."""

    def __init__(self, path: str | Path, itemsize: int, n_rows: int):
        self.path = Path(path)
        self.itemsize = int(itemsize)
        self.n_rows = int(n_rows)
        size = self.path.stat().st_size
        if size != self.itemsize * self.n_rows:
            raise ValueError(f"{self.path}: {size} bytes, expected {self.n_rows} records "
                             f"of {self.itemsize} bytes")
        self._fd = os.open(self.path, os.O_RDONLY)

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def read_ranges(self, starts: np.ndarray, ends: np.ndarray, out: memoryview) -> None:
        """Read rows ``[starts[i], ends[i])`` one range after the other into ``out``.

        The ranges are issued in file order over a thread pool; each lands at
        its own position in ``out``, so the order of ``out`` is the order given.
        """
        lengths = (ends - starts) * self.itemsize
        dest = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        if int(lengths.sum()) != len(out):
            raise ValueError("output buffer does not match the requested rows")
        order = np.argsort(starts, kind="stable")
        n = len(order)
        if n == 0:
            return
        # Groups of consecutive (file-ordered) ranges of about equal size.
        n_groups = max(1, min(READ_THREADS, n, int(lengths.sum()) // (1 << 20) + 1))
        bounds = np.searchsorted(np.cumsum(lengths[order]),
                                 np.linspace(0, lengths.sum(), n_groups + 1)[1:-1])
        groups = np.split(order, bounds)
        fd, item = self._fd, self.itemsize

        def work(idx):
            for i in idx:
                if lengths[i]:
                    _pread_into(fd, out[dest[i]:dest[i] + lengths[i]], int(starts[i]) * item)

        if len(groups) == 1:
            work(groups[0])
            return
        for f in [_pool().submit(work, g) for g in groups if len(g)]:
            f.result()

    def read_slice(self, start: int, end: int, out: memoryview | None = None) -> memoryview:
        if out is None:
            out = memoryview(bytearray((end - start) * self.itemsize))
        self.read_ranges(np.array([start]), np.array([end]), out)
        return out


def row_runs(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Maximal runs of consecutive row indices, as ``(starts, ends)``."""
    if rows.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    brk = np.flatnonzero(np.diff(rows) != 1) + 1
    starts = rows[np.concatenate([[0], brk])]
    ends = rows[np.concatenate([brk - 1, [rows.size - 1]])] + 1
    return starts.astype(np.int64), ends.astype(np.int64)


def _empty_records(n: int) -> Tensor:
    return torch.empty((n, RECORD.itemsize), dtype=torch.uint8)


class ObservationStore:
    """The record file of a store: batch reads, read-ahead and sequential scans."""

    def __init__(self, path: str | Path, n_obs: int):
        self.path = Path(path)
        self.file = RecordFile(self.path / "obs.bin", RECORD.itemsize, n_obs)
        self.n_obs = int(n_obs)
        self._pending: list[tuple[Tensor, object]] = []
        self._last: tuple[Tensor, Tensor] | None = None
        #: Seconds callers spent blocked in :meth:`read`, and bytes read.
        self.wait_s = 0.0
        self.bytes_read = 0

    def close(self) -> None:
        self._pending, self._last = [], None
        self.file.close()

    def _read(self, rows: Tensor) -> Tensor:
        r = rows.numpy()
        out = _empty_records(r.size)
        starts, ends = row_runs(r)
        self.file.read_ranges(starts, ends, memoryview(out.numpy()).cast("B"))
        return out

    def prefetch(self, rows: Tensor) -> None:
        """Start reading ``rows`` in the background; :meth:`read` picks it up."""
        rows = rows.detach().to("cpu", torch.int64)
        if any(_same(r, rows) for r, _ in self._pending):
            return
        self._pending.append((rows, _pool("ahead").submit(self._read, rows)))
        del self._pending[:-MAX_PENDING]

    def read(self, rows: Tensor) -> Tensor:
        """Records of ``rows`` (any int tensor of row indices), ``(n, itemsize)`` uint8.

        A read issued by :meth:`prefetch` for the same rows is waited for
        instead of repeated, and the last batch is kept, so a second read of one
        batch (diagnostics) costs no I/O.
        """
        rows = rows.detach().to("cpu", torch.int64)
        if self._last is not None and _same(self._last[0], rows):
            return self._last[1]
        t0 = time.perf_counter()
        hit = next((i for i, (r, _) in enumerate(self._pending) if _same(r, rows)), None)
        if hit is not None:
            rec = self._pending.pop(hit)[1].result()
        else:
            rec = self._read(rows)
        self.wait_s += time.perf_counter() - t0
        self.bytes_read += rec.numel()
        self._last = (rows, rec)
        return rec

    def scan(self, chunk_rows: int = 1 << 22, start: int = 0, end: int | None = None):
        """``(row0, records)`` over ``[start, end)`` in order, one chunk read ahead."""
        end = self.n_obs if end is None else int(end)
        bounds = list(range(int(start), end, int(chunk_rows))) + [end]
        spans = list(zip(bounds[:-1], bounds[1:]))
        if not spans:
            return

        def load(s, e):
            out = np.empty(e - s, dtype=RECORD)
            self.file.read_slice(s, e, memoryview(out).cast("B"))
            return out

        fut = _pool("ahead").submit(load, *spans[0])
        for i, (s, _) in enumerate(spans):
            rec = fut.result()
            if i + 1 < len(spans):
                fut = _pool("ahead").submit(load, *spans[i + 1])
            yield s, rec


def _same(a: Tensor, b: Tensor) -> bool:
    return a is b or (a.shape == b.shape and bool(torch.equal(a, b)))


@dataclass
class StoreObsInfo:
    """The per-crystal part of :class:`ObsInfo` for a store.

    Per-observation fields are not held; :meth:`PreparedStore.batch` returns
    them for the rows asked for.
    """

    n_crystals: int
    crystal_feat: Tensor
    crystal_res: Tensor             # (n_crystals,) diffraction limit
    crystal_beam_energy: Tensor     # (n_crystals,) photon energy of the crystal's image
    has_sigma: bool
    delay_bin_idx: Tensor | None = None
    unique_timepoints: Tensor | None = None


class PreparedStore:
    """A :class:`~torchsx.merging.observations.PreparedObs` whose observations
    stay on disk.

    Carries the same per-reflection fields; per-observation data is read in
    batches with :meth:`batch`.  Per-reflection sums of the whole set (the
    unity merge and the plain mean) and a fixed random row sample are written
    with the records, so nothing reads the full file when training starts.
    """

    partiality = None
    obs_weights = None

    def __init__(self, path: str | Path):
        self.path = Path(path)
        meta = json.loads((self.path / "meta.json").read_text())
        if meta.get("format") != STORE_FORMAT or meta.get("version") != STORE_VERSION:
            raise ValueError(f"{self.path} is not a version-{STORE_VERSION} prepared store")
        if meta.get("record") != [list(f) for f in RECORD.descr]:
            raise ValueError(f"{self.path}: record layout differs from this torchsx")
        self.meta = meta
        t = torch.load(self.path / "tables.pt", weights_only=True)
        self.tables = t
        self.n_obs_total = int(meta["n_obs"])
        self.n_valid = int(meta["n_valid"])
        self.hkl_valid = t["hkl_valid"]
        self.d_spacing = t["d_spacing"]
        self.median_cell = t["median_cell"]
        self.obs_counts = t["obs_counts"]
        self.shell_id_fw = t["shell_id_fw"]
        self.is_centric = t["is_centric"]
        self.crystal_offsets = t["crystal_offsets"]
        self.n_panels = int(meta["n_panels"])
        self.obs_info = StoreObsInfo(
            n_crystals=int(meta["n_crystals"]),
            crystal_feat=t["crystal_feat"],
            crystal_res=t["crystal_res"],
            crystal_beam_energy=t["crystal_beam_energy"],
            has_sigma=bool(meta["has_sigma"]),
            delay_bin_idx=t.get("delay_bin_idx"),
            unique_timepoints=t.get("unique_timepoints"),
        )
        self.store = ObservationStore(self.path, self.n_obs_total)
        # s^2 = 1 / (4 d^2) per reflection: the per-observation value is a lookup
        self.s_sq = 1.0 / (4.0 * self.d_spacing.clamp(min=0.1) ** 2)
        self._device_tables: dict = {}

    def close(self) -> None:
        self.store.close()

    def to(self, device) -> "PreparedStore":
        """The observations stay on disk; batches are read to their device."""
        return self

    def crystal_counts(self) -> Tensor:
        return self.crystal_offsets[1:] - self.crystal_offsets[:-1]

    # ── reflection sums ────────────────────────────────────────────────

    def unity_merge(self, *, floor: float = 1e-3) -> Tensor:
        """:func:`~torchsx.nn.merge_target.unity_merge` over every observation."""
        from torchsx.nn.merge_target import unity_from_sums
        return unity_from_sums(self.tables["sum_wI"], self.tables["sum_w"], floor=floor)

    def plain_mean(self, *, floor: float = 1e-3) -> Tensor:
        """:func:`~torchsx.nn.merge_target.plain_mean` over every observation."""
        from torchsx.nn.merge_target import plain_mean_from_sums
        return plain_mean_from_sums(self.tables["sum_I"], self.tables["count"], floor=floor)

    # ── observations ───────────────────────────────────────────────────

    def _tables_on(self, device) -> dict:
        device = torch.device(device)
        oi = self.obs_info
        got = self._device_tables.get(str(device))
        # the split labels may be replaced after loading (a held-out split folded in)
        if got is None or got["src"][0] is not oi.delay_bin_idx or got["src"][1] is not oi.unique_timepoints:
            got = {
                "src": (oi.delay_bin_idx, oi.unique_timepoints),
                "offsets": self.crystal_offsets.to(device),
                "s_sq": self.s_sq.to(device),
                "res": oi.crystal_res.to(device),
                "energy": oi.crystal_beam_energy.to(device),
                "feat": oi.crystal_feat.to(device),
                "delay": None if oi.delay_bin_idx is None else oi.delay_bin_idx.to(device),
                "times": None if oi.unique_timepoints is None else oi.unique_timepoints.to(device),
            }
            self._device_tables[str(device)] = got
        return got

    def decode(self, records: Tensor, rows: Tensor) -> ObsInfo:
        """Records and their row indices (same device) -> an :class:`ObsInfo`.

        Matches ``PreparedObs.obs_info.gather`` of the same observations, with
        ``obs_id`` the row in this store and ``I_obs``/``bg`` attached.
        """
        n = records.shape[0]
        f = records.view(torch.float32).reshape(n, RECORD_WIDTH)
        i = records.view(torch.int32).reshape(n, RECORD_WIDTH)
        t = self._tables_on(records.device)
        rows = rows.to(records.device, torch.int64)
        cid = torch.searchsorted(t["offsets"], rows, right=True) - 1
        hkl = i[:, _COL["hkl_idx"]].long()

        def col(name):
            return f[:, _COL[name]].contiguous()

        return ObsInfo(
            crystal_id=cid,
            s_sq=t["s_sq"][hkl],
            hkl_idx=hkl,
            n_crystals=self.obs_info.n_crystals,
            panel_id=i[:, _COL["panel_id"]].long(),
            fs=col("fs"), ss=col("ss"), peak=col("peak"), background=col("background"),
            ewald_offset=col("ewald_offset"),
            sigma=col("sigma") if self.obs_info.has_sigma else None,
            q_dir=f[:, _COL["qx"]:_COL["qz"] + 1].contiguous(),
            n_pix=col("n_pix"),
            beam_energy=t["energy"][cid],
            crystal_res=t["res"][cid],
            crystal_feat=t["feat"],
            obs_id=rows,
            delay_bin_idx=t["delay"],
            unique_timepoints=t["times"],
            I_obs=col("I_obs"),
            bg=col("bg"),
        )

    def prefetch(self, rows: Tensor) -> None:
        self.store.prefetch(rows)

    def batch(self, rows: Tensor, device) -> ObsInfo:
        """The observations at ``rows``, on ``device``."""
        rec = self.store.read(rows)
        device = torch.device(device)
        rec = rec.to(device)
        return self.decode(rec, rows.to(device, non_blocking=True))

    def crystal_of_row(self, row: int) -> int:
        return int(torch.searchsorted(self.crystal_offsets, torch.tensor([int(row)]), right=True)[0]) - 1

    def random_rows(self, n_sample: int, seed: int) -> ObsInfo:
        """``batch`` of ``torch.randint(n_obs, (n_sample,))`` under ``seed``, on the host.

        Served from the sample written with the store when it was drawn the
        same way; otherwise read with one pass over the file.
        """
        idx = torch.randint(self.n_obs_total, (int(n_sample),),
                            generator=torch.Generator().manual_seed(int(seed)))
        spec = self.meta.get("row_sample") or {}
        if spec.get("n") == int(n_sample) and spec.get("seed") == int(seed):
            rec = np.load(self.path / spec["file"])
            return self.decode(torch.from_numpy(rec.view(np.uint8).reshape(len(rec), -1)), idx)
        return self.gather(idx)

    def materialize(self):
        """The whole store as an in-memory ``PreparedObs``, rows in store order.

        For small data and for checking the store against the in-memory path:
        the trainer gives the same results on either.
        """
        from torchsx.merging.observations import PreparedObs

        oi = self.gather(torch.arange(self.n_obs_total))
        I_obs, bg = oi.I_obs, oi.bg
        oi.I_obs = oi.bg = None
        return PreparedObs(
            hkl_valid=self.hkl_valid, d_spacing=self.d_spacing, n_valid=self.n_valid,
            median_cell=self.median_cell, I_obs=I_obs, hkl_id=oi.hkl_idx, bg=bg,
            obs_counts=self.obs_counts, n_obs_total=self.n_obs_total, obs_info=oi,
            partiality=None, shell_id_fw=self.shell_id_fw, is_centric=self.is_centric,
        )

    def gather(self, idx: Tensor) -> ObsInfo:
        """``batch`` of arbitrary rows (any order, repeats allowed) on the host,
        read with one sequential pass over the file."""
        idx = idx.detach().to("cpu", torch.int64)
        rec = gather_records(self.store, idx.numpy())
        return self.decode(torch.from_numpy(rec.view(np.uint8).reshape(len(rec), -1)), idx)


def gather_records(store: ObservationStore, idx: np.ndarray, chunk_rows: int = 1 << 22) -> np.ndarray:
    """Records at arbitrary rows (any order, repeats allowed) by one sequential pass."""
    order = np.argsort(idx, kind="stable")
    sidx = idx[order]
    out = np.empty(len(idx), dtype=RECORD)
    lo = 0
    for s, rec in store.scan(chunk_rows):
        hi = np.searchsorted(sidx, s + len(rec), side="left")
        if hi > lo:
            out[order[lo:hi]] = rec[sidx[lo:hi] - s]
            lo = hi
    return out


class StoreWriter:
    """Append records to a new store directory; :meth:`finish` makes it readable."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path / "obs.bin", "wb", buffering=1 << 24)
        self.n_obs = 0
        self._sample = None

    def draw_sample(self, n_obs_total: int, n_sample: int, seed: int) -> None:
        """Collect the records of :meth:`PreparedStore.random_rows` ``(n_sample, seed)``
        while writing; ``n_obs_total`` is the final row count."""
        idx = torch.randint(int(n_obs_total), (int(n_sample),),
                            generator=torch.Generator().manual_seed(int(seed))).numpy()
        order = np.argsort(idx, kind="stable")
        self._sample = dict(n=int(n_sample), seed=int(seed), order=order, sorted=idx[order],
                            out=np.empty(len(idx), dtype=RECORD), lo=0)

    def write(self, records: np.ndarray) -> None:
        assert records.dtype == RECORD
        s = self._sample
        if s is not None:
            hi = np.searchsorted(s["sorted"], self.n_obs + len(records), side="left")
            if hi > s["lo"]:
                s["out"][s["order"][s["lo"]:hi]] = records[s["sorted"][s["lo"]:hi] - self.n_obs]
                s["lo"] = hi
        self._f.write(memoryview(records).cast("B"))
        self.n_obs += len(records)

    def finish(self, tables: dict, meta: dict) -> None:
        self._f.close()
        meta = dict(meta, format=STORE_FORMAT, version=STORE_VERSION, n_obs=self.n_obs,
                    record=[list(f) for f in RECORD.descr])
        s = self._sample
        if s is not None:
            if s["lo"] != len(s["sorted"]):
                raise ValueError("row sample drawn for a different row count")
            np.save(self.path / "sample.npy", s["out"])
            meta["row_sample"] = {"n": s["n"], "seed": s["seed"], "file": "sample.npy"}
        torch.save(tables, self.path / "tables.pt")
        (self.path / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
