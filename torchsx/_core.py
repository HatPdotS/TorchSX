"""Columnar stream cores: BaseStream → SFXStreamCore → ShoeboxStreamCore.

The three-level SX data model (observation / crystal / image) is stored as a set
of :class:`~torchsx.levels.Level` columnar tables joined by integer key columns
(``crystal_id``: obs→crystal, ``image_id``: crystal→image). ``BaseStream`` holds
the generic columnar/join machinery; ``SFXStreamCore`` adds the SX field layout,
the legacy keyword constructor, and validation; ``ShoeboxStreamCore`` is just an
``SFXStreamCore`` that additionally carries per-observation pixel patches.

``_SFXStreamCore`` is kept as an alias of ``SFXStreamCore`` so existing imports and
the keyword constructor ``_SFXStreamCore(hkl=…, I=…, …)`` keep working unchanged;
attribute access (``core.hkl``, ``core.cell``, ``core.beam_energy``) is served by a
``__getattr__`` shim over the levels.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


def abspath_str(p: str) -> str:
    """Resolve a path string to an absolute, ``~``-expanded path.

    Streams store filesystem paths (``source_path``, per-image ``filename``) so
    that downstream consumers — finetuning / refinement reloads frames by
    ``metadata['image']['filename']``, ``write_stream`` copies a header from
    ``source_path`` — resolve correctly regardless of the current working
    directory. Normalising to absolute at every ingress makes that robust.

    Empty strings (e.g. an un-indexed image's filename placeholder) and
    non-strings pass through unchanged; the call is idempotent on paths that are
    already absolute.
    """
    if not isinstance(p, str) or not p:
        return p
    return os.path.abspath(os.path.expanduser(p))


def absolutize_image_filenames(metadata) -> None:
    """In place: rewrite ``metadata['image']['filename']`` entries to absolute paths.

    Used at the load boundary so a stream written with relative paths comes back
    with absolute ones. Safe on freshly-built metadata; no-op when the image
    filename list is absent.
    """
    img = metadata.get("image") if metadata else None
    if not img:
        return
    fnames = img.get("filename")
    if fnames:
        img["filename"] = [abspath_str(f) for f in fnames]

from torchsx.geometry import Geometry
from torchsx.io._metadata import StreamMetadata
from torchsx.levels import ColumnSpec, Level


# Field → level routing for the SX layout (drives the legacy keyword constructor).
_OBS_FIELDS = ["hkl", "I", "sigma", "background", "peak", "fs", "ss", "panel_id", "crystal_id"]
_CRYSTAL_FIELDS = [
    "astar", "bstar", "cstar", "cell", "profile_rad", "resolution",
    "n_refl", "image_id", "crystal_in_image",
]
_IMAGE_FIELDS = ["beam_energy", "det_shift_x", "det_shift_y", "n_peaks", "camera_length"]

_FIELD_TO_LEVEL: dict[str, str] = {
    **{f: "obs" for f in _OBS_FIELDS},
    **{f: "crystal" for f in _CRYSTAL_FIELDS},
    **{f: "image" for f in _IMAGE_FIELDS},
}

# Columns whose canonical dtype must be set explicitly (everything else float32/given).
_CANONICAL_INT_DTYPE = {
    "hkl": torch.int16,
    "panel_id": torch.int16,
    "crystal_id": torch.int32,
    "n_refl": torch.int32,
    "image_id": torch.int32,
    "crystal_in_image": torch.int32,
    "n_peaks": torch.int32,
}


@dataclass
class JoinSpec:
    """A directed join edge: ``child_level.key_column → parent_level``."""

    child_level: str
    key_column: str
    parent_level: str


def _default_sx_joins() -> list[JoinSpec]:
    return [
        JoinSpec("obs", "crystal_id", "crystal"),
        JoinSpec("crystal", "image_id", "image"),
    ]


def _sx_joins_with_peaks() -> list[JoinSpec]:
    """Join set for a stream that carries a raw peak-search ``peak`` level.

    The ``peak`` level holds one row per spot found by the peak finder (indexed or
    not) and joins straight to ``image`` (peaks are an image-level product, not a
    crystal one). Used by :func:`~torchsx.datareduction.to_stream.build_stream`;
    the two-join default is kept for peak-less streams (parsed/legacy).
    """
    return _default_sx_joins() + [JoinSpec("peak", "peak_image_id", "image")]


class BaseStream:
    """Generic columnar container: named :class:`Level`s + a join registry.

    Provenance-agnostic. Filtering/transforms return new objects of the most
    specific subclass (:meth:`refine_type`); attribute access falls through to
    the level columns via :meth:`__getattr__`.
    """

    LEVEL_ORDER = ("obs", "crystal", "image")

    def __init__(
        self,
        levels: dict[str, Level],
        joins: list[JoinSpec] | None = None,
        geometry: Geometry | None = None,
        metadata: StreamMetadata | None = None,
        spacegroup: str = "P 1",
        source_path: str = "",
        friedel_merged: bool = True,
    ):
        self.levels = levels
        self.joins = joins if joins is not None else _default_sx_joins()
        self.geometry = geometry
        self.metadata = metadata if metadata is not None else StreamMetadata()
        self.spacegroup = spacegroup
        # Always store provenance paths absolute (idempotent; "" passes through).
        self.source_path = abspath_str(source_path)
        # Whether ``hkl`` has already been Friedel-merged (canonicalized to the
        # P-1 ASU, mates collapsed).  Drives anomalous-aware MTZ writing in
        # :meth:`torchsx.merging.posterior.Posterior.integrate`.  Default True
        # for backwards compatibility (the historical behaviour).
        self.friedel_merged = bool(friedel_merged)
        # Optional detector good-pixel mask ``(P, ss, fs)`` bool (True=good) built by
        # the masking pass. Carried here so it can be embedded in the .sfx; not threaded
        # through transforms (set on a freshly built/loaded core). See torchsx.io._safetensors.
        self.detector_mask: torch.Tensor | None = None

    # ---- attribute-compat shim ----------------------------------------

    def __getattr__(self, name: str) -> torch.Tensor:
        # Only invoked when normal lookup fails; guard against recursion pre-init.
        levels = self.__dict__.get("levels")
        if levels:
            for lvl in levels.values():
                if name in lvl:
                    return lvl[name]
        raise AttributeError(name)

    # ---- sizes ---------------------------------------------------------

    @property
    def n_observations(self) -> int:
        return self.levels["obs"].n

    @property
    def n_crystals(self) -> int:
        return self.levels["crystal"].n

    @property
    def n_images(self) -> int:
        return self.levels["image"].n

    @property
    def device(self) -> torch.device:
        return self.levels["obs"].device

    # ---- construction helpers -----------------------------------------

    @classmethod
    def _make(
        cls,
        levels: dict[str, Level],
        joins: list[JoinSpec],
        geometry: Geometry | None,
        metadata: StreamMetadata,
        spacegroup: str,
        source_path: str,
        friedel_merged: bool = True,
    ) -> "BaseStream":
        """Build a new stream from levels, bypassing the legacy constructor,
        and refine to the most specific subclass given which columns exist."""
        target = _classify(levels)
        obj = target.__new__(target)
        BaseStream.__init__(
            obj, levels, joins, geometry, metadata, spacegroup, source_path,
            friedel_merged=friedel_merged,
        )
        obj._validate()
        return obj

    def refine_type(self) -> "BaseStream":
        """Return self re-typed to the most specific subclass for its columns."""
        target = _classify(self.levels)
        if type(self) is target:
            return self
        return self._make(
            self.levels, self.joins, self.geometry, self.metadata,
            self.spacegroup, self.source_path,
            friedel_merged=self.friedel_merged,
        )

    def _validate(self) -> None:
        """Generic validation: join-key columns index into their parent level."""
        for join in self.joins:
            child, parent = self.levels.get(join.child_level), self.levels.get(join.parent_level)
            if child is None or parent is None or join.key_column not in child:
                continue
            if child.n > 0:
                key = child[join.key_column]
                assert key.min() >= 0, f"{join.key_column} has negative values"
                assert key.max() < parent.n, (
                    f"{join.key_column} max {int(key.max())} >= "
                    f"n_{join.parent_level} {parent.n}"
                )

    # ---- immutable transforms -----------------------------------------

    def to(self, device) -> "BaseStream":
        new_levels = {name: lvl.to(device) for name, lvl in self.levels.items()}
        meta = self.metadata.to(device) if self.metadata else StreamMetadata()
        return self._make(
            new_levels, self.joins, self.geometry, meta, self.spacegroup, self.source_path,
            friedel_merged=self.friedel_merged,
        )

    def select_cascade(self, masks: dict[str, torch.Tensor]) -> "BaseStream":
        """Apply per-level boolean masks and rebuild contiguous join keys.

        ``masks`` need only supply a bool mask for the levels the caller cares
        about (typically obs/crystal/image). Any child level present in the
        stream but *absent* from ``masks`` (e.g. the raw ``peak`` level) has its
        mask **derived** from its parent's: a child row survives iff its parent
        does. This keeps every level join-consistent without each filter needing
        to know about every level. Each column is then subset in one loop and
        each :class:`JoinSpec`'s key column is remapped onto the surviving parent
        rows.
        """
        # Derive masks for child levels that the caller didn't supply, from the
        # parent mask through the join (fixpoint: handles chains parent→child→…).
        masks = dict(masks)
        changed = True
        while changed:
            changed = False
            for join in self.joins:
                child = self.levels.get(join.child_level)
                if child is None or join.key_column not in child:
                    continue
                if join.child_level in masks:
                    continue
                parent_mask = masks.get(join.parent_level)
                if parent_mask is None:
                    continue
                key = child[join.key_column].long()
                masks[join.child_level] = parent_mask[key]
                changed = True

        new_levels: dict[str, Level] = {}
        for name, lvl in self.levels.items():
            mask = masks.get(name)
            new_levels[name] = lvl.select(mask) if mask is not None else lvl

        for join in self.joins:
            child_level = self.levels.get(join.child_level)
            if child_level is None or join.key_column not in child_level:
                continue
            child_mask = masks.get(join.child_level)
            parent_mask = masks.get(join.parent_level)
            if parent_mask is None:
                continue
            old_key = self.levels[join.child_level][join.key_column]
            kept_parent = parent_mask.nonzero(as_tuple=True)[0]
            remap = torch.full(
                (self.levels[join.parent_level].n,), -1,
                dtype=old_key.dtype, device=old_key.device,
            )
            remap[kept_parent] = torch.arange(
                kept_parent.shape[0], dtype=old_key.dtype, device=old_key.device
            )
            sub_key = old_key[child_mask] if child_mask is not None else old_key
            new_key = remap[sub_key.long()]
            new_levels[join.child_level].__setitem__(
                join.key_column, new_key,
                self.levels[join.child_level].schema[join.key_column],
            )

        meta = _subset_metadata(self.metadata, masks)
        return self._make(
            new_levels, self.joins, self.geometry, meta, self.spacegroup, self.source_path,
            friedel_merged=self.friedel_merged,
        )

    def with_columns(self, level: str, **columns: torch.Tensor) -> "BaseStream":
        """Return a new stream with extra/overwritten columns on ``level``.

        Unlike :meth:`replace` (which only routes known SX fields), this adds
        arbitrary new columns (e.g. ``shoebox``, ``peak_snr``, indexing
        confidence) — the columnar extensibility hook.
        """
        if level not in self.levels:
            raise KeyError(f"unknown level {level!r}")
        new_levels = dict(self.levels)
        base = self.levels[level]
        fresh = Level()
        for cname, tensor in base.columns.items():
            fresh.__setitem__(cname, columns.get(cname, tensor), base.schema.get(cname))
        for cname, tensor in columns.items():
            if cname not in fresh:
                fresh.__setitem__(cname, tensor)
        new_levels[level] = fresh
        return self._make(
            new_levels, self.joins, self.geometry, self.metadata,
            self.spacegroup, self.source_path,
            friedel_merged=self.friedel_merged,
        )

    def broadcast(self, column: str, from_level: str, to_level: str = "obs") -> torch.Tensor:
        """Gather a column from ``from_level`` down to ``to_level`` through joins.

        e.g. ``broadcast("beam_energy", "image", "obs")`` →
        ``beam_energy[image_id][crystal_id]``.
        """
        col = self.levels[from_level][column]
        path = []
        cur = to_level
        while cur != from_level:
            join = next(j for j in self.joins if j.child_level == cur)
            path.append(self.levels[cur][join.key_column])
            cur = join.parent_level
        for key in reversed(path):
            col = col[key.long()]
        return col

    def replace(self, **overrides) -> "BaseStream":
        """Return a new stream with the given column/attribute overrides.

        Accepts SX field names (rerouted to their level), level names, and the
        scalar attributes ``geometry``/``metadata``/``spacegroup``/``source_path``.
        Mirrors ``dataclasses.replace`` for the facade.
        """
        attrs: dict[str, object] = {
            "geometry": self.geometry, "metadata": self.metadata,
            "spacegroup": self.spacegroup, "source_path": self.source_path,
            "friedel_merged": self.friedel_merged,
        }
        level_over: dict[str, Level] = {}
        col_over: dict[str, dict[str, torch.Tensor]] = {}
        for key, value in overrides.items():
            if key in attrs:
                attrs[key] = value
            elif key in self.levels:
                level_over[key] = value
            else:
                level_name = _FIELD_TO_LEVEL.get(key)
                if level_name is None:
                    raise KeyError(f"replace: unknown field/level {key!r}")
                col_over.setdefault(level_name, {})[key] = value

        new_levels: dict[str, Level] = {}
        for name, lvl in self.levels.items():
            if name in level_over:
                new_levels[name] = level_over[name]
                continue
            overs = col_over.get(name)
            if not overs:
                new_levels[name] = lvl  # unchanged — reuse immutable level
                continue
            # Rebuild fresh so an override may change the level's row count
            # (e.g. expand-mode update_observations replaces all obs columns).
            fresh = Level()
            for cname in list(lvl.columns.keys()) + [c for c in overs if c not in lvl.columns]:
                val = overs.get(cname, lvl.columns.get(cname))
                fresh.__setitem__(cname, val, lvl.schema.get(cname))
            new_levels[name] = fresh

        obj = self._make(
            new_levels, self.joins, attrs["geometry"], attrs["metadata"],
            attrs["spacegroup"], attrs["source_path"],
            friedel_merged=attrs["friedel_merged"],
        )
        obj.detector_mask = self.detector_mask  # detector-level constant; survives replace
        return obj

    @classmethod
    def concat(cls, cores: list["BaseStream"]) -> "BaseStream":
        """Row-wise concatenate cores, offsetting join keys onto the merged levels.

        Works on arbitrary columns (a ``shoebox`` obs column rides along) and any
        level set; generalises the hand-rolled per-field concat in the facade.
        """
        cores = list(cores)
        if not cores:
            raise ValueError("concat requires at least one core")
        if len(cores) == 1:
            return cores[0]

        joins = cores[0].joins
        level_names = list(cores[0].levels.keys())
        running: dict[str, int] = {ln: 0 for ln in level_names}
        per_core: list[dict[str, Level]] = []

        for core in cores:
            adjusted = {
                ln: Level(dict(lvl.columns), dict(lvl.schema), n=lvl.n)
                for ln, lvl in core.levels.items()
            }
            for join in joins:
                base = running.get(join.parent_level, 0)
                child = adjusted.get(join.child_level)
                if base and child is not None and join.key_column in child:
                    spec = child.schema.get(join.key_column)
                    child.__setitem__(join.key_column, child[join.key_column] + base, spec)
            per_core.append(adjusted)
            for ln, lvl in core.levels.items():
                running[ln] = running.get(ln, 0) + lvl.n

        out_levels = {
            ln: Level.concat([pc[ln] for pc in per_core]) for ln in level_names
        }
        merged_meta = _merge_metadata([c.metadata for c in cores])
        fm0 = bool(cores[0].friedel_merged)
        for i, c in enumerate(cores[1:], start=1):
            if bool(c.friedel_merged) != fm0:
                raise ValueError(
                    f"concat: cores disagree on friedel_merged "
                    f"(core[0]={fm0}, core[{i}]={bool(c.friedel_merged)}). "
                    "All inputs must share the same Friedel-merge state."
                )
        return cls._make(
            out_levels, joins, cores[0].geometry, merged_meta,
            cores[0].spacegroup, cores[0].source_path,
            friedel_merged=fm0,
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n_observations={self.n_observations}, "
            f"n_crystals={self.n_crystals}, n_images={self.n_images})"
        )


class SFXStreamCore(BaseStream):
    """Serial-crystallography stream core with the canonical three-level layout.

    Provides the legacy keyword constructor and the original ``__post_init__``
    validation (row-count + join-key-range checks, ``camera_length`` synthesis).
    """

    def __init__(
        self,
        hkl: torch.Tensor,
        I: torch.Tensor,
        sigma: torch.Tensor,
        background: torch.Tensor,
        peak: torch.Tensor,
        fs: torch.Tensor,
        ss: torch.Tensor,
        panel_id: torch.Tensor,
        crystal_id: torch.Tensor,
        astar: torch.Tensor,
        bstar: torch.Tensor,
        cstar: torch.Tensor,
        cell: torch.Tensor,
        profile_rad: torch.Tensor,
        resolution: torch.Tensor,
        n_refl: torch.Tensor,
        image_id: torch.Tensor,
        crystal_in_image: torch.Tensor,
        beam_energy: torch.Tensor,
        det_shift_x: torch.Tensor,
        det_shift_y: torch.Tensor,
        n_peaks: torch.Tensor,
        camera_length: torch.Tensor | None = None,
        geometry: Geometry | None = None,
        metadata: StreamMetadata | None = None,
        spacegroup: str = "P 1",
        source_path: str = "",
        friedel_merged: bool = True,
    ):
        n_obs, n_crystals, n_images = hkl.shape[0], cell.shape[0], beam_energy.shape[0]
        if camera_length is None:
            camera_length = torch.zeros(n_images, dtype=torch.float32)

        local = locals()
        # Row-count validation up front so the documented AssertionError fires
        # (rather than Level's ValueError) on a mismatched field.
        _expected = {"obs": n_obs, "crystal": n_crystals, "image": n_images}
        for fname, level_name in _FIELD_TO_LEVEL.items():
            t = local[fname]
            exp = _expected[level_name]
            assert t.shape[0] == exp, f"{fname} has {t.shape[0]} rows, expected {exp}"

        levels = {"obs": Level(), "crystal": Level(), "image": Level()}
        for fname, level_name in _FIELD_TO_LEVEL.items():
            spec = None
            if fname in _CANONICAL_INT_DTYPE:
                spec = ColumnSpec(dtype=_CANONICAL_INT_DTYPE[fname])
            levels[level_name].__setitem__(fname, local[fname], spec)

        super().__init__(
            levels, _default_sx_joins(), geometry, metadata, spacegroup, source_path,
            friedel_merged=friedel_merged,
        )
        self._validate()


class ShoeboxStreamCore(SFXStreamCore):
    """An ``SFXStreamCore`` that additionally carries per-observation patches.

    Distinguished purely by data richness (an obs ``shoebox`` or ``shoebox_id``
    column), never by provenance.
    """


def _classify(levels: dict[str, Level]) -> type:
    """Pick the most specific core class supported by the present columns."""
    obs = levels.get("obs")
    if obs is not None and ("shoebox" in obs or "shoebox_id" in obs):
        return ShoeboxStreamCore
    if obs is not None and {"hkl", "I", "sigma"}.issubset(obs.keys()):
        return SFXStreamCore
    return BaseStream


def _subset_metadata(
    metadata: StreamMetadata, masks: dict[str, torch.Tensor]
) -> StreamMetadata:
    """Subset per-level StreamMetadata entries by the level masks.

    Transitional: kept until per-level metadata is folded into columns. Mirrors
    the old ``filter._subset_metadata`` length-heuristic.
    """
    if not metadata:
        return StreamMetadata()
    new = StreamMetadata()
    for top_key, sub in metadata.items():
        if not isinstance(sub, dict) or top_key not in masks:
            new[top_key] = sub
            continue
        mask = masks[top_key]
        n_old = int(mask.shape[0])
        new[top_key] = {}
        for field_name, value in sub.items():
            if isinstance(value, torch.Tensor) and value.shape[0] == n_old:
                new[top_key][field_name] = value[mask]
            elif isinstance(value, list) and len(value) == n_old:
                keep = mask.nonzero(as_tuple=True)[0].tolist()
                new[top_key][field_name] = [value[i] for i in keep]
            else:
                new[top_key][field_name] = value
    return new


def _merge_metadata(metas: list[StreamMetadata]) -> StreamMetadata:
    """Concatenate per-level StreamMetadata lists/tensors across cores."""
    merged = StreamMetadata()
    for meta in metas:
        if not meta:
            continue
        for top_key, sub in meta.items():
            if not isinstance(sub, dict):
                continue
            dst = merged.setdefault(top_key, {})
            for field_name, values in sub.items():
                bucket = dst.setdefault(field_name, [])
                if isinstance(values, list):
                    bucket.extend(values)
                elif isinstance(values, torch.Tensor):
                    bucket.append(values)
    for top_key, sub in merged.items():
        if isinstance(sub, dict):
            for field_name, values in sub.items():
                if isinstance(values, list) and values and isinstance(values[0], torch.Tensor):
                    sub[field_name] = torch.cat(values)
    return merged


# Backwards-compatible alias: existing imports and the keyword constructor.
_SFXStreamCore = SFXStreamCore
