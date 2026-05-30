"""Columnar ``Level`` primitive: a table of named tensor columns sharing a row dim.

A :class:`Level` is one level of the three-level SX data model (observation /
crystal / image), represented as a dict of named tensors that all share the same
leading dimension ``N`` (the *only* invariant; trailing dims are free). This makes
the data model **extensible** (add a column without editing a class),
**vectorizable** (filtering is one indexing op over every column), and
**disk-mappable** (each column is one safetensors entry).

Categorical/string columns reuse the string-table scheme proven in
:mod:`torchsx.io._metadata`: the column stores an int32 index tensor and the
table of unique strings lives in the column's :class:`ColumnSpec`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace as _dc_replace

import torch


# dtype <-> short string, for JSON-serialisable schema and canonical-dtype restore.
_DTYPE_TO_STR = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.bool: "bool",
}
_STR_TO_DTYPE = {v: k for k, v in _DTYPE_TO_STR.items()}

# safetensors cannot store these; upcast on write, restore canonical dtype on read.
_UPCAST_ON_SAVE = {torch.int16: torch.int32}


@dataclass
class ColumnSpec:
    """Per-column metadata not recoverable from the tensor alone.

    ``element_shape`` (trailing dims) is stored explicitly so an empty level
    (``N == 0``) still validates and round-trips. ``string_table`` carries the
    categorical decode table. ``role`` tags columns for special handling:

    - ``"join_key"`` — an int column pointing into ``join_target`` level.
    - ``"categorical"`` — int index column decoded via ``string_table``.
    - ``"derived"`` — a cache (e.g. ``hkl_id``) dropped on any reorder/subset.
    - ``None`` — ordinary data column.
    """

    dtype: torch.dtype
    element_shape: tuple[int, ...] = ()
    role: str | None = None
    join_target: str | None = None
    string_table: list[str] | None = None
    derived_from: tuple[str, ...] | None = None

    def to_dict(self) -> dict:
        return {
            "dtype": _DTYPE_TO_STR[self.dtype],
            "element_shape": list(self.element_shape),
            "role": self.role,
            "join_target": self.join_target,
            "string_table": self.string_table,
            "derived_from": list(self.derived_from) if self.derived_from else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ColumnSpec":
        return cls(
            dtype=_STR_TO_DTYPE[d["dtype"]],
            element_shape=tuple(d.get("element_shape", ())),
            role=d.get("role"),
            join_target=d.get("join_target"),
            string_table=d.get("string_table"),
            derived_from=tuple(d["derived_from"]) if d.get("derived_from") else None,
        )


class Level:
    """A columnar table: ``dict[str, Tensor]`` whose columns share row dim ``N``.

    All transforms (:meth:`select`, :meth:`gather`, :meth:`concat`, :meth:`to`)
    return a NEW ``Level`` and never mutate in place. ``__setitem__`` mutates and
    is the builder/escape-hatch for constructing a level column by column.
    """

    def __init__(
        self,
        columns: dict[str, torch.Tensor] | None = None,
        schema: dict[str, ColumnSpec] | None = None,
        n: int | None = None,
    ):
        self.columns: dict[str, torch.Tensor] = {}
        self.schema: dict[str, ColumnSpec] = {}
        self._n: int | None = n
        if columns:
            for name, tensor in columns.items():
                spec = schema.get(name) if schema else None
                self.__setitem__(name, tensor, spec)

    # ---- introspection -------------------------------------------------

    @property
    def n(self) -> int:
        """Number of rows. 0 if the level is empty-with-known-size, else inferred."""
        return self._n if self._n is not None else 0

    @property
    def device(self) -> torch.device | None:
        for t in self.columns.values():
            return t.device
        return None

    def keys(self):
        return self.columns.keys()

    def __iter__(self):
        return iter(self.columns)

    def __contains__(self, key: str) -> bool:
        return key in self.columns

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        cols = ", ".join(self.columns.keys())
        return f"Level(n={self.n}, columns=[{cols}])"

    # ---- column access / mutation -------------------------------------

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.columns[key]

    def __setitem__(
        self, key: str, tensor: torch.Tensor, spec: ColumnSpec | None = None
    ) -> None:
        """Add/replace a column (mutating; builder escape hatch).

        Validates the shared-N invariant and infers ``element_shape``/``dtype``
        into a :class:`ColumnSpec` when one is not supplied.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"column {key!r} must be a Tensor, got {type(tensor)}")
        if self._n is None:
            self._n = tensor.shape[0]
        elif tensor.shape[0] != self._n:
            raise ValueError(
                f"column {key!r} has {tensor.shape[0]} rows; level has {self._n}"
            )
        if spec is None:
            spec = ColumnSpec(dtype=tensor.dtype, element_shape=tuple(tensor.shape[1:]))
        else:
            # keep the spec but trust the live tensor's trailing shape
            spec = _dc_replace(spec, element_shape=tuple(tensor.shape[1:]))
        self.columns[key] = tensor
        self.schema[key] = spec

    def get(self, key: str, default=None):
        return self.columns.get(key, default)

    # ---- categorical helpers (string-table, cf. io/_metadata.py) -------

    def set_categorical(self, key: str, values: list[str]) -> None:
        """Store a list of strings as an int32 index column + decode table."""
        if len(values) != (self._n or len(values)):
            raise ValueError(
                f"categorical {key!r} has {len(values)} entries; level has {self._n}"
            )
        table = list(dict.fromkeys(values))  # unique, order-preserving
        lookup = {s: i for i, s in enumerate(table)}
        idx = torch.tensor([lookup[s] for s in values], dtype=torch.int32)
        self.__setitem__(
            key, idx, ColumnSpec(dtype=torch.int32, role="categorical", string_table=table)
        )

    def get_categorical(self, key: str) -> list[str]:
        """Decode a categorical column back to a list of strings."""
        spec = self.schema[key]
        if spec.role != "categorical" or spec.string_table is None:
            raise ValueError(f"column {key!r} is not categorical")
        table = spec.string_table
        return [table[i] for i in self.columns[key].tolist()]

    # ---- immutable transforms -----------------------------------------

    def select(self, index: torch.Tensor) -> "Level":
        """Return a new Level with every column advanced-indexed on dim 0.

        ``index`` may be a bool mask or a long index tensor. Columns tagged
        ``role="derived"`` are dropped (their values are order/content-dependent
        and must be recomputed).
        """
        out = Level()
        for name, tensor in self.columns.items():
            spec = self.schema[name]
            if spec.role == "derived":
                continue
            out.__setitem__(name, tensor[index], spec)
        if not out.columns:
            # preserve row count even when nothing carries over
            if index.dtype == torch.bool:
                out._n = int(index.sum().item())
            else:
                out._n = int(index.shape[0])
        return out

    gather = select  # same op; alias for intent clarity

    def to(self, device) -> "Level":
        out = Level(n=self._n)
        for name, tensor in self.columns.items():
            out.__setitem__(name, tensor.to(device), self.schema[name])
        return out

    @classmethod
    def concat(cls, levels: list["Level"]) -> "Level":
        """Row-wise concatenate levels that share the same column set.

        dtype and ``element_shape`` must agree per column. Categorical columns
        are remapped onto a unified string table (different shards may carry
        different tables).
        """
        levels = [lv for lv in levels if lv._n != 0 or lv.columns]
        if not levels:
            return cls(n=0)
        if len(levels) == 1:
            return levels[0]

        ref = levels[0]
        ref_keys = set(ref.columns.keys())
        for lv in levels[1:]:
            if set(lv.columns.keys()) != ref_keys:
                raise ValueError(
                    "cannot concat levels with differing column sets: "
                    f"{sorted(ref_keys)} vs {sorted(lv.columns.keys())}"
                )

        out = Level()
        for name in ref.columns:
            spec = ref.schema[name]
            for lv in levels[1:]:
                other = lv.schema[name]
                if other.dtype != spec.dtype:
                    raise ValueError(
                        f"column {name!r} dtype mismatch: {spec.dtype} vs {other.dtype}"
                    )
                if other.element_shape != spec.element_shape:
                    raise ValueError(
                        f"column {name!r} element_shape mismatch: "
                        f"{spec.element_shape} vs {other.element_shape}"
                    )

            if spec.role == "categorical":
                table: list[str] = []
                lookup: dict[str, int] = {}
                remapped = []
                for lv in levels:
                    ltable = lv.schema[name].string_table or []
                    for s in ltable:
                        if s not in lookup:
                            lookup[s] = len(table)
                            table.append(s)
                    old = lv.columns[name]
                    remap = torch.tensor(
                        [lookup[s] for s in ltable], dtype=torch.int32, device=old.device
                    )
                    remapped.append(remap[old.long()].to(torch.int32))
                out.__setitem__(
                    name,
                    torch.cat(remapped, dim=0),
                    ColumnSpec(dtype=torch.int32, role="categorical", string_table=table),
                )
            else:
                out.__setitem__(
                    name, torch.cat([lv.columns[name] for lv in levels], dim=0), spec
                )
        return out

    # ---- persistence helpers (file orchestration lives in BaseStream) --

    def state_dict(self, prefix: str) -> tuple[dict[str, torch.Tensor], dict]:
        """Return (tensors keyed ``prefix/col``, schema) for safetensors save.

        int16 columns are upcast to int32 on the way out; the canonical dtype is
        recorded in the per-column schema so :meth:`from_state` restores it.
        """
        tensors: dict[str, torch.Tensor] = {}
        schema: dict = {"n": self._n, "columns": {}}
        for name, tensor in self.columns.items():
            spec = self.schema[name]
            out = tensor
            save_dtype = _UPCAST_ON_SAVE.get(tensor.dtype)
            if save_dtype is not None:
                out = out.to(save_dtype)
            tensors[f"{prefix}/{name}"] = out.contiguous().cpu()
            schema["columns"][name] = spec.to_dict()
        return tensors, schema

    @classmethod
    def from_state(
        cls, prefix: str, tensors: dict[str, torch.Tensor], schema: dict
    ) -> "Level":
        out = cls(n=schema.get("n"))
        for name, spec_d in schema["columns"].items():
            spec = ColumnSpec.from_dict(spec_d)
            t = tensors[f"{prefix}/{name}"]
            if t.dtype != spec.dtype:
                t = t.to(spec.dtype)
            out.__setitem__(name, t, spec)
        return out
