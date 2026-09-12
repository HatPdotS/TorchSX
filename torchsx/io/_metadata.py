"""StreamMetadata: nested dict ↔ tensor conversion for per-crystal/per-image metadata."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from torchsx._categorical import decode_categorical, encode_categorical


class StreamMetadata(dict):
    """Nested dict-like metadata container with tensor conversion.

    Stores heterogeneous per-crystal/per-image metadata as nested dicts.
    Can be condensed to a flat tensor representation and restored exactly.

    Examples
    --------
    >>> meta = StreamMetadata()
    >>> meta["crystal"] = {"cell": [[14.9, 18.8, 18.9, 89.4, 85.0, 67.8]], "scale": [1.0]}
    >>> meta["image"] = {"filename": ["a.h5", "b.h5"], "energy": [12960.0, 12960.0]}
    >>> tensors, schema = meta.to_tensors()
    >>> meta2 = StreamMetadata.from_tensors(tensors, schema)
    >>> assert meta == meta2
    """

    def to_tensors(self) -> tuple[dict[str, torch.Tensor], dict]:
        """Convert all numeric fields to tensors, encode strings as int tensors.

        Returns (tensor_dict, schema) where schema stores types/shapes for
        exact round-trip reconstruction. String fields are encoded via a
        string table (unique strings → int indices → tensor).
        """
        tensor_dict = {}
        schema = {}

        for top_key, sub_dict in self.items():
            if not isinstance(sub_dict, dict):
                # Scalar or non-dict value — store in schema directly
                schema[top_key] = {"type": "scalar", "value": sub_dict}
                continue
            for sub_key, value in sub_dict.items():
                flat_key = f"{top_key}/{sub_key}"
                entry = _encode_value(value, flat_key, tensor_dict)
                schema[flat_key] = entry

        return tensor_dict, schema

    @classmethod
    def from_tensors(cls, tensors: dict[str, torch.Tensor], schema: dict) -> StreamMetadata:
        """Reconstruct from tensors + schema. Inverse of to_tensors()."""
        result = cls()

        for flat_key, entry in schema.items():
            if "/" not in flat_key:
                result[flat_key] = entry["value"]
                continue

            # Split flat key into top/sub
            parts = flat_key.split("/", 1)
            top_key, sub_key = parts[0], parts[1]

            if top_key not in result:
                result[top_key] = {}

            result[top_key][sub_key] = _decode_value(entry, flat_key, tensors)

        return result

    def to(self, device) -> StreamMetadata:
        """Move all tensor values to device."""
        new = StreamMetadata()
        for top_key, sub_dict in self.items():
            if isinstance(sub_dict, dict):
                new[top_key] = {}
                for sub_key, value in sub_dict.items():
                    if isinstance(value, torch.Tensor):
                        new[top_key][sub_key] = value.to(device)
                    else:
                        new[top_key][sub_key] = value
            else:
                new[top_key] = sub_dict
        return new

    def save(self, path: str | Path):
        """Save via safetensors (tensors) + JSON (schema + string tables)."""
        from safetensors.torch import save_file

        path = Path(path)
        tensors, schema = self.to_tensors()

        if tensors:
            save_file(tensors, str(path.with_suffix(".safetensors")))
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(schema, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> StreamMetadata:
        """Load from saved format."""
        from safetensors.torch import load_file

        path = Path(path)
        tensor_path = path.with_suffix(".safetensors")
        json_path = path.with_suffix(".json")

        with open(json_path) as f:
            schema = json.load(f)

        tensors = {}
        if tensor_path.exists():
            tensors = load_file(str(tensor_path))

        return cls.from_tensors(tensors, schema)


def _encode_value(value, flat_key: str, tensor_dict: dict) -> dict:
    """Encode a single value into tensor_dict and return its schema entry."""
    if isinstance(value, list) and len(value) > 0:
        first = value[0]

        if isinstance(first, str):
            # String table encoding
            unique_strings, indices = encode_categorical(value)
            tensor_dict[flat_key] = indices
            return {
                "type": "string_table",
                "strings": unique_strings,
                "length": len(value),
            }
        elif isinstance(first, (list, tuple)):
            # 2D numeric
            t = torch.tensor(value, dtype=torch.float32)
            tensor_dict[flat_key] = t
            return {
                "type": "tensor",
                "dtype": "float32",
                "shape": list(t.shape),
            }
        elif isinstance(first, float):
            t = torch.tensor(value, dtype=torch.float32)
            tensor_dict[flat_key] = t
            return {"type": "tensor", "dtype": "float32", "shape": list(t.shape)}
        elif isinstance(first, int):
            t = torch.tensor(value, dtype=torch.int64)
            tensor_dict[flat_key] = t
            return {"type": "tensor", "dtype": "int64", "shape": list(t.shape)}
        else:
            # Fallback: try as float tensor
            t = torch.tensor(value, dtype=torch.float32)
            tensor_dict[flat_key] = t
            return {"type": "tensor", "dtype": "float32", "shape": list(t.shape)}
    elif isinstance(value, (int, float)):
        return {"type": "scalar", "value": value}
    elif isinstance(value, list) and len(value) == 0:
        return {"type": "empty_list"}
    else:
        return {"type": "scalar", "value": value}


def _decode_value(entry: dict, flat_key: str, tensors: dict):
    """Decode a schema entry back to the original Python value."""
    vtype = entry["type"]

    if vtype == "scalar":
        return entry["value"]
    elif vtype == "empty_list":
        return []
    elif vtype == "string_table":
        return decode_categorical(entry["strings"], tensors[flat_key])
    elif vtype == "tensor":
        t = tensors[flat_key]
        dtype = entry["dtype"]
        shape = entry["shape"]
        if dtype == "int64":
            # Return as list of ints for exact round-trip
            if len(shape) == 1:
                return t.to(torch.int64).tolist()
            return t.to(torch.int64).tolist()
        else:
            # Return as list of floats
            if len(shape) == 1:
                return t.to(torch.float32).tolist()
            return t.to(torch.float32).tolist()
    else:
        raise ValueError(f"Unknown schema type: {vtype}")
