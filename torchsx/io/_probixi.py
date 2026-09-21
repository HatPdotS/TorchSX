from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

PROBIXI_COLUMNS = ("n_pix", "bg_model", "bg_model_var")


def read_reflection_columns(path: str | Path, names: tuple[str, ...], n_obs: int) -> dict[str, torch.Tensor]:
    """``{name: (n_obs,) float32}`` for the trailing columns of every reflection row.

    A row lacking the extras (a crystal written without prediction) yields NaN,
    which the observation filter later rejects in a mode that needs them.
    """
    names = tuple(names)
    if not names:
        return {}
    out = np.full((len(names), n_obs), np.nan, dtype=np.float32)
    i = 0
    in_table = False
    with open(path, "rb") as f:
        for raw in f:
            line = raw.rstrip(b"\r\n").decode("utf-8", "replace")
            if not in_table:
                if line.startswith("Reflections measured after indexing"):
                    in_table = True
                    header = True
                continue
            if line.startswith("End of reflections"):
                in_table = False
                continue
            if header:               # the column-name line
                header = False
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            try:
                int(parts[0]); int(parts[1]); int(parts[2])
                for k in range(3, 9):
                    float(parts[k])
            except (ValueError, IndexError):
                continue
            if i >= n_obs:
                raise ValueError(f"{path}: more reflection rows than the parsed {n_obs:,} observations")
            if len(parts) >= 10 + len(names):
                for j in range(len(names)):
                    try:
                        out[j, i] = float(parts[10 + j])
                    except ValueError:
                        pass
            i += 1
    if i != n_obs:
        raise ValueError(f"{path}: {i:,} reflection rows but {n_obs:,} parsed observations")
    return {name: torch.from_numpy(out[j]) for j, name in enumerate(names)}
