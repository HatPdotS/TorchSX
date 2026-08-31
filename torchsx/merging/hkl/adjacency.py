"""±1 Miller-index adjacency graph + Dirichlet-energy helper.

The graph itself is a pure indexing primitive (it describes structure of the
HKL set, not any statistical model).  ``MillerGraphSmoothness`` in
``torchsx.merging.regularizers`` wraps it into a regulariser.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Adjacency:
    """Directed edge list over a list of HKLs (edge src → dst).

    Each undirected pair of ±1 Miller neighbours shows up twice (once each
    direction); that is a deliberate symmetry — downstream penalties like
    ``dirichlet_energy`` just absorb the factor of 2 into the weight.
    """

    edge_src: Tensor          # (E,) long
    edge_dst: Tensor          # (E,) long
    n_neighbors: Tensor       # (N,) float — out-degree per node

    def to(self, device) -> "Adjacency":
        return Adjacency(
            edge_src=self.edge_src.to(device),
            edge_dst=self.edge_dst.to(device),
            n_neighbors=self.n_neighbors.to(device),
        )


def build_adjacency(hkl: Tensor) -> Adjacency:
    """Build the ±1 Miller-index adjacency graph for a list of HKLs.

    Parameters
    ----------
    hkl : (N, 3) integer tensor.
    """
    n = hkl.shape[0]
    lookup: dict[tuple[int, int, int], int] = {}
    for i in range(n):
        key = (int(hkl[i, 0]), int(hkl[i, 1]), int(hkl[i, 2]))
        lookup[key] = i

    deltas = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    edge_src: list[int] = []
    edge_dst: list[int] = []
    for i in range(n):
        h, k, l = int(hkl[i, 0]), int(hkl[i, 1]), int(hkl[i, 2])
        for dh, dk, dl in deltas:
            j = lookup.get((h + dh, k + dk, l + dl))
            if j is not None:
                edge_src.append(j)
                edge_dst.append(i)

    edge_src_t = torch.tensor(edge_src, dtype=torch.long)
    edge_dst_t = torch.tensor(edge_dst, dtype=torch.long)
    n_neighbors = torch.bincount(edge_dst_t, minlength=n).float().clamp(min=1)
    return Adjacency(edge_src=edge_src_t, edge_dst=edge_dst_t, n_neighbors=n_neighbors)


def dirichlet_energy(values: Tensor, adj: Adjacency) -> Tensor:
    """Σ_(i,j)∈E (values[i] − values[j])².  Each undirected edge is counted
    twice (see :class:`Adjacency`)."""
    diff = values[adj.edge_src] - values[adj.edge_dst]
    return (diff * diff).sum()
