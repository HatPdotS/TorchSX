"""Dirichlet smoothing over the ±1 Miller-index adjacency graph.

Absorbs the three hand-rolled per-parameter Dirichlet blocks that lived
in ``merging_clean.loss.regularizers_only`` (on ``log_I0``, ``log_alpha``,
``log_beta``) into one parameterised regulariser.
"""

from __future__ import annotations

from typing import Literal

from torch import Tensor

from torchsx.merging.indexing.adjacency import Adjacency, dirichlet_energy
from torchsx.merging.parameters import Parameters
from torchsx.merging.regularizers.base import Regularizer

ParamName = Literal["log_I0", "log_alpha", "log_beta"]


class MillerGraphSmoothness(Regularizer):
    """Σ_(i,j)∈E (params.<target>[i] − params.<target>[j])².

    Parameters
    ----------
    adjacency : ±1 Miller adjacency graph over the valid HKLs
        (from :func:`torchsx.merging.indexing.build_adjacency`).
    target : which per-HKL parameter field to smooth (``log_I0``,
        ``log_alpha``, or ``log_beta``).
    """

    def __init__(self, adjacency: Adjacency, target: ParamName) -> None:
        super().__init__()
        self.register_buffer("edge_src", adjacency.edge_src)
        self.register_buffer("edge_dst", adjacency.edge_dst)
        self.target = target

    def forward(self, params: Parameters) -> Tensor:
        adj = Adjacency(
            self.edge_src, self.edge_dst,
            # n_neighbors is not used by dirichlet_energy; pass an empty tensor.
            self.edge_src.new_empty(0, dtype=self.edge_src.dtype).float(),
        )
        values = getattr(params, self.target)
        return dirichlet_energy(values, adj)
