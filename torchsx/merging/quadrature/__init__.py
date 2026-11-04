"""Latent-signal quadrature: integrates ``S`` out of the joint per observation.

:class:`~torchsx.merging.quadrature.split_log_grid.SplitLogGrid` places its
nodes per observation and carries the two-width and mixture scatter paths.  It
has Triton fast paths and an eager fallback that must agree.
"""

from torchsx.merging.quadrature.base import Quadrature
from torchsx.merging.quadrature.split_log_grid import SplitLogGrid

__all__ = ["Quadrature", "SplitLogGrid"]
