"""Latent-signal quadrature: integrates ``S`` out of the joint per observation.

Two grids are in use.  :class:`~torchsx.merging.quadrature.split_log_grid.SplitLogGrid`
places its nodes per observation and carries the two-width and mixture scatter
paths; it is what training integrates against.
:class:`~torchsx.merging.quadrature.fixed_log_grid.FixedLogGrid` shares one wide
grid across observations, which is what the posterior readout wants.  Both have
Triton fast paths and eager fallbacks that must agree.
"""

from torchsx.merging.quadrature.base import Quadrature
from torchsx.merging.quadrature.fixed_log_grid import FixedLogGrid
from torchsx.merging.quadrature.split_log_grid import SplitLogGrid

__all__ = ["FixedLogGrid", "Quadrature", "SplitLogGrid"]
