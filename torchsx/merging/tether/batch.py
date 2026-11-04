"""The set of observations the tether's terms are evaluated on.

Built once per training batch, or once over a whole merge, and shared by every
term: the scale prediction, each observation's reflection and crystal, and the
smooth per-reflection basis.  A term that needs further quantities of the
observations -- their counts, or a covariate built from them -- takes them from
here, so that they are formed once for all terms.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["TetherBatch"]


class TetherBatch:
    """One set of observations as the tether sees it.

    ``log_K`` the scale prediction (in the graph during training), each
    observation's reflection and crystal, the number of crystals, the smooth
    per-reflection basis ``phi`` (``n_hkl x p``, without its constant column)
    and, with pseudo-translation classes, their per-reflection columns ``phase``
    (:meth:`~torchsx.merging.scale.pseudo_translation.PhaseClasses.columns`).
    """

    def __init__(self, log_K: Tensor, hkl_idx: Tensor, crystal_id: Tensor, n_crystals: int, phi: Tensor,
                 *, phase: Tensor | None = None):
        self.log_K = log_K
        self.hkl_idx = hkl_idx.long()
        self.crystal_id = crystal_id.long()
        self.n_crystals = int(n_crystals)
        self.phi = phi
        self.phase = (torch.zeros(phi.shape[0], 0, device=phi.device, dtype=phi.dtype) if phase is None
                      else phase.to(device=phi.device, dtype=phi.dtype))

    @property
    def n_hkl(self) -> int:
        return self.phi.shape[0]
