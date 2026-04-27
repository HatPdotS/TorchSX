"""Base class for merging-loss targets.

Inherits from the generic ``torchref.refinement.targets.base.Target`` so
``LossState`` sees a standard callable-returning-scalar interface.  The
domain-specific helpers (data, prior, regulariser, scale-reg) subclass
this.
"""

from __future__ import annotations

from torchref.refinement.targets.base import Target


class MergingTarget(Target):
    """Thin subclass of torchref's ``Target`` for the SFX merging pipeline.

    We inherit the ``forward() → scalar`` contract and the ``LossState``
    integration hooks; no additional state is needed at this layer.
    """

    name: str = "merging/base"

    def __init__(self) -> None:
        super().__init__(verbose=0, target_value=0.0, sigma=1.0)
