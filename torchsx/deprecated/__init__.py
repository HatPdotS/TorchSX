"""Deprecated legacy subpackages.

The modules under this namespace predate the modular ``torchsx.merging``
pipeline and are no longer part of the active merge path.  They are kept
importable so older experiment scripts keep running; new code should not
depend on anything here.

Subpackages:
- ``torchsx.deprecated.merge``      — pre-``merging_clean`` EM merger, neural
                                      corrections, noise model, structure-factor
                                      utilities.  The *live* indexing helpers
                                      were lifted into
                                      ``torchsx.merging.indexing``.
- ``torchsx.deprecated.partiality`` — older partiality modelling pipeline.
- ``torchsx.deprecated.forward``    — pixel forward-model / spot-prediction.
- ``torchsx.deprecated.scaling``    — older composite scaling loss.
- ``torchsx.deprecated.target``     — older scaling targets (KL etc.).
- ``torchsx.deprecated.cli``        — CLI scripts that depend on the
                                      deprecated packages above.
"""

import warnings as _warnings

_warnings.warn(
    "torchsx.deprecated is a legacy namespace kept only for backward "
    "compatibility with older experiment scripts; prefer torchsx.merging "
    "for current work.",
    DeprecationWarning,
    stacklevel=2,
)
