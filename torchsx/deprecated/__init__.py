"""Deprecated legacy subpackages.

The modules here predate the modular ``torchsx.merging`` pipeline and are not on
the active merge path. No *live* ``torchsx`` code imports them; they are kept
importable only because parts of the test suite still exercise the science they
implement. New code should not depend on anything here.

Status of each subpackage (see also each subpackage's ``README.md``):

FROZEN — still unique, NO live replacement (kept until a live model exists):
- ``torchsx.deprecated.forward``    — pixel-level differentiable forward model /
                                      spot prediction. No equivalent in
                                      ``torchsx.merging``. Exercised by
                                      ``test_forward*``, ``test_sparse_pixel``,
                                      ``test_soft_assignment``, ``test_profile_fitted``,
                                      ``test_censored_poisson``, ``test_attraction_loss``.
- ``torchsx.deprecated.partiality`` — profile-fit / excitation-error partiality
                                      models (``test_partiality``).
- ``torchsx.deprecated.merge``      — empirical-distribution fitting + EM merger
                                      (``_distribution``/``_em``/``_corrections``/
                                      ``_neural*``; ``test_distribution``).

SUPERSEDED — a live equivalent exists, but the deprecated copy is RETAINED:
the frozen subpackages above (and many ``experiments/`` scripts) still import
these leaves, so deleting them would cascade into code with no live
replacement. Their *tests* were repointed at the live modules, so coverage now
tracks the maintained implementation:
- ``torchsx.deprecated.merge._filter`` / ``_hkl`` / ``_stats``
                                      → tests use ``torchsx.merging.hkl.{filter,hkl,stats}``.
- ``torchsx.deprecated.scaling._absolute`` / ``_bscale`` / ``_loss``
                                      ↔ ``torchsx.merging.scale.*`` (not a 1:1 API;
                                      ``test_forward`` still consumes these).
- ``torchsx.deprecated.target._kl`` / ``_self_consistent``
                                      ↔ ``torchsx.merging.targets.data`` (conceptual,
                                      not drop-in).

These can only be deleted once ``forward`` / ``partiality`` gain live
replacements and the experiment scripts are migrated.
"""

import warnings as _warnings

_warnings.warn(
    "torchsx.deprecated is a legacy namespace kept only for backward "
    "compatibility with older experiment scripts; prefer torchsx.merging "
    "for current work.",
    DeprecationWarning,
    stacklevel=2,
)
