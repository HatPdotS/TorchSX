"""Single source of truth for the spot-table column schema.

A *spot table* is the common currency of the data-reduction pipeline: a
``(M, C)`` float tensor with one row per detected spot. Both the pf8/threshold
finders (:mod:`spotfinding`) and the native-torch / probabilistic finders
(:mod:`peakfinder`) emit this layout, and everything downstream
(:mod:`to_stream`, indexing) reads it by these column indices.

Column legend::

    0  PANEL      panel index
    1  SS         assembled slow-scan row (panel-offset already folded in)
    2  FS         fast-scan column within the panel
    3  INTENSITY  integrated peak intensity (photons)
    4  PROB       per-peak probability in [0, 1]  (probabilistic finder only)

Threshold / pf8 finders emit ``(M, 4)`` (``PANEL..INTENSITY``); the
probabilistic finder emits ``(M, 5)`` (adds ``PROB``). Downstream code probes
``table.shape[1] > PROB`` to tell the two apart.
"""

from __future__ import annotations

PANEL, SS, FS, INTENSITY, PROB = 0, 1, 2, 3, 4

N_BASE_COLS = 4  # PANEL, SS, FS, INTENSITY
N_PROB_COLS = 5  # + PROB

__all__ = [
    "PANEL",
    "SS",
    "FS",
    "INTENSITY",
    "PROB",
    "N_BASE_COLS",
    "N_PROB_COLS",
]
