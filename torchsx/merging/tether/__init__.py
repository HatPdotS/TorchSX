"""The sum-of-spots tether: directions the likelihood leaves to the model, held at the raw counts.

The likelihood sees an observation only through ``K I0``, so some directions of
the model are not fixed by the fit: the smooth gauge
``K -> K g(h), I0 -> I0 / g(h)`` exactly.  A model left to choose there chooses
for reasons unrelated to the data, and differently from run to run.  The raw
counts fix it: in rotation data the full intensity is the sum of a reflection's
partials, and the observations of a reflection in still images sample the
excitation error evenly over a window that covers the whole rocking curve, so
the arithmetic mean of ``K`` over them is that integral -- the sum of the
spots.

The tether holds the model to it, as a penalty in the training loss computed per
batch in closed form, from the batch alone.  It is not a prior on the
intensities: its weight does not fall as the data grow, and at its minimum
every term is where the counts put it.  Each direction is a term
(:mod:`~torchsx.merging.tether.terms`): the arithmetic mean of ``K`` carries no
trend along the term's columns (:mod:`~torchsx.merging.tether.kmean`).  The
smooth reciprocal basis is the gauge term, which pins the rocking-curve integral
at every resolution and in every direction; the pseudo-translation classes are
the phase term, which pins it in every family of reflections.  The quantities the terms share are
built once per batch (:class:`~torchsx.merging.tether.batch.TetherBatch`).
"""

from torchsx.merging.tether.batch import TetherBatch
from torchsx.merging.tether.kmean import (
    integral_mode,
    integral_step,
    mode_mean_square,
    psd_solve,
    smooth_columns,
    step_mean_square,
)
from torchsx.merging.tether.terms import TERMS, KMeanTerm, PhaseGauge, SmoothGauge
from torchsx.merging.tether.tether import Tether, parse_terms

__all__ = [
    "TERMS",
    "KMeanTerm",
    "PhaseGauge",
    "SmoothGauge",
    "Tether",
    "TetherBatch",
    "integral_mode",
    "integral_step",
    "mode_mean_square",
    "parse_terms",
    "psd_solve",
    "smooth_columns",
    "step_mean_square",
]
