"""The sum-of-spots tether: directions the likelihood leaves to the model, held at the raw counts.

The likelihood sees an observation through ``K I0`` and a scatter model, and
some directions in that parametrisation are fixed weakly or not at all by the
fit: the smooth gauge ``K -> K g(h), I0 -> I0 / g(h)`` exactly, how much of an
observation's dimness the scale and how much the scatter explains nearly.  A
model left to choose there chooses for reasons unrelated to the data, and
differently from run to run.  The raw counts fix these directions: the sum of a
reflection's spots, ``sum (n - bg)``, depends on no model.

The tether holds the model to it in each such direction, as a penalty in the
training loss computed per batch in closed form, from the batch alone.  It is
not a prior on the intensities -- its weight does not fall as the data grow,
and at its minimum every term is where the data put it.  Each direction is a
term (:mod:`~torchsx.merging.tether.terms`) of one of two families:

* the arithmetic mean of ``K`` carries no trend along given columns
  (:mod:`~torchsx.merging.tether.kmean`): the smooth reciprocal basis, which
  pins the rocking-curve integral at every resolution, and the reflection's
  strength;
* the likelihood's estimating equation at each group's sum of spots carries no
  trend along a covariate (:mod:`~torchsx.merging.tether.response`): the
  reflection's strength.

The raw-count quantities the terms share -- parts of the crystals, sums of
spots, the cross-fitted strength covariate -- are built once per batch
(:class:`~torchsx.merging.tether.batch.TetherBatch`).
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
from torchsx.merging.tether.response import response_slope, slope_curvature
from torchsx.merging.tether.terms import (
    TERMS,
    KMeanTerm,
    ResponseTerm,
    SmoothGauge,
    StrengthResponse,
    StrengthSlope,
)
from torchsx.merging.tether.tether import Curves, Tether, parse_terms

__all__ = [
    "TERMS",
    "Curves",
    "KMeanTerm",
    "ResponseTerm",
    "SmoothGauge",
    "StrengthResponse",
    "StrengthSlope",
    "Tether",
    "TetherBatch",
    "integral_mode",
    "integral_step",
    "mode_mean_square",
    "parse_terms",
    "psd_solve",
    "response_slope",
    "slope_curvature",
    "smooth_columns",
    "step_mean_square",
]
