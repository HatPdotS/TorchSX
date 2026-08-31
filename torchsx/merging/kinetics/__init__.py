"""Kinetic-ensemble merging: complex F per state, time-resolved occupancies.

This subpackage extends the scalar single-dataset merge (``Parameters``,
``DataNLLTarget``) to a kinetic ensemble where the observed structure
factor is a complex sum over kinetic states with time-dependent
occupancies:

.. math::

    F_{\\text{obs}}(h, t) = \\sum_k w_k(t) \\cdot F_k(h) \\in \\mathbb C
    \\qquad
    I_{\\text{obs}}(h, t) = \\bigl|F_{\\text{obs}}(h, t)\\bigr|^2.

Per-HKL phase gauge: the dark state ``F_dark(h) \\in \\mathbb R_+`` is held
real and non-negative; relative phases of excited states are refined via
their imaginary parts (``F_im_k``), with a soft restraint keeping the
phase rotation small.

The kinetic occupancies + Gaussian instrument response function are
provided by :class:`torchref.kinetic.KineticModel` — already a torchsx
dependency.  This subpackage glues the kinetic populations to the
per-observation forward model and provides a sigma-aware initial fit
against per-timepoint pre-merged amplitudes.
"""

from torchsx.merging.kinetics.parameters import KineticEnsembleParameters
from torchsx.merging.kinetics.mixing import KineticMixing
from torchsx.merging.kinetics.phase_restraint import PhaseRestraint
from torchsx.merging.kinetics.regularizers import ExcitedAmplitudeAnchor
from torchsx.merging.kinetics.initial_fit import (
    PerBinAmplitudes,
    fit_initial_kinetic_ensemble,
    per_bin_premerging,
)
from torchsx.merging.kinetics.data_target import KineticDataNLLTarget
from torchsx.merging.kinetics.loss import KineticLogSpaceLoss
from torchsx.merging.kinetics.output import (
    write_kinetic_params_json,
    write_per_state_mtzs,
    write_predicted_bin_mtzs,
)

__all__ = [
    "KineticEnsembleParameters",
    "KineticMixing",
    "PhaseRestraint",
    "ExcitedAmplitudeAnchor",
    "PerBinAmplitudes",
    "fit_initial_kinetic_ensemble",
    "per_bin_premerging",
    "KineticDataNLLTarget",
    "KineticLogSpaceLoss",
    "write_kinetic_params_json",
    "write_per_state_mtzs",
    "write_predicted_bin_mtzs",
]
