"""Settings for :func:`torchsx.merging.observations.prepare_observations`.

Everything a stream needs on the way to a ``PreparedObs``: which reflections
survive, how the resolution shells are cut for the scatter-width fit, and how
detector counts are converted to photons.  Nothing here reaches training or the
merge — those read the CLI flags and the checkpoint recipe.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MergeConfig:
    """Observation-preparation settings.

    Attributes
    ----------
    min_obs
        Drop a reflection unless it has at least this many observations with
        positive intensity.  ``0`` keeps every reflection.
    n_shells
        Equal-population resolution shells used to fit the pooled GenGauss
        scatter width beta.  Reflections are ranked by d-spacing and split into
        this many bins.
    snr_thresh
        Minimum pooled I/sigma for a reflection to enter that beta fit.  Weak
        reflections would otherwise pull the fitted width toward the
        background.
    beta_pool_cap
        Cap on the observations pooled per shell for the beta fit, taken as an
        evenly spaced stride so the reads stay sequential.  ``0`` pools every
        observation.  A cap shifts the fitted beta slightly, so a run that
        needs to reproduce an uncapped number must leave it at ``0``.
    gain
        ADU per photon.  ``"stream"`` reads the declared ``adu_per_photon``, or
        ``adu_per_eV`` times the mean recorded beam energy; ``"empirical"``
        runs the Mandel calibration on negatives; a float pins it.
    fw_n_shells
        Equal-population resolution shells for the French-Wilson prior
        metadata attached to each surviving reflection.
    """

    min_obs: int = 0
    n_shells: int = 20
    snr_thresh: float = 3.0
    beta_pool_cap: int = 0
    gain: str | float = "stream"
    fw_n_shells: int = 20
