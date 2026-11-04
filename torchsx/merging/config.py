"""Settings for :func:`torchsx.merging.observations.prepare_observations`.

Everything a stream needs on the way to a ``PreparedObs``: which reflections
survive and how detector counts are converted to photons.  Nothing here reaches
training or the merge — those read the CLI flags and the checkpoint recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from torchsx.measurement import DEFAULT_GAIN_TOLERANCE


@dataclass
class MergeConfig:
    """Observation-preparation settings.

    Attributes
    ----------
    min_obs
        Drop a reflection unless it has at least this many observations with
        positive intensity.  ``0`` keeps every reflection.
    gain
        ADU per photon: auto compares input and calibrated estimates;
        trust_input uses the declaration; calibrate forces estimation; a number
        supplies the gain directly. Auto uses a 50% discrepancy threshold.
    fw_n_shells
        Equal-population resolution shells for the French-Wilson prior
        metadata attached to each surviving reflection.
    disk_size
        Disk pixels per observation: ``nominal`` (the ring mask) or ``recorded``
        (the source's own count, probixi's ``n_pixels``).
    background
        ``annulus`` (ring mean times disk pixels) or ``model`` (the source's
        background model, probixi's ``background_model``).
    per_image_k0
        Ewald offsets at each image's own wavelength instead of the median.
    """

    min_obs: int = 0
    gain: str | float = "auto"
    gain_tolerance: float = DEFAULT_GAIN_TOLERANCE
    fw_n_shells: int = 20
    disk_size: str = "nominal"
    background: str = "annulus"
    per_image_k0: bool = False
