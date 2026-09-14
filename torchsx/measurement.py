"""Measurement semantics and recipe resolution, independent of the producer.

Adapters attach this small, versioned record to the existing stream metadata.
It describes the *stored* values; conversion to photon counts happens only in
``prepare_observations``. A run is assumed to use one integration recipe.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math


MEASUREMENT_KEY = "_torchsx_measurement"
GAIN_POLICY_VERSION = 1
DEFAULT_GAIN_TOLERANCE = 0.5


@dataclass(frozen=True)
class MeasurementSpec:
    schema_version: int = 1
    producer: str = "unknown"
    producer_version: str | None = None
    integration_method: str | None = None
    integration_modifiers: tuple[str, ...] = ()
    intensity_unit: str = "unknown"
    intensity_definition: str = "unknown"
    background_unit: str = "unknown"
    background_definition: str = "unknown"
    uncertainty_definition: str = "unknown"
    # (scope: 'global' or panel name, convention, value); None means unparsed.
    gain_declarations: tuple[tuple[str, str, float | None], ...] = ()
    radii: tuple[float, float, float] | None = None
    nominal_n_peak: int | None = None
    nominal_n_bg: int | None = None
    pixel_count_definition: str = "unknown"
    upstream_corrections: tuple[str, ...] = ()
    source_files: tuple[str, ...] = ()
    provenance: tuple[tuple[str, str], ...] = ()
    unresolved: tuple[str, ...] = ()

    def __post_init__(self):
        for key in ("integration_modifiers", "upstream_corrections", "source_files", "unresolved"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        for key in ("gain_declarations", "provenance"):
            object.__setattr__(self, key, tuple(tuple(row) for row in getattr(self, key)))
        if self.radii is not None:
            object.__setattr__(self, "radii", tuple(self.radii))

    def to_dict(self) -> dict:
        # JSON-shaped, including tuples, so checkpoint/cache equality survives
        # JSON round trips without float32 conversion of recipe values.
        import json
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, record: dict) -> MeasurementSpec:
        if record.get("schema_version") != 1:
            raise ValueError(f"Unsupported measurement schema {record.get('schema_version')!r}")
        return cls(**record)

    @classmethod
    def from_metadata(cls, metadata) -> MeasurementSpec | None:
        entry = metadata.get(MEASUREMENT_KEY)
        return cls.from_dict(entry["record"]) if entry else None


def combine_measurements(specs) -> MeasurementSpec | None:
    """Retain semantics and input provenance for a homogeneous run."""
    specs = list(specs)
    if not specs or any(s is None for s in specs):
        return None
    return replace(specs[0], source_files=tuple(dict.fromkeys(
        path for spec in specs for path in spec.source_files)),
        provenance=tuple(dict.fromkeys(item for spec in specs for item in spec.provenance)))


def parse_radii(value) -> tuple[float, float, float]:
    parts = ([float(x) for x in value.replace(":", ",").split(",")]
             if isinstance(value, str) else [float(x) for x in value])
    if (len(parts) != 3 or not all(math.isfinite(x) for x in parts)
            or not 0 <= parts[0] < parts[1] <= parts[2]):
        raise ValueError(f"--int-radius wants 0 <= r_peak < r_gap <= r_bg, got {value!r}")
    return tuple(parts)


def parse_gain(value) -> str | float:
    if value is None:
        return "auto"
    if str(value) == "stream":  # Older recipes used this name for trust_input.
        return "trust_input"
    if str(value) in ("trust_input", "auto", "calibrate"):
        return str(value)
    error = (f"Invalid --gain {value!r}. Expected 'auto' (default), 'trust_input', "
             "'calibrate', or a positive finite integer/float (ADU/photon).")
    try:
        gain = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(error) from exc
    if isinstance(value, bool) or not math.isfinite(gain) or gain <= 0:
        raise ValueError(error)
    return gain


def validate_measurement(spec: MeasurementSpec | None, *, explicit_rings: bool) -> None:
    """Gate the present likelihood by semantics, never by producer name."""
    if spec is None:  # legacy cache/reader, with an explicit historical recipe
        if explicit_rings:
            return
        raise ValueError("Measurement metadata unavailable; supply --int-radius and --gain.")
    if spec.schema_version != 1:
        raise ValueError(f"Unsupported measurement schema {spec.schema_version}")
    expected = {
        "intensity_definition": "background_subtracted_sum",
        "background_definition": "local_mean_per_pixel",
        "uncertainty_definition": "supplied_sigma_of_intensity",
    }
    bad = [key for key, val in expected.items() if getattr(spec, key) != val]
    if (spec.intensity_unit, spec.background_unit) not in (("adu", "adu/pixel"),
                                                         ("photon", "photon/pixel")):
        bad.append("units")
    if spec.upstream_corrections:
        bad.append("upstream_corrections")
    if spec.integration_method not in (None, "rings"):
        bad.append("integration_method")
    if bad:
        raise ValueError("Unsupported measurement semantics: " + ", ".join(bad))
    if (spec.integration_method is None or "integration_modifiers" in spec.unresolved) and not explicit_rings:
        raise ValueError("Integration recipe is unresolved; confirm the ring recipe with --int-radius.")


@dataclass(frozen=True)
class ResolvedMeasurement:
    radii: tuple[float, float, float]
    gain: str | float
    provenance: tuple[tuple[str, str], ...]
    overrides: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"schema_version": 1, "radii": list(self.radii), "gain": self.gain,
                "provenance": dict(self.provenance), "overrides": list(self.overrides)}


def resolve_measurement(spec: MeasurementSpec | None, *, radii=None, gain=None,
                        defaults: dict | None = None) -> ResolvedMeasurement:
    """Explicit values > checkpoint recipe > automatic source-gain validation.

    Gain modes remain requests for the existing preparation-time calibration.
    New checkpoints also record the numeric result, so their recipe is reusable.
    """
    d = defaults or {}
    checkpoint_gain = (d.get("measurement") or {}).get("gain_adu_per_photon", d.get("gain"))
    source_radii = spec.radii if spec else None
    checkpoint_radii = d.get("radii")
    resolved_radii = radii if radii is not None else (
        checkpoint_radii if checkpoint_radii is not None else source_radii)
    if resolved_radii is None:
        raise ValueError("Integration radii unavailable; supply --int-radius R_PEAK,R_GAP,R_BG.")
    resolved_radii = parse_radii(resolved_radii)
    from torchsx.merging.hkl.noise import _crystfel_pixel_counts
    if min(_crystfel_pixel_counts(*resolved_radii)) <= 0:
        raise ValueError("--int-radius must define nonempty peak and background masks.")
    explicit_rings = radii is not None or d.get("radii") is not None
    validate_measurement(spec, explicit_rings=explicit_rings)
    resolved_gain = parse_gain(gain if gain is not None else checkpoint_gain)
    legacy_stream_request = spec is None and resolved_gain == "trust_input"
    if resolved_gain == "trust_input" and not legacy_stream_request and (spec is None or (
            spec.intensity_unit != "photon" and not spec.gain_declarations)):
        raise ValueError("Detector gain unavailable; supply --gain (ADU/photon), auto or calibrate.")
    provenance = (("radii", "explicit" if radii is not None else "checkpoint" if d.get("radii") is not None else "source"),
                  ("gain", "explicit" if gain is not None else "checkpoint" if checkpoint_gain is not None else "source"))
    overrides = []
    if source_radii is not None and resolved_radii != source_radii:
        overrides.append(f"radii {source_radii} from source -> {resolved_radii}")
    if gain is not None and checkpoint_gain is not None and parse_gain(gain) != parse_gain(checkpoint_gain):
        overrides.append(f"checkpoint gain {checkpoint_gain} -> {resolved_gain}")
    if resolved_gain not in ("trust_input", "auto") and spec and spec.gain_declarations:
        overrides.append(f"source gain declarations overridden by {resolved_gain}")
    return ResolvedMeasurement(resolved_radii, resolved_gain, provenance, tuple(overrides))


def legacy_stream_gain(core) -> float:
    """Retain the old direct-reader/explicit legacy-recipe behavior visibly.

    The new CLI resolver never selects this path implicitly for missing metadata.
    Probixi keeps its previous preparation behavior until its adapter is added.
    """
    import warnings
    warnings.warn("Legacy input has no measurement specification; using the historical "
                  "geometry-based gain, whose missing declarations cannot be checked. "
                  "Supply an explicit gain for a reproducible recipe.", stacklevel=2)
    adu_per_eV = float(getattr(core.geometry, "adu_per_eV", 1.0) or 1.0)
    gain = adu_per_eV * float(core.beam_energy.float().mean().item())
    return gain if gain > 0 else 1.0


def gain_from_measurement(spec: MeasurementSpec | None, beam_energy_eV: float) -> float:
    """Resolve one run-wide gain without inventing a missing declaration."""
    if spec is None:
        raise ValueError("Legacy input has no gain declaration; supply an explicit --gain.")
    if spec.intensity_unit == "photon":
        return 1.0
    values = []
    for scope, convention, value in spec.gain_declarations:
        if value is None or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid {scope}/{convention} gain declaration; supply --gain.")
        if convention == "adu_per_eV":
            value *= beam_energy_eV
        elif convention != "adu_per_photon":
            raise ValueError(f"Unsupported gain convention {convention!r}; supply --gain.")
        values.append(value)
    if not values or not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("Gain/beam energy unavailable or invalid; supply --gain.")
    if not all(math.isclose(v, values[0], rel_tol=1e-6) for v in values):
        raise ValueError("Gain declarations do not resolve to one scalar; supply --gain.")
    return values[0]
