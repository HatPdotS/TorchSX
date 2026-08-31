"""Output writers for the photon-level kinetic merge.

Three artefacts from a refined :class:`KineticLogSpaceLoss`:

* :func:`write_per_state_mtzs` — one MTZ per kinetic state with
  ``F = |F_s(h)|``.  For sequential A → I → B (``n_excited=2``):

    state_A.mtz : F = F_dark                (ground)
    state_I.mtz : F = √(F_re[0]² + F_im[0]²) (intermediate)
    state_B.mtz : F = √(F_re[1]² + F_im[1]²) (long-time product)

* :func:`write_predicted_bin_mtzs` — for a user-supplied list of
  timepoints ``t_b``, evaluate ``F_total(h, t_b) = Σ_s p_s(t_b)·F_s(h)``
  and write one MTZ per timepoint.  Cross-checks the photon-level
  kinetic merge vs the model-free time-binned merge bin-for-bin.

* :func:`write_kinetic_params_json` — rate constants, time constants,
  IRF width, flow chart, final NLL.

σ on F is the MAP-square-root fallback used by the standard posterior
when full Bayesian uncertainty isn't available: ``σ_F = ½ · σ_logI · F``
with ``σ_logI = 0.1`` (10 % posterior width — a placeholder until a
kinetic Laplace posterior is implemented).  Per-state σ is the same
across states; downstream tools should treat |F_s| as a point estimate
and use difference-map confidence from the model-free per-bin σ.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from torchsx.merging.kinetics.loss import KineticLogSpaceLoss
from torchsx.merging.kinetics.parameters import KineticEnsembleParameters


_SIGMA_LOG_I_FLOOR = 0.1   # placeholder posterior width on log_I per HKL


def _state_F_magnitudes(
    params: KineticEnsembleParameters,
) -> list[tuple[str, Tensor]]:
    """Return ``[(label, F_magnitude)]`` per state, length n_states.

    Labels follow the convention ``state_A``, ``state_I``, ``state_B``,
    … for n_states ≤ 4; beyond that, ``state_S0``, ``state_S1``, …
    """
    out: list[tuple[str, Tensor]] = []
    out.append(("state_A", params.F_dark.detach().cpu().float()))

    if params.n_excited == 0:
        return out

    labels = ["state_I", "state_B", "state_C", "state_D"]
    F_re = params.F_re.detach().cpu().float()
    F_im = params.F_im.detach().cpu().float()
    for k in range(params.n_excited):
        F_mag = torch.sqrt(F_re[k] ** 2 + F_im[k] ** 2 + 1e-30)
        if k < len(labels):
            label = labels[k]
        else:
            label = f"state_S{k + 1}"
        out.append((label, F_mag))

    return out


def _write_mtz_with_F(
    loss: KineticLogSpaceLoss,
    F: Tensor,
    out_path: Path,
) -> None:
    """Build a :class:`ReflectionData` from ``F`` and save to ``out_path``.

    Mirrors the MTZ-writing branch of
    :meth:`torchsx.merging.posterior.Posterior.integrate` so the output
    parses identically in phenix / gemmi (anomalous columns when the
    stream was loaded with ``merge_friedel=False``).
    """
    from torchref import ReflectionData
    from torchref.symmetry import Cell, SpaceGroup

    prep = loss._prep
    median_cell = prep.median_cell.detach().cpu().tolist()
    sg = loss._stream._core.spacegroup or "P 1"
    cell = Cell(median_cell)
    spacegroup = SpaceGroup(sg)

    F_cpu = F.detach().cpu().float().clamp(min=1e-6)
    # MAP-square-root fallback σ — placeholder until kinetic Laplace.
    F_sigma_cpu = (_SIGMA_LOG_I_FLOOR * F_cpu).clamp(min=1e-6)

    rfree_flags = torch.zeros(prep.n_valid, dtype=torch.bool)
    friedel_merged = bool(getattr(loss._stream, "friedel_merged", True))
    data = ReflectionData.from_tensors(
        hkl=prep.hkl_valid.detach().cpu().int(),
        F=F_cpu,
        F_sigma=F_sigma_cpu,
        cell=cell,
        spacegroup=spacegroup,
        rfree_flags=rfree_flags,
        friedel_merged=friedel_merged,
    )
    # I = F²; σ_I ≈ 2 F σ_F.
    I = F_cpu * F_cpu
    sigma_I = 2.0 * F_cpu * F_sigma_cpu
    data.I = I
    data.I_sigma = sigma_I

    out_path.parent.mkdir(parents=True, exist_ok=True)
    data.write_mtz(str(out_path))


def write_per_state_mtzs(
    loss: KineticLogSpaceLoss,
    output_dir: str | Path,
) -> list[Path]:
    """Write one MTZ per kinetic state (``state_A.mtz``, ``state_I.mtz``, …).

    Returns the list of paths written.
    """
    output_dir = Path(output_dir)
    written: list[Path] = []
    for label, F in _state_F_magnitudes(loss.kinetic_params):
        out_path = output_dir / f"{label}.mtz"
        _write_mtz_with_F(loss, F, out_path)
        print(
            f"[kinetic-output] wrote {out_path}  "
            f"(F median {float(F.median()):.3f}, F p99 "
            f"{float(F.quantile(0.99)):.2f}, n_HKLs={F.numel()})",
            flush=True,
        )
        written.append(out_path)
    return written


def write_predicted_bin_mtzs(
    loss: KineticLogSpaceLoss,
    output_dir: str | Path,
    times_ps: Sequence[float],
    output_template: str = "bin_{idx:02d}_t{t_center:+.1f}.mtz",
) -> list[Path]:
    """Reconstruct ``|F_total(h, t_b)|`` per requested ``t_b`` and write MTZs.

    The kinetic model is evaluated at the supplied timepoints (which
    need not match the original ``unique_timepoints`` of the stream — a
    *fresh* :class:`KineticMixing` is built on ``times_ps`` reusing the
    fitted rate constants and IRF).  Useful for movie reconstruction or
    head-to-head comparison with the model-free per-bin MTZs.

    Returns the list of paths written.
    """
    from torchsx.merging.kinetics.mixing import KineticMixing

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    src_mix = loss.kinetic_mixing
    params = loss.kinetic_params

    # Pull the fitted kinetic state from the live ``KineticModel`` and
    # rebuild a fresh ``KineticMixing`` on the user-requested timepoints.
    # We reuse the same flow chart, activation policy, instrument
    # function, and IRF width.
    rate_constants = src_mix.kinetic.get_rate_constants()
    instrument_width = math.exp(
        float(src_mix.kinetic.log_instrument_width.detach())
    ) if src_mix.kinetic.log_instrument_width is not None else 0.0
    flow_chart = src_mix.kinetic.flow_chart
    light_activated = src_mix.light_activated
    activation_level = float(src_mix.activation_level)
    instrument_function = src_mix.instrument_function if instrument_width > 0 else "none"

    eval_mix = KineticMixing(
        flow_chart=flow_chart,
        unique_timepoints=torch.tensor(list(times_ps), dtype=torch.float64),
        instrument_width=instrument_width,
        light_activated=light_activated,
        activation_level=activation_level,
        rate_constants=rate_constants,
        instrument_function=instrument_function,
        verbose=0,
    ).to(params.log_F_dark.device)

    with torch.no_grad():
        F_re_bin, F_im_bin = eval_mix.F_total_per_bin(params)   # (n_t, N_hkl)
        F_mag_bin = torch.sqrt(F_re_bin ** 2 + F_im_bin ** 2 + 1e-30)

    written: list[Path] = []
    for idx, t in enumerate(times_ps):
        name = output_template.format(idx=idx, t_center=float(t))
        out_path = output_dir / name
        _write_mtz_with_F(loss, F_mag_bin[idx], out_path)
        print(
            f"[kinetic-output] wrote {out_path}  "
            f"(t={t:+.1f} ps, F median "
            f"{float(F_mag_bin[idx].median()):.3f})",
            flush=True,
        )
        written.append(out_path)
    return written


def write_kinetic_params_json(
    loss: KineticLogSpaceLoss,
    output_dir: str | Path,
    *,
    extra: dict | None = None,
) -> Path:
    """Dump kinetic parameters + diagnostics to ``kinetic_params.json``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "kinetic_params.json"

    mix = loss.kinetic_mixing
    rate_constants = mix.kinetic.get_rate_constants()
    time_constants = mix.kinetic.get_time_constants()
    irf_sigma = None
    if mix.kinetic.log_instrument_width is not None:
        irf_sigma = math.exp(float(mix.kinetic.log_instrument_width.detach()))

    payload = {
        "flow_chart": mix.kinetic.flow_chart,
        "rate_constants": rate_constants,
        "time_constants": time_constants,
        "irf_sigma_ps": irf_sigma,
        "irf_fwhm_ps": irf_sigma * (2 * math.sqrt(2 * math.log(2)))
            if irf_sigma is not None else None,
        "n_excited": loss.kinetic_params.n_excited,
        "n_states": loss.kinetic_params.n_states,
        "n_hkl": loss.kinetic_params.n_hkl,
        "n_obs": int(loss._prep.n_obs_total),
        "gain": loss._gain,
        "final_data_nll": loss._final_nll,
        "unique_timepoints_ps": loss.kinetic_mixing.unique_timepoints.detach().cpu().tolist(),
    }
    if extra is not None:
        payload.update(extra)

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[kinetic-output] wrote {out_path}", flush=True)
    return out_path
