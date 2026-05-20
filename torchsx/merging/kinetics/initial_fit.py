"""Sigma-aware initial fit for the kinetic-ensemble parameters.

Two stages:

**Stage A** — run the standard single-state merging pipeline at each
discrete timepoint, producing ``(|F_bin(h)|, σ_F_bin(h))`` per bin via
the existing posterior machinery (no new likelihood).  Implemented as
:func:`per_bin_premerging` which loops over
:meth:`SFXStream.filter_images` and calls
:meth:`SFXStream.merge` (which internally uses the canonical
``Merger``).

**Stage B** — minimise

::

    Σ_{h,b} (|F_total(h, t_b)| − |F_bin(h, t_b)|)² / σ_F_bin(h, t_b)²
      + λ_phase · Σ_{k ≥ 1, h ∉ centric} F_im_k(h)²

over ``(log_F_dark, F_re, F_im, log_rate_constants,
log_instrument_width, …)`` with LBFGS.  Operates on per-HKL
amplitudes only — orders of magnitude cheaper than a photon-level
joint refinement.  The full Poisson × GenGauss joint fit is a
follow-up step that reuses these parameters as a warm start.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from torchsx.merging.kinetics.mixing import KineticMixing
from torchsx.merging.kinetics.parameters import KineticEnsembleParameters
from torchsx.merging.kinetics.phase_restraint import PhaseRestraint


# ── Stage A: per-bin pre-merging ────────────────────────────────────


@dataclass
class PerBinAmplitudes:
    """Pre-merged amplitudes ``|F(h, t_b)|`` per kinetic-model timepoint.

    Each list entry corresponds to one row of
    :attr:`KineticMixing.unique_timepoints` (in the same order).

    Notes
    -----
    The HKL set of bin ``b`` is a subset of the full-stream ASU
    (some HKLs may not be observed in a given delay bin).  We carry
    explicit ``ref_hkl_idx_per_bin`` to map each bin's HKLs to the
    ``F_dark`` reference HKL table — built once by
    :meth:`build_ref_index`.
    """

    timepoints: Tensor
    hkl_per_bin: list[Tensor]                       # n_t × (n_hkl_b, 3) int64
    F_per_bin: list[Tensor]                         # n_t × (n_hkl_b,)
    sigma_F_per_bin: list[Tensor]                   # n_t × (n_hkl_b,)
    ref_hkl_idx_per_bin: list[Tensor] | None = None  # n_t × (n_hkl_b,) int64

    def build_ref_index(self, ref_hkl: Tensor) -> None:
        """Resolve each bin's HKLs against ``ref_hkl`` (typically
        ``KineticEnsembleParameters``' HKL table).

        Bin HKLs not present in ``ref_hkl`` are dropped (filtered out
        of ``F_per_bin`` and ``sigma_F_per_bin`` too).  Builds the
        ``ref_hkl_idx_per_bin`` table in-place.
        """
        ref = ref_hkl.long()
        n_ref = ref.shape[0]
        # Linearise into a hashable int64 — works for typical HKL range
        # (≤ 16-bit per component).
        ref_key = _hkl_to_key(ref)
        ref_lookup: dict[int, int] = {int(k.item()): i for i, k in enumerate(ref_key)}

        new_hkl: list[Tensor] = []
        new_F: list[Tensor] = []
        new_sigma: list[Tensor] = []
        new_idx: list[Tensor] = []

        for b, (h_b, F_b, s_b) in enumerate(
            zip(self.hkl_per_bin, self.F_per_bin, self.sigma_F_per_bin)
        ):
            keys = _hkl_to_key(h_b.long())
            mapped = torch.tensor(
                [ref_lookup.get(int(k.item()), -1) for k in keys],
                dtype=torch.int64,
            )
            keep = mapped >= 0
            if not keep.all():
                n_drop = int((~keep).sum())
                print(
                    f"  [initial-fit] bin {b} (t={self.timepoints[b].item()}): "
                    f"dropped {n_drop} HKLs absent from reference set"
                )
            new_hkl.append(h_b[keep])
            new_F.append(F_b[keep])
            new_sigma.append(s_b[keep])
            new_idx.append(mapped[keep])

        self.hkl_per_bin = new_hkl
        self.F_per_bin = new_F
        self.sigma_F_per_bin = new_sigma
        self.ref_hkl_idx_per_bin = new_idx


def _hkl_to_key(hkl: Tensor) -> Tensor:
    """Pack (h,k,l) int triples into single int64 keys."""
    # Shift to non-negative then bit-pack.  Assumes |h|,|k|,|l| < 2^16.
    h = hkl[:, 0].to(torch.int64) + (1 << 15)
    k = hkl[:, 1].to(torch.int64) + (1 << 15)
    l = hkl[:, 2].to(torch.int64) + (1 << 15)
    return (h << 32) | (k << 16) | l


def per_bin_premerging(
    stream,
    *,
    verbose: bool = True,
) -> tuple[PerBinAmplitudes, "MergedResult"]:
    """Run Stage A: a current-pipeline merge at each unique timepoint.

    Returns the per-bin amplitudes (with σ_F via the delta method on
    σ_I0) plus a full-stream merge to seed ``log_F_dark``.

    Requires the stream to be timepoint-labelled (see
    :meth:`SFXStream.assign_timepoints`).
    """
    image_tp = stream.image_timepoints
    if not image_tp:
        raise ValueError(
            "Stream is not timepoint-labelled. Call "
            "stream.assign_timepoints(times_file) first."
        )
    unique_t = stream.unique_timepoints()
    if not unique_t:
        raise ValueError("Stream has no finite timepoints labelled")

    if verbose:
        print(
            f"[Stage A] per-bin pre-merging at {len(unique_t)} timepoints: "
            f"{unique_t}"
        )

    image_tp_t = torch.tensor(image_tp, dtype=torch.float64)

    timepoints_out: list[float] = []
    hkl_per_bin: list[Tensor] = []
    F_per_bin: list[Tensor] = []
    sigma_F_per_bin: list[Tensor] = []

    for t_b in unique_t:
        mask = (image_tp_t == t_b)
        if not mask.any():
            continue
        stream_b = stream.filter_images(mask)
        if verbose:
            print(
                f"  bin t={t_b}: {stream_b.n_images} images, "
                f"{stream_b.n_crystals} crystals, {stream_b.n_observations} obs"
            )
        merged_b = stream_b.merge()
        F_b = torch.sqrt(merged_b.I0.clamp(min=1e-12))
        # delta-method: σ_F = σ_I0 / (2 √I0) = σ_I0 / (2 F).
        sigma_F_b = merged_b.sigma_I0 / (2.0 * F_b.clamp(min=1e-6))
        timepoints_out.append(float(t_b))
        hkl_per_bin.append(merged_b.hkl.long())
        F_per_bin.append(F_b.float())
        sigma_F_per_bin.append(sigma_F_b.float())

    full_merged = stream.merge()

    return (
        PerBinAmplitudes(
            timepoints=torch.tensor(timepoints_out, dtype=torch.float64),
            hkl_per_bin=hkl_per_bin,
            F_per_bin=F_per_bin,
            sigma_F_per_bin=sigma_F_per_bin,
        ),
        full_merged,
    )


# ── Stage B: sigma-aware kinetic-ensemble fit ────────────────────────


def fit_initial_kinetic_ensemble(
    *,
    dark_seed,                            # MergedResult — for F_dark + ref HKL + centric
    per_bin: PerBinAmplitudes,
    n_excited: int,
    flow_chart: str,
    space_group: str = "P 1",
    instrument_width: float = 10.0,
    light_activated: bool = True,
    activation_level: float = 0.5,
    rate_constants=None,
    lam_phase: float = 1.0,
    max_iter: int = 200,
    history_size: int = 20,
    tolerance_grad: float = 1e-5,
    device: str | torch.device = "cpu",
    verbose: bool = True,
) -> tuple[KineticEnsembleParameters, KineticMixing, dict]:
    """Stage B: fit the kinetic ensemble to per-bin amplitudes.

    Loss
    ----

    ::

        L = Σ_{h,b} (|F_total(h, t_b)| − |F_bin(h, t_b)|)² / σ_F²
            + λ_phase · Σ_{k ≥ 1, h ∉ centric} F_im_k(h)²

    where ``F_total(h, t_b) = w_dark(t_b) F_dark(h) + Σ_{k ≥ 1}
    w_k(t_b) (F_re_k + i F_im_k)`` and ``w(t)`` comes from the
    kinetic ODE + IRF (:class:`KineticMixing`).

    The per-HKL σ_F enters as the inverse-variance weight on the
    intensity residuals — well-measured HKLs dominate the fit, weak
    ones get pulled toward the dark + restraint floor.

    Returns
    -------
    ``(params, mixing, diagnostics)`` — both modules on ``device``.
    """
    device = torch.device(device)

    # ── Build parameters (centric mask, F_dark seed) from dark_seed ─
    params = KineticEnsembleParameters.from_merged_result(
        dark_seed, n_excited=n_excited, space_group=space_group,
    ).to(device)

    # ── Resolve per-bin HKLs against params' reference set ──────────
    if per_bin.ref_hkl_idx_per_bin is None:
        per_bin.build_ref_index(dark_seed.hkl)

    # ── Build mixing layer at the same timepoints as per_bin ────────
    mixing = KineticMixing(
        flow_chart=flow_chart,
        unique_timepoints=per_bin.timepoints,
        instrument_width=instrument_width,
        light_activated=light_activated,
        activation_level=activation_level,
        rate_constants=rate_constants,
        verbose=0,
    ).to(device)

    if mixing.n_states != params.n_states:
        raise ValueError(
            f"flow_chart implies {mixing.n_states} states but "
            f"n_excited+1 = {params.n_states}; adjust n_excited so they match"
        )

    phase_restraint = PhaseRestraint(lam=lam_phase).to(device)

    # ── Move per-bin targets to device once ─────────────────────────
    F_obs = [F.to(device) for F in per_bin.F_per_bin]
    sigma_F = [s.clamp(min=1e-6).to(device) for s in per_bin.sigma_F_per_bin]
    ref_idx = [r.to(device) for r in per_bin.ref_hkl_idx_per_bin]
    n_bins = len(F_obs)

    # ── LBFGS over all trainable parameters ─────────────────────────
    trainable = [p for p in params.parameters() if p.requires_grad]
    trainable += [p for p in mixing.parameters() if p.requires_grad]

    opt = torch.optim.LBFGS(
        trainable,
        lr=1.0,
        max_iter=max_iter,
        history_size=history_size,
        tolerance_grad=tolerance_grad,
        line_search_fn="strong_wolfe",
    )

    history: list[dict] = []
    n_calls = {"n": 0}

    def closure() -> Tensor:
        opt.zero_grad(set_to_none=False)
        F_re_grid, F_im_grid = mixing.F_total_per_bin(params)
        # (n_t, N_hkl) for both — pick out bin/HKL pairs as residuals.
        loss_data = F_re_grid.new_zeros(())
        for b in range(n_bins):
            idx = ref_idx[b]                              # (n_hkl_b,)
            re_b = F_re_grid[b, idx]                      # (n_hkl_b,)
            im_b = F_im_grid[b, idx]
            F_mod = torch.sqrt(re_b ** 2 + im_b ** 2 + 1e-30)
            r = (F_mod - F_obs[b]) / sigma_F[b]
            loss_data = loss_data + (r * r).sum()
        loss_phase = phase_restraint(params)
        loss = loss_data + loss_phase
        loss.backward()
        n_calls["n"] += 1
        history.append({
            "n_call": n_calls["n"],
            "loss": float(loss.detach()),
            "loss_data": float(loss_data.detach()),
            "loss_phase": float(loss_phase.detach()),
        })
        return loss

    # Initial-loss baseline (single forward + backward — already counted
    # in n_calls when LBFGS calls closure inside step()).
    final_loss = opt.step(closure)

    # Enforce centric F_im = 0 hard constraint after the optimization
    # step (the L2 restraint shrinks but doesn't zero).
    params.zero_centric_imag()

    diagnostics = {
        "n_closure_calls": n_calls["n"],
        "final_loss": float(final_loss.detach()) if isinstance(final_loss, Tensor) else None,
        "history": history,
        "rate_constants": mixing.kinetic.get_rate_constants(),
        "time_constants": mixing.kinetic.get_time_constants(),
        "instrument_width": math.exp(float(mixing.kinetic.log_instrument_width.detach()))
            if mixing.kinetic.log_instrument_width is not None else None,
        "n_excited": n_excited,
        "n_bins": n_bins,
        "n_hkl": params.n_hkl,
        "lam_phase": lam_phase,
    }

    if verbose:
        print(f"[Stage B] LBFGS done in {n_calls['n']} closure calls; "
              f"final loss {diagnostics['final_loss']:.6g}")
        for k, v in diagnostics["rate_constants"].items():
            print(f"  rate {k} = {v:.4g}  (τ = {1.0/max(v, 1e-30):.4g})")
        if diagnostics["instrument_width"] is not None:
            print(f"  IRF σ = {diagnostics['instrument_width']:.4g}")
        # Phase-magnitude summary.
        if params.n_excited > 0:
            with torch.no_grad():
                acentric = ~params.centric_mask
                if acentric.any():
                    F_re = params.F_re[:, acentric]
                    F_im = params.F_im[:, acentric]
                    abs_F = torch.sqrt(F_re ** 2 + F_im ** 2 + 1e-30)
                    phi = torch.atan2(F_im, F_re)
                    print(
                        f"  acentric phase |φ| (deg): "
                        f"median {phi.abs().median().item() * 180 / math.pi:.2f}, "
                        f"q95 {phi.abs().quantile(0.95).item() * 180 / math.pi:.2f}; "
                        f"|F_k| / |F_dark| median "
                        f"{(abs_F / params.F_dark[acentric].unsqueeze(0).clamp(min=1e-6)).median().item():.3g}"
                    )

    return params, mixing, diagnostics
