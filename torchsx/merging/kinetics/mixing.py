"""Mixing layer: kinetic populations → per-observation complex F_total.

Wraps :class:`torchref.kinetic.KineticModel` (matrix-exponential ODE
solver with refinable rate constants and a refinable Gaussian
instrument response function) so each crystal's observed timepoint
indexes into the correct row of the IRF-convolved population matrix.

Usage
-----

::

    mixing = KineticMixing(
        flow_chart="A->B",
        unique_timepoints=stream.unique_timepoints(),
        instrument_width=10.0,
    )
    populations = mixing.populations()      # (n_timepoints, K), rows sum to 1

    F_total_re, F_total_im = mixing.mix(
        params,                              # KineticEnsembleParameters
        hkl_idx,                             # (Nobs,) int64
        crystal_delay_bin=obs_info.delay_bin_idx,  # (n_crystals,) int64
        crystal_id=obs_info.crystal_id,            # (Nobs,) int64
    )
    F_total_sq = F_total_re ** 2 + F_total_im ** 2

The K = 1 reduction (``flow_chart="A"``) gives populations ≡ 1 and
``F_total = F_dark``, recovering the scalar single-state model.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor


class KineticMixing(nn.Module):
    """Glue between :class:`torchref.kinetic.KineticModel` and the merging forward model.

    Parameters
    ----------
    flow_chart :
        Kinetic scheme string passed verbatim to
        :class:`torchref.kinetic.KineticModel` (e.g. ``"A->B"``,
        ``"A->B,B->C"``).  K = 1 sanity-reduction path uses ``"A"``
        (a single non-reactive state).
    unique_timepoints :
        Sorted unique timepoints at which to evaluate populations.
        Aligns with ``ObsInfo.delay_bin_idx`` (which is a per-crystal
        index into this list).
    instrument_width :
        Initial Gaussian IRF σ.  Refinable.
    light_activated :
        If True, the kinetic model adds an inactive ``A*`` state
        (back-reactions to the ground state become A* and cannot
        re-photoactivate).  Default True for pump-probe SX.
    activation_level :
        Initial guess for the photoactivated fraction ∈ [0, 1].
    rate_constants :
        Optional initial rate constants (dict or list).  ``None`` →
        smart defaults inside ``KineticModel``.
    instrument_function :
        ``"gaussian"`` or ``"none"``.  Default ``"gaussian"`` (the IRF
        the time-resolved fit needs).
    """

    def __init__(
        self,
        flow_chart: str,
        unique_timepoints: Sequence[float] | Tensor,
        *,
        instrument_width: float = 10.0,
        light_activated: bool = True,
        activation_level: float = 0.5,
        rate_constants=None,
        instrument_function: str = "gaussian",
        verbose: int = 0,
    ) -> None:
        super().__init__()
        from torchref.kinetic import KineticModel

        if isinstance(unique_timepoints, Tensor):
            t_list = unique_timepoints.detach().cpu().tolist()
        else:
            t_list = list(unique_timepoints)
        if len(t_list) < 1:
            raise ValueError("unique_timepoints must have at least one entry")

        timepoints_t = torch.tensor(t_list, dtype=torch.float32)
        self.register_buffer("unique_timepoints", timepoints_t)

        self.kinetic = KineticModel(
            flow_chart=flow_chart,
            timepoints=timepoints_t,
            rate_constants=rate_constants,
            instrument_function=instrument_function,
            instrument_width=instrument_width,
            light_activated=light_activated,
            activation_level=activation_level,
            verbose=verbose,
        )
        # Light-activated mode adds an internal "A*" (photolocked) state
        # that is structurally identical to A; we collapse its population
        # into A so the user-facing ``n_states`` matches the number of
        # structurally distinct states.
        self.light_activated = bool(light_activated)
        if self.light_activated:
            init_state = self.kinetic.initial_state
            inactive_state = init_state + "*"
            idx_active = self.kinetic.state_to_idx[init_state]
            idx_inactive = self.kinetic.state_to_idx.get(inactive_state)
            if idx_inactive is None:
                self._collapse_idx = None       # nothing to merge
            else:
                self._collapse_idx = (idx_active, idx_inactive)
        else:
            self._collapse_idx = None

    # ── Forward views ───────────────────────────────────────────────

    def populations(self) -> Tensor:
        """``(n_timepoints, n_states)`` IRF-convolved occupancies.

        ``n_states`` here counts *structurally distinct* states — the
        internal photolocked ``A*`` state of the kinetic model is
        merged into ``A`` so that the kinetic occupancies align with
        :class:`KineticEnsembleParameters`' ``F_dark + F_re/F_im[k]``.
        """
        raw = self.kinetic()                            # (n_t, n_internal_states)
        if self._collapse_idx is None:
            return raw
        idx_a, idx_a_star = self._collapse_idx
        a_combined = raw[:, idx_a] + raw[:, idx_a_star]
        # Drop the A* column and overwrite A with the combined population.
        keep = [i for i in range(raw.shape[1]) if i != idx_a_star]
        out = raw[:, keep]
        new_idx_a = keep.index(idx_a)
        # Build the output without an in-place op so autograd is happy.
        return out.clone().index_copy_(
            1, torch.tensor([new_idx_a], device=out.device), a_combined.unsqueeze(1),
        )

    @property
    def n_states(self) -> int:
        return self.kinetic.n_states - (1 if self._collapse_idx is not None else 0)

    @property
    def n_timepoints(self) -> int:
        return int(self.unique_timepoints.numel())

    # ── The mixing kernel ───────────────────────────────────────────

    def mix(
        self,
        params,                       # KineticEnsembleParameters
        hkl_idx: Tensor,              # (Nobs,) int64
        crystal_id: Tensor,           # (Nobs,) int64
        crystal_delay_bin: Tensor,    # (n_crystals,) int64
    ) -> tuple[Tensor, Tensor]:
        """Build ``(F_total_re, F_total_im)`` per observation.

        Notes
        -----
        ``crystal_delay_bin`` is per-crystal (full table), so we
        gather first to per-observation then index ``populations``.
        ``KineticEnsembleParameters.n_excited + 1 == n_states`` must
        hold — checked once per call.
        """
        if params.n_excited + 1 != self.n_states:
            raise ValueError(
                f"KineticEnsembleParameters has {params.n_states} states "
                f"but KineticMixing has {self.n_states}"
            )

        pops = self.populations()                        # (n_t, K)
        obs_bin = crystal_delay_bin[crystal_id]          # (Nobs,)
        # Sentinel -1 (unlabelled crystal) is *not* allowed here — caller
        # should have filtered it out at PreparedObs construction time.
        if (obs_bin < 0).any():
            raise ValueError(
                "Found observations with unlabelled timepoint "
                "(crystal_delay_bin == -1). Filter the stream first."
            )

        w_obs = pops[obs_bin]                            # (Nobs, K)

        F_dark = params.F_dark[hkl_idx]                  # (Nobs,)
        w_dark = w_obs[:, 0]                             # (Nobs,)
        F_total_re = w_dark * F_dark
        F_total_im = torch.zeros_like(F_total_re)

        if params.n_excited > 0:
            # F_re/F_im are (K-1, N_hkl); gather per-obs then mix.
            F_re_obs = params.F_re[:, hkl_idx]           # (K-1, Nobs)
            F_im_obs = params.F_im[:, hkl_idx]           # (K-1, Nobs)
            w_exc = w_obs[:, 1:].T                       # (K-1, Nobs)
            F_total_re = F_total_re + (w_exc * F_re_obs).sum(dim=0)
            F_total_im = F_total_im + (w_exc * F_im_obs).sum(dim=0)

        return F_total_re, F_total_im

    def F_total_per_bin(
        self,
        params,                      # KineticEnsembleParameters
    ) -> tuple[Tensor, Tensor]:
        """Compute ``F_total`` for every (bin, HKL) pair.

        Returns ``(F_re, F_im)`` each shaped ``(n_timepoints, N_hkl)``.
        This is the per-timepoint amplitude grid against which the
        sigma-aware initial fit compares pre-merged amplitudes.
        """
        pops = self.populations()                        # (n_t, K)
        F_dark = params.F_dark.unsqueeze(0)              # (1, N_hkl)
        F_re = pops[:, 0:1] * F_dark                     # (n_t, N_hkl)
        F_im = torch.zeros_like(F_re)
        if params.n_excited > 0:
            w_exc = pops[:, 1:]                          # (n_t, K-1)
            F_re = F_re + w_exc @ params.F_re            # (n_t, N_hkl)
            F_im = F_im + w_exc @ params.F_im
        return F_re, F_im
