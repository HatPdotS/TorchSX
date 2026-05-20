"""Per-HKL learnable parameters for the kinetic-ensemble forward model.

Holds:

- ``log_F_dark``  (N_hkl,)  — real, ≥ 0 dark amplitude in log space.
  Replaces the scalar ``log_I0`` of the single-state ``Parameters``;
  at ``K = 1`` we have ``log_I0 = 2 · log_F_dark``.

- ``F_re``, ``F_im``  (K - 1, N_hkl)  — cartesian complex components
  of the excited states.  Relative to the dark phase gauge
  (``F_dark`` is real positive), so the phase of state ``k`` is
  ``atan2(F_im_k, F_re_k)``.

A boolean ``centric_mask`` is stored for downstream consumers
(phase restraint, initial fit) so they can hard-zero ``F_im`` at
centric HKLs without having to recompute the mask.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class KineticEnsembleParameters(nn.Module):
    """Per-HKL parameters for ``K``-state kinetic ensemble.

    Parameters
    ----------
    log_F_dark_init :
        ``(N_hkl,)`` initial log-amplitude of the dark state.
    n_excited :
        Number of excited states ``K - 1`` (so total states = ``K``).
        May be ``0`` for the K = 1 sanity-reduction path.
    centric_mask :
        ``(N_hkl,)`` bool — True at centric HKLs.  ``F_im`` is held at 0
        at these HKLs (zeroed on init and after each gradient step via
        :meth:`zero_centric_imag`; restraints + initial-fit objectives
        also skip them).
    excited_init :
        Optional ``(K - 1, N_hkl, 2)`` tensor seeding ``(F_re, F_im)``.
        Defaults to all zeros — symmetry-broken later by the
        sigma-aware initial fit.
    """

    def __init__(
        self,
        log_F_dark_init: Tensor,
        n_excited: int,
        centric_mask: Tensor,
        excited_init: Tensor | None = None,
    ) -> None:
        super().__init__()
        if log_F_dark_init.ndim != 1:
            raise ValueError(
                f"log_F_dark_init must be 1-D, got shape {tuple(log_F_dark_init.shape)}"
            )
        n_hkl = log_F_dark_init.shape[0]
        if centric_mask.shape != log_F_dark_init.shape:
            raise ValueError(
                f"centric_mask shape {tuple(centric_mask.shape)} != "
                f"log_F_dark_init shape {tuple(log_F_dark_init.shape)}"
            )
        if n_excited < 0:
            raise ValueError(f"n_excited must be ≥ 0, got {n_excited}")

        self.n_hkl = n_hkl
        self.n_excited = n_excited

        self.log_F_dark = nn.Parameter(log_F_dark_init.clone().float())
        self.register_buffer("centric_mask", centric_mask.clone().bool())

        if n_excited == 0:
            # Register zero-sized parameters so attribute access never
            # KeyErrors and ``.parameters()`` enumerates a stable set.
            self.F_re = nn.Parameter(torch.zeros((0, n_hkl), dtype=torch.float32))
            self.F_im = nn.Parameter(torch.zeros((0, n_hkl), dtype=torch.float32))
        else:
            if excited_init is None:
                F_re = torch.zeros((n_excited, n_hkl), dtype=torch.float32)
                F_im = torch.zeros((n_excited, n_hkl), dtype=torch.float32)
            else:
                if excited_init.shape != (n_excited, n_hkl, 2):
                    raise ValueError(
                        f"excited_init must have shape ({n_excited}, {n_hkl}, 2), "
                        f"got {tuple(excited_init.shape)}"
                    )
                F_re = excited_init[..., 0].clone().float()
                F_im = excited_init[..., 1].clone().float()
            self.F_re = nn.Parameter(F_re)
            self.F_im = nn.Parameter(F_im)

        # Initialize centric F_im to zero (phase ∈ {0, π} forced into
        # the real axis under the dark gauge).
        with torch.no_grad():
            self.zero_centric_imag()

    # ── Convenience views ───────────────────────────────────────────

    @property
    def F_dark(self) -> Tensor:
        """``(N_hkl,)`` real, non-negative dark amplitudes."""
        return torch.exp(self.log_F_dark)

    @property
    def n_states(self) -> int:
        return self.n_excited + 1

    # ── Maintenance ─────────────────────────────────────────────────

    @torch.no_grad()
    def zero_centric_imag(self) -> None:
        """Force ``F_im[:, centric] = 0``.

        Call after each optimizer step so the centric phase constraint
        (phase ∈ {0, π} → ``F_im ≡ 0`` under the dark gauge) is
        maintained exactly.  The phase restraint already discourages
        movement of ``F_im`` at centrics, but it does not strictly
        forbid it; this guard provides the hard constraint.
        """
        if self.n_excited == 0:
            return
        mask = self.centric_mask
        if mask.any():
            self.F_im.data[:, mask] = 0.0

    # ── Reductions ──────────────────────────────────────────────────

    @classmethod
    def from_merged_result(
        cls,
        merged,  # MergedResult — avoid circular import at module load
        n_excited: int,
        space_group: str = "P 1",
        excited_init: Tensor | None = None,
    ) -> "KineticEnsembleParameters":
        """Seed from a single-state ``MergedResult``.

        ``log_F_dark = 0.5 · log(I0)``.  Centric mask is recomputed from
        the merged ``hkl`` indices using
        :func:`torchref.base.french_wilson.is_centric_from_hkl`.
        """
        from torchref.base.french_wilson import is_centric_from_hkl

        I0 = merged.I0.clamp(min=1e-12)
        log_F_dark_init = 0.5 * torch.log(I0)

        try:
            centric = is_centric_from_hkl(
                merged.hkl.long(), space_group=space_group,
            ).bool()
        except Exception as exc:  # noqa: BLE001
            print(
                f"  [kinetics] centric detection failed ({exc!r}); "
                f"treating all reflections as acentric"
            )
            centric = torch.zeros(I0.shape[0], dtype=torch.bool)

        return cls(
            log_F_dark_init=log_F_dark_init.float(),
            n_excited=n_excited,
            centric_mask=centric,
            excited_init=excited_init,
        )
