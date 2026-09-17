"""Leave-one-out in-batch merge target, blended with a running EMA.

**Everything happens in observation space.  The data are never scaled.**
``I_true`` is scaled *up* by ``K`` to meet the observation; the observation is
never divided down to meet ``I_true``.  Concretely, the target is the
least-squares solution of

.. code-block:: text

    min_I0  Sum_j (n'_j - K_j I0)^2 / V_j        n'_j = n_j - bg_j
    V_j     = sigma_j^2 + (K_j I0)^2 (e^{w_j^2} - 1)
    I0^(-j) = (Sum_{i != j} K_i n'_i / V_i) / (Sum_{i != j} K_i^2 / V_i)

built from **its peers only**, plus an EMA prior.

The equivalent-looking form "inverse-variance mean of ``y_j = n'_j / K_j``"
is algebraically identical, and was what this module did first.  It is the
wrong framing and it constructs a quantity the data never contained: 2.88 % of
observations on the real dark sit at ``log K < -5``, where ``y`` is an
observation inflated ~150-fold and then handed a weight ~22,000x smaller to
undo it.  Accumulating ``K n'/V`` and ``K^2/V`` instead keeps every term at the
scale of the measurement.

No merged intensity is a free parameter.  There is no ``I0`` vector to optimise,
no EM step and no LBFGS — the merge happens inside the loss, which is what makes
"differentiate through the merge" a property of the design rather than a
feature to add.

Everything here is computed under ``no_grad`` from a **detached** prediction.
The gradient path to the network runs through ``log_mean = log I0^(-j) + log K``
in the objective, not through the target.  Detaching it is what makes the
two-pass step well defined and stops the network lowering the loss by dragging
the merge toward itself — leave-one-out already blocks the per-observation
version of that; detachment blocks the collective version.

The EMA blend is the cold start *and* the co-adaptation mitigation, with one
knob::

    lambda -> inf   fixed target, re-merged once per epoch  (the EM fallback)
    lambda -> 0     full in-batch merging

Start high so the network first learns ``K`` against a stationary reference
seeded from an existing merge, then decay.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from torchsx.nn.pooling import group_sum, loo_ratio

__all__ = ["MergeTargetOut", "LOOInBatchTarget", "unity_merge"]


def unity_merge(
    I: Tensor, sigma: Tensor, hkl_id: Tensor, n_hkl: int, *, floor: float = 1e-3
) -> Tensor:
    """Inverse-variance merge at unit scale — the right EMA seed.

    Deliberately **linear and over all observations**, including the negatives.
    ``PreparedObs.mu_init`` is a log-space MLE fitted to *positive* observations
    only; with 32.4 % of this dataset negative, exponentiating it gives a target
    biased high, and every observation then sits systematically low against it.
    That is not a subtle effect — seeding from ``exp(mu_init)`` put the initial
    normalised residual at ``var_z = 31`` on the real dark, against a target
    of 1.

    It is also the natural cold start: at ``K == 1`` this *is* the unity merge,
    and ``partialator --model=unity`` already reaches CC1/2 0.985 on this data.
    """
    # The K == 1 case of the same observation-space regression.
    w = 1.0 / sigma.clamp(min=1e-6) ** 2
    num = group_sum(w * I, hkl_id, n_hkl)
    den = group_sum(w, hkl_id, n_hkl)
    return torch.where(den > 0, num / den.clamp(min=1e-12), torch.zeros_like(num)).clamp(
        min=floor
    )


@dataclass
class MergeTargetOut:
    """Per-observation merge target plus everything the metrics want."""

    log_I0: Tensor       # (n,) log of the (floored) leave-one-out target
    I0: Tensor           # (n,) the target itself
    resid: Tensor        # (n,) n' - K*I0, in OBSERVATION space
    var: Tensor          # (n,) V_j, the observation-space variance
    z: Tensor            # (n,) resid / sqrt(V) — the normalised residual
    valid: Tensor        # (n,) bool — a usable peer target existed
    n_floored: int       # how many targets hit the positivity floor
    var_frac_scatter: Tensor  # (n,) share of V from predicted scatter


class LOOInBatchTarget(nn.Module):
    """Leave-one-out inverse-variance merge with an EMA prior.

    Parameters
    ----------
    I0_init
        ``(n_hkl,)`` seed intensities — use the existing EM Poisson merge.
        Also the fallback wherever no peer target exists.
    lam
        Prior strength λ.  Mutable during training (decay schedule).
    ema_decay
        Decay for the running ``(I0, den)`` used as the prior.
    floor
        Positivity floor on the target.  The multiplicative-scatter model needs
        a positive latent (it is a density on ``log S``), while an
        inverse-variance mean of noisy weak observations can come out negative.
        Floored observations are counted, not hidden — a rising
        ``n_floored`` means λ has been decayed too far.
    """

    def __init__(
        self,
        I0_init: Tensor,
        *,
        lam: float = 10.0,
        ema_decay: float = 0.99,
        floor: float = 1e-3,
    ) -> None:
        super().__init__()
        n_hkl = I0_init.shape[0]
        self.n_hkl = int(n_hkl)
        self.lam = float(lam)
        self.ema_decay = float(ema_decay)
        self.floor = float(floor)
        self.register_buffer("ema_I0", I0_init.detach().clone().float())
        self.register_buffer("ema_den", torch.zeros(n_hkl))
        self.register_buffer("_den_seeded", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def forward(
        self,
        log_K: Tensor,       # (n,) detached
        log_w: Tensor,       # (n,) detached
        n_ph: Tensor,        # (n,) photon counts
        bg: Tensor,          # (n,)
        sigma: Tensor,       # (n,) detector sigma on (n - bg)
        hkl_id: Tensor,      # (n,)
        *, update_state: bool = True,
    ) -> MergeTargetOut:
        K = log_K.exp().clamp(min=1e-8)
        n_signal = n_ph - bg                       # the data, left alone

        # V uses the EMA intensity as the scatter reference: stable, and it
        # keeps the weights out of the circular K -> target -> K loop.
        I0_ref = self.ema_I0[hkl_id].clamp(min=self.floor)
        w2 = (log_w.exp() ** 2).clamp(max=20.0)
        var_scatter = (K * I0_ref) ** 2 * torch.expm1(w2)
        var = (sigma**2 + var_scatter).clamp(min=1e-12)

        a = K * n_signal / var                     # both at measurement scale
        b = K * K / var

        den_batch = group_sum(b, hkl_id, self.n_hkl)
        prior_den = self.ema_den
        if not bool(self._den_seeded):
            prior_den = den_batch
            if update_state:
                self.ema_den.copy_(den_batch)
                self._den_seeded.fill_(True)

        prior_b = self.lam * prior_den
        target, valid = loo_ratio(
            a, b, hkl_id, self.n_hkl,
            fill=self.ema_I0[hkl_id],
            prior_a=prior_b * self.ema_I0,
            prior_b=prior_b,
        )

        floored = target < self.floor
        I0 = target.clamp(min=self.floor)

        resid = n_signal - K * I0                  # observation space
        z = resid / var.sqrt()

        if update_state:
            self._update_ema(a, b, hkl_id, den_batch)

        return MergeTargetOut(
            log_I0=I0.log(),
            I0=I0,
            resid=resid,
            var=var,
            z=z,
            valid=valid & ~floored,
            n_floored=int(floored.sum()),
            var_frac_scatter=var_scatter / var,
        )

    @torch.no_grad()
    def _update_ema(
        self, a: Tensor, b: Tensor, hkl_id: Tensor, den_batch: Tensor
    ) -> None:
        """Roll the running merge forward on the reflections this batch saw."""
        num_batch = group_sum(a, hkl_id, self.n_hkl)
        seen = den_batch > 0
        if not bool(seen.any()):
            return
        batch_I0 = torch.zeros_like(self.ema_I0)
        batch_I0[seen] = num_batch[seen] / den_batch[seen]
        d = self.ema_decay
        self.ema_I0[seen] = d * self.ema_I0[seen] + (1 - d) * batch_I0[seen]
        self.ema_den[seen] = d * self.ema_den[seen] + (1 - d) * den_batch[seen]
