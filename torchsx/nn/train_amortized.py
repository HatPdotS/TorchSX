"""Training loop for the amortised scale model.

Batches are drawn over **crystals**, not observations, because the
leave-one-out context pool needs a crystal's reflections to be complete: a
truncated crystal gives a biased context.  The merge target is the opposite —
it is an average, so a subsample of each reflection's observations is unbiased,
and at 8192 crystals a batch holds ~77 observations per reflection.

That is why the batch is sized in *observations* rather than crystals: crystals
carry a variable number (median 178, long tail), so a fixed crystal count makes
memory jump around while a fixed observation budget keeps it flat.

Each step is two passes:

1. ``no_grad``: predict, then build the leave-one-out merge target from
   **detached** ``K`` and ``w``.
2. with grad: predict again, form ``log_mean = log I0^(-j) + log K``, and
   evaluate the marginal NLL.

Detaching the target is what makes the scheme well defined.  Leave-one-out
blocks an observation from pulling its *own* target; detachment blocks the
network from pulling the whole merge toward itself collectively.

Adam only — no LBFGS anywhere, on the network or elsewhere.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from torchsx.merging.observations import PreparedObs
from torchsx.nn.merge_target import (
    LOOInBatchTarget, MergeTargetOut, unity_merge,
)
from torchsx.nn.metrics import (
    MetricLogger,
    calibration_metrics,
    condition_auc,
    grad_norms,
    hkl_permutation_sensitivity,
    scale_metrics,
)

__all__ = ["CrystalIndex", "TrainConfig", "AmortizedTrainer"]


class CrystalIndex:
    """Crystal-major view of the hkl-sorted observation list.

    ``PreparedObs`` is sorted by ``hkl_id`` (the merge needs contiguous
    reflections).  Training needs the opposite grouping, so this builds the
    permutation once and serves ``crystal -> observation rows`` from CSR
    offsets.
    """

    def __init__(self, crystal_id: Tensor, n_crystals: int) -> None:
        order = crystal_id.argsort(stable=True)
        counts = torch.bincount(crystal_id, minlength=n_crystals)
        offsets = torch.zeros(n_crystals + 1, dtype=torch.long)
        offsets[1:] = counts.cumsum(0)
        self.order = order
        self.counts = counts
        self.offsets = offsets
        self.n_crystals = int(n_crystals)
        self.nonempty = torch.nonzero(counts > 0, as_tuple=True)[0]

    def rows(self, crystals: Tensor) -> Tensor:
        """All observation rows of the given crystals, as a flat index.

        Vectorised rather than a per-crystal Python loop: a batch holds ~8600
        crystals, and building one tensor each cost 38.8 ms against 15.1 ms
        for the arithmetic form.
        """
        if crystals.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        starts = self.offsets[crystals]
        counts = self.offsets[crystals + 1] - starts
        total = int(counts.sum())
        if total == 0:
            return torch.empty(0, dtype=torch.long)
        grp_start = torch.cumsum(counts, 0) - counts
        within = torch.arange(total, device=starts.device) - grp_start.repeat_interleave(counts)
        return self.order[within + starts.repeat_interleave(counts)]

    def batches(
        self, max_obs: int, *, generator: torch.Generator | None = None,
        crystals: Tensor | None = None, shuffle: bool = True,
    ):
        """Yield ``(crystal_sel, obs_rows)`` with at most ``max_obs`` rows each."""
        pool = self.nonempty if crystals is None else crystals
        if shuffle:
            pool = pool[torch.randperm(pool.numel(), generator=generator)]
        c = self.counts[pool]
        start, running = 0, 0
        for i in range(pool.numel()):
            ci = int(c[i])
            if running + ci > max_obs and i > start:
                sel = pool[start:i]
                yield sel, self.rows(sel)
                start, running = i, 0
            running += ci
        if start < pool.numel():
            sel = pool[start:]
            yield sel, self.rows(sel)


@dataclass
class TrainConfig:
    #: Warn when the hkl-permutation sensitivity exceeds this.  A scale that
    #: depends on hkl is degenerate with I0 and can manufacture a difference
    #: peak; measured 0.003 in abismal and 0.586 in a collapsed run here.
    hkl_permutation_warn: float = 0.02
    max_obs_per_batch: int = 1_500_000
    epochs: int = 10
    lr: float = 1e-3
    #: Linear LR warmup.  Matters once the encoder is a deep stack of exchange
    #: blocks: the zero-initialised output projections make every block start
    #: as the identity, so all early gradient lands on the heads (measured
    #: 4161 at step 0 against ~1 later) and a constant LR spends the first
    #: steps taking large, badly-scaled steps.  0 disables.
    warmup_steps: int = 0
    #: Decoupled (AdamW) weight decay.  Applied to matrix parameters ONLY --
    #: norm gains and biases are excluded, because decaying an RMSNorm gain
    #: shrinks the network's effective scale rather than regularising it.
    #: Note this is genuinely decoupled: plain ``Adam(weight_decay=)`` adds L2
    #: to the gradient, which Adam then divides by sqrt(v), so parameters with
    #: small gradients would be decayed hardest -- the decay strength ends up
    #: coupled to gradient history rather than being a clean prior.
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    lam0: float = 10.0            # EMA prior strength at step 0
    lam_min: float = 0.0
    lam_halflife_steps: int = 200  # lam decays by 1/2 every this many steps
    #: Penalty on the predicted scatter width, ``lam_spread * mean(log w)``.
    #: The design doc argued no explicit penalty on w is needed because the
    #: GenGauss normaliser already contains -log(alpha).  That is true *at the
    #: likelihood optimum*; it says nothing about which optimum you reach.
    #: Empirically the model settles into a wide-scatter minimum where the
    #: width absorbs all per-observation variation (var_frac_scatter 0.99,
    #: within_frac 0.02) and there is then almost no gradient pressure left to
    #: refine K per observation.  This penalty is a tool for escaping that
    #: basin, not a claim about the statistics -- so it is annealed away and
    #: the final w is judged on calibration (var_z), not on this term.
    #: ``"loo"``   — leave-one-out inverse-variance estimator (Gaussian WLS
    #:               surrogate; cheap, and the default).
    #: ``"posterior"`` — integrate I0 out on a local grid.  Never fits I0, so
    #:               the surrogate/objective mismatch that makes detaching only
    #:               approximate disappears.  Costs ~k_coarse inner-quadrature
    #:               passes (measured: one pass is 4.4 % of a step, so k=31 is
    #:               roughly 2.3x).  Requires a quadrature with the Triton path
    #:               (a wide ``FixedLogGrid``) to be worth it.
    merge_target: str = "loo"
    #: ``"per_obs"``  — score each observation against the merged target.
    #: ``"evidence"`` — use the per-reflection marginal likelihood directly.
    #:                  Only valid with ``merge_target="posterior"``.  No target
    #:                  is involved at all, so no leave-one-out bookkeeping is
    #:                  needed: what protects it is the *feature* contract, not
    #:                  target exclusion.
    loss: str = "per_obs"
    #: Estimator used to MERGE for the split-half metrics, independent of
    #: ``merge_target``.  Tying the two confounds the comparison: the estimator
    #: alone is worth +0.007 CC1/2 at K = 1 (0.9871 WLS vs 0.9944 posterior),
    #: so scoring a leave-one-out run with WLS and a posterior run with the
    #: posterior measures both changes at once.  Default posterior for every
    #: run, so CC1/2 differences reflect the scale model only.
    merge_estimator: str = "posterior"
    posterior_k_coarse: int = 31
    posterior_m_fine: int = 257
    posterior_n_sigma: float = 8.0
    posterior_prior: str = "flat_I0"
    lam_spread: float = 0.0
    #: Halflife of the spread penalty in steps; ``0`` disables the ramp and
    #: holds ``lam_spread`` constant for the whole run.
    lam_spread_halflife_steps: int = 400
    log_every: int = 10
    eval_lam: float = 0.0
    heldout_frac: float = 0.05
    #: Crystals scored from the TRAINING set under the same conditions as the
    #: held-out set (eval_lam, no spread penalty).  Without this the logged
    #: train and heldout numbers differ in both the penalty and lambda, so
    #: their gap is bookkeeping rather than generalisation — and that gap is
    #: precisely the diagnostic for whether capacity binds.
    train_eval_crystals: int = 3500
    #: Keep the parameters that scored best on held-out.  The 300-epoch run
    #: peaked at ~epoch 230 (held-out 5.044) and then degraded to 5.15+ as the
    #: spread penalty annealed and the model over-committed to within-crystal
    #: structure -- the endpoint was not the model we wanted.
    keep_best: bool = True
    #: Run the split-half merge every N epochs (0 disables).  Costs one
    #: no-grad pass over the whole dataset.
    split_half_every: int = 20
    seed: int = 0
    outdir: str = "."
    device: str = "cuda"
    shell_bins: int = 10
    extra: dict = field(default_factory=dict)


class AmortizedTrainer:
    """Ties scale model, merge target, objective and metrics together."""

    def __init__(
        self,
        prep: PreparedObs,
        scale,
        objective,
        params,
        *,
        config: TrainConfig | None = None,
        is_light: Tensor | None = None,
    ) -> None:
        self.cfg = config or TrainConfig()
        self.device = torch.device(self.cfg.device)
        self.prep = prep
        self.scale = scale.to(self.device)
        self.objective = objective
        self.params = params
        self.is_light = is_light

        oi = prep.obs_info
        self.index = CrystalIndex(oi.crystal_id.cpu(), oi.n_crystals)
        self.n_ph = (prep.I_obs + prep.bg).clamp(min=0)
        self.sigma = (
            oi.sigma if oi.sigma is not None
            else (self.n_ph + prep.bg.clamp(min=0) + 1.0).sqrt()
        )
        seed_I0 = unity_merge(prep.I_obs, self.sigma, prep.hkl_id, prep.n_valid)
        if self.cfg.merge_target == "posterior":
            from torchsx.nn.posterior_target import PosteriorMergeTarget
            self.target = PosteriorMergeTarget(
                seed_I0,
                k_coarse=self.cfg.posterior_k_coarse,
                m_fine=self.cfg.posterior_m_fine,
                n_sigma=self.cfg.posterior_n_sigma,
                prior=self.cfg.posterior_prior,
            ).to(self.device)
        elif self.cfg.merge_target == "loo":
            self.target = LOOInBatchTarget(seed_I0, lam=self.cfg.lam0).to(self.device)
        else:
            raise ValueError(
                f"merge_target must be 'loo' or 'posterior', got {self.cfg.merge_target!r}"
            )
        if self.cfg.loss == "evidence" and self.cfg.merge_target != "posterior":
            raise ValueError("loss='evidence' requires merge_target='posterior'")
        if self.cfg.merge_estimator not in ("posterior", "wls"):
            raise ValueError(
                f"merge_estimator must be 'posterior' or 'wls', got "
                f"{self.cfg.merge_estimator!r}"
            )
        # Scoring estimator, deliberately independent of the training target.
        if self.cfg.merge_estimator == "posterior":
            if self.cfg.merge_target == "posterior":
                self.score_target = self.target      # share state, don't duplicate
            else:
                from torchsx.nn.posterior_target import PosteriorMergeTarget
                self.score_target = PosteriorMergeTarget(
                    seed_I0,
                    k_coarse=self.cfg.posterior_k_coarse,
                    m_fine=self.cfg.posterior_m_fine,
                    n_sigma=self.cfg.posterior_n_sigma,
                    prior=self.cfg.posterior_prior,
                ).to(self.device)
        else:
            self.score_target = None
        # obs_scale = n'/I0 needs a reference merge.  Seed it from the same
        # unity merge (linear, all observations) rather than exp(mu_init),
        # which is a log-space MLE over positives only and biased high.
        feats = getattr(self.scale, "features", None)
        if feats is not None and hasattr(feats, "set_reference_I0"):
            feats.set_reference_I0(seed_I0.to(self.device))

        self.gen = torch.Generator().manual_seed(self.cfg.seed)
        pool = self.index.nonempty
        perm = pool[torch.randperm(pool.numel(), generator=self.gen)]
        n_held = int(self.cfg.heldout_frac * perm.numel())
        self.heldout_crystals = perm[:n_held]
        self.train_crystals = perm[n_held:]

        decay, no_decay = [], []
        for name, prm in self.scale.named_parameters():
            if not prm.requires_grad:
                continue
            (no_decay if prm.ndim <= 1 else decay).append(prm)
        # Merge-quality quartets, at MATCHED crystal count.  Merge statistics
        # depend strongly on multiplicity, so comparing a 66,822-crystal train
        # merge against a 3,516-crystal validation merge would measure the
        # count, not the model.  Both sets are subsampled to the same size and
        # each is split in half, giving CC1/2 within seen crystals and within
        # unseen crystals on equal footing.
        n_match = min(self.train_crystals.numel(), self.heldout_crystals.numel())
        n_match -= n_match % 2
        gq = torch.Generator().manual_seed(self.cfg.seed + 991)
        tr_sub = self.train_crystals[
            torch.randperm(self.train_crystals.numel(), generator=gq)[:n_match]
        ]
        va_sub = self.heldout_crystals[
            torch.randperm(self.heldout_crystals.numel(), generator=gq)[:n_match]
        ]
        h = n_match // 2
        self.merge_sets = {
            "train_A": tr_sub[:h], "train_B": tr_sub[h:],
            "val_A": va_sub[:h], "val_B": va_sub[h:],
        }
        self.n_merge_crystals = n_match

        # FULL-dataset split-half, for an absolute CC1/2 comparable to the
        # published figures (partialator unity 0.9850, EM Poisson 0.9872).
        # The matched train/val sets above answer a different question and are
        # much shallower: 3,516 crystals per set is ~21 observations per
        # reflection per half, against ~375 when everything is merged, and
        # merge statistics depend strongly on multiplicity -- so those numbers
        # are NOT comparable to the literature and only their train-vs-val
        # difference is meaningful.
        allc = self.index.nonempty[
            torch.randperm(self.index.nonempty.numel(), generator=gq)
        ]
        half = allc.numel() // 2
        self.merge_sets["full_A"] = allc[:half]
        self.merge_sets["full_B"] = allc[half:]
        # A second, independent split of the same data: comparing the two gives
        # the scatter of the statistic itself, so we know which differences
        # between runs are real.
        allc2 = self.index.nonempty[
            torch.randperm(self.index.nonempty.numel(), generator=gq)
        ]
        self.merge_sets["rep_A"] = allc2[:half]
        self.merge_sets["rep_B"] = allc2[half:]

        self.opt = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.cfg.lr,
        )
        self.n_decayed = sum(p.numel() for p in decay)
        self.n_undecayed = sum(p.numel() for p in no_decay)
        self._base_lr = self.cfg.lr
        self.logger = MetricLogger(self.cfg.outdir)
        self.step_i = 0
        self.best_heldout = float("inf")
        self.best_step = -1
        self._best_state: dict | None = None

        # Resolution shell per observation, for the calibration breakdown.
        d = prep.d_spacing[prep.hkl_id]
        rank = d.argsort().argsort().float() / max(d.numel(), 1)
        self.shell_of_obs = (rank * self.cfg.shell_bins).long().clamp(
            max=self.cfg.shell_bins - 1
        )

    # ── schedule ───────────────────────────────────────────────────────

    def _lam(self) -> float:
        hl = max(self.cfg.lam_halflife_steps, 1)
        return self.cfg.lam_min + (self.cfg.lam0 - self.cfg.lam_min) * (
            0.5 ** (self.step_i / hl)
        )

    def _lam_spread(self) -> float:
        if self.cfg.lam_spread <= 0:
            return 0.0
        hl = self.cfg.lam_spread_halflife_steps
        if hl <= 0:
            # No ramp: hold the penalty constant.  A decaying penalty makes the
            # optimised loss non-comparable across a run for a reason that has
            # nothing to do with the model -- and at lam_spread = 0.5 it did
            # real damage, driving the per-shell CC1/2 non-monotone in
            # resolution (shells 3-5 scored above shell 1, which is not
            # physically possible).  A weak constant penalty is a prior, not a
            # schedule, and is what the shrinkage is meant to be.
            return self.cfg.lam_spread
        return self.cfg.lam_spread * (0.5 ** (self.step_i / hl))

    # ── one step ───────────────────────────────────────────────────────

    def _forward(self, rows: Tensor, *, grad: bool):
        meta = self.prep.obs_info.gather(rows)
        n_ph = self.n_ph[rows]
        bg = self.prep.bg[rows]
        sigma = self.sigma[rows]

        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            # ONE forward.  The target needs a *detached* prediction, and this
            # prediction detached is exactly that -- same weights, same inputs,
            # same numbers.  Running a separate no_grad pass first (as this did)
            # bought nothing and cost a full forward: with gradient
            # checkpointing the grad pass is ~3 forward-equivalents, so the
            # redundant pass was ~25 % of the step.
            pred = self.scale.prime(meta)
            if self.cfg.merge_target == "posterior":
                tgt, log_Z = self._posterior_step(pred, n_ph, bg, sigma, meta)
            else:
                if grad:
                    self.target.lam = self._lam()
                log_Z = None
                with torch.no_grad():
                    tgt = self.target(
                        pred.log_K.detach(), pred.log_w.detach(),
                        n_ph, bg, sigma, meta.hkl_idx,
                    )
            # log_I0 is a LINEAR (mean) estimate; the GenGauss is centred on
            # log_mean, i.e. a median.  Shift by -log E[exp(eps)] so the two
            # agree -- otherwise the target is biased high by a factor that
            # grows with the width, which is a runaway loop.
            if self.cfg.loss == "evidence":
                # The marginal likelihood per reflection.  I0 appears nowhere.
                nll = -log_Z[self.target.last_seen].sum()
            else:
                log_mean = tgt.log_I0 + pred.log_K - pred.log_e_s
                nll = self.objective(
                    self.scale.scatter, self.params, log_mean, n_ph, bg, meta,
                    mask=tgt.valid,
                )
        return meta, pred, tgt, nll, n_ph, bg, sigma

    def _posterior_step(self, pred, n_ph, bg, sigma, meta):
        """Posterior merge: returns a ``MergeTargetOut`` plus per-HKL log Z."""
        post = self.target(
            self.objective, self.scale.scatter, self.params, pred,
            n_ph, bg, sigma, meta.hkl_idx, meta,
        )
        self.target.last_seen = post.seen
        self.last_n_hkl = int(post.seen.sum())
        self.last_edge_mass = post.edge_mass
        I0 = post.I0_obs
        K = pred.log_K.exp().clamp(min=1e-8)
        w2 = (pred.log_w.exp() ** 2).clamp(max=20.0)
        var_scatter = (K * I0) ** 2 * torch.expm1(w2)
        # `sigma` is the counting variance alone.  When the likelihood carries
        # the background ESTIMATE's variance the model claims
        # lam + sigma_b^2 + scatter, and omitting that term makes `var_z` rise
        # purely because the width head shrank -- so the metric would report a
        # calibration change when nothing about the residuals moved.  At this
        # data's median bg the term is 135 against sigma^2 = 136, so it is not
        # a refinement.
        bvf = float(getattr(getattr(self.objective, "likelihood", None),
                            "bg_var_factor", 0.0) or 0.0)
        var = (sigma**2 + bvf * bg.clamp(min=0.0)
               + var_scatter).clamp(min=1e-12)
        resid = (n_ph - bg) - K * I0
        tgt = MergeTargetOut(
            log_I0=I0.clamp(min=1e-12).log(), I0=I0, resid=resid, var=var,
            z=resid / var.sqrt(),
            valid=torch.isfinite(I0) & (I0 > 0),
            n_floored=post.n_floored,
            var_frac_scatter=var_scatter / var,
        )
        return tgt, post.log_Z

    @torch.no_grad()
    def _canonical_nll(self, rows: Tensor) -> float:
        """Data NLL on this batch at **fixed** lambda and with no penalty.

        The optimised loss moves for reasons that have nothing to do with the
        model getting better: ``lam_spread`` is annealing, and ``lambda``
        changes the merge target itself (as it decays the target shifts from a
        stationary EMA to a noisier in-batch merge, which raises the loss on its
        own).  So the training curve is not comparable across a run.  This is
        the same batch scored under conditions that never change, which is the
        number to read for "is the model actually improving".
        """
        self.scale.eval()
        sched = self.target.lam
        self.target.lam = self.cfg.eval_lam
        try:
            _, _, tgt, nll, *_ = self._forward(rows, grad=False)
        finally:
            self.target.lam = sched
            self.scale.train()
        self.scale.clear()
        return float(nll.item()) / max(int(tgt.valid.sum()), 1)

    def step(self, rows: Tensor) -> dict:
        self.scale.train()
        if self.cfg.warmup_steps > 0:
            scale_lr = min(1.0, (self.step_i + 1) / self.cfg.warmup_steps)
            for grp in self.opt.param_groups:
                grp["lr"] = self._base_lr * scale_lr
        t0 = time.time()
        meta, pred, tgt, nll, n_ph, bg, sigma = self._forward(rows, grad=True)

        n_used = int(tgt.valid.sum())
        nll_data = float(nll.detach().item()) / max(n_used, 1)   # no penalty
        loss = nll / max(n_used, 1)
        lam_s = self._lam_spread()
        if lam_s > 0:
            loss = loss + lam_s * pred.log_w.mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = grad_norms(self.scale)
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.scale.parameters(), self.cfg.grad_clip
        )
        gn["gradnorm_total"] = float(total_norm)          # pre-clip global norm
        gn["grad_was_clipped"] = bool(float(total_norm) > self.cfg.grad_clip)
        self.opt.step()
        self.scale.clear()

        rec = {
            "step": self.step_i,
            "phase": "train",
            # What the optimiser actually minimised (penalty included).
            "loss_total": float(loss.item()),
            # ``nll_per_obs`` is the DATA term in both train and held-out
            # records.  It previously aliased ``loss_total`` here while
            # ``evaluate()`` wrote the penalty-free objective under the same
            # key -- two different quantities under one name.  That made the
            # spread penalty (negative, since log w < 0) look like training
            # progress: loss_total fell 5.37 -> 4.67 while the model's actual
            # data term ROSE 5.12 -> 5.37.
            "nll_per_obs": nll_data,
            "nll_data": nll_data,
            # Under ``loss="evidence"`` the numerator is a sum over
            # REFLECTIONS (-sum_h log Z_h) while the denominator counts
            # OBSERVATIONS, so "per obs" is only a scale convention -- and the
            # two are not in a fixed ratio.  log Z_h carries one O(1) term per
            # reflection that does NOT grow with multiplicity, so the per-obs
            # number picks up an offset ~ n_hkl_used / n_obs.  Train batches
            # here hold 1.5 M observations and the held-out set 0.63 M over
            # nearly the same reflections, so that offset differs by ~2.4x
            # between them: a train-vs-heldout GAP under this loss is partly an
            # artefact of the normaliser, not generalisation.  Read each curve's
            # trend, and use ``evidence_per_hkl`` when comparing sets.
            "evidence_per_hkl": (
                nll_data * max(n_used, 1) / max(getattr(self, "last_n_hkl", 0), 1)
                if self.cfg.loss == "evidence" else float("nan")
            ),
            "n_hkl_used": getattr(self, "last_n_hkl", 0),
            "n_obs": int(rows.numel()),
            "n_crystals": int(meta.crystal_id.unique().numel()),
            "n_used": n_used,
            "n_floored": tgt.n_floored,
            "lam": self.target.lam,
            "lam_spread": lam_s,
            "lr": float(self.opt.param_groups[0]["lr"]),
            "sec": time.time() - t0,
            **gn,
        }
        if self.step_i % self.cfg.log_every == 0:
            # Fixed-lambda, no-penalty score on this same batch: the only
            # training number that is comparable across the whole run.
            rec["nll_canonical"] = self._canonical_nll(rows)
            rec.update(self._diagnostics(meta, pred, tgt, rows))
        self.step_i += 1
        self.logger.log(rec)
        return rec

    # ── metrics ────────────────────────────────────────────────────────

    @torch.no_grad()
    def _diagnostics(self, meta, pred, tgt, rows) -> dict:
        # Observation-space normalised residual: (n' - K*I0) / sqrt(V).
        z = torch.where(tgt.valid, tgt.z, torch.full_like(tgt.z, float("nan")))

        out: dict = {
            # A. loss decomposition — never log only the total.  nll falling
            # while median_w rises is the network buying fit with uncertainty,
            # which is invisible in the total and is exactly how Sdfac -> 4.75.
            "median_w": float(pred.log_w.exp().median().item()),
            "mean_log_w": float(pred.log_w.mean().item()),
            "median_beta": float(pred.beta.median().item()),
            "mean_log_alpha": float(pred.log_alpha.mean().item()),
            # F. is the width head doing anything?
            "var_frac_scatter": float(tgt.var_frac_scatter.median().item()),
            # The in-batch merge needs within-batch agreement: with n_hkl
            # reflections, a batch of N crystals gives only ~N*obs/n_hkl
            # observations per reflection.  Shrink the batch (e.g. for an
            # O(L^2) encoder) and the merge target silently degrades -- which
            # is the one thing everything else depends on.  Log it.
            "obs_per_reflection_in_batch": float(
                torch.bincount(meta.hkl_idx, minlength=self.prep.n_valid)
                .float().clamp(min=1).median().item()
            ),
            "reflections_in_batch": int(meta.hkl_idx.unique().numel()),
        }
        em = getattr(self, "last_edge_mass", None)
        if em is not None:
            # Validity condition for detaching the grid geometry: if the grid
            # truncates, the integral DOES depend on node placement and the
            # detached gradient is wrong, not merely unnecessary.
            out["posterior_edge_mass_med"] = float(em.median().item())
            out["posterior_edge_mass_max"] = float(em.max().item())
        gm = getattr(self.objective, "last_grid_mass", None)
        if gm is not None:
            # Share of the scatter density actually inside the quadrature grid.
            # Near 1 => the truncation correction is immaterial; well below 1
            # => the scatter is wider than the grid can represent and the
            # likelihood is only a density because we divided this out.
            out["grid_mass_median"] = float(gm.median().item())
            out["grid_mass_p01"] = float(gm.quantile(0.01).item())
        ios = (self.n_ph[rows] - self.prep.bg[rows]) / self.sigma[rows]
        out.update(calibration_metrics(
            z, shell_id=self.shell_of_obs[rows],
            n_shells=self.cfg.shell_bins, i_over_sigma=ios,
        ))
        # Is the width head actually using its per-observation freedom?  var_z
        # runs 0.01 -> 1.21 across I/sigma deciles, which a per-observation
        # width should be able to flatten.  If median w is flat across the same
        # deciles the head is predicting a constant in all but name.
        from torchsx.nn.metrics import _rank
        dec = (_rank(ios) / max(ios.numel(), 1) * 10).long().clamp(max=9)
        out["median_w_by_isigma_decile"] = [
            float(pred.log_w[dec == d].exp().median().item())
            if int((dec == d).sum()) > 8 else float("nan")
            for d in range(10)
        ]
        out["median_beta_by_isigma_decile"] = [
            float(pred.beta[dec == d].median().item())
            if int((dec == d).sum()) > 8 else float("nan")
            for d in range(10)
        ]
        out["var_frac_scatter_by_isigma_decile"] = [
            float(tgt.var_frac_scatter[dec == d].median().item())
            if int((dec == d).sum()) > 8 else float("nan")
            for d in range(10)
        ]
        out.update(scale_metrics(
            pred.log_K, meta.crystal_id, meta.n_crystals,
            features={
                "s_sq": meta.s_sq,
                "ewald_offset": meta.ewald_offset,
                "d_rank": self.shell_of_obs[rows].float(),
            },
        ))
        out["spearman_log_w_ewald"] = (
            float(_spearman_safe(pred.log_w, meta.ewald_offset.abs()))
            if meta.ewald_offset is not None else float("nan")
        )
        if self.is_light is not None:
            out["condition_auc"] = condition_auc(
                pred.log_K, self.is_light[meta.crystal_id]
            )
        return out

    @torch.no_grad()
    def evaluate(self, crystals: Tensor, *, phase: str = "heldout") -> dict:
        """Score crystals the network never trained on.

        In-batch loss falling while this does not is co-adaptation through the
        merge rather than learning — the diagnostic that was missing throughout
        the abismal investigation.

        Scored at a **fixed** ``cfg.eval_lam``, not the current schedule value.
        The training NLL is not comparable across the λ decay, because λ changes
        the merge target itself: as λ falls the target moves from a stationary
        EMA to a noisier in-batch merge, and the loss rises for that reason
        alone.  Pinning λ here makes the held-out curve an actual yardstick.
        """
        self.scale.eval()
        sched_lam = self.target.lam
        tot, n_tot, n_hkl, recs = 0.0, 0, 0, []
        for sel, rows in self.index.batches(
            self.cfg.max_obs_per_batch, crystals=crystals, shuffle=False
        ):
            self.target.lam = self.cfg.eval_lam
            meta, pred, tgt, nll, *_ = self._forward(rows, grad=False)
            n_used = int(tgt.valid.sum())
            tot += float(nll.item())
            n_tot += n_used
            n_hkl += getattr(self, "last_n_hkl", 0)
            recs.append(
                calibration_metrics(torch.where(tgt.valid, tgt.z, torch.nan))
            )
            self.scale.clear()
        self.target.lam = sched_lam
        rec = {
            "step": self.step_i, "phase": phase,
            "nll_per_obs": tot / max(n_tot, 1),
            # See step(): under loss="evidence" this is a reflection sum over an
            # observation count, so it is comparable along its own curve but not
            # against the training curve, which has a different multiplicity.
            "evidence_per_hkl": (
                tot / max(n_hkl, 1) if self.cfg.loss == "evidence" else float("nan")
            ),
            "n_hkl_used": n_hkl,
            "n_obs": n_tot,
            "lam": self.cfg.eval_lam,
            "var_z": _mean_of([r.get("var_z") for r in recs]),
            "frac_gt3": _mean_of([r.get("frac_gt3") for r in recs]),
        }
        if (
            self.cfg.keep_best and phase == "heldout"
            and rec["nll_per_obs"] == rec["nll_per_obs"]
            and rec["nll_per_obs"] < self.best_heldout
        ):
            self.best_heldout = rec["nll_per_obs"]
            self.best_step = self.step_i
            self._best_state = {
                k: v.detach().clone() for k, v in self.scale.state_dict().items()
            }
        rec["best_heldout"] = self.best_heldout
        self.logger.log(rec)
        return rec

    def save_best(self, path=None) -> "Path | None":
        """Persist the best held-out parameters so a finished run can be
        re-analysed without retraining it.

        Nothing was written to disk before this, so every post-hoc question
        (merge statistics, a different split, per-shell breakdowns) meant a
        full re-run.
        """
        if self._best_state is None:
            return None
        path = Path(path) if path is not None else Path(self.cfg.outdir) / "best.pt"
        torch.save(
            {
                "state_dict": self._best_state,
                "best_step": self.best_step,
                "best_heldout": self.best_heldout,
                "n_crystals": self.index.n_crystals,
                "n_hkl": self.prep.n_valid,
                "ema_I0": self.target.ema_I0.detach().cpu(),
                "heldout_crystals": self.heldout_crystals.cpu(),
                "train_crystals": self.train_crystals.cpu(),
            },
            path,
        )
        return path

    def restore_best(self) -> bool:
        """Load the best-scoring held-out parameters.  Returns False if none."""
        if self._best_state is None:
            return False
        self.scale.load_state_dict(self._best_state)
        return True

    @torch.no_grad()
    def _merge_set(self, crystals: Tensor) -> tuple[Tensor, Tensor]:
        """Merge one crystal set with the current network.

        Deliberately **no separate estimator**: whatever the training objective
        merges with is what gets scored.  Otherwise CC1/2 measures a different
        quantity from the one being minimised, and improving the loss need not
        improve the metric.

        For the posterior target that means accumulating the same coarse
        log-likelihood curve the loss uses (additive over observations, hence
        over batches) and integrating once.  For the leave-one-out target it is
        the same inverse-variance sums that target is built from.
        """
        n_hkl = self.prep.n_valid
        posterior = self.score_target is not None
        tgt = self.score_target
        if posterior:
            if tgt.node_placement == "wilson":
                v_nodes, log_w_nodes = tgt.nodes()
            else:
                # Locate the peak on THIS crystal subset first.  Using the
                # closed-form width here instead collapsed the merge onto the
                # shared ema_I0 (per-shell CC1/2 read exactly 1.0000).
                scan = tgt.scan_grid()
                acc_s = torch.zeros(n_hkl, scan.shape[1], device=self.device)
                for _, r in self.index.batches(
                    self.cfg.max_obs_per_batch, crystals=crystals, shuffle=False
                ):
                    r = r.to(self.device)
                    m = self.prep.obs_info.gather(r)
                    pr = self.scale.prime(m)
                    acc_s = acc_s + tgt.accumulate(
                        self.objective, self.scale.scatter, self.params, pr,
                        self.n_ph[r], self.prep.bg[r], m.hkl_idx, m, scan,
                    )
                    self.scale.clear()
                c, sd = tgt.peak_from_scan(acc_s)
                v_nodes = c[:, None] + sd[:, None] * tgt.u_coarse[None, :]
                log_w_nodes = None
                tgt._loc = (c, sd)
            acc = torch.zeros(n_hkl, v_nodes.shape[1], device=self.device)
        else:
            num = torch.zeros(n_hkl, device=self.device)
            den = torch.zeros(n_hkl, device=self.device)

        for _, rows in self.index.batches(
            self.cfg.max_obs_per_batch, crystals=crystals, shuffle=False
        ):
            rows = rows.to(self.device)
            meta = self.prep.obs_info.gather(rows)
            pred = self.scale.prime(meta)
            n_signal = self.n_ph[rows] - self.prep.bg[rows]
            if posterior:
                acc = acc + tgt.accumulate(
                    self.objective, self.scale.scatter, self.params, pred,
                    self.n_ph[rows], self.prep.bg[rows], meta.hkl_idx, meta,
                    v_nodes,
                )
            else:
                K = pred.log_K.exp().clamp(min=1e-8)
                I0_ref = self.target.ema_I0[meta.hkl_idx].clamp(
                    min=self.target.floor
                )
                w2 = (pred.log_w.exp() ** 2).clamp(max=20.0)
                V = (
                    self.sigma[rows] ** 2 + (K * I0_ref) ** 2 * torch.expm1(w2)
                ).clamp(min=1e-12)
                num.index_add_(0, meta.hkl_idx, K * n_signal / V)
                den.index_add_(0, meta.hkl_idx, K * K / V)
            self.scale.clear()

        if posterior:
            _, I0, *_ = tgt.integrate(acc, v_nodes, log_w_nodes)
            seen = (acc.abs().sum(1) > 0).float()
            # (value, weight) so the caller's ratio form works unchanged
            return I0 * seen, seen
        return num, den

    @torch.no_grad()
    def split_half_metrics(self, *, n_shells: int = 10, log: bool = True) -> dict:
        """CC1/2 and R_split on TRAIN and VALIDATION crystals, matched in count.

        Four merges of equal size: ``train_A/train_B`` and ``val_A/val_B``.
        ``cc_half_train`` is merge quality on crystals the network has trained
        on; ``cc_half_val`` is the same statistic on crystals it has never
        seen, at identical multiplicity.  The gap between them is the
        generalisation of the *scale model* — a per-crystal G has no such gap
        by construction, because it is refit for every crystal.

        Computed on **intensities**, not amplitudes.  Note this makes the
        numbers NOT directly comparable to the figures in
        ``experiments/benchmark_refactored_genpoiss/RESULTS.md`` (EM Poisson
        0.9872, partialator unity 0.9850) — and those were measured on three
        negative-delay streams from the ``sfx_scan`` dataset, not on this one,
        so they were never a valid baseline here regardless.  The baseline that
        counts is the same-data, same-protocol unity merge.
        """
        from torchsx.merging.hkl.stats import (
            MatchedHalves, cc_half, cc_star, r_split, shell_stats,
        )

        was_training = self.scale.training
        self.scale.eval()
        merged = {k: self._merge_set(v) for k, v in self.merge_sets.items()}
        # CC1/2 on INTENSITIES, not amplitudes.  ``to_F = I.clamp(min=0).sqrt()``
        # is the repo's benchmark convention, but clamping sends every
        # negative merged intensity to exactly 0, and identical values in both
        # halves read to Pearson as perfect agreement.  Measured here the
        # both-negative fraction is under 1 % so it changes little (0.9918 vs
        # 0.9928 overall), but intensities are the honest choice: negatives are
        # real measurements and must be allowed to disagree.

        def _pair(a: str, b: str) -> tuple[MatchedHalves, Tensor]:
            (na, da), (nb, db) = merged[a], merged[b]
            ok = (da > 0) & (db > 0)
            Ia = torch.where(da > 0, na / da.clamp(min=1e-30), torch.zeros_like(na))
            Ib = torch.where(db > 0, nb / db.clamp(min=1e-30), torch.zeros_like(nb))
            return MatchedHalves(
                F1=Ia[ok].cpu(), F2=Ib[ok].cpu(),
                d_spacing=self.prep.d_spacing[ok].cpu(),
                hkl=self.prep.hkl_valid[ok].cpu(),
            ), ok

        rec = {
            "step": self.step_i, "phase": "split_half",
            "n_merge_crystals": int(self.n_merge_crystals),
        }
        for tag, (a, b) in {
            "train": ("train_A", "train_B"),
            "val": ("val_A", "val_B"),
            "full": ("full_A", "full_B"),
            "rep": ("rep_A", "rep_B"),
        }.items():
            m, ok = _pair(a, b)
            cc = cc_half(m)
            rec[f"cc_half_{tag}"] = cc
            rec[f"cc_star_{tag}"] = cc_star(cc)
            rec[f"r_split_{tag}"] = r_split(m)
            rec[f"n_matched_{tag}"] = int(ok.sum())
            if tag in ("val", "full"):
                sh = shell_stats(m, n_shells=n_shells)
                # Per tag, not shared.  Shells are quantiles of d over the
                # MATCHED reflections and ``ok`` differs between the pairings,
                # so a single ``shell_d_min`` key was whichever tag ran last
                # (``full``) while ``shell_cc_half_val`` was still printed
                # against it -- the val shells carried the full-merge
                # boundaries.
                rec[f"shell_d_min_{tag}"] = [x.d_min for x in sh]
                rec[f"shell_cc_half_{tag}"] = [x.cc_half for x in sh]
                if tag == "full":
                    rec["shell_d_min"] = rec["shell_d_min_full"]  # back-compat
        rec["cc_half_gap"] = rec["cc_half_train"] - rec["cc_half_val"]
        # |full - rep| is two independent splits of identical data, so it is a
        # direct read on the statistic's own scatter -- the floor below which
        # differences between runs mean nothing.
        rec["cc_half_split_scatter"] = abs(rec["cc_half_full"] - rec["cc_half_rep"])
        if log:
            self.logger.log(rec)
        if was_training:
            self.scale.train()
        return rec

    @torch.no_grad()
    def audit(self, rows: Tensor) -> dict:
        """Per-epoch leakage battery.  Run before trusting any headline number."""
        self.scale.eval()
        meta = self.prep.obs_info.gather(rows)
        rec = {
            "step": self.step_i,
            "phase": "audit",
            # D. a scale that depends on hkl is degenerate with I0 and is the
            # one channel that can manufacture a difference peak.
            "hkl_permutation_frac": hkl_permutation_sensitivity(self.scale, meta),
        }
        frac = rec["hkl_permutation_frac"]
        if frac == frac and frac > self.cfg.hkl_permutation_warn:
            warnings.warn(
                f"hkl-permutation sensitivity {frac:.3f} exceeds "
                f"{self.cfg.hkl_permutation_warn}: the scale is becoming a "
                f"function of hkl, which is degenerate with I0 and is the one "
                f"channel that can manufacture a difference peak. Drop hkl from "
                f"the features before any light/dark work.",
                RuntimeWarning, stacklevel=2,
            )
        self.logger.log(rec)
        return rec

    # ── driver ─────────────────────────────────────────────────────────

    def _refresh_wilson_sigma(self) -> None:
        """Keep the Wilson Sigma on the shell mean of the running merge."""
        t = self.target
        if hasattr(t, "set_wilson_sigma"):
            sid = self.prep.shell_id_fw.to(self.device)
            t.set_wilson_sigma(sid, int(sid.max().item()) + 1)

    def _refresh_feature_reference(self) -> None:
        """Roll the obs_scale reference forward with the running merge."""
        feats = getattr(self.scale, "features", None)
        if feats is not None and hasattr(feats, "set_reference_I0"):
            feats.set_reference_I0(self.target.ema_I0)

    def run(self) -> None:
        for epoch in range(self.cfg.epochs):
            self._refresh_feature_reference()
            self._refresh_wilson_sigma()
            for _, rows in self.index.batches(
                self.cfg.max_obs_per_batch,
                generator=self.gen,
                crystals=self.train_crystals,
            ):
                self.step(rows.to(self.device))
            if self.heldout_crystals.numel():
                self.evaluate(self.heldout_crystals)
            if self.cfg.train_eval_crystals > 0:
                self.evaluate(
                    self.train_crystals[: self.cfg.train_eval_crystals],
                    phase="train_eval",
                )
            probe = self.index.rows(self.train_crystals[:256])
            self.audit(probe.to(self.device))
            self.save_best()
            if (
                self.cfg.split_half_every > 0
                and (epoch + 1) % self.cfg.split_half_every == 0
            ):
                sh = self.split_half_metrics()
                print(
                    f"    [merge] ep {epoch}  "
                    f"train CC1/2={sh['cc_half_train']:.4f} Rsplit={sh['r_split_train']:.4f} | "
                    f"val CC1/2={sh['cc_half_val']:.4f} Rsplit={sh['r_split_val']:.4f} | "
                    f"gap={sh['cc_half_gap']:+.4f}  ({sh['n_merge_crystals']:,} xtals each)",
                    flush=True,
                )

    def close(self) -> None:
        """Close the metric sink.  ``run()`` deliberately does not — callers
        routinely keep auditing and evaluating after training finishes."""
        self.logger.close()


def _spearman_safe(a, b) -> float:
    from torchsx.nn.metrics import spearman
    return spearman(a, b)


def _mean_of(vals) -> float:
    good = [v for v in vals if v is not None and v == v]
    return sum(good) / len(good) if good else float("nan")
