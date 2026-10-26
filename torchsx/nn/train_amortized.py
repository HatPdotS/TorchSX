"""Training loop for the amortised scale model.

Batches are drawn over **crystals**, not observations, because the
leave-one-out context pool needs a crystal's reflections to be complete: a
truncated crystal gives a biased context.  The merge target is the opposite —
it is an average, so a subsample of each reflection's observations is unbiased,
and a batch of a few thousand crystals holds tens of observations per reflection.

That is why the batch is sized in *observations* rather than crystals: crystals
carry a variable number (long tail), so a fixed crystal count makes
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

import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from torchsx.merging.observations import PreparedObs
from torchsx.nn.merge_target import MergeTargetOut, unity_merge
from torchsx.nn.metrics import (
    MetricLogger,
    _diagnostic_quantile,
    calibration_metrics,
    condition_auc,
    grad_norms,
    hkl_permutation_sensitivity,
    hkl_permutation_sensitivity_shape,
    scale_metrics,
)
from torchsx.nn.shape_prior import pack_single, shape_prior_kl

__all__ = ["CrystalIndex", "TrainConfig", "AmortizedTrainer"]


def _bg_mean_offset(objective) -> float:
    """The detection background-mean offset the objective's likelihood carries."""
    return float(getattr(getattr(objective, "likelihood", None),
                         "bg_mean_offset", 0.0) or 0.0)


class CrystalIndex:
    """Crystal-major view of the hkl-sorted observation list.

    ``PreparedObs`` is sorted by ``hkl_id`` (the merge needs contiguous
    reflections).  Training needs the opposite grouping, so this builds the
    permutation once and serves ``crystal -> observation rows`` from CSR
    offsets.  A :class:`~torchsx.merging.store.PreparedStore` is crystal-major
    already: :meth:`from_counts` serves its rows with no permutation.
    """

    def __init__(
        self, crystal_id: Tensor, n_crystals: int, condition: Tensor | None = None,
    ) -> None:
        self._setup(crystal_id.argsort(stable=True),
                    torch.bincount(crystal_id, minlength=n_crystals), condition)

    @classmethod
    def from_counts(cls, counts: Tensor, condition: Tensor | None = None) -> "CrystalIndex":
        """Index of crystal-major rows: crystal ``c`` owns rows ``[offsets[c], offsets[c+1])``."""
        index = cls.__new__(cls)
        index._setup(None, counts.to(torch.long).cpu(), condition)
        return index

    def _setup(self, order: Tensor | None, counts: Tensor, condition: Tensor | None) -> None:
        n_crystals = int(counts.numel())
        offsets = torch.zeros(n_crystals + 1, dtype=torch.long)
        offsets[1:] = counts.cumsum(0)
        #: rows in crystal order, or None when the rows already are
        self.order = order
        self.counts = counts
        self.offsets = offsets
        self.n_crystals = n_crystals
        self.nonempty = torch.nonzero(counts > 0, as_tuple=True)[0]
        # Per-crystal condition (light/dark, delay bin).  Batches are then
        # drawn from ONE condition each: the in-batch merge target for a
        # reflection must never pool dark and light, or the difference between
        # them is averaged away before the loss ever sees it.  The label is
        # ``ObsInfo.delay_bin_idx``, i.e. the stream's image-level timepoint.
        if condition is not None:
            condition = condition.detach().to(torch.long).cpu()
            if condition.shape[0] != self.n_crystals:
                raise ValueError(
                    f"condition must have one entry per crystal "
                    f"({self.n_crystals}), got {tuple(condition.shape)}"
                )
            if bool((condition[self.nonempty] < 0).any()):
                n_bad = int((condition[self.nonempty] < 0).sum())
                raise ValueError(
                    f"{n_bad} crystals carry condition -1 (unlabelled image). "
                    "Drop them from the stream before training, or do not label it."
                )
        self.condition = condition
        self.n_conditions = (
            int(condition.max()) + 1 if condition is not None and condition.numel() else 1
        )

    def rows(self, crystals: Tensor) -> Tensor:
        """All observation rows of the given crystals, as a flat index.

        Vectorised rather than a per-crystal Python loop: a batch holds
        thousands of crystals, and building one tensor each costs more than
        twice the arithmetic form.
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
        flat = within + starts.repeat_interleave(counts)
        return flat if self.order is None else self.order[flat]

    def batches(
        self, max_obs: int, *, generator: torch.Generator | None = None,
        crystals: Tensor | None = None, shuffle: bool = True,
    ):
        """Yield ``(crystal_sel, obs_rows)`` with at most ``max_obs`` rows each.

        With a condition label every batch holds crystals of ONE condition: the
        (shuffled) pool is split per condition, each part is filled greedily
        as before, and the resulting batches are then shuffled as a list so
        the conditions alternate rather than run dark-then-light.  Without a
        label (or with a single condition) this is bit-identical to the
        unconditioned batcher for the same generator state.
        """
        pool = self.nonempty if crystals is None else crystals
        if shuffle:
            pool = pool[torch.randperm(pool.numel(), generator=generator)]
        pools = [pool]
        if self.condition is not None and self.n_conditions > 1:
            cb = self.condition[pool]
            pools = [pool[cb == b] for b in range(self.n_conditions)]
            pools = [q for q in pools if q.numel()]
        sels: list[Tensor] = []
        for q in pools:
            c = self.counts[q]
            start, running = 0, 0
            for i in range(q.numel()):
                ci = int(c[i])
                if running + ci > max_obs and i > start:
                    sels.append(q[start:i])
                    start, running = i, 0
                running += ci
            if start < q.numel():
                sels.append(q[start:])
        if shuffle and len(pools) > 1:
            order = torch.randperm(len(sels), generator=generator).tolist()
            sels = [sels[i] for i in order]
        for sel in sels:
            yield sel, self.rows(sel)


def lr_factor(step: int, *, warmup: int, total: int, schedule: str = "constant",
              min_frac: float = 0.0) -> float:
    """Learning-rate multiplier at update ``step`` (0-based).

    Linear warm-up over ``warmup`` updates; then 1 (``"constant"``) or a cosine
    from 1 at the end of the warm-up to ``min_frac`` at update ``total``.
    """
    if schedule not in ("constant", "cosine"):
        raise ValueError(f"lr_schedule must be 'constant' or 'cosine', got {schedule!r}")
    f = min(1.0, (step + 1) / warmup) if warmup > 0 else 1.0
    if schedule == "cosine" and total > warmup:
        t = min(max((step - warmup) / (total - warmup), 0.0), 1.0)
        f *= min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * t))
    return f


@dataclass
class TrainConfig:
    #: Warn when the hkl-permutation sensitivity exceeds this.  A scale that
    #: depends on hkl is degenerate with I0 and can manufacture a difference
    #: peak.
    hkl_permutation_warn: float = 0.02
    max_obs_per_batch: int = 1_500_000
    #: Size the batch by the in-batch REDUNDANCY it buys, not by an observation
    #: count.  The in-batch merge target for a reflection is built from roughly
    #: ``max_obs_per_batch / n_hkl`` observations, and that -- not the number of
    #: crystals or observations -- is what decides whether the target is any
    #: good.  The same 1.5 M observations give ~100 obs/reflection on a small
    #: cell with ~10 k unique reflections and ~10 on a protein cell with
    #: ~100 k, so a batch size carried across data sets silently changes the
    #: objective.  When set, ``max_obs_per_batch`` is derived as
    #: ``target_redundancy * n_hkl`` and the configured value is ignored.
    target_redundancy: float | None = None
    epochs: int = 10
    lr: float = 1e-3
    #: Learning-rate multiplier for the mixture head's parameters.  The head
    #: reads the full embedding like the other heads; a large multiplier drives
    #: the lower widths to extreme alpha within a few hundred updates.
    mixture_lr_mult: float = 1.0
    #: Learning-rate multiplier for the pseudo-translation profile coefficients
    #: (``PhaseWidth``, ``PhaseBroad``): a handful of scalars that must travel
    #: O(1) in log width and log fraction, where Adam at the network's rate
    #: moves them ~lr per update.
    pseudo_translation_lr_mult: float = 30.0
    #: Learning-rate multiplier for the learned detector noise (per-panel log A,
    #: log B).  They may have to move by several nats within the budget -- B by 2.3
    #: for a gain off by 10x, A up from its near-zero start -- and
    #: Adam moves a parameter by at most ~lr per update.  Each is projected into
    #: its bounds after every step, so a persistent push cannot run away.
    noise_lr_mult: float = 30.0
    #: Linear LR warmup.  Matters once the encoder is a deep stack of exchange
    #: blocks: the zero-initialised output projections make every block start
    #: as the identity, so all early gradient lands on the heads (orders of
    #: magnitude above its later size) and a constant LR spends the first
    #: steps taking large, badly-scaled steps.  0 disables.
    warmup_steps: int = 0
    #: Learning rate after the warm-up: ``"constant"``, or ``"cosine"`` decay
    #: from the base rate at the end of the warm-up to ``min_lr_frac`` of it at
    #: ``total_steps``.
    lr_schedule: str = "constant"
    total_steps: int = 0
    min_lr_frac: float = 0.0
    #: Decoupled (AdamW) weight decay.  Applied to matrix parameters ONLY --
    #: norm gains and biases are excluded, because decaying an RMSNorm gain
    #: shrinks the network's effective scale rather than regularising it.
    #: Note this is genuinely decoupled: plain ``Adam(weight_decay=)`` adds L2
    #: to the gradient, which Adam then divides by sqrt(v), so parameters with
    #: small gradients would be decayed hardest -- the decay strength ends up
    #: coupled to gradient history rather than being a clean prior.
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    #: Penalty on the predicted scatter width, ``lam_spread * mean(log w)``.
    #: The GenGauss normaliser already contains -log(alpha), so no explicit
    #: penalty on w is needed *at the likelihood optimum*; that says nothing
    #: about which optimum you reach.  The model can settle into a wide-scatter
    #: minimum where the width absorbs nearly all per-observation variation,
    #: and there is then almost no gradient pressure left to
    #: refine K per observation.  This penalty is a tool for escaping that
    #: basin, not a claim about the statistics -- so it is annealed away and
    #: the final w is judged on calibration (var_z), not on this term.
    lam_spread: float = 0.0
    #: Halflife of the spread penalty in steps; ``0`` disables the ramp and
    #: holds ``lam_spread`` constant for the whole run.
    lam_spread_halflife_steps: int = 400
    #: Recompute each outer quadrature node in backward instead of keeping all
    #: ``k_coarse`` node graphs alive at once.  Trades ~2x compute in
    #: ``accumulate`` for a large drop in retained activations.  Only worth it
    #: when the batch is memory-bound: a large cell gets several times less
    #: in-batch redundancy than a small one at the same batch size, and
    #: redundancy is what decides whether the in-batch merge target is any good.
    #: Keep ``prep`` on the host and copy only the current batch's rows to the
    #: device.  On a large data set ``prep`` (mostly ``obs_info``) is a large
    #: share of device memory and does not shrink with the batch, so it is pure
    #: overhead; the transfer is milliseconds against a multi-second
    #: step.  Off by default: it is a loss on a data set that already fits.
    #: Crystals per encoder chunk (0 = the whole batch at once).  The encoder
    #: is separable across crystals -- only ``leave_one_out_mean`` couples rows,
    #: and only within a crystal -- so this is exact, not an approximation.
    encoder_chunk_crystals: int = 0
    stream_batches: bool = False
    posterior_checkpoint_nodes: bool = False
    posterior_k_coarse: int = 31
    posterior_m_fine: int = 257
    posterior_n_sigma: float = 8.0
    #: ``"wilson"``: the Wilson intensity prior on the smooth surface
    #: ``log Sigma(h) = phi(h) @ c``, whose coefficients are trained with the
    #: network by the same evidence (type-II maximum likelihood) at
    #: ``wilson_lr_mult`` times the base rate, started at the Wilson fit to the
    #: unity merge.  ``"none"``: integrate the likelihood in positive intensity.
    posterior_prior: str = "wilson"
    wilson_lr_mult: float = 10.0
    wilson_degree: int = 4
    #: Outer grid half-width cap in nats (see PosteriorMergeTarget.max_halfwidth).
    posterior_max_halfwidth: float = 12.0
    #: Hierarchical prior on each observation's scatter density
    #: (:mod:`torchsx.nn.shape_prior`): ``lam_shape_prior`` pseudo-observations
    #: per observation drawn from the batch's population density, i.e.
    #: ``lam * mean_j KL(p_bar || p_j)`` added to the per-observation loss.
    #: Weak observations, whose likelihood barely depends on their scatter
    #: shape, take the population shape; strong ones keep what their data say.
    #: Constant over the run (it is a prior, not an escape schedule).
    lam_shape_prior: float = 0.0
    #: observations sampled per update to estimate the prior term
    shape_prior_sample: int = 2048
    log_every: int = 10
    heldout_frac: float = 0.05
    #: Condition (``ObsInfo.delay_bin_idx`` value) the held-out crystals and
    #: the split-half merge sets are drawn from, so every validation number is
    #: a same-condition statistic.  Bins are sorted by timepoint, so 0 is the
    #: earliest label -- the dark set when dark is labelled with a negative
    #: delay.  ``None`` pools all conditions and is only allowed when the
    #: data carries a single one.
    heldout_condition: int | None = 0
    #: Explicit held-out crystals (indices into the prepared data), e.g. a fixed
    #: external held-out stream shared by many runs so their held-out NLL is on
    #: the same crystals.  They must belong to ``heldout_condition``; they are
    #: scored, never trained on, and the export leaves them out
    #: (``external_heldout``).  None: a random ``heldout_frac`` of that condition.
    heldout_crystal_ids: object = None
    #: Crystals scored from the TRAINING set under the same conditions as the
    #: held-out set (no spread penalty).  Without this the logged train and
    #: heldout numbers differ in the penalty, so their gap is bookkeeping
    #: rather than generalisation — and that gap is precisely the diagnostic
    #: for whether capacity binds.
    train_eval_crystals: int = 3500
    #: Keep the parameters that scored best on held-out.  A long run can peak
    #: before its end and then degrade as the spread penalty anneals and the
    #: model over-commits to within-crystal structure -- the endpoint need not
    #: be the model we want.
    keep_best: bool = True
    #: Run the split-half merge every N epochs (0 disables).  Costs one
    #: no-grad pass over the whole dataset.
    split_half_every: int = 20
    #: Scan half-widths for the merge's locating passes, one full data pass
    #: each.  A single wide round mis-centres the grid by about half a nat
    #: whatever the multiplicity (see ``merge``); shrinking the window squares
    #: that away.  Set to a single value for a one-pass locate.
    locate_windows: tuple[float, ...] = (8.0, 1.0, 0.125)
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
        from torchsx.merging.store import PreparedStore

        self.cfg = config or TrainConfig()
        self.device = torch.device(self.cfg.device)
        self.prep = prep
        #: Observations on disk: batches are read per step, never held whole.
        self._store = prep if isinstance(prep, PreparedStore) else None
        #: Updates dropped for being non-finite (see ``step``).
        self.n_skipped_steps = 0
        self.scale = scale.to(self.device)
        self.objective = objective
        self.params = params

        oi = prep.obs_info
        # Condition per crystal from the stream's image-level timepoint label;
        # a stream that was never labelled is one condition.
        self.condition = (
            oi.delay_bin_idx.detach().to(torch.long).cpu()
            if oi.delay_bin_idx is not None
            else torch.zeros(oi.n_crystals, dtype=torch.long)
        )
        if self._store is not None:
            self.index = CrystalIndex.from_counts(prep.crystal_counts(), condition=self.condition)
        else:
            self.index = CrystalIndex(oi.crystal_id.cpu(), oi.n_crystals,
                                      condition=self.condition)
        self.n_conditions = self.index.n_conditions
        self._cond = 0                      # condition of the batch in flight
        if is_light is None and self.n_conditions > 1:
            is_light = self.condition != 0
        self.is_light = None if is_light is None else is_light.to(self.device)
        # Batches are condition-pure, so an in-batch condition AUC is
        # undefined; keep a subsample of log K from the latest batch of every
        # condition and score across them.
        self._auc_pool: list = [None] * self.n_conditions
        if self.n_conditions > 1:
            n_per = torch.bincount(self.condition[self.index.nonempty],
                                   minlength=self.n_conditions).tolist()
            print(f"  [condition] {self.n_conditions} conditions, crystals per "
                  f"condition {n_per}; batches are condition-pure", flush=True)
        if self.cfg.target_redundancy is not None:
            want = int(round(self.cfg.target_redundancy * prep.n_valid))
            avail = int(prep.n_obs_total)
            self.cfg.max_obs_per_batch = min(want, avail)
            print(
                f"  [batch] target_redundancy={self.cfg.target_redundancy:g} "
                f"x {prep.n_valid:,} reflections -> max_obs_per_batch="
                f"{self.cfg.max_obs_per_batch:,}"
                + ("  (capped at the whole set)" if want > avail else "")
                + f"  ~{self.cfg.max_obs_per_batch / max(prep.n_valid, 1):.0f}"
                f" obs/reflection",
                flush=True,
            )
        if self.cfg.encoder_chunk_crystals:
            self.scale.set_encoder_chunking(self.cfg.encoder_chunk_crystals)
        if self._store is None:
            self.n_ph = (prep.I_obs + prep.bg).clamp(min=0)
            self.sigma = (
                oi.sigma if oi.sigma is not None
                else (self.n_ph + prep.bg.clamp(min=0) + 1.0).sqrt()
            )
        if self.cfg.posterior_prior not in ("none", "wilson"):
            raise ValueError(f"posterior_prior must be 'none' or 'wilson', got "
                             f"{self.cfg.posterior_prior!r}")
        # One posterior target PER CONDITION: dark and light keep their own
        # merge and Wilson surface while the network is shared.  None carries
        # per-reflection state between batches; the unity merge only starts
        # each merge's grid search and the Wilson fit.
        from torchsx.nn.posterior_target import PosteriorMergeTarget
        I_start = (prep.unity_merge() if self._store is not None else
                   unity_merge(prep.I_obs, self.sigma, prep.hkl_id, prep.n_valid))
        self.targets: list = []
        for _ in range(self.n_conditions):
            t = PosteriorMergeTarget(
                I_start,
                k_coarse=self.cfg.posterior_k_coarse,
                m_fine=self.cfg.posterior_m_fine,
                n_sigma=self.cfg.posterior_n_sigma,
                prior=self.cfg.posterior_prior,
                max_halfwidth=self.cfg.posterior_max_halfwidth,
            ).to(self.device)
            t.checkpoint_nodes = bool(self.cfg.posterior_checkpoint_nodes)
            t.set_is_centric(prep.is_centric.to(self.device))
            self.targets.append(t)
        self._best_wilson: list | None = None

        self.gen = torch.Generator().manual_seed(self.cfg.seed)
        pool = self.index.nonempty
        # Held-out crystals come from ONE condition (cfg.heldout_condition), so
        # the held-out NLL and every split-half statistic below are
        # same-condition numbers.  Training gets the rest of that condition
        # plus every crystal of the others.
        hc = self.cfg.heldout_condition
        if hc is None:
            if self.n_conditions > 1:
                raise ValueError(
                    "heldout_condition=None pools conditions in the validation "
                    f"set; the data carries {self.n_conditions}. Pick one."
                )
            ref_pool, other = pool, pool[:0]
        else:
            if not 0 <= hc < self.n_conditions:
                raise ValueError(
                    f"heldout_condition={hc} but the data has "
                    f"{self.n_conditions} condition(s)"
                )
            in_ref = self.condition[pool] == hc
            ref_pool, other = pool[in_ref], pool[~in_ref]
        perm = ref_pool[torch.randperm(ref_pool.numel(), generator=self.gen)]
        self.external_heldout: Tensor | None = None
        if self.cfg.heldout_crystal_ids is not None:
            ids = torch.as_tensor(self.cfg.heldout_crystal_ids, dtype=torch.long).cpu()
            if hc is not None and bool((self.condition[ids] != hc).any()):
                raise ValueError(f"heldout_crystal_ids must all belong to condition {hc}")
            held = torch.isin(perm, ids)
            self.heldout_crystals = perm[held]
            ref_train = perm[~held]
            self.external_heldout = ids
        else:
            n_held = int(self.cfg.heldout_frac * perm.numel())
            self.heldout_crystals = perm[:n_held]
            ref_train = perm[n_held:]
        self.train_crystals = torch.cat([ref_train, other])

        decay, no_decay, mixture, profile = [], [], [], []
        for name, prm in self.scale.named_parameters():
            if not prm.requires_grad:
                continue
            if name.startswith("heads.mixture.") and self.cfg.mixture_lr_mult != 1.0:
                mixture.append(prm)
                continue
            if name.startswith(("features.phase_width.", "features.phase_broad.")):
                profile.append(prm)
                continue
            (no_decay if prm.ndim <= 1 else decay).append(prm)
        # the objective may carry its own scalars (the miss fraction); they are
        # fitted with the network, undecayed.  The detector noise gets its own rate.
        noise_ids = {id(p) for p in (self.objective.detector_noise.parameters()
                                     if getattr(self.objective, "detector_noise", None) is not None else [])}
        noise = []
        for prm in self.objective.parameters():
            if prm.requires_grad:
                (noise if id(prm) in noise_ids else no_decay).append(prm)
        # Merge-quality quartets, at MATCHED crystal count.  Merge statistics
        # depend strongly on multiplicity, so comparing a full train merge
        # against a validation merge a twentieth its size would measure the
        # count, not the model.  Both sets are subsampled to the same size and
        # each is split in half, giving CC1/2 within seen crystals and within
        # unseen crystals on equal footing.
        n_match = min(ref_train.numel(), self.heldout_crystals.numel())
        n_match -= n_match % 2
        gq = torch.Generator().manual_seed(self.cfg.seed + 991)
        tr_sub = ref_train[
            torch.randperm(ref_train.numel(), generator=gq)[:n_match]
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

        # FULL-dataset split-half, for an absolute CC1/2 comparable to other
        # programs' full-data split-half figures.
        # The matched train/val sets above answer a different question and are
        # much shallower: each half carries an order of magnitude fewer
        # observations per reflection than the full merge, and
        # merge statistics depend strongly on multiplicity -- so those numbers
        # are NOT comparable to the literature and only their train-vs-val
        # difference is meaningful.
        allc = ref_pool[torch.randperm(ref_pool.numel(), generator=gq)]
        half = allc.numel() // 2
        self.merge_sets["full_A"] = allc[:half]
        self.merge_sets["full_B"] = allc[half:]
        # A second, independent split of the same data: comparing the two gives
        # the scatter of the statistic itself, so we know which differences
        # between runs are real.
        allc2 = ref_pool[torch.randperm(ref_pool.numel(), generator=gq)]
        self.merge_sets["rep_A"] = allc2[:half]
        self.merge_sets["rep_B"] = allc2[half:]

        groups = [
            {"params": decay, "weight_decay": self.cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        if mixture:
            groups.append({"params": mixture, "weight_decay": 0.0,
                           "lr": self.cfg.lr * self.cfg.mixture_lr_mult,
                           "lr_mult": float(self.cfg.mixture_lr_mult)})
        if noise:
            groups.append({"params": noise, "weight_decay": 0.0,
                           "lr": self.cfg.lr * self.cfg.noise_lr_mult,
                           "lr_mult": float(self.cfg.noise_lr_mult)})
        if profile:
            groups.append({"params": profile, "weight_decay": 0.0,
                           "lr": self.cfg.lr * self.cfg.pseudo_translation_lr_mult,
                           "lr_mult": float(self.cfg.pseudo_translation_lr_mult)})
        self.opt = torch.optim.AdamW(groups, lr=self.cfg.lr)
        self.n_decayed = sum(p.numel() for p in decay)
        self.n_undecayed = sum(p.numel() for p in no_decay)
        self._base_lr = self.cfg.lr
        self.logger = MetricLogger(self.cfg.outdir)
        self.step_i = 0
        self.best_heldout = float("inf")
        self.best_step = -1
        self._best_state: dict | None = None
        self._best_obj_state: dict | None = None

        #: Free-form provenance a caller (the CLI) stamps on the checkpoint:
        #: d_min, spacegroup, gain, integration radii.  Not consulted by
        #: training -- it exists so a later merge rebuilds the same inputs
        #: instead of re-deriving them from flags that may have drifted.
        self.data_recipe: dict = {}
        #: Names of the crystal splits, in split-index order (empty = one
        #: unnamed split).  Travels to the checkpoint and names the outputs.
        self.split_names: list[str] = []

        # Resolution shell (equal observation count, low d first), for the
        # calibration breakdown.  Every observation of a reflection shares its
        # d, so the shells are cut over reflections weighted by multiplicity and
        # kept per reflection; a batch looks its observations up.
        bins = self.cfg.shell_bins
        if self._store is not None:
            counts = prep.obs_counts.double()
        else:
            counts = torch.bincount(prep.hkl_id, minlength=prep.n_valid).double()
        order = prep.d_spacing.to(counts.device).argsort()
        below = counts[order].cumsum(0) - counts[order]
        shell_h = torch.empty(prep.n_valid, dtype=torch.int16, device=counts.device)
        shell_h[order] = (below / max(int(prep.n_obs_total), 1) * bins).floor().clamp(
            max=bins - 1).to(torch.int16)
        #: per reflection; an observation's shell is that of its reflection
        self.shell_of_hkl = shell_h

        # A Wilson surface from the start, on a basis with no symmetry beyond
        # the identity; a caller that knows the space group installs its own
        # (set_reciprocal_basis), which replaces this one.
        self.reciprocal_basis = None
        if prep.median_cell is not None:
            from torchsx.merging.priors.reciprocal_basis import ReciprocalBasis
            self.set_reciprocal_basis(ReciprocalBasis.from_hkl(
                prep.hkl_valid.long(), prep.median_cell, None, degree=self.cfg.wilson_degree))

    # ── condition-indexed state ────────────────────────────────────────

    @property
    def target(self):
        """Merge target of the condition currently in flight."""
        return self.targets[self._cond]

    # ── schedule ───────────────────────────────────────────────────────

    def _lam_spread(self) -> float:
        if self.cfg.lam_spread <= 0:
            return 0.0
        hl = self.cfg.lam_spread_halflife_steps
        if hl <= 0:
            # No ramp: hold the penalty constant.  A decaying penalty makes the
            # optimised loss non-comparable across a run for a reason that has
            # nothing to do with the model -- and a strong decaying penalty can
            # drive the per-shell CC1/2 non-monotone in resolution, which is
            # not physically possible.  A weak constant penalty is a prior, not a
            # schedule, and is what the shrinkage is meant to be.
            return self.cfg.lam_spread
        return self.cfg.lam_spread * (0.5 ** (self.step_i / hl))

    # ── one step ───────────────────────────────────────────────────────

    def _batch(self, rows: Tensor):
        """``(meta, n_ph, bg, sigma)`` for one batch, on the compute device.

        With ``stream_batches`` the per-observation state lives on the host and
        only the gathered rows cross the bus; otherwise this is the plain
        indexing it replaces, so the two paths give identical tensors.
        """
        if self._store is not None:
            meta = self._store.batch(rows, self.device)
            if self.n_conditions > 1 and rows.numel():
                self._cond = int(self.condition[self._store.crystal_of_row(int(rows[0]))])
            n_ph = (meta.I_obs + meta.bg).clamp(min=0)
            sigma = (meta.sigma if meta.sigma is not None
                     else (n_ph + meta.bg.clamp(min=0) + 1.0).sqrt())
            return meta, n_ph, meta.bg, sigma
        if self.n_conditions > 1 and rows.numel():
            # The batch is condition-pure (CrystalIndex guarantees it), so the
            # first row names the condition for the whole batch.  This is the
            # one choke point every forward, evaluate, merge and diagnostic
            # path passes through, which is what keeps ``self.target`` right.
            cid = self.prep.obs_info.crystal_id
            self._cond = int(self.condition[int(cid[rows[0].to(cid.device)])])
        if not self.cfg.stream_batches:
            return (self.prep.obs_info.gather(rows), self.n_ph[rows],
                    self.prep.bg[rows], self.sigma[rows])
        r = rows.to(self.prep.bg.device)
        return (
            self.prep.obs_info.gather(r).to(self.device),
            self.n_ph[r].to(self.device, non_blocking=True),
            self.prep.bg[r].to(self.device, non_blocking=True),
            self.sigma[r].to(self.device, non_blocking=True),
        )

    def prefetch(self, rows: Tensor) -> None:
        """Start reading the observations of a coming batch (a no-op in memory)."""
        if self._store is not None:
            self._store.prefetch(rows)

    def _batches(self, max_obs: int, **kwargs):
        """``CrystalIndex.batches`` with the next batch read while this one is used."""
        it = self.index.batches(max_obs, **kwargs)
        if self._store is None:
            yield from it
            return
        nxt = next(it, None)
        while nxt is not None:
            cur, nxt = nxt, next(it, None)
            self.prefetch(cur[1])
            if nxt is not None:
                self.prefetch(nxt[1])
            yield cur

    def _forward(self, rows: Tensor, *, grad: bool):
        meta, n_ph, bg, sigma = self._batch(rows)

        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            # ONE forward.  The target needs a *detached* prediction, and this
            # prediction detached is exactly that -- same weights, same inputs,
            # same numbers.  A separate no_grad pass first would buy nothing
            # and cost a full forward: with gradient checkpointing the grad
            # pass is ~3 forward-equivalents, so that pass would be ~25 % of
            # the step.
            pred = self.scale.prime(meta)
            tgt, log_Z = self._posterior_step(pred, n_ph, bg, sigma, meta)
            # The marginal likelihood per reflection.  I0 appears nowhere.
            nll = -log_Z[self.target.last_seen].sum()
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
        # calibration change when nothing about the residuals moved.  At a
        # typical background the term is as large as sigma^2 itself, so it is
        # not a refinement.
        bvf = float(getattr(getattr(self.objective, "likelihood", None),
                            "bg_var_factor", 0.0) or 0.0)
        var_count = sigma**2 + bvf * bg.clamp(min=0.0)
        noise = (self.objective.noise_terms(meta)
                 if hasattr(self.objective, "noise_terms") else None)
        if noise is not None:
            # the learned detector variance A + B (counting)
            nA, nB = (t.detach() for t in noise)
            var_count = nA + nB * var_count
        var = (var_count + var_scatter).clamp(min=1e-12)
        # Same argument for the mean: under a bg_mean_offset the model claims
        # E[n] = K*I0 + bg + offset, so the residual this metric standardises
        # has to subtract the same background the loss used.  (Small at a
        # typical background -- it keeps the metric honest, it is not a fix.)
        resid = (n_ph - bg - _bg_mean_offset(self.objective)) - K * I0
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
        """Data NLL on this batch in eval mode and with no penalty.

        The optimised loss moves for reasons that have nothing to do with the
        model getting better while ``lam_spread`` anneals, so the training
        curve is not comparable across a run.  This is the same batch scored
        under conditions that never change.
        """
        self.scale.eval()
        try:
            _, _, tgt, nll, *_ = self._forward(rows, grad=False)
        finally:
            self.scale.train()
        self.scale.clear()
        return float(nll.item()) / max(int(tgt.valid.sum()), 1)

    def step(self, rows: Tensor) -> dict:
        self.scale.train()
        if self.cfg.warmup_steps > 0 or self.cfg.lr_schedule != "constant":
            scale_lr = lr_factor(self.step_i, warmup=self.cfg.warmup_steps, total=self.cfg.total_steps,
                                 schedule=self.cfg.lr_schedule, min_frac=self.cfg.min_lr_frac)
            for grp in self.opt.param_groups:
                grp["lr"] = self._base_lr * scale_lr * grp.get("lr_mult", 1.0)
        t0 = time.time()
        io0 = self._store.store.wait_s if self._store is not None else 0.0
        meta, pred, tgt, nll, n_ph, bg, sigma = self._forward(rows, grad=True)
        io_wait = (self._store.store.wait_s - io0) if self._store is not None else 0.0

        n_used = int(tgt.valid.sum())
        nll_data = float(nll.detach().item()) / max(n_used, 1)   # no penalty
        loss = nll / max(n_used, 1)
        lam_s = self._lam_spread()
        if lam_s > 0:
            loss = loss + lam_s * pred.log_w.mean()
        shape_rec = {}
        if self.cfg.lam_shape_prior > 0:
            mix = pred.mix if pred.mix is not None else pack_single(pred.log_alpha, pred.beta, pred.log_alpha_hi)
            kl, sd_bar, out_bar = shape_prior_kl(mix, pred.log_e_s, n_sample=self.cfg.shape_prior_sample)
            loss = loss + self.cfg.lam_shape_prior * kl
            shape_rec = {"shape_prior_kl": float(kl.detach()), "shape_prior_sd_bar": sd_bar,
                         "shape_prior_off_grid": out_bar}
        self.opt.zero_grad(set_to_none=True)
        if self.cfg.posterior_prior == "wilson" and self.targets[0].wilson_coef_param is None:
            raise RuntimeError("the Wilson surface is trained with the network and needs a "
                               "reciprocal basis: prep.median_cell or set_reciprocal_basis()")
        loss.backward()
        gn = grad_norms(self.scale)
        for t in self.targets:
            p = getattr(t, "wilson_coef_param", None)
            if p is not None and p.grad is not None and not bool(torch.isfinite(p.grad).all()):
                gn["n_nonfinite_grads"] = gn.get("n_nonfinite_grads", 0) + 1
        # A single non-finite batch must not end the run.  Applying an update
        # from NaN gradients makes every later batch NaN -- the whole model is
        # poisoned, n_used goes to 0 and the remaining epochs compute nothing.
        # One batch's worth of gradient is not worth that, so drop it and carry
        # on from the last good weights.  Counted, never silent.
        skipped = bool(gn.get("n_nonfinite_grads", 0)) or not bool(
            torch.isfinite(loss.detach()))
        if skipped:
            self.n_skipped_steps += 1
            self.opt.zero_grad(set_to_none=True)
            total_norm = torch.tensor(float("nan"))
        else:
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.scale.parameters(), self.cfg.grad_clip
            )
            self.opt.step()
            dn = getattr(self.objective, "detector_noise", None)
            if dn is not None:
                dn.project()          # the noise terms stay inside their bounds
            for t in self.targets:
                t.sync_evidence_prior()
        gn["gradnorm_total"] = float(total_norm)          # pre-clip global norm
        gn["grad_was_clipped"] = bool(float(total_norm) > self.cfg.grad_clip)
        gn["step_skipped"] = skipped
        self.scale.clear()

        rec = {
            "step": self.step_i,
            "phase": "train",
            # What the optimiser actually minimised (penalty included).
            "loss_total": float(loss.item()),
            # ``nll_per_obs`` is the DATA term in both train and held-out
            # records, the penalty-free objective ``evaluate()`` writes under
            # the same key.  ``loss_total`` includes the spread penalty
            # (negative, since log w < 0), which can fall while the data term
            # rises and would otherwise look like training progress.
            "nll_per_obs": nll_data,
            "nll_data": nll_data,
            # Under ``loss="evidence"`` the numerator is a sum over
            # REFLECTIONS (-sum_h log Z_h) while the denominator counts
            # OBSERVATIONS, so "per obs" is only a scale convention -- and the
            # two are not in a fixed ratio.  log Z_h carries one O(1) term per
            # reflection that does NOT grow with multiplicity, so the per-obs
            # number picks up an offset ~ n_hkl_used / n_obs.  Train batches
            # and the held-out set differ in observation count over nearly the
            # same reflections, so that offset differs
            # between them: a train-vs-heldout GAP under this loss is partly an
            # artefact of the normaliser, not generalisation.  Read each curve's
            # trend, and use ``evidence_per_hkl`` when comparing sets.
            "evidence_per_hkl": (
                nll_data * max(n_used, 1) / max(getattr(self, "last_n_hkl", 0), 1)
            ),
            "n_hkl_used": getattr(self, "last_n_hkl", 0),
            "n_obs": int(rows.numel()),
            "n_crystals": int(meta.crystal_id.unique().numel()),
            "n_used": n_used,
            "n_floored": tgt.n_floored,
            "lam_spread": lam_s,
            **shape_rec,
            "lr": float(self.opt.param_groups[0]["lr"]),
            "sec": time.time() - t0,
            # time this update waited for its observations to be read from disk
            "io_wait": io_wait,
            **gn,
        }
        if self.step_i % self.cfg.log_every == 0:
            # No-penalty score on this same batch: the only training number
            # that is comparable across the whole run.
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
            # which is invisible in the total.
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
            out["grid_mass_p01"] = float(_diagnostic_quantile(gm, 0.01).item())
        _m_, _n_, _b_, _s_ = self._batch(rows)
        ios = (_n_ - _b_) / _s_
        shell = self.shell_of_hkl.to(meta.hkl_idx.device)[meta.hkl_idx]
        out.update(calibration_metrics(
            z, shell_id=shell.to(z.device).long(),
            n_shells=self.cfg.shell_bins, i_over_sigma=ios,
        ))
        # Is the width head actually using its per-observation freedom?  var_z
        # varies strongly across I/sigma deciles, which a per-observation
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
        if pred.mix is not None:
            from torchsx.merging.scatter.mixture import split_mix
            from torchsx.merging.scatter.mixture import N_BLOCKS
            K = pred.mix.shape[1] // N_BLOCKS
            log_pi, offset, la_lo, la_hi, lb = split_mix(pred.mix.detach(), K)
            out["mix_weight_median"] = [float(v) for v in log_pi.exp().median(dim=0).values]
            out["mix_offset_median"] = [float(v) for v in offset.median(dim=0).values]
            out["mix_alpha_lo_median"] = [float(v) for v in la_lo.exp().median(dim=0).values]
            out["mix_alpha_hi_median"] = [float(v) for v in la_hi.exp().median(dim=0).values]
            out["mix_beta_median"] = [float(v) for v in lb.exp().median(dim=0).values]
            out["mix_log_e_s_median"] = float(pred.log_e_s.median().item())
            out["mix_weight_median_by_isigma_decile"] = [
                [float(v) for v in log_pi[dec == d].exp().median(dim=0).values]
                if int((dec == d).sum()) > 8 else [float("nan")] * K
                for d in range(10)
            ]
        out.update(scale_metrics(
            pred.log_K, meta.crystal_id, meta.n_crystals,
            features={
                "s_sq": meta.s_sq,
                "ewald_offset": meta.ewald_offset,
                "d_rank": shell.float().to(pred.log_K.device),
            },
        ))
        out["spearman_log_w_ewald"] = (
            float(_spearman_safe(pred.log_w, meta.ewald_offset.abs()))
            if meta.ewald_offset is not None else float("nan")
        )
        if self.is_light is not None:
            if self.n_conditions == 1:
                out["condition_auc"] = condition_auc(
                    pred.log_K, self.is_light[meta.crystal_id]
                )
            else:
                lk = pred.log_K.detach()
                take = torch.randperm(lk.numel(), device=lk.device)[:4096]
                self._auc_pool[self._cond] = lk[take]
                if all(v is not None for v in self._auc_pool):
                    labels = torch.cat([
                        torch.full((v.numel(),), c != 0, dtype=torch.bool, device=lk.device)
                        for c, v in enumerate(self._auc_pool)
                    ])
                    out["condition_auc"] = condition_auc(torch.cat(self._auc_pool), labels)
        return out

    @torch.no_grad()
    def evaluate(self, crystals: Tensor, *, phase: str = "heldout") -> dict:
        """Score crystals the network never trained on.

        In-batch loss falling while this does not is co-adaptation through the
        merge rather than learning.
        """
        self.scale.eval()
        tot, n_tot, n_hkl, recs = 0.0, 0, 0, []
        for sel, rows in self._batches(
            self.cfg.max_obs_per_batch, crystals=crystals, shuffle=False
        ):
            meta, pred, tgt, nll, *_ = self._forward(rows, grad=False)
            n_used = int(tgt.valid.sum())
            tot += float(nll.item())
            n_tot += n_used
            n_hkl += getattr(self, "last_n_hkl", 0)
            recs.append(
                calibration_metrics(torch.where(tgt.valid, tgt.z, torch.nan))
            )
            self.scale.clear()
        rec = {
            "step": self.step_i, "phase": phase,
            "nll_per_obs": tot / max(n_tot, 1),
            # See step(): under loss="evidence" this is a reflection sum over an
            # observation count, so it is comparable along its own curve but not
            # against the training curve, which has a different multiplicity.
            "evidence_per_hkl": tot / max(n_hkl, 1),
            "n_hkl_used": n_hkl,
            "n_obs": n_tot,
            "var_z": _mean_of([r.get("var_z") for r in recs]),
            "frac_gt3": _mean_of([r.get("frac_gt3") for r in recs]),
        }
        dn = getattr(self.objective, "detector_noise", None)
        if dn is not None:
            # the fitted variance terms per panel
            for t, v in dn.values().items():
                rec[f"noise_{t}"] = v
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
            # the objective's own scalars (miss fraction) belong to the same snapshot
            self._best_obj_state = {
                k: v.detach().clone() for k, v in self.objective.state_dict().items()
            }
            self._best_wilson = [t.wilson_coef.detach().clone() for t in self.targets]
        rec["best_heldout"] = self.best_heldout
        rec.update(self.wilson_fit_info())
        self.logger.log(rec)
        return rec

    def _detector_noise_record(self) -> dict | None:
        """Configuration and fitted (best-state) values of the detector noise."""
        dn = getattr(self.objective, "detector_noise", None)
        if dn is None:
            return None
        best = getattr(self, "_best_obj_state", None) or {}
        state = {k: best.get(f"detector_noise.{k}", v).detach().cpu().tolist()
                 for k, v in dn.state_dict().items()}
        return {"config": dn.config(), "state": state}

    def save_best(self, path=None) -> "Path | None":
        """Persist the best held-out parameters so a finished run can be
        re-analysed (merge statistics, a different split, per-shell
        breakdowns) without retraining it.
        """
        # With no held-out set (a two-parameter global fit, say) nothing is ever
        # scored and there is no "best" -- write the current state rather than
        # letting a finished run produce no checkpoint at all.
        state = self._best_state if self._best_state is not None else {
            k: v.detach().cpu().clone() for k, v in self.scale.state_dict().items()
        }
        path = Path(path) if path is not None else Path(self.cfg.outdir) / "best.pt"
        wilson = self._best_wilson or [t.wilson_coef.detach().clone() for t in self.targets]
        torch.save(
            {
                "state_dict": state,
                # so a later merge builds the SAME inner integral as training
                "quadrature": {
                    # The "+bgmean" suffix is load-bearing: it is the only thing
                    # that makes a build without bg_mean_offset REFUSE this
                    # checkpoint (see _pipeline's kind check) instead of merging
                    # it at the uncorrected background convention.
                    "kind": type(getattr(self.objective, "quadrature", None)).__name__
                            + ("+bgmean" if _bg_mean_offset(self.objective) else ""),
                    "n_points": getattr(getattr(self.objective, "quadrature", None),
                                        "n_points", None),
                },
                # so a later merge builds the SAME forward, not the default one
                "encoder": {
                    k: getattr(self.scale.encoder, k, None)
                    for k in ("context_folds", "obs_context_layers",
                              "obs_self_layers", "d_model", "n_layers",
                              "n_crystal_features", "peer_attention_layers",
                              "peer_heads", "peer_dim")
                },
                # everything a merge must not have to guess: the likelihood it
                # was fitted under, the data recipe, and the scale geometry.  A
                # silent mismatch here changes the merge without any error.
                "likelihood": {
                    "kind": type(getattr(self.objective, "likelihood", None)).__name__
                            + ("+bgmean" if _bg_mean_offset(self.objective) else ""),
                    "bg_var_factor": float(getattr(
                        getattr(self.objective, "likelihood", None),
                        "bg_var_factor", 0.0) or 0.0),
                    "bg_mean_offset": _bg_mean_offset(self.objective),
                },
                "data": dict(self.data_recipe or {}),
                "scale": {
                    "d_model": getattr(self.scale.encoder, "d_model", None),
                    "n_layers": getattr(self.scale.encoder, "n_layers", None),
                    "include_hkl": bool(getattr(self.scale.features, "include_hkl", False)),
                    "self_features": list(getattr(self.scale.features, "self_features", ())),
                    "context_features": list(getattr(self.scale.features, "context_features", ())),
                    "crystal_features": list(getattr(self.scale.features, "crystal_features", ())),
                    "predict_width": bool(getattr(self.scale.heads, "predict_width", True)),
                    "predict_shape": bool(getattr(self.scale.heads, "predict_shape", True)),
                    "anchor_log_k": bool(getattr(self.scale, "anchor_log_k", True)),
                    "anchor_stat": getattr(self.scale, "anchor_stat", "median"),
                    "pseudo_translation": (
                        None if getattr(self.scale.features, "phase_broad", None) is None
                        else {"width": self.scale.features.phase_width.definition(),
                              "broad": self.scale.features.phase_broad.definition()}),
                },
                "splits": list(self.split_names or []),
                # scatter family and miss component, so a merge rebuilds the
                # SAME likelihood the network was fitted under
                "scatter": {
                    "predict_skew": bool(getattr(self.scale.heads, "predict_skew", False)),
                    "family": getattr(self.scale.heads, "scatter_family", "gengauss"),
                    "n_components": int(getattr(self.scale.heads, "n_components", 1)),
                    "component_shape": (self.scale.heads.mixture.shape
                                        if getattr(self.scale.heads, "scatter_family", "gengauss") == "mixture" else None),
                    "miss_logit": (None if getattr(self.objective, "miss_logit", None) is None
                                   else float((getattr(self, "_best_obj_state", None) or {}).get(
                                       "miss_logit", self.objective.miss_logit.detach()))),
                    "outlier_logit": (None if getattr(self.objective, "outlier_logit", None) is None
                                      else float((getattr(self, "_best_obj_state", None) or {}).get(
                                          "outlier_logit", self.objective.outlier_logit.detach()))),
                },
                "detector_noise": self._detector_noise_record(),
                "best_step": self.best_step,
                "best_heldout": self.best_heldout,
                "n_crystals": self.index.n_crystals,
                "n_hkl": self.prep.n_valid,
                "n_conditions": self.n_conditions,
                # the Wilson surface as a function of hkl: the basis definition
                # and each state's trained coefficients, so a merge of any
                # reflection set evaluates the same surface
                "wilson_surface": (None if getattr(self, "reciprocal_basis", None) is None else {
                    "basis": self.reciprocal_basis.definition(),
                    "coef": [c.cpu() for c in wilson],
                }),
                # outer-grid geometry, so a later merge integrates on the SAME grid
                "posterior_grid": {
                    "k_coarse": self.cfg.posterior_k_coarse, "m_fine": self.cfg.posterior_m_fine,
                    "n_sigma": self.cfg.posterior_n_sigma, "max_halfwidth": self.cfg.posterior_max_halfwidth,
                    "prior": self.cfg.posterior_prior,
                    "wilson_degree": self.cfg.wilson_degree,
                },
                "timepoints": (
                    None if self.prep.obs_info.unique_timepoints is None
                    else self.prep.obs_info.unique_timepoints.detach().cpu().tolist()
                ),
                "heldout_condition": self.cfg.heldout_condition,
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
        if getattr(self, "_best_obj_state", None):
            self.objective.load_state_dict(self._best_obj_state)
        if self._best_wilson is not None:
            for t, c in zip(self.targets, self._best_wilson):
                t.set_wilson_coef(c)
                if t.wilson_coef_param is not None:
                    t.wilson_coef_param.data.copy_(c)
        return True

    @torch.no_grad()
    def merge(
        self,
        *,
        crystals: Tensor | None = None,
        condition: int | None = None,
        estimator: str = "posterior",
        return_likelihood: bool = False,
    ) -> dict:
        """Merge a crystal set and return everything an MTZ needs.

        The single merge implementation the CLIs use.  ``_merge_set`` keeps its
        own shorter path: it feeds the training-time CC1/2 and R_split, which
        stay on one fixed code path.

        ``estimator``:

        ``posterior``  Wilson-posterior mean of ``sqrt(I0)`` over the very
                       weights the merge computed, or the flat-in-intensity
                       mean when ``posterior_prior='none'``. The selected prior
                       governs I, F, uncertainties and edge diagnostics alike.
        ``flat``       the same moment under the measure alone (no Wilson
                       prior).
        ``ml``         ``sqrt(I0)`` at the likelihood maximum, prior-free;
                       sigma from the flat posterior in both cases.
        ``pooled``     ``sum(n - bg) / sum(K)``, count-weighted, shape-free.
        ``mean``       equal-weight mean of ``(n - bg) / K``, partialator's
                       weighting.
        ``wls``        inverse-variance mean of ``(n - bg) / K`` with the
                       variance the trained scale model predicts for each
                       observation: counting variance plus the multiplicative
                       scatter ``(K I0)^2 expm1(w^2)`` from the width head,
                       the weights ``_merge_set`` uses for the split-half
                       metrics.  The width is fitted by the shape likelihood,
                       so this readout inherits that likelihood's location
                       bias; it is kept as a diagnostic, not a product.
                       ``pooled``/``mean``/``wls`` return intensities
                       (``I``, ``I_sd``, ``n_meas``) rather than F: they are
                       moment estimators and belong on the same French-Wilson
                       path partialator's own output takes.

        Returns a dict with ``F``/``F_sd`` (or ``I``/``I_sd``/``n_meas``),
        ``I0``, ``seen`` and the grid diagnostics ``edge_mass``, ``loc_sd``.
        """
        if estimator not in ("posterior", "flat", "ml", "pooled", "mean", "wls"):
            raise ValueError(f"unknown estimator {estimator!r}")
        if condition is not None:
            if not (0 <= condition < self.n_conditions):
                raise ValueError(
                    f"condition {condition} outside 0..{self.n_conditions - 1}"
                )
            self._cond = int(condition)
            in_cond = (self.condition == condition).nonzero(as_tuple=True)[0]
            crystals = (in_cond if crystals is None
                        else crystals[torch.isin(crystals.cpu(), in_cond)])
        elif crystals is None:
            crystals = torch.arange(self.index.n_crystals)
        if self.n_conditions > 1 and condition is None:
            cb = self.condition[crystals.cpu()].unique()
            if cb.numel() != 1:
                raise ValueError(
                    f"merge: crystals span conditions {cb.tolist()}; "
                    "merge one condition at a time"
                )
            self._cond = int(cb[0])
        self.scale.eval()
        n_hkl = self.prep.n_valid
        dev = self.device
        tgt = self.target
        batches = lambda: self._batches(
            self.cfg.max_obs_per_batch, crystals=crystals, shuffle=False
        )

        if estimator in ("pooled", "mean", "wls"):
            # Moment estimators on the trained K: no scatter shape, no prior,
            # no mean/median correction -- a weighted mean of (n - bg)/K
            # estimates E[I p] whatever the partiality distribution is.
            I0_ref = tgt.I_start.to(dev)
            floor = float(getattr(tgt, "floor", 1e-6) or 1e-6)
            bvf = float(getattr(self.objective.likelihood, "bg_var_factor", 0.0) or 0.0)
            S = {k: torch.zeros(n_hkl, device=dev, dtype=torch.float64)
                 for k in ("cnt", "r", "r2", "sig", "K", "var", "sig2", "sigK", "K2",
                           "num", "den", "A", "B", "C")}
            for _, rows in batches():
                meta, n_ph, bg, _ = self._batch(rows.to(dev))
                pred = self.scale.prime(meta)
                K = pred.log_K.exp().double()
                sig = (n_ph - bg).double()
                h = meta.hkl_idx
                ratio = sig / K
                var = ((K * I0_ref[h].double()).clamp(min=0)
                       + bg.double() * (1.0 + bvf))
                # per-observation variance of (n - bg) under the trained model:
                # counting term plus the multiplicative log-scatter of width w.
                # The counting term is the model's (K I0_ref + bg), not the
                # observation's own sigma: a weight that depends on the observed
                # count biases the weighted mean low, by an amount that depends
                # on multiplicity.  Floors: I0_ref at the target's floor (the
                # unity merge of a weak reflection can be zero or negative) and the
                # counting term at one count, so no single zero-background
                # observation can dominate a reflection.
                w2 = (pred.log_w.exp().double() ** 2).clamp(max=20.0)
                mu = K * I0_ref[h].double().clamp(min=floor)
                V = ((mu + bg.double() * (1.0 + bvf)).clamp(min=1.0)
                     + mu * mu * torch.expm1(w2))
                g = K / V                     # weight on (n - bg) for the WLS mean
                for k, v in (("cnt", torch.ones_like(sig)), ("r", ratio),
                             ("r2", ratio * ratio), ("sig", sig), ("K", K),
                             ("var", var), ("sig2", sig * sig),
                             ("sigK", sig * K), ("K2", K * K),
                             ("num", g * sig), ("den", g * K),
                             ("A", g * g * sig * sig), ("B", g * g * sig * K),
                             ("C", g * g * K * K)):
                    S[k].index_add_(0, h, v)
                self.scale.clear()
            cnt = S["cnt"].clamp(min=1)
            if estimator == "wls":
                den = S["den"].clamp(min=1e-300)
                I = S["num"] / den
                # model variance 1 / sum(K^2 / V), and the sandwich variance of
                # the same weighted mean; report the larger (see ``pooled``).
                var_model = 1.0 / den
                ss = (S["A"] - 2.0 * I * S["B"] + I * I * S["C"]).clamp(min=0)
                bessel = cnt / (cnt - 1).clamp(min=1)
                I_sd = torch.maximum(var_model, ss * bessel / (den * den)).sqrt()
            elif estimator == "mean":
                I = S["r"] / cnt
                I_sd = ((S["r2"] / cnt - I * I).clamp(min=0) / cnt).sqrt()
            else:
                sumK = S["K"].clamp(min=1e-12)
                I = S["sig"] / sumK
                # Counting variance alone understates the error of a merge whose
                # observations also scatter crystal-to-crystal (partiality,
                # residual scale error).  Add the count-weighted empirical
                # (sandwich) variance of the ratios, sum K^2 (r - I)^2 / (sum K)^2
                # with an n/(n-1) correction, and report the larger of the two:
                # the empirical term is noisy at low multiplicity, the counting
                # term is a floor.
                var_pois = S["var"] / (sumK * sumK)
                ss = (S["sig2"] - 2.0 * I * S["sigK"] + I * I * S["K2"]).clamp(min=0)
                bessel = cnt / (cnt - 1).clamp(min=1)
                var_emp = ss * bessel / (sumK * sumK)
                I_sd = torch.maximum(var_pois, var_emp).sqrt()
            log_c = self.scale.readout_log_factor(torch.arange(n_hkl, device=dev))
            if log_c is not None:
                I, I_sd = I * log_c.exp().to(I), I_sd * log_c.exp().to(I_sd)
            return {
                "estimator": estimator,
                "I": I.float(), "I_sd": I_sd.float(),
                "n_meas": S["cnt"].long(),
                "seen": S["cnt"] >= 2,
                "n_crystals": int(crystals.numel()),
                "condition": self._cond,
            }

        # Locate the peak on THIS crystal set, then accumulate the same coarse
        # curve the loss uses -- additive over observations, hence over batches.
        n_bad_loc = 0
        # Locating rounds, each a full pass: anchor on the previous
        # estimate and shrink the scan window.
        #
        # One round is not enough at high multiplicity.  The scan is
        # 9 nodes over +-scan_halfwidth (2-nat spacing by default) and the
        # peak is read off a parabola through three of them, but the
        # accumulated log-likelihood is ``A v - B e^v`` -- exponential above
        # the peak, linear below -- so the fitted vertex carries a bias of
        # about half a nat that does NOT shrink with multiplicity (both the
        # y-differences and the curvature scale with A, so the vertex does
        # not move).  Meanwhile the grid built around it is n_sigma * sd
        # wide with sd ~ 1/sqrt(A).  The two cross between 1e3 and 1e4
        # counts per reflection: below that the grid still covers the peak,
        # above it every reflection puts all its mass on the grid edge.
        # The vertex bias is O(h^2), so each round of a shrinking
        # window squares it away, and the grid covers the peak at every
        # multiplicity.
        hw0 = tgt.scan_halfwidth
        centre = tgt.start_centre()
        loc_sd = None
        try:
            for hw in self.cfg.locate_windows or (hw0,):
                tgt.scan_halfwidth = float(hw)
                scan = tgt.scan_grid(centre)
                acc_s = torch.zeros(n_hkl, scan.shape[1], device=dev)
                for _, rows in batches():
                    meta, n_ph, bg, _ = self._batch(rows.to(dev))
                    pred = self.scale.prime(meta)
                    c = tgt.accumulate(
                        self.objective, self.scale.scatter, self.params, pred,
                        n_ph, bg, meta.hkl_idx, meta, scan,
                    )
                    bad = ~torch.isfinite(c)
                    if bool(bad.any()):
                        n_bad_loc += int(bad.any(1).sum())
                        c = c.masked_fill(bad, 0.0)
                    acc_s = acc_s + c
                    self.scale.clear()
                centre, sd_r = tgt.peak_from_scan(acc_s, centre)
                if loc_sd is None:
                    # WIDTH comes from the wide round only.  Its trust /
                    # one-sided / sd_max logic is scaled to the wide scan:
                    # at a 0.125-nat window the returned sd is a local
                    # curvature, which for a weak one-sided reflection is
                    # tiny and yields a grid far too narrow: refining the
                    # width too puts such reflections on the grid edge.
                    # The wide round's width already tracks 1/sqrt(A); only
                    # the CENTRE carries the parabola bias, so only the
                    # centre is refined.
                    loc_sd = sd_r
        finally:
            tgt.scan_halfwidth = hw0
        tgt._loc = (centre, loc_sd)
        v_nodes = centre[:, None] + loc_sd[:, None] * tgt.u_coarse[None, :]

        # A non-finite contribution is dropped where it is non-finite, not
        # summed in.  ``accumulate`` is additive over observations, so a NaN
        # from one batch otherwise propagates to every reflection that batch
        # touched -- and since ``seen`` is ``acc.abs().sum(1) > 0`` and
        # ``NaN > 0`` is False, the whole split comes back empty.  Losing
        # those reflections' share of one batch is worth far less than the
        # split.
        acc = torch.zeros(n_hkl, v_nodes.shape[1], device=dev)
        n_bad_acc = 0
        for _, rows in batches():
            meta, n_ph, bg, _ = self._batch(rows.to(dev))
            pred = self.scale.prime(meta)
            c = tgt.accumulate(
                self.objective, self.scale.scatter, self.params, pred,
                n_ph, bg, meta.hkl_idx, meta, v_nodes,
            )
            bad = ~torch.isfinite(c)
            if bool(bad.any()):
                n_bad_acc += int(bad.any(1).sum())
                c = c.masked_fill(bad, 0.0)
            acc = acc + c
            self.scale.clear()
        _, I0, I0_sd, w_post, v, edge = tgt.integrate(acc)
        log_c = self.scale.readout_log_factor(torch.arange(n_hkl, device=dev))
        if log_c is not None:
            # Fitted core intensity -> total recorded intensity: the grid, and
            # with it every moment below, moves by log(1 + rho) per reflection.
            log_c = log_c.to(v)
            v = v + log_c[:, None]
            I0, I0_sd = I0 * log_c.exp(), I0_sd * log_c.exp()

        # F as a posterior moment of sqrt(I0) on the same weights, not the
        # plug-in sqrt(E[I]) -- they differ exactly where the relative
        # posterior width is large, i.e. in the weak outer shells.
        rootI = v.exp().clamp(min=0).sqrt()
        F = (w_post * rootI).sum(1)
        F_sd = ((w_post * (rootI - F[:, None]) ** 2).sum(1)).clamp(min=0).sqrt()
        if estimator in ("flat", "ml"):
            # Same nodes, same accumulated curve, weighted by the measure alone
            # (dI0 = I0 dv): the prior-free estimate and a sigma that carries no
            # prior shrinkage.
            S_fine = tgt._interp(acc)
            du = float(2 * tgt.n_sigma / (tgt.m_fine - 1))
            _, sig_h = tgt._loc
            lw = (sig_h * du).log()[:, None].expand_as(v).clone() + v
            lw[:, 0] -= math.log(2.0)
            lw[:, -1] -= math.log(2.0)
            w_flat = torch.softmax(S_fine + lw, dim=1)
            F_flat = (w_flat * rootI).sum(1)
            F_sd = ((w_flat * (rootI - F_flat[:, None]) ** 2).sum(1)).clamp(min=0).sqrt()
            F = (F_flat if estimator == "flat"
                 else rootI.gather(1, S_fine.argmax(1, keepdim=True)).squeeze(1))
        result = {
            "estimator": estimator, "prior": self.cfg.posterior_prior,
            "F": F, "F_sd": F_sd, "I0": I0, "I0_sd": I0_sd,
            "seen": acc.abs().sum(1) > 0,
            "edge_mass": edge,
            "edge_low": w_post[:, 0], "edge_high": w_post[:, -1],
            "loc_sd": loc_sd,
            "scan_halfwidth": tgt.scan_halfwidth,
            "n_nonfinite_rows": n_bad_loc + n_bad_acc,
            "n_crystals": int(crystals.numel()),
            "condition": self._cond,
        }
        if return_likelihood:
            result['log_likelihood'] = tgt._interp(acc).detach().cpu()
            result['log_intensity_grid'] = v.detach().cpu()
            result['wilson_sigma'] = tgt.wilson_sigma.detach().cpu().clone()
            result['centric'] = tgt.is_centric.detach().cpu().clone()
        return result

    @torch.no_grad()
    def _merge_set(self, crystals: Tensor) -> tuple[Tensor, Tensor]:
        """Merge one crystal set with the current network.

        Deliberately **no separate estimator**: whatever the training objective
        merges with is what gets scored.  Otherwise CC1/2 measures a different
        quantity from the one being minimised, and improving the loss need not
        improve the metric.

        That means accumulating the same coarse log-likelihood curve the loss
        uses (additive over observations, hence over batches) and integrating
        once.
        """
        n_hkl = self.prep.n_valid
        if self.n_conditions > 1:
            # A merge is a same-condition quantity; pin the target BEFORE
            # reading it, since the last training batch may have been the
            # other condition.
            cb = self.condition[crystals.cpu()].unique()
            if cb.numel() != 1:
                raise ValueError(
                    f"_merge_set: crystals span conditions {cb.tolist()}; "
                    "merge one condition at a time"
                )
            self._cond = int(cb[0])
        tgt = self.target
        # Locate the peak on THIS crystal subset first.  A closed-form width
        # here would collapse the merge onto the shared grid centre (per-shell
        # CC1/2 a spurious 1).
        start = tgt.start_centre()
        scan = tgt.scan_grid(start)
        acc_s = torch.zeros(n_hkl, scan.shape[1], device=self.device)
        for _, r in self._batches(
            self.cfg.max_obs_per_batch, crystals=crystals, shuffle=False
        ):
            r = r.to(self.device)
            m, _n, _b, _ = self._batch(r)
            pr = self.scale.prime(m)
            acc_s = acc_s + tgt.accumulate(
                self.objective, self.scale.scatter, self.params, pr,
                _n, _b, m.hkl_idx, m, scan,
            )
            self.scale.clear()
        c, sd = tgt.peak_from_scan(acc_s, start)
        v_nodes = c[:, None] + sd[:, None] * tgt.u_coarse[None, :]
        tgt._loc = (c, sd)
        acc = torch.zeros(n_hkl, v_nodes.shape[1], device=self.device)
        for _, rows in self._batches(
            self.cfg.max_obs_per_batch, crystals=crystals, shuffle=False
        ):
            rows = rows.to(self.device)
            meta, _nph, _bg, _sig = self._batch(rows)
            pred = self.scale.prime(meta)
            acc = acc + tgt.accumulate(
                self.objective, self.scale.scatter, self.params, pred,
                _nph, _bg, meta.hkl_idx, meta, v_nodes,
            )
            self.scale.clear()
        _, I0, *_ = tgt.integrate(acc)
        seen = (acc.abs().sum(1) > 0).float()
        # (value, weight) so the caller's ratio form works unchanged
        return I0 * seen, seen

    @torch.no_grad()
    def split_half_metrics(self, *, n_shells: int = 10, log: bool = True) -> dict:
        """CC1/2 and R_split on TRAIN and VALIDATION crystals, matched in count.

        Four merges of equal size: ``train_A/train_B`` and ``val_A/val_B``.
        ``cc_half_train`` is merge quality on crystals the network has trained
        on; ``cc_half_val`` is the same statistic on crystals it has never
        seen, at identical multiplicity.  The gap between them is the
        generalisation of the *scale model* — a per-crystal G has no such gap
        by construction, because it is refit for every crystal.

        Computed on **intensities**, not amplitudes, so the numbers are not
        directly comparable to an amplitude CC1/2.  The baseline that
        counts is the same-data, same-protocol unity merge.
        """
        from torchsx.merging.hkl.stats import (
            MatchedHalves, cc_half, cc_star, r_split, shell_stats,
        )

        was_training = self.scale.training
        self.scale.eval()
        merged = {k: self._merge_set(v) for k, v in self.merge_sets.items()}
        # CC1/2 on INTENSITIES, not amplitudes.  ``I.clamp(min=0).sqrt()``
        # sends every negative merged intensity to exactly 0, and identical
        # values in both halves read to Pearson as perfect agreement.
        # Intensities are the honest choice: negatives are real measurements
        # and must be allowed to disagree.

        def _pair(a: str, b: str) -> tuple[MatchedHalves, Tensor]:
            (na, da), (nb, db) = merged[a], merged[b]
            ok = (da > 0) & (db > 0)
            Ia = torch.where(da > 0, na / da.clamp(min=1e-30), torch.zeros_like(na))
            Ib = torch.where(db > 0, nb / db.clamp(min=1e-30), torch.zeros_like(nb))
            return MatchedHalves(
                F1=Ia[ok].cpu(), F2=Ib[ok].cpu(),
                # per-reflection and host-resident under stream_batches, so
                # index with a host mask rather than the device one
                d_spacing=self.prep.d_spacing[ok.cpu()].cpu(),
                hkl=self.prep.hkl_valid[ok.cpu()].cpu(),
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
        meta, _, _, _ = self._batch(rows)
        rec = {
            "step": self.step_i,
            "phase": "audit",
            # D. a scale that depends on hkl is degenerate with I0 and is the
            # one channel that can manufacture a difference peak.
            "hkl_permutation_frac": hkl_permutation_sensitivity(self.scale, meta),
            # the same test on the predicted width and mixture shape
            "hkl_permutation_frac_shape": hkl_permutation_sensitivity_shape(self.scale, meta),
        }
        frac = max(rec["hkl_permutation_frac"], rec["hkl_permutation_frac_shape"])
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

    def set_reciprocal_basis(self, basis) -> None:
        """Install one smooth reciprocal-space basis on the prior and the gauge pin.

        ``basis`` is a :class:`~torchsx.merging.priors.reciprocal_basis.ReciprocalBasis`
        over ``prep.hkl_valid``.  The scale model (when in ``basis`` anchor mode)
        projects the ensemble-mean component of ``log K`` in its span out.  With
        the Wilson prior, each target's surface ``log Sigma = phi @ c`` starts
        at the Wilson fit to the unity merge and ``c`` joins the optimiser, to
        be trained by the evidence with the network.  Replaces the basis and
        surface installed before, optimiser group included.
        """
        phi = basis.phi.to(self.device)
        if phi.shape[0] != self.prep.n_valid:
            raise ValueError(f"basis rows {phi.shape[0]} != n_valid {self.prep.n_valid}")
        old = {id(t.wilson_coef_param) for t in self.targets if t.wilson_coef_param is not None}
        if old:
            self.opt.param_groups = [g for g in self.opt.param_groups
                                     if not any(id(q) in old for q in g["params"])]
            for q in [q for q in self.opt.state if id(q) in old]:
                del self.opt.state[q]
        self.reciprocal_basis = basis
        for t in self.targets:
            t.set_basis(phi)
        if hasattr(self.scale, "set_gauge_basis"):
            self.scale.set_gauge_basis(phi)
        if self.cfg.posterior_prior == "wilson":
            for t in self.targets:
                t.fit_wilson_surface()
            mult = float(self.cfg.wilson_lr_mult)
            self.opt.add_param_group({"params": [t.enable_evidence_prior() for t in self.targets],
                                      "lr": self.cfg.lr * mult, "lr_mult": mult,
                                      "weight_decay": 0.0, "name": "wilson_surface"})

    def wilson_fit_info(self) -> dict:
        """Diagnostics of the Wilson surface on the first target (empty without a basis)."""
        t = self.targets[0]
        if not getattr(t, "has_basis", False) or getattr(self, "reciprocal_basis", None) is None:
            return {}
        from torchsx.merging.priors.reciprocal_basis import wilson_b_equivalent
        b = self.reciprocal_basis
        coef = t.wilson_coef.detach().cpu()
        info = {"wilson_B": wilson_b_equivalent(b.to(coef.device), coef),
                "wilson_fit_iters": int(t.wilson_fit_iters)}
        for i in range(b.n_aniso):
            info[f"wilson_aniso_{i}"] = float(coef[b.degree + 1 + i])
        return info

    def run(self) -> None:
        for epoch in range(self.cfg.epochs):
            for _, rows in self._batches(
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
