"""Per-observation features for the amortised scale model, with the tiering
that keeps the design honest.

Two tiers, and the split is load-bearing:

``self`` features
    What the scale head sees for the observation it is scaling.  Smooth
    diffraction geometry only.  An observation's own intensity is **forbidden**
    here — with it, ``K_j = I_j / I0_h`` fits perfectly and means nothing.

``context`` features
    What goes into the pooled branch.  May include intensity and σ, because
    :func:`~torchsx.nn.pooling.leave_one_out_mean` removes observation ``j``
    from its own context before it is used.

    The peers are not a fallback for missing geometry — they are a **better
    measurement than ``j``'s own geometry can be**.  Reflection ``j``'s
    excitation error is computed from a refined orientation that is itself in
    error, and it carries no information at all about the pulse spectrum, which
    at SASE jitters shot to shot.  The other reflections *measure the crystal's
    state*: which of them came out strong or weak, given where they nominally
    sat, constrains the orientation error and the spectral content, and those
    are what actually set ``j``'s partiality.  (This dataset also has
    ``--fix-profile-radius=4e7``, so the one geometric partiality parameter is
    constant across every crystal.)

    That makes the context a **regression problem**: fit the crystal's scale
    function from ``(geometry, observed scale)`` pairs over the peers, then
    evaluate it at ``j``.  Both halves of the pair must therefore be present —
    geometry alone says nothing, and raw intensity alone conflates "this
    reflection is intrinsically strong" (I0) with "this reflection was well
    positioned" (partiality).  ``obs_scale`` separates them.

Both guards are enforced in ``__init__``, not left to convention.

Condition blindness
-------------------
The scale model must never see ``delay_bin_idx`` or ``unique_timepoints``.
*Any* degree of freedom that can distinguish light from dark will absorb the
difference signal — that is the one failure mode which would produce a
confident, publishable, wrong answer.  Conditions are handled
partialator-style: one shared network, separate merge targets per condition.
Construction raises if a condition field is requested.

This module works from ``PreparedObs`` / ``ObsInfo`` (the merging pipeline's
representation).
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
from torch import Tensor

from torchsx.merging.observations import CRYSTAL_FEAT_COLUMNS, ObsInfo, PreparedObs

__all__ = [
    "SELF_FEATURES",
    "CONTEXT_ONLY_FEATURES",
    "RANK_CONTEXT_FEATURES",
    "PEER_SCALES",
    "CRYSTAL_FEATURES",
    "FORBIDDEN_IN_SELF",
    "CONDITION_FEATURES",
    "ObsFeatureSpec",
]

#: Geometry the scale head may see for its own observation.
SELF_FEATURES: tuple[str, ...] = (
    "s_sq",            # resolution, (sin θ / λ)²  — a MAGNITUDE
    "ewald_offset",    # signed excitation error (Å⁻¹) → partiality
    # Lab-frame scattering DIRECTION.  Without it an orientation error is not
    # representable at all: a small misorientation ``delta`` shifts the
    # excitation error by ``(delta x q) . s_hat``, which depends on where the
    # reflection sits, not how far out it is.  "Reflections on this side of the
    # pattern run high" needs a direction.  Lab frame, not crystal frame — the
    # crystal-frame direction is just hkl and would be degenerate with I0.
    "qx", "qy", "qz",
    "beam_energy",     # per-shot photon energy — SASE spectrum jitters
    "crystal_res",     # per-crystal diffraction limit (crystal quality)
    "fs", "ss",        # detector position (absorption, geometry)
    "panel",           # detector panel
    "h", "k", "l",     # Miller indices — see the permutation monitor
)

#: Measurement channels, legitimate ONLY in the leave-one-out pooled branch.
CONTEXT_ONLY_FEATURES: tuple[str, ...] = (
    "signed_log_I",    # sign(I)·log1p(|I|) — keeps the 32.5 % negatives
    "log_sigma",
    "i_over_sigma",    # the dimensionless one that actually carries weight
    "log_bg",
    # n'/I0 — the peer's *observed scale*, K.S, with the reflection's intrinsic
    # brightness divided out.  This is the channel that carries partiality:
    # raw intensity only tells the network how bright the crystal is, which is
    # why a raw-intensity context learns G and nothing else (between_frac 0.96,
    # within_frac 0.04).  Linear, so the 32.4 % negative observations survive.
    "obs_scale",
    "obs_scale_over_sigma",
    "log_I0_ref",      # how bright the peer's reflection should be
)

#: The measurement channels with the peer's observed scale given as a RANK:
#: the normal score of the observation's position among all observations of
#: its reflection.  It needs no per-reflection intensity estimate, so nothing
#: has to be merged, refreshed or carried in a checkpoint to form it; it is
#: bounded, keeps the negative observations, and is unchanged by any monotone
#: rescaling of a reflection (its intrinsic brightness drops out).  The ratio
#: channels are replaced; ``log_I0_ref`` stays so the network can tell how
#: informative a peer's rank is.
RANK_CONTEXT_FEATURES: tuple[str, ...] = (
    "signed_log_I",
    "log_sigma",
    "i_over_sigma",
    "log_bg",
    "rank_score",
    "log_I0_ref",
)

#: How a peer's observed scale enters the context: ``"ratio"`` -- n'/I0 against
#: the reference intensity (:data:`CONTEXT_ONLY_FEATURES`); ``"rank"`` -- its
#: within-reflection rank (:data:`RANK_CONTEXT_FEATURES`).
PEER_SCALES: tuple[str, ...] = ("ratio", "rank")

FORBIDDEN_IN_SELF: frozenset[str] = frozenset(CONTEXT_ONLY_FEATURES) | {"rank_score"}

#: Per-CRYSTAL metadata, one row per crystal (``ObsInfo.crystal_feat``).  Read
#: by the encoder's crystal-level input head, once per crystal rather than once
#: per observation.  No intensity, so no exclusion is needed.  ``beam_energy``
#: and ``crystal_res`` also remain per-observation self features for now.
CRYSTAL_FEATURES: tuple[str, ...] = CRYSTAL_FEAT_COLUMNS

#: Never allowed in either tier.  See the module docstring.
CONDITION_FEATURES: frozenset[str] = frozenset(
    {"delay_bin_idx", "unique_timepoints", "timepoint", "delay", "condition"}
)


class ObsFeatureSpec(nn.Module):
    """Builds ``(x_self, x_ctx)`` from an ``ObsInfo`` slice or gather.

    Raw per-observation channels (``I``, ``bg``, ``sigma``) and the Miller table
    are held as buffers and addressed by ``meta.obs_id``, so this works
    identically under a contiguous ``ObsInfo.slice`` and an arbitrary
    crystal-batched ``ObsInfo.gather``.

    Normalisation statistics are computed once over the full dataset at
    construction and stored as buffers, so a batch is never normalised against
    its own statistics (which would leak batch composition into the features).
    """

    def __init__(
        self,
        prep: PreparedObs,
        *,
        self_features: tuple[str, ...] = SELF_FEATURES,
        context_features: tuple[str, ...] | None = None,
        include_hkl: bool = True,
        crystal_features: tuple[str, ...] | None = None,
        peer_scale: str = "ratio",
    ) -> None:
        super().__init__()
        #: Recorded so a checkpoint can say which feature set it was fitted
        #: under; the flag itself is consumed a few lines below.
        self.include_hkl = bool(include_hkl)
        if peer_scale not in PEER_SCALES:
            raise ValueError(f"peer_scale must be one of {PEER_SCALES}, got {peer_scale!r}")
        self.peer_scale = peer_scale

        if context_features is None:
            context_features = tuple(self_features) + (
                CONTEXT_ONLY_FEATURES if peer_scale == "ratio" else RANK_CONTEXT_FEATURES
            )
        if not include_hkl:
            self_features = tuple(f for f in self_features if f not in ("h", "k", "l"))
            context_features = tuple(
                f for f in context_features if f not in ("h", "k", "l")
            )

        leaks = FORBIDDEN_IN_SELF.intersection(self_features)
        if leaks:
            raise ValueError(
                f"{sorted(leaks)} may not be self features: routing an "
                f"observation's own intensity to the scale head makes "
                f"K_j = I_j / I0_h fit perfectly and mean nothing. "
                f"Put them in context_features, where leave-one-out excludes "
                f"the observation from its own pool."
            )
        crystal_features = (
            tuple(crystal_features) if crystal_features is not None
            else (CRYSTAL_FEATURES if prep.obs_info.crystal_feat is not None else ())
        )
        unknown = set(crystal_features) - set(CRYSTAL_FEAT_COLUMNS)
        if unknown:
            raise KeyError(f"unknown crystal features {sorted(unknown)}")
        if crystal_features and prep.obs_info.crystal_feat is None:
            raise ValueError(
                "crystal_features requested but ObsInfo.crystal_feat is None "
                "(prepare_observations builds it from the stream core)"
            )
        for tier, names in (("self", self_features), ("context", context_features),
                            ("crystal", crystal_features)):
            bad = CONDITION_FEATURES.intersection(names)
            if bad:
                raise ValueError(
                    f"{sorted(bad)} requested as {tier} features. The scale "
                    f"model must be blind to the light/dark condition — any "
                    f"degree of freedom that distinguishes them absorbs the "
                    f"difference signal. Use separate merge targets instead."
                )

        self.self_features = tuple(self_features)
        self.context_features = tuple(context_features)
        self.crystal_features = tuple(crystal_features)

        oi = prep.obs_info
        if oi.obs_id is None:
            raise ValueError("ObsInfo.obs_id is required (prepare_observations sets it)")

        n = prep.n_obs_total
        I = prep.I_obs.float()
        bg = prep.bg.float()
        sigma = (
            oi.sigma.float() if oi.sigma is not None
            else (I.abs() + bg.clamp(min=0) + 1.0).sqrt()   # Poisson fallback
        )
        # persistent=False on every buffer that is a COPY OF THE DATA.  They
        # are rebuilt from the prep at construction, so saving them puts the
        # data set in the checkpoint: three per-observation arrays, 429 MB on a
        # 35 M-observation probe and ~10 GB on the full 794 M-observation set,
        # rewritten at every epoch's save_best.  Worse, a strict load would then
        # demand that a *different* data set have the same number of rows.  The
        # standardisation statistics below stay persistent: those are fitted,
        # and a merge that recomputed them would be standardising on its own
        # subset.
        self.register_buffer("_I", I, persistent=False)
        self.register_buffer("_bg", bg, persistent=False)
        self.register_buffer("_sigma", sigma.clamp(min=1e-6), persistent=False)
        self.register_buffer("_hkl", prep.hkl_valid.float(), persistent=False)
        self.has_supplied_sigma = oi.sigma is not None
        # Reference merged intensity per reflection, used to form obs_scale.
        # Always detached: it is a lookup, never a gradient path.  Starts flat;
        # the trainer installs the running merge (set_reference_I0) before the
        # first forward.
        self.register_buffer("_I0_ref", torch.ones(prep.n_valid), persistent=False)
        # Within-reflection rank, fixed by the data alone: computed once over
        # every observation of the prepared set (all conditions pooled, as the
        # single reference is), never refreshed.
        self.register_buffer(
            "_rank_score",
            within_reflection_normal_score(I, oi.hkl_idx, prep.n_valid)
            if "rank_score" in self.context_features else torch.zeros(0),
            persistent=False,
        )

        # Normalisation stats over the FULL dataset (never per batch — that
        # would leak batch composition into the features).
        self._stat_obs_info = oi
        self._stat_names = tuple(
            sorted(set(self.self_features) | set(self.context_features))
        )
        self.register_buffer("_mean", torch.zeros(len(self._stat_names)))
        self.register_buffer("_std", torch.ones(len(self._stat_names)))
        self.register_buffer(
            "_const", torch.zeros(len(self._stat_names), dtype=torch.bool)
        )
        self._reported_const: list[str] = []

        # Per-crystal table.  Standardised over CRYSTALS (rows of the table),
        # with the same constant-column rule as the per-observation tiers.
        n_cf = len(self.crystal_features)
        if n_cf:
            idx = [CRYSTAL_FEAT_COLUMNS.index(k) for k in self.crystal_features]
            raw = oi.crystal_feat.float()[:, idx]
            if raw.shape[0] != oi.n_crystals:
                raise ValueError(
                    f"crystal_feat has {raw.shape[0]} rows for {oi.n_crystals} crystals"
                )
        else:
            raw = torch.zeros(oi.n_crystals, 0)
        self.register_buffer("_crys_raw", raw, persistent=False)
        self.register_buffer("_crys_mean", torch.zeros(n_cf))
        self.register_buffer("_crys_std", torch.ones(n_cf))
        self.register_buffer("_crys_const", torch.zeros(n_cf, dtype=torch.bool))
        if n_cf:
            # Only crystals that actually have observations define the stats.
            present = torch.bincount(oi.crystal_id, minlength=oi.n_crystals) > 0
            rows = raw[present.to(raw.device)]
            mean = rows.mean(dim=0)
            std = rows.std(dim=0) if rows.shape[0] > 1 else torch.zeros(n_cf)
            scale = std.abs() + mean.abs() + 1.0
            const = std <= self.CONST_REL_TOL * scale
            self._crys_mean.copy_(mean)
            self._crys_std.copy_(torch.where(const, torch.ones_like(std),
                                             std.clamp(min=1e-6)))
            self._crys_const.copy_(const)
        self._recompute_stats()

    #: A column whose spread is below this fraction of its own scale carries
    #: no information and is treated as constant.
    CONST_REL_TOL: float = 1e-6

    #: Rows used to estimate the per-feature mean and std.  Full-dataset stats
    #: over 133 M observations buy nothing a 2 M sample does not already give.
    STAT_SAMPLE: int = 2_000_000

    @torch.no_grad()
    def _recompute_stats(self) -> None:
        """Refresh normalisation over the full dataset.

        Must be redone whenever the reference I0 moves: ``obs_scale`` and
        ``log_I0_ref`` are both defined relative to it, so statistics computed
        against a stale reference leave those channels badly off-scale.  That
        is not a cosmetic issue — with the stats frozen at the initial
        reference, the synthetic recovery test fell from rho = 0.96 to 0.11.
        """
        # Subsample.  These are per-feature means and standard deviations, and
        # 2 M rows fix them to ~0.07 %; using all 133 M costs a 4 GB host-device
        # transfer every epoch once `prep` is host-resident, because the
        # module's own `_I`/`_sigma` buffers live on the compute device while
        # the cached stat meta does not.  Deterministic, so the normalisation
        # does not jitter between calls.
        meta_s = self._stat_obs_info
        n = int(meta_s.hkl_idx.numel())
        if n > self.STAT_SAMPLE:
            # With replacement: a full randperm(n) to take 2 M rows costs
            # seconds at 1e8 observations and buys nothing for a mean and std.
            g = torch.Generator(device="cpu").manual_seed(0)
            idx = torch.randint(n, (self.STAT_SAMPLE,), generator=g)
            meta_s = meta_s.gather(idx.to(meta_s.hkl_idx.device))
        meta_s = meta_s.to(self._I.device)
        full = self._raw_columns(meta_s, self._stat_names)
        mean = torch.stack([full[k].mean() for k in self._stat_names])
        raw_std = torch.stack([full[k].std() for k in self._stat_names])

        # Near-constant columns must be zeroed, not standardised.  Dividing by
        # a std of ~0 amplifies float noise without limit: ``beam_energy`` is
        # ~12960 eV with essentially no spread, and normalising it produced
        # feature values of order 1e6 and a training loss of 5.7e6 on the first
        # epoch.  ``profile_rad`` is exactly constant on this dataset
        # (--fix-profile-radius), so this is not a corner case.
        scale = raw_std.abs() + mean.abs() + 1.0
        const = raw_std <= self.CONST_REL_TOL * scale
        self._mean.copy_(mean)
        self._std.copy_(torch.where(const, torch.ones_like(raw_std),
                                    raw_std.clamp(min=1e-6)))
        self._const.copy_(const)
        dropped = [n for n, c in zip(self._stat_names, const.tolist()) if c]
        if dropped and dropped != self._reported_const:
            self._reported_const = dropped
            warnings.warn(
                f"ObsFeatureSpec: {dropped} are constant to within "
                f"{self.CONST_REL_TOL:g} and carry no information; they are "
                f"emitted as zeros rather than standardised.",
                RuntimeWarning, stacklevel=2,
            )

    @torch.no_grad()
    def set_reference_I0(self, I0: Tensor) -> None:
        """Update the per-reflection reference used by ``obs_scale``.

        Call with the running merge (the target's EMA).  Detached by
        construction — this is a feature lookup, not part of the objective.
        """
        if I0.shape != self._I0_ref.shape:
            raise ValueError(
                f"I0 must have shape {tuple(self._I0_ref.shape)}, "
                f"got {tuple(I0.shape)}"
            )
        self._I0_ref.copy_(I0.detach().clamp(min=1e-3))
        self._recompute_stats()

    # ── raw channels ───────────────────────────────────────────────────

    def _raw_columns(self, meta: ObsInfo, names) -> dict[str, Tensor]:
        oid = meta.obs_id
        out: dict[str, Tensor] = {}
        need = set(names)

        I = sig = None
        if {"signed_log_I", "i_over_sigma", "obs_scale",
            "obs_scale_over_sigma"} & need:
            I = self._I[oid.to(self._I.device)]
        if {"log_sigma", "i_over_sigma", "obs_scale_over_sigma"} & need:
            sig = self._sigma[oid.to(self._sigma.device)]
        if {"obs_scale", "obs_scale_over_sigma", "log_I0_ref"} & need:
            # `_I0_ref` is a buffer and follows the module's device, but the
            # cached full-dataset meta used by `_recompute_stats` stays on the
            # host once `prep` is host-resident (stream_batches).  Index on the
            # meta's device: it is (n_hkl,), so moving it is free, whereas
            # moving the meta would defeat the point of streaming.
            I0 = self._I0_ref.to(meta.hkl_idx.device)[meta.hkl_idx]
            I = I.to(I0.device) if I is not None else I
            sig = sig.to(I0.device) if sig is not None else sig

        for k in need:
            if k == "s_sq":
                out[k] = meta.s_sq.float()
            elif k == "ewald_offset":
                out[k] = _or_zeros(meta.ewald_offset, meta)
            elif k == "fs":
                out[k] = _or_zeros(meta.fs, meta)
            elif k == "ss":
                out[k] = _or_zeros(meta.ss, meta)
            elif k in ("qx", "qy", "qz"):
                out[k] = (
                    meta.q_dir[:, "xyz".index(k[1])].float()
                    if meta.q_dir is not None else torch.zeros_like(meta.s_sq)
                )
            elif k == "beam_energy":
                out[k] = _or_zeros(meta.beam_energy, meta)
            elif k == "crystal_res":
                out[k] = _or_zeros(meta.crystal_res, meta)
            elif k == "panel":
                out[k] = (
                    meta.panel_id.float() if meta.panel_id is not None
                    else torch.zeros_like(meta.s_sq)
                )
            elif k in ("h", "k", "l"):
                out[k] = self._hkl[meta.hkl_idx, "hkl".index(k)]
            elif k == "signed_log_I":
                out[k] = torch.sign(I) * torch.log1p(I.abs())
            elif k == "log_sigma":
                out[k] = torch.log(sig)
            elif k == "i_over_sigma":
                out[k] = (I / sig).clamp(-50.0, 50.0)
            elif k == "log_bg":
                out[k] = torch.log1p(self._bg[oid].clamp(min=0.0))
            elif k == "obs_scale":
                # K.S for this peer.  Clipped, not logged: negatives are real.
                out[k] = (I / I0).clamp(-20.0, 20.0)
            elif k == "obs_scale_over_sigma":
                # How well determined that observed scale is.
                out[k] = (I / sig).clamp(-50.0, 50.0) * (sig / I0).clamp(max=50.0)
            elif k == "log_I0_ref":
                out[k] = I0.log()
            elif k == "rank_score":
                out[k] = self._rank_score[oid.to(self._rank_score.device)].to(meta.s_sq.device)
            else:
                raise KeyError(f"unknown feature {k!r}")
        return out

    def _matrix(self, meta: ObsInfo, names: tuple[str, ...]) -> Tensor:
        cols = self._raw_columns(meta, names)
        idx = [self._stat_names.index(k) for k in names]
        x = torch.stack([cols[k] for k in names], dim=-1)
        out = (x - self._mean[idx]) / self._std[idx]
        return out.masked_fill(self._const[idx], 0.0)

    # ── public ─────────────────────────────────────────────────────────

    @property
    def n_self(self) -> int:
        return len(self.self_features)

    @property
    def n_context(self) -> int:
        return len(self.context_features)

    @property
    def n_crystal(self) -> int:
        return len(self.crystal_features)

    def crystal_matrix(self) -> Tensor:
        """``(n_crystals, n_crystal)`` standardised per-crystal table."""
        x = (self._crys_raw - self._crys_mean) / self._crys_std
        return x.masked_fill(self._crys_const, 0.0)

    def build(self, meta: ObsInfo) -> tuple[Tensor, Tensor, Tensor]:
        """``meta`` (sliced or gathered) → ``(x_self, x_ctx, x_crys)``, all normalised.

        ``x_crys`` is the full ``(n_crystals, n_crystal)`` per-crystal table --
        indexed by ``crystal_id``, not by row -- so it is the same object for
        every slice, gather or chunk of the dataset.
        """
        if meta.obs_id is None:
            raise ValueError("ObsFeatureSpec.build needs ObsInfo.obs_id")
        x_ctx = self._matrix(meta, self.context_features)
        return self._matrix(meta, self.self_features), x_ctx, self.crystal_matrix()


@torch.no_grad()
def within_reflection_normal_score(I: Tensor, hkl_id: Tensor, n_hkl: int) -> Tensor:
    """Normal score of each observation's rank among its reflection's observations.

    With ``c`` observations of a reflection, the one at position ``p`` (0-based,
    ascending intensity; ties take their mean position) has the rank
    ``r = (p + 1/2) / c`` and the score ``Phi^-1(r)``.  One sort of a combined
    ``(reflection, intensity)`` key: the float intensity is mapped to an
    order-preserving unsigned 32-bit code below the reflection index.
    """
    n = int(I.numel())
    if n == 0:
        return torch.zeros(0)
    I = I.detach().float().contiguous().cpu()
    h = hkl_id.detach().to(torch.int64).cpu()
    bits = I.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    code = torch.where(bits >= 0x80000000, 0xFFFFFFFF - bits, bits + 0x80000000)
    key, order = torch.sort(h * (1 << 32) + code)
    del bits, code
    new = torch.ones(n, dtype=torch.bool)
    new[1:] = key[1:] != key[:-1]
    run = new.cumsum(0) - 1
    start = new.nonzero().squeeze(1)
    length = torch.bincount(run)
    mid = start[run].double() + 0.5 * (length[run] - 1).double()
    del new, run, start, length
    h_sorted = key >> 32
    counts = torch.bincount(h, minlength=n_hkl)
    first = counts.cumsum(0) - counts
    r = (mid - first[h_sorted].double() + 0.5) / counts[h_sorted].double()
    score = torch.empty(n, dtype=torch.float32)
    score[order] = (math.sqrt(2.0) * torch.erfinv(2.0 * r - 1.0)).float()
    return score


def _or_zeros(t: Tensor | None, meta: ObsInfo) -> Tensor:
    return t.float() if t is not None else torch.zeros_like(meta.s_sq)
