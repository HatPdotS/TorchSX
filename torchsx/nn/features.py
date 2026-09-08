"""Dataloader feature head for neural reflection scaling (multi-system).

Turns one or more :class:`~torchsx.stream.SFXStream`s into crystal-grouped,
padded token batches ready for a transformer-over-reflections, paired with the
flat per-observation tensors that :class:`torchsx.nn.consistency_loss.ScaleConsistencyNLL`
consumes.  Each frame (crystal) is a context; each reflection is a token.

The whole point of this module is to make the **feature contract** that keeps
the consistency loss collapse-free *structural* rather than a comment.  Token
features are split into three tiers:

* ``scale_features`` — what the (future) scale head is allowed to consume:
  smooth diffraction geometry (s², Ewald offset, fs/ss, panel, beam energy,
  profile radius, resolution) plus **per-crystal aggregate intensity** (the
  crystal's brightness and resolution-falloff slope).  The per-crystal
  aggregates are the legitimate "how strong is this crystal" cue — they drive a
  per-crystal scalar and do not reveal any single reflection's residual.
* ``group_fields`` — discrete ``(h,k,l)`` and ``system_id``.  These build the
  global ``hkl_id`` used by the *loss* grouping.  They are exposed to the *net*
  only under multi-system training (``include_hkl_in_features=True``), never in
  the single-system case, because a single-system net could memorise
  ``(h,k,l)→intensity`` and absorb the structure factor into the "scale".
* ``target_fields`` — ``I_obs``, ``sigma_det``, ``peak``, ``background``.
  Consumed by the **loss only**.  They are returned in a *separate* dict from
  ``scale_features`` so routing an observation's own intensity into the scale
  head requires a deliberate, visible act (which would reintroduce the
  always-fatal own-intensity collapse).

**Multi-system** is first-class: ``log_I0`` is keyed per ``(system, hkl)``.  Each
system gets a disjoint block of the global HKL index (and of the Wilson shells),
so a shared network cannot tie one ``I`` value to an HKL identity across
systems — the mechanism that makes the geometry/HKL features safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from torchsx.merging.observations import _ewald_offset
from torchsx.merging.hkl import compute_hkl_index
from torchsx.nn.pad import group_offsets_from_counts, pack, unpack

# Default smooth-geometry + per-crystal-aggregate features the scale head may see.
# (Order is the column order of the emitted ``scale_features`` matrix.)
DEFAULT_SCALE_FEATURES: tuple[str, ...] = (
    "s_sq",              # (sin θ / λ)² — resolution of this reflection
    "ewald_offset",      # signed Ewald excitation error (Å⁻¹) → partiality
    "fs", "ss",          # detector position (absorption / geometry)
    "panel",             # detector panel index (float; a real net would embed)
    "beam_energy",       # per-image incident energy
    "profile_rad",       # per-crystal reflection profile radius
    "resolution",        # per-crystal resolution limit
    "crystal_log_mean_I",  # per-crystal aggregate brightness (safe scale cue)
    "crystal_falloff",     # per-crystal d log I / d s² slope (safe scale cue)
)

# Fields that must never enter ``scale_features`` (own-intensity leak).
TARGET_FIELDS: tuple[str, ...] = ("I_obs", "sigma_det", "peak", "background")


@dataclass
class ReflectionBatch:
    """One crystal-grouped, padded batch + the flat per-obs loss inputs.

    Conventions follow :class:`torchsx.nn.batch.GroupedBatch`: ``B`` groups
    (crystals) padded to ``L`` tokens; ``padding_mask[b, l]`` is ``True`` at PAD
    positions (the ``src_key_padding_mask`` convention).  All flat tensors are in
    the batch's token order and have length ``n = sum(n_per_group)``;
    :meth:`unpack` maps a padded ``(B, L, *)`` tensor (e.g. the net's output)
    back to that flat order.
    """

    scale_features: Tensor                 # (B, L, F) padded, normalised
    feature_names: tuple[str, ...]
    padding_mask: Tensor                   # (B, L) bool, True = PAD
    n_per_group: Tensor                    # (B,)
    group_offsets: Tensor                  # (B+1,) into the flat tensors

    # Flat per-observation loss inputs (batch order):
    I_obs: Tensor                          # (n,)
    sigma_det: Tensor                      # (n,)
    hkl_id: Tensor                         # (n,) int64 — GLOBAL (system,hkl) index
    system_id: Tensor                      # (n,) int64
    crystal_id: Tensor                     # (n,) int64 — per-system crystal index
    obs_idx: Tensor                        # (n,) int64 — row in the source system core

    patches: Tensor | None = None          # (B, L, h, w) or None
    target: dict[str, Tensor] = field(default_factory=dict)  # padded target fields

    def unpack(self, padded: Tensor) -> Tensor:
        """Padded ``(B, L, *)`` (e.g. net output) → flat ``(n, *)`` in batch order."""
        return unpack(padded, self.group_offsets)

    def to(self, device) -> "ReflectionBatch":
        m = lambda t: None if t is None else t.to(device)
        return ReflectionBatch(
            scale_features=self.scale_features.to(device),
            feature_names=self.feature_names,
            padding_mask=self.padding_mask.to(device),
            n_per_group=self.n_per_group.to(device),
            group_offsets=self.group_offsets.to(device),
            I_obs=self.I_obs.to(device),
            sigma_det=self.sigma_det.to(device),
            hkl_id=self.hkl_id.to(device),
            system_id=self.system_id.to(device),
            crystal_id=self.crystal_id.to(device),
            obs_idx=self.obs_idx.to(device),
            patches=m(self.patches),
            target={k: v.to(device) for k, v in self.target.items()},
        )


class _SystemBlock:
    """Crystal-sorted resident arrays for one system, ready to slice into batches."""

    __slots__ = (
        "system_idx", "j_offset", "n_unique", "shell_offset", "n_shells",
        "feat", "I_obs", "sigma_det", "hkl_id_global", "crystal_id", "obs_idx",
        "patches", "peak", "background", "offsets", "n_groups",
    )


def _crystal_falloff(cid: Tensor, s_sq: Tensor, log_I: Tensor, n_crystals: int) -> Tensor:
    """Per-crystal least-squares slope d(log I)/d(s²), 0 where undetermined."""
    one = torch.ones_like(s_sq)
    n = torch.zeros(n_crystals).scatter_add(0, cid, one)
    sx = torch.zeros(n_crystals).scatter_add(0, cid, s_sq)
    sy = torch.zeros(n_crystals).scatter_add(0, cid, log_I)
    sxx = torch.zeros(n_crystals).scatter_add(0, cid, s_sq * s_sq)
    sxy = torch.zeros(n_crystals).scatter_add(0, cid, s_sq * log_I)
    denom = (n * sxx - sx * sx)
    slope = torch.where(denom.abs() > 1e-8, (n * sxy - sx * sy) / denom, torch.zeros_like(denom))
    return slope[cid]


class ReflectionDataset:
    """Multi-system reflection dataset → crystal-grouped token batches.

    Parameters
    ----------
    streams
        One :class:`SFXStream` or a list of them.  Each is a distinct *system*
        and receives its own disjoint block of the global ``(system, hkl)``
        index used to size :class:`ScaleConsistencyNLL`'s ``log_I0``.
    noise_model
        A :class:`~torchsx.merging.likelihood.gaussian_polynomial.GaussianPolynomialVariance`
        (or any object with ``sigma_squared_obs(n, bg)``) used to compute the
        per-observation detector σ from ``(peak, background)``.  ``None`` →
        pure-Poisson default (σ² = max(peak, bg)).  For real data, pass a
        *calibrated* model.
    scale_features
        Ordered feature names for the ``scale_features`` matrix.  Defaults to
        :data:`DEFAULT_SCALE_FEATURES`.  A guard rejects any
        :data:`TARGET_FIELDS` name (the own-intensity contract).
    include_hkl_in_features
        Append the (normalised) discrete ``h, k, l`` to ``scale_features``.
        Only legitimate with ≥2 systems (asserted); see module docstring.
    groups_per_batch, shuffle, seed, device
        Batch size in crystals, batch-order shuffling, RNG seed, target device.
    """

    def __init__(
        self,
        streams,
        *,
        noise_model=None,
        scale_features: tuple[str, ...] = DEFAULT_SCALE_FEATURES,
        include_hkl_in_features: bool = False,
        groups_per_batch: int = 256,
        shuffle: bool = False,
        seed: int | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        if hasattr(streams, "_core"):
            streams = [streams]
        self.streams = list(streams)
        if not self.streams:
            raise ValueError("need at least one stream")

        bad = [f for f in scale_features if f in TARGET_FIELDS]
        if bad:
            raise ValueError(
                f"scale_features may not contain target fields {bad}: feeding an "
                f"observation's own intensity to the scale head collapses the loss "
                f"(see torchsx.nn.features module docstring)."
            )
        if include_hkl_in_features and len(self.streams) < 2:
            raise ValueError(
                "include_hkl_in_features requires ≥2 systems — a single-system net "
                "can memorise (h,k,l)→intensity and absorb the structure factor."
            )
        self.scale_feature_names = tuple(scale_features) + (
            ("h", "k", "l") if include_hkl_in_features else ()
        )

        if noise_model is None:
            from torchsx.merging.likelihood.gaussian_polynomial import (
                GaussianPolynomialVariance,
            )
            noise_model = GaussianPolynomialVariance(var_linear=1.0)
        self.noise_model = noise_model

        self.groups_per_batch = int(groups_per_batch)
        self.shuffle = bool(shuffle)
        self.device = device
        self._gen = torch.Generator()
        if seed is not None:
            self._gen.manual_seed(seed)

        # First pass: per-system raw (native-order) feature dicts, for global
        # normalisation stats.
        raw_per_system = [self._raw_features(s._core) for s in self.streams]
        all_feat = {
            name: torch.cat([r["feat"][name] for r in raw_per_system])
            for name in self.scale_feature_names
        }
        self._feat_mean = {n: all_feat[n].mean() for n in self.scale_feature_names}
        self._feat_std = {
            n: all_feat[n].std().clamp(min=1e-6) for n in self.scale_feature_names
        }

        # Second pass: build crystal-sorted blocks with a disjoint J-index per system.
        self.blocks: list[_SystemBlock] = []
        j_offset = 0
        shell_offset = 0
        for sidx, (stream, raw) in enumerate(zip(self.streams, raw_per_system)):
            blk = self._build_block(sidx, raw, j_offset, shell_offset)
            self.blocks.append(blk)
            j_offset += blk.n_unique
            shell_offset += blk.n_shells
        self.n_hkl = j_offset
        self.n_shells_total = shell_offset

    # ── feature computation ─────────────────────────────────────────────

    def _raw_features(self, core) -> dict:
        """Native-order per-obs feature dict + raw I/peak/bg + local hkl index."""
        h = core.hkl[:, 0].float()
        k = core.hkl[:, 1].float()
        l = core.hkl[:, 2].float()
        cid = core.crystal_id.long()
        n_crystals = int(cid.max().item()) + 1

        q = (h[:, None] * core.astar[cid] + k[:, None] * core.bstar[cid]
             + l[:, None] * core.cstar[cid])
        qn = q.norm(dim=1)
        s_sq = (qn * qn) / 4.0

        I = core.I.float()
        peak = core.peak.float() if core.peak is not None else (I + (
            core.background.float() if core.background is not None else 0.0))
        bg = core.background.float() if core.background is not None else torch.zeros_like(I)

        log_I = I.clamp(min=1e-3).log()
        sumI = torch.zeros(n_crystals).scatter_add(0, cid, I.clamp(min=1e-3))
        cnt = torch.zeros(n_crystals).scatter_add(0, cid, torch.ones_like(I))
        crystal_log_mean_I = (sumI / cnt.clamp(min=1.0)).log()[cid]
        crystal_falloff = _crystal_falloff(cid, s_sq, log_I, n_crystals)

        feat = {
            "s_sq": s_sq,
            "ewald_offset": _ewald_offset(core).float(),
            "fs": core.fs.float() if core.fs is not None else torch.zeros_like(I),
            "ss": core.ss.float() if core.ss is not None else torch.zeros_like(I),
            "panel": core.panel_id.float() if core.panel_id is not None else torch.zeros_like(I),
            "beam_energy": core.broadcast("beam_energy", "image", "obs").float(),
            "profile_rad": core.broadcast("profile_rad", "crystal", "obs").float(),
            "resolution": core.broadcast("resolution", "crystal", "obs").float(),
            "crystal_log_mean_I": crystal_log_mean_I,
            "crystal_falloff": crystal_falloff,
            "h": h, "k": k, "l": l,
        }
        sigma_det = self.noise_model.sigma_squared_obs(peak, bg).clamp(min=1e-12).sqrt()
        idx = compute_hkl_index(core.hkl)
        return {
            "feat": {n: feat[n] for n in self.scale_feature_names},
            "I_obs": I, "peak": peak, "background": bg,
            "sigma_det": sigma_det,
            "hkl_id_local": idx.hkl_id.long(),
            "n_unique": idx.n_unique,
            "crystal_id": cid,
        }

    def _build_block(self, sidx, raw, j_offset, shell_offset) -> _SystemBlock:
        cid = raw["crystal_id"]
        order = torch.argsort(cid, stable=True)
        uniq, counts = torch.unique_consecutive(cid[order], return_counts=True)

        feat_mat = torch.stack([
            (raw["feat"][n] - self._feat_mean[n]) / self._feat_std[n]
            for n in self.scale_feature_names
        ], dim=1)[order]

        blk = _SystemBlock()
        blk.system_idx = sidx
        blk.j_offset = j_offset
        blk.n_unique = raw["n_unique"]
        blk.feat = self._move(feat_mat)
        blk.I_obs = self._move(raw["I_obs"][order])
        blk.sigma_det = self._move(raw["sigma_det"][order])
        blk.hkl_id_global = self._move(raw["hkl_id_local"][order] + j_offset)
        blk.crystal_id = self._move(cid[order])
        blk.obs_idx = self._move(torch.arange(cid.shape[0])[order])
        blk.peak = self._move(raw["peak"][order])
        blk.background = self._move(raw["background"][order])
        blk.patches = None
        blk.offsets = self._move(group_offsets_from_counts(counts))
        blk.n_groups = int(uniq.shape[0])

        # Wilson shells for this system (disjoint block of the global shell index).
        n_shells = self._wilson_shells(sidx, raw, shell_offset)
        blk.shell_offset = shell_offset
        blk.n_shells = n_shells
        return blk

    def _move(self, t: Tensor) -> Tensor:
        return t if self.device is None else t.to(self.device)

    # ── patches ─────────────────────────────────────────────────────────

    def attach_patches(self, patch_size: int = 21, **excise_kwargs) -> None:
        """Excise ``patch_size×patch_size`` patches per system and align them.

        Plumbing only — no model consumes patches yet.  Stores a
        ``(n_obs, h, w)`` tensor per block (native-order patches reordered into
        crystal order) so :meth:`__iter__` can pack a ``(B, L, h, w)`` channel.
        """
        for stream, blk in zip(self.streams, self.blocks):
            ex = stream.excise_patches(patch_size=patch_size, **excise_kwargs)
            patches = _aligned_patches(ex, stream._core.n_observations, patch_size)
            # blk.obs_idx maps crystal-sorted rows → native rows.
            blk.patches = self._move(patches[blk.obs_idx.cpu()].float())

    # ── Wilson prior ────────────────────────────────────────────────────

    def _wilson_shells(self, sidx, raw, shell_offset, n_shells: int = 20) -> int:
        """Stash per-valid-HKL shell ids + centric flags for this system.

        Returns the number of shells (always ``n_shells``); the arrays are kept
        on the dataset keyed by system for :meth:`wilson_prior`.
        """
        if not hasattr(self, "_wilson"):
            self._wilson = {}
        core = self.streams[sidx]._core
        idx = compute_hkl_index(core.hkl)
        unique_hkl = idx.unique_hkl
        n_unique = idx.n_unique

        # Resolution-rank shells over the unique HKLs (median cell).
        cell = core.cell.median(dim=0).values
        from torchref.base.reciprocal.hkl import get_d_spacing
        d = get_d_spacing(unique_hkl.float(), cell)
        rank = d.argsort().argsort()
        shell_id = (rank.float() / max(n_unique, 1) * n_shells).long().clamp(max=n_shells - 1)

        sg = getattr(self.streams[sidx], "spacegroup", None)
        is_centric = torch.zeros(n_unique, dtype=torch.bool)
        if sg and sg != "P 1":
            try:
                from torchref.symmetry import SpaceGroup
                is_centric = SpaceGroup(sg).is_centric(unique_hkl.long())
            except Exception:  # noqa: BLE001 — fall back to acentric
                pass
        self._wilson[sidx] = (shell_id + shell_offset, is_centric)
        return n_shells

    def wilson_prior(self, mean_kind: str = "arithmetic"):
        """Build a :class:`WilsonPrior` over the global ``(system, hkl)`` index.

        Each system contributes its own disjoint shells, so the prior's shell
        means are computed within a system — the per-system gauge anchor.
        """
        from torchsx.merging.priors.wilson import WilsonPrior

        shell_id = torch.cat([self._wilson[b.system_idx][0] for b in self.blocks])
        is_centric = torch.cat([self._wilson[b.system_idx][1] for b in self.blocks])
        prior = WilsonPrior(
            shell_id=shell_id, is_centric=is_centric,
            n_shells=self.n_shells_total, mean_kind=mean_kind,
        )
        return prior if self.device is None else prior.to(self.device)

    def mu_init(self) -> Tensor:
        """Per-global-HKL seed for ``log_I0``: log mean positive obs intensity."""
        mu = torch.zeros(self.n_hkl)
        for blk in self.blocks:
            I = blk.I_obs.detach().cpu()
            hid = blk.hkl_id_global.detach().cpu()
            pos = I.clamp(min=1e-3)
            n = mu.new_zeros(self.n_hkl).scatter_add(0, hid, torch.ones_like(pos))
            s = mu.new_zeros(self.n_hkl).scatter_add(0, hid, pos)
            mask = n > 0
            mu[mask] = (s[mask] / n[mask]).log()
        return mu if self.device is None else mu.to(self.device)

    # ── iteration ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return sum(
            (b.n_groups + self.groups_per_batch - 1) // self.groups_per_batch
            for b in self.blocks
        )

    def __iter__(self):
        # Each batch stays within one system (a frame belongs to one system).
        units: list[tuple[_SystemBlock, Tensor]] = []
        for blk in self.blocks:
            gorder = (
                torch.randperm(blk.n_groups, generator=self._gen)
                if self.shuffle else torch.arange(blk.n_groups)
            )
            for start in range(0, blk.n_groups, self.groups_per_batch):
                units.append((blk, gorder[start:start + self.groups_per_batch]))
        if self.shuffle:
            perm = torch.randperm(len(units), generator=self._gen).tolist()
            units = [units[i] for i in perm]
        for blk, gsel in units:
            yield self._build_batch(blk, gsel)

    def _build_batch(self, blk: _SystemBlock, gsel: Tensor) -> ReflectionBatch:
        gsel = gsel.to(blk.offsets.device)
        starts = blk.offsets[:-1][gsel]
        stops = blk.offsets[1:][gsel]
        counts = stops - starts
        row_idx = torch.cat([
            torch.arange(int(s), int(e), device=blk.offsets.device)
            for s, e in zip(starts.tolist(), stops.tolist())
        ]) if gsel.numel() else torch.empty(0, dtype=torch.long, device=blk.offsets.device)
        local_offsets = group_offsets_from_counts(counts)

        feats_padded, mask = pack(blk.feat[row_idx], local_offsets)
        patches_padded = None
        if blk.patches is not None:
            patches_padded, _ = pack(blk.patches[row_idx], local_offsets)

        sysid = torch.full((row_idx.shape[0],), blk.system_idx,
                           dtype=torch.long, device=row_idx.device)
        target = {}
        for name, src in (("I_obs", blk.I_obs), ("sigma_det", blk.sigma_det),
                          ("peak", blk.peak), ("background", blk.background)):
            target[name], _ = pack(src[row_idx], local_offsets)

        return ReflectionBatch(
            scale_features=feats_padded,
            feature_names=self.scale_feature_names,
            padding_mask=mask,
            n_per_group=counts,
            group_offsets=local_offsets,
            I_obs=blk.I_obs[row_idx],
            sigma_det=blk.sigma_det[row_idx],
            hkl_id=blk.hkl_id_global[row_idx],
            system_id=sysid,
            crystal_id=blk.crystal_id[row_idx],
            obs_idx=blk.obs_idx[row_idx],
            patches=patches_padded,
            target=target,
        )


def _aligned_patches(ex, n_obs: int, patch_size: int) -> Tensor:
    """Best-effort align an ``ExcisedPatches`` result to native obs order.

    The excision API sorts internally by (file, event); if it exposes an index
    back to the original observations we use it, otherwise we assume the result
    is already in native order.
    """
    patches = getattr(ex, "patches", ex)
    if not torch.is_tensor(patches):
        patches = torch.as_tensor(patches)
    for attr in ("obs_index", "obs_idx", "order", "index"):
        idx = getattr(ex, attr, None)
        if idx is not None:
            inv = torch.empty(n_obs, dtype=torch.long)
            inv[torch.as_tensor(idx).long()] = torch.arange(len(idx))
            return patches[inv]
    return patches
