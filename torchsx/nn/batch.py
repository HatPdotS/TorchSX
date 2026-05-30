"""Grouped, padded batching of a columnar stream for transformer training.

:class:`StreamBatcher` groups a stream's observations by crystal (or frame),
sorts once, and yields :class:`GroupedBatch`es of padded ``(B, L_max, *T)``
column tensors plus a ``src_key_padding_mask`` — each batch item being the
variable-length token set of one crystal/frame. Per-group (crystal/image)
columns are carried one-row-per-group, not L_max-expanded.

Single-process and GPU-resident: requested obs columns are moved to ``device``
once after the group-sort; batch slicing is then pure indexing. (Lazy/mmap
shoebox fetch and a prefetch thread are future optimisations.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from torchsx.nn.pad import group_offsets_from_counts, pack, unpack


@dataclass
class GroupedBatch:
    """One batch of grouped, padded observations."""

    columns: dict[str, torch.Tensor]          # name → (B, L_max, *T) padded
    padding_mask: torch.Tensor                # (B, L_max) bool, True = PAD
    n_per_group: torch.Tensor                 # (B,)
    group_offsets: torch.Tensor               # (B+1,) into obs_idx
    obs_idx: torch.Tensor                     # (sum n,) flat row indices (sorted order)
    group_columns: dict[str, torch.Tensor] = field(default_factory=dict)  # name → (B, *T)
    group_id: torch.Tensor | None = None      # (B,) parent-level id per group

    def to(self, device) -> "GroupedBatch":
        return GroupedBatch(
            columns={k: v.to(device) for k, v in self.columns.items()},
            padding_mask=self.padding_mask.to(device),
            n_per_group=self.n_per_group.to(device),
            group_offsets=self.group_offsets.to(device),
            obs_idx=self.obs_idx.to(device),
            group_columns={k: v.to(device) for k, v in self.group_columns.items()},
            group_id=None if self.group_id is None else self.group_id.to(device),
        )

    def unpack(self, name: str) -> torch.Tensor:
        """Flatten a padded column back to ``(sum n, *T)`` in this batch's order."""
        return unpack(self.columns[name], self.group_offsets)


class StreamBatcher:
    """Iterate a stream as grouped, padded batches keyed by crystal or frame."""

    def __init__(
        self,
        stream,
        *,
        group_by: str = "crystal",
        obs_columns: list[str] | None = None,
        group_columns: tuple[str, ...] = (),
        groups_per_batch: int = 256,
        shuffle: bool = False,
        device: str | torch.device | None = None,
        seed: int | None = None,
        max_len: int | None = None,
    ):
        core = getattr(stream, "_core", stream)  # accept SFXStream or a core
        self.core = core
        self.group_by = group_by
        self.groups_per_batch = groups_per_batch
        self.shuffle = shuffle
        self.max_len = max_len
        self._gen = torch.Generator()
        if seed is not None:
            self._gen.manual_seed(seed)

        obs = core.levels["obs"]
        if obs_columns is None:
            obs_columns = [c for c in obs.keys() if c != "crystal_id"]
        self.obs_columns = obs_columns

        # group key per observation
        if group_by == "crystal":
            gkey = core.crystal_id.long()
            self._parent_level = "crystal"
        elif group_by in ("image", "frame"):
            gkey = core.broadcast("image_id", "crystal", "obs").long()
            self._parent_level = "image"
        else:
            raise ValueError(f"group_by must be 'crystal' or 'image', got {group_by!r}")

        order = torch.argsort(gkey, stable=True)
        self.uniq, counts = torch.unique_consecutive(gkey[order], return_counts=True)
        self.offsets = group_offsets_from_counts(counts)
        self.n_groups = int(self.uniq.shape[0])

        dev = device
        # move resident obs columns once, in sorted order
        self._cols = {
            name: obs[name][order].to(dev) if dev is not None else obs[name][order]
            for name in obs_columns
        }
        self.obs_idx = torch.arange(core.n_observations)[order]
        if dev is not None:
            self.obs_idx = self.obs_idx.to(dev)
            self.offsets = self.offsets.to(dev)
            self.uniq = self.uniq.to(dev)

        # per-group (parent-level) columns, one row per group
        parent = core.levels[self._parent_level]
        self._group_cols = {
            name: (parent[name][self.uniq.cpu().long()].to(dev)
                   if dev is not None else parent[name][self.uniq.long()])
            for name in group_columns
        }

    def __len__(self) -> int:
        return (self.n_groups + self.groups_per_batch - 1) // self.groups_per_batch

    def __iter__(self):
        gorder = (
            torch.randperm(self.n_groups, generator=self._gen)
            if self.shuffle else torch.arange(self.n_groups)
        )
        for start in range(0, self.n_groups, self.groups_per_batch):
            gsel = gorder[start:start + self.groups_per_batch]
            yield self._build_batch(gsel)

    def _build_batch(self, gsel: torch.Tensor) -> GroupedBatch:
        gsel = gsel.to(self.offsets.device)
        starts = self.offsets[:-1][gsel]
        stops = self.offsets[1:][gsel]
        counts = stops - starts
        # gather the (variable) rows of the selected groups, in group order
        row_idx = torch.cat([torch.arange(int(s), int(e), device=self.offsets.device)
                             for s, e in zip(starts.tolist(), stops.tolist())]) \
            if gsel.numel() else torch.empty(0, dtype=torch.long, device=self.offsets.device)
        local_offsets = group_offsets_from_counts(counts)

        padded_cols = {}
        mask = None
        for name, col in self._cols.items():
            sub = col[row_idx]
            p, m = pack(sub, local_offsets, max_len=self.max_len)
            padded_cols[name] = p
            mask = m
        if mask is None:  # no obs columns requested
            _, mask = pack(torch.zeros(row_idx.shape[0], 1, device=self.offsets.device),
                           local_offsets, max_len=self.max_len)

        gsel_cpu = gsel.cpu()
        return GroupedBatch(
            columns=padded_cols,
            padding_mask=mask,
            n_per_group=counts,
            group_offsets=local_offsets,
            obs_idx=self.obs_idx[row_idx],
            group_columns={k: v[gsel_cpu] if v.device.type == "cpu" else v[gsel]
                           for k, v in self._group_cols.items()},
            group_id=self.uniq[gsel],
        )
