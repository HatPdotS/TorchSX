"""Resolution-binned completeness and multiplicity of a stream.

Both count unique reflections in the stream's space group with Friedel mates
merged (its Laue class): the observed indices are mapped onto their CCP4 ASU
representatives first, so the numbers do not depend on whether the stream was
symmetrized.  Resolution comes from the median cell, and the shells are equal in
reciprocal volume (``1/d^3``) between the lowest and highest resolution observed.
"""

from __future__ import annotations

import torch

from torchsx._core import SFXStreamCore
from torchsx.merging.hkl.hkl import compute_hkl_index

# Miller indices of a stream fit in int16; offsetting by 2^15 packs (h, k, l)
# into one non-negative int64 key for set membership.
_OFFSET, _SPAN = 1 << 15, 1 << 16


def _key(hkl: torch.Tensor) -> torch.Tensor:
    h = hkl.long() + _OFFSET
    return (h[:, 0] * _SPAN + h[:, 1]) * _SPAN + h[:, 2]


def _shell_edges(d: torch.Tensor, n_bins: int) -> torch.Tensor:
    """``n_bins + 1`` edges in ``s = 1/d``, equal steps in ``s^3``."""
    s = 1.0 / d
    lo, hi = s.min() ** 3, s.max() ** 3
    steps = torch.linspace(0.0, 1.0, n_bins + 1, dtype=s.dtype, device=s.device)
    return (lo + (hi - lo) * steps) ** (1.0 / 3.0)


def _assign(d: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Shell of each reflection; the end points fall into the outer shells."""
    n_bins = edges.numel() - 1
    return (torch.bucketize(1.0 / d, edges, right=True) - 1).clamp(0, n_bins - 1)


def _observed(core: SFXStreamCore):
    """Unique ASU reflections of the stream and the observations of each.

    Returns ``(group, cell, hkl, n_obs, d)``: the space group, the median cell
    (float64), the unique ASU indices ``(N, 3)``, the observation count of each
    ``(N,)`` and its resolution ``(N,)`` in the median cell.
    """
    from torchref.base.reciprocal.hkl import get_d_spacing
    from torchref.symmetry import SpaceGroup

    if core.hkl.shape[0] == 0:
        raise ValueError("the stream holds no observations")
    group = SpaceGroup(core.spacegroup or "P 1")
    # Canonicalize the distinct indices only, not every observation.
    raw = compute_hkl_index(core.hkl)
    asu = group.canonicalize_hkl(raw.unique_hkl.int(), include_friedel=True, sort=False)[0]
    merged = compute_hkl_index(asu)
    per_raw = torch.bincount(raw.hkl_id, minlength=raw.n_unique)
    n_obs = torch.zeros(merged.n_unique, dtype=torch.int64, device=per_raw.device)
    n_obs.scatter_add_(0, merged.hkl_id.to(per_raw.device), per_raw)
    cell = core.cell.double().median(dim=0).values
    hkl = merged.unique_hkl.to(cell.device)
    d = get_d_spacing(hkl.double(), cell)
    return group, cell, hkl, n_obs.to(cell.device), d


def completeness(core: SFXStreamCore, n_bins: int = 20) -> torch.Tensor:
    """Resolution-binned completeness: observed over possible unique reflections.

    The possible reflections are those of the space group's ASU within the
    observed resolution range, systematic absences excluded; an observed index
    that is systematically absent does not count.  Returns ``(n_bins,)``,
    lowest resolution first; a shell with no possible reflection reads 0.
    """
    from torchref.base.reciprocal.hkl import generate_possible_hkl, get_d_spacing

    group, cell, hkl_obs, _, d_obs = _observed(core)
    d_min, d_max = d_obs.min().item(), d_obs.max().item()

    # Generate with slack, then cut the range on the ASU indices with the same
    # d computation as the observed ones, so a reflection on the boundary is
    # judged identically on both sides.
    possible = generate_possible_hkl(cell, d_min * (1 - 1e-6))
    possible = possible[~group.is_absent(possible).to(possible.device)]
    possible = group.canonicalize_hkl(possible.int(), include_friedel=True, sort=False)[0]
    possible = torch.unique(possible, dim=0)
    d_possible = get_d_spacing(possible.double(), cell)
    in_range = (d_possible >= d_min) & (d_possible <= d_max)
    possible, d_possible = possible[in_range], d_possible[in_range]

    edges = _shell_edges(d_obs, n_bins)
    observed = torch.isin(_key(hkl_obs), _key(possible))
    n_seen = torch.bincount(_assign(d_obs[observed], edges), minlength=n_bins)
    n_possible = torch.bincount(_assign(d_possible, edges), minlength=n_bins)
    return (n_seen.double() / n_possible.clamp(min=1).double()).float().cpu()


def multiplicity(core: SFXStreamCore, n_bins: int = 20) -> torch.Tensor:
    """Resolution-binned mean multiplicity: observations per observed unique reflection.

    Returns ``(n_bins,)``, lowest resolution first; a shell with no observation
    reads 0.
    """
    _, _, _, n_obs, d = _observed(core)
    shell = _assign(d, _shell_edges(d, n_bins))
    total = torch.bincount(shell, weights=n_obs.double(), minlength=n_bins)
    unique = torch.bincount(shell, minlength=n_bins)
    return (total / unique.clamp(min=1).double()).float().cpu()
