"""The raw-count quantities a set of observations offers the tether's terms.

Everything here comes from the counts and the crystal assignment, built once
per batch (or once per merge) and shared by every term.  The crystals are split
into parts alternately in the order of their ids -- not by the parity of the id
itself, which other structure in the data, such as states assigned in turn, may
share.  A (reflection, part) group is a reflection's observations in one part.

A covariate that describes a reflection must not share its noise with the
observations it is set against: a sample of fuller spots raises a reflection's
mean ``K`` and its plain mean together.  The strength covariate of a group is
therefore taken from the *other half* of the crystals.
"""

from __future__ import annotations

from functools import cached_property

import torch
from torch import Tensor

from torchsx.merging.tether.kmean import psd_solve

__all__ = ["TetherBatch"]


class TetherBatch:
    """One set of observations as the tether sees it.

    ``n_signal`` (``n - bg - offset``, the background the likelihood's mean
    uses) and ``var`` (its counting variance) per observation, ``log_K`` the
    scale prediction (in the graph during training), each observation's
    reflection and crystal, the smooth per-reflection basis ``phi`` (without
    its constant column) and a resolution shell per reflection.
    """

    def __init__(self, n_signal: Tensor, var: Tensor, log_K: Tensor, hkl_idx: Tensor,
                 crystal_id: Tensor, n_crystals: int, phi: Tensor, shell: Tensor, *, floor: float = 0.05):
        self.n_signal = n_signal.to(torch.float64)
        self.var = var.to(torch.float64)
        self.log_K = log_K
        self.hkl_idx = hkl_idx.long()
        self.crystal_id = crystal_id.long()
        self.n_crystals = int(n_crystals)
        self.phi = phi.to(device=n_signal.device, dtype=torch.float64)
        self.shell = shell.to(n_signal.device).long()
        self.floor = float(floor)

    @property
    def n_hkl(self) -> int:
        return self.phi.shape[0]

    @cached_property
    def crystal_rank(self) -> Tensor:
        return torch.unique(self.crystal_id, return_inverse=True)[1]

    def rows(self, n_parts: int) -> Tensor:
        """Each observation's (reflection, part) group: ``h + n_hkl * part``."""
        return self.hkl_idx + self.n_hkl * (self.crystal_rank % n_parts)

    def half_of_row(self, n_parts: int) -> Tensor:
        """The half-split group that each ``n_parts`` group lies in (parts alternate, so
        part ``q`` lies in half ``q % 2``)."""
        r = torch.arange(n_parts * self.n_hkl, device=self.n_signal.device)
        return r % self.n_hkl + self.n_hkl * ((r // self.n_hkl) % 2)

    def _sum(self, x: Tensor, rows: Tensor, n_rows: int) -> Tensor:
        return torch.zeros(n_rows, device=x.device, dtype=x.dtype).index_add_(0, rows, x)

    def groups(self, n_parts: int) -> dict:
        """Per (reflection, part) group, outside the graph: ``count``, the sum of spots
        ``I_lin = sum n_signal / sum K``, its counting SD ``sd`` and its relative SD
        ``rel_sd`` (both independent of the scale of ``K``)."""
        rows, n_rows = self.rows(n_parts), n_parts * self.n_hkl
        with torch.no_grad():
            count = self._sum(torch.ones_like(self.n_signal), rows, n_rows)
            signal = self._sum(self.n_signal, rows, n_rows)
            sum_K = self._sum(self.log_K.detach().to(torch.float64).exp(), rows, n_rows).clamp(min=1e-300)
            sd_sig = self._sum(self.var, rows, n_rows).clamp(min=0).sqrt()
        return {"rows": rows, "n_rows": n_rows, "count": count, "I_lin": signal / sum_K,
                "sd": sd_sig / sum_K, "rel_sd": sd_sig / signal.abs().clamp(min=1e-300)}

    @cached_property
    def strength(self) -> dict:
        """The cross-fitted strength covariate.

        A (reflection, half) group's strength is the log plain mean
        ``sum n_signal / N`` of the reflection over the *other* half -- free of
        ``K`` and of this half's counting noise -- floored at ``floor`` times
        that half's mean over the reflection's shell, then made orthogonal to a
        constant and the basis (weighted by the group's observations), so it
        carries strength and no resolution trend.  Returns per group ``x_row``
        (0 where the reflection lacks either half), ``valid`` and ``bright``
        (strength above its shell's median), and ``x`` per observation.
        """
        n_hkl, dev = self.n_hkl, self.n_signal.device
        rows = self.rows(2)
        with torch.no_grad():
            s = self._sum(self.n_signal, rows, 2 * n_hkl)
            c = self._sum(torch.ones_like(self.n_signal), rows, 2 * n_hkl)
            pm = s / c.clamp(min=1.0)
            n_sh = int(self.shell.max()) + 1 if self.shell.numel() else 1
            key = self.shell.repeat(2) * 2 + torch.arange(2, device=dev).repeat_interleave(n_hkl)
            has = c > 0
            tot = torch.zeros(2 * n_sh, device=dev, dtype=torch.float64).index_add_(0, key[has], pm[has])
            cnt = torch.zeros(2 * n_sh, device=dev, dtype=torch.float64).index_add_(
                0, key[has], torch.ones_like(pm[has]))
            fl = self.floor * (tot / cnt.clamp(min=1.0)).clamp(min=1e-12)
            other = torch.cat([torch.arange(n_hkl, 2 * n_hkl, device=dev), torch.arange(n_hkl, device=dev)])
            valid = has & has[other]
            x = torch.maximum(pm[other], fl[key[other]]).log()
            A = torch.cat([torch.ones(2 * n_hkl, 1, device=dev, dtype=torch.float64), self.phi.repeat(2, 1)], 1)
            wc = torch.where(valid, c, torch.zeros_like(c))
            coef = psd_solve((A * wc[:, None]).T @ A,
                             (A * wc[:, None]).T @ torch.where(valid, x, torch.zeros_like(x)))
            x_row = torch.where(valid, x - A @ coef, torch.zeros_like(x))
            bright = torch.zeros(2 * n_hkl, dtype=torch.bool, device=dev)
            sh_row = self.shell.repeat(2)
            for k in range(n_sh):
                m = valid & (sh_row == k)
                if bool(m.any()):
                    bright[m] = x_row[m] > x_row[m].median()
        return {"x": x_row.index_select(0, rows), "x_row": x_row, "valid": valid, "bright": bright}
