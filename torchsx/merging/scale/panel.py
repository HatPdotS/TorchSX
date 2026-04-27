"""Per-panel gain offset (+ optional fs/ss linear gradient)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from torchsx.merging.observations import ObsInfo
from torchsx.merging.parameters import Parameters
from torchsx.merging.scale.base import ScaleModel


class PanelScale(ScaleModel):
    """Log-gain per panel, optionally with a linear fs/ss gradient.

    ``log K_j = gain[p_j]  ( + grad_fs[p_j]·fs_j + grad_ss[p_j]·ss_j )``
    """

    def __init__(
        self,
        n_panels: int,
        *,
        fit_gradient: bool = False,
        panel_prior: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_panels = int(n_panels)
        self.fit_gradient = bool(fit_gradient)
        self.panel_prior = float(panel_prior)
        self.gain = nn.Parameter(torch.zeros(n_panels))
        if fit_gradient:
            self.grad_fs = nn.Parameter(torch.zeros(n_panels))
            self.grad_ss = nn.Parameter(torch.zeros(n_panels))
        else:
            self.register_parameter("grad_fs", None)
            self.register_parameter("grad_ss", None)

    @classmethod
    def from_obs_info(
        cls,
        obs_info: ObsInfo,
        *,
        fit_gradient: bool = False,
        panel_prior: float = 1.0,
    ) -> "PanelScale":
        if obs_info.panel_id is None:
            raise ValueError("PanelScale requires obs_info.panel_id")
        n_panels = int(obs_info.panel_id.max().item()) + 1
        return cls(n_panels=n_panels, fit_gradient=fit_gradient, panel_prior=panel_prior)

    def log_k(
        self,
        params: Parameters,
        meta: ObsInfo,
        obs_range: tuple[int, int] | None = None,
    ) -> Tensor:
        del params, obs_range
        pid = meta.panel_id
        assert pid is not None, "PanelScale requires obs_info.panel_id"
        out = self.gain[pid]
        if self.grad_fs is not None:
            out = out + self.grad_fs[pid] * meta.fs + self.grad_ss[pid] * meta.ss
        return out

    def regularizer(self, params: Parameters) -> Tensor:
        del params
        reg = self.panel_prior * (self.gain * self.gain).sum()
        if self.grad_fs is not None:
            reg = reg + self.panel_prior * (self.grad_fs * self.grad_fs).sum()
            reg = reg + self.panel_prior * (self.grad_ss * self.grad_ss).sum()
        return reg
