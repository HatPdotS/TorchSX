"""Known scale of an observation from the disk it was summed over.

The scale model and the sum-of-spots tether assume that every observation
captures the same fraction of its spot at a given resolution.  A source that
sums reflections over disks of different size breaks that: a larger disk
captures more of the spot, by a factor that jumps wherever the disk changes.
:class:`DiskCapture` holds that factor relative to one reference disk,
``flux(disk used) / flux(reference disk)``, as a function of resolution for each
disk size, and returns its logarithm per observation.  The scale model adds it
to ``log K``, so the network's own scale and the merged intensities stay on the
reference disk's scale.

The reference is the table's choice.  With the source's smallest disk as the
reference, only the larger disks carry a row and the smallest disk is left
alone (``base_pixels`` is its pixel count).  With the largest disk as the
reference, every disk carries a row of values at or below zero and
``base_pixels`` is 0: the merged intensities then follow the flux in the
largest disk at every resolution.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor

__all__ = ["DiskCapture"]


class DiskCapture:
    """Log capture ratio per observation, from the resolution and the disk's pixel count.

    Parameters
    ----------
    q
        ``(n_q,)`` knots in ``1/d`` (1/A), increasing.
    n_pixels
        ``(n_disk,)`` pixel counts of the full disks that carry a row,
        increasing, all above ``base_pixels``.
    log_ratio
        ``(n_disk, n_q)`` log of the captured flux relative to the reference
        disk at each knot; linear between knots and constant beyond them.
    base_pixels
        Observations with at most this many pixels get no offset: the pixel
        count of the smallest disk when it is the reference, 0 when every disk
        carries a row.

    An observation belongs to the smallest disk that holds its pixel count: a
    disk that lost pixels to a mask or to a neighbour counts as the next
    smaller one once it holds no more pixels than that one.
    """

    def __init__(self, q, n_pixels, log_ratio, base_pixels: float) -> None:
        self.q = torch.as_tensor(q, dtype=torch.float32).flatten()
        self.n_pixels = torch.as_tensor(n_pixels, dtype=torch.float32).flatten()
        self.log_ratio = torch.as_tensor(log_ratio, dtype=torch.float32).reshape(
            self.n_pixels.numel(), self.q.numel())
        self.base_pixels = float(base_pixels)
        if self.q.numel() < 1 or bool((self.q[1:] <= self.q[:-1]).any()):
            raise ValueError("capture knots must be increasing")
        if bool((self.n_pixels[1:] <= self.n_pixels[:-1]).any()) or (
                self.n_pixels.numel() and float(self.n_pixels[0]) <= self.base_pixels):
            raise ValueError("capture disks must be increasing and above base_pixels")

    @classmethod
    def from_definition(cls, d: dict) -> "DiskCapture":
        return cls(d["q"], d["n_pixels"], d["log_ratio"], d["base_pixels"])

    @classmethod
    def from_file(cls, path) -> "DiskCapture":
        return cls.from_definition(json.loads(Path(path).read_text()))

    def definition(self) -> dict:
        return {"q": self.q.tolist(), "n_pixels": self.n_pixels.tolist(),
                "log_ratio": self.log_ratio.tolist(), "base_pixels": self.base_pixels}

    def log_offset(self, s_sq: Tensor, n_pix: Tensor) -> Tensor:
        """``(n,)`` log capture ratio; ``s_sq = 1 / (4 d^2)``, ``n_pix`` the disk's pixels."""
        out = torch.zeros_like(s_sq, dtype=torch.float32)
        if self.n_pixels.numel() == 0:
            return out
        dev = s_sq.device
        knots, table = self.q.to(dev), self.log_ratio.to(dev)
        q = (2.0 * s_sq.float().clamp(min=0.0).sqrt()).clamp(float(knots[0]), float(knots[-1]))
        hi = torch.bucketize(q, knots).clamp(1, max(knots.numel() - 1, 1))
        if knots.numel() == 1:
            lo, t = torch.zeros_like(hi), torch.zeros_like(q)
            hi = lo
        else:
            lo = hi - 1
            t = (q - knots[lo]) / (knots[hi] - knots[lo])
        n = n_pix.float()
        disk = torch.bucketize(n, self.n_pixels.to(dev)).clamp(max=self.n_pixels.numel() - 1)
        value = table[disk, lo] * (1.0 - t) + table[disk, hi] * t
        return torch.where(n > self.base_pixels, value, out)
