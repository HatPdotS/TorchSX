"""What gets written out besides the loss.

Each group answers a question the loss alone cannot, and each is computed from
tensors the training step already produces — none needs an extra pass.

A. Loss decomposition — is one term paying for another?
B. Calibration — is the predicted uncertainty honest?
C. Scale quality — is the network learning a scale at all?
D. Leakage — is the number real, or an artefact of the setup?
E. Merge quality — does it beat the baseline?  (epoch level, see the trainer)
F. Optimisation health.

The one that decides the project is ``between_frac`` in group C.  A per-crystal
G model puts 100 % of its ``log K`` variance between crystals.  The entire
premise of amortising is that the *within*-crystal part — partiality — is real
and predictable from a leave-one-out context.  If ``within`` stays near zero the
network has merely rediscovered G, and there is no reason to prefer it.

The one that protects the science is ``condition_auc`` in group D.  Any degree
of freedom that can separate light from dark absorbs the difference signal, and
would do so while every other metric improved.
"""

from __future__ import annotations

import datetime
import json
import math
import os
from pathlib import Path

import torch
from torch import Tensor

__all__ = [
    "calibration_metrics",
    "scale_metrics",
    "hkl_permutation_sensitivity",
    "hkl_permutation_sensitivity_shape",
    "condition_auc",
    "spearman",
    "grad_norms",
    "MetricLogger",
]


def _rank(x: Tensor) -> Tensor:
    """Average-tie-free ranks (ordinal); adequate for monitoring."""
    order = x.argsort()
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(x.numel(), device=x.device)
    return ranks.float()


def spearman(a: Tensor, b: Tensor) -> float:
    """Rank correlation.  Used to ask *where* the predicted scale comes from."""
    if a.numel() < 3:
        return float("nan")
    ra, rb = _rank(a.flatten()), _rank(b.flatten())
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = ra.norm() * rb.norm()
    return float((ra @ rb / denom).item()) if denom > 0 else float("nan")


def _binned(values: Tensor, bin_id: Tensor, n_bins: int, fn) -> list[float]:
    out = []
    for b in range(n_bins):
        m = bin_id == b
        out.append(float(fn(values[m]).item()) if int(m.sum()) > 8 else float("nan"))
    return out


# ── B. calibration ─────────────────────────────────────────────────────


def calibration_metrics(
    z: Tensor,
    *,
    shell_id: Tensor | None = None,
    n_shells: int = 10,
    i_over_sigma: Tensor | None = None,
) -> dict:
    """Health of the normalised residual ``z = (y - ybar) / tau``.

    ``var_z`` is the primary number and its target is exactly 1.0.  It pins the
    *level* of the scatter as a constraint rather than a fitted parameter,
    which is what both known failures were missing — abismal's ``Sdfac``
    running to 4.75 and torchsx's own ``g1 ~ 14`` are the same defect: one
    global number forced to describe every observation when scale determinacy
    varies >20x across crystals.

    ``frac_gt3`` against the Gaussian 0.27 % is the heavy-tail monitor.  A tail
    is what collapsed abismal until ``-t 32``: 0.19 % of reflections held
    46.9 % of the chi-squared.  Watch it from step 0, not after divergence.

    ``kurtosis`` decides whether the predicted ``beta`` is earning its keep: a
    Gaussian is 0, a Laplace +3.  With ``beta`` predicted this is a direct
    readout rather than a deferred question.

    Per-shell / per-decile spread matters more than the aggregate: *flat*
    miscalibration is a level error, whereas miscalibration that **trends**
    with a covariate is a model-form error and names the missing feature.
    """
    finite = torch.isfinite(z)
    z = z[finite]
    if z.numel() < 8:
        return {"var_z": float("nan"), "n": int(z.numel())}

    zc = z - z.mean()
    var = float((zc**2).mean().item())
    m4 = float((zc**4).mean().item())
    out = {
        "n": int(z.numel()),
        "var_z": var,
        "mean_z": float(z.mean().item()),
        "frac_gt3": float((z.abs() > 3).float().mean().item()),
        "frac_gt5": float((z.abs() > 5).float().mean().item()),
        "kurtosis": m4 / max(var**2, 1e-12) - 3.0,
    }
    if shell_id is not None and shell_id.numel() == finite.numel():
        out["var_z_by_shell"] = _binned(
            z, shell_id[finite], n_shells, lambda v: v.var()
        )
    if i_over_sigma is not None and i_over_sigma.numel() == finite.numel():
        ios = i_over_sigma[finite]
        dec = (_rank(ios) / max(ios.numel(), 1) * 10).long().clamp(max=9)
        out["var_z_by_isigma_decile"] = _binned(z, dec, 10, lambda v: v.var())
    return out


# ── C. is the network learning a scale? ────────────────────────────────


def _diagnostic_quantile(values: Tensor, q: float) -> Tensor:
    """Compute a scalar linear quantile beyond torch.quantile's size limit."""
    if values.numel() <= 2**24:
        return values.quantile(q)
    values = values.flatten().sort().values
    rank = q * (values.numel() - 1)
    lower, upper = math.floor(rank), math.ceil(rank)
    lo = values[lower]
    result = lo if lower == upper else lo.lerp(
        values[upper], rank - lower
    )
    return torch.where(values.isnan().any(), result.new_full((), float("nan")), result)


def scale_metrics(
    log_K: Tensor,
    crystal_id: Tensor,
    n_crystals: int,
    *,
    features: dict[str, Tensor] | None = None,
    baseline_log_K: Tensor | None = None,
) -> dict:
    """Spread, structure and provenance of the predicted scale.

    ``between_frac`` is the decisive one — see the module docstring.  A
    per-crystal G+B model cannot produce within-crystal variance at fixed
    resolution, so any ``within_frac`` above noise is signal that the classic
    parameterisation could not have captured.
    """
    log_K = log_K.detach().float()
    out = {
        "sd_log_K": float(log_K.std().item()),
        "mean_log_K": float(log_K.mean().item()),
        "p01_log_K": float(_diagnostic_quantile(log_K, 0.01).item()),
        "p99_log_K": float(_diagnostic_quantile(log_K, 0.99).item()),
    }

    # variance decomposition: between-crystal vs within-crystal
    counts = torch.bincount(crystal_id, minlength=n_crystals).float()
    sums = torch.zeros(n_crystals, device=log_K.device).index_add_(0, crystal_id, log_K)
    means = sums / counts.clamp(min=1)
    within = log_K - means[crystal_id]
    seen = counts > 0
    grand = log_K.mean()
    between_ss = float((counts[seen] * (means[seen] - grand) ** 2).sum().item())
    within_ss = float((within**2).sum().item())
    total = between_ss + within_ss
    out["between_frac"] = between_ss / total if total > 0 else float("nan")
    out["within_frac"] = within_ss / total if total > 0 else float("nan")
    out["sd_log_K_within"] = float(within.std().item())

    if baseline_log_K is not None:
        a = log_K - log_K.mean()
        b = baseline_log_K.detach().float()
        b = b - b.mean()
        d = a.norm() * b.norm()
        out["cc_vs_baseline"] = float((a @ b / d).item()) if d > 0 else float("nan")

    if features:
        # Where does the scale come from?  Also the interpretability check:
        # the predicted width should track ewald_offset, which is the
        # partiality signal --fix-profile-radius erased from the geometry.
        out["spearman_log_K"] = {k: spearman(log_K, v) for k, v in features.items()}
    return out


# ── D. leakage and degeneracy ──────────────────────────────────────────


@torch.no_grad()
def hkl_permutation_sensitivity(scale, meta, *, seed: int = 0) -> float:
    """Fraction of ``log K`` variance that moves when hkl labels are permuted.

    A scale that depends on hkl is degenerate with ``I0`` and is the **one
    channel that can manufacture a difference peak**.  It was 0.3 % in abismal.
    Run this every epoch, not once: if it grows as the model strengthens, drop
    hkl from the features before any light/dark work.

    Only the **self** branch is permuted.  Once the context carries
    ``obs_scale = n'/I0``, permuting the whole meta also permutes the *peers'*
    I0 lookup, and the monitor then reports legitimate use of per-reflection I0
    as if it were leakage (measured: 0.33 on a model with no hkl feature at
    all).  The degeneracy that matters is whether ``log_K[j]`` responds to
    ``j``'s *own* reflection identity, which is a self-branch property.
    """
    import dataclasses

    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(meta.hkl_idx.numel(), generator=g).to(meta.hkl_idx.device)
    shuffled = dataclasses.replace(meta, hkl_idx=meta.hkl_idx[perm])

    if hasattr(scale, "predict_split"):
        base = scale.predict_split(meta, meta).log_K
        moved = scale.predict_split(shuffled, meta).log_K
    else:
        base = scale.predict(meta).log_K
        moved = scale.predict(shuffled).log_K

    denom = float(base.var().item())
    numer = float((moved - base).var().item())
    if denom <= 1e-12:
        # A constant log_K has no hkl dependence by definition.  Returning NaN
        # here makes the guard silently un-assertable in exactly the case where
        # it should read zero (e.g. an untrained or collapsed model).
        return 0.0 if numer <= 1e-12 else float("inf")
    return numer / denom


@torch.no_grad()
def hkl_permutation_sensitivity_shape(scale, meta, *, seed: int = 0) -> float:
    """The same self-branch permutation test for the predicted scatter shape.

    Largest variance fraction that moves, over ``log w`` and every column of
    the packed mixture ``mix`` (when present).  A shape predicted from the
    embedding inherits the feature contract, but a width or tail that tracks a
    reflection's own identity would still let hkl leak into the merge weights,
    so this is watched alongside :func:`hkl_permutation_sensitivity`.
    """
    import dataclasses

    if not hasattr(scale, "predict_split"):
        return float("nan")
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(meta.hkl_idx.numel(), generator=g).to(meta.hkl_idx.device)
    shuffled = dataclasses.replace(meta, hkl_idx=meta.hkl_idx[perm])
    base = scale.predict_split(meta, meta)
    moved = scale.predict_split(shuffled, meta)
    cols = [(base.log_w, moved.log_w)]
    if base.mix is not None:
        cols += [(base.mix[:, j], moved.mix[:, j]) for j in range(base.mix.shape[1])]
    worst = 0.0
    for b, m in cols:
        denom, numer = float(b.var().item()), float((m - b).var().item())
        if denom > 1e-12:
            worst = max(worst, numer / denom)
        elif numer > 1e-12:
            return float("inf")
    return worst


def condition_auc(log_K: Tensor, is_light: Tensor) -> float:
    """AUC of ``log K`` as a light/dark classifier.  **Must be 0.5.**

    This is the assertion the whole difference experiment rests on.  A scale
    that can tell the conditions apart will absorb the difference signal, and
    it will do so while CC1/2, R_split and the calibration all *improve*.
    """
    log_K = log_K.detach().float().flatten()
    is_light = is_light.bool().flatten()
    n_pos = int(is_light.sum())
    n_neg = int((~is_light).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _rank(log_K) + 1.0
    return float(((r[is_light].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item())


# ── F. optimisation health ─────────────────────────────────────────────


def grad_norms(module: torch.nn.Module, groups: tuple[str, ...] = ("encoder", "head")) -> dict:
    """Per-group and **per-depth** gradient norms, plus a NaN/inf counter.

    The per-depth breakdown is what matters once the encoder is a deep stack of
    exchange blocks: an aggregate encoder norm cannot distinguish "healthy" from
    "the last two blocks are learning and block 0 sees nothing".  Blocks are
    identified by the integer following ``blocks.`` in the parameter name —
    ``context_blocks`` and ``self_blocks`` in the Deep Sets encoder.

    ``gradnorm_depth_ratio`` is last-block / first-block.  Values far above 1
    mean the early blocks are starved; note it is legitimately large at step 0,
    because the zero-initialised output projections make every block start as
    the identity and no gradient reaches the encoder until the heads move.
    """
    out: dict[str, float] = {}
    acc: dict[str, float] = {g: 0.0 for g in groups}
    per_depth: dict[int, float] = {}
    other = 0.0
    n_bad = 0
    for name, p in module.named_parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            n_bad += 1
            continue
        sq = float((p.grad.detach() ** 2).sum().item())
        if "blocks." in name:
            try:
                d = int(name.split("blocks.")[1].split(".")[0])
                per_depth[d] = per_depth.get(d, 0.0) + sq
            except (IndexError, ValueError):
                pass
        for g in groups:
            if g in name:
                acc[g] += sq
                break
        else:
            other += sq

    if per_depth:
        depths = sorted(per_depth)
        norms = [math.sqrt(per_depth[d]) for d in depths]
        out["gradnorm_by_depth"] = norms
        first, last = norms[0], norms[-1]
        out["gradnorm_depth_ratio"] = last / first if first > 0 else float("inf")
    for g, v in acc.items():
        out[f"gradnorm_{g}"] = math.sqrt(v)
    out["gradnorm_other"] = math.sqrt(other)
    out["n_nonfinite_grads"] = n_bad
    return out


# ── sink ───────────────────────────────────────────────────────────────


class MetricLogger:
    """JSONL sink, one line per logged step, plus a printed table.

    Matches the repo's existing convention (JSONL/CSV + PNG + a table into the
    SLURM ``.out``); there is no tensorboard or wandb anywhere in torchsx and
    the compute nodes have no outbound network.
    """

    def __init__(self, outdir: str | Path, *, filename: str = "metrics.jsonl") -> None:
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.path = self.outdir / filename
        self._fh = self.path.open("a", buffering=1)
        # Append mode keeps history across reruns, which is wanted — but
        # without a marker the records silently interleave, and an analysis
        # that does not filter will mix a buggy run with its replacement.
        # That has happened twice.  Stamp every record with a run id.
        self.run_id = os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}"
        self._fh.write(json.dumps({
            "phase": "run_start", "run_id": self.run_id,
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
        }) + "\n")

    def log(self, record: dict) -> None:
        if self._fh.closed:
            self._fh = self.path.open("a", buffering=1)
        record = {**record, "run_id": self.run_id}
        self._fh.write(json.dumps(_jsonable(record)) + "\n")

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "MetricLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, Tensor):
        return _jsonable(o.detach().cpu().tolist())
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o
