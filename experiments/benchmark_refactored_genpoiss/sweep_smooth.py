#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Sweep Dirichlet smoothing strength on α and β with σ-Cauchy DISABLED.

Starting point was the pure data-MLE run (λ_σ=0, Dirichlet=500): CC½=0.987,
R_split=0.082, (α,β)=(3.2, 4.3).  Increasing Dirichlet should further
stabilise per-HKL noise without biasing the median (smoothing is
location-invariant), hopefully tightening R_split without losing CC.
"""

from __future__ import annotations

from pathlib import Path

import torch

from torchsx.stream import SFXStream
from torchsx.merge import match_halves, cc_half, r_split
from torchsx.merging_clean import Merger, MergeConfig, ScaleBCorrection

STREAM_DIR = Path("/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams")
OUTDIR = Path("/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/benchmark_refactored_genpoiss")
DELAYS = [-91, -136, -182, -227]
CELL = torch.tensor([14.97, 18.85, 18.89, 89.37, 84.94, 67.84], dtype=torch.float32)
D_MIN, D_MAX = 0.9, 10.0
SEED = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}", flush=True)

streams = [SFXStream.load(STREAM_DIR / f"delay_{d}.sfx") for d in DELAYS]
s = SFXStream.concat(streams)
h1, h2 = s.split_random(n_parts=2, seed=SEED)
print(f"{s.n_crystals} crystals → halves {h1.n_crystals}/{h2.n_crystals}", flush=True)


# Sweep Dirichlet strength.  500 = current 10× rule baseline.
SMOOTH_VALUES = [100.0, 250.0, 500.0, 1_000.0, 2_500.0, 5_000.0, 10_000.0, 25_000.0]


def fit_one(stream, smooth: float):
    cfg = MergeConfig(
        min_obs=5,
        # Vary Dirichlet (same for α and β)
        lam_alpha_smooth=smooth,
        lam_beta_smooth=smooth,
        # Anchors off, σ-Cauchy off — isolates the smoothing effect
        lam_alpha_anchor=0.0,
        lam_beta_anchor=0.0,
        lam_sigma_cauchy=0.0,
        sig_lo_override=1e-4,
        max_iter=150,
        history_size=10,
        obs_chunk_size=1_000_000,
    )
    merger = Merger(stream, config=cfg, corrections=[ScaleBCorrection], device=device)
    result = merger.merge()
    # Capture adjacency for neighbour-diff metric
    edge_src = merger.loss.edge_src.cpu()
    edge_dst = merger.loss.edge_dst.cpu()
    return result, edge_src, edge_dst


def width_sigma(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    log_c = 0.5 * (torch.lgamma(3.0 / beta) - torch.lgamma(1.0 / beta))
    return alpha * torch.exp(log_c)


def neighbour_rmsd(values_per_hkl: torch.Tensor, edge_src: torch.Tensor,
                   edge_dst: torch.Tensor) -> float:
    """Short-range roughness: RMS of (value_i - value_j) over ±1 Miller edges.
    Global trends don't register here; only local fluctuation does."""
    diff = values_per_hkl[edge_src] - values_per_hkl[edge_dst]
    return float(torch.sqrt((diff * diff).mean()))


results: list[dict] = []
for sm in SMOOTH_VALUES:
    print(f"\n── lam_smooth = {sm} (α and β) ──", flush=True)
    r1, es1, ed1 = fit_one(h1, sm)
    r2, _, _ = fit_one(h2, sm)

    F1 = r1.I0.clamp(min=0).sqrt()
    F2 = r2.I0.clamp(min=0).sqrt()
    match = match_halves(r1.hkl, F1, r2.hkl, F2, CELL, d_min=D_MIN, d_max=D_MAX)
    cc = cc_half(match)
    rs = r_split(match)

    sigma1 = width_sigma(r1.alpha, r1.beta)
    # Local roughness: RMS of neighbour differences in log-space
    log_a = torch.log(r1.alpha.clamp(min=1e-6))
    log_b = torch.log(r1.beta.clamp(min=1e-6))
    rms_log_a = neighbour_rmsd(log_a, es1, ed1)
    rms_log_b = neighbour_rmsd(log_b, es1, ed1)
    row = dict(
        smooth=sm,
        cc_half=cc,
        r_split=rs,
        med_alpha=r1.alpha.median().item(),
        rms_dlog_alpha=rms_log_a,
        med_beta=r1.beta.median().item(),
        rms_dlog_beta=rms_log_b,
        med_sigma=sigma1.median().item(),
        med_I0=r1.I0.median().item(),
        min_I0=r1.I0.min().item(),
        std_logI0=torch.log(r1.I0.clamp(min=1e-8)).std().item(),
    )
    results.append(row)
    print(
        f"  CC½={cc:.4f}  R_split={rs:.4f}  "
        f"med(α)={row['med_alpha']:.3f}  rms(Δlog α)_nb={rms_log_a:.3f}  "
        f"med(β)={row['med_beta']:.3f}  rms(Δlog β)_nb={rms_log_b:.3f}  "
        f"med(σ)={row['med_sigma']:.3f}  "
        f"min(I₀)={row['min_I0']:.4f}  std(log I₀)={row['std_logI0']:.3f}",
        flush=True,
    )


print("\n" + "=" * 120)
print(f"  {'smooth':>8s}  {'CC½':>7s}  {'R_split':>8s}  "
      f"{'med α':>6s}  {'rmsΔα':>7s}  {'med β':>6s}  {'rmsΔβ':>7s}  "
      f"{'med σ':>6s}  {'min I₀':>8s}  {'std logI₀':>10s}")
print("  " + "-" * 110)
for r in results:
    print(
        f"  {r['smooth']:>8.0f}  {r['cc_half']:>7.4f}  {r['r_split']:>8.4f}  "
        f"{r['med_alpha']:>6.3f}  {r['rms_dlog_alpha']:>7.4f}  "
        f"{r['med_beta']:>6.3f}  {r['rms_dlog_beta']:>7.4f}  "
        f"{r['med_sigma']:>6.3f}  {r['min_I0']:>8.4f}  {r['std_logI0']:>10.3f}"
    )

torch.save(results, OUTDIR / "sweep_smooth_results.pt")
print(f"\nSaved to sweep_smooth_results.pt", flush=True)
