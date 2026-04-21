#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -p gpu
#SBATCH -t 2:00:00
#SBATCH -o /das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/benchmark_refactored_genpoiss/sweep_sigma_cauchy.out

"""Sweep the half-Cauchy shrinkage prior on the GenGauss width σ=α·c(β).

τ (scale) held fixed; λ varied.  Reports CC½, R_split, and summary stats
that diagnose whether the penalty is (a) tightening σ where the data has
little to say, (b) being evaded via β-drift (shouldn't, because σ is
α↔β-invariant), and (c) helping or hurting dynamic range.
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

LAM_VALUES = [0.0, 1.0, 5.0, 25.0, 50.0, 100.0, 250.0, 500.0]
TAU = 1.0  # what we call "innocent" width


def fit_one(stream, lam: float):
    # Keep Dirichlet smoothing at the principled 10× rule value (500).
    # Anchors off (as decided).  Vary only σ-Cauchy.
    cfg = MergeConfig(
        min_obs=5,
        lam_alpha_smooth=500.0,
        lam_beta_smooth=500.0,
        lam_alpha_anchor=0.0,
        lam_beta_anchor=0.0,
        lam_sigma_cauchy=lam,
        sigma_cauchy_scale=TAU,
        sig_lo_override=1e-4,
        max_iter=150,
        history_size=10,
        obs_chunk_size=1_000_000,
    )
    merger = Merger(stream, config=cfg, corrections=[ScaleBCorrection], device=device)
    return merger.merge()


results: list[dict] = []
for lam in LAM_VALUES:
    print(f"\n── lam_sigma_cauchy = {lam} (τ={TAU}) ──", flush=True)
    r1 = fit_one(h1, lam)
    r2 = fit_one(h2, lam)

    F1 = r1.I0.clamp(min=0).sqrt()
    F2 = r2.I0.clamp(min=0).sqrt()
    m = match_halves(r1.hkl, F1, r2.hkl, F2, CELL, d_min=D_MIN, d_max=D_MAX)
    cc = cc_half(m)
    rs = r_split(m)

    # Width stats
    beta = r1.beta
    alpha = r1.alpha
    import math
    c_beta = torch.exp(
        0.5 * (torch.lgamma(3.0 / beta) - torch.lgamma(1.0 / beta))
    )
    sigma = alpha * c_beta

    std_logI0 = torch.log(r1.I0.clamp(min=1e-6)).std().item()
    min_I0 = r1.I0.min().item()

    row = dict(
        lam=lam,
        cc_half=cc,
        r_split=rs,
        med_alpha=alpha.median().item(),
        med_beta=beta.median().item(),
        med_sigma=sigma.median().item(),
        med_I0=r1.I0.median().item(),
        min_I0=min_I0,
        std_logI0=std_logI0,
        n_matched=m.F1.numel(),
    )
    results.append(row)
    print(
        f"  CC½={cc:.4f}  R_split={rs:.4f}  "
        f"med(α)={row['med_alpha']:.3f}  med(β)={row['med_beta']:.3f}  "
        f"med(σ)={row['med_sigma']:.3f}  "
        f"min(I₀)={min_I0:.4f}  med(I₀)={row['med_I0']:.2f}  "
        f"std(log I₀)={std_logI0:.3f}",
        flush=True,
    )

# ── Summary ────────────────────────────────────────────────
print("\n" + "=" * 110)
print(f"  {'lam':>6s}  {'CC½':>7s}  {'R_split':>8s}  "
      f"{'med α':>6s}  {'med β':>6s}  {'med σ':>6s}  "
      f"{'min I₀':>8s}  {'med I₀':>8s}  {'std logI₀':>10s}")
print("  " + "-" * 100)
for r in results:
    print(
        f"  {r['lam']:>6.1f}  {r['cc_half']:>7.4f}  {r['r_split']:>8.4f}  "
        f"{r['med_alpha']:>6.3f}  {r['med_beta']:>6.3f}  {r['med_sigma']:>6.3f}  "
        f"{r['min_I0']:>8.4f}  {r['med_I0']:>8.2f}  {r['std_logI0']:>10.3f}"
    )

torch.save(results, OUTDIR / "sweep_sigma_cauchy_results.pt")
print(f"\nSaved to sweep_sigma_cauchy_results.pt", flush=True)
