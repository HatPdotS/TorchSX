#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u
"""Scale all regularisation strengths by a multiplier and watch where (α, β)
plateau.

All reg weights (Dirichlet smoothing, anchors, σ-Cauchy) are multiplied by
the same scalar `m`.  As m → 0 the data-MLE emerges; as m → ∞ the priors
dominate.  A plateau in (α, β) vs m means the data-MLE is well-identified
and small changes in reg strength don't move the estimate.
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


# Multipliers applied uniformly to every reg weight
MULTIPLIERS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

# Reference "m=1" settings — same as current default baseline
BASE = dict(
    lam_alpha_smooth=500.0,
    lam_beta_smooth=500.0,
    lam_alpha_anchor=0.0,
    lam_beta_anchor=0.0,
    lam_sigma_cauchy=50.0,
    sigma_cauchy_scale=1.0,
)


def fit(stream, m: float):
    cfg = MergeConfig(
        min_obs=5,
        lam_alpha_smooth=BASE["lam_alpha_smooth"] * m,
        lam_beta_smooth=BASE["lam_beta_smooth"] * m,
        lam_alpha_anchor=BASE["lam_alpha_anchor"] * m,
        lam_beta_anchor=BASE["lam_beta_anchor"] * m,
        lam_sigma_cauchy=BASE["lam_sigma_cauchy"] * m,
        sigma_cauchy_scale=BASE["sigma_cauchy_scale"],
        sig_lo_override=1e-4,
        max_iter=150,
        history_size=10,
        obs_chunk_size=1_000_000,
    )
    merger = Merger(stream, config=cfg, corrections=[ScaleBCorrection], device=device)
    return merger.merge()


def width_sigma(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """σ = α · √(Γ(3/β)/Γ(1/β))."""
    log_c = 0.5 * (torch.lgamma(3.0 / beta) - torch.lgamma(1.0 / beta))
    return alpha * torch.exp(log_c)


results: list[dict] = []
for m in MULTIPLIERS:
    print(f"\n── m = {m} ──", flush=True)
    r1 = fit(h1, m)
    r2 = fit(h2, m)

    F1 = r1.I0.clamp(min=0).sqrt()
    F2 = r2.I0.clamp(min=0).sqrt()
    match = match_halves(r1.hkl, F1, r2.hkl, F2, CELL, d_min=D_MIN, d_max=D_MAX)
    cc = cc_half(match)
    rs = r_split(match)

    sigma1 = width_sigma(r1.alpha, r1.beta)
    std_logI0 = torch.log(r1.I0.clamp(min=1e-8)).std().item()

    row = dict(
        m=m,
        cc_half=cc,
        r_split=rs,
        med_alpha=r1.alpha.median().item(),
        med_beta=r1.beta.median().item(),
        med_sigma=sigma1.median().item(),
        med_I0=r1.I0.median().item(),
        min_I0=r1.I0.min().item(),
        std_logI0=std_logI0,
    )
    results.append(row)
    print(
        f"  CC½={cc:.4f}  R_split={rs:.4f}  "
        f"med(α)={row['med_alpha']:.3f}  med(β)={row['med_beta']:.3f}  "
        f"med(σ)={row['med_sigma']:.3f}  "
        f"min(I₀)={row['min_I0']:.4f}  med(I₀)={row['med_I0']:.2f}  "
        f"std(log I₀)={std_logI0:.3f}",
        flush=True,
    )


print("\n" + "=" * 110)
print(f"  {'m':>5s}  {'CC½':>7s}  {'R_split':>8s}  "
      f"{'med α':>6s}  {'med β':>6s}  {'med σ':>6s}  "
      f"{'min I₀':>8s}  {'med I₀':>8s}  {'std logI₀':>10s}")
print("  " + "-" * 100)
for r in results:
    print(
        f"  {r['m']:>5.2f}  {r['cc_half']:>7.4f}  {r['r_split']:>8.4f}  "
        f"{r['med_alpha']:>6.3f}  {r['med_beta']:>6.3f}  {r['med_sigma']:>6.3f}  "
        f"{r['min_I0']:>8.4f}  {r['med_I0']:>8.2f}  {r['std_logI0']:>10.3f}"
    )

torch.save(results, OUTDIR / "sweep_reg_strength_results.pt")
print(f"\nSaved to sweep_reg_strength_results.pt", flush=True)
