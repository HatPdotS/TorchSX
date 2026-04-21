#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -p gpu
#SBATCH -t 2:00:00
#SBATCH -o /das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/benchmark_refactored_genpoiss/sweep_sig_lo.out

"""Screen the quadrature-grid lower bound `sig_lo` on the Merger pipeline.

For each sig_lo in {1e-4 .. 1}, fit half1 and half2 and report CC½, R_split,
med(α), med(β), and std(log I₀).  Compares to the baseline (adaptive
per-HKL Poisson floor) and to EM Poisson saved earlier.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
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

# ── Load + split ────────────────────────────────────────────
streams = [SFXStream.load(STREAM_DIR / f"delay_{d}.sfx") for d in DELAYS]
s = SFXStream.concat(streams)
h1, h2 = s.split_random(n_parts=2, seed=SEED)
print(f"{s.n_crystals} crystals → halves {h1.n_crystals}/{h2.n_crystals}", flush=True)

# Sweep values (log-spaced) plus None (= current default)
SIG_LO_VALUES = [None, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]


def fit_one(stream, sig_lo):
    cfg = MergeConfig(
        min_obs=5,
        lam_alpha_smooth=50.0,
        lam_beta_smooth=500.0,
        lam_alpha_anchor=100.0,
        lam_beta_anchor=1_000.0,
        max_iter=150,
        history_size=10,
        obs_chunk_size=1_000_000,
        sig_lo_override=sig_lo,
    )
    merger = Merger(stream, config=cfg, corrections=[ScaleBCorrection], device=device)
    return merger.merge()


results: list[dict] = []
for sl in SIG_LO_VALUES:
    label = "adaptive" if sl is None else f"{sl:.0e}"
    print(f"\n── sig_lo = {label} ──", flush=True)

    r1 = fit_one(h1, sl)
    r2 = fit_one(h2, sl)

    F1 = r1.I0.clamp(min=0).sqrt()
    F2 = r2.I0.clamp(min=0).sqrt()
    m = match_halves(r1.hkl, F1, r2.hkl, F2, CELL, d_min=D_MIN, d_max=D_MAX)
    cc = cc_half(m)
    rs = r_split(m)

    log_I0 = torch.log(r1.I0.clamp(min=1e-6))
    std_logI = log_I0.std().item()

    # What fraction of HKLs have I₀ at or below the sig_lo floor (proxy for
    # compression)?
    if sl is not None:
        frac_floor = (r1.I0 <= 2 * sl).float().mean().item()
    else:
        frac_floor = float("nan")

    row = dict(
        sig_lo=label,
        cc_half=cc,
        r_split=rs,
        med_alpha=r1.alpha.median().item(),
        med_beta=r1.beta.median().item(),
        med_I0=r1.I0.median().item(),
        std_logI0=std_logI,
        frac_near_floor=frac_floor,
        n_matched=m.F1.numel(),
    )
    results.append(row)
    print(
        f"  CC½={cc:.4f}  R_split={rs:.4f}  "
        f"med(α)={row['med_alpha']:.3f}  med(β)={row['med_beta']:.3f}  "
        f"med(I₀)={row['med_I0']:.2f}  std(log I₀)={std_logI:.3f}  "
        f"frac≤2·sig_lo={frac_floor:.3f}",
        flush=True,
    )


# ── Summary table ───────────────────────────────────────────
print("\n" + "=" * 95)
print(f"  {'sig_lo':>10s}  {'CC½':>7s}  {'R_split':>8s}  "
      f"{'med α':>6s}  {'med β':>6s}  {'med I₀':>8s}  "
      f"{'std log I₀':>11s}  {'≤2·floor':>8s}")
print("  " + "-" * 85)
for r in results:
    floor_str = "---" if np.isnan(r["frac_near_floor"]) else f"{r['frac_near_floor']:.3f}"
    print(
        f"  {r['sig_lo']:>10s}  {r['cc_half']:>7.4f}  {r['r_split']:>8.4f}  "
        f"{r['med_alpha']:>6.3f}  {r['med_beta']:>6.3f}  "
        f"{r['med_I0']:>8.2f}  {r['std_logI0']:>11.3f}  {floor_str:>8s}"
    )

torch.save(results, OUTDIR / "sweep_sig_lo_results.pt")
print(f"\nSaved results to sweep_sig_lo_results.pt", flush=True)
