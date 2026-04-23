#!/das/work/units/LBR-FEL/p17490/Peter/torchsx_bg_anchor/.venv/bin/python -u
"""Sweep the weak log-I₀ floor prior to see where dark HKLs end up.

With the bg anchor active, reflections that look like pure background already
contribute ~0 gradient to log I₀.  Adding a weak Gaussian prior centred at a
very small log_I0_floor_target pulls those HKLs toward 0 while leaving strong
HKLs at their data MLE.

Reports the I₀ distribution and CC½ / R_split for each λ.
"""

from __future__ import annotations

from pathlib import Path
import math

import torch

from torchsx.stream import SFXStream
from torchsx.merge import match_halves, cc_half, r_split
from torchsx.merging_clean import Merger, MergeConfig, ScaleBCorrection

STREAM_DIR = Path("/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams")
OUTDIR = Path(__file__).parent
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

LAM_VALUES = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0]
TARGET = -11.5  # log(1e-5)


def fit_one(stream, lam: float):
    cfg = MergeConfig(
        min_obs=5,
        lam_I0_floor=lam,
        log_I0_floor_target=TARGET,
        max_iter=150,
        history_size=10,
    )
    merger = Merger(stream, config=cfg, corrections=[ScaleBCorrection], device=device)
    return merger.merge()


results: list[dict] = []
for lam in LAM_VALUES:
    print(f"\n── lam_I0_floor = {lam}  target={TARGET} ──", flush=True)
    r1 = fit_one(h1, lam)
    r2 = fit_one(h2, lam)

    F1 = r1.I0.clamp(min=0).sqrt()
    F2 = r2.I0.clamp(min=0).sqrt()
    m = match_halves(r1.hkl, F1, r2.hkl, F2, CELL, d_min=D_MIN, d_max=D_MAX)
    cc = cc_half(m); rs = r_split(m)

    I0 = r1.I0
    log_I0 = torch.log(I0.clamp(min=1e-12))
    q = lambda p: torch.quantile(I0, p).item()
    row = dict(
        lam=lam,
        cc_half=cc, r_split=rs,
        med_alpha=r1.alpha.median().item(),
        med_beta=r1.beta.median().item(),
        min_I0=I0.min().item(),
        p001=q(0.001), p01=q(0.01), p10=q(0.10),
        p50=q(0.50), p90=q(0.90), p99=q(0.99),
        max_I0=I0.max().item(),
        std_logI0=log_I0.std().item(),
    )
    results.append(row)
    print(
        f"  CC½={cc:.4f}  R={rs:.4f}  "
        f"med(α)={row['med_alpha']:.3f}  med(β)={row['med_beta']:.3f}  ",
        flush=True,
    )
    print(
        f"  I₀ quantiles: min={row['min_I0']:.2e}  0.1%={row['p001']:.2e}  "
        f"1%={row['p01']:.2e}  10%={row['p10']:.2e}  50%={row['p50']:.2f}  "
        f"90%={row['p90']:.1f}  99%={row['p99']:.1f}  max={row['max_I0']:.0f}  "
        f"std(log)={row['std_logI0']:.2f}",
        flush=True,
    )

print("\n" + "=" * 130)
print(f"  {'lam':>6s}  {'CC½':>7s}  {'R':>7s}  {'med α':>6s}  {'med β':>6s}  "
      f"{'min I₀':>10s}  {'0.1%':>10s}  {'1%':>10s}  {'10%':>8s}  "
      f"{'50%':>6s}  {'std logI':>9s}")
print("  " + "-" * 120)
for r in results:
    print(
        f"  {r['lam']:>6.3f}  {r['cc_half']:>7.4f}  {r['r_split']:>7.4f}  "
        f"{r['med_alpha']:>6.3f}  {r['med_beta']:>6.3f}  "
        f"{r['min_I0']:>10.2e}  {r['p001']:>10.2e}  {r['p01']:>10.2e}  "
        f"{r['p10']:>8.3f}  {r['p50']:>6.2f}  {r['std_logI0']:>9.3f}"
    )

torch.save(results, OUTDIR / "sweep_I0_floor_results.pt")
print(f"\nSaved to sweep_I0_floor_results.pt", flush=True)
