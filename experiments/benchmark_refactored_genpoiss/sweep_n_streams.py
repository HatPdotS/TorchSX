#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=256G
#SBATCH -p gpu
#SBATCH -t 8:00:00
#SBATCH -o /das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/benchmark_refactored_genpoiss/sweep_n_streams.out

"""Data-size scan: fit with 1, 2, 4, 8, 16, 32, 64, 136 streams and track
how (α, β) behave as the per-HKL observation count grows.

Fixed config: Dirichlet=500 on α/β (rational 10× rule), no anchors, no
σ-Cauchy, sig_lo=1e-4.  Reg is N-scaled (reg_scale_ref_obs=2.2 M), so the
reg-to-data ratio is constant across dataset sizes.

Tracks also LBFGS convergence: n_closure_calls, max |grad| at exit,
final_data_nll.  If (α, β) plateau with N, the generative model is
self-consistent; otherwise we're seeing missing structure.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch

from torchsx.stream import SFXStream
from torchsx.merging_clean import Merger, MergeConfig, ScaleBCorrection

STREAM_DIR = Path(
    "/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams"
)
OUTDIR = Path(
    "/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/benchmark_refactored_genpoiss"
)

# Pick streams by filename order (time-delay order).  Always include the
# four original dark streams first so N=1..4 matches the existing benchmark.
ORIGINAL_DARK = ["delay_-91", "delay_-136", "delay_-182", "delay_-227"]


def stream_order() -> list[str]:
    all_sfx = sorted(STREAM_DIR.glob("*.sfx"))
    names = [p.stem for p in all_sfx]
    # Move dark streams to the front, preserving their order
    rest = [n for n in names if n not in ORIGINAL_DARK]
    return ORIGINAL_DARK + rest


STREAM_NAMES = stream_order()
print(f"Total available streams: {len(STREAM_NAMES)}", flush=True)

N_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, len(STREAM_NAMES)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)


def fit_streams(names: list[str], max_iter: int) -> dict:
    """Load `names`, concat, fit, return summary dict."""
    t0 = time.perf_counter()
    streams = [SFXStream.load(STREAM_DIR / f"{n}.sfx") for n in names]
    s = SFXStream.concat(streams) if len(streams) > 1 else streams[0]
    t_load = time.perf_counter() - t0

    # Memory scaling: quadrature buffers are O(N_obs · n_quad).  Drop n_quad
    # and chunk size for large datasets so the 40 GB A100 stays within budget.
    n_obs_est = sum(int(Path(STREAM_DIR / f"{nm}.sfx").stat().st_size / 50) for nm in names)
    if n_obs_est > 40_000_000:
        n_quad = 21
        chunk = 200_000
    elif n_obs_est > 10_000_000:
        n_quad = 31
        chunk = 300_000
    else:
        n_quad = 51
        chunk = 1_000_000

    cfg = MergeConfig(
        min_obs=5,
        lam_alpha_smooth=500.0,
        lam_beta_smooth=500.0,
        lam_alpha_anchor=0.0,
        lam_beta_anchor=0.0,
        lam_sigma_cauchy=0.0,
        sig_lo_override=1e-4,
        max_iter=max_iter,
        history_size=10,
        obs_chunk_size=chunk,
        n_quad=n_quad,
    )
    print(f"    (memory: n_quad={n_quad}, chunk={chunk})", flush=True)
    t1 = time.perf_counter()
    merger = Merger(s, config=cfg, corrections=[ScaleBCorrection], device=device)
    result = merger.merge()
    t_fit = time.perf_counter() - t1

    # Derived diagnostics
    beta = result.beta
    alpha = result.alpha
    log_c = 0.5 * (torch.lgamma(3.0 / beta) - torch.lgamma(1.0 / beta))
    sigma = alpha * torch.exp(log_c)

    I0 = result.I0
    log_I0 = torch.log(I0.clamp(min=1e-8))

    return dict(
        n_streams=len(names),
        n_crystals=s.n_crystals,
        n_obs=int(result.n_obs.sum().item()),
        n_valid_hkl=int(alpha.numel()),
        t_load=t_load,
        t_fit=t_fit,
        med_alpha=alpha.median().item(),
        med_beta=beta.median().item(),
        med_sigma=sigma.median().item(),
        med_I0=I0.median().item(),
        min_I0=I0.min().item(),
        std_logI0=log_I0.std().item(),
        # Convergence
        n_closure_calls=result.diagnostics.get("n_closure_calls"),
        max_grad=result.diagnostics.get("max_grad"),
        final_data_nll=result.diagnostics.get("final_data_nll"),
        final_reg=result.diagnostics.get("final_reg"),
    )


# Resume: load prior results and skip N values already done
RESULTS_FILE = OUTDIR / "sweep_n_streams_results.pt"
if RESULTS_FILE.exists():
    results_default: list[dict] = torch.load(RESULTS_FILE, weights_only=False)
    done_N = {r["n_streams"] for r in results_default}
    print(f"Resuming — already have N = {sorted(done_N)}", flush=True)
else:
    results_default = []
    done_N = set()

for n in N_SIZES:
    if n in done_N:
        continue
    names = STREAM_NAMES[:n]
    print(f"\n── N={n} streams ──", flush=True)
    try:
        row = fit_streams(names, max_iter=150)
    except Exception as exc:
        print(f"  FAILED: {exc}", flush=True)
        continue
    results_default.append(row)
    # Save progressively so an OOM at the next N doesn't lose this row
    torch.save(results_default, RESULTS_FILE)
    print(
        f"  N={n:>3d} streams  {row['n_crystals']:>5d} crystals  "
        f"{row['n_obs']/1e6:>5.2f}M obs  {row['n_valid_hkl']} HKLs  "
        f"(load {row['t_load']:.1f}s, fit {row['t_fit']:.1f}s)",
        flush=True,
    )
    print(
        f"    med(α)={row['med_alpha']:.3f}  med(β)={row['med_beta']:.3f}  "
        f"med(σ)={row['med_sigma']:.3f}  "
        f"min(I₀)={row['min_I0']:.4f}  std(log I₀)={row['std_logI0']:.3f}",
        flush=True,
    )
    print(
        f"    LBFGS: n_closure={row['n_closure_calls']}  "
        f"max|grad|={row['max_grad']:.2e}  "
        f"data_nll={row['final_data_nll']:.3e}",
        flush=True,
    )


# Summary table
print("\n" + "=" * 130)
print(f"  {'N':>3s}  {'obs (M)':>8s}  {'HKLs':>6s}  "
      f"{'med α':>6s}  {'med β':>6s}  {'med σ':>6s}  "
      f"{'min I₀':>8s}  {'std logI₀':>10s}  "
      f"{'n_cls':>6s}  {'|grad|':>8s}  {'fit (s)':>8s}")
print("  " + "-" * 120)
for r in results_default:
    print(
        f"  {r['n_streams']:>3d}  {r['n_obs']/1e6:>8.2f}  {r['n_valid_hkl']:>6d}  "
        f"{r['med_alpha']:>6.3f}  {r['med_beta']:>6.3f}  {r['med_sigma']:>6.3f}  "
        f"{r['min_I0']:>8.4f}  {r['std_logI0']:>10.3f}  "
        f"{r['n_closure_calls']:>6d}  {r['max_grad']:>8.2e}  {r['t_fit']:>8.1f}"
    )

torch.save(results_default, OUTDIR / "sweep_n_streams_results.pt")
print(f"\nSaved to sweep_n_streams_results.pt", flush=True)
