#!/das/work/units/LBR-FEL/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -p gpu
#SBATCH -t 1:00:00
#SBATCH -o /das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/benchmark_refactored_genpoiss/bench_closure.out

"""Stage-0 closure benchmark for the unified merge loss.

Validates:
  1. data_nll matches a reference value at known parameters.
  2. Chunked == unchunked (loss + gradient).
  3. Gradient correctness against finite differences on a few params.
  4. Wall-clock + peak memory across several chunk sizes.

Runs on a SLURM A100 node, on the same 3 streams used in benchmark_final.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from torchsx.stream import SFXStream
from torchsx.merging_clean import MergeConfig, ScaleBCorrection
from torchsx.merging_clean.corrections import ObsInfo
from torchsx.merging_clean.loss import UnifiedMergeLoss
from torchsx.merging_clean.observations import prepare_observations

STREAM_DIR = Path(
    "/das/work/units/LBR-FEL/p17490/Peter/torchsx/experiments/sfx_scan/streams"
)
# Same 3 streams we have been using before (per user instruction)
DELAYS = [-91, -136, -182, -227]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)


# ─────────────────────────────────────────────────────────────
# Load and prepare
# ─────────────────────────────────────────────────────────────

print("── Loading streams ──", flush=True)
t0 = time.perf_counter()
streams = [SFXStream.load(STREAM_DIR / f"delay_{d}.sfx") for d in DELAYS]
stream = SFXStream.concat(streams)
print(
    f"  loaded in {time.perf_counter()-t0:.1f}s: "
    f"{stream.n_crystals} crystals, {len(stream._core.I)} observations",
    flush=True,
)

print("── prepare_observations ──", flush=True)
t0 = time.perf_counter()
config = MergeConfig(obs_chunk_size=0)  # unchunked by default; we'll vary later
prep = prepare_observations(stream, config)
prep_cpu = prep  # keep a copy on CPU for cross-chunk param copy
print(
    f"  prep in {time.perf_counter()-t0:.1f}s  "
    f"n_valid={prep.n_valid} n_obs={prep.n_obs_total}",
    flush=True,
)

prep = prep.to(device)

# Build a base loss (CPU params we will copy for each test)
corrections = [ScaleBCorrection.from_stream(stream, prep.obs_info).to(device)]
loss_ref = UnifiedMergeLoss(prep, corrections, config).to(device)


def fresh_loss() -> UnifiedMergeLoss:
    """Create a loss with identical parameters to loss_ref."""
    c = ScaleBCorrection.from_stream(stream, prep.obs_info).to(device)
    L = UnifiedMergeLoss(prep, [c], config).to(device)
    L.load_state_dict(loss_ref.state_dict())
    return L


# ─────────────────────────────────────────────────────────────
# 1. Forward pass numerically finite
# ─────────────────────────────────────────────────────────────

print("\n── 1. forward pass ──", flush=True)
L = fresh_loss()
with torch.no_grad():
    t0 = time.perf_counter()
    nll = L.data_nll(0, L.n_obs_total)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    reg = L.regularizers_only()
print(f"  data_nll = {float(nll):.4f}  (elapsed {dt:.2f}s)", flush=True)
print(f"  reg      = {float(reg):.4f}", flush=True)
assert torch.isfinite(nll), "data_nll is not finite!"
assert torch.isfinite(reg), "regularizer not finite!"


# ─────────────────────────────────────────────────────────────
# 2. Chunked == unchunked (loss + gradient)
# ─────────────────────────────────────────────────────────────

print("\n── 2. chunked vs unchunked ──", flush=True)


def grads_and_loss(L: UnifiedMergeLoss, chunk_size: int) -> tuple[float, dict]:
    for p in L.parameters():
        if p.grad is not None:
            p.grad.zero_()
    total = torch.zeros((), device=device)
    if chunk_size <= 0:
        nll = L.data_nll(0, L.n_obs_total)
        nll.backward()
        total = nll.detach()
    else:
        for s, e in L.obs_chunks(chunk_size):
            nll = L.data_nll(s, e)
            nll.backward()
            total = total + nll.detach()
    grads = {n: p.grad.detach().clone() for n, p in L.named_parameters() if p.grad is not None}
    return float(total), grads


L_full = fresh_loss()
total_full, grads_full = grads_and_loss(L_full, 0)

# Overall gradient magnitude (for relative-tolerance comparisons below)
grad_scale = max((g.abs().max().item() for g in grads_full.values()), default=1.0)
REL_TOL = 1e-4   # float32 accumulation noise over ~5M terms

for cs in [50_000, 200_000, 1_000_000]:
    L_c = fresh_loss()
    total_c, grads_c = grads_and_loss(L_c, cs)
    rel_loss = abs(total_full - total_c) / max(abs(total_full), 1.0)
    # Relative gradient error: max |Δgrad| / max |grad|  (single global scale so
    # entries near zero don't blow up the ratio)
    max_dg = max((grads_full[k] - grads_c[k]).abs().max().item() for k in grads_full)
    rel_grad = max_dg / max(grad_scale, 1e-12)
    tag = "OK" if rel_loss < REL_TOL and rel_grad < REL_TOL else "!!"
    print(
        f"  {tag}  chunk={cs:>7d}  rel|Δloss|={rel_loss:.2e}  "
        f"rel|Δgrad|={rel_grad:.2e}",
        flush=True,
    )
print(f"  (REL_TOL = {REL_TOL:.0e};  grad_scale = max|grad| = {grad_scale:.3e})", flush=True)


# ─────────────────────────────────────────────────────────────
# 3. Wall-clock + peak memory across chunk sizes
# ─────────────────────────────────────────────────────────────

print("\n── 3. timing + memory ──", flush=True)
print(f"  {'chunk_size':>10}  {'fwd+bwd (s)':>12}  {'peak_mem (MB)':>13}", flush=True)


def time_and_mem(chunk_size: int, n_repeat: int = 5) -> tuple[float, float]:
    L = fresh_loss()
    # warm-up
    _ = grads_and_loss(L, chunk_size)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        _ = grads_and_loss(L, chunk_size)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n_repeat
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e6
    else:
        peak = float("nan")
    return dt, peak


for cs in [50_000, 100_000, 200_000, 500_000, 1_000_000, 0]:
    try:
        dt, peak = time_and_mem(cs)
        print(f"  {cs:>10d}  {dt:>12.3f}  {peak:>13.1f}", flush=True)
    except torch.cuda.OutOfMemoryError as exc:  # noqa: PERF203
        print(f"  {cs:>10d}  OOM: {exc}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────
# 4. Steady-state timing (10 back-to-back closure calls)
# ─────────────────────────────────────────────────────────────

print("\n── 4. steady-state 10-call closure ──", flush=True)
for cs in [200_000, 0]:
    L = fresh_loss()
    # Warm-up
    _ = grads_and_loss(L, cs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        _ = grads_and_loss(L, cs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    total = time.perf_counter() - t0
    tag = f"chunk={cs}" if cs else "full"
    print(f"  {tag}: 10 closures in {total:.1f}s  ({total/10:.2f}s each)", flush=True)

print("\nDone.", flush=True)
