"""Is the inner-integral kernel actually limited by transcendental throughput?

Three independent ways to answer it, because the arithmetic estimate alone is
not evidence:

1. **Scaling.** Sweep K at fixed M.  If ``t = a + b*K`` with a small intercept,
   the inner loop is in a throughput regime and ``b*M`` is the cost per
   integrand evaluation.  Compare that against the HBM and SFU rooflines.
2. **Ablation.**  A probe kernel with the identical loop, memory pattern and
   dependency chain, but with ``log2``/``exp`` replaced by cheap arithmetic.
   The gap between the two IS the transcendental cost -- no modelling.
3. **ncu.**  ``--single`` issues exactly one launch so Nsight Compute can
   report the XU (transcendental) pipe utilisation directly.

A100-PCIE-40GB reference numbers used for the rooflines below: 108 SM,
1.41 GHz boost, 16 SFU/SM/clk -> 2.44e12 transcendentals/s; 19.5 TFLOP/s fp32
FMA; 1555 GB/s HBM.
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import triton
import triton.language as tl

SM, CLK, SFU_PER_SM = 108, 1.41e9, 16
SFU_PEAK = SM * SFU_PER_SM * CLK
FMA_PEAK = SM * 64 * 2 * CLK
HBM = 1555e9

LN2 = 0.6931471805599453   # host-side only; Triton forbids non-constexpr globals


@triton.jit
def _probe(
    log_mean_ptr, n_ph_ptr, bg_ptr, log_alpha_ptr, log_beta_ptr,
    log_S_grid_ptr, S_grid_ptr, log_w_ptr, out_ptr,
    M, K, MODE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Same shape as ``_fwd_kernel``; MODE 0 keeps the transcendentals, MODE 1
    swaps every ``log2``/``exp`` for a multiply-add of the same operand, so the
    loads, the loop, the online-max rescale and the register pressure all
    survive and only the XU work disappears."""
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    log_mean = tl.load(log_mean_ptr + offs_m, mask=mask_m, other=0.0)
    n_ph = tl.load(n_ph_ptr + offs_m, mask=mask_m, other=0.0)
    bg = tl.load(bg_ptr + offs_m, mask=mask_m, other=0.0)
    log_alpha = tl.load(log_alpha_ptr + offs_m, mask=mask_m, other=0.0)
    log_beta = tl.load(log_beta_ptr + offs_m, mask=mask_m, other=0.0)

    if MODE == 0:
        alpha = tl.exp(log_alpha)
        beta = tl.exp(log_beta)
    else:
        alpha = 1.0 + log_alpha * 0.5
        beta = 1.0 + log_beta * 0.5
    inv_alpha = 1.0 / tl.where(alpha > 1e-8, alpha, 1e-8)

    m_max = tl.full((BLOCK_M,), -1e30, dtype=tl.float32)
    T0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        log_S_k = tl.load(log_S_grid_ptr + offs_k, mask=mask_k, other=0.0)
        S_k = tl.load(S_grid_ptr + offs_k, mask=mask_k, other=0.0)
        log_w_k = tl.load(log_w_ptr + offs_k, mask=mask_k, other=0.0)

        eps = log_S_k[None, :] - log_mean[:, None]
        z = tl.abs(eps) * inv_alpha[:, None]
        z_eff = tl.where(z > 1e-12, z, 1e-12)
        lam = S_k[None, :] + bg[:, None]
        lam = tl.where(lam > 1e-6, lam, 1e-6)
        if MODE == 0:
            log_z = 0.6931471805599453 * tl.math.log2(z_eff)
            z_pow = tl.exp(beta[:, None] * log_z)
            log_lam = 0.6931471805599453 * tl.math.log2(lam)
        else:
            log_z = 0.6931471805599453 * (z_eff * 0.5 + 0.25)
            z_pow = beta[:, None] * log_z * 0.5 + 0.25
            log_lam = 0.6931471805599453 * (lam * 0.5 + 0.25)
        log_int = -z_pow + n_ph[:, None] * log_lam - lam + log_w_k[None, :]
        log_int = tl.where(mask_k[None, :], log_int, -1e30)
        m_chunk = tl.max(log_int, axis=1)
        m_new = tl.maximum(m_max, m_chunk)
        if MODE == 0:
            T0 = T0 * tl.exp(m_max - m_new) + tl.sum(tl.exp(log_int - m_new[:, None]), axis=1)
        else:
            T0 = T0 * (m_max - m_new) + tl.sum(log_int - m_new[:, None], axis=1)
        m_max = m_new
    tl.store(out_ptr + offs_m, m_max + T0, mask=mask_m)


def _bench(fn, reps=5):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / reps


def _inputs(M, K, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(0)
    log_S = torch.linspace(-34.5, 34.5, K, device=dev)
    return dict(
        log_mean=torch.randn(M, device=dev, generator=g),
        n_ph=(torch.rand(M, device=dev, generator=g) * 50).round(),
        bg=torch.rand(M, device=dev, generator=g) * 5 + 0.5,
        log_alpha=torch.randn(M, device=dev, generator=g) * 0.2 + 0.8,
        log_beta=torch.zeros(M, device=dev) + 0.35,
        log_S=log_S, S=log_S.exp(),
        log_w=torch.full((K,), math.log(69.0 / (K - 1)), device=dev),
    )


def _run_probe(d, M, K, mode, BM=64, BK=16):
    out = torch.empty(M, device="cuda")
    _probe[(triton.cdiv(M, BM),)](
        d["log_mean"], d["n_ph"], d["bg"], d["log_alpha"], d["log_beta"],
        d["log_S"], d["S"], d["log_w"], out, M, K,
        MODE=mode, BLOCK_M=BM, BLOCK_K=BK, num_warps=4, num_stages=3,
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true",
                    help="one launch of the real kernel, for ncu")
    ap.add_argument("-M", type=int, default=1 << 20)
    ap.add_argument("-K", type=int, default=2001)
    args = ap.parse_args()

    from torchsx.merging.quadrature._fixed_log_grid_triton import (
        marginalize_signal_triton,
    )
    d = _inputs(args.M, args.K)
    if args.single:
        marginalize_signal_triton(
            d["log_mean"], d["n_ph"], d["bg"], d["log_alpha"], d["log_beta"],
            d["log_S"], d["log_w"],
        )
        torch.cuda.synchronize()
        return

    print(f"A100 rooflines: SFU {SFU_PEAK:.2e}/s   FMA {FMA_PEAK:.2e} flop/s   "
          f"HBM {HBM/1e9:.0f} GB/s\n")

    # ── 1. K scaling of the production kernel ──────────────────────────
    M = args.M
    print(f"1  production kernel, M = {M:,}")
    print(f"   {'K':>6} {'ms':>9} {'evals/s':>11} {'SFU/s (x5)':>12} "
          f"{'%SFU':>7} {'GB/s':>8} {'%HBM':>7}")
    ks = [129, 257, 513, 1025, 2001, 4001, 8001]
    ts = []
    for K in ks:
        dd = _inputs(M, K)
        t = _bench(lambda: marginalize_signal_triton(
            dd["log_mean"], dd["n_ph"], dd["bg"], dd["log_alpha"],
            dd["log_beta"], dd["log_S"], dd["log_w"]))
        ts.append(t)
        ev = M * K / t
        byt = (7 * M + 5 * M + 3 * K) * 4
        print(f"   {K:>6} {t*1e3:>9.2f} {ev:>11.2e} {ev*5:>12.2e} "
              f"{100*ev*5/SFU_PEAK:>6.1f}% {byt/t/1e9:>8.1f} "
              f"{100*byt/t/HBM:>6.1f}%")
    # linear fit t = a + b K
    n = len(ks)
    mk = sum(ks) / n
    mt = sum(ts) / n
    b = sum((k - mk) * (t - mt) for k, t in zip(ks, ts)) / sum((k - mk) ** 2 for k in ks)
    a = mt - b * mk
    print(f"   fit t = {a*1e3:.3f} ms + {b*1e6:.4f} us/K   "
          f"-> intercept is {100*a/ts[-1]:.1f}% of the K=8001 time")
    print(f"   steady-state cost per integrand eval: {b/M*1e12:.3f} ps")

    # ── 2. ablation: same kernel, no transcendentals ───────────────────
    print(f"\n2  ablation at K = {args.K}: identical loop, SFU work removed")
    dd = _inputs(M, args.K)
    t0 = _bench(lambda: _run_probe(dd, M, args.K, 0))
    t1 = _bench(lambda: _run_probe(dd, M, args.K, 1))
    print(f"   with transcendentals   {t0*1e3:8.2f} ms")
    print(f"   without                {t1*1e3:8.2f} ms")
    print(f"   -> transcendentals are {100*(t0-t1)/t0:5.1f}% of the runtime")
    if t1 > 0.5 * t0:
        print("   -> NOT SFU bound: removing every exp/log leaves most of the "
              "time behind.")
    else:
        print("   -> SFU dominated: the transcendentals are the runtime.")


if __name__ == "__main__":
    main()
