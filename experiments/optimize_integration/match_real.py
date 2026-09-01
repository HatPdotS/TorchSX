"""Fit ``make_problem`` parameters to the real batch's distributions.

The synthetic problems used earlier put the operating point in the wrong place
-- median n_ph 26 against the real 136, and no background to speak of -- which
is why they reported the inner grid as converged when it is not.  This solves
for the generative parameters that reproduce the real marginals, so the
certified oracle can be applied at the place production actually runs.

Matched: n_ph (median and 99th percentile, via the Wilson Sigma scale), base
spread (via sd_log_K), bg, alpha and beta ranges, and the multiplicity ladder.
Everything else follows from the generative model.
"""
from __future__ import annotations

import math

import torch

from oracle import make_problem


def _q(t, p):
    return float(t.double().quantile(torch.tensor(p, dtype=torch.float64)))


def main() -> None:
    d = torch.load("batch.pt", map_location="cpu")
    tgt = {
        "n50": _q(d["n_ph"], 0.5), "n99": _q(d["n_ph"], 0.99),
        "base_sd": float(d["base"].double().std()),
    }
    print(f"target:  n_ph median {tgt['n50']:.1f}  q99 {tgt['n99']:.0f}   "
          f"base sd {tgt['base_sd']:.3f}\n")

    # alpha -> w: log alpha = log w - 0.5[lgamma(3/b) - lgamma(1/b)]
    b = float(d["log_beta"].exp().median())
    shift = 0.5 * (math.lgamma(3.0 / b) - math.lgamma(1.0 / b))
    a_lo, a_hi = _q(d["log_alpha"].exp(), 0.01), _q(d["log_alpha"].exp(), 0.99)
    w_range = (a_lo * math.exp(shift), a_hi * math.exp(shift))
    # log-uniform has geometric-mean median, so pin the geometric mean to the
    # real median rather than taking the 1-99 % range literally: bg sets where
    # the Poisson factor sits, and getting its centre right matters more than
    # its tails.
    bg_hi = _q(d["bg"], 0.99)
    bg_med = _q(d["bg"], 0.50)
    bg_range = (bg_med ** 2 / bg_hi, bg_hi)
    print(f"  beta {b:.3f}  ->  w_range {w_range[0]:.3f}..{w_range[1]:.3f}")
    print(f"  bg_range {bg_range[0]:.2f}..{bg_range[1]:.1f}")

    MULTS = (16, 48, 96, 200, 400)              # real: q01 16, med 93, q99 405

    N_RUNG = 16

    def ladder(scale, span):
        """Log-spaced Wilson Sigma ladder: ``scale`` sets the level (hence
        n_ph's median), ``span`` its dynamic range (hence the upper tail).

        Sixteen rungs, not four.  The real data is extremely heavy-tailed --
        median n - bg is 7.3, essentially noise, while the maximum is 269,615,
        five orders of magnitude up -- and a mixture of exponentials over a
        handful of rungs cannot reach that far.  The tail is where the float32
        cancellation in `n log(lam) - lam - log n!` lives, so reproducing it is
        the entire point of this fit.
        """
        return tuple(
            scale * span ** (i / (N_RUNG - 1) - 0.5) for i in range(N_RUNG)
        )

    def build(scale, sd_k, span=8.0, n_per=8):
        return make_problem(
            n_per_mult=n_per, mults=MULTS,
            sigma_ladder=ladder(scale, span),
            w_range=w_range, beta_range=(b, b + 1e-6), bg_range=bg_range,
            sd_log_K=sd_k, seed=0, device="cpu",
        )

    # Sigma scale sets n_ph; sd_log_K sets the base spread.  They are nearly
    # independent, so two 1-D bisections rather than a joint search.
    # sd_log_K is NOT identifiable from base's spread here: base = log_K -
    # log_e_s, and with beta ~ 1.07 the mean/median correction log_e_s is
    # violently sensitive to alpha (E[e^eps] diverges as beta -> 1, alpha -> 1),
    # so the measured alpha range alone produces 1.55 of spread against the real
    # 1.166.  The real network predicts log_K and log_w from the same encoder,
    # so its two terms are correlated and partly cancel; an independent-draw
    # model cannot reproduce that.  Fix sd_log_K at a physical value instead and
    # accept a base that is ~33 % wider than reality -- which makes the test
    # harder, not easier.
    sd_k = 0.35

    # Level sets n_ph's median, span its 99th percentile; each shifts the other
    # a little, so alternate.  Both bisections must use the SAME sd_log_K the
    # final problem is built with, since log_K spread feeds straight into n_ph.
    scale, span = 50.0, 8.0
    for _ in range(6):
        lo, hi = 1e-3, 1e5
        for _ in range(30):
            mid = math.sqrt(lo * hi)
            got = _q(build(mid, sd_k, span).n_ph, 0.5)
            lo, hi = (mid, hi) if got < tgt["n50"] else (lo, mid)
        scale = math.sqrt(lo * hi)
        lo, hi = 1.05, 1e7
        for _ in range(30):
            mid = math.sqrt(lo * hi)
            got = _q(build(scale, sd_k, mid).n_ph, 0.99)
            lo, hi = (mid, hi) if got < tgt["n99"] else (lo, mid)
        span = math.sqrt(lo * hi)

    print(f"\nfitted:  level {scale:.3f}  span {span:.1f}  sd_log_K {sd_k:.3f} (fixed)")
    print(f"  sigma_ladder {tuple(round(x, 3) for x in ladder(scale, span))}")
    p = build(scale, sd_k, span, n_per=16)
    print(f"\n{'quantity':<10} {'real':>12} {'matched':>12}")
    for nm, real, got in (
        ("n_ph 1%", _q(d["n_ph"], .01), _q(p.n_ph, .01)),
        ("n_ph 25%", _q(d["n_ph"], .25), _q(p.n_ph, .25)),
        ("n_ph 50%", tgt["n50"], _q(p.n_ph, .5)),
        ("n_ph 90%", _q(d["n_ph"], .9), _q(p.n_ph, .9)),
        ("n_ph 99%", tgt["n99"], _q(p.n_ph, .99)),
        ("n_ph 99.9%", _q(d["n_ph"], .999), _q(p.n_ph, .999)),
        ("n_ph max", float(d["n_ph"].max()), float(p.n_ph.max())),
        ("n-bg 50%", _q(d["n_ph"] - d["bg"], .5), _q(p.n_ph - p.bg, .5)),
        ("n-bg max", float((d["n_ph"] - d["bg"]).max()),
         float((p.n_ph - p.bg).max())),
        ("bg 50%", _q(d["bg"], .5), _q(p.bg, .5)),
        ("alpha 50%", _q(d["log_alpha"].exp(), .5), _q(p.log_alpha.exp(), .5)),
        ("base sd", tgt["base_sd"], float(p.base.std())),
        ("base 50%", _q(d["base"], .5), _q(p.base, .5)),
    ):
        print(f"{nm:<10} {real:>12.3f} {got:>12.3f}")
    neg_r = float(((d["n_ph"] - d["bg"]) < 0).float().mean())
    neg_p = float(((p.n_ph - p.bg) < 0).double().mean())
    print(f"{'neg n-bg':<10} {neg_r*100:>11.1f}% {neg_p*100:>11.1f}%")
    for thr in (1e3, 1e4, 1e5):
        fr = float((d["n_ph"] > thr).float().mean()) * 100
        fp = float((p.n_ph > thr).double().mean()) * 100
        print(f"{'n>%.0e' % thr:<10} {fr:>11.3f}% {fp:>11.3f}%")
    print(f"\nM = {p.M:,}   n_hkl = {p.n_hkl}")
    print("\n--- preset for oracle.py ---")
    print(f"REAL_LIKE = dict(\n"
          f"    mults={MULTS},\n"
          f"    sigma_ladder={tuple(round(x, 3) for x in ladder(scale, span))},\n"
          f"    w_range=({w_range[0]:.3f}, {w_range[1]:.3f}),\n"
          f"    beta_range=({b:.4f}, {b + 1e-6:.4f}),\n"
          f"    bg_range=({bg_range[0]:.2f}, {bg_range[1]:.1f}),\n"
          f"    sd_log_K={sd_k:.3f},\n)")


if __name__ == "__main__":
    main()
