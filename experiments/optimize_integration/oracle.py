"""Dense synthetic oracle for the posterior integration over I0.

What it is authoritative about
------------------------------
Both integrals, not just the inner one:

    log p_j(v) = log INT p_pois(n_j | S + bg_j)
                      . GG(log S - v - base_j; alpha_j, beta_j) d log S      (inner)
    log Z_h    = log INT exp(sum_{j in h} log p_j(v)) . p_W(e^v) e^v dv      (outer)
    E[I0_h], SD[I0_h] from the same outer measure

plus the gradients of ``sum_h log Z_h`` with respect to ``base``, ``log alpha``
and ``log beta`` -- which is what actually trains the network, and which a
value-only check cannot police: a quadrature grid that follows ``log_mean`` can
reproduce the values and still get the derivatives wrong, because the analytic
backward assumes the nodes do not move with the parameters.

How it is cheap enough to be dense
----------------------------------
The outer node enters the inner integral **only as a translation** of the
GenGauss, so ``log p_j`` is the log of a convolution::

    G_q[mu] = sum_i f_j(x_i) q(x_i - mu),      f_j = Poisson factor, q = kernel

One FFT per observation therefore evaluates it at *every* mu at once, which is
what makes 2^18 inner nodes affordable.  Better, the three parameter
derivatives are three more convolutions against closed-form kernels::

    dk/dmu       = k . beta z^beta / t
    dk/dlog alpha = k . (beta z^beta - 1)
    dk/dlog beta  = k . (-beta z^beta log z + 1 + psi(1/beta)/beta)
    d log p/dtheta = G_{dk/dtheta} / G_k

So the oracle's gradients come from a completely different numerical route
than the Triton kernel's ``S1..S4`` softmax statistics -- an independent
implementation, not autograd through the same formula.

Two kinds of check, which catch different bugs
----------------------------------------------
*Numerical*: halve the node spacing, widen the extent -- the change is the
oracle's own error floor, and every candidate difference below it is noise.
*Statistical*: the problem is generated **from the model** (Wilson I0 -> scale
K -> GenGauss scatter -> Poisson counts), so ``E[I0|data]`` must converge on
``I0_true`` as multiplicity grows.  Numerical agreement proves we evaluate our
formula correctly; recovery proves it is the right formula.

beta < 1 is out of range by construction.  The GenGauss tail is ``exp(-z^beta)``
so covering it to 1e-13 needs ``z ~ 30^(1/beta)`` -- at beta = 0.5 that is
z = 900, a grid 30x wider for the same resolution.  That is a real statement
about heavy-tailed scatter, not an oracle limitation; the head's floor is
beta_min = 0.3, so if the fitted beta ever goes below 1 this oracle stops
applying and must be rebuilt around an analytic tail.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["Problem", "make_problem", "oracle", "OracleOut", "REAL_LIKE"]

#: Generative parameters fitted to a real 1.5 M-observation batch of
#: ``dark.stream`` (see ``match_real.py``).  Use this, not the defaults: the
#: earlier synthetic problems sat at median n_ph 26 with negligible background
#: and reported the inner grid as converged when on real data it is not.
#:
#: Matched well: the bulk of n_ph (quartile 57 vs 56, median 162 vs 136, 90th
#: 734 vs 639), bg (median 118 vs 116), alpha (median 1.244 vs 1.261), beta
#: (1.071, a single global scalar because the runs use predict_shape=False),
#: and the fraction of negative n - bg (36.5 % vs 32.4 %).
#:
#: Deliberately NOT matched, both conservative:
#:  * the far tail is 2-3x too heavy (n > 1e4 at 0.80 % vs 0.28 %) -- further
#:    fitting just chases sampling noise in a five-decade tail, and an excess of
#:    bright observations over-represents the regime that is hard to integrate;
#:  * ``base`` is ~33 % wider than reality (sd 1.59 vs 1.166).  It is not
#:    identifiable: with beta ~ 1.07 the mean/median correction log_e_s is
#:    violently sensitive to alpha, so the measured alpha range alone produces
#:    1.55 of spread.  The real network predicts log_K and log_w from one
#:    encoder, so its two terms correlate and partly cancel; independent draws
#:    cannot reproduce that.
REAL_LIKE = dict(
    mults=(16, 48, 96, 200, 400),
    sigma_ladder=(0.001, 0.002, 0.006, 0.018, 0.054, 0.158, 0.461, 1.351,
                  3.958, 11.591, 33.947, 99.418, 291.161, 852.706, 2497.272,
                  7313.621),
    w_range=(1.147, 2.161),
    beta_range=(1.0711, 1.0712),
    bg_range=(16.16, 835.9),
    sd_log_K=0.350,
)

#: Relative floor for the FFT convolution, ~32 sqrt(L) eps.  Below this the
#: result is round-off, so ``log p`` is trustworthy to about 26 nats under its
#: own peak.  Measured, not assumed: ``verify_oracle.py`` sweeps mu across the
#: whole outer range and reports the error against a direct sum as a function
#: of depth below peak.
FFT_NOISE = 4e-12


# ── problem generation ─────────────────────────────────────────────────


@dataclass
class Problem:
    """Everything the integration consumes, plus the truth it was drawn from."""

    base: Tensor          # (M,)  log K - log E[e^eps]
    n_ph: Tensor          # (M,)  observed counts
    bg: Tensor            # (M,)
    log_alpha: Tensor     # (M,)
    log_beta: Tensor      # (M,)
    hkl_id: Tensor        # (M,)
    n_hkl: int
    wilson_sigma: Tensor  # (n_hkl,) shell mean intensity Sigma
    # generative truth
    I0_true: Tensor       # (n_hkl,)
    log_K: Tensor         # (M,)
    log_w: Tensor         # (M,)  SD of the log-scatter
    eps: Tensor           # (M,)  the scatter that was actually drawn
    mult: Tensor          # (n_hkl,)
    bg_true: Tensor       # (M,)  the background that GENERATED the counts

    @property
    def M(self) -> int:
        return int(self.base.shape[0])


def make_problem(
    *,
    n_per_mult: int = 16,
    mults: tuple[int, ...] = (2, 16, 128, 751),
    sigma_ladder: tuple[float, ...] = (0.5, 5.0, 50.0, 200.0),
    w_range: tuple[float, float] = (0.4, 3.0),
    beta_range: tuple[float, float] = (1.0, 3.0),
    bg_range: tuple[float, float] = (0.5, 40.0),
    sd_log_K: float = 0.35,
    bg_noise: bool = False,
    bg_per_reflection: bool = False,
    n_peak_over_n_bg: float = 149.0 / 128.0,
    seed: int = 0,
    device: str = "cuda",
) -> Problem:
    """Draw a problem from the model the posterior inverts.

    ``mults`` deliberately spans 2 to 751 -- the real data's median is 751
    observations per reflection, but the outer integral behaves completely
    differently when a reflection has two, and the weak/low-multiplicity corner
    is where every previous grid failure showed up.  ``sigma_ladder`` spans the
    shells: at Sigma = 0.5 with bg up to 40 the likelihood is essentially
    uninformative and the posterior *is* the prior, which is the 84 %-of-
    observations regime and the one that must be right.
    """
    from torchsx.merging.scatter.gengauss import (
        log_alpha_from_log_width, log_e_s_gengauss,
    )
    torch.manual_seed(seed)
    dev = device

    n_shell = len(sigma_ladder)
    n_hkl = n_per_mult * len(mults)
    lad = torch.tensor(sigma_ladder, device=dev, dtype=torch.float64)
    Sigma = lad[torch.arange(n_hkl, device=dev) % n_shell]
    # Wilson, acentric (P1): p(I) = exp(-I/Sigma)/Sigma
    I0_true = -Sigma * torch.rand(n_hkl, device=dev, dtype=torch.float64).clamp(min=1e-12).log()

    mult = torch.tensor(mults, device=dev).repeat_interleave(n_per_mult)
    hkl_id = torch.arange(n_hkl, device=dev).repeat_interleave(mult)
    M = int(mult.sum())

    log_K = torch.randn(M, device=dev, dtype=torch.float64) * sd_log_K
    lw_lo, lw_hi = math.log(w_range[0]), math.log(w_range[1])
    log_w = torch.rand(M, device=dev, dtype=torch.float64) * (lw_hi - lw_lo) + lw_lo
    beta = (torch.rand(M, device=dev, dtype=torch.float64)
            * (beta_range[1] - beta_range[0]) + beta_range[0])
    log_alpha = log_alpha_from_log_width(log_w, beta)
    base = log_K - log_e_s_gengauss(log_alpha, beta)

    lb_lo, lb_hi = math.log(bg_range[0]), math.log(bg_range[1])
    if bg_per_reflection:
        # Background is a property of where a reflection sits on the detector,
        # so it must be constant across a reflection's observations.  Drawing it
        # i.i.d. per observation makes every reflection's MEAN background
        # identical once ~865 of them are averaged, which leaves a
        # bg-dependence test with no leverage at all.
        bgr = (torch.rand(n_hkl, device=dev, dtype=torch.float64)
               * (lb_hi - lb_lo) + lb_lo).exp()
        bg_true = bgr[hkl_id]
    else:
        bg_true = (torch.rand(M, device=dev, dtype=torch.float64)
                   * (lb_hi - lb_lo) + lb_lo).exp()
    bg = bg_true

    # GenGauss(0, alpha, beta):  |t| = alpha * G^(1/beta), G ~ Gamma(1/beta, 1)
    g = torch._standard_gamma(1.0 / beta)
    sign = torch.where(torch.rand(M, device=dev, dtype=torch.float64) < 0.5, -1.0, 1.0)
    eps = sign * log_alpha.exp() * g.pow(1.0 / beta)

    log_S = I0_true[hkl_id].log() + base + eps
    n_ph = torch.poisson(log_S.exp() + bg_true)

    if bg_noise:
        # The background handed to the likelihood is ESTIMATED from the 9-11 px
        # annulus, so it carries its own error:
        #     sigma(bg_est) = sqrt(bg * n_peak / n_bg) ~ 1.08 sqrt(bg)
        # which at bg = 116 is 11.6 counts against a median signal of 7.3.  The
        # Poisson marginal treats bg as exact, and the error rectifies rather
        # than averaging away: an UPWARD fluctuation makes n - bg_est apparently
        # negative, which S = K I0 exp(eps) > 0 cannot represent, so it is
        # absorbed by pushing I0 up; a downward fluctuation meets no such
        # barrier.  That is a bias, not noise, so 865 observations do not
        # remove it -- and it scales as sqrt(bg).
        bg = (bg_true + torch.randn_like(bg_true)
              * (bg_true * n_peak_over_n_bg).sqrt()).clamp(min=1e-3)

    return Problem(
        base=base, n_ph=n_ph, bg=bg, log_alpha=log_alpha, log_beta=beta.log(),
        hkl_id=hkl_id, n_hkl=n_hkl, wilson_sigma=Sigma,
        I0_true=I0_true, log_K=log_K, log_w=log_w, eps=eps, mult=mult,
        bg_true=bg_true,
    )


# ── the oracle ─────────────────────────────────────────────────────────


@dataclass
class OracleOut:
    log_Z: Tensor         # (n_hkl,)
    I0_mean: Tensor       # (n_hkl,)
    I0_sd: Tensor         # (n_hkl,)
    grad_base: Tensor     # (M,)  d sum_h log Z_h / d base
    grad_log_alpha: Tensor
    grad_log_beta: Tensor
    v: Tensor             # (n_hkl, n_outer) outer nodes
    S: Tensor             # (n_hkl, n_outer) summed inner log-likelihood curve
    w_post: Tensor        # (n_hkl, n_outer)
    edge_mass: Tensor     # (n_hkl,) outer-grid truncation check
    meta: dict


def _cubic(G: Tensor, idx_f: Tensor) -> Tensor:
    """Catmull-Rom interpolation of ``G`` (rows) at fractional indices.

    Linear would leave an O(h^2 f'') error of ~1e-6 here, which is above the
    accuracy the oracle is supposed to certify; cubic is O(h^4) and costs three
    extra gathers.
    """
    N = G.shape[1]
    i0 = idx_f.floor()
    t = (idx_f - i0)
    i0 = i0.long()
    gm = G.gather(1, (i0 - 1).clamp(0, N - 1))
    g0 = G.gather(1, i0.clamp(0, N - 1))
    g1 = G.gather(1, (i0 + 1).clamp(0, N - 1))
    g2 = G.gather(1, (i0 + 2).clamp(0, N - 1))
    return 0.5 * (
        2 * g0
        + (-gm + g1) * t
        + (2 * gm - 5 * g0 + 4 * g1 - g2) * t * t
        + (-gm + 3 * g0 - 3 * g1 + g2) * t * t * t
    )


def _conv_bank(
    n_ph: Tensor, bg: Tensor, log_alpha: Tensor, log_beta: Tensor,
    x: Tensor, h: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """``(log G_k, G_dmu/G_k, G_dloga/G_k, G_dlogb/G_k)`` on the x grid.

    Both factors are normalised by their own maxima before the FFT: the
    Poisson factor and the GenGauss each span hundreds of decades on their own,
    but their *ratio* to their peak is in [0, 1], and the offsets are added back
    in log space afterwards.  The three ratios need no offset at all -- it
    cancels.
    """
    N = x.shape[0]
    L = 2 * N
    dev, f64 = x.device, torch.float64

    # Poisson factor f(x) = p(n | e^x + bg), peak-normalised
    lam = (x[None, :].exp() + bg[:, None]).clamp(min=1e-300)
    log_f = n_ph[:, None] * lam.log() - lam - torch.lgamma(n_ph[:, None] + 1)
    mf = log_f.max(dim=1, keepdim=True).values
    f = (log_f - mf).exp()
    del log_f, lam

    # kernels on the wrapped lag axis: kappa[u] = q(-u_signed * h)
    u = ((torch.arange(L, device=dev) + N) % L) - N
    t = (-u).to(f64) * h                                     # (L,)
    beta = log_beta.exp()
    z = t.abs()[None, :] / log_alpha.exp()[:, None]
    zb = z.clamp(min=1e-300).pow(beta[:, None])
    k = (-zb).exp()                                          # peak-normalised
    # dk/dmu = k * beta z^beta / t   (odd; 0 at t = 0)
    tt = torch.where(t.abs() > 0, t, torch.ones_like(t))
    k_mu = torch.where(t.abs() > 0, k * beta[:, None] * zb / tt[None, :],
                       torch.zeros_like(k))
    k_a = k * (beta[:, None] * zb - 1.0)
    psi = torch.special.digamma(1.0 / beta)
    k_b = k * (-beta[:, None] * zb * z.clamp(min=1e-300).log()
               + 1.0 + psi[:, None] / beta[:, None])
    del z, zb, tt

    fp = torch.zeros(f.shape[0], L, device=dev, dtype=f64)
    fp[:, :N] = f
    F = torch.fft.rfft(fp, n=L)
    del fp, f

    out = []
    for q in (k, k_mu, k_a, k_b):
        out.append(torch.fft.irfft(F * torch.fft.rfft(q, n=L), n=L)[:, :N])
    G0, G1, Ga, Gb = out

    # FFT convolution carries an ABSOLUTE round-off of order
    # sqrt(L) * eps * max(G0), so ``log p`` is meaningless once it drops more
    # than ``-log(FFT_NOISE)`` ~ 26 nats below its own peak -- and there the
    # convolution can even come out negative.  Floor at that noise level rather
    # than letting it through: a sentinel like -1e30 would be *interpolated*
    # against valid neighbours by the cubic and manufacture values of +-1e29,
    # which is exactly what it did (S ran to 1e30 at four nodes out of 512 and
    # took the whole outer integral with it).  Flooring can only raise values
    # that are already far below the peak, so it cannot displace the posterior;
    # ``dyn_range`` reports how far down the result is still trustworthy.
    gmax = G0.max(dim=1, keepdim=True).values.clamp(min=torch.finfo(f64).tiny)
    G0s = torch.maximum(G0, gmax * FFT_NOISE)
    # log alpha and the GG normaliser were divided out of k; put them back
    log_norm = beta.log() - math.log(2.0) - torch.lgamma(1.0 / beta) - log_alpha
    log_p = G0s.log() + mf + log_norm[:, None] + math.log(h)
    return log_p, G1 / G0s, Ga / G0s, Gb / G0s


def _direct_bank(
    n_ph: Tensor, bg: Tensor, log_alpha: Tensor, log_beta: Tensor,
    mu: Tensor, x: Tensor, h: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Same four quantities by literal summation at the given ``mu``.

    No dynamic-range limit -- everything stays in log space — but the cost is
    ``chunk x n_outer x n_inner`` instead of one FFT, so this is for validating
    the convolution route on a small problem, not for production oracle runs.
    """
    lam = (x[None, :].exp() + bg[:, None]).clamp(min=1e-300)
    log_f = n_ph[:, None] * lam.log() - lam - torch.lgamma(n_ph[:, None] + 1)
    beta, alpha = log_beta.exp(), log_alpha.exp()
    log_norm = beta.log() - math.log(2.0) - torch.lgamma(1.0 / beta) - log_alpha
    psi = torch.special.digamma(1.0 / beta)
    lp, r1, ra, rb = [], [], [], []
    for c in range(mu.shape[1]):
        t = x[None, :] - mu[:, c, None]
        z = t.abs() / alpha[:, None]
        zl = z.clamp(min=1e-300).log()
        zb = (beta[:, None] * zl).exp()
        li = -zb + log_norm[:, None] + log_f + math.log(h)
        lp.append(torch.logsumexp(li, dim=1))
        W = torch.softmax(li, dim=1)
        tt = torch.where(t.abs() > 0, t, torch.ones_like(t))
        r1.append((W * torch.where(t.abs() > 0, beta[:, None] * zb / tt,
                                   torch.zeros_like(t))).sum(1))
        ra.append((W * (beta[:, None] * zb - 1.0)).sum(1))
        rb.append((W * (-beta[:, None] * zb * zl + 1.0
                        + psi[:, None] / beta[:, None])).sum(1))
    return (torch.stack(lp, 1), torch.stack(r1, 1),
            torch.stack(ra, 1), torch.stack(rb, 1))


@torch.no_grad()
def oracle(
    prob: Problem, *,
    n_inner: int = 1 << 18,
    n_outer: int = 8192,
    v_lo: float = -25.0,
    v_hi: float = 6.0,
    z_tail: float = 30.0,
    x_hi_pad: float = 3.0,
    chunk: int = 24,
    method: str = "fft",
    need_grads: bool = True,
) -> OracleOut:
    """Dense two-level integration.  Everything in float64.

    ``z_tail`` sets how far into the GenGauss tail the inner grid reaches, in
    units where the density is ``exp(-z^beta)``; 30 gives a truncated mass of
    ``1e-13`` at the worst case ``beta = 1``.  ``v_lo = -25`` truncates the
    Wilson prior at a relative mass of ``e^-25 = 1.4e-11``.  Both are checked,
    not assumed: :func:`convergence` re-runs with them widened.
    """
    dev, f64 = prob.base.device, torch.float64
    M, H = prob.M, prob.n_hkl

    v_u = torch.linspace(v_lo, v_hi, n_outer, device=dev, dtype=f64)
    v = prob.wilson_sigma.log()[:, None] + v_u[None, :]          # (H, n_outer)

    # inner extent: the Poisson factor dies above log(n + few sqrt(n)); below,
    # it PLATEAUS at p(n|bg) and only the GenGauss cuts the integrand off, so
    # the lower limit is set by the scatter and must follow z_tail * alpha.
    alpha = prob.log_alpha.exp()
    beta = prob.log_beta.exp()
    reach = float((alpha * z_tail ** (1.0 / beta)).max())
    mu_lo = float((v[:, 0][prob.hkl_id] + prob.base).min())
    x_hi = float((prob.n_ph + 5 * prob.n_ph.sqrt() + prob.bg).clamp(min=1).log().max()) + x_hi_pad
    x_lo = mu_lo - reach
    x = torch.linspace(x_lo, x_hi, n_inner, device=dev, dtype=f64)
    h = float(x[1] - x[0])

    S = torch.zeros(H, n_outer, device=dev, dtype=f64)
    # The three (M, n_outer) derivative tables dominate the memory -- 3.6 GB
    # each at M = 2.2e5, n_outer = 2048 -- and a recovery check needs only the
    # moments.  Skip them when no gradient is asked for.
    z = torch.zeros(1, 1, device=dev, dtype=f64)
    R1, Ra, Rb = ((torch.empty(M, n_outer, device=dev, dtype=f64)
                   for _ in range(3)) if need_grads else (z, z, z))

    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        mu = v[prob.hkl_id[s:e]] + prob.base[s:e, None]          # (c, n_outer)
        if method == "direct":
            out = _direct_bank(
                prob.n_ph[s:e], prob.bg[s:e],
                prob.log_alpha[s:e], prob.log_beta[s:e], mu, x, h,
            )
        else:
            bank = _conv_bank(
                prob.n_ph[s:e], prob.bg[s:e],
                prob.log_alpha[s:e], prob.log_beta[s:e], x, h,
            )
            idx = ((mu - x_lo) / h).clamp(0.0, n_inner - 1.0)
            out = tuple(_cubic(t, idx) for t in bank)
        S.index_add_(0, prob.hkl_id[s:e], out[0])
        if need_grads:
            R1[s:e], Ra[s:e], Rb[s:e] = out[1], out[2], out[3]

    # outer measure: Wilson prior, dI = e^v dv
    dv = float(v_u[1] - v_u[0])
    log_w = (math.log(dv) + v - v.exp() / prob.wilson_sigma[:, None]
             - prob.wilson_sigma.log()[:, None])
    log_w[:, 0] -= math.log(2.0)
    log_w[:, -1] -= math.log(2.0)

    logits = S + log_w
    log_Z = torch.logsumexp(logits, dim=1)
    w_post = torch.softmax(logits, dim=1)
    I = v.exp()
    I0_mean = (w_post * I).sum(1)
    I0_sd = (w_post * (I - I0_mean[:, None]) ** 2).sum(1).clamp(min=0).sqrt()
    n_edge = max(1, int(round(0.05 * n_outer)))
    edge_mass = w_post[:, :n_edge].sum(1) + w_post[:, -n_edge:].sum(1)

    # d sum_h log Z_h / d theta_j  =  sum_c w_post[h,c] . d log p_j(c)/d theta_j
    if need_grads:
        wj = w_post[prob.hkl_id]
        gb, ga, gl = (wj * R1).sum(1), (wj * Ra).sum(1), (wj * Rb).sum(1)
    else:
        gb = ga = gl = torch.zeros(M, device=dev, dtype=f64)
    return OracleOut(
        log_Z=log_Z, I0_mean=I0_mean, I0_sd=I0_sd,
        grad_base=gb, grad_log_alpha=ga, grad_log_beta=gl,
        v=v, S=S, w_post=w_post, edge_mass=edge_mass,
        meta={"n_inner": n_inner, "n_outer": n_outer, "h": h,
              "x_lo": x_lo, "x_hi": x_hi, "v_lo": v_lo, "v_hi": v_hi,
              "z_tail": z_tail, "reach": reach, "method": method,
              "dyn_range": -math.log(FFT_NOISE) if method == "fft" else float("inf")},
    )
