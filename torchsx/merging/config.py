"""Configuration for the modular merging pipeline.

Fields are grouped by role.  Per-term λ-weights end up on
``LossState.set_weight("reg/name", lam)`` at Merger-build time — the config
is the single source of truth for hyperparameters, the LossState is what
applies them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MergeConfig:
    # ── Observation filtering / shell setup ────────────────────────────
    min_obs: int = 0
    n_shells: int = 20
    snr_thresh: float = 3.0
    #: Cap on the observations pooled per resolution shell for the GenGauss β
    #: fit.  Pooling every observation is ~4.5 min of the full-data prep; a 2 M
    #: cap cuts that to ~1 min.  OFF by default (0) because it does move β --
    #: max |Δβ| 0.007 and α_init 0.5 % on a 159 M-observation subset -- and
    #: every scored baseline was measured uncapped.  Opt in for exploratory
    #: runs where prep time matters more than reproducing an old number.
    beta_pool_cap: int = 0

    # ── Dirichlet smoothing over ±1 Miller graph (per-HKL α, β, I₀) ─────
    # λ ≈ 0.4 · N_h_max keeps per-HKL data pull ≤ 10% from neighbour mean;
    # for BR/tubulin scale (~1000 obs/HKL) that is ~400–500.
    lam_alpha_smooth: float = 500.0
    lam_beta_smooth: float = 500.0
    lam_I0_smooth: float = 0.0

    # Optional Gaussian anchors toward shell-fit init (biased by
    # positive-only init → β overestimated), off by default.
    lam_alpha_anchor: float = 0.0
    lam_beta_anchor: float = 0.0

    # Weak Gaussian prior on log I₀ centred at ``log_I0_floor_target``.
    # MAP arithmetic: HKLs with N_h ≥ ~100 ignore this; pure-bg HKLs go
    # to the target.  Still kept for back-compat (new runs set to 0 and
    # use the Wilson prior instead).
    lam_I0_floor: float = 0.1
    log_I0_floor_target: float = -11.5

    # Half-Cauchy prior on GenGauss width σ = α · c(β).
    lam_sigma_cauchy: float = 0.0
    sigma_cauchy_scale: float = 1.0

    # ── LBFGS ─────────────────────────────────────────────────────────
    n_outer_iters: int = 1
    max_iter: int = 200
    history_size: int = 10
    tolerance_grad: float = 1e-6
    tolerance_change: float = 1e-9
    lr: float = 1.0

    # DEPRECATED: regulariser scaling is disabled in :mod:`merger` —
    # ``s`` is hardcoded to 1.0 and ``cfg.lam_*`` weights are used as
    # absolute losses.  This field is retained for backwards compatibility
    # with config files / pickles but is no longer consulted.
    reg_scale_ref_obs: int = 2_200_000

    # ── Chunking / quadrature ─────────────────────────────────────────
    obs_chunk_size: int = 1_000_000   # 0 disables chunking
    n_quad: int = 51
    n_sigma_quad: float = 5.0
    # Constant grid lower bound on signal (None ⇒ per-HKL Poisson floor).
    sig_lo_override: float | None = 1e-4
    apply_bias_correction: bool = False
    # Optional settings for AdaptiveHybridLogGrid experiments.
    n_quad_adaptive: int = 51
    adaptive_tail_tol: float = 1e-8
    adaptive_u_max_cap: float = 30.0
    adaptive_detach_grid: bool = True

    # ── Scatter family ───────────────────────────────────────────────
    # "gengauss" — generalised-Gaussian on ε=log(S/μ), per-HKL α + β
    # "gamma"    — Gamma on linear S (β unused), naturally covers S→0
    scatter: str = "gengauss"

    # ── Background mixture ───────────────────────────────────────────
    # "off" | "global" | "global_learned" | "per_hkl" — swap the
    # DetectionLikelihood subclass.
    #   global         : fixed 50/50 mix (no learnable π)
    #   global_learned : single learnable π shared across all obs
    #   per_hkl        : per-HKL learned π_h (degenerate when data is weak)
    bg_mode: str = "global"
    # Initial value of the learned global π (only used for global_learned).
    pi_global_init: float = 0.1
    pi_beta_a: float = 1.0
    pi_beta_b: float = 2.0
    lam_pi_beta: float = 1.0

    # ── French–Wilson MAP prior on I₀ ─────────────────────────────────
    lam_frac_wilson: float = 0.0
    fw_n_shells: int = 20
    fw_eps: float = 1e-6
    # Shell-mean estimator for Σ in WilsonPrior:
    # "arithmetic" — Σ = mean(I0) (linear; insensitive to collapsed HKLs).
    # "geometric"  — Σ = exp(mean(log I0) + γ) (sensitive on equal log
    #                footing; collapsed HKLs perturb Σ and feed back through
    #                the per-shell coupling).
    wilson_mean_kind: str = "arithmetic"

    # ── Wilson distribution-match prior (KL per shell × centricity) ───
    lam_wilson_match: float = 0.0
    wilson_match_min_count: int = 20

    # ── Wilson anisotropic prior on log_I0 (alternative to lam_frac_wilson) ──
    # Replaces the per-shell discrete Σ with a single global log_I_mean and a
    # 6-parameter anisotropic ADP-like decay U so log_Σ_h = log_I_mean
    # − 2π² · sᵀ U s. Convention matches torchref.scaling.scaler_base. The
    # population mean of per-crystal B is pulled toward zero by a soft
    # regularizer (lam_wilson_aniso_b_center) so the global resolution decay
    # flows into U rather than being absorbed by the per-crystal scale.
    # Mutually exclusive with lam_frac_wilson at runtime.
    lam_wilson_aniso: float = 4.0
    lam_wilson_aniso_U_ridge: float = 1.0
    lam_wilson_aniso_b_center: float = 1.0
    wilson_aniso_init_from_data: bool = True

    # ── Detector gain ─────────────────────────────────────────────────
    #   "stream"    → ``adu_per_eV × photon_energy_eV`` from the stream
    #   "empirical" → Mandel calibration on negatives
    #   float       → explicit ADU/photon
    gain: str | float = "stream"

    # ── Posterior moments at writeout ────────────────────────────────
    # With ``posterior_enabled`` the Merger integrates p(I₀|data) per HKL
    # to report ⟨I₀⟩ and √Var[I₀] rather than the bare MAP.  Expensive
    # step is the *inner* quadrature over the latent signal; we avoid
    # redoing it at every outer log-I₀ point by evaluating each
    # observation's likelihood on a coarse μ-grid (``K_coarse`` points)
    # and linearly interpolating onto the fine integration grid
    # (``M_fine`` points).  Both grids span ``± n_sigma · σ_Laplace``
    # around the MAP.  A flat prior and a Wilson prior (with Σ_h fixed at
    # the shell-mean of the converged MAP) are evaluated by default; the
    # primary ``MergedResult.I0`` column is the flat-prior posterior mean.
    posterior_enabled: bool = True
    posterior_n_grid_coarse: int = 31
    posterior_n_grid_fine: int = 101
    posterior_n_sigma: float = 6.0
    posterior_obs_chunk_size: int = 500_000
    # Cap on σ_Laplace inside the integration.  HKLs with near-flat data
    # Hessians have arbitrarily large σ, which without a cap blows the
    # grid out to thousands of log-units and overflows exp.  Default 2.0
    # (multiplicative factor of ~7.4) is far wider than any plausibly
    # informative HKL; HKLs above the cap fall back to MAP via
    # ``map_fallback_log_I0``.
    posterior_sigma_max: float = 2.0
    # Which posterior to use as the primary ``I0`` / ``sigma_I0`` column.
    # Both named variants are still stored in diagnostics.
    posterior_primary: str = "wilson"
    # How to summarise the per-HKL posterior in linear-I space.
    #   "linear_mean"  — I0 = E[I | data]   (default; legacy behaviour)
    #   "log_mean"     — I0 = exp(E[log I | data]); sigma via delta method.
    # Linear-mean is biased upward by ~exp(σ²_logI / 2) for broad posteriors
    # (weak HKLs); log-mean is unbiased w.r.t. log_I_truth.  See
    # experiments/synthetic_gengauss/integrand_diag/fix_bias for the derivation.
    posterior_summary: str = "linear_mean"
    # How to lay out the dlog_I integration grid for the posterior moments.
    #   "laplace" — MAP-centered, ±n_sigma·σ_laplace per HKL  (default)
    #   "global"  — single fixed log_I grid, identical for every HKL.
    # ``"global"`` removes the Laplace approx and any MAP-centering bias.
    # When the quadrature exposes ``log_S_grid`` (e.g. ``FixedLogGrid``),
    # those nodes are used directly; otherwise the explicit range below
    # parameterises a fresh grid.
    posterior_grid_mode: str = "laplace"
    posterior_global_log_I_min: float = -6.907755     # log(1e-3)
    posterior_global_log_I_max: float = 13.815511     # log(1e6)
    posterior_global_n_points: int = 1025
