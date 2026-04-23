"""Configuration for the unified merging pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MergeConfig:
    # Observation filtering / shell setup
    min_obs: int = 20
    n_shells: int = 20
    snr_thresh: float = 3.0

    # Regularizer weights — Dirichlet energy over ±1 Miller graph.
    # Set by the rational rule: λ ≥ 0.4 · N_h_max so the per-HKL data pull is at
    # most 10% of the way from neighbours to its MLE.  For this dataset
    # (N_h_max ~ 800–1000 obs/HKL) that gives λ ≈ 400–500.  Same scale for α and β.
    lam_alpha_smooth: float = 500.0
    lam_beta_smooth: float = 500.0
    lam_I0_smooth: float = 0.0

    # Gaussian anchor toward the shell-fit initial value.  Off by default —
    # strong Dirichlet smoothing (above) handles per-HKL regularisation, and
    # the shell-fit is itself biased (positive-only obs → β overestimated), so
    # anchoring locks in that bias.  Re-enable if you want a stricter scale
    # constraint, but budget carefully.
    lam_alpha_anchor: float = 0.0
    lam_beta_anchor: float = 0.0

    # Half-Cauchy prior on the GenGauss distribution width σ = α · c(β), where
    # c(β) = √(Γ(3/β) / Γ(1/β)).  This is a proper shrinkage prior: mode at 0,
    # log-tails, invariant under α ↔ β trade-offs.  Answers the question "do
    # you really need this noise width?" per HKL.
    #   log prior:  −log(1 + σ²/τ²)  →  adds lam · Σ log(τ² + σ²) to the NLL.
    #   lam_sigma_cauchy = 0    → disabled
    #   lam_sigma_cauchy = 1    → proper Bayes-factor scale
    #   lam_sigma_cauchy ~ 5–10 → stronger shrinkage (overrides weak data)
    # τ is the width you consider "innocent" — below τ the penalty is flat.
    lam_sigma_cauchy: float = 0.0
    sigma_cauchy_scale: float = 1.0

    # LBFGS — outer loop runs up to `n_outer_iters` calls of `opt.step()`,
    # each of which runs up to `max_iter` internal LBFGS iterations using
    # persisted history.  Early-stops if max|grad| falls below
    # `tolerance_grad` between rounds.  Use this to "do more steps" without
    # a single giant step() call, and to get round-by-round diagnostics.
    n_outer_iters: int = 1
    max_iter: int = 200
    history_size: int = 10
    tolerance_grad: float = 1e-6
    tolerance_change: float = 1e-9
    lr: float = 1.0

    # Regularization is scaled by N_obs / reg_scale_ref_obs so that the
    # effective reg-to-data ratio is invariant to dataset size.  Without this,
    # halves and full converge to different (α, β) because halves are
    # relatively 2× more regularised.  Interpret every `lam_*` below as
    # "weight per `reg_scale_ref_obs` observations".
    # Default 2.2 M ≈ one half of the tubulin dataset — where the σ-Cauchy
    # sweep was calibrated.  Other datasets should set this to their own
    # typical N_obs if they want the calibration to transfer.
    reg_scale_ref_obs: int = 2_200_000

    # Chunking / quadrature
    # 1 M obs/chunk → ~5 GB peak on A100 with ~0% throughput hit vs full-batch
    # (bench_closure on merged delays -91/-136/-182/-227).  0 disables chunking.
    obs_chunk_size: int = 1_000_000
    n_quad: int = 51
    n_sigma_quad: float = 5.0
    # Quadrature grid lower bound on signal.  Sweep over this showed a plateau
    # in [1e-4, 1e-2] where the grid fully contains the integrand's natural
    # support for every HKL — anything lower adds redundant zero-weight points.
    # Default 1e-4 is the most conservative value inside the plateau.
    #   None  → per-HKL Poisson detection floor: 0.5 · √(bg / n_obs)  (= adaptive,
    #           compresses weak HKLs toward a common value around 0.2–0.4)
    #   float → constant across all observations
    sig_lo_override: float | None = 1e-4

    # Truncation-bias correction
    apply_bias_correction: bool = True
