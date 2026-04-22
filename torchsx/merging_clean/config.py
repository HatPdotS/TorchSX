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

    # Weak Gaussian prior on log I₀ centred at `log_I0_floor_target`.
    # Designed so HKLs with no signal evidence (bg-anchor dominates) get
    # pulled toward the target value ≈ 0 intensity, while strong HKLs keep
    # their data MLE.  The transition is governed by MAP arithmetic:
    #   log I₀_MAP  ≈  (F_data·log I_MLE + τ·log I_target) / (F_data + τ)
    # with τ = lam_I0_floor and F_data ~ N_h.  With the default τ=0.01 the
    # prior loses to the data when N_h ≥ ~100; pure-bg HKLs (F_data≈0) go to
    # the target.  Off by default — opt-in via a positive λ.
    lam_I0_floor: float = 0.1
    log_I0_floor_target: float = -11.5   # log(10⁻⁵): "zero" intensity

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

    # LBFGS
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

    # Truncation-bias correction (Mills ratio).  Made redundant by
    #   (a) sig_lo_override = 1e-4 (grid captures integrand support)
    #   (b) the bg-anchor in data_nll (genuinely-dark HKLs are handled by the
    #       background hypothesis instead of being pulled down post-hoc).
    # Defaults to False; flip to True to re-enable the Gaussian-tail Mills
    # correction for datasets where those two assumptions don't hold.
    apply_bias_correction: bool = False
