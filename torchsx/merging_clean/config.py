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

    # Truncation-bias correction (Mills ratio).  Made redundant by
    #   (a) sig_lo_override = 1e-4 (grid captures integrand support)
    #   (b) the bg-anchor in data_nll (genuinely-dark HKLs are handled by the
    #       background hypothesis instead of being pulled down post-hoc).
    # Defaults to False; flip to True to re-enable the Gaussian-tail Mills
    # correction for datasets where those two assumptions don't hold.
    apply_bias_correction: bool = False

    # Background-hypothesis mixing.  Three modes:
    #   "off"     — no bg term; pure GenGauss+Poisson signal likelihood.
    #   "global"  — append log Poisson(n|bg) as a (K+1)th logsumexp term;
    #               implicit 50/50 mixture per observation (fixed).
    #   "per_hkl" — learn a per-HKL Bernoulli weight π_h ∈ [0, 1]:
    #               p(obs) = (1−π_h)·p_sig_marginal + π_h·p_bg.
    #               Dark HKLs commit π_h → 1 and their data gradient on I₀
    #               vanishes, so the floor prior (or just gradient-zero) can
    #               place I₀ at 0 cleanly.  Bright HKLs commit π_h → 0 and
    #               the GenGauss scatter model is unperturbed.
    bg_mode: str = "global"

    # Beta(a, b) prior on π_h when bg_mode = "per_hkl".  Choose a < b to bias
    # toward signal (most HKLs really do have signal); a > b biases toward
    # dark.  Strength scales with lam_pi_beta (set to 0 for uniform prior).
    pi_beta_a: float = 1.0
    pi_beta_b: float = 2.0
    lam_pi_beta: float = 1.0

    # French–Wilson log-prior on I₀ (per-shell, gradient-flowing ⟨I⟩).
    #   acentric:  -log p = I/Σ + log Σ
    #   centric:   -log p = I/(2Σ) + 0.5·log(I/Σ) + 0.5·log Σ
    # Σ = ⟨I⟩_shell is recomputed each forward pass from exp(log_I0) via
    # scatter_add over `shell_id_fw` (no detach — with O(n_HKL_per_shell)
    # ≫ 1 the self-coupling is negligible and the prior behaves as EM).
    # Shell-calibrated shrinkage replaces the coarse `lam_I0_floor`
    # Gaussian pull. Off by default; opt-in via a positive λ.
    lam_frac_wilson: float = 0.0
    fw_n_shells: int = 20
    fw_eps: float = 1e-6

    # Wilson **distribution-match** prior — KL between the empirical Gaussian
    # fit of log(I/Σ) and the Wilson-target (mean, var), per shell × centricity
    # group. Preserves the *shape* of the Wilson distribution (wide spread,
    # exponential decay) rather than shrinking all HKLs toward the mode.
    #   lam_wilson_match = 0      → disabled (default)
    #   lam_wilson_match ~ 1e-3   → soft shape-matching (recommended starting point)
    #   lam_wilson_match ~ 1e-1   → stronger; comparable to per-HKL MAP prior
    # Groups with fewer than `wilson_match_min_count` HKLs are skipped (bad
    # empirical-moment estimates).
    lam_wilson_match: float = 0.0
    wilson_match_min_count: int = 20

    # Detector gain (ADU per photon).  The Poisson likelihood assumes the
    # intensities fed to the loss are photon counts, not ADU; with an
    # integrating detector at protein wavelengths this is ~30–400 ADU/photon.
    # Treating ADU as photons underestimates shot noise by a factor of `gain`,
    # which biases α/β and I₀.
    #   "stream"  → read `adu_per_eV` from the geometry header and multiply
    #               by each image's `photon_energy_eV`.  Most robust default.
    #   "empirical" → empirical Mandel calibration on the negatives
    #               (`PoissonNoiseModel.calibrate_gain`).  Requires enough
    #               negative background-subtracted observations.
    #   float    → explicit gain value (ADU/photon), e.g. 330.0.
    gain: str | float = "stream"
