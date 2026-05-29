"""``torchsx-merge`` — production CLI for the May-2026 merging recipe.

Stream + spacegroup → MTZ.  Bakes in the defaults validated on tubulin
(T7, R-free = 0.2285) and BR (R-free = 0.180, 1.24 pp better than
partialator unity):

* ``scatter_shape="aniso_quad"`` via the two-pass swap (free per-HKL →
  LSTSQ projection onto the 14-parameter symmetric quadratic form →
  refine again).
* ``likelihood="gaussian_polynomial"`` with σ² = max(n, bg) — symmetric
  Gaussian, Poisson-moment-match in σ.  Pass ``--variance`` to tweak.
* Wilson-aniso (4.0), Miller-graph smoothness (500), per-crystal scale
  ridge (1.0), min_obs=5 — the canonical prior recipe.
* Posterior with flat prior on a MAP-centred Δ-grid → MTZ with both
  ``F-obs/SIGF-obs`` and ``I-obs/SIGI-obs``.

The output MTZ is in torchsx-native orientation.  To refine against a
published model, transform the model PDB once via
``transform_pdb_to_torchsx_orientation.py``.

Examples
--------
::

    torchsx-merge BR_sub.sfx "P 63" --output unified_br.mtz
    torchsx-merge BR_sub.sfx "P 63" --output out.mtz --scatter-shape global
    torchsx-merge tubulin.sfx "P 21 21 21" -o out.mtz --variance "0,2.4,0"
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from torchsx.stream import SFXStream


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="torchsx-merge",
        description="Streamlined merging following the May 2026 recipe.",
    )
    p.add_argument("stream", type=Path, help="Input .sfx (or .stream) file")
    p.add_argument(
        "spacegroup",
        help='Crystallographic space group (e.g. "P 63"). Stamped into the '
        "output MTZ; mismatches against the stream's recorded spacegroup "
        "are surfaced as a warning.",
    )
    p.add_argument(
        "-o", "--output", type=Path, required=True,
        help="Output MTZ path (will be created/overwritten)",
    )
    p.add_argument(
        "--scatter-shape",
        choices=["aniso_quad", "global", "free"],
        default="aniso_quad",
        help="α/β parametrisation (default: aniso_quad — two-pass: free → "
        "LSTSQ-warm-start aniso_quad → refine. global is simpler and only "
        "marginally worse; free preserves per-HKL freedom for diagnostics.)",
    )
    p.add_argument("--device", default="cuda", help="'cpu' or 'cuda' (default cuda)")
    p.add_argument(
        "--gain", default="auto",
        help="ADU/photon: 'auto' | 'stream' | 'empirical' | float (default auto)",
    )
    p.add_argument("--n-points", type=int, default=513)
    p.add_argument("--min-obs", type=int, default=5)
    p.add_argument("--smooth", type=float, default=500.0)
    p.add_argument("--wilson-aniso", type=float, default=4.0)
    p.add_argument("--scale-reg-weight", type=float, default=1.0)
    p.add_argument("--max-iter", type=int, default=150)
    p.add_argument(
        "--obs-chunk-size", type=int, default=-1,
        help="Per-obs chunk for the data target. Default (-1) auto-sizes to "
        "max(1024, min(50_000, 75M / n_points)). 0 disables chunking but "
        "OOMs on real-sized SFX (37M obs).",
    )
    p.add_argument(
        "--prior", choices=["flat", "wilson", "wilson_aniso"], default="flat",
        help="Posterior prior for the integrated MTZ (default flat)",
    )
    p.add_argument(
        "--hessian-filter", default=None,
        help="If set, restrict the integration-grid Laplace Hessian to "
             "targets whose name starts with this prefix.  Use "
             "'merging/data' for data-only Hessian — gives σ reflecting "
             "the likelihood's curvature, decoupled from prior shrinkage.",
    )
    p.add_argument(
        "--also-write-map", action="store_true",
        help="In addition to the integrated MTZ at --output, write a "
             "second MTZ at <output_stem>_map.mtz built from MAP "
             "I0 = exp(log_I0_MAP) and σ_I0 from a data-only Laplace "
             "Hessian (delta method, σ_F = σ_I0 / (2·F)). MAP itself "
             "still benefits from prior shrinkage; only σ is prior-free.",
    )
    p.add_argument(
        "--no-friedel", dest="merge_friedel",
        action="store_false", default=True,
        help="Don't merge Friedel mates on stream load (default: merge)",
    )
    p.add_argument(
        "--r-peak", type=float, default=7.0,
        help="CrystFEL peak-ring radius in pixels (default 7). Must match "
             "the indexamajig integration radii of the upstream stream — "
             "Rhodopsin uses 2/4/6, tubulin 2/3/6; mismatch corrupts both "
             "the truncnorm gain calibration and bg-in-peak.",
    )
    p.add_argument(
        "--r-gap", type=float, default=9.0,
        help="CrystFEL gap-ring radius in pixels (default 9).",
    )
    p.add_argument(
        "--r-bg", type=float, default=11.0,
        help="CrystFEL bg-ring outer radius in pixels (default 11).",
    )
    p.add_argument(
        "--likelihood",
        choices=["gaussian_polynomial", "gaussian_polynomial_bg_mixture", "poisson"],
        default="gaussian_polynomial",
        help="Per-observation detection likelihood.  Default "
             "'gaussian_polynomial' uses a symmetric Gaussian with "
             "σ² = g₀ + g₁·μ_obs + g₂·μ_obs² evaluated at the data-side "
             "Poisson MLE μ_obs = max(n, bg) — well-conditioned in the "
             "Poisson-limited regime.  Coefficients set via --variance "
             "(default 'stream' = (0, 1, 0), pure Poisson moment-match).  "
             "'gaussian_polynomial_bg_mixture' adds a per-observation "
             "background-hypothesis branch mixed with weight q(|r|) = "
             "sigmoid(q_a + q_b · |r| · 1e3); the two scalars (q_a, q_b) "
             "are co-fit with LBFGS.  "
             "Pass --likelihood poisson for the raw Poisson PMF (debug "
             "fallback; T7 production basin uses gaussian_polynomial).",
    )
    p.add_argument(
        "--variance",
        default="stream",
        help="σ² polynomial coefficients for --likelihood gaussian_polynomial.  "
             "'stream' (default) → (g0, g1, g2) = (0, 1, 0) — pure Poisson "
             "moment-match at the stream-derived gain.  Triplet 'g0,g1,g2' → "
             "explicit coefficients in photons²: g₀ = read-noise + bg-est "
             "variance floor, g₁ = Poisson-dispersion factor (g₁=γ for a "
             "calibrated overdispersion γ), g₂ = (RMS relative gain "
             "uncertainty)² for Jungfrau-style gain switching.  Ignored "
             "when --likelihood poisson.",
    )
    p.add_argument(
        "--refine-detector-noise", action="store_true",
        help="EXPERIMENTAL: optimise (g₀, g₁, g₂) jointly with the rest of "
             "the model via LBFGS.  At production scale (50k tubulin) this "
             "branch converges to non-physical g₁≈14 and is NaN-prone in "
             "single precision; the data NLL does not pin these "
             "coefficients.  Keep off for production.",
    )
    return p


def _parse_gain(value):
    if value in ("auto", "stream", "empirical"):
        return value
    try:
        return float(value)
    except ValueError:
        raise SystemExit(
            "--gain must be a float or one of {auto, stream, empirical}, "
            f"got {value!r}"
        )


def _parse_variance(value: str) -> tuple[float, float, float]:
    """``stream`` → (0, 1, 0); ``"g0,g1,g2"`` → explicit triplet."""
    if value == "stream":
        return (0.0, 1.0, 0.0)
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise SystemExit(
            f"--variance must be 'stream' or a triplet 'g0,g1,g2', got {value!r}"
        )
    try:
        g0, g1, g2 = (float(p) for p in parts)
    except ValueError:
        raise SystemExit(
            f"--variance triplet entries must be floats, got {value!r}"
        )
    return (g0, g1, g2)


def _override_spacegroup(stream: SFXStream, requested: str) -> SFXStream:
    """Stamp `requested` into the stream's core; warn on mismatch."""
    core = stream._core
    have = (core.spacegroup or "").strip()
    want = requested.strip()
    if have and have != want:
        print(
            f"[merge] WARN: stream spacegroup={have!r} differs from requested "
            f"{want!r}; using requested.",
            flush=True,
        )
    if have == want:
        return stream
    stream._core = replace(core, spacegroup=want)
    return stream


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    args.gain = _parse_gain(args.gain)
    if args.obs_chunk_size < 0:
        args.obs_chunk_size = max(1024, min(50_000, 75_000_000 // args.n_points))

    var_const, var_linear, var_quadratic = _parse_variance(args.variance)
    if args.likelihood == "poisson" and args.variance != "stream":
        print(
            f"[merge] WARN: --variance {args.variance!r} ignored with "
            "--likelihood poisson (Poisson PMF has no variance polynomial).",
            file=sys.stderr,
            flush=True,
        )
    if args.refine_detector_noise:
        print(
            "[merge] EXPERIMENTAL: --refine-detector-noise on — (g0, g1, g2) "
            "are trainable. This branch does not converge usefully at "
            "production scale; use only for diagnostic experiments.",
            file=sys.stderr,
            flush=True,
        )

    print(
        f"[merge] stream={args.stream}  spacegroup={args.spacegroup!r}  "
        f"scatter_shape={args.scatter_shape}  device={args.device}  "
        f"gain={args.gain}  obs_chunk_size={args.obs_chunk_size}  "
        f"r_peak/r_gap/r_bg={args.r_peak}/{args.r_gap}/{args.r_bg}  "
        f"likelihood={args.likelihood}  "
        f"variance=({var_const:.4g},{var_linear:.4g},{var_quadratic:.4g})",
        flush=True,
    )

    stream = SFXStream.load(args.stream, merge_friedel=args.merge_friedel)
    stream = _override_spacegroup(stream, args.spacegroup)

    # Pass 1: free per-HKL α/β with the canonical prior recipe.  Gives the
    # gold-standard log_I0 and per-HKL α/β distribution that pass 2 will
    # project onto the lower-rank parametrisation.
    pass1_shape = "free" if args.scatter_shape == "aniso_quad" else args.scatter_shape
    print(f"[merge] pass 1: scatter_shape={pass1_shape!r}", flush=True)
    loss = stream.make_loss(
        n_points=args.n_points,
        gain=args.gain, min_obs=args.min_obs,
        scale="b_per_crystal",
        scale_reg_weight=args.scale_reg_weight,
        scatter_shape=pass1_shape,
        smooth=args.smooth,
        wilson_aniso=args.wilson_aniso,
        obs_chunk_size=args.obs_chunk_size,
        device=args.device,
        r_peak=args.r_peak, r_gap=args.r_gap, r_bg=args.r_bg,
        likelihood=args.likelihood,
        var_const=var_const,
        var_linear=var_linear,
        var_quadratic=var_quadratic,
        trainable_polynomial_var=args.refine_detector_noise,
    )
    print(
        f"[merge] pass 1: {loss.n_valid} HKLs, "
        f"{sum(p.numel() for p in loss._collect_parameters())} trainable elements",
        flush=True,
    )
    loss.refine(max_iter=args.max_iter, history_size=10)
    print(f"[merge] pass 1 done.  final_nll={loss.final_nll:.4e}", flush=True)

    # Pass 2 (only if user picked aniso_quad): swap to lower-rank, refit.
    if args.scatter_shape == "aniso_quad":
        print(
            "[merge] swap_scatter_shape('aniso_quad') — LSTSQ-projecting "
            "per-HKL α/β onto the 7-param symmetric quadratic basis",
            flush=True,
        )
        loss.swap_scatter_shape("aniso_quad")
        n_train = sum(p.numel() for p in loss._collect_parameters())
        print(f"[merge] pass 2 trainable elements = {n_train}", flush=True)
        loss.refine(max_iter=args.max_iter, history_size=10)
        print(f"[merge] pass 2 done.  final_nll={loss.final_nll:.4e}", flush=True)

    # Posterior integration → MTZ in torchsx-native orientation.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[merge] integrating posterior(prior={args.prior!r}"
        + (f", hessian_filter={args.hessian_filter!r}"
           if args.hessian_filter else "")
        + ") ...",
        flush=True,
    )
    posterior = loss.posterior(
        prior=args.prior, hessian_filter=args.hessian_filter,
    )
    data = posterior.integrate()
    data.write_mtz(str(args.output))
    print(f"[merge] wrote {args.output}", flush=True)

    if args.also_write_map:
        from torchsx.merging.posterior import build_map_only_data
        map_path = args.output.parent / (args.output.stem + "_map.mtz")
        print(f"[merge] also writing MAP MTZ (hessian_filter='merging/data') ...",
              flush=True)
        map_data = build_map_only_data(loss, hessian_filter="merging/data")
        map_data.write_mtz(str(map_path))
        print(f"[merge] wrote {map_path}", flush=True)

    print(
        f"[merge] summary: n_hkl={loss.n_valid}  gain={loss.stats.get('gain'):.4g}  "
        f"final_nll={loss.final_nll:.6e}",
        flush=True,
    )
    if args.likelihood == "gaussian_polynomial_bg_mixture":
        # Show fitted (q_a, q_b) and the corresponding q at a few |r| anchors,
        # so the bench log captures the calibration the LBFGS converged to.
        import torch
        q_a = float(loss._data_target.params.q_a.item())
        q_b = float(loss._data_target.params.q_b.item())
        anchors = [0.0, 1.0, 3.0, 6.0, 10.0, 15.0]  # ×10⁻³ Å⁻¹
        q_vals = [float(torch.sigmoid(torch.tensor(q_a + q_b * r))) for r in anchors]
        anchor_str = "  ".join(f"q(|r|={r:>4.1f})={q:.3f}"
                                for r, q in zip(anchors, q_vals))
        print(f"[merge] bg_mixture: q_a={q_a:+.4f}  q_b={q_b:+.4f}", flush=True)
        print(f"[merge] bg_mixture: {anchor_str}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
