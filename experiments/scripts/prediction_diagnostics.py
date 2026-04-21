"""Compare CrystFEL observations vs forward-model predictions.

Uses CrystFEL's own orientation (UB from stream), beam energy, and geometry
— no refinement applied (delta_omega = 0). This measures how well our
prediction engine reproduces what CrystFEL indexed.

Usage:
    python scripts/prediction_diagnostics.py [sfx_path] [--n-crystals N]
"""

from __future__ import annotations

import argparse
import time

import torch


def main():
    parser = argparse.ArgumentParser(description="Prediction diagnostics")
    parser.add_argument(
        "sfx_path",
        nargs="?",
        default="sfx_scan/streams/delay_-45.sfx",
        help="Path to .sfx file",
    )
    parser.add_argument(
        "--n-crystals", type=int, default=0,
        help="Limit to first N crystals (0 = all)",
    )
    parser.add_argument(
        "--sigma-cutoff", type=float, default=3.0,
        help="Excitation error cutoff in sigma units",
    )
    parser.add_argument(
        "--relp-sigma", type=float, default=None,
        help="Relp width in A^-1 (default: use profile_rad from stream)",
    )
    parser.add_argument(
        "--d-min", type=float, default=None,
        help="Resolution cutoff in A (default: min of per-crystal resolutions)",
    )
    parser.add_argument(
        "--no-friedel", action="store_true",
        help="Disable Friedel-mate matching (exact HKL only)",
    )
    args = parser.parse_args()

    from torchsx import SFXStream
    from torchsx.scaling import CrystalOrientation, prepare_scaler_inputs
    from torchsx.forward import RefinableGeometry
    from torchsx.forward.predict import (
        predict_reflections,
        compare_predictions,
    )

    # --- Load data ---
    print(f"Loading {args.sfx_path} ...")
    t0 = time.time()
    stream = SFXStream.load(args.sfx_path)
    core = stream._core
    print(
        f"  Loaded in {time.time() - t0:.1f}s: "
        f"{core.n_observations:,} obs, {core.n_crystals:,} crystals, "
        f"{core.n_images:,} images"
    )

    # --- Optionally limit crystals ---
    if args.n_crystals > 0 and args.n_crystals < core.n_crystals:
        from torchsx.merge._filter import filter_crystals
        mask = torch.zeros(core.n_crystals, dtype=torch.bool)
        mask[:args.n_crystals] = True
        core = filter_crystals(core, mask)
        print(
            f"  Filtered to {core.n_crystals} crystals, "
            f"{core.n_observations:,} obs"
        )

    # --- Build orientation from CrystFEL's UB (no refinement) ---
    UB = torch.stack([core.astar, core.bstar, core.cstar], dim=-1)
    orientation = CrystalOrientation(UB)
    # delta_omega stays at 0 — pure CrystFEL orientation

    geometry = RefinableGeometry(core.geometry)
    wavelength = 12398.4 / core.beam_energy.median().item()

    # Use median profile_rad as relp_sigma if not specified
    relp_sigma = args.relp_sigma
    if relp_sigma is None:
        relp_sigma = core.profile_rad.median().item()
        if relp_sigma <= 0 or relp_sigma > 1.0:
            relp_sigma = 0.05  # fallback
    print(f"\n  Wavelength: {wavelength:.4f} A")
    print(f"  Beam energy: {core.beam_energy.median().item():.1f} eV")
    print(f"  Relp sigma: {relp_sigma:.4f} A^-1")
    print(f"  Sigma cutoff: {args.sigma_cutoff}")

    # Resolution: use provided d_min or minimum per-crystal resolution
    if args.d_min is not None:
        d_min = args.d_min
    else:
        d_min = core.resolution.min().item()
        d_min = max(d_min, 0.8)  # don't go below 0.8 A
    print(f"  d_min: {d_min:.2f} A (per-crystal range: {core.resolution.min().item():.2f}-{core.resolution.max().item():.2f} A)")

    # --- Predict reflections ---
    print(f"\nPredicting reflections ...")
    t0 = time.time()
    predicted = predict_reflections(
        orientation, geometry, core.cell,
        d_min=d_min,
        wavelength=wavelength,
        sigma_cutoff=args.sigma_cutoff,
        relp_sigma=relp_sigma,
    )
    t_pred = time.time() - t0
    print(f"  {predicted.hkl.shape[0]:,} predictions in {t_pred:.1f}s")

    # --- Compare ---
    print(f"\nComparing predictions vs observations ...")
    t0 = time.time()
    diag = compare_predictions(predicted, core, core.cell, n_shells=10, friedel=not args.no_friedel)
    t_diag = time.time() - t0
    print(f"  Done in {t_diag:.1f}s\n")

    print(diag.summary())

    # --- Extra details ---
    print(f"\n--- Excitation error stats ---")
    if diag.excitation_error_matched.numel() > 0:
        em = diag.excitation_error_matched.abs()
        print(
            f"  Matched: median |r| = {em.median().item():.5f} A^-1, "
            f"mean = {em.mean().item():.5f}, max = {em.max().item():.5f}"
        )
    if diag.excitation_error_predicted_only.numel() > 0:
        ep = diag.excitation_error_predicted_only.abs()
        print(
            f"  Predicted-only: median |r| = {ep.median().item():.5f} A^-1, "
            f"mean = {ep.mean().item():.5f}, max = {ep.max().item():.5f}"
        )

    # --- Per-crystal stats ---
    pc = diag.per_crystal_counts
    if pc.shape[0] > 0:
        per_crystal_hit = pc[:, 0] / pc[:, 1].clamp(min=1)
        per_crystal_rec = pc[:, 0] / pc[:, 2].clamp(min=1)
        active = pc[:, 2] > 0  # crystals with observations
        if active.sum() > 0:
            print(f"\n--- Per-crystal stats ({active.sum().item()} active crystals) ---")
            print(
                f"  Hit rate:      "
                f"median={per_crystal_hit[active].median().item():.1%}, "
                f"mean={per_crystal_hit[active].mean().item():.1%}, "
                f"min={per_crystal_hit[active].min().item():.1%}"
            )
            print(
                f"  Recovery rate: "
                f"median={per_crystal_rec[active].median().item():.1%}, "
                f"mean={per_crystal_rec[active].mean().item():.1%}, "
                f"min={per_crystal_rec[active].min().item():.1%}"
            )

            # Flag crystals with low recovery (possible indexing problems)
            low_rec = (per_crystal_rec[active] < 0.3).sum().item()
            high_oo = ((pc[active, 2] - pc[active, 0]) / pc[active, 2].clamp(min=1) > 0.3).sum().item()
            print(f"  Crystals with recovery < 30%: {low_rec}")
            print(f"  Crystals with >30% observed-only: {high_oo}")

    # --- Sweep sigma_cutoff ---
    print(f"\n--- Sigma cutoff sweep ---")
    for sc in [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]:
        pred_sc = predict_reflections(
            orientation, geometry, core.cell,
            d_min=d_min, wavelength=wavelength,
            sigma_cutoff=sc, relp_sigma=relp_sigma,
        )
        diag_sc = compare_predictions(pred_sc, core, core.cell, n_shells=0, friedel=not args.no_friedel)
        print(
            f"  sigma_cutoff={sc:4.1f}: "
            f"pred={pred_sc.hkl.shape[0]:>8,}, "
            f"matched={diag_sc.n_matched:>7,}, "
            f"hit={diag_sc.hit_rate:.1%}, "
            f"rec={diag_sc.recovery_rate:.1%}"
        )

    # --- Sweep relp_sigma ---
    print(f"\n--- Relp sigma sweep (sigma_cutoff={args.sigma_cutoff}) ---")
    for rs in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
        pred_rs = predict_reflections(
            orientation, geometry, core.cell,
            d_min=d_min, wavelength=wavelength,
            sigma_cutoff=args.sigma_cutoff, relp_sigma=rs,
        )
        diag_rs = compare_predictions(pred_rs, core, core.cell, n_shells=0, friedel=not args.no_friedel)
        print(
            f"  relp_sigma={rs:.3f}: "
            f"pred={pred_rs.hkl.shape[0]:>8,}, "
            f"matched={diag_rs.n_matched:>7,}, "
            f"hit={diag_rs.hit_rate:.1%}, "
            f"rec={diag_rs.recovery_rate:.1%}"
        )


if __name__ == "__main__":
    main()
