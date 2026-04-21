#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u
"""Rank-concordance refinement of crystal orientations and wavelengths."""

import sys
from pathlib import Path
import torch
import numpy as np

# Ensure imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from partiality._data import load_and_prepare
from partiality._excitation import ExcitationErrorModel
from partiality._concordance import RankConcordanceLoss
from partiality._refine import refine_orientations, RefinementConfig
from partiality._diagnostics import evaluate_concordance, plot_diagnostics


def main():
    stream_dir = Path(__file__).resolve().parent.parent / "sfx_scan" / "streams"
    sfx_path = [
        stream_dir / "delay_-91.sfx",
        stream_dir / "delay_-136.sfx",
        stream_dir / "delay_-182.sfx",
        stream_dir / "delay_-227.sfx",
    ]
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load and prepare
    pair_data = load_and_prepare(
        sfx_path,
        device=device,
        I_threshold=20.0,
        min_mult=3,
        max_pairs_per_hkl=500,
    )

    # Compute initial |r|
    model_init = ExcitationErrorModel(
        pair_data.n_crystals, pair_data.UB, pair_data.lambda_0,
    ).to(device)
    with torch.no_grad():
        abs_r_before = model_init.abs_excitation_error(
            pair_data.hkl_obs, pair_data.crystal_id,
        ).cpu().numpy()

    # Initial diagnostics
    print("\nBefore refinement:")
    stats_before = evaluate_concordance(
        abs_r_before, pair_data.I_corr.cpu().numpy(),
        pair_data.hkl_id.cpu().numpy(), pair_data.crystal_id.cpu().numpy(),
    )
    print(f"  median ρ(I, |r|) = {stats_before['median_rho']:.4f}  "
          f"({stats_before['n_hkl']} HKLs, {stats_before['frac_negative']*100:.1f}% negative)")

    # Refine
    config = RefinementConfig(
        n_steps=300,
        lr_omega=1e-4,
        lr_lambda=1e-5,
        beta_start=5.0,
        beta_end=30.0,
        restraint_omega=300.0,   # optimal from scan
        restraint_lambda=2e4,
        device=device,
    )
    result = refine_orientations(pair_data, config)

    # Compute refined |r|
    with torch.no_grad():
        abs_r_after = result.model.abs_excitation_error(
            pair_data.hkl_obs, pair_data.crystal_id,
        ).cpu().numpy()

    # After diagnostics
    print("\nAfter refinement:")
    stats_after = evaluate_concordance(
        abs_r_after, pair_data.I_corr.cpu().numpy(),
        pair_data.hkl_id.cpu().numpy(), pair_data.crystal_id.cpu().numpy(),
    )
    print(f"  median ρ(I, |r|) = {stats_after['median_rho']:.4f}  "
          f"({stats_after['n_hkl']} HKLs, {stats_after['frac_negative']*100:.1f}% negative)")

    # Plot
    print("\nGenerating figures...")
    plot_diagnostics(result, pair_data, abs_r_before, abs_r_after, output_dir)

    # Summary
    omega = result.delta_omega_xy.numpy()
    omega_mag = np.degrees(np.sqrt((omega ** 2).sum(axis=1))) * 1000
    dlam = result.delta_lambda.numpy()

    print("\n" + "=" * 65)
    print("RANK-CONCORDANCE REFINEMENT SUMMARY")
    print("=" * 65)
    print(f"  Crystals:     {pair_data.n_crystals:,}")
    print(f"  Pairs:        {pair_data.n_pairs:,}")
    print(f"  Steps:        {config.n_steps}")
    print(f"  ρ before:     {stats_before['median_rho']:.4f}")
    print(f"  ρ after:      {stats_after['median_rho']:.4f}")
    print(f"  Δρ:           {stats_after['median_rho'] - stats_before['median_rho']:.4f}")
    print(f"  |δω| median:  {np.median(omega_mag):.1f} millideg")
    print(f"  |δλ| median:  {np.median(np.abs(dlam))*1e4:.2f} ×10⁻⁴ Å")
    print("=" * 65)

    # Save
    torch.save({
        "delta_omega_xy": result.delta_omega_xy,
        "delta_lambda": result.delta_lambda,
        "loss_history": result.loss_history,
    }, output_dir / "refinement_result.pt")
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
