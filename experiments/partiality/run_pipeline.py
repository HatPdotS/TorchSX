#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u
"""Run the iterative partiality-aware merging pipeline."""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from partiality._pipeline import run_pipeline, PipelineConfig

OUTDIR = Path(__file__).resolve().parent / "results"
OUTDIR.mkdir(exist_ok=True)


def main():
    from torchsx.stream import SFXStream

    stream_dir = Path(__file__).resolve().parent.parent / "sfx_scan" / "streams"
    names = ["delay_-91", "delay_-136", "delay_-182", "delay_-227"]

    streams = []
    for name in names:
        path = stream_dir / f"{name}.sfx"
        print(f"Loading {name}...")
        s = SFXStream.load(str(path))
        streams.append(s)
        print(f"  {s.n_observations:,} obs, {s.n_crystals:,} crystals")

    config = PipelineConfig(
        n_outer_cycles=4,
        n_em_cycles=10,
        rank_n_steps=200,
        rank_restraint_omega=300.0,
        rank_restraint_lambda=2e4,
        part_n_steps=500,
        part_lr=1e-3,
    )

    result = run_pipeline(streams, config)

    # Save
    save_dict = {
        "delta_omega_xy": result.state.excitation_model.delta_omega_xy.detach(),
        "delta_lambda": result.state.excitation_model.delta_lambda.detach(),
        "diagnostics": result.cycle_diagnostics,
    }
    if result.state.partiality_model is not None:
        save_dict["partiality_state_dict"] = result.state.partiality_model.state_dict()
    torch.save(save_dict, OUTDIR / "pipeline_result.pt")

    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    for d in result.cycle_diagnostics:
        print(f"  Cycle {d['cycle']}: ρ = {d['med_rho']:.4f}, "
              f"neg = {d['frac_neg']:.0f}%, σ_LN = {d['sigma_LN_med']:.4f}")
    print("=" * 60)
    print(f"\nSaved to {OUTDIR / 'pipeline_result.pt'}")


if __name__ == "__main__":
    main()
