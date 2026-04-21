#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u

#SBATCH -o /das/work/p17/p17490/Peter/torchsx/sfx_scan/train_neural_correction.out
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 4:00:00
#SBATCH --mem=64G

"""First experiment: can a neural network learn per-observation corrections
from binary ROI shapes and geometric metadata?

Steps:
1. Load dark streams → SFXStream
2. Run joint fit → frozen (I₀, α, β)
3. Excise ROIs (21×21) from raw HDF5 data
4. Train neural correction network (50 epochs)
5. Monitor: does free-set NLL decrease?

If free-set NLL decreases → the net learns real corrections from spatial patterns.
"""

import sys
import os

# Unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, "/das/work/p17/p17490/Peter/torchsx")

from pathlib import Path
import torch

from torchsx.stream import SFXStream
from torchsx.merge._distribution import fit_gengauss
from torchsx.merge._noise import PoissonNoiseModel
from torchsx.merge._corrections import ScaleBCorrection
from torchsx.io._roi_store import ROIStore

# ── Configuration ──
STREAM_DIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/streams")
DELAYS = [-91, -136, -182, -227, -273, -45]
SPACEGROUP = "P-1"
OUTDIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/neural_correction")
OUTDIR.mkdir(parents=True, exist_ok=True)

ROI_PATH = OUTDIR / "rois.safetensors"
PATCH_SIZE = 21
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")

# ══════════════════════════════════════════════════════════════
# 1. Load dark streams
# ══════════════════════════════════════════════════════════════
print("\n── Loading streams ──")
streams = []
for d in DELAYS:
    p = STREAM_DIR / f"delay_{d}.stream"
    print(f"  {p.name} ...", end=" ", flush=True)
    st = SFXStream.from_stream(str(p), spacegroup=SPACEGROUP, n_workers=1)
    print(f"{st.n_crystals} crystals, {st.n_observations} obs")
    streams.append(st)

s = SFXStream.concat(streams)
print(f"\nTotal: {s.n_crystals} crystals, {s.n_observations} observations")

# ══════════════════════════════════════════════════════════════
# 2. Joint fit → frozen (I₀, α, β)
# ══════════════════════════════════════════════════════════════
FIT_KW = dict(
    poisson_noise=PoissonNoiseModel(gain=1.0),
    mode="joint",
    min_obs=20,
    n_shells=20,
    n_smooth_iter=15,
    smooth_lambda=0.3,
    joint_device=DEVICE,
)

print("\n── Step 1: Joint fit WITHOUT corrections ──")
joint_fit = fit_gengauss(
    s, n_em_cycles=1, scale_crystals=False, **FIT_KW,
)
print(f"  {joint_fit.hkl.shape[0]} HKLs, med(α)={joint_fit.alpha.median():.4f}")
print(f"  I₀: min={joint_fit.I0.min():.2f}, med={joint_fit.I0.median():.2f}, "
      f"max={joint_fit.I0.max():.2f}")

# ══════════════════════════════════════════════════════════════
# 3. Excise ROIs (if not cached)
# ══════════════════════════════════════════════════════════════
if ROI_PATH.exists():
    print(f"\n── Loading cached ROIs from {ROI_PATH} ──")
    roi_store = ROIStore(str(ROI_PATH))
else:
    print(f"\n── Excising ROIs ({PATCH_SIZE}×{PATCH_SIZE}) ──")
    roi_store = s.excise_rois(
        path=str(ROI_PATH),
        patch_size=PATCH_SIZE,
        n_workers=4,
    )
    print(f"  Saved to {ROI_PATH}")

patches_shape = roi_store["patches"].shape
print(f"  ROI shape: {patches_shape}")

# ══════════════════════════════════════════════════════════════
# 4. EM: alternate neural net training ↔ refit (I₀, α)
# ══════════════════════════════════════════════════════════════
from torchsx.merge._train_correction import train_correction, TrainingConfig

config = TrainingConfig(
    n_epochs=30,
    lr=1e-3,
    weight_decay=1e-4,
    n_crystals_per_batch=512,
    patience=5,
    n_quad=21,
    device=DEVICE,
    d_model=128,
    n_layers=3,
    n_heads=4,
    patch_embed_dim=32,
    dropout=0.1,
)

N_EM_OUTER = 3
noise = PoissonNoiseModel(gain=1.0)

for em_round in range(N_EM_OUTER):
    print(f"\n{'='*60}")
    print(f"  EM round {em_round+1}/{N_EM_OUTER}")
    print(f"{'='*60}")

    # ── S-step: train neural correction on RAW observations ──
    # No previous corrections applied — net learns everything from scratch
    print(f"\n── S-step: Train neural correction (raw observations) ──")
    print(f"  Frozen I₀: med={joint_fit.I0.median():.2f}, α: med={joint_fit.alpha.median():.4f}")

    model, log = train_correction(
        stream=s,
        roi_store=roi_store,
        joint_fit=joint_fit,
        noise_model=noise,
        config=config,
    )

    print(f"  Train NLL: {log.train_nll[0]:.4f} → {log.train_nll[log.best_epoch-1]:.4f}")
    print(f"  Free NLL:  {log.free_nll[0]:.4f} → {log.free_nll[log.best_epoch-1]:.4f}")

    if log.free_nll[log.best_epoch-1] < log.free_nll[0]:
        delta = log.free_nll[0] - log.free_nll[log.best_epoch-1]
        print(f"  ✓ Free-set NLL decreased by {delta:.4f}")
    else:
        print(f"  ✗ Free-set NLL did not decrease — stopping EM")
        break

    # ── D-step: refit (I₀, α) with neural corrections applied ──
    print(f"\n── D-step: Refit (I₀, α) with neural corrections ──")

    # Compute corrections from trained model for ALL observations
    from torchsx.merge._train_correction import compute_corrections
    log_c_all = compute_corrections(
        model=model,
        stream=s,
        roi_store=roi_store,
        joint_fit=joint_fit,
        noise_model=noise,
        config=config,
    )
    # Apply corrections: create a corrected stream (or pass corrections to fit)
    # For now: store corrections and pass via a FixedCorrection wrapper
    from torchsx.merge._corrections import FixedCorrection
    fixed_corr = FixedCorrection(log_c_all)

    joint_fit = fit_gengauss(
        s, n_em_cycles=1, scale_crystals=False,
        corrections=[fixed_corr],
        **FIT_KW,
    )
    print(f"  Refitted: med(I₀)={joint_fit.I0.median():.2f}, med(α)={joint_fit.alpha.median():.4f}")

# ══════════════════════════════════════════════════════════════
# 5. Summary
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"EM complete after {em_round+1} rounds")
print(f"  Final I₀: med={joint_fit.I0.median():.2f}, α: med={joint_fit.alpha.median():.4f}")
print(f"{'='*60}")

# Save model
model_path = OUTDIR / "neural_correction.pt"
torch.save({
    "model_state": model.state_dict(),
    "config": config,
    "log": log,
}, str(model_path))
print(f"Saved model to {model_path}")
