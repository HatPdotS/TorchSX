#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python
"""Histograms of per-crystal and per-image metadata from stream files."""

from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchsx.stream import SFXStream

STREAM_DIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/streams")
DELAYS = [-91, -136, -182, -227, -273]
SPACEGROUP = "P-1"
OUTDIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/resolution_study")
N_WORKERS = 16

# ── Load ──
streams = []
for d in DELAYS:
    stem = STREAM_DIR / f"delay_{d}.stream"
    cache = stem.with_suffix(".sfx")
    print(f"Loading {stem.name} ...")
    s = SFXStream.from_stream(stem, spacegroup=SPACEGROUP,
                              cache=cache, n_workers=N_WORKERS)
    print(f"  {s.n_crystals} crystals, {s.n_observations} obs")
    streams.append(s)

s = SFXStream.concat(streams)
c = s._core
print(f"\nTotal: {s.n_crystals} crystals, {s.n_observations} obs, "
      f"{c.beam_energy.shape[0]} images")

OUTDIR.mkdir(parents=True, exist_ok=True)

# ── Figure 1: Cell parameters ──
cell = c.cell.numpy()  # (N_crystals, 6): a, b, c, α, β, γ
labels = ["a (Å)", "b (Å)", "c (Å)", "α (°)", "β (°)", "γ (°)"]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, (ax, lab) in enumerate(zip(axes.flat, labels)):
    vals = cell[:, i]
    ax.hist(vals, bins=100, color="C0", edgecolor="none")
    ax.axvline(np.median(vals), color="r", ls="--",
               label=f"med={np.median(vals):.3f}")
    ax.set_xlabel(lab)
    ax.set_ylabel("Count")
    ax.set_title(f"{lab}  (σ={np.std(vals):.4f})")
    ax.legend(fontsize=7)
fig.suptitle("Unit cell parameter distributions", fontsize=14)
fig.tight_layout()
fig.savefig(str(OUTDIR / "hist_cell_params.png"), dpi=150)
plt.close(fig)
print(f"Saved {OUTDIR / 'hist_cell_params.png'}")

# ── Figure 2: Crystal-level quality metrics ──
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# profile_radius
ax = axes[0, 0]
pr = c.profile_rad.numpy()
ax.hist(pr, bins=100, color="C1", edgecolor="none")
ax.set_xlabel("Profile radius (Å⁻¹)")
ax.set_title(f"Profile radius (med={np.median(pr):.5f})")

# resolution (d_min)
ax = axes[0, 1]
res = c.resolution.numpy()
ax.hist(res, bins=100, color="C2", edgecolor="none")
ax.set_xlabel("Resolution d_min (Å)")
ax.set_title(f"Resolution (med={np.median(res):.2f} Å)")

# n_refl
ax = axes[0, 2]
nr = c.n_refl.numpy()
ax.hist(nr, bins=100, color="C3", edgecolor="none")
ax.set_xlabel("Number of reflections")
ax.set_title(f"n_refl (med={np.median(nr):.0f})")

# det_shift_x
ax = axes[1, 0]
dsx = c.det_shift_x[c.image_id.long()].numpy()
ax.hist(dsx, bins=100, color="C4", edgecolor="none")
ax.set_xlabel("det_shift_x (mm)")
ax.set_title(f"Detector shift X (σ={np.std(dsx):.3f})")

# det_shift_y
ax = axes[1, 1]
dsy = c.det_shift_y[c.image_id.long()].numpy()
ax.hist(dsy, bins=100, color="C5", edgecolor="none")
ax.set_xlabel("det_shift_y (mm)")
ax.set_title(f"Detector shift Y (σ={np.std(dsy):.3f})")

# beam_energy per crystal (via image_id)
ax = axes[1, 2]
be = c.beam_energy[c.image_id.long()].numpy()
ax.hist(be, bins=100, color="C6", edgecolor="none")
ax.set_xlabel("Beam energy (eV)")
ax.set_title(f"Beam energy (med={np.median(be):.0f} eV)")

fig.suptitle("Crystal / image quality metrics", fontsize=14)
fig.tight_layout()
fig.savefig(str(OUTDIR / "hist_crystal_metrics.png"), dpi=150)
plt.close(fig)
print(f"Saved {OUTDIR / 'hist_crystal_metrics.png'}")

# ── Figure 3: Reciprocal vectors — |a*|, |b*|, |c*| ──
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, (ax, name) in enumerate(zip(axes, ["a*", "b*", "c*"])):
    vec = [c.astar, c.bstar, c.cstar][i]
    mag = vec.norm(dim=1).numpy()
    ax.hist(mag, bins=100, color=f"C{i}", edgecolor="none")
    ax.set_xlabel(f"|{name}| (Å⁻¹)")
    ax.set_title(f"|{name}| (med={np.median(mag):.5f}, σ={np.std(mag):.5f})")
fig.suptitle("Reciprocal vector magnitudes", fontsize=14)
fig.tight_layout()
fig.savefig(str(OUTDIR / "hist_recip_vectors.png"), dpi=150)
plt.close(fig)
print(f"Saved {OUTDIR / 'hist_recip_vectors.png'}")

# ── Figure 4: 2D correlations ──
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# n_refl vs resolution
ax = axes[0]
ax.scatter(res, nr, s=1, alpha=0.1, color="C0")
ax.set_xlabel("Resolution d_min (Å)")
ax.set_ylabel("n_refl")
ax.set_title("Reflections vs resolution")

# profile_radius vs resolution
ax = axes[1]
ax.scatter(res, pr, s=1, alpha=0.1, color="C1")
ax.set_xlabel("Resolution d_min (Å)")
ax.set_ylabel("Profile radius (Å⁻¹)")
ax.set_title("Profile radius vs resolution")

# det_shift_x vs det_shift_y
ax = axes[2]
ax.scatter(dsx, dsy, s=1, alpha=0.1, color="C4")
ax.set_xlabel("det_shift_x (mm)")
ax.set_ylabel("det_shift_y (mm)")
ax.set_title("Detector shifts")
ax.set_aspect("equal")

fig.tight_layout()
fig.savefig(str(OUTDIR / "hist_correlations.png"), dpi=150)
plt.close(fig)
print(f"Saved {OUTDIR / 'hist_correlations.png'}")

# ── Print summary stats ──
print(f"\n── Summary statistics ──")
print(f"  Cell a: {cell[:,0].mean():.3f} ± {cell[:,0].std():.4f} Å")
print(f"  Cell b: {cell[:,1].mean():.3f} ± {cell[:,1].std():.4f} Å")
print(f"  Cell c: {cell[:,2].mean():.3f} ± {cell[:,2].std():.4f} Å")
print(f"  Cell α: {cell[:,3].mean():.3f} ± {cell[:,3].std():.4f}°")
print(f"  Cell β: {cell[:,4].mean():.3f} ± {cell[:,4].std():.4f}°")
print(f"  Cell γ: {cell[:,5].mean():.3f} ± {cell[:,5].std():.4f}°")
print(f"  Profile rad: {np.median(pr):.5f} ± {np.std(pr):.5f} Å⁻¹")
print(f"  Resolution: {np.median(res):.2f} ± {np.std(res):.2f} Å")
print(f"  n_refl: {np.median(nr):.0f} ± {np.std(nr):.0f}")
print(f"  Beam energy: {np.median(be):.0f} ± {np.std(be):.0f} eV")

# Cell parameter std as fractional strain
print(f"\n── Cell parameter variation as strain ──")
for i, name in enumerate(["a", "b", "c"]):
    frac = cell[:, i].std() / cell[:, i].mean()
    print(f"  Δ{name}/{name} = {frac:.5f}  ({frac*100:.3f}%)")
