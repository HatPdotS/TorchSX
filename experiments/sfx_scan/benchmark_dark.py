#!/das/work/p17/p17490/Peter/small_molecule_torchref/small_molecule_torchref/bin/python -u

#SBATCH -o /das/work/p17/p17490/Peter/torchsx/sfx_scan/benchmark_dark.out
#SBATCH -c 16
#SBATCH -p day
#SBATCH -t 4:00:00

"""Benchmark: merge dark data with GenGauss, refine against known structure.

Compare R-work / R-free with partialator unity baseline.
"""

import sys
from pathlib import Path

# Ensure torchsx is importable (editable install may need explicit path)
sys.path.insert(0, "/das/work/p17/p17490/Peter/torchsx")

import torch
import numpy as np

from torchsx.stream import SFXStream
from torchsx.merge._distribution import fit_gengauss, HKLDistribution
from torchsx.merge._corrections import ScaleBCorrection, PanelCorrection
from torchsx.merge._stats import match_halves, cc_half, r_split, shell_stats, spearman_cc

# ── Paths ────────────────────────────────────────────────────
STREAM_DIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/streams")
DARK_STREAMS = sorted(STREAM_DIR.glob("delay_-*.stream"))
REF_PDB = "/das/work/p17/p17490/Peter/small_molecule_torchref/data/refined_against_self_reference.pdb"
REF_CIF = "/das/work/p17/p17490/Peter/small_molecule_torchref/data/Merged_restraints_all_opened.cif"
PART_DIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/partialator")
OUTDIR = Path("/das/work/p17/p17490/Peter/torchsx/sfx_scan/benchmark_dark")
OUTDIR.mkdir(parents=True, exist_ok=True)

CELL = torch.tensor([14.97, 18.85, 18.89, 89.37, 84.94, 67.84], dtype=torch.float32)
SPACEGROUP = "P 1"
D_MIN = 0.9
D_MAX = 10.0
WAVELENGTH = 0.9537
N_EM = 10

# ── Load dark streams ────────────────────────────────────────
print("Loading dark streams...")
streams = []
for p in DARK_STREAMS:
    print(f"  {p.name} ...", end=" ", flush=True)
    st = SFXStream.from_stream(str(p), spacegroup=SPACEGROUP, n_workers=1)
    print(f"{st.n_crystals} crystals, {st.n_observations} obs")
    streams.append(st)
s = SFXStream.concat(streams)
print(f"\nTotal: {s.n_crystals} crystals, {s.n_observations} observations")

# ── Merge methods ─────────────────────────────────────────────
from torchsx.merge._noise import PoissonNoiseModel
FIT_KW = dict(min_obs=20, n_smooth_iter=15, smooth_lambda=0.3)

# Method 1: GenGauss log-space (baseline)
print(f"\n── GenGauss log-space + EM×{N_EM} + ScaleB ──")
fit_log = fit_gengauss(s, n_em_cycles=N_EM, scale_crystals=True, **FIT_KW)
print(f"  {fit_log.hkl.shape[0]} HKLs, med(α)={fit_log.alpha.median():.4f}, med(β)={fit_log.beta.median():.3f}")

# Method 2: Joint Poisson+GenGauss (LBFGS, quadrature)
print(f"\n── Joint + ScaleB + Poisson ──")
noise = PoissonNoiseModel(r_peak=7, r_gap=9, r_bg=11)
fit_joint = fit_gengauss(s, n_em_cycles=5, scale_crystals=True,
                         poisson_noise=noise, **FIT_KW)
print(f"  {fit_joint.hkl.shape[0]} HKLs, med(α)={fit_joint.alpha.median():.4f}")
n_zero = (fit_joint.I0 <= 1e-6).sum().item()
n_weak = (fit_joint.I0 < 1.0).sum().item()
print(f"  I₀ ≈ 0: {n_zero}, I₀ < 1: {n_weak}")


# ── Convert to MTZ ──────────────────────────────────────────
def gengauss_to_mtz(fit: HKLDistribution, cell, spacegroup, path, rfree_frac=0.05):
    """Write HKLDistribution to MTZ via torchref."""
    from torchref import Cell, SpaceGroup, ReflectionData

    I0 = fit.I0.clamp(min=0)
    F = I0.sqrt()
    # sigma(F) from Fisher information: sigma(I0)/I0 = sigma(mu), sigma(F) = F * sigma(mu)/2
    sigma_mu = fit.sigma_mu if hasattr(fit, 'sigma_mu') and fit.sigma_mu is not None else fit.alpha / (fit.obs_counts.float().sqrt() + 1e-6)
    sigma_F = F * sigma_mu / 2.0
    sigma_F = sigma_F.clamp(min=1e-6)

    hkl = fit.hkl.int()

    # Deterministic R-free flags from Miller indices
    h, k, l = hkl[:, 0].long(), hkl[:, 1].long(), hkl[:, 2].long()
    hashes = (h * 2654435761) ^ (k * 2246822519) ^ (l * 3266489917)
    denom = round(1.0 / rfree_frac)
    rfree = (hashes % denom) == 0

    cell_obj = Cell(cell.tolist(), device="cpu")
    sg = SpaceGroup(spacegroup)

    data = ReflectionData.from_tensors(
        hkl=hkl, F=F, F_sigma=sigma_F,
        cell=cell_obj, spacegroup=sg,
        rfree_flags=rfree, device="cpu",
    )
    data.write_mtz(str(path))
    n_rfree = rfree.sum().item()
    print(f"  Wrote {path} ({hkl.shape[0]} reflections, {n_rfree} R-free)")
    return data


mtz_log = OUTDIR / "gengauss_log.mtz"
mtz_joint = OUTDIR / "gengauss_joint.mtz"
gengauss_to_mtz(fit_log, CELL, SPACEGROUP, mtz_log)
gengauss_to_mtz(fit_joint, CELL, SPACEGROUP, mtz_joint)


# ── Refine ──────────────────────────────────────────────────
def refine_against_model(mtz_path, label):
    """Refine starting model against MTZ, return R-work/R-free."""
    from torchref import ModelFT, ReflectionData, Scaler
    from torchref.symmetry import Cell
    from torchref.refinement.targets import (
        MaximumLikelihoodXrayTarget,
        TotalGeometryTarget,
        TotalADPTarget,
    )

    print(f"\n── Refining: {label} ──")
    device = "cpu"

    data = ReflectionData(device=device).load_mtz(str(mtz_path))
    data.cell = Cell(CELL.tolist(), device=device)
    data.filter_by_resolution(d_min=D_MIN, d_max=D_MAX)

    model = ModelFT(device=device, wavelength=WAVELENGTH)
    model.load_pdb(REF_PDB)
    model.cell = Cell(CELL.tolist(), device=device)
    model.setup_grid()
    model.set_restraints_cif(REF_CIF)

    scaler = Scaler(model, data, device=device, nbins=10)
    scaler.calc_initial_scale()
    scaler.setup_anisotropy_correction()
    scaler.refine_lbfgs()

    target = MaximumLikelihoodXrayTarget(model=model, data=data, scaler=scaler)
    geo_loss = TotalGeometryTarget(model=model)
    adp_loss = TotalADPTarget(model=model)

    rw, rf = scaler.rfactor()
    print(f"  Initial  R-work={rw:.5f}  R-free={rf:.5f}")

    params = [*scaler.parameters(), *model.parameters()]
    optimizer = torch.optim.LBFGS(params, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = target() + 0.1 * geo_loss() + 0.1 * adp_loss()
        loss.backward()
        return loss

    for step in range(50):
        loss = optimizer.step(closure)
        if (step + 1) % 10 == 0:
            rw, rf = scaler.rfactor()
            print(f"  Step {step+1:3d}: R-work={rw:.5f}  R-free={rf:.5f}")

    rw, rf = scaler.rfactor()
    rw, rf = float(rw), float(rf)
    print(f"  Final    R-work={rw:.5f}  R-free={rf:.5f}  gap={rf-rw:.4f}")
    return rw, rf


# Also refine against partialator unity
mtz_part = OUTDIR / "partialator_unity.mtz"
if not mtz_part.exists():
    # Convert partialator HKL to MTZ
    from torchsx.merge._stats import match_halves
    from torchref import Cell as TRCell, SpaceGroup, ReflectionData

    def parse_crystfel_hkl(path):
        h, k, l, I_ = [], [], [], []
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    h.append(int(parts[0])); k.append(int(parts[1]))
                    l.append(int(parts[2])); I_.append(float(parts[3]))
                except ValueError:
                    continue
        return torch.tensor(list(zip(h, k, l)), dtype=torch.int32), torch.tensor(I_, dtype=torch.float32)

    def _hkl_to_mtz(hkl_path, mtz_path, label):
        hkl_t, I_t = parse_crystfel_hkl(hkl_path)
        F_t = I_t.clamp(min=0).sqrt()
        sigma_t = F_t * 0.05  # rough 5% sigma
        hh, kk, ll = hkl_t[:, 0].long(), hkl_t[:, 1].long(), hkl_t[:, 2].long()
        hashes = (hh * 2654435761) ^ (kk * 2246822519) ^ (ll * 3266489917)
        rfree = (hashes % 20) == 0
        cell_obj = TRCell(CELL.tolist(), device="cpu")
        sg = SpaceGroup(SPACEGROUP)
        data_t = ReflectionData.from_tensors(
            hkl=hkl_t, F=F_t, F_sigma=sigma_t,
            cell=cell_obj, spacegroup=sg,
            rfree_flags=rfree, device="cpu",
        )
        data_t.write_mtz(str(mtz_path))
        print(f"  Wrote {mtz_path} ({label}, {hkl_t.shape[0]} reflections)")

    _hkl_to_mtz(PART_DIR / "neg.hkl", mtz_part, "unity")

    # Partialator xsphere
    mtz_xsph = OUTDIR / "partialator_xsphere.mtz"
    xsph_hkl = PART_DIR / "xsphere.hkl"
    has_xsphere = xsph_hkl.exists()
    if has_xsphere:
        _hkl_to_mtz(xsph_hkl, mtz_xsph, "xsphere")

results = {}
results["partialator_unity"] = refine_against_model(mtz_part, "Partialator unity")
if has_xsphere:
    results["partialator_xsphere"] = refine_against_model(mtz_xsph, "Partialator xsphere")
results["gengauss_log"] = refine_against_model(mtz_log, "GenGauss log-space")
results["gengauss_joint"] = refine_against_model(mtz_joint, "GenGauss joint+Poisson")


# ── Summary ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"{'Method':>30s} {'R-work':>8s} {'R-free':>8s} {'Gap':>8s}")
print(f"{'-'*60}")
for name, (rw, rf) in results.items():
    print(f"{name:>30s} {rw:8.5f} {rf:8.5f} {rf-rw:8.4f}")
print(f"{'='*60}")
