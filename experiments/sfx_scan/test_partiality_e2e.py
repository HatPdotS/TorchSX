#!/das/work/p17/p17490/Peter/torchsx/.venv/bin/python -u
"""Quick end-to-end test of joint partiality+scaling on synthetic data (CPU).

Uses physically realistic geometry: reflections placed near the Ewald sphere
so partiality has meaningful variation (not all clamped to p_min).
"""

import math
import torch
from torchsx.scaling._partiality import EwaldPartiality
from torchsx.scaling._absolute import AbsoluteScale
from torchsx.scaling._bscale import BScale
from torchsx.target._self_consistent import SelfConsistentTarget

torch.manual_seed(42)

# --- Synthetic geometry ---
n_crystals = 50
n_unique = 200
n_obs_per_crystal = 100
n_obs = n_crystals * n_obs_per_crystal
n_shells = 5
wavelength = 1.2  # Å
inv_wl = 1.0 / wavelength

# Reciprocal lattice: ~50Å unit cell → a* ≈ 0.02 Å⁻¹
a_star = 0.02
UB_base = a_star * torch.eye(3)

# Per-crystal random rotations (small, ~1-2°)
true_rotations = 0.03 * torch.randn(n_crystals, 3)
UB_true = torch.zeros(n_crystals, 3, 3)
for i in range(n_crystals):
    skew = torch.zeros(3, 3)
    w = true_rotations[i]
    skew[0, 1], skew[1, 0] = -w[2], w[2]
    skew[0, 2], skew[2, 0] = w[1], -w[1]
    skew[1, 2], skew[2, 1] = -w[0], w[0]
    R = torch.linalg.matrix_exp(skew)
    UB_true[i] = R @ UB_base

# True orientation perturbations to recover (small, ~0.5°)
delta_omega_true = 0.005 * torch.randn(n_crystals, 3)

# Generate Miller indices that land near the Ewald sphere.
# For each crystal, pick integer hkl and compute q = R(dw) @ UB @ h.
# Only keep those with small excitation error.
s0 = torch.tensor([0.0, 0.0, inv_wl])

all_crystal_id = []
all_hkl_id = []
all_hkl_obs = []
all_s_sq = []
all_shell = []

# Generate a pool of possible hkl
hkl_pool = []
for h in range(-15, 16):
    for k in range(-15, 16):
        for l in range(-5, 6):
            if h == 0 and k == 0 and l == 0:
                continue
            hkl_pool.append([h, k, l])
hkl_pool = torch.tensor(hkl_pool, dtype=torch.float32)

# Map unique hkl to indices (use first n_unique from the pool)
unique_hkl_map = {}
unique_counter = 0

for c in range(n_crystals):
    # Apply true perturbation to UB
    w = delta_omega_true[c]
    skew = torch.zeros(3, 3)
    skew[0, 1], skew[1, 0] = -w[2], w[2]
    skew[0, 2], skew[2, 0] = w[1], -w[1]
    skew[1, 2], skew[2, 1] = -w[0], w[0]
    R_true = torch.linalg.matrix_exp(skew)
    RUB = R_true @ UB_true[c]

    # Compute excitation error for all hkl in pool
    q = (RUB @ hkl_pool.T).T  # (N_pool, 3)
    q_s0 = q + s0
    r = q_s0.norm(dim=-1) - inv_wl

    # Select those with |r| < 0.005 Å⁻¹ (near the Ewald sphere)
    near = r.abs() < 0.005
    idx_near = near.nonzero(as_tuple=True)[0]

    if len(idx_near) == 0:
        continue

    # Take up to n_obs_per_crystal reflections
    if len(idx_near) > n_obs_per_crystal:
        perm = torch.randperm(len(idx_near))[:n_obs_per_crystal]
        idx_near = idx_near[perm]

    for j in idx_near:
        hkl = tuple(hkl_pool[j].int().tolist())
        if hkl not in unique_hkl_map:
            if unique_counter < n_unique:
                unique_hkl_map[hkl] = unique_counter
                unique_counter += 1
            else:
                # Wrap around
                unique_hkl_map[hkl] = hash(hkl) % n_unique

        all_crystal_id.append(c)
        all_hkl_id.append(unique_hkl_map[hkl])
        all_hkl_obs.append(list(hkl))
        d = 1.0 / (2.0 * (q[j].norm().item() + 1e-10))
        all_s_sq.append((1.0 / (2.0 * max(d, 0.1))) ** 2)

crystal_id = torch.tensor(all_crystal_id, dtype=torch.int32)
hkl_id = torch.tensor(all_hkl_id, dtype=torch.long)
hkl_obs = torch.tensor(all_hkl_obs, dtype=torch.float32)
s_sq = torch.tensor(all_s_sq, dtype=torch.float32)
n_obs = len(all_crystal_id)
n_unique_actual = unique_counter
shell = torch.randint(0, n_shells, (n_obs,))

print(f"Generated {n_obs} obs near Ewald sphere, {n_unique_actual} unique reflections")

# True intensities (log-normal)
log_I_true = torch.randn(n_unique) * 2.0 + 5.0
I_true = torch.exp(log_I_true)

# True scale factors
G_true = torch.exp(0.3 * torch.randn(n_crystals))
B_true = torch.randn(n_crystals) * 0.5

# Compute true partiality
partiality_true = EwaldPartiality(n_crystals, UB_true, wavelength)
with torch.no_grad():
    partiality_true.delta_omega.copy_(delta_omega_true)
    log_p_true = partiality_true.log_scale_per_obs(hkl_obs, crystal_id)
    p_true = torch.exp(log_p_true)

cid = crystal_id.long()
total_scale_true = G_true[cid] * torch.exp(-2.0 * B_true[cid] * s_sq) * p_true

# Generate observations with lognormal noise
I_pred_true = I_true[hkl_id] * total_scale_true
noise = torch.exp(0.1 * torch.randn(n_obs))
I_obs = (I_pred_true * noise).clamp(min=0.1)
I_bg = torch.zeros(n_obs)

print(f"True partiality: median={p_true.median():.3f}, min={p_true.min():.3f}, max={p_true.max():.3f}")
print(f"  p>0.5: {(p_true > 0.5).sum()}/{n_obs}, p>0.1: {(p_true > 0.1).sum()}/{n_obs}")
print(f"I_obs range: [{I_obs.min():.1f}, {I_obs.max():.1f}]")

# --- Joint optimisation ---
abs_scale = AbsoluteScale(n_crystals)
b_scale = BScale(n_crystals)
target = SelfConsistentTarget(n_unique, n_shells)
# Start partiality from scratch (UB known, delta_omega=0)
partiality = EwaldPartiality(n_crystals, UB_true, wavelength)

# Calibrate with unit scale
target.calibrate(I_obs, torch.ones(n_obs), hkl_id, shell, crystal_id)

params = (
    list(abs_scale.parameters())
    + list(b_scale.parameters())
    + list(target.parameters())
    + list(partiality.parameters())
)
optimizer = torch.optim.LBFGS(params, lr=1.0, max_iter=20, history_size=10,
                               line_search_fn="strong_wolfe")

hkl_obs_float = hkl_obs.float()
losses = []

print("\nJoint optimisation (3 cycles × 30 steps)...")
for cycle in range(3):
    if cycle > 0:
        with torch.no_grad():
            log_p = partiality.log_scale_per_obs(hkl_obs_float, crystal_id, s_sq=s_sq)
            ts = torch.exp(abs_scale.log_G[cid] - 2.0 * b_scale.B[cid] * s_sq + log_p)
        target.update_sigma(I_obs, ts, hkl_id)
        optimizer = torch.optim.LBFGS(params, lr=1.0, max_iter=20, history_size=10,
                                       line_search_fn="strong_wolfe")

    for step in range(30):
        current_loss = [None]

        def closure():
            optimizer.zero_grad()
            log_G = abs_scale.log_G[cid]
            log_B = -2.0 * b_scale.B[cid] * s_sq
            log_p = partiality.log_scale_per_obs(hkl_obs_float, crystal_id, s_sq=s_sq)
            total_scale = torch.exp(log_G + log_B + log_p)
            loss = target(I_obs, total_scale, hkl_id, I_bg)
            loss = loss + 100.0 * abs_scale.log_G.mean() ** 2
            loss = loss + 1e-3 * (abs_scale.log_G ** 2).mean()
            loss = loss + 1e-3 * (b_scale.B ** 2).mean()
            loss = loss + 1e-2 * (partiality.delta_omega ** 2).mean()
            loss.backward()
            current_loss[0] = loss.item()
            return loss

        optimizer.step(closure)
        losses.append(current_loss[0])

    with torch.no_grad():
        log_p_fit = partiality.log_scale_per_obs(hkl_obs_float, crystal_id, s_sq=s_sq)
        dw_deg = partiality.delta_omega.norm(dim=-1) * (180.0 / math.pi)
        sigma_0 = torch.exp(partiality.log_sigma_0).item()
        B_relp = torch.exp(partiality.log_B_relp).item()

    print(f"  cycle {cycle}: loss={current_loss[0]:.4f}, "
          f"δω median={dw_deg.median():.3f}° max={dw_deg.max():.3f}°, "
          f"σ_0={sigma_0:.5f} Å⁻¹ B_relp={B_relp:.3f}")

# --- Check recovery ---
print("\n--- Recovery diagnostics ---")

with torch.no_grad():
    I_fit = target.I_true
cc = torch.corrcoef(torch.stack([I_true.log(), I_fit.log()]))[0, 1]
print(f"log(I_true) CC: {cc:.4f}")

with torch.no_grad():
    log_G_fit = abs_scale.log_G
cc_G = torch.corrcoef(torch.stack([G_true.log(), log_G_fit]))[0, 1]
print(f"log(G) CC: {cc_G:.4f}")

with torch.no_grad():
    log_p_fit = partiality.log_scale_per_obs(hkl_obs_float, crystal_id, s_sq=s_sq)
# Only evaluate CC on observations not clamped to floor in truth
not_floor = log_p_true > (math.log(0.01) + 0.01)
if not_floor.sum() > 10:
    cc_p = torch.corrcoef(torch.stack([log_p_true[not_floor], log_p_fit[not_floor]]))[0, 1]
    print(f"log(p) CC (non-floor): {cc_p:.4f}  ({not_floor.sum()} obs)")
else:
    cc_p = torch.tensor(float('nan'))
    print(f"log(p) CC: N/A (all at floor)")

dw_fit = partiality.delta_omega.detach()
cc_dw = torch.corrcoef(torch.stack([delta_omega_true.flatten(), dw_fit.flatten()]))[0, 1]
print(f"δω CC: {cc_dw:.4f}")

print(f"\nLoss: {losses[0]:.4f} → {losses[-1]:.4f} (ratio {losses[-1]/losses[0]:.3f})")

ok = cc > 0.9 and losses[-1] < losses[0]
print(f"\n{'PASS' if ok else 'FAIL'}: I_true CC={cc:.3f}, loss decreased={losses[-1]<losses[0]}")
