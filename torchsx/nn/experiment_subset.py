"""Disjoint-data preparation for the versioned TD1 independent-half test."""
from dataclasses import replace
import numpy as np
import torch


def subset_prepared(prep, assignment, half):
    assignment = torch.as_tensor(assignment)
    if assignment.shape != (prep.obs_info.n_crystals,) or half not in (0, 1):
        raise ValueError('Invalid half assignment')
    keep = assignment == half
    if not keep.any() or keep.all():
        raise ValueError('Empty independent half')
    oi = prep.obs_info
    rows = keep[oi.crystal_id].nonzero(as_tuple=True)[0]
    meta = oi.gather(rows)
    mapping = torch.full((oi.n_crystals,), -1, dtype=torch.long)
    mapping[keep] = torch.arange(int(keep.sum()))
    meta.crystal_id = mapping[meta.crystal_id]
    meta.n_crystals = int(keep.sum())
    meta.obs_id = torch.arange(rows.numel())
    meta.crystal_feat = None if oi.crystal_feat is None else oi.crystal_feat[keep]
    meta.delay_bin_idx = None if oi.delay_bin_idx is None else oi.delay_bin_idx[keep]
    I, h = prep.I_obs[rows], prep.hkl_id[rows]
    counts = torch.bincount(h, minlength=prep.n_valid)
    seed = torch.zeros(prep.n_valid).index_add_(0, h, I) / counts.clamp(min=1)
    return replace(prep, I_obs=I, hkl_id=h, bg=prep.bg[rows],
                   obs_counts=counts, n_obs_total=rows.numel(), obs_info=meta,
                   mu_init=seed.clamp(min=1e-3).log(),
                   alpha_init=torch.full_like(prep.alpha_init, 1.65),
                   beta_perhkl=torch.full_like(prep.beta_perhkl, 2.),
                   partiality=None if prep.partiality is None else prep.partiality[rows],
                   obs_weights=None if prep.obs_weights is None else prep.obs_weights[rows])


def maybe_subset(bundle, args):
    file = getattr(args, 'experiment_partition', None)
    half = getattr(args, 'experiment_half', None)
    if file is None and half is None:
        return bundle
    if file is None or half not in (0, 1):
        raise ValueError('Supply both experiment partition and half')
    with np.load(file) as data:
        assignment, conditions = data['half'], data['condition']
    if not np.array_equal(conditions, bundle.prep.obs_info.delay_bin_idx.numpy()):
        raise ValueError('Partition raw crystal order/conditions differ from prepared data')
    bundle.prep = subset_prepared(bundle.prep, assignment, half)
    print(f'Independent image half {half}: {bundle.prep.obs_info.n_crystals} crystals; '
          'all intensity-derived initialization rebuilt from this half', flush=True)
    return bundle
