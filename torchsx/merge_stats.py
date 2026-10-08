"""Statistics of merged reflection files: one merge, two independent halves, differences.

Every merge is read onto the canonical CCP4 asymmetric unit, Friedel mates
merged, one row per unique reflection. Comparisons run on the reflections
common to the merges involved, after scaling, in equal-count resolution shells:

* ``summary_stats``: one merge on its own -- completeness, mean intensity and
  I/sigma(I), mean multiplicity where the file carries counts, and the Wilson
  falloff of <I/epsilon>.
* ``halves_stats``: two independent halves of one dataset. Half B is scaled
  onto half A (one log scale plus six anisotropic terms on normalised hkl,
  sigma-weighted robust least squares on amplitudes), then CC, R_split, the
  sigma-normalised difference and the noise-to-signal ratio ``ln N/S``.
* ``difference_stats``: dataset X against a reference R, both as two
  independent arms A and B. In each arm X is scaled onto R, arm B onto arm A by
  the reference merges; the arms' Delta F = F_X - F_R are compared per shell
  (CC, ``ln N/S``) and summed into information in bits over resolution shells x
  quantiles of sigma(Delta F).
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

AMPLITUDES = ('french-wilson', 'sqrt')
FORMATS = ('csv', 'json', 'plots')
COUNT_LABELS = ('N', 'NOBS', 'N_OBS', 'NMEAS', 'MULT', 'MULTIPLICITY', 'REDUNDANCY')
LABEL_KEYS = ('F', 'SIGF', 'I', 'SIGI')
WILSON_D_MAX = 4.0
CC_CLIP = 0.999


@dataclass
class Merge:
    """One merge on the canonical ASU: one row per unique reflection.

    ``F``/``SIGF`` are read, or derived from intensities and NaN where no
    amplitude could be derived; ``I``/``SIGI`` are present only when the file
    carries measured intensities.
    """
    hkl: np.ndarray
    F: np.ndarray
    SIGF: np.ndarray
    d: np.ndarray
    cell: np.ndarray
    spacegroup: str
    I: np.ndarray | None = None
    SIGI: np.ndarray | None = None
    counts: np.ndarray | None = None
    amplitudes: str = 'file'
    path: str = ''

    def __len__(self):
        return len(self.hkl)

    def take(self, ix):
        pick = lambda a: None if a is None else a[ix]          # noqa: E731
        return dataclasses.replace(self, hkl=self.hkl[ix], F=self.F[ix], SIGF=self.SIGF[ix], d=self.d[ix],
                                   I=pick(self.I), SIGI=pick(self.SIGI), counts=pick(self.counts))

    def with_amplitudes(self):
        """The rows that carry an amplitude, which every comparison needs."""
        ok = np.isfinite(self.F)
        return self if ok.all() else self.take(np.flatnonzero(ok))

    @property
    def laue(self):
        return laue_key(self.spacegroup)


def _spacegroup(name):
    from torchref.symmetry import SpaceGroup
    return SpaceGroup(name)


def _tensor(a):
    return torch.from_numpy(np.ascontiguousarray(a))


def laue_key(spacegroup):
    """The group's rotations closed under inversion, in its setting.

    Merges with equal keys share one reciprocal asymmetric unit, so their
    canonical indices can be matched.
    """
    m = _spacegroup(spacegroup).matrices.round().to(torch.int64)
    return frozenset(tuple(r.flatten().tolist()) for r in torch.cat([m, -m]))


def d_spacing(hkl, cell):
    """Resolution of each Miller index, in angstrom."""
    from torchref.base.reciprocal.hkl import get_d_spacing
    return get_d_spacing(torch.as_tensor(np.asarray(hkl), dtype=torch.float64),
                         torch.as_tensor(np.asarray(cell, dtype=np.float64))).cpu().numpy()


def _orbit_representatives(spacegroup, hkl, chunk=200_000):
    """Whether each row is its orbit's representative: the smallest packed index
    among its symmetry equivalents and their Friedel mates."""
    sg = _spacegroup(spacegroup)
    out = []
    for start in range(0, len(hkl), chunk):
        part = hkl[start:start + chunk]
        eq = sg.expand_reciprocal(_tensor(part)).cpu().numpy().astype(np.int64)
        keys = _key(np.concatenate([eq, -eq]).reshape(-1, 3)).reshape(2 * len(eq), -1)
        out.append(_key(part) == keys.min(axis=0))
    return np.concatenate(out) if out else np.zeros(0, bool)


def possible_reflections(cell, spacegroup, d_min, d_max=None):
    """One index per unique, not systematically absent reflection with ``d_min <= d``
    (``<= d_max``), Friedel mates merged; and its d.

    Indices are orbit representatives, not ASU indices: they count reflections.
    The search box is |h| <= a / d_min per axis, which holds for any cell.
    """
    sg = _spacegroup(spacegroup)
    a, b, c = (int(math.floor(x / d_min * (1 + 1e-9))) for x in np.asarray(cell, dtype=np.float64)[:3])
    k, l = np.meshgrid(np.arange(-b, b + 1), np.arange(-c, c + 1), indexing='ij')
    k, l = k.ravel(), l.ravel()
    rows, ds = [], []
    for h in range(-a, a + 1):
        plane = np.stack([np.full_like(k, h), k, l], 1)
        plane = plane[(plane != 0).any(axis=1)]
        d = d_spacing(plane, cell)
        keep = d >= d_min * (1 - 1e-9)
        if d_max is not None:
            keep &= d <= d_max
        plane, d = plane[keep], d[keep]
        keep = ~sg.is_absent(_tensor(plane)).cpu().numpy() & _orbit_representatives(spacegroup, plane)
        rows.append(plane[keep])
        ds.append(d[keep])
    return np.concatenate(rows), np.concatenate(ds)


def file_format(path) -> str:
    """``'mtz'``, ``'hkl'`` (CrystFEL reflection list, including ``.hkl1``/``.hkl2``) or ``'cif'``."""
    name = Path(path).name.lower().removesuffix('.gz')
    suffix = Path(name).suffix
    if suffix == '.mtz':
        return 'mtz'
    if suffix in ('.hkl', '.hkl1', '.hkl2'):
        return 'hkl'
    if suffix in ('.cif', '.mmcif', '.ent'):
        return 'cif'
    raise ValueError(f'{path}: unknown reflection format; expected .mtz, CrystFEL .hkl or mmCIF')


def _mtz_counts(dataset):
    """A per-reflection count column of a reciprocalspaceship dataset, matched by label."""
    for col in dataset.columns:
        if str(col).upper() in COUNT_LABELS:
            return np.asarray(dataset[col].to_numpy(), dtype=np.float64)
    return None


def _read(path, cell, spacegroup, labels):
    """``(data dict, cell, spacegroup, counts)`` from TorchRef's readers, Friedel mates merged."""
    fmt = file_format(path)
    if fmt == 'mtz':
        from torchref.io.mtz import MTZReader
        reader = MTZReader(column_names=labels or None, anomalous=False).read(str(path))
        data, cell_, sg = reader()
        return data, cell_, sg, _mtz_counts(reader.mtz_data)
    if fmt == 'cif':
        from torchref.io.cif_readers import ReflectionCIFReader
        data, cell_, sg = ReflectionCIFReader(str(path), anomalous=False)()
        return data, cell_, sg, None
    if cell is None or spacegroup is None:
        raise ValueError(f'{path}: a CrystFEL reflection list carries no cell or space group; '
                         'give --cell and --spacegroup, or another input that has them')
    from torchref.io.hkl import HKLReader
    reader = HKLReader().read(str(path), cell, spacegroup)
    data, cell_, sg = reader()
    return data, cell_, sg, reader.nmeas.astype(np.float64)


def _column(data, key):
    return None if key not in data else np.asarray(data[key], dtype=np.float64)


def _average_duplicates(hkl, columns):
    """Inverse-variance average of rows that share a canonical index.

    ``columns`` maps a name to ``(value, sigma)``, or to ``(value, None)`` for a
    count that is summed.
    """
    unique, inverse = np.unique(hkl, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    if len(unique) == len(hkl):
        return hkl, columns
    out = {}
    for name, (value, sigma) in columns.items():
        if sigma is None:
            out[name] = (np.bincount(inverse, value, len(unique)), None)
            continue
        w = 1.0 / sigma**2
        sw = np.bincount(inverse, w, len(unique))
        out[name] = (np.bincount(inverse, w * value, len(unique)) / sw, 1.0 / np.sqrt(sw))
    return unique, out


def _french_wilson(hkl, I, SIGI, d, spacegroup):
    """French-Wilson amplitudes, the prior fitted to these intensities; the rows it accepts."""
    from torchref.io.datasets import french_wilson_auto
    as64 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)  # noqa: E731
    F, S, valid = french_wilson_auto(as64(I), as64(SIGI), _tensor(hkl), as64(d), str(spacegroup))
    F, S = F.cpu().double().numpy(), S.cpu().double().numpy()
    keep = valid.cpu().numpy() & np.isfinite(F) & (F > 0)
    return F, S, keep


def merge_from_arrays(hkl, *, cell, spacegroup, F=None, SIGF=None, I=None, SIGI=None, counts=None,
                      amplitudes='french-wilson', dmin=None, dmax=None, path='') -> Merge:
    """Canonicalise, average duplicates, cut resolution and derive amplitudes.

    Amplitudes the file carries are used as given. From intensities alone,
    ``amplitudes='french-wilson'`` takes TorchRef's French-Wilson posterior,
    ``'sqrt'`` takes F = sqrt(I), sigma(F) = sigma(I) / 2F. Reflections without
    an amplitude (French-Wilson rejections, I <= 0 under ``'sqrt'``) keep their
    row with F = NaN: they count for the single-merge summary and drop out of
    comparisons.
    """
    if amplitudes not in AMPLITUDES:
        raise ValueError(f'amplitudes must be one of {AMPLITUDES}')
    hkl = np.asarray(hkl, dtype=np.int64).reshape(-1, 3)
    cell = np.asarray(cell, dtype=np.float64).reshape(6)
    columns = {}
    keep = (hkl != 0).any(axis=1)
    if F is not None and SIGF is not None:
        F, SIGF = np.asarray(F, dtype=np.float64), np.asarray(SIGF, dtype=np.float64)
        keep &= np.isfinite(F) & np.isfinite(SIGF) & (F > 0) & (SIGF > 0)
        columns['F'] = (F, SIGF)
    if I is not None and SIGI is not None:
        I, SIGI = np.asarray(I, dtype=np.float64), np.asarray(SIGI, dtype=np.float64)
        keep &= np.isfinite(I) & np.isfinite(SIGI) & (SIGI > 0)
        columns['I'] = (I, SIGI)
    if not columns:
        raise ValueError(f'{path}: needs amplitudes with sigmas or intensities with sigmas')
    if counts is not None:
        counts = np.asarray(counts, dtype=np.float64)
        columns['counts'] = (np.where(np.isfinite(counts), counts, 0.0), None)
    canonical = _spacegroup(spacegroup).canonicalize_hkl(_tensor(hkl[keep]), sort=False)[0]
    hkl, columns = _average_duplicates(canonical.cpu().numpy().astype(np.int64),
                                       {k: (v[keep], None if s is None else s[keep]) for k, (v, s) in columns.items()})
    d = d_spacing(hkl, cell)
    cut = np.ones(len(hkl), bool)
    if dmin is not None:
        cut &= d >= dmin
    if dmax is not None:
        cut &= d <= dmax
    hkl, d = hkl[cut], d[cut]
    columns = {k: (v[cut], None if s is None else s[cut]) for k, (v, s) in columns.items()}
    I, SIGI = columns.get('I', (None, None))
    if 'F' in columns:
        F, SIGF = columns['F']
        source = 'file'
    elif amplitudes == 'sqrt':
        ok = I > 0
        F = np.sqrt(np.where(ok, I, 1.0))
        SIGF = SIGI / (2 * F)
        source = 'sqrt'
    else:
        F, SIGF, ok = _french_wilson(hkl, I, SIGI, d, spacegroup)
        source = 'french-wilson'
    if source != 'file':
        F, SIGF = np.where(ok, F, np.nan), np.where(ok, SIGF, np.nan)
    counts = columns['counts'][0] if 'counts' in columns else None
    if not len(hkl):
        raise ValueError(f'{path}: no usable reflections')
    return Merge(hkl=hkl, F=F, SIGF=SIGF, d=d, cell=cell, spacegroup=str(spacegroup), I=I, SIGI=SIGI,
                 counts=counts, amplitudes=source, path=str(path))


def load_merge(path, *, cell=None, spacegroup=None, labels=None, amplitudes='french-wilson',
               dmin=None, dmax=None) -> Merge:
    """Read an MTZ, CrystFEL ``.hkl`` or mmCIF structure-factor file into a :class:`Merge`.

    ``cell`` and ``spacegroup`` are needed for CrystFEL reflection lists, which
    carry neither; MTZ and mmCIF files bring their own. ``labels`` pins MTZ
    columns (keys ``F``, ``SIGF``, ``I``, ``SIGI``).
    """
    data, cell_, sg, counts = _read(path, cell, spacegroup, labels)
    return merge_from_arrays(data['HKL'], cell=cell_, spacegroup=sg, F=_column(data, 'F'),
                             SIGF=_column(data, 'SIGF'), I=_column(data, 'I'), SIGI=_column(data, 'SIGI'),
                             counts=counts, amplitudes=amplitudes, dmin=dmin, dmax=dmax, path=path)


def _key(hkl):
    h = hkl + 1024
    if (h < 0).any() or (h > 2047).any():
        raise ValueError('Miller indices beyond +-1023')
    return (h[:, 0] << 22) | (h[:, 1] << 11) | h[:, 2]


def common_indices(*merges):
    """Per merge, the row indices of the reflections present in every merge, in one shared order."""
    keys = [_key(m.hkl) for m in merges]
    both = keys[0]
    for k in keys[1:]:
        both = np.intersect1d(both, k, assume_unique=True)
    if not len(both):
        raise ValueError('no reflections common to ' + ', '.join(m.path or '?' for m in merges))
    out = []
    for k in keys:
        order = np.argsort(k)
        out.append(order[np.searchsorted(k, both, sorter=order)])
    return out


def resolution_shells(d, n_shells):
    """Equal-count shells of row indices, low to high resolution."""
    order = np.argsort(-d, kind='stable')
    return [ix for ix in np.array_split(order, max(1, min(n_shells, len(order)))) if len(ix)]


def _shell(j, d, ix):
    return {'shell': j, 'd_max': float(d[ix].max()), 'd_min': float(d[ix].min()),
            'inv_d2': float(np.mean(1.0 / d[ix] ** 2))}


def pearson(x, y) -> float:
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def noise_signal(x, y):
    """``(S, N, ln N/S)`` of two independent estimates of the same quantities.

    ``S = Cov(x, y)`` is the variance of the common (true) part and ``N = Var(x
    - y) / 2`` the error variance of one estimate; ``CC = 1 / (1 + N/S)``.
    ``ln N/S`` is NaN where no common signal is left (``S <= 0``).
    """
    if x.size < 3:
        return float('nan'), float('nan'), float('nan')
    S = float(np.cov(x, y)[0, 1])
    N = float(0.5 * np.var(x - y, ddof=1))
    return S, N, float(np.log(N / S)) if S > 0 and N > 0 else float('nan')


def design(hkl):
    """Log-scale design: overall term plus six anisotropic terms on hkl / max|hkl|."""
    q = hkl / np.maximum(np.abs(hkl).max(axis=0), 1)
    h, k, l = q.T
    return np.stack([np.ones_like(h), h * h, k * k, l * l, 2 * h * k, 2 * h * l, 2 * k * l], 1)


def fit_scale(X, FA, SA, FB, SB):
    """Coefficients ``c`` of the correction ``F_B -> F_B * exp(X @ c)`` onto ``F_A``.

    Sigma-weighted soft-L1 least squares on amplitudes.
    """
    from scipy.optimize import least_squares

    def resid(c):
        g = np.exp(X @ c)
        return (FA - g * FB) / np.sqrt(SA**2 + (g * SB) ** 2)
    c0 = np.zeros(X.shape[1])
    c0[0] = np.median(np.log(FA / FB))
    return least_squares(resid, c0, loss='soft_l1', f_scale=3.0).x


def scale_onto(target: Merge, source: Merge):
    """Per-row correction factor of ``source`` onto ``target`` (rows aligned) and its coefficients."""
    X = design(target.hkl)
    c = fit_scale(X, target.F, target.SIGF, source.F, source.SIGF)
    return np.exp(X @ c), c


def pair_stats(FA, SA, FB, SB, IA, IB) -> dict:
    z = (FA - FB) / np.sqrt(SA**2 + SB**2)
    S_I, N_I, lnsr_I = noise_signal(IA, IB)
    return {
        'n': int(FA.size),
        'cc_I': pearson(IA, IB),
        'cc_F': pearson(FA, FB),
        'rsplit_I': float(np.sqrt(2) * np.abs(IA - IB).sum() / (IA + IB).sum()),
        'r_F': float(np.abs(FA - FB).sum() / (0.5 * (FA + FB).sum())),
        'rms_z': float(np.sqrt(np.mean(z**2))),
        'frac_z_gt3': float(np.mean(np.abs(z) > 3)),
        'S_I': S_I, 'N_I': N_I, 'lnsr_I': lnsr_I,
        'lnsr_F': noise_signal(FA, FB)[2],
    }


def _finite_mean(values):
    v = [x for x in values if np.isfinite(x)]
    return float(np.mean(v)) if v else float('nan')


def halves_stats(a: Merge, b: Merge, n_shells=20) -> dict:
    """Two independent halves of one dataset, half B scaled onto half A.

    Intensity statistics use the measured intensities when both halves carry
    them, else F^2. ``lowres_drift`` is the unscaled median ln(I_B / I_A) of the
    lowest-resolution 2 % of the common reflections minus that of the middle
    40 %: the smooth falloff difference the anisotropic model cannot absorb.
    """
    a, b = a.with_amplitudes(), b.with_amplitudes()
    ia, ib = common_indices(a, b)
    A, B = a.take(ia), b.take(ib)
    d = A.d
    g, c = scale_onto(A, B)
    ci = fit_scale(np.stack([np.ones_like(d), 1.0 / d**2], 1), A.F, A.SIGF, B.F, B.SIGF)
    FB, SB = B.F * g, B.SIGF * g
    measured = A.I is not None and B.I is not None
    IA, IB = (A.I, B.I * g**2) if measured else (A.F**2, FB**2)
    out = {
        'n_A': len(a), 'n_B': len(b), 'n_common': len(A),
        'intensities': 'measured' if measured else 'F^2',
        'amplitudes': [a.amplitudes, b.amplitudes],
        'scale': {
            'model': 'F_B *= exp(c . [1, h2, k2, l2, 2hk, 2hl, 2kl]) on hkl / max|hkl|',
            'c': [float(v) for v in c],
            'iso_log_k': float(ci[0]),
            # F ~ exp(-B s^2 / 4) with s = 1/d: positive = B falls off faster than A
            'iso_delta_B': float(4.0 * ci[1]),
            'g_range': [float(g.min()), float(g.max())],
        },
        'overall': pair_stats(A.F, A.SIGF, FB, SB, IA, IB),
    }
    shells = []
    for j, ix in enumerate(resolution_shells(d, n_shells)):
        shells.append({**_shell(j, d, ix), **pair_stats(A.F[ix], A.SIGF[ix], FB[ix], SB[ix], IA[ix], IB[ix])})
    out['shells'] = shells
    order = np.argsort(-d, kind='stable')
    lr = 2 * np.log(B.F[order] / A.F[order])
    n = len(order)
    lowres_drift = float(np.median(lr[: max(1, n // 50)]) - np.median(lr[int(0.3 * n): max(int(0.7 * n), int(0.3 * n) + 1)]))
    o = out['overall']
    out['headline'] = {
        'n_common': len(A), 'cc_I': o['cc_I'], 'cc_I_outer': shells[-1]['cc_I'],
        'cc_I_shell_mean': _finite_mean(s['cc_I'] for s in shells),
        'cc_F': o['cc_F'], 'rsplit_I': o['rsplit_I'], 'rms_z': o['rms_z'], 'rms_z_outer': shells[-1]['rms_z'],
        'lnsr_I': o['lnsr_I'], 'lnsr_I_shell_mean': _finite_mean(s['lnsr_I'] for s in shells),
        'lnsr_I_outer': shells[-1]['lnsr_I'], 'lowres_drift': lowres_drift,
        'iso_delta_B': out['scale']['iso_delta_B'],
    }
    return out


def information_bits(cc) -> float:
    """Information of one estimate about the signal, per reflection: ``-1/2 log2(1 - CC)``.

    For x_A = s + e_A, x_B = s + e_B, ``CC = S / (S + N)`` and this equals
    ``1/2 log2(1 + S/N)``. CC is clipped to +-0.999 and keeps its sign, so
    noise around CC = 0 averages out; an undefined CC contributes nothing.
    """
    if not np.isfinite(cc):
        return 0.0
    return -0.5 * math.log2(1.0 - min(max(cc, -CC_CLIP), CC_CLIP))


def difference_stats(ref: tuple[Merge, Merge], other: tuple[Merge, Merge], n_shells=20, n_sigma=5) -> dict:
    """Delta F = F_X - F_R of two independent arms, compared between the arms.

    On the reflections common to all four merges: in each arm X is scaled onto
    R, then arm B onto arm A by the reference merges, the same correction
    applied to X_B. Per resolution shell: CC and ``ln N/S`` of Delta F between
    the arms, and information in bits, the sum of ``n_b * information_bits(CC_b)``
    over ``n_sigma`` quantile bins of hypot(sigma_dF_A, sigma_dF_B) within the
    shell. Binning by sigma keeps a down-weighting of noisy reflections from
    counting as information. ``bits_res_only`` uses the shell CC alone.
    """
    merges = tuple(m.with_amplitudes() for m in (ref[0], other[0], ref[1], other[1]))
    RA, XA, RB, XB = (m.take(i) for m, i in zip(merges, common_indices(*merges)))
    gXA, cXA = scale_onto(RA, XA)
    gXB, cXB = scale_onto(RB, XB)
    gB, cB = scale_onto(RA, RB)
    FRA, SRA = RA.F, RA.SIGF
    FXA, SXA = XA.F * gXA, XA.SIGF * gXA
    FRB, SRB = RB.F * gB, RB.SIGF * gB
    FXB, SXB = XB.F * gXB * gB, XB.SIGF * gXB * gB
    dA, dB = FXA - FRA, FXB - FRB
    sigma = np.hypot(np.hypot(SRA, SXA), np.hypot(SRB, SXB))
    d = RA.d
    shells, bins = [], []
    for j, ix in enumerate(resolution_shells(d, n_shells)):
        cc = pearson(dA[ix], dB[ix])
        S, N, lnsr = noise_signal(dA[ix], dB[ix])
        bits = 0.0
        for k, jx in enumerate(np.array_split(ix[np.argsort(sigma[ix], kind='stable')], max(1, min(n_sigma, len(ix))))):
            cc_b = pearson(dA[jx], dB[jx])
            b = len(jx) * information_bits(cc_b)
            bits += b
            bins.append({'shell': j, 'sigma_bin': k, 'd_max': float(d[jx].max()), 'd_min': float(d[jx].min()),
                         'sigma_min': float(sigma[jx].min()), 'sigma_max': float(sigma[jx].max()),
                         'n': len(jx), 'cc_dF': cc_b, 'bits': b})
        shells.append({**_shell(j, d, ix), 'n': len(ix), 'cc_dF': cc, 'S_dF': S, 'N_dF': N, 'lnsr_dF': lnsr,
                       'bits': bits, 'bits_res_only': len(ix) * information_bits(cc)})
    return {
        'n_common': len(d),
        'scale': {'X_onto_R_A': [float(v) for v in cXA], 'X_onto_R_B': [float(v) for v in cXB],
                  'B_onto_A_reference': [float(v) for v in cB]},
        'shells': shells,
        'bins': bins,
        'headline': {'n_common': len(d), 'bits': float(sum(s['bits'] for s in shells)),
                     'bits_res_only': float(sum(s['bits_res_only'] for s in shells))},
    }


def summary_stats(m: Merge, n_shells=20, dmax=None) -> dict:
    """One merge on its own, per equal-count resolution shell and overall.

    Completeness is the fraction of the unique, not systematically absent
    reflections of the space group that the merge holds; the lowest shell
    reaches out to ``dmax`` (all low-order reflections when ``None``).
    ``n_absent`` counts merged reflections that are systematically absent. I/sigma(I) uses measured intensities when the
    file carries them, else F^2 / 2 F sigma(F). ``wilson_B`` is the slope of
    ln <I/epsilon> against s^2 = 1/4d^2 over the shells beyond 4 A, without
    scattering-factor normalisation: comparable between merges of one crystal,
    not an absolute B.
    """
    sg = _spacegroup(m.spacegroup)
    _, d_possible = possible_reflections(m.cell, m.spacegroup, float(m.d.min()), dmax)
    allowed = ~sg.is_absent(_tensor(m.hkl)).cpu().numpy()
    I, SIGI = (m.I, m.SIGI) if m.I is not None else (m.F**2, 2 * m.F * m.SIGF)
    eps = sg.epsilon(_tensor(m.hkl), friedel=False).cpu().double().numpy()
    shells_ix = resolution_shells(m.d, n_shells)
    s2 = 1.0 / m.d**2
    edges = [0.5 * (s2[a].max() + s2[b].min()) for a, b in zip(shells_ix[:-1], shells_ix[1:])]
    n_possible = np.bincount(np.searchsorted(edges, 1.0 / d_possible**2), minlength=len(shells_ix))
    n_held = np.bincount(np.searchsorted(edges, s2[allowed]), minlength=len(shells_ix))
    shells = []
    for j, ix in enumerate(shells_ix):
        shells.append({**_shell(j, m.d, ix), 'n': len(ix), 'n_possible': int(n_possible[j]),
                       'completeness': n_held[j] / n_possible[j] if n_possible[j] else float('nan'),
                       'mean_I': float(I[ix].mean()), 'mean_I_over_sigma': float((I[ix] / SIGI[ix]).mean()),
                       'mean_I_eps': float((I[ix] / eps[ix]).mean()),
                       'multiplicity': float(m.counts[ix].mean()) if m.counts is not None else float('nan')})
    fit = [s for s in shells if s['inv_d2'] > 1 / WILSON_D_MAX**2 and s['mean_I_eps'] > 0]
    if len(fit) >= 3:
        slope, intercept = np.polyfit([s['inv_d2'] / 4 for s in fit], [math.log(s['mean_I_eps']) for s in fit], 1)
        wilson_B, wilson_lnk = float(-slope / 2), float(intercept)
    else:
        wilson_B = wilson_lnk = float('nan')
    headline = {'n': len(m), 'd_max': float(m.d.max()), 'd_min': float(m.d.min()),
                'completeness': int(allowed.sum()) / n_possible.sum() if n_possible.sum() else float('nan'),
                'n_absent': int((~allowed).sum()),
                'mean_I_over_sigma': float((I / SIGI).mean()),
                'multiplicity': float(m.counts.mean()) if m.counts is not None else float('nan'),
                'wilson_B': wilson_B}
    return {'n': len(m), 'intensities': 'measured' if m.I is not None else 'F^2', 'amplitudes': m.amplitudes,
            'spacegroup': m.spacegroup, 'cell': [float(v) for v in m.cell],
            'wilson': {'B': wilson_B, 'ln_k': wilson_lnk, 'n_shells': len(fit), 'd_max_fit': WILSON_D_MAX},
            'shells': shells, 'headline': headline}


# Categorical slots in fixed order; more datasets than slots take one ordered blue ramp.
SERIES = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948')
RAMP = ('#86b6ef', '#0d366b')
INK, INK_2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e3e2dc', '#fcfcfb'
D_TICKS = (20, 10, 8, 6, 5, 4, 3.5, 3, 2.5, 2.2, 2, 1.8, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8,
           0.7, 0.6, 0.5, 0.4, 0.3)


def _colors(n):
    if n <= len(SERIES):
        return list(SERIES[:n])
    from matplotlib.colors import LinearSegmentedColormap, to_hex
    cmap = LinearSegmentedColormap.from_list('ramp', RAMP)
    return [to_hex(cmap(t)) for t in np.linspace(0, 1, n)]


def _resolution_axis(ax, d_min):
    """x in 1/d^2 from 0 to just past ``d_min``; d labels from the high-resolution end, at least 1/7 apart."""
    hi = 1.03 / d_min**2
    ticks = []
    for t in sorted(D_TICKS):
        if 1 / t**2 <= hi and all(abs(1 / t**2 - 1 / u**2) >= hi / 7 for u in ticks):
            ticks.append(t)
    ax.set_xlim(0, hi)
    ax.set_xticks([1 / t**2 for t in ticks])
    ax.set_xticklabels([f'{t:g}' for t in ticks])
    ax.set_xlabel('resolution (Å)', color=INK_2)


def _style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc='left')
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)


def _panels(series, metrics, filename, folder, heading, refs=None):
    """One figure of per-shell curves: ``series`` maps a label to its shell rows."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 3.9), facecolor=SURFACE)
    colors = _colors(len(series))
    for ax, (key, title) in zip(np.atleast_1d(axes), metrics):
        _style(ax, title)
        for color, (label, shells) in zip(colors, series.items()):
            ax.plot([s['inv_d2'] for s in shells], [s[key] for s in shells], color=color, linewidth=1.6, marker='o', markersize=4,
                    markeredgecolor=SURFACE, markeredgewidth=0.8, label=label)
        if refs and key in refs:
            ax.axhline(refs[key], color=INK_2, linewidth=0.8, linestyle=(0, (4, 3)))
        _resolution_axis(ax, min(s['d_min'] for shells in series.values() for s in shells))
    if len(series) > 1:
        fig.legend(*np.atleast_1d(axes)[0].get_legend_handles_labels(), loc='lower center', ncol=min(len(series), 6),
                   frameon=False, labelcolor=INK, fontsize=9)
    fig.suptitle(heading, color=INK, fontsize=12, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.09 if len(series) > 1 else 0, 1, 0.95))
    fig.savefig(folder / filename, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def _wilson_panel(series, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), facecolor=SURFACE)
    colors = _colors(len(series))
    for ax, (key, title) in zip(axes[:2], (('completeness', 'completeness'), ('mean_I_over_sigma', '⟨I/σ(I)⟩'))):
        _style(ax, title)
        for color, (label, res) in zip(colors, series.items()):
            ax.plot([s['inv_d2'] for s in res['shells']], [s[key] for s in res['shells']], color=color, linewidth=1.6, marker='o', markersize=4,
                    markeredgecolor=SURFACE, markeredgewidth=0.8, label=label)
        _resolution_axis(ax, min(s['d_min'] for res in series.values() for s in res['shells']))
    ax = axes[2]
    _style(ax, 'Wilson plot: ln⟨I/ε⟩')
    for color, (label, res) in zip(colors, series.items()):
        sh = [s for s in res['shells'] if s['mean_I_eps'] > 0]
        x = np.array([s['inv_d2'] / 4 for s in sh])
        ax.plot(x, [math.log(s['mean_I_eps']) for s in sh], color=color, linewidth=0, marker='o', markersize=4,
                markeredgecolor=SURFACE, markeredgewidth=0.8, label=f"{label}  B = {res['wilson']['B']:.1f} Å²")
        w = res['wilson']
        if np.isfinite(w['B']):
            xf = x[x > 1 / (4 * w['d_max_fit'] ** 2)]
            ax.plot(xf, w['ln_k'] - 2 * w['B'] * xf, color=color, linewidth=1.2)
    ax.set_xlabel('s² = 1/4d² (Å⁻²)', color=INK_2)
    ax.legend(frameon=False, labelcolor=INK, fontsize=8)
    if len(series) > 1:
        fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=min(len(series), 6),
                   frameon=False, labelcolor=INK, fontsize=9)
    fig.suptitle('single merges', color=INK, fontsize=12, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.09 if len(series) > 1 else 0, 1, 0.95))
    fig.savefig(folder / 'summary.png', dpi=200, facecolor=SURFACE)
    plt.close(fig)


def render(results, folder):
    """PNG figures, one per kind present: ``summary.png``, ``halves.png``, ``difference.png``."""
    folder = Path(folder)
    summaries = {r['name']: r for r in results['datasets'] if r['kind'] == 'summary'}
    halves = {r['name']: r['shells'] for r in results['datasets'] if r['kind'] == 'halves'}
    diffs = {f"{r['name']} − {r['reference']}": r['shells'] for r in results['differences']}
    written = []
    if summaries:
        _wilson_panel(summaries, folder)
        written.append('summary.png')
    if halves:
        _panels(halves, (('cc_I', 'CC(I), A vs B'), ('rms_z', 'RMS z of F_A − F_B'), ('lnsr_I', 'ln N/S of I')),
                'halves.png', folder, 'independent halves (B scaled onto A)', refs={'rms_z': 1.0})
        written.append('halves.png')
    if diffs:
        _panels(diffs, (('cc_dF', 'CC(ΔF), arm A vs arm B'), ('lnsr_dF', 'ln N/S of ΔF'), ('bits', 'information (bits)')),
                'difference.png', folder, 'differences against the reference', refs={'cc_dF': 0.0, 'bits': 0.0})
        written.append('difference.png')
    return written
