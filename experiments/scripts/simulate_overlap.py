"""
Interactive Ewald sphere offset × partiality simulation for SFX.

Supports:
  - Relp models: Gaussian approximation vs sinc² (true shape transform)
  - Beam profiles: top-hat vs Gaussian
  - Sliders for bandwidth, N unit cells, cell parameter, d-spacing

Usage:
  python ewald_partiality_interactive.py
  (requires matplotlib with interactive backend)
"""

import numpy as np
from scipy.special import erf
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

# ── Partiality functions ─────────────────────────────────────────────

def partiality_gauss_tophat(delta, sigma_relp, shell_hw):
    """Gaussian relp × top-hat Ewald shell."""
    if shell_hw < 1e-14:
        return np.exp(-0.5 * delta**2 / sigma_relp**2)
    sq2s = np.sqrt(2) * sigma_relp
    return 0.5 * (erf((shell_hw - delta) / sq2s) - erf((-shell_hw - delta) / sq2s))


def partiality_gauss_gauss(delta, sigma_relp, sigma_beam):
    """Gaussian relp × Gaussian beam profile."""
    sigma_eff = np.sqrt(sigma_relp**2 + sigma_beam**2)
    return (sigma_relp / sigma_eff) * np.exp(-0.5 * delta**2 / sigma_eff**2)


def sinc2_profile(x, L):
    """sinc²(π L x) — true shape transform of N unit cells."""
    arg = np.pi * L * x
    out = np.ones_like(arg, dtype=float)
    mask = np.abs(arg) > 1e-10
    out[mask] = (np.sin(arg[mask]) / arg[mask])**2
    return out


def partiality_sinc2_tophat(delta, n_cells, a_cell, shell_hw, n_quad=500):
    """sinc² relp × top-hat Ewald shell via numerical quadrature."""
    L = n_cells * a_cell
    span = 5.0 / L
    x = np.linspace(-span, span, n_quad)
    dx = x[1] - x[0]
    profile = sinc2_profile(x, L)
    norm_val = np.sum(profile) * dx

    if np.isscalar(delta):
        delta = np.array([delta])

    p = np.zeros_like(delta, dtype=float)
    for i, d in enumerate(delta):
        mask = (x >= d - shell_hw) & (x <= d + shell_hw)
        p[i] = np.sum(profile[mask]) * dx / norm_val if norm_val > 0 else 0
    return p


def partiality_sinc2_gauss(delta, n_cells, a_cell, sigma_beam, n_quad=500):
    """sinc² relp × Gaussian beam profile via numerical convolution."""
    L = n_cells * a_cell
    span = max(5.0 / L, 4 * sigma_beam)
    x = np.linspace(-span, span, n_quad)
    dx = x[1] - x[0]
    profile = sinc2_profile(x, L)
    norm_relp = np.sum(profile) * dx

    if np.isscalar(delta):
        delta = np.array([delta])

    p = np.zeros_like(delta, dtype=float)
    for i, d in enumerate(delta):
        beam = np.exp(-0.5 * (x - d)**2 / sigma_beam**2)
        beam /= np.sum(beam) * dx  # normalize beam
        conv = np.sum(profile * beam) * dx
        p[i] = conv / (norm_relp / (n_quad * dx)) if norm_relp > 0 else 0
        # simpler: fraction of relp captured
        p[i] = np.sum(profile * beam) * dx / (np.sum(profile) * dx) * (np.sum(beam) * dx) * n_quad
    # redo properly: p = ∫ relp(x) beam(x-Δ) dx / ∫ relp(x) dx, with beam normalized
    p = np.zeros_like(delta, dtype=float)
    beam_norm_factor = sigma_beam * np.sqrt(2 * np.pi)
    for i, d in enumerate(delta):
        beam = np.exp(-0.5 * (x - d)**2 / sigma_beam**2) / beam_norm_factor
        # partiality = shell_equivalent_width * beam_weighted_relp / relp_total
        # For Gaussian beam, effective shell width ~ sigma_beam * sqrt(2pi)
        conv = np.sum(profile * beam) * dx
        p[i] = conv * beam_norm_factor / (norm_relp) if norm_relp > 0 else 0
    return np.clip(p, 0, 1)


def compute_partiality(delta, params, relp_model, beam_profile):
    """Dispatch to the right partiality function."""
    if relp_model == 'Gaussian':
        if beam_profile == 'Top-hat':
            return partiality_gauss_tophat(delta, params['sigma_relp'], params['shell_hw'])
        else:
            return partiality_gauss_gauss(delta, params['sigma_relp'], params['sigma_beam'])
    else:  # sinc²
        if beam_profile == 'Top-hat':
            return partiality_sinc2_tophat(delta, params['n_cells'], params['a_cell'], params['shell_hw'])
        else:
            return partiality_sinc2_gauss(delta, params['n_cells'], params['a_cell'], params['sigma_beam'])


# ── Monte Carlo + histogram helpers ──────────────────────────────────

def run_mc(params, relp_model, beam_profile, n_mc=80000, threshold=0.01):
    """Run MC sampling, return observed partialities above threshold."""
    cutoff = params['cutoff']
    offsets = np.random.uniform(-cutoff, cutoff, n_mc)
    p_obs = compute_partiality(offsets, params, relp_model, beam_profile)
    mask = p_obs > threshold
    return p_obs[mask]


def make_params(bw_pct, n_cells, a_cell, d_spacing):
    """Compute derived parameters from physical inputs."""
    bw = bw_pct / 100.0
    s = 1.0 / d_spacing
    relp_fwhm = 1.0 / (n_cells * a_cell)
    sigma_relp = relp_fwhm / 2.355
    shell_hw = 0.5 * bw * s
    sigma_beam = shell_hw / 2.355
    cutoff = 3.0 * (sigma_relp * 3 + max(shell_hw, 3 * sigma_beam))
    return {
        'sigma_relp': sigma_relp,
        'shell_hw': shell_hw,
        'sigma_beam': sigma_beam,
        'n_cells': n_cells,
        'a_cell': a_cell,
        'relp_fwhm': relp_fwhm,
        'cutoff': cutoff,
        'ratio': 2 * shell_hw / relp_fwhm,
    }


# ── Bandwidth sweep ──────────────────────────────────────────────────

def bw_sweep(n_cells, a_cell, d_spacing, relp_model, beam_profile,
             bw_list=None, n_mc=40000, threshold=0.01):
    """Sweep bandwidth, return σ(log p) for each."""
    if bw_list is None:
        bw_list = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]
    sigmas = []
    for bw_pct in bw_list:
        params = make_params(bw_pct, n_cells, a_cell, d_spacing)
        p_det = run_mc(params, relp_model, beam_profile, n_mc, threshold)
        if len(p_det) > 10:
            log_p = np.log(p_det)
            sigmas.append(np.std(log_p))
        else:
            sigmas.append(np.nan)
    return bw_list, sigmas


# ── Main interactive plot ────────────────────────────────────────────

def main():
    # Initial values
    init_bw = 0.40
    init_ncells = 10
    init_acell = 40.0
    init_dspacing = 3.0

    fig = plt.figure(figsize=(15, 11))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.30, hspace=0.40, wspace=0.30)

    ax_offset = fig.add_subplot(2, 2, 1)
    ax_pdist = fig.add_subplot(2, 2, 2)
    ax_logp = fig.add_subplot(2, 2, 3)
    ax_sweep = fig.add_subplot(2, 2, 4)

    # Slider axes
    slider_color = '#CECBF6'
    ax_bw = fig.add_axes([0.12, 0.18, 0.35, 0.025])
    ax_nc = fig.add_axes([0.12, 0.14, 0.35, 0.025])
    ax_ac = fig.add_axes([0.12, 0.10, 0.35, 0.025])
    ax_ds = fig.add_axes([0.12, 0.06, 0.35, 0.025])

    s_bw = Slider(ax_bw, 'BW (%)', 0.05, 2.0, valinit=init_bw, valstep=0.05, color=slider_color)
    s_nc = Slider(ax_nc, 'N cells', 2, 60, valinit=init_ncells, valstep=1, color=slider_color)
    s_ac = Slider(ax_ac, 'Cell (Å)', 10, 200, valinit=init_acell, valstep=5, color=slider_color)
    s_ds = Slider(ax_ds, 'd (Å)', 1.5, 10, valinit=init_dspacing, valstep=0.5, color=slider_color)

    # Radio buttons
    ax_beam = fig.add_axes([0.60, 0.06, 0.12, 0.12])
    ax_relp = fig.add_axes([0.78, 0.06, 0.12, 0.12])
    radio_beam = RadioButtons(ax_beam, ['Top-hat', 'Gaussian'], active=0)
    radio_relp = RadioButtons(ax_relp, ['sinc²', 'Gaussian'], active=0)
    ax_beam.set_title('Beam profile', fontsize=10, fontweight='bold')
    ax_relp.set_title('Relp model', fontsize=10, fontweight='bold')

    # Stats text
    ax_stats = fig.add_axes([0.55, 0.18, 0.40, 0.05])
    ax_stats.axis('off')
    stats_text = ax_stats.text(0, 0.5, '', fontsize=11, fontfamily='monospace',
                               verticalalignment='center')

    def update(_=None):
        bw_pct = s_bw.val
        n_cells = int(s_nc.val)
        a_cell = s_ac.val
        d_spacing = s_ds.val
        beam_profile = radio_beam.value_selected
        relp_model = radio_relp.value_selected

        params = make_params(bw_pct, n_cells, a_cell, d_spacing)

        # ── Partiality vs offset curve ──
        cutoff = params['cutoff']
        delta_range = np.linspace(-cutoff, cutoff, 500)
        p_curve = compute_partiality(delta_range, params, relp_model, beam_profile)

        ax_offset.cla()
        ax_offset.plot(delta_range * 1e3, p_curve, color='#534AB7', lw=2)
        ax_offset.fill_between(delta_range * 1e3, p_curve, alpha=0.15, color='#534AB7')
        ax_offset.axhline(0.01, color='gray', ls=':', lw=1, alpha=0.5)
        ax_offset.set_xlabel('Ewald offset (×10⁻³ Å⁻¹)')
        ax_offset.set_ylabel('Partiality')
        ax_offset.set_title('Partiality vs Ewald offset')
        ax_offset.set_ylim(-0.05, 1.1)

        # ── Monte Carlo ──
        p_det = run_mc(params, relp_model, beam_profile, n_mc=80000)

        if len(p_det) < 20:
            for ax in [ax_pdist, ax_logp, ax_sweep]:
                ax.cla()
                ax.text(0.5, 0.5, 'Too few detections', ha='center', va='center',
                        transform=ax.transAxes)
            stats_text.set_text('Too few detected reflections')
            fig.canvas.draw_idle()
            return

        log_p = np.log(p_det)
        mean_p = np.mean(p_det)
        sigma_ln = np.std(log_p)
        mean_log_p = np.mean(log_p)
        det_frac = len(p_det) / 80000 * 100

        stats_text.set_text(
            f'σ(log p) = {sigma_ln:.3f}    '
            f'⟨p⟩ = {mean_p:.3f}    '
            f'shell/relp = {params["ratio"]:.2f}    '
            f'detected = {det_frac:.1f}%'
        )

        # ── Partiality distribution ──
        ax_pdist.cla()
        ax_pdist.hist(p_det, bins=60, range=(0, 1), density=True,
                      color='#1D9E75', alpha=0.6, edgecolor='#0F6E56', lw=0.5)
        ax_pdist.set_xlabel('Partiality')
        ax_pdist.set_ylabel('Density')
        ax_pdist.set_title('Observed partiality distribution')

        # ── Log-partiality distribution ──
        ax_logp.cla()
        log_min = max(-8, log_p.min() - 0.5)
        ax_logp.hist(log_p, bins=60, range=(log_min, 0.5), density=True,
                     color='#D85A30', alpha=0.6, edgecolor='#993C1D', lw=0.5,
                     label='Observed')
        x_fit = np.linspace(log_min, 0.5, 200)
        gauss_fit = norm.pdf(x_fit, mean_log_p, sigma_ln)
        ax_logp.plot(x_fit, gauss_fit, color='#534AB7', lw=2, label=f'Gaussian (σ={sigma_ln:.2f})')
        ax_logp.set_xlabel('log(partiality)')
        ax_logp.set_ylabel('Density')
        ax_logp.set_title('log(partiality) distribution')
        ax_logp.legend(fontsize=9)

        # ── BW sweep ──
        ax_sweep.cla()
        bw_list, sigmas = bw_sweep(n_cells, a_cell, d_spacing, relp_model, beam_profile)
        ax_sweep.plot(bw_list, sigmas, 'o-', color='#534AB7', lw=2, markersize=5)
        ax_sweep.axhline(sigma_ln, color='#D85A30', ls='--', lw=1, alpha=0.7)
        ax_sweep.axvline(bw_pct, color='#D85A30', ls='--', lw=1, alpha=0.7)
        ax_sweep.set_xlabel('Bandwidth (%)')
        ax_sweep.set_ylabel('σ(log p)')
        ax_sweep.set_title('σ(log p) vs bandwidth')

        fig.canvas.draw_idle()

    s_bw.on_changed(update)
    s_nc.on_changed(update)
    s_ac.on_changed(update)
    s_ds.on_changed(update)
    radio_beam.on_clicked(update)
    radio_relp.on_clicked(update)

    fig.suptitle('Ewald offset × partiality simulation for SFX', fontsize=14, fontweight='bold')
    update()
    plt.show()


if __name__ == '__main__':
    main()