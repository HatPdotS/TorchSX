"""Synthetic validation of the collapse-free scale-consistency loss.

These tests pin the scientific claims of :class:`torchsx.nn.consistency_loss.ScaleConsistencyNLL`
on data where the answer is known:

* it recovers the latent per-HKL intensities ``J_h`` and the scale width;
* it is a proper scoring rule (uncertainty inflation *and* over-confidence both
  raise the loss);
* the global scale gauge is a real flat direction of the data term (so an
  anchor is required), and the scale-mean centering term breaks it;
* the two information-leak degeneracies are real — a scale that sees its own
  ``I_obs``, or a free per-HKL scale, makes the loss independent of ``J_h`` —
  and the feature contract (constrained, intensity-blind scale) restores the
  coupling;
* multi-system training with a shared scale function recovers each system's
  relative intensities.
"""

from __future__ import annotations

import math

import torch

from torchsx.nn.consistency_loss import ScaleConsistencyNLL


def _make_synth(
    *, n_hkl=40, n_cry=80, obsper=25, sigma_scale=0.3, snr="med", seed=0,
):
    """Generate observations from a known (J_h, per-crystal scale, σ_scale)."""
    g = torch.Generator().manual_seed(seed)
    true_logI0 = torch.randn(n_hkl, generator=g) * 0.8 + 4.0
    true_scale = torch.randn(n_cry, generator=g) * 0.5
    hkl_id = torch.randint(0, n_hkl, (n_cry * obsper,), generator=g)
    cid = torch.arange(n_cry).repeat_interleave(obsper)
    log_mean = true_logI0[hkl_id] + true_scale[cid]
    log_S = log_mean + torch.randn(log_mean.shape, generator=g) * sigma_scale
    S = log_S.exp()
    frac = {"med": 0.1, "high": 0.01}[snr]
    sigma_det = frac * S.abs() + (0.5 if snr == "high" else 1.0)
    I_obs = S + torch.randn(S.shape, generator=g) * sigma_det
    return {
        "true_logI0": true_logI0, "true_scale": true_scale,
        "hkl_id": hkl_id, "cid": cid, "I_obs": I_obs, "sigma_det": sigma_det,
        "n_hkl": n_hkl, "n_cry": n_cry, "sigma_scale": sigma_scale,
    }


def _mu_seed(I_obs, hkl_id, n_hkl):
    mu = torch.zeros(n_hkl)
    cnt = torch.zeros(n_hkl)
    mu.scatter_add_(0, hkl_id, I_obs.clamp(min=1e-3))
    cnt.scatter_add_(0, hkl_id, torch.ones_like(I_obs))
    return (mu / cnt.clamp(min=1)).log()


def _fit(d, *, steps=400, lr=0.05, n_points=1024):
    """Joint fit of log_I0 + mean-centered per-crystal scale + global σ_scale."""
    mu = _mu_seed(d["I_obs"], d["hkl_id"], d["n_hkl"])
    m = ScaleConsistencyNLL(d["n_hkl"], mu_init=mu, n_points=n_points,
                            S_min=1e-2, S_max=1e5)
    raw = torch.zeros(d["n_cry"], requires_grad=True)
    logsig = torch.tensor(-1.0, requires_grad=True)
    opt = torch.optim.Adam([m.log_I0, raw, logsig], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        lsm = (raw - raw.mean())[d["cid"]]
        lss = logsig.expand(d["hkl_id"].shape[0])
        m(lsm, lss, d["I_obs"], d["sigma_det"], d["hkl_id"]).backward()
        opt.step()
    return m, raw.detach(), float(logsig.detach().exp())


# ── recovery ────────────────────────────────────────────────────────────


def test_recovers_latent_intensities():
    d = _make_synth(seed=0)
    m, _, _ = _fit(d)
    rec = m.log_I0.detach()
    a = rec - rec.mean()
    b = d["true_logI0"] - d["true_logI0"].mean()
    corr = torch.corrcoef(torch.stack([a, b]))[0, 1]
    assert corr > 0.97, f"log_I0 recovery corr={corr:.3f}"
    assert (a - b).std() < 0.15


def test_recovers_scale_width():
    d = _make_synth(seed=1, sigma_scale=0.35)
    _, _, fitted_sigma = _fit(d)
    assert abs(fitted_sigma - d["sigma_scale"]) < 0.1, fitted_sigma


# ── proper scoring rule (no inflation, no over-confidence) ────────────────


def test_no_uncertainty_inflation():
    """Loss has an interior minimum in σ_scale — broadening *or* sharpening hurts."""
    d = _make_synth(seed=2)
    # Place log_I0 / scale at the (gauge-fixed) truth so only σ_scale varies.
    log_I0 = d["true_logI0"] + d["true_scale"].mean()
    m = ScaleConsistencyNLL(d["n_hkl"], mu_init=log_I0, n_points=1024,
                            S_min=1e-2, S_max=1e5)
    lsm = (d["true_scale"] - d["true_scale"].mean())[d["cid"]]

    log_sigmas = torch.linspace(math.log(0.05), math.log(2.0), 25)
    losses = []
    for ls in log_sigmas:
        lss = ls.expand(d["hkl_id"].shape[0])
        with torch.no_grad():
            losses.append(float(m.data_nll_per_obs(
                lsm, lss, d["I_obs"], d["sigma_det"], d["hkl_id"]).sum()))
    losses = torch.tensor(losses)
    k = int(losses.argmin())
    assert 0 < k < len(losses) - 1, f"σ minimum on boundary (k={k})"
    # Strictly worse when the predicted width is doubled or halved.
    assert losses[k] < losses[0] and losses[k] < losses[-1]


# ── gauge ─────────────────────────────────────────────────────────────────


def test_data_term_global_gauge_is_flat_and_centering_pins_it():
    d = _make_synth(seed=3)
    m = ScaleConsistencyNLL(d["n_hkl"], mu_init=d["true_logI0"].clone(),
                            n_points=768, S_min=1e-2, S_max=1e5)
    lsm = torch.zeros(d["hkl_id"].shape[0])
    lss = torch.full((d["hkl_id"].shape[0],), -1.0)
    base = m.data_nll_per_obs(lsm, lss, d["I_obs"], d["sigma_det"], d["hkl_id"]).sum()

    # Joint shift (log_I0 += c, scale -= c) leaves the data term invariant.
    c = 0.7
    m.log_I0.data += c
    shifted = m.data_nll_per_obs(lsm - c, lss, d["I_obs"], d["sigma_det"],
                                 d["hkl_id"]).sum()
    assert torch.allclose(base, shifted, atol=1e-3), (float(base), float(shifted))

    # The scale-mean centering term is NOT invariant — it pins the gauge.
    m.center_scale_mean = True
    m.lambda_center = 1.0
    l0 = m(torch.zeros_like(lsm), lss, d["I_obs"], d["sigma_det"], d["hkl_id"])
    lc = m(torch.zeros_like(lsm) - c, lss, d["I_obs"], d["sigma_det"], d["hkl_id"])
    assert not torch.allclose(l0, lc, atol=1.0)


# ── information-leak degeneracies (the feature contract) ──────────────────


def test_own_intensity_leak_makes_loss_independent_of_J():
    """A scale that sees its own I_obs fits any J_h → loss flat in J_h."""
    d = _make_synth(seed=4, snr="high")
    I, hid, sd = d["I_obs"], d["hkl_id"], d["sigma_det"]
    pos = I > 1.0  # the cheat needs a positive target to hit
    I, hid, sd = I[pos], hid[pos], sd[pos]
    cid = d["cid"][pos]

    m = ScaleConsistencyNLL(d["n_hkl"], n_points=1024, S_min=1e-2, S_max=1e5)
    lss = torch.full((I.shape[0],), math.log(0.05))

    logI0_A = d["true_logI0"].clone()
    logI0_B = d["true_logI0"] + torch.randn(d["n_hkl"], generator=torch.Generator().manual_seed(9)) * 2.0

    def cheat_nll(logI0):
        m.log_I0.data = logI0.clone()
        # scale head outputs s_k = log(I_k) - log_I0[h]  ⇒  log_mean = log(I_k)
        lsm = I.clamp(min=1e-3).log() - logI0[hid]
        return float(m.data_nll_per_obs(lsm, lss, I, sd, hid).sum())

    def honest_nll(logI0):
        m.log_I0.data = logI0.clone()
        lsm = torch.zeros(I.shape[0])  # intensity-blind per-crystal scale (here 0)
        return float(m.data_nll_per_obs(lsm, lss, I, sd, hid).sum())

    a_cheat, b_cheat = cheat_nll(logI0_A), cheat_nll(logI0_B)
    a_hon, b_hon = honest_nll(logI0_A), honest_nll(logI0_B)
    # Cheat: J_h is irrelevant (loss ~identical for very different J).
    assert abs(a_cheat - b_cheat) / abs(a_cheat) < 0.02
    # Contract: the intensity-blind scale makes the loss strongly J-dependent.
    assert abs(a_hon - b_hon) > 10.0 * abs(a_cheat - b_cheat)


def test_free_per_hkl_scale_leaves_J_unidentified():
    """A free per-HKL scale is a per-HKL gauge confounded with log_I0."""
    d = _make_synth(seed=5)
    m = ScaleConsistencyNLL(d["n_hkl"], mu_init=d["true_logI0"].clone(),
                            n_points=768, S_min=1e-2, S_max=1e5)
    lss = torch.full((d["hkl_id"].shape[0],), -1.0)
    base = m.data_nll_per_obs(torch.zeros(d["hkl_id"].shape[0]), lss,
                              d["I_obs"], d["sigma_det"], d["hkl_id"]).sum()

    delta = torch.randn(d["n_hkl"], generator=torch.Generator().manual_seed(7))
    m.log_I0.data += delta                       # move J_h arbitrarily per HKL
    lsm = (-delta)[d["hkl_id"]]                   # a per-HKL scale absorbs it
    shifted = m.data_nll_per_obs(lsm, lss, d["I_obs"], d["sigma_det"],
                                 d["hkl_id"]).sum()
    assert torch.allclose(base, shifted, atol=1e-3)


# ── multi-system guardrail ────────────────────────────────────────────────


def test_multisystem_shared_scale_recovers_both():
    """One shared geometry-driven scale + per-(system,hkl) J recovers both systems."""
    g = torch.Generator().manual_seed(11)
    w_true = 0.6
    systems = []
    offset = 0
    for s in range(2):
        n_hkl, n_cry, obsper = 30, 50, 20
        true_logI0 = torch.randn(n_hkl, generator=g) * 0.8 + 4.0
        feat_c = torch.randn(n_cry, generator=g)          # per-crystal geometry cue
        scale_c = w_true * feat_c                          # SHARED scale function
        hkl_id = torch.randint(0, n_hkl, (n_cry * obsper,), generator=g)
        cid = torch.arange(n_cry).repeat_interleave(obsper)
        log_mean = true_logI0[hkl_id] + scale_c[cid]
        S = (log_mean + torch.randn(log_mean.shape, generator=g) * 0.25).exp()
        sd = 0.1 * S.abs() + 1.0
        I = S + torch.randn(S.shape, generator=g) * sd
        systems.append({
            "true_logI0": true_logI0, "hkl_id": hkl_id + offset,
            "feat": feat_c[cid], "I": I, "sd": sd, "n_hkl": n_hkl,
            "slice": slice(offset, offset + n_hkl),
        })
        offset += n_hkl

    n_hkl_total = offset
    hkl_id = torch.cat([s["hkl_id"] for s in systems])
    feat = torch.cat([s["feat"] for s in systems])
    I = torch.cat([s["I"] for s in systems])
    sd = torch.cat([s["sd"] for s in systems])

    mu = _mu_seed(I, hkl_id, n_hkl_total)
    m = ScaleConsistencyNLL(n_hkl_total, mu_init=mu, n_points=1024,
                            S_min=1e-2, S_max=1e5)
    w = torch.tensor(0.0, requires_grad=True)        # ONE shared scale parameter
    logsig = torch.tensor(-1.0, requires_grad=True)
    opt = torch.optim.Adam([m.log_I0, w, logsig], lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        lsm = w * feat
        lss = logsig.expand(hkl_id.shape[0])
        m(lsm, lss, I, sd, hkl_id).backward()
        opt.step()

    rec = m.log_I0.detach()
    for s in systems:
        a = rec[s["slice"]] - rec[s["slice"]].mean()
        b = s["true_logI0"] - s["true_logI0"].mean()
        corr = torch.corrcoef(torch.stack([a, b]))[0, 1]
        assert corr > 0.9, f"per-system recovery corr={corr:.3f}"
    assert abs(float(w.detach()) - w_true) < 0.15


# ── posterior ─────────────────────────────────────────────────────────────


def test_posterior_effective_sigma_shrinks_with_multiplicity():
    # Skewed HKL frequencies so multiplicity spans a wide range (rare → common).
    d = _make_synth(seed=6)
    g = torch.Generator().manual_seed(60)
    n_obs = d["hkl_id"].shape[0]
    skewed = (torch.rand(n_obs, generator=g).pow(3) * d["n_hkl"]).long().clamp(max=d["n_hkl"] - 1)
    d["hkl_id"] = skewed

    m, raw, sigma = _fit(d, steps=250)
    lsm = (raw - raw.mean())[d["cid"]]
    lss = torch.full((d["hkl_id"].shape[0],), math.log(sigma))
    I0, sigma_eff = m.posterior(lsm, lss, d["I_obs"], d["sigma_det"], d["hkl_id"])

    assert (sigma_eff > 0).all() and torch.isfinite(sigma_eff).all()
    # Compare the *log-space* posterior width (σ_eff = I0·σ_log; the I0 factor
    # otherwise confounds the multiplicity trend). More (consistent)
    # observations ⇒ tighter posterior on the latent.
    sigma_log = (sigma_eff / I0)
    mult = torch.bincount(d["hkl_id"], minlength=d["n_hkl"]).float()
    seen = mult > 0
    corr = torch.corrcoef(torch.stack([mult[seen].log(), sigma_log[seen].log()]))[0, 1]
    assert corr < -0.5, f"σ_log vs multiplicity corr={corr:.3f}"
