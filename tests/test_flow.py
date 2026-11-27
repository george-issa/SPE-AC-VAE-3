"""Conditional flow-matching posterior: coordinates, constraints, training signal."""

import math

import numpy as np
import pytest
import torch

from spevae.data.label_bank import _dispersive_gauge
from spevae.data.pole_prior import PolePriorGenerator
from spevae.models.flow import A_DUST, FlowPosterior, PoleCoords, load_flow, save_flow


@pytest.fixture(scope="module")
def batch():
    g = PolePriorGenerator(beta_range=(8.0, 8.0), dtau_choices=(0.125,),
                           residue_mode="dispersive")
    g.seed_scalars(0)
    return g.batch(32, torch.Generator().manual_seed(0))


def test_gauge_matches_the_label_bank_reference(batch):
    lab = batch.poles_true
    c = PoleCoords().gauge(lab["a"].double(), lab["b"].double(), lab["a"] > 0)
    for i in range(8):
        m = int(lab["n_true"][i])
        a, b = lab["a"][i, :m].double().numpy(), lab["b"][i, :m].double().numpy()
        ref = _dispersive_gauge(a, b)
        assert np.abs(b + float(c[i]) * a).max() == pytest.approx(np.abs(b + ref * a).max(),
                                                                   rel=1e-9, abs=1e-12)


def test_coordinates_roundtrip_the_labels(batch):
    pc = PoleCoords()
    lab = {k: batch.poles_true[k].double() for k in ("eps", "gamma", "a", "b")}
    ps = pc.decode(pc.encode(lab))
    valid = lab["a"] > 0
    for k in ("eps", "gamma"):
        assert torch.allclose(getattr(ps, k)[valid], lab[k][valid], rtol=1e-6, atol=1e-6)
    assert torch.allclose(ps.a[valid], lab["a"][valid], rtol=1e-6, atol=1e-9)
    assert torch.allclose(ps.b[valid], lab["b"][valid], rtol=1e-6, atol=1e-9)
    assert float(ps.a[~valid].abs().max()) == 0.0 and float(ps.b[~valid].abs().max()) == 0.0


def test_samples_satisfy_the_constraints(batch):
    torch.manual_seed(0)
    m = FlowPosterior(d_model=32, enc_layers=1, flow_layers=1, n_heads=2, d_ff=64).eval()
    ps = m.sample(batch.G[:2], batch.x, batch.beta, sigma=batch.sigma[:2], n_samples=5,
                  steps=3, generator=torch.Generator().manual_seed(1))
    assert ps.a.shape == (10, 12)
    assert float(ps.a.min()) >= 0 and torch.allclose(ps.a.sum(-1), torch.ones(10), atol=1e-2)
    assert torch.allclose(ps.b.sum(-1), torch.zeros(10), atol=1e-5)
    assert float(ps.gamma.min()) > m.coords.gamma_min
    assert float(ps.eps.abs().max()) <= m.nu_max(batch.beta) + 1e-4


def test_permuting_pole_tokens_permutes_the_velocity(batch):
    torch.manual_seed(0)
    m = FlowPosterior(d_model=32, enc_layers=1, flow_layers=2, n_heads=2, d_ff=64).eval()
    for p in m.out.parameters():
        torch.nn.init.normal_(p, std=0.1)
    mem, keep, _ = m.context(batch.G[:2], batch.x, batch.beta, sigma=batch.sigma[:2])
    x = torch.randn(2, 12, 5)
    perm = torch.randperm(12)
    t = torch.tensor([0.3, 0.7])
    v = m.velocity(x, t, mem, keep)
    vp = m.velocity(x[:, perm], t, mem, keep)
    assert torch.allclose(v[:, perm], vp, atol=1e-5)


def test_loss_decreases_and_checkpoint_roundtrips(batch, tmp_path):
    torch.manual_seed(0)
    m = FlowPosterior(d_model=32, enc_layers=1, flow_layers=1, n_heads=2, d_ff=64)
    m.coords.fit([batch.poles_true])
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)
    first = None
    for _ in range(60):
        loss = m.loss(batch, generator=g)
        first = float(loss) if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.7 * first
    p = save_flow(m, str(tmp_path / "f.pt"), steps=60)
    m2 = load_flow(p)
    assert torch.equal(m2.coords.mean, m.coords.mean)
    x = torch.randn(1, 12, 5)
    mem, keep, _ = m.eval().context(batch.G[:1], batch.x, batch.beta, sigma=batch.sigma[:1])
    assert torch.allclose(m.velocity(x, torch.tensor([0.5]), mem, keep),
                          m2.velocity(x, torch.tensor([0.5]), mem, keep))


def test_refine_pulls_samples_onto_the_data_and_the_tether_holds():
    from spevae.evaluation import cases as C
    from spevae.models.flow import refine_samples
    torch.manual_seed(0)
    inst = C.draw(C.realize("irregular triplet"), 1e-3, 7)
    tr = inst.truth
    m = FlowPosterior(d_model=32, enc_layers=1, flow_layers=1, n_heads=2, d_ff=64).eval()
    lab = {"eps": tr.poles.eps, "gamma": tr.poles.gamma, "a": tr.poles.a, "b": tr.poles.b}
    m.coords.fit([lab])
    z_true = torch.cat([m.coords.encode(lab), torch.full((1, 9, 5), 0.0)], 1)
    z_true[0, 3:, 4] = -1.0                                  # 9 empty slots
    z0 = z_true.repeat(4, 1, 1)
    z0[..., :4] += 0.3 * torch.randn(z0[..., :4].shape) * (z0[..., 4:] > 0)
    norm = torch.ones(4)
    G, s = inst.G.expand(4, -1), inst.sigma.expand(4, -1)
    ps0 = m.coords.decode(z0, norm=norm, nu_max=m.nu_max(8.0))
    from spevae.physics import PoleGreens
    d0 = ((PoleGreens(tr.quad)(ps0, tr.x) - G) / s) @ inst.isr.T
    chi0 = (d0 * d0).mean(-1)
    ps, chi = refine_samples(m, z0, norm, 8.0, tr.x, G, s, inst.isr, lam=0.1, steps=300)
    assert float(chi.max()) < 3.0 < float(chi0.min())       # pulled to the noise level
    assert (ps.a[:, 3:] == 0).all()                          # presence pattern kept
    _, chi_stiff = refine_samples(m, z0, norm, 8.0, tr.x, G, s, inst.isr, lam=1e9, steps=50)
    assert torch.allclose(chi_stiff, chi0, rtol=1e-3)        # an infinite tether does not move
    # the default solver is the least-squares one; the first-order loop stays selectable
    # and, on the same samples and objective, does not fit better
    _, chi_adam = refine_samples(m, z0, norm, 8.0, tr.x, G, s, inst.isr, lam=0.1, steps=300,
                                 solver="adam")
    assert float(chi_adam.max()) < 3.0
    assert float(chi.mean()) <= float(chi_adam.mean()) + 1e-3
    with pytest.raises(ValueError):
        refine_samples(m, z0, norm, 8.0, tr.x, G, s, inst.isr, lam=0.1, solver="sgd")


def test_feedback_flow_sees_the_likelihood_and_trains(batch, tmp_path):
    """Simulator feedback: the Gauss-Newton step vanishes at the truth, is
    non-zero away from it, and a feedback model trains, samples and reloads."""
    torch.manual_seed(0)
    m = FlowPosterior(d_model=32, enc_layers=1, flow_layers=1, n_heads=2, d_ff=64, feedback=True)
    lab = {k: batch.poles_true[k] for k in ("eps", "gamma", "a", "b")}
    m.coords.fit([lab])
    x1 = m.coords.encode(lab)
    _, _, norm = m.context(batch.G_clean, batch.x, batch.beta, sigma=batch.sigma)
    data = m.feedback_data(batch.G_clean, batch.x, batch.beta, batch.sigma, norm,
                           inv_sqrt_R=batch.inv_sqrt_R)
    step, logchi = m.gn_feedback(x1, data)                 # noise-free data, the true poles
    assert step.shape == (32, 12, 4) and float(step.abs().max()) < 5e-2
    assert (step[x1[..., 4] < 0] == 0).all()               # absent tokens get no step
    xp = x1.clone()
    xp[..., :4] += 0.3 * torch.randn(xp[..., :4].shape) * (x1[..., 4:] > 0)
    step_p, logchi_p = m.gn_feedback(xp, data)
    assert float(step_p.abs().mean()) > 10 * float(step.abs().mean())
    assert float(logchi_p.median()) > float(logchi.median())
    # trains: the feedback layers start at zero and move
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    g = torch.Generator().manual_seed(0)
    first = None
    for _ in range(25):
        loss = m.loss(batch, generator=g)
        first = float(loss.detach()) if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss.detach()) < first and float(m.fb_proj.weight.abs().sum()) > 0
    z, _ = m.eval().sample_coords(batch.G[:2], batch.x, batch.beta, sigma=batch.sigma[:2],
                                  n_samples=3, steps=3, inv_sqrt_R=batch.inv_sqrt_R,
                                  generator=torch.Generator().manual_seed(1))
    assert z.shape == (6, 12, 5) and torch.isfinite(z).all()
    with pytest.raises(ValueError):
        m.sample_coords(batch.G[:1], batch.x, batch.beta, n_samples=2, steps=2)
    path = save_flow(m, str(tmp_path / "fb.pt"))
    m2 = load_flow(path)
    assert m2.feedback and torch.equal(m2.fb_proj.weight, m.fb_proj.weight)
    assert not FlowPosterior(d_model=32, enc_layers=1, flow_layers=1, n_heads=2, d_ff=64).feedback


def test_lowered_floor_keeps_the_proposals_and_lets_the_fit_narrow_a_line():
    from spevae.models.flow import LoweredFloor, refine_samples
    torch.manual_seed(0)
    beta = 8.0
    m = FlowPosterior(d_model=32, enc_layers=1, flow_layers=1, n_heads=2, d_ff=64).eval()
    low = LoweredFloor(m, 4.0)
    assert abs(low.coords.gamma_min - m.coords.gamma_min / 4.0) < 1e-15 and low.P == m.P
    # the same token coordinates, re-expressed, are the same poles
    z = torch.randn(5, m.P, 5)
    p0 = m.coords.decode(z, nu_max=m.nu_max(beta))
    p1 = low.coords.decode(low.recode(z), nu_max=low.nu_max(beta))
    for k in ("eps", "gamma", "a", "b"):
        assert torch.allclose(getattr(p0, k), getattr(p1, k), rtol=1e-5, atol=1e-6), k
    with pytest.raises(ValueError):
        LoweredFloor(m, 0.5)
    # follow_within: a pole within 1.25 trained floors keeps its offset above the floor
    # (so it sits as close to the lowered floor as it sat to the trained one); wider poles keep their width
    z = z.clone()
    z[:, ::3, 1] = -5.0                                       # every third pole pinned at the floor
    p0 = m.coords.decode(z, nu_max=m.nu_max(beta))
    p2 = low.coords.decode(low.recode(z, follow_within=1.25), nu_max=low.nu_max(beta))
    g0, drop = m.coords.gamma_min, m.coords.gamma_min - low.coords.gamma_min
    pinned = p0.gamma <= 1.25 * g0
    assert bool(pinned.any()) and bool((~pinned).any())
    assert torch.allclose(p2.gamma[pinned], p0.gamma[pinned] - drop, rtol=1e-4, atol=1e-6)
    assert torch.allclose(p2.gamma[~pinned], p0.gamma[~pinned], rtol=1e-5, atol=1e-6)
    assert torch.allclose(p2.eps, p0.eps) and torch.allclose(p2.a, p0.a, atol=1e-6)
    # two exact lines at omega = +-1 (a Hubbard atom); proposals: two poles there, 1.5x the floor wide
    x = torch.linspace(0.0, 1.0, 65)
    e = torch.tensor([-1.0, 1.0], dtype=torch.float64) * beta
    G = (0.5 * torch.exp(-x.double()[:, None] * e) / (1.0 + torch.exp(-e))).sum(-1).float()
    sigma = torch.full_like(G, 1e-4 * float(G.max()))
    lab = {"eps": e.float()[None], "gamma": torch.full((1, 2), 1.5 * m.coords.gamma_min),
           "a": torch.full((1, 2), 0.5), "b": torch.zeros(1, 2)}
    z0 = torch.cat([m.coords.encode(lab), torch.zeros(1, m.P - 2, 5)], 1)
    z0[0, 2:, 4] = -1.0
    z0 = z0.repeat(3, 1, 1)
    args = (torch.ones(3), beta, x, G.expand(3, -1), sigma.expand(3, -1), torch.eye(65), 1.0)
    ps, chi = refine_samples(m, z0, *args)
    ps_low, chi_low = refine_samples(low, low.recode(z0), *args)
    assert float(ps.gamma[:, :2].min()) >= m.coords.gamma_min                # the trained floor binds
    assert float(ps_low.gamma[:, :2].max()) < m.coords.gamma_min             # the lowered one is used
    assert float(ps_low.gamma[:, :2].min()) >= low.coords.gamma_min
    assert float(chi_low.max()) < 0.5 * float(chi.min())                     # and the lines fit better


def test_tokens_marked_as_lines_decode_and_fit_as_zero_width_lines():
    from spevae.models.flow import LoweredFloor, add_pole, refine_samples
    torch.manual_seed(0)
    beta = 8.0
    m = FlowPosterior(d_model=32, enc_layers=1, flow_layers=1, n_heads=2, d_ff=64).eval()
    z = torch.randn(5, m.P, 5)
    lines = torch.zeros(5, m.P, dtype=torch.bool)
    lines[:, ::2] = True
    p = m.coords.decode(z, nu_max=m.nu_max(beta), lines=lines)
    live = p.a != 0
    assert bool((lines & live).any()) and bool((~lines & live).any())
    assert (p.gamma[lines & live] == 0).all() and (p.b[lines & live] == 0).all()
    assert (p.gamma[~lines & live] >= m.coords.gamma_min).all()
    assert torch.allclose(p.a.sum(-1), torch.ones(5), atol=1e-5) and float(p.b.sum(-1).abs().max()) < 1e-5
    p0 = m.coords.decode(z, nu_max=m.nu_max(beta))
    p1 = m.coords.decode(z, nu_max=m.nu_max(beta), lines=torch.zeros_like(lines))
    for k in ("eps", "gamma", "a", "b"):                      # no line marked: the same poles
        assert torch.allclose(getattr(p0, k), getattr(p1, k), atol=1e-6), k
    assert torch.equal(p.eps, p0.eps) and torch.equal(p.a, p0.a)
    # two exact lines (a Hubbard atom) at error bars of 1e-5; proposals: two poles pinned at the
    # floor, each a tenth of a unit off, with the wrong weights
    x = torch.linspace(0.0, 1.0, 65, dtype=torch.float64)
    e, w = torch.tensor([-1.2, 0.3], dtype=torch.float64) * beta, torch.tensor([0.3, 0.7], dtype=torch.float64)
    G = (w * torch.exp(-x[:, None] * e) / (1.0 + torch.exp(-e))).sum(-1)
    sigma = torch.full_like(G, 1e-5 * float(G.max()))
    G = G + sigma * torch.randn(65, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    g0 = m.coords.gamma_min
    lab = {"eps": torch.zeros(1, m.P), "gamma": torch.ones(1, m.P), "a": torch.zeros(1, m.P),
           "b": torch.zeros(1, m.P)}
    lab["eps"][0, :2] = torch.tensor([-1.1, 0.4]) * beta
    lab["gamma"][0, :2] = 1.05 * g0
    lab["a"][0, :2] = 0.5
    low = LoweredFloor(m, 4.0)
    z0 = low.recode(m.coords.encode(lab), follow_within=1.25).double().repeat(3, 1, 1)
    ln = torch.zeros(3, m.P, dtype=torch.bool)
    ln[:, :2] = True
    args = (torch.ones(3, dtype=torch.float64), beta, x, G.expand(3, -1), sigma.expand(3, -1),
            torch.eye(65, dtype=torch.float64))
    ps, chi, u = refine_samples(low, z0, *args, lam=10.0, anneal_from=1e-2, return_coords=True, lines=ln)
    assert float(chi.max()) < 1.6                              # the lines reach the data
    assert (ps.gamma[:, :2] == 0).all() and (ps.b == 0).all()
    assert float((ps.eps[:, :2] / beta - torch.tensor([-1.2, 0.3])).abs().max()) < 2e-3
    assert float((ps.a[:, :2] - w).abs().max()) < 2e-3
    # the same proposals as Lorentzians of the lowered floor cannot: a Lorentzian has tails
    _, chi_wide = refine_samples(low, z0, *args, lam=10.0, anneal_from=1e-2)
    assert float(chi_wide.min()) > 5.0
    # a birth borrows a line from a line
    zc = torch.cat([u, z0[..., 4:5]], -1)
    z_new, born, ln_new = add_pole(low, zc, ps, generator=torch.Generator().manual_seed(1), lines=ln)
    assert bool(born.all()) and (ln_new.sum(-1) == 3).all() and (ln.sum(-1) == 2).all()
    assert ((z_new[..., 4] > 0).sum(-1) == 3).all()
