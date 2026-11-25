"""Exactly solvable models: every formula checked against an independent route."""

import math

import numpy as np
import pytest
import torch

from spevae.data import exact_models as X


def _bz_sum_2d(eps_fn, z, n=600):
    k = (np.arange(n) + 0.5) * 2 * math.pi / n
    k1, k2 = np.meshgrid(k, k, indexing="ij")
    return np.mean(1.0 / (z - eps_fn(k1, k2)))


@pytest.mark.parametrize("z", [0.3 + 0.4j, -1.7 + 0.25j, 2.9 + 0.6j])
def test_lattice_greens_match_brute_force_brillouin_zone_sums(z):
    t, tp = 0.8, 0.25
    sq = _bz_sum_2d(lambda a, b: -2 * t * (np.cos(a) + np.cos(b)) - 4 * tp * np.cos(a) * np.cos(b), z)
    assert abs(X.lattice_greens("tb_square", z, t=t, tp=tp) - sq) < 1e-9
    tri = _bz_sum_2d(lambda a, b: -2 * t * (np.cos(a) + np.cos(b) + np.cos(a - b)), z)
    assert abs(X.lattice_greens("tb_triangular", z, t=t) - tri) < 1e-9
    k = (np.arange(600) + 0.5) * 2 * math.pi / 600
    k1, k2 = np.meshgrid(k, k, indexing="ij")
    f2 = t * t * (3 + 2 * np.cos(k1) + 2 * np.cos(k2) + 2 * np.cos(k1 - k2))
    assert abs(X.lattice_greens("tb_honeycomb", z, t=t) - np.mean(z / (z * z - f2))) < 1e-9
    kk = (np.arange(4000) + 0.5) * 2 * math.pi / 4000
    assert abs(X.lattice_greens("tb_chain", z, t=t) - np.mean(1 / (z + 2 * t * np.cos(kk)))) < 1e-9
    t1, t2 = 1.1, 0.6
    ssh = np.mean(z / (z * z - np.abs(t1 + t2 * np.exp(1j * kk)) ** 2))
    assert abs(X.lattice_greens("ssh_chain", z, t1=t1, t2=t2) - ssh) < 1e-9


def test_square_lattice_matches_the_elliptic_integral_closed_form():
    mp = pytest.importorskip("mpmath")
    t = 1.0
    for z in (0.7 + 0.3j, 3.1 + 0.2j, -5.0 + 0.5j):
        ref = complex(2 / (mp.pi * z) * mp.ellipk((4 * t / z) ** 2))
        assert abs(X.lattice_greens("tb_square", z, t=t) - ref) < 1e-9


def test_bethe_and_hypercubic_bands_against_quadrature():
    t = 1.3
    z = 0.4 + 0.3j
    th = (np.arange(20000) + 0.5) * math.pi / 20000
    e = 2 * t * np.cos(th)                                    # semicircle: rho = sqrt(4t^2-e^2)/(2 pi t^2)
    bethe = np.sum(np.sqrt(4 * t * t - e * e) / (2 * math.pi * t * t) / (z - e) * 2 * t * np.sin(th)) * math.pi / 20000
    assert abs(X.lattice_greens("tb_bethe", z, t=t) - bethe) < 1e-7
    x = np.linspace(-12 * t, 12 * t, 400001)
    gau = np.trapezoid(np.exp(-x * x / (2 * t * t)) / (t * math.sqrt(2 * math.pi)) / (z - x), x)
    assert abs(X.lattice_greens("tb_hypercubic", z, t=t) - gau) < 1e-8


def test_derived_models_reduce_to_their_limits():
    z = np.array([0.3 + 0.2j, -1.4 + 0.5j, 2.2 + 0.1j])
    bp = {"t": 1.0}
    g0 = X.lattice_greens("tb_bethe", z, **bp)
    # BCS with zero gap is the normal band; resonant level with V = 0 is a free level
    assert np.allclose(X.lattice_greens("bcs", z, band="tb_bethe", band_params=bp, gap=1e-9), g0, atol=1e-8)
    assert np.allclose(X.lattice_greens("resonant_level", z, bath="tb_bethe", bath_params=bp,
                                        e_d=0.4, V=0.0), 1 / (z - 0.4))
    # Falicov-Kimball at U = 0 is the Bethe lattice, and it solves its cubic
    assert np.allclose(X.lattice_greens("falicov_kimball", z, t=1.0, U=0.0, w1=0.5), g0, atol=1e-10)
    t, U, w1 = 1.0, 3.0, 0.4
    G = X.lattice_greens("falicov_kimball", z, t=t, U=U, w1=w1)
    rhs = (1 - w1) / (z - t * t * G) + w1 / (z - U - t * t * G)
    assert np.allclose(G, rhs, atol=1e-11) and (G.imag < 0).all()


def test_bcs_gap_and_fk_gap_are_where_theory_puts_them():
    w = np.linspace(-4, 4, 1601)
    A = -X.lattice_greens("bcs", w + 0.01j, band="tb_bethe", band_params={"t": 1.0}, gap=0.5).imag / math.pi
    assert A[np.abs(w) < 0.3].max() < 0.02 and A[(np.abs(w) > 0.5) & (np.abs(w) < 0.7)].max() > 0.5
    # the Falicov-Kimball gap on the Bethe lattice opens at U = 2t (half filling)
    mid = lambda U: -X.lattice_greens("falicov_kimball", np.array([U / 2 + 0.005j]), t=1.0, U=U, w1=0.5).imag[0] / math.pi
    assert mid(1.0) > 0.1 and mid(3.5) < 0.01


@pytest.mark.parametrize("family", X.EXACT_CONTINUUM)
def test_continuum_truths_are_normalised_positive_and_self_consistent(family):
    rng = torch.Generator().manual_seed(3)
    for beta in (2.0, 8.0):
        tr = X.sample_exact_truth(family, beta, rng)
        eta, L = tr.params["eta"], tr.support_bound()
        assert eta >= 1.05 * X.GAMMA_MIN / beta - 1e-12
        w = np.linspace(-L - 60.0, L + 60.0, 6001)
        A = tr.A(w)
        assert A.min() > -1e-12
        # a Lorentzian-broadened spectrum keeps weight ~ 2 eta / (pi R) beyond +-R
        assert abs(np.trapezoid(A, w) - 1.0) < 2 * eta / (math.pi * 60.0) + 5e-3
        # the transform behaves as 1/z, and on the Matsubara axis it is the integral of A
        assert abs(1j * 1e5 * tr.stieltjes(np.array([1e5j]))[0] - 1.0) < 1e-3
        om, F = tr.matsubara(4)
        num = np.trapezoid(A[None, :] / (1j * om[:, None] - w[None, :]), w, axis=1)
        tail = 2 * eta / (math.pi * 60.0) / om            # what the window leaves out, at most
        assert (np.abs(num - F) < tail + 2e-2 * np.abs(F)).all()


def test_mollified_spectrum_is_the_gaussian_convolution():
    tr = X.sample_exact_truth("tb_square", 8.0, torch.Generator().manual_seed(0))
    w = np.linspace(-3, 3, 7)
    sig = math.pi / 8.0
    u = np.linspace(-12, 12, 24001)                   # the Gaussian is 1e-100 beyond this
    Au = tr.A(u)
    ref = [np.trapezoid(Au * np.exp(-0.5 * ((x - u) / sig) ** 2) / (sig * math.sqrt(2 * math.pi)), u)
           for x in w]
    assert np.allclose(tr.mollified_A(w, sig), ref, atol=2e-6)


def test_hubbard_atom_matches_the_textbook_two_pole_spectrum():
    U, mu, beta = 3.0, 1.1, 2.5
    s = X.hubbard_cluster(np.zeros((1, 1)), U, mu, beta)
    Z = 1 + 2 * math.exp(beta * mu) + math.exp(beta * (2 * mu - U))
    n_dn = (math.exp(beta * mu) + math.exp(beta * (2 * mu - U))) / Z
    assert np.allclose(s.energies, [-mu, U - mu], atol=1e-10)
    assert np.allclose(s.weights, [1 - n_dn, n_dn], atol=1e-10)
    # G(0) + G(beta) = 1 and G(beta) = <n>
    G = s.greens(np.array([0.0, 1.0]))
    assert abs(G.sum() - 1.0) < 1e-12 and abs(G[1] - n_dn) < 1e-10


def test_noninteracting_clusters_give_the_single_particle_levels():
    t = 0.9
    d = X.hubbard_cluster(np.array([[0, -t], [-t, 0.0]]), 0.0, 0.0, 6.0)
    assert np.allclose(d.energies, [-t, t], atol=1e-9) and np.allclose(d.weights, [0.5, 0.5], atol=1e-9)
    hop = np.zeros((4, 4))
    for i in range(4):
        hop[i, (i + 1) % 4] = hop[(i + 1) % 4, i] = -t
    r = X.hubbard_cluster(hop, 0.0, 0.3, 4.0)
    assert np.allclose(r.energies, [-2 * t - 0.3, -0.3, 2 * t - 0.3], atol=1e-9)
    assert np.allclose(r.weights, [0.25, 0.5, 0.25], atol=1e-9)


def test_half_filled_hubbard_dimer_is_particle_hole_symmetric_with_known_poles():
    t, U = 1.0, 4.0
    s = X.hubbard_cluster(np.array([[0, -t], [-t, 0.0]]), U, U / 2, 40.0)       # T -> 0
    c = math.sqrt(U * U + 16 * t * t)
    want = sorted([t + 0.5 * (c - U) + 0, -(t + 0.5 * (c - U))] + [0.5 * (c + U) - t + 0, -(0.5 * (c + U) - t)])
    # zero-temperature poles of the half-filled dimer: +-(c - U)/2 -+ ... reduce to +-(c/2 -+ t) -+ U/2 ...
    poles = sorted(s.energies[s.weights > 1e-6])
    assert len(poles) == 4 and np.allclose(poles, -np.array(poles)[::-1], atol=1e-8)
    assert np.allclose(sorted(abs(p) for p in poles)[::2], [0.5 * c - t, 0.5 * c + t], atol=1e-8)
    assert abs(s.weights.sum() - 1) < 1e-12 and abs((s.weights * s.energies).sum()) < 1e-8
    del want


def test_holstein_atom_is_the_independent_boson_ladder():
    om, g, mu, U = 1.0, 1.2, -2.0, 20.0
    lam = (g / om) ** 2
    # mu below the polaron level and a large U: the T -> 0 ground state is the empty site
    # with no phonons, so electron addition is the textbook independent-boson spectrum --
    # lines at (-lam*om - mu) + n*om with Poisson weights exp(-lam) lam^n / n!
    s = X.holstein_atom(om, g, U, mu, beta=60.0, n_boson=60)
    lines = [(e, w) for e, w in zip(s.energies, s.weights) if w > 1e-9]
    e0 = min(e for e, _ in lines)
    assert abs(e0 - (-lam * om - mu)) < 1e-8
    for n in range(6):
        w = sum(w for e, w in lines if abs(e - (e0 + n * om)) < 1e-6)
        assert abs(w - math.exp(-lam) * lam ** n / math.factorial(n)) < 1e-8
    # truncation converged: more phonons change nothing
    s2 = X.holstein_atom(om, g, U, mu, beta=60.0, n_boson=80)
    x = np.linspace(0, 1, 9)
    assert abs(s2.greens(x) - s.greens(x)).max() < 1e-12


@pytest.mark.parametrize("family", X.EXACT_DISCRETE)
def test_discrete_samples_obey_the_sum_rule_and_the_greens_function_identity(family):
    rng = torch.Generator().manual_seed(11)
    for beta in (2.0, 8.0):
        s = X.sample_discrete_spectrum(family, beta, rng)
        assert abs(s.weights.sum() - 1.0) < 1e-10 and (s.weights > 0).all()
        G = s.greens(np.array([0.0, 0.5, 1.0]))
        assert abs(G[0] + G[2] - 1.0) < 1e-10 and (G > 0).all()
        eps, gam, a, kept = s.poles(0.05, 12)
        assert len(eps) <= 12 and abs(a.sum() - 1) < 1e-12 and 0 < kept <= 1 + 1e-12


def test_exact_pole_truths_are_valid_pole_sets_inside_the_band_box():
    rng = torch.Generator().manual_seed(5)
    seen = set()
    for _ in range(40):
        eps, gam, a, b, fam = X.sample_exact_poles(8.0, rng)
        seen.add(fam)
        assert 1 <= len(eps) <= 12 and abs(sum(a) - 1) < 1e-9 and all(v == 0.0 for v in b)
        assert min(gam) >= 1.05 * X.GAMMA_MIN / 8.0 - 1e-12 and max(abs(e) for e in eps) < 20.0
    assert len(seen) >= 4


def test_line_spectra_go_through_the_canonical_label_protocol():
    from spevae.data.label_bank import extract_smooth_label
    sp = X.holstein_atom(omega0=0.8, g=1.1, U=0.0, mu=0.4, beta=8.0)
    # the protocol's inputs, each against its definition
    om, F = sp.matsubara(8)
    w = sp.weights / sp.weights.sum()
    assert np.allclose(F, [(w / (1j * o - sp.energies)).sum() for o in om])
    grid = np.linspace(-40.0, 40.0, 16001)
    assert abs(np.trapezoid(sp.mollified_A(grid, 0.3), grid) - 1.0) < 1e-6
    assert sp.support_bound() >= np.abs(sp.energies[w > 1e-6]).max()
    # a dense ladder of lines is representable: the label reproduces the exact G(tau)
    lab = extract_smooth_label(sp, P_max=12, residue_mode="dispersive")
    from spevae.physics import PoleGreens, PoleSet, SpectralQuadrature
    ps = PoleSet(*(torch.tensor(np.asarray(v), dtype=torch.float64).unsqueeze(0)
                   for v in (lab.eps * 8.0, lab.gamma * 8.0, lab.a, lab.b)))
    x = np.linspace(0.0, 1.0, 65)
    G = PoleGreens(SpectralQuadrature.for_beta(8.0, dtype=torch.float64))(ps, torch.tensor(x))[0].numpy()
    assert np.abs(G - sp.greens(x)).max() < 1e-4
