"""Step-2 label bank: closed forms, the canonical extraction, the bank.

Load-bearing claims, each measured during build 1: the closed-form
Stieltjes transforms are exact; the constrained-ladder extraction is
deterministic, lives in the decoder's output space, and reproduces the
truth's G to noise-resolution; the rank ladder decays geometrically and
the supported rank grows as the tolerance tightens; a truth AT the
resolution floor gets an honestly-recorded floor-limited label instead of
a silently damaged one.
"""

import math

import numpy as np
import pytest
import torch

from spevae.data.label_bank import (
    DIP_TOL,
    EPS_MAX,
    GAM_MAX,
    GAMMA_MIN,
    SMOOTH_FAMILIES,
    LabelBank,
    SmoothTruth,
    _pack,
    _residual_jac,
    _residual_vec,
    build_label_bank,
    extract_smooth_label,
    sample_smooth_truth,
)
from spevae.physics.forward import PoleGreens, SpectralQuadrature

BETA = 8.0

#: One pinned instance per family (explicit params: no sampling, so every
#: number in this file is reproducible by hand).
PINNED = {
    "gauss_mixture": {"weights": [0.6, 0.4], "mus": [-1.2, 1.8],
                      "sigmas": [0.5, 0.9], "W": 4.0},
    "gapped_pair": {"weights": [0.45, 0.55], "mus": [-2.2, 2.2],
                    "sigmas": [0.4, 0.4], "W": 4.0},
    "semicircle": {"centre": 0.3, "radius": 4.0, "sigma_res": 0.05, "W": 4.0},
    "box": {"lo": -3.0, "hi": 4.0, "sigma_res": 0.05, "W": 4.0},
    "pseudogap": {"weights": [1.2062, -0.2062],
                  "mus": [0.1, 0.1], "sigmas": [2.0, 0.4851], "W": 4.0},
}


def pinned(family, beta=BETA):
    return SmoothTruth(family, beta, dict(PINNED[family]))


def truth_greens(truth, quad, x):
    """G from the truth's exact A through the standard forward map."""
    beta = truth.beta
    Ad = torch.tensor(truth.A(quad.nodes.numpy() / beta)) / beta
    K = PoleGreens(quad).kernel_matrix(x).double()
    return (Ad * quad.weights.double()) @ K


def test_stieltjes_closed_forms_match_brute_force():
    om = (2.0 * np.arange(8) + 1.0) * math.pi / BETA
    z = 1j * om
    w = np.linspace(-60.0, 60.0, 200_001)
    for fam in SMOOTH_FAMILIES:
        tr = pinned(fam)
        S = tr.stieltjes(z)
        Aw = tr.A(w)
        S_brute = np.trapezoid(Aw[None, :] / (z[:, None] - w[None, :]), w, axis=1)
        assert np.abs(S - S_brute).max() < 1e-8, fam
        # normalisation two ways: the A integral and the 1/z tail of S
        assert np.trapezoid(Aw, w) == pytest.approx(1.0, abs=1e-6)
        assert (200j * tr.stieltjes(np.array([200j]))).real[0] == \
            pytest.approx(1.0, abs=1e-3)


def test_sampler_floors_positivity_and_reconstruction():
    rng = torch.Generator().manual_seed(7)
    w = np.linspace(-40.0, 40.0, 20_001)
    floor = 1.05 * GAMMA_MIN / BETA
    for fam in SMOOTH_FAMILIES:
        tr = sample_smooth_truth(fam, BETA, rng)
        assert tr.A(w).min() > -1e-12, fam          # positive truths
        widths = tr.params.get("sigmas", [tr.params.get("sigma_res", 1.0)])
        assert min(abs(s) for s in widths) >= 0.999 * min(floor, GAMMA_MIN / BETA)
        # the bank stores (family, beta, params) only; reconstruction is exact
        tr2 = SmoothTruth(tr.family, tr.beta, tr.params)
        probe = np.array([-1.3, 0.0, 2.7])
        assert np.array_equal(tr.A(probe), tr2.A(probe))


def test_polish_jacobian_matches_finite_differences():
    tr = pinned("gauss_mixture")
    om, F = tr.matsubara()
    Fs = np.abs(F).max()
    gfloor = 1.05 * GAMMA_MIN / BETA
    n = 4
    cfg = (np.array([-2.0, -0.5, 1.0, 3.0]), np.array([0.5, 0.3, 0.8, 0.4]),
           np.array([0.3, 0.2, 0.4, 0.1]), np.array([0.05, -0.02, -0.06, 0.03]))
    for cap in (0.5, 0.25, 0.0):
        th = _pack(*cfg, gfloor, cap)
        J = _residual_jac(th, n, om, F, Fs, gfloor, cap)
        h = 1e-7
        for k in range(th.size):
            tp, tm = th.copy(), th.copy()
            tp[k] += h
            tm[k] -= h
            col = (_residual_vec(tp, n, om, F, Fs, gfloor, cap)
                   - _residual_vec(tm, n, om, F, Fs, gfloor, cap)) / (2 * h)
            assert np.abs(J[:, k] - col).max() < 1e-6, (cap, k)


def test_labels_reproduce_truth_to_noise_resolution():
    """The step-2 build-order gate: the label's G matches the truth's G
    below the quietest training noise (3e-5), through the same forward map
    training uses.  Pointwise A is NOT the metric -- at finite beta the
    kernel has a null space and the label may legitimately differ there."""
    quad = SpectralQuadrature(nu_c=BETA * 20.0, h=GAMMA_MIN / 3.5, n_tail=32,
                              dtype=torch.float64)
    x = torch.linspace(0.0, 1.0, 65, dtype=torch.float64)
    for fam in SMOOTH_FAMILIES:
        tr = pinned(fam)
        lab = extract_smooth_label(tr)
        assert lab.resid < 1e-5, fam
        G_t = truth_greens(tr, quad, x)
        ps = lab.poleset().to(torch.float64).to_dimensionless(BETA)
        G_l = PoleGreens(quad)(ps, x)[0]
        assert float((G_l - G_t).abs().max()) < 3e-5, fam


def test_floor_limited_truth_is_recorded_honestly():
    """Lobes AT the resolution floor at beta=2: the output space cannot
    follow, the certificate forces the asymmetry cap down, and the label
    ships with the floor-limited residual on record instead of a silent
    quality collapse (the unbounded-shrink failure measured 5.7e-2)."""
    beta = 2.0
    sig = 1.05 * GAMMA_MIN / beta
    tr = SmoothTruth("gapped_pair", beta,
                     {"weights": [0.4, 0.6], "mus": [-1.85, 1.85],
                      "sigmas": [sig, sig], "W": 1.8})
    lab = extract_smooth_label(tr)
    assert 1e-4 < lab.resid < 5e-2
    assert lab.neg_weight <= 1.1 * DIP_TOL
    quad = SpectralQuadrature(nu_c=beta * 20.0, h=GAMMA_MIN / 3.5, n_tail=32,
                              dtype=torch.float64)
    x = torch.linspace(0.0, 1.0, 41, dtype=torch.float64)
    G_t = truth_greens(tr, quad, x)
    G_l = PoleGreens(quad)(lab.poleset().to(torch.float64)
                           .to_dimensionless(beta), x)[0]
    assert float((G_l - G_t).abs().max()) < 2.0 * lab.resid


def test_ladder_decays_geometrically_and_rank_grows_with_snr():
    for fam in ("gauss_mixture", "semicircle"):
        lab = extract_smooth_label(pinned(fam))
        res = dict(lab.ladder)
        ranks = sorted(res)
        # strict decay until the ladder's own plateau (each truth has its
        # own floor -- Gaussian tails plateau near 1e-7 at beta=8, branch
        # cuts near 1e-13)
        floor = 100.0 * min(res.values())
        for lo, hi_ in zip(ranks, ranks[1:]):
            if res[lo] > max(floor, 1e-8):
                assert res[hi_] <= 1.001 * res[lo], (fam, lo)
        # geometric: >= 3x per rung on the way to 1e-6
        k = min(n for n in ranks if res[n] <= 1e-6)
        assert k <= 10, fam
        assert res[ranks[0]] / res[k] >= 3.0 ** (k - ranks[0]), fam
        # the SNR ladder: rank the data supports grows as noise shrinks
        n_hi, n_mid, n_lo = (lab.rank_at(t) for t in (3e-2, 3e-4, 1e-6))
        assert n_hi <= n_mid <= n_lo
        assert n_hi < n_lo


def test_extraction_deterministic_and_in_the_output_space():
    tr = pinned("pseudogap")
    lab1 = extract_smooth_label(tr)
    lab2 = extract_smooth_label(tr)
    for u, v in ((lab1.eps, lab2.eps), (lab1.gamma, lab2.gamma),
                 (lab1.a, lab2.a), (lab1.b, lab2.b)):
        assert np.array_equal(u, v)
    assert lab1.ladder == lab2.ladder
    # every package constraint, exactly
    assert lab1.n <= 12
    assert float(lab1.a.sum()) == pytest.approx(1.0, abs=1e-9)
    assert float(lab1.b.sum()) == pytest.approx(0.0, abs=1e-12)
    assert (np.abs(lab1.b) <= lab1.a * (1 + 1e-9)).all()
    assert (lab1.gamma * BETA >= 1.05 * GAMMA_MIN * (1 - 1e-9)).all()
    assert (np.abs(lab1.eps) <= EPS_MAX).all()
    assert (lab1.gamma <= GAM_MAX * (1 + 1e-9)).all()
    assert lab1.neg_weight <= 1.1 * DIP_TOL


@pytest.fixture(scope="module")
def small_bank():
    return build_label_bank(n_per_family=1, seed=5, beta_range=(6.0, 10.0))


def test_bank_layout_padding_and_conventions(small_bank):
    bank = small_bank
    assert len(bank) == len(SMOOTH_FAMILIES)
    assert bank.family == list(SMOOTH_FAMILIES)
    for i in range(len(bank)):
        m = int(bank.n_true[i])
        assert 1 <= m <= bank.P_max
        assert (bank.a[i, m:] == 0).all(), "padding must carry zero weight"
        assert (bank.gamma[i, m:] == 1.0).all(), "inert padding widths"
        assert float(bank.a[i, :m].sum()) == pytest.approx(1.0, rel=1e-5)
        ps = bank.label(i)
        assert ps.shape == (1, m)
    lab = bank.labels_dimensionless()
    i = 0
    m = int(bank.n_true[i])
    assert torch.allclose(lab["eps"][i, :m],
                          bank.eps[i, :m] * bank.beta[i])
    assert (lab["gamma"][i, m:] == 1.0).all()
    assert (lab["gamma"][i, :m] >= GAMMA_MIN * (1 - 1e-6)).all(), \
        "dimensionless width floor"


def test_bank_deterministic_and_roundtrips(small_bank, tmp_path):
    bank2 = build_label_bank(n_per_family=1, seed=5, beta_range=(6.0, 10.0))
    assert torch.equal(small_bank.eps, bank2.eps)
    assert torch.equal(small_bank.a, bank2.a)
    p = tmp_path / "bank.pt"
    small_bank.save(p)
    loaded = LabelBank.load(p)
    for k in ("beta", "eps", "gamma", "a", "b", "n_true", "resid"):
        assert torch.equal(getattr(loaded, k), getattr(small_bank, k)), k
    assert loaded.params == small_bank.params
    assert loaded.ladders == small_bank.ladders
    # the truth is reconstructible from the stored record alone
    tr = loaded.truth(2)
    assert tr.family == SMOOTH_FAMILIES[2]
    assert np.isfinite(tr.A(np.array([0.0]))).all()


def test_truncation_residual_pinned_per_family(small_bank):
    """The bank's first unit test, per the roadmap: the label-truncation
    residual per family.  Away from the resolution floor (beta in [6,10])
    every family labels below 1e-4, the roadmap's estimated bound; the
    measured build-1 values sit at 1e-7 and below."""
    r = small_bank.resid
    for i, fam in enumerate(small_bank.family):
        assert float(r[i]) < 1e-4, (fam, float(r[i]))
    assert float(r.median()) < 1e-6
