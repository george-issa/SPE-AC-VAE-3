"""The versioned DQMC run with a known answer: the half-filled Hubbard dimer."""

import math

import numpy as np

from spevae.evaluation import cases as C


def test_dimer_fixture_agrees_with_the_exact_greens_function():
    mg, exact = C.hubbard_dimer_cell()
    assert mg.n_bins == 400 and mg.n_tau == 161 and mg.beta == 8.0
    assert mg.check_sum_rule()["ok"]
    # exact spectrum: four lines, symmetric, weights summing to one
    big = exact.weights > 1e-3
    assert big.sum() == 4 and abs(exact.weights.sum() - 1) < 1e-12
    e = np.sort(exact.energies[big])
    assert np.allclose(e, -e[::-1]) and np.allclose(e[2:], [1.82842712, 3.82842712], atol=1e-7)
    # the Monte Carlo mean agrees with the exact G(tau) at the level of its error bars
    G = exact.greens(mg.x.double().numpy())
    sem = (mg.sigma / math.sqrt(mg.n_bins)).double().numpy()
    z = ((mg.mean.double().numpy() - G) / np.maximum(sem, 1e-300))[1:-1]   # endpoints are pinned
    assert np.sqrt((z ** 2).mean()) < 1.5 and np.abs(z).max() < 4.0
    assert np.abs(mg.mean.double().numpy() - G).max() < 1e-2
