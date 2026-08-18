"""Tests for the feature-level spectral metrics."""

import numpy as np

from spevae.metrics import find_peaks, gap_metrics, matched_peak_error, spectral_report

W = np.linspace(-8, 8, 3201)


def gaussians(centres, heights, widths):
    A = np.zeros_like(W)
    for c, h, s in zip(centres, heights, widths):
        A += h * np.exp(-0.5 * ((W - c) / s) ** 2)
    return A


def test_find_peaks_locates_centres_and_heights():
    A = gaussians([-2.0, 1.5], [1.0, 0.6], [0.3, 0.4])
    peaks = find_peaks(A, W)
    assert len(peaks) == 2
    found = sorted(p.omega for p in peaks)
    assert abs(found[0] + 2.0) < 0.02 and abs(found[1] - 1.5) < 0.02
    assert abs(peaks[0].height - 1.0) < 1e-3


def test_find_peaks_rejects_a_shoulder_but_keeps_a_satellite():
    """Prominence, not height: a bump on a big peak is not a separate feature."""
    shoulder = gaussians([0.0, 0.45], [1.0, 0.35], [0.5, 0.25])
    satellite = gaussians([0.0, 4.0], [1.0, 0.12], [0.5, 0.25])
    assert len(find_peaks(shoulder, W)) == 1
    assert len(find_peaks(satellite, W)) == 2


def test_matched_peak_error_is_zero_for_identical_spectra():
    A = gaussians([-1.0, 2.0], [1.0, 0.5], [0.3, 0.3])
    m = matched_peak_error(A, A, W)
    assert m["n_matched"] == 2 and m["missed"] == 0 and m["spurious"] == 0
    assert m["d_omega"] < 1e-9 and m["rel_d_height"] < 1e-9


def test_matched_peak_error_measures_displacement_not_order():
    """Peaks are a set: relabelling must not change the score."""
    true = gaussians([-2.0, 2.0], [1.0, 0.8], [0.3, 0.3])
    pred = gaussians([2.3, -1.8], [1.0, 0.8], [0.3, 0.3])   # emitted in the other order
    m = matched_peak_error(pred, true, W)
    assert m["n_matched"] == 2
    assert 0.2 < m["d_omega"] < 0.3


def test_missed_and_spurious_peaks_are_counted_separately():
    true = gaussians([-2.0, 0.0, 2.0], [1.0, 0.9, 1.0], [0.25, 0.25, 0.25])
    pred = gaussians([-2.0, 2.0], [1.0, 1.0], [0.25, 0.25])
    m = matched_peak_error(pred, true, W)
    assert m["missed"] == 1 and m["spurious"] == 0
    m2 = matched_peak_error(true, pred, W)
    assert m2["spurious"] == 1 and m2["missed"] == 0


def test_gap_metrics_detects_and_sizes_a_gap():
    gapped = gaussians([-2.0, 2.0], [1.0, 1.0], [0.35, 0.35])
    g = gap_metrics(gapped, W)
    assert g["gapped"]
    assert 2.0 < g["gap"] < 3.5
    assert g["edge_lower"] < 0 < g["edge_upper"]


def test_gap_metrics_reports_no_gap_for_a_metal():
    metal = gaussians([0.0], [1.0], [0.8])
    assert not gap_metrics(metal, W)["gapped"]
    assert gap_metrics(metal, W)["rel_A_at_zero"] > 0.9


def test_spectral_report_covers_the_reportable_quantities():
    true = gaussians([-2.0, 2.0], [1.0, 1.0], [0.35, 0.35])
    true /= np.trapezoid(true, W)
    pred = gaussians([-2.4, 1.7], [1.0, 1.1], [0.4, 0.3])
    pred /= np.trapezoid(pred, W)
    rep = spectral_report(pred, true, W)
    for key in ("l2", "l2_rel", "linf", "w1", "min_A_pred",
                "peak_d_omega", "peak_rel_d_height", "gap_error"):
        assert key in rep and np.isfinite(rep[key])
    assert rep["min_A_pred"] >= 0.0
    assert abs(rep["norm_pred"] - 1.0) < 1e-6
    assert 0.2 < rep["peak_d_omega"] < 0.5
    assert rep["w1"] > 0.0
