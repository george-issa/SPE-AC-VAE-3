"""The evaluation suite: frozen cases, the shared scorer, statistics, the runner."""

import math
import os

import numpy as np
import pytest
import torch

from spevae.models import SPEVAE
from spevae.evaluation import cases as C
from spevae.evaluation import methods as M
from spevae.evaluation.stats import compare_models, summarize
from spevae.physics import PoleGreens
from spevae.physics.poles import PoleSet


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return SPEVAE(latent_dim=4, num_poles=6, omega_c=20.0, d_model=16, enc_layers=1,
                  dec_layers=1, n_heads=2, d_ff=32).eval()


# ---------------------------------------------------------------- cases --- #
@pytest.mark.parametrize("name", list(C.CASES))
def test_every_case_realises_normalised(name):
    tr = C.realize(name)
    assert tr.A.shape == (1, tr.quad.nodes.numel())
    assert tr.G_clean.shape == (1, C.NTAU + 1)
    norm = float((tr.A * tr.quad.weights).sum())
    assert abs(norm - 1.0) < 2e-3, norm
    # fermionic sum rule G(0) + G(beta) = 1 on the endpoint grid
    assert abs(float(tr.G_clean[0, 0] + tr.G_clean[0, -1]) - 1.0) < 2e-3


def test_pole_cases_carry_their_count():
    for c in C.CASES.values():
        if isinstance(c.spec, C.PoleSpec):
            assert c.n_true == len(c.spec.eps)


def test_fano_band_matches_the_judge_construction():
    """The registry reproduces judge_map_screen.py's float path exactly."""
    beta, quad = 8.0, C.realize("fano band").quad
    ks = np.linspace(-1, 1, 5)
    eps = (0.3 + ks * 1.2).tolist()
    a, t = [0.2] * 5, np.clip(0.7 * ks, -0.45, 0.45).tolist()
    a_, t_ = np.array(a), np.array(t)
    b = a_ * (t_ - (a_ * t_).sum() / a_.sum())
    ps = PoleSet(eps=torch.tensor([eps]) * beta, gamma=torch.tensor([[0.5] * 5]) * beta,
                 a=torch.tensor([a]).double().float(), b=torch.tensor([b.tolist()]).float())
    x = torch.linspace(0.0, 1.0, 65)
    tr = C.realize("fano band")
    assert torch.equal(tr.A, ps.spectral(quad.nodes))
    assert torch.equal(tr.G_clean, PoleGreens(quad)(ps, x))


def test_noise_draws_are_deterministic_and_seeded():
    tr = C.realize("asym triplet")
    a, b, c = C.draw(tr, 1e-3, 7), C.draw(tr, 1e-3, 7), C.draw(tr, 1e-3, 8)
    assert torch.equal(a.G, b.G)
    assert not torch.equal(a.G, c.G)
    assert C.seeds_for(4) == [7, 8, 9, 10]


# --------------------------------------------------------------- scorer --- #
def test_truth_scores_zero_w1_and_noise_level_chi2():
    tr = C.realize("irregular triplet")
    inst = C.draw(tr, 1e-3, 7)
    rec, _ = M.score_synthetic(M.Output("oracle", poles=tr.poles), inst)
    assert rec["w1"] < 1e-6
    assert 0.3 < rec["chi2"] < 3.0          # one whitened noise draw, 65 points
    assert rec["neg_weight"] == 0.0
    assert rec["n_poles"] == 3


def test_grid_output_of_the_truth_scores_near_zero():
    name = "semicircle"
    tr = C.realize(name)
    w = np.linspace(-15, 15, 6001)
    out = M.Output("grid", omega=w, A=C.truth_on_grid(name, w))
    rec, _ = M.score_synthetic(out, C.draw(tr, 1e-4, 7))
    assert rec["w1"] < 5e-3


def test_scorer_chi2_equals_z_descent_internal(tiny_model):
    inst = C.draw(C.realize("fano band"), 1e-3, 7)
    out = M.make_method("zd5", tiny_model).synthetic(inst)
    rec, _ = M.score_synthetic(out, inst)
    assert math.isclose(rec["chi2"], rec["chi2_internal"], rel_tol=1e-3)


def test_amortized_is_deterministic(tiny_model):
    inst = C.draw(C.realize("lorentz pair"), 1e-4, 9)
    r1, _ = M.score_synthetic(M.make_method("amortized", tiny_model).synthetic(inst), inst)
    r2, _ = M.score_synthetic(M.make_method("amortized", tiny_model).synthetic(inst), inst)
    assert r1["w1"] == r2["w1"]


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        M.make_method("magic")


# ---------------------------------------------------------------- stats --- #
def _recs(model, scale, seeds=range(7, 15), runs=1):
    out = []
    for k in range(runs):
        tag = f"{model}#{k}" if runs > 1 else model
        for s in seeds:
            out.append({"kind": "synthetic", "model": tag, "method": "amortized",
                        "case": "fano band", "noise": 1e-3, "seed": s,
                        "w1": scale * (1.0 + 0.01 * s + 0.1 * k)})
    return out


def test_summary_interval_brackets_the_mean():
    rows = summarize(_recs("A", 1.0))
    w = rows[0]["w1"]
    assert w["lo"] <= w["mean"] <= w["hi"]
    assert w["n"] == 8 and w["runs"] == 1
    assert w["seed7"] == pytest.approx(1.07)


def test_training_runs_widen_the_interval():
    one = summarize(_recs("A", 1.0, runs=1))[0]["w1"]
    three = summarize(_recs("A", 1.0, runs=3))[0]["w1"]
    assert three["runs"] == 3 and len(three["run_means"]) == 3
    assert (three["hi"] - three["lo"]) > (one["hi"] - one["lo"])


def test_paired_comparison_recovers_a_pure_scale():
    rows = compare_models(_recs("A", 1.0) + _recs("B", 0.5), "A", "B", "amortized")
    assert rows[0]["ratio"] == pytest.approx(0.5)
    assert rows[0]["b_better"] == 1.0


# --------------------------------------------------------------- runner --- #
def test_holstein_fixture_matches_pin():
    path = os.path.join(os.path.dirname(C.__file__), "fixtures", C.HOLSTEIN_FIXTURE["file"])
    if not os.path.exists(path):
        pytest.skip("Holstein fixture not built (scripts/eval/build_holstein_fixture.py)")
    mg = C.holstein_cell()
    assert mg.n_bins == 400 and mg.n_tau == 81
    assert mg.covariance(1e-10, method="structured")["chi2_target"] == pytest.approx(0.17529,
                                                                                      abs=1e-5)


def test_run_suite_end_to_end(tmp_path, tiny_model):
    from spevae.evaluation import run_suite, write_report
    from spevae.io import save_model
    ck = str(tmp_path / "tiny.pt")
    save_model(tiny_model, ck)
    card = run_suite([("tiny", ck), ("tiny", ck)], tier="fast", workers=1,
                     cases_sel=["fano band"], skip=("zd800", "zd3000", "canonical", "maxent",
                                                    "deac_shipped", "streams", "real"),
                     progress=lambda *_: None)
    assert not card["failures"], card["failures"]
    syn = [r for r in card["records"] if r["kind"] == "synthetic"]
    assert {r["model"] for r in syn} == {"tiny#0", "tiny#1"}
    assert len(syn) == 2 * 2 * 8              # 2 runs x 2 noises x 8 seeds
    write_report(card, str(tmp_path / "rep"), figures=False)
    assert (tmp_path / "rep" / "scorecard.json").exists()
    assert (tmp_path / "rep" / "summary.md").exists()


def test_paired_interval_does_not_flag_two_agreeing_pairs():
    """Two pairs that agree in sign are not evidence; eight consistent ones are."""
    from spevae.evaluation.stats import compare_pairs

    def recs(ratios):
        out = []
        for s, r in enumerate(ratios):
            for g, v in (("A", 1.0), ("B", r)):
                out.append({"kind": "synthetic", "model": g, "method": "m", "case": "c",
                            "noise": 1e-3, "seed": s, "w1": v})
        return out

    c = compare_pairs(recs([0.9, 0.7]), ("A", "m"), ("B", "m"))[0]
    assert c["n"] == 2 and c["lo"] < 1.0 < c["hi"] and abs(c["ratio"] - (0.9 * 0.7) ** 0.5) < 1e-12
    c = compare_pairs(recs([0.8, 0.75, 0.85, 0.7, 0.9, 0.8, 0.78, 0.82]), ("A", "m"), ("B", "m"))[0]
    assert c["hi"] < 1.0
    c = compare_pairs(recs([0.8]), ("A", "m"), ("B", "m"))[0]
    assert c["lo"] == 0.0 and c["hi"] == float("inf")
