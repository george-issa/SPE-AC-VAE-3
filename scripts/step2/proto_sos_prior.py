"""Prototype: SOS residue sampler vs the current priors, matched positions.

A nonneg univariate polynomial is a sum of two squares, so given pole
locations z_p the WHOLE positive set of spectra A = P/(pi Q) at those
locations is parameterised by P = q1^2 + q2^2 (deg q <= n-1):
positivity exact by construction (zero certificate calls, zero shrink
bias), sum(b) = 0 automatic (deg P <= 2n-2), sum(a) = 1 by one rescale.
Residues: r_p = -2i P(z_p)/Q'(z_p) (verified analytically for n = 1 and
numerically below).  The one reachability filter: partial fractions can
give a_p < 0 (positive total does not imply positive per-pole weights),
and those truths are softmax-unreachable -> rejected; the rejection rate
is a headline measurement.

Compares three samplers on IDENTICAL position/width draws (same seed;
residue modes consume no extra torch randomness): the step-1 map
("complex"), the M1 rescaled-t dispersive prior, and SOS.  Metrics:
|b|/a reach, negative weight, certify/shrink activity, induced weight
structure (effective pole count), speed.  Artifacts:
out/step2/sos_prior_proto.{json,png}.
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import spevae.data.pole_prior as pp
from spevae.physics.positivity import certify

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9,
                     "axes.labelsize": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5})

OUT = Path("out/step2")
BETA = 8.0
N_TRUTHS = 300
SEED = 3


def sos_residues(eps, gam, rng, tries=40):
    """Sample reachable (a, b) uniformly-in-q over the positive set at
    fixed poles.  Returns (a, b, tries_used) or None if every try had a
    softmax-unreachable negative weight."""
    eps = np.asarray(eps, float)
    gam = np.asarray(gam, float)
    n = eps.size
    c = eps.mean()
    s = max((eps.max() - eps.min()) / 2.0 + gam.max(), 1.0)
    u_lo = (eps - c) / s - 1j * gam / s          # lower-half poles, u-space
    roots = np.concatenate([u_lo, u_lo.conj()])
    Qp = np.empty(n, complex)
    for p in range(n):
        others = np.delete(roots, p)
        Qp[p] = np.prod(u_lo[p] - others)
    for t in range(tries):
        q1 = rng.standard_normal(n)              # deg n-1 coefficients
        q2 = rng.standard_normal(n)
        Pz = np.polyval(q1, u_lo) ** 2 + np.polyval(q2, u_lo) ** 2
        r = -2j * Pz / Qp
        a, b = r.real, r.imag
        if (a > 0).all():
            S = a.sum()
            return a / S, b / S, t + 1
    return None


def self_test(rng):
    """Verify the residue formula: A from the pole set == P/(pi Q) >= 0."""
    eps = np.array([-1.2, 0.3, 1.7])
    gam = np.array([0.5, 0.35, 0.8])
    out = sos_residues(eps, gam, rng)
    a, b, _ = out
    w = np.linspace(-30, 30, 20001)
    z = eps - 1j * gam
    A = (-1 / math.pi) * ((a + 1j * b)[None, :] / (w[:, None] - z[None, :])
                          ).sum(1).imag
    assert A.min() > -1e-13, A.min()
    assert abs(np.trapezoid(A, w) - 1.0) < 2e-2   # 1/w^2 tail truncation
    assert abs(b.sum()) < 1e-12
    print(f"self-test ok: min A = {A.min():.1e}, sum b = {b.sum():.1e}")


def collect(mode, rng_np=None):
    """Draw N_TRUTHS via sample_poles; instrument certify; return metrics."""
    gen = pp.PolePriorGenerator(beta_range=(BETA, BETA), P_max=12,
                                residue_mode=("complex" if mode == "sos"
                                              else mode))
    gen.seed_scalars(SEED)
    calls = {"n": 0}
    orig = pp.certify

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    pp.certify = counting
    rows, examples = [], []
    t0 = time.time()
    try:
        for i in range(N_TRUTHS):
            e, g, a, b = gen.sample_poles(BETA)
            before = calls["n"]
            if mode == "sos":
                out = sos_residues(e, g, rng_np)
                if out is None:
                    rows.append(None)
                    continue
                a, b, tries = out
                a, b = a.tolist(), b.tolist()
            else:
                tries = None
            rep = certify(e, g, list(a), list(b), atol=1e-9)
            negw = 0.0 if rep.is_positive else float(rep.negative_weight)
            an = np.array(a)
            bn = np.array(b)
            rows.append({
                "n": len(e),
                "max_boa": float((np.abs(bn) / np.maximum(an, 1e-12)).max()),
                "max_b": float(np.abs(bn).max()),
                "negw": negw,
                "eff_n": float(1.0 / (an ** 2).sum()),
                "min_a": float(an.min()),
                "sos_tries": tries,
                "certify_calls": calls["n"] - before,  # inside sample_poles: 0 here
            })
            if len(examples) < 4:
                examples.append((np.array(e), np.array(g), an, bn))
    finally:
        pp.certify = orig
    wall = time.time() - t0
    ok = [r for r in rows if r is not None]
    return ok, len(rows) - len(ok), calls["n"], wall, examples


def q(v, p):
    return float(np.quantile(v, p))


def main():
    rng_np = np.random.default_rng(SEED)
    self_test(rng_np)
    report = {}
    all_ex = {}
    for mode in ("complex", "dispersive", "sos"):
        ok, rejected, ncert, wall, ex = collect(mode, rng_np)
        boa = [r["max_boa"] for r in ok]
        report[mode] = {
            "n_ok": len(ok), "rejected": rejected,
            "certify_calls_total": ncert,
            "wall_s": round(wall, 2),
            "max_boa_med": q(boa, 0.5), "max_boa_p95": q(boa, 0.95),
            "frac_b_gt_a": float(np.mean([v > 1 for v in boa])),
            "negw_max": max(r["negw"] for r in ok),
            "frac_dipping": float(np.mean([r["negw"] > 1e-9 for r in ok])),
            "eff_n_med": q([r["eff_n"] for r in ok], 0.5),
            "min_a_med": q([r["min_a"] for r in ok], 0.5),
        }
        if mode == "sos":
            tries = [r["sos_tries"] for r in ok]
            report[mode]["sos_tries_med"] = q(tries, 0.5)
            report[mode]["sos_tries_p95"] = q(tries, 0.95)
        all_ex[mode] = ex
        r = report[mode]
        print(f"{mode:10s}: ok={r['n_ok']:3d} rej={rejected:3d} "
              f"certify={ncert:4d} boa med/p95={r['max_boa_med']:.2f}/"
              f"{r['max_boa_p95']:.2f} frac|b|>a={r['frac_b_gt_a']:.0%} "
              f"negw_max={r['negw_max']:.1e} dip%={r['frac_dipping']:.0%} "
              f"effn={r['eff_n_med']:.1f} ({r['wall_s']}s)")

    with open(OUT / "sos_prior_proto.json", "w") as fh:
        json.dump(report, fh, indent=1)

    # figure: example spectra per sampler + metric histograms
    fig = plt.figure(figsize=(12.5, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 4)
    colors = {"complex": "#4269d0", "dispersive": "#efb118", "sos": "#3ca951"}
    w = np.linspace(-12, 12, 3000)
    for i, mode in enumerate(("complex", "dispersive", "sos")):
        for j in range(3):
            ax = fig.add_subplot(gs[i, j])
            e, g, a, b = all_ex[mode][j]
            z = e - 1j * g
            A = (-1 / math.pi) * ((a + 1j * b)[None, :]
                                  / (w[:, None] - z[None, :])).sum(1).imag
            ax.plot(w, A, color=colors[mode], lw=1.2)
            ax.axhline(0, color="0.5", lw=0.6)
            boa = float((np.abs(b) / np.maximum(a, 1e-12)).max())
            ax.set_title(f"{mode}  n={e.size}  max|b|/a={boa:.2f}", fontsize=8)
            if i == 2:
                ax.set_xlabel(r"$\omega$")
    # histograms on a fresh matched stream (numpy rng reseeded for SOS)
    fig2_data = {}
    for mode in ("complex", "dispersive", "sos"):
        ok, _, _, _, _ = collect(mode, np.random.default_rng(SEED + 1))
        fig2_data[mode] = ok
    ax = fig.add_subplot(gs[0, 3])
    for mode, ok in fig2_data.items():
        ax.hist([r["max_boa"] for r in ok], bins=np.linspace(0, 6, 25),
                histtype="step", lw=1.3, color=colors[mode], label=mode)
    ax.axvline(1.0, color="0.3", ls=":", lw=1)
    ax.set_title("max |b|/a per truth")
    ax.legend(frameon=False, fontsize=7)
    ax = fig.add_subplot(gs[1, 3])
    for mode, ok in fig2_data.items():
        v = np.log10(np.maximum([r["negw"] for r in ok], 1e-12))
        ax.hist(v, bins=24, histtype="step", lw=1.3, color=colors[mode])
    ax.axvline(math.log10(5e-3), color="0.3", ls=":", lw=1)
    ax.set_title("log10 negative weight (dip bound dotted)")
    ax = fig.add_subplot(gs[2, 3])
    for mode, ok in fig2_data.items():
        ax.hist([r["eff_n"] for r in ok], bins=np.linspace(1, 12, 23),
                histtype="step", lw=1.3, color=colors[mode])
    ax.set_title("effective pole count 1/sum a^2")
    fig.suptitle("SOS residue sampler vs current priors "
                 "(matched position/width draws, beta = 8)", fontsize=10)
    fig.savefig(OUT / "sos_prior_proto.png", dpi=150)
    print(f"-> {OUT}/sos_prior_proto.png, .json")


if __name__ == "__main__":
    main()
