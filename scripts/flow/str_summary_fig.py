"""Summary figures for the sample-then-refine (STR) judgement of a flow.

Reads the final.json files written by `judge_str.py final` (no reruns) and
renders two figures next to the redraw result:

  summary_full.png   full training distribution (SBC set): calibration KS,
                     interval / band coverage, sample chi2, posterior-mean W1,
                     and the Holstein chi2/target spread
  summary_suite.png  the 14 suite truths at noise 1e-3 and 1e-4: posterior-mean
                     W1 against the VAE's amortized W1, and 90% band coverage

    python scripts/flow/str_summary_fig.py out/flow/str_F60k_redraw out/flow/str_F60k
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, GREEN, AMBER = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1f1e1c", "#52514e", "#d9d8d4"


def style(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)


def fig_full(red, nored, path):
    # rows top to bottom: the headline method first
    rows = [("refined + redraw", red["sbc"]["str"], GREEN),
            ("refined, no redraw", nored["sbc"]["str"], AMBER),
            ("raw flow", red["sbc"]["raw"], BLUE),
            ("VAE (ctrl-60k)", red["sbc"]["vae"], ORANGE)]
    names = [r[0] for r in rows]
    y = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.7), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1.5, 1, 1, 1]})

    def dots(ax, key, ref=None, ref_lbl=None, fmt="{:.2f}"):
        for yi, (_, m, c) in zip(y, rows):
            ax.plot([0, m[key]], [yi, yi], color=c, lw=2, solid_capstyle="round")
            ax.plot(m[key], yi, "o", color=c, ms=8, mec="white", mew=1.2, zorder=3)
            ax.annotate(fmt.format(m[key]), (m[key], yi), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=7.5, color=INK)
        if ref is not None:
            ax.axvline(ref, color=MUTED, ls="--", lw=1)
            ax.annotate(ref_lbl, (ref, y[-1] - 0.55), xytext=(3, 0), textcoords="offset points",
                        fontsize=7.5, color=MUTED, va="center")
        ax.set_yticks(y, names if ax is axes[0] else [""] * len(y))
        ax.set_ylim(y[-1] - 0.8, y[0] + 0.6)
        style(ax)

    ks_crit = rows[0][1]["ks_crit_5pct"]
    dots(axes[0], "ks_max", ks_crit, f"5% critical {ks_crit:.3f}")
    axes[0].set_xlim(0, 0.32)
    axes[0].set_title("SBC calibration: max KS over 9 statistics\n(below the dashed line passes)",
                      fontsize=9, color=INK)

    ax = axes[1]
    groups = [("cover50", "50% interval", 0.5), ("cover90", "90% interval", 0.9),
              ("band90", "90% pointwise band", 0.9)]
    off = np.linspace(-0.27, 0.27, len(rows))
    for gi, (key, _, nom) in enumerate(groups):
        ax.plot([gi - 0.4, gi + 0.4], [nom, nom], color=MUTED, ls="--", lw=1)
        for k, (_, m, c) in enumerate(rows):
            ax.plot(gi + off[k], m[key], "o", color=c, ms=8, mec="white", mew=1.2, zorder=3)
    ax.set_xticks(range(len(groups)), [g[1] for g in groups])
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("empirical coverage", fontsize=8, color=INK)
    ax.set_title("Coverage of the truth (dashed: nominal)", fontsize=9, color=INK)
    style(ax, "y")
    for k, (n, _, c) in enumerate(rows):
        ax.plot([], [], "o", color=c, ms=7, label=n)
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    dots(axes[2], "chi2_median", 1.0, "truth = 1")
    axes[2].set_xlim(0, 1.75)
    axes[2].set_title("Sample fit: median $\\chi^2$ per point\n(the true spectrum scores ≈ 1)",
                      fontsize=9, color=INK)
    dots(axes[3], "w1_mean", fmt="{:.2f}")
    axes[3].set_xlim(0, 0.7)
    axes[3].set_title("Accuracy: W$_1$ of the posterior mean\n(lower is better)", fontsize=9,
                      color=INK)

    ax = axes[4]
    hol = [("refined + redraw", red, GREEN), ("refined, no redraw", nored, AMBER)]
    for yi, (n, r, c) in zip([1, 0], hol):
        lo, med, hi = r["holstein"]["chi2_over_target_q05_q50_q95"]
        ax.plot([lo, hi], [yi, yi], color=c, lw=2, solid_capstyle="round")
        ax.plot(med, yi, "o", color=c, ms=8, mec="white", mew=1.2, zorder=3)
        ax.annotate(f"median {med:.2f}", (med, yi), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=7.5, color=INK)
    ax.axvline(1.0, color=MUTED, ls="--", lw=1)
    ax.annotate("target", (1.0, -0.45), xytext=(3, 0), textcoords="offset points",
                fontsize=7.5, color=MUTED, va="center")
    ax.set_xscale("log")
    ax.set_xlim(0.25, 6)
    ax.set_xticks([0.3, 1, 3], ["0.3", "1", "3"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([1, 0], [h[0] for h in hol])
    ax.set_ylim(-0.6, 1.6)
    ax.set_title("Holstein: sample $\\chi^2/\\chi^2_{\\rm target}$\n(5–95% of 200 samples)",
                 fontsize=9, color=INK)
    style(ax)

    fig.suptitle("Sample-then-refine on the 60k flow, judged on 96 held-out draws of the training "
                 "distribution (32 samples each, mixed noise)", fontsize=10.5, color=INK)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_suite(red, path):
    cases = list(red["suite@0.001"])
    y = np.arange(len(cases))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for ax, nz in zip(axes[:2], ("0.001", "0.0001")):
        rows = red[f"suite@{nz}"]
        for yi, c in zip(y, cases):
            s, v = rows[c]["str_w1_mean"], rows[c]["vae_amortized_w1"]
            ax.plot([s, v], [yi, yi], color=GRID, lw=2, zorder=1)
            ax.plot(v, yi, "o", color=ORANGE, ms=7.5, mec="white", mew=1.1, zorder=3)
            ax.plot(s, yi, "o", color=GREEN, ms=7.5, mec="white", mew=1.1, zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(0.008, 1.5)
        ax.set_yticks(y, cases if ax is axes[0] else [""] * len(y))
        ax.set_xlabel("W$_1$ to the truth (lower is better)", fontsize=8, color=INK)
        ax.set_title(f"Suite truths, noise {float(nz):g}", fontsize=9.5, color=INK)
        style(ax)

    ax = axes[2]
    for nz, mk, lbl in (("0.001", "o", "noise 1e-3"), ("0.0001", "^", "noise 1e-4")):
        rows = red[f"suite@{nz}"]
        dy = 0.17 if nz == "0.001" else -0.17
        ax.plot([rows[c]["band90_cover"] for c in cases], y + dy, mk, color=GREEN, ms=7.5,
                mec="white" if mk == "o" else GREEN, mew=1.1 if mk == "o" else 1.6,
                mfc=GREEN if mk == "o" else "white", label=f"90% band, {lbl}", zorder=3)
    ax.axvline(0.9, color=MUTED, ls="--", lw=1)
    ax.set_xlim(0.55, 1.02)
    ax.set_yticks(y, [""] * len(y))
    ax.set_xlabel("fraction of |ω| ≤ 8 inside the 90% band", fontsize=8, color=INK)
    ax.set_title("Does the 90% band contain the truth?\n(dashed: nominal 0.9)", fontsize=9.5,
                 color=INK)
    style(ax)
    axes[0].plot([], [], "o", color=GREEN, ms=7, label="W$_1$: refined + redraw, posterior mean")
    axes[0].plot([], [], "o", color=ORANGE, ms=7, label="W$_1$: VAE ctrl-60k, amortized (one pass)")
    hs = [h for a in (axes[0], axes[2]) for h in a.get_legend_handles_labels()[0]]
    fig.legend(handles=hs, loc="outside lower center", ncol=4, fontsize=8, frameon=False)
    for a in axes:
        a.set_ylim(y[-1] - 0.7, y[0] + 0.7)
    fig.suptitle("Refined 60k flow (with noise redraw) on the 14 suite truths, one draw per noise "
                 "level", fontsize=10.5, color=INK)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    red_dir, nored_dir = Path(sys.argv[1]), Path(sys.argv[2])
    red = json.load(open(red_dir / "final.json"))
    nored = json.load(open(nored_dir / "final.json"))
    assert red["perturb"] and not nored["perturb"], "expected (redraw, no-redraw) order"
    fig_full(red, nored, red_dir / "summary_full.png")
    fig_suite(red, red_dir / "summary_suite.png")
    print(f"-> {red_dir}/summary_full.png, summary_suite.png")


if __name__ == "__main__":
    main()
