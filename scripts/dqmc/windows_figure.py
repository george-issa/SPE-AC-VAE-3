"""Window by window: the exact weight of a known-answer DQMC run against what the report states.

Reads two results of `judge_dimer.py` on the same bins, the report as it is
and the report with the spectrum stated to be discrete, and draws for every
frequency window the range the data alone allow, the two stated 90% intervals
(the 50% intervals inside them) and the exact weight.

    python scripts/dqmc/windows_figure.py DEFAULT.json DISCRETE.json OUT.png
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OFF, ON, GREY, INK = "#2a78d6", "#0b7a75", "#c9c8c3", "#0b0b0b"


def main():
    a, b, out = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
    wa, wb = a["flow"]["windows"], b["flow"]["windows"]
    f, ax = plt.subplots(figsize=(11.5, 4.6), constrained_layout=True)
    for j, (x, y) in enumerate(zip(wa, wb)):
        assert x["window"] == y["window"]
        ax.fill_between([j - 0.36, j + 0.36], [x["lo"]] * 2, [x["hi"]] * 2, color=GREY, alpha=0.55, lw=0)
        for w, col, dx in ((x, OFF, -0.14), (y, ON, 0.14)):
            ax.plot([j + dx] * 2, [w["q05"], w["q95"]], color=col, lw=2.2, solid_capstyle="butt")
            ax.plot([j + dx] * 2, [w["q25"], w["q75"]], color=col, lw=6.5, solid_capstyle="butt")
        ax.plot([j - 0.36, j + 0.36], [x["exact"]] * 2, color=INK, lw=1.8)
    name = lambda w: "[{:g}, {:g})".format(*w["window"]).replace("inf", "∞")
    ax.set_xticks(range(len(wa)), [name(w) for w in wa])
    ax.set_xlabel(r"frequency window ($\omega$)")
    ax.set_ylabel("share of the spectral weight")
    ax.plot([], [], color=INK, lw=1.8, label="exact")
    ax.plot([], [], color=OFF, lw=4, label="the report as it is: 90% (thin) and 50% (thick) intervals")
    ax.plot([], [], color=ON, lw=4, label="with the spectrum stated to be discrete")
    ax.fill_between([], [], [], color=GREY, alpha=0.55, lw=0, label="allowed by the data alone")
    ax.legend(fontsize=8.5, frameon=False, loc="upper center", ncol=2)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.18)
    ha, hb = a["flow"]["window_hits"], b["flow"]["window_hits"]
    what = {"dimer": "Hubbard dimer"}.get(a["model"], f"Hubbard ring of {a['model'][4:]} sites")
    ax.set_title(f"Real DQMC data, {what}, U = {a['U']:g}, beta = {a['beta']:g}: exact weight inside the stated "
                 f"90% interval in {ha['inside_90']} of {len(wa)} windows as it is, {hb['inside_90']} of {len(wb)} "
                 "with the statement", fontsize=10)
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    f.savefig(out, dpi=150)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
