r"""Scorecard files and figures.

``write_report(card, out_dir)`` writes:

- ``scorecard.json`` -- provenance, raw per-draw records, the aggregated
  summary and paired comparisons (the machine-readable result);
- ``summary.md`` -- the same headline numbers as tables;
- ``w1_scorecard.png`` -- W1 per case, method and model with 95% intervals;
- ``holstein.png`` -- chi^2/target on the real cell, the manifold floor,
  and the spectra each method returns;
- ``spectra_<noise>.png`` -- truth vs reconstructions on the banked draw.

Colours follow the entity (a model keeps its colour in every figure, in
the order the models were given); MaxEnt and other references are drawn
in neutral ink with their own marker, never in a series colour.
"""

from __future__ import annotations

import json
import os

import numpy as np

from . import cases as C
from .stats import compare_pairs, group_of, summarize

__all__ = ["write_report"]

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "D", "^", "v", "P"]
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#9a9893", "#fcfcfb"
REF = {"maxent": dict(color=INK2, marker="o", mfc="none", ls="--"),
       "deac": dict(color=INK2, marker="x", mfc=INK2, ls=":"),
       "deac_shipped": dict(color=INK, marker="+", mfc=INK, ls="-")}
MODEL_METHODS = ("amortized", "zd800", "zd1500", "zd3000", "zdlm", "scan_full",
                 "scan_minimal")


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE, "axes.grid": True, "grid.color": "#e4e3df",
        "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 1.4})
    return plt


def _groups(card):
    seen = []
    for tag in card["models"]:
        g = group_of(tag)
        if g not in seen:
            seen.append(g)
    return seen


def _methods(card, kind="synthetic"):
    present = []
    for r in card["records"]:
        if r.get("kind") == kind and r["method"] not in present:
            present.append(r["method"])
    order = list(MODEL_METHODS) + ["maxent", "deac", "deac_shipped"]
    return sorted(present, key=lambda m: order.index(m) if m in order else 99)


# ------------------------------------------------------------- figures --- #
def _fig_w1(card, rows, path):
    plt = _style()
    groups, methods = _groups(card), _methods(card)
    noises = card["suite"]["noises"]
    names = [n for n in C.CASES if any(r["case"] == n for r in rows)]
    model_methods = [m for m in methods if m in MODEL_METHODS]
    has_ref = "maxent" in methods
    if not model_methods:
        return
    idx = {(r["model"], r["method"], r["case"], r["noise"]): r for r in rows}
    fig, axes = plt.subplots(len(noises), len(model_methods), sharey=True, sharex=True,
                             figsize=(2.9 * len(model_methods) + 1.6, 0.26 * len(names) * len(noises) + 1.2),
                             squeeze=False, constrained_layout=True)
    y = np.arange(len(names))[::-1]
    off = np.linspace(-0.22, 0.22, max(len(groups), 1)) if len(groups) > 1 else [0.0]
    for i, nz in enumerate(noises):
        for j, meth in enumerate(model_methods):
            ax = axes[i, j]
            for k, g in enumerate(groups):
                pts = [(yy + off[k], idx.get((g, meth, n, nz))) for yy, n in zip(y, names)]
                pts = [(yy, r["w1"]) for yy, r in pts if r and "w1" in r]
                if not pts:
                    continue
                yy = np.array([p[0] for p in pts])
                mu = np.array([p[1]["mean"] for p in pts])
                lo = np.array([p[1]["lo"] for p in pts])
                hi = np.array([p[1]["hi"] for p in pts])
                ax.errorbar(mu, yy, xerr=[mu - lo, hi - mu], fmt=MARKERS[k], ms=4.5,
                            color=SERIES[k], ecolor=SERIES[k], elinewidth=1.0, capsize=0,
                            mec=SURFACE, mew=0.6, label=g if (i == 0 and j == 0) else None)
            if has_ref:
                ref = [(yy, idx.get(("-", "maxent", n, nz))) for yy, n in zip(y, names)]
                ref = [(yy, r["w1"]["mean"]) for yy, r in ref if r and "w1" in r]
                if ref:
                    ax.plot([v for _, v in ref], [yy for yy, _ in ref], ls="none",
                            marker="o", ms=6, mfc="none", mec=INK2, mew=0.9,
                            label="MaxEnt" if (i == 0 and j == 0) else None)
            ax.set_xscale("log")
            ax.set_title(f"{meth}  |  noise {nz:g}", color=INK)
            if i == len(noises) - 1:
                ax.set_xlabel(r"$W_1$ to truth  (mean, 95% CI over draws)")
    for ax in axes[:, 0]:
        ax.set_yticks(y)
        ax.set_yticklabels(names)
    fig.legend(loc="outside lower center", ncol=len(groups) + int(has_ref))
    fig.suptitle("Spectral error per test case", color=INK, fontsize=10)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _fig_holstein(card, path):
    plt = _style()
    real = [r for r in card["records"] if r.get("kind") == "real"]
    floors = [r for r in card["records"] if r.get("kind") == "floor"]
    if not real:
        return
    groups = _groups(card)
    omega = np.asarray(card["plot_omega"])
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 3.9), constrained_layout=True,
                                   gridspec_kw={"width_ratios": [1, 1.35]})
    labels, k = [], 0
    for r in sorted(real, key=lambda r: (r["model"] == "-", r["model"], r["method"])):
        g = group_of(r["model"])
        if g in groups:
            c, mk, mfc = SERIES[groups.index(g)], MARKERS[groups.index(g)], None
        else:
            st = REF.get(r["method"], REF["maxent"])
            c, mk, mfc = st["color"], st["marker"], st["mfc"]
        line_marker = mk in ("x", "+", "|")
        ax0.plot(r["ratio"], k, marker=mk, ms=7 if line_marker else 6, color=c, ls="none",
                 mfc=mfc or c, mew=1.4 if line_marker else 0.6,
                 mec=c if (mfc == "none" or line_marker) else SURFACE)
        labels.append(f"{r['method']}" + (f"  ({g})" if g in groups else ""))
        k += 1
    for r in floors:
        g = group_of(r["model"])
        c = SERIES[groups.index(g)] if g in groups else INK2
        ax0.plot(r["ratio"], k, marker="|", ms=12, mew=2, color=c, ls="none")
        labels.append(f"manifold floor  ({g})"
                      + ("" if r.get("floor_converged", True) else ", step cap"))
        k += 1
        if "floor_lm_ratio" in r:
            ax0.plot(r["floor_lm_ratio"], k, marker="|", ms=12, mew=2, color=c, ls="none")
            labels.append(f"floor, least squares  ({g})")
            k += 1
    ax0.axvline(1.0, color=INK2, lw=0.9, ls="--")
    ax0.annotate("noise floor", xy=(1.0, 1.0), xycoords=("data", "axes fraction"),
                 xytext=(4, -2), textcoords="offset points", color=INK2, fontsize=7,
                 va="top")
    ax0.set_yticks(range(k))
    ax0.set_yticklabels(labels)
    ax0.invert_yaxis()
    ax0.set_xscale("log")
    ax0.set_xlabel(r"$\chi^2/\chi^2_{\rm target}$ on the Holstein cell")
    ax0.set_title(r"Data fit ($\chi^2_{\rm target}$ = %.3f)" % real[0]["chi2_target"])

    core = np.abs(omega) <= 6
    for r in real:
        g = group_of(r["model"])
        if g in groups:
            if r["method"] != max((m for m in MODEL_METHODS if any(
                    q["method"] == m and group_of(q["model"]) == g for q in real)),
                    key=MODEL_METHODS.index):
                continue
            ax1.plot(omega[core], np.asarray(r["A_plot"])[core], color=SERIES[groups.index(g)],
                     lw=1.6, label=f"{g} ({r['method']})")
        else:
            st = REF.get(r["method"], REF["maxent"])
            ax1.plot(omega[core], np.asarray(r["A_plot"])[core], color=st["color"],
                     ls=st["ls"], lw=1.1, label=r["method"])
    ax1.set_xlabel(r"$\omega$")
    ax1.set_ylabel(r"$A(\omega)$")
    ax1.set_title("Spectra returned on the Holstein cell")
    ax1.legend(fontsize=7.5)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _fig_spectra(card, noise, path):
    plt = _style()
    groups = _groups(card)
    omega = np.asarray(card["plot_omega"])
    recs = [r for r in card["records"] if r.get("kind") == "synthetic"
            and r["noise"] == noise and "A_plot" in r]
    if not recs:
        return
    names = [n for n in C.CASES if any(r["case"] == n for r in recs)]
    ncol = min(5, len(names))
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 1.95 * nrow),
                             constrained_layout=True, squeeze=False)
    core = np.abs(omega) <= 6
    for ax, name in zip(axes.flat, names):
        ax.plot(omega[core], C.truth_on_grid(name, omega)[core], color=INK, lw=2.0,
                label="truth")
        for k, g in enumerate(groups):
            mine = [r for r in recs if r["case"] == name and group_of(r["model"]) == g]
            if not mine:
                continue
            best = max(mine, key=lambda r: MODEL_METHODS.index(r["method"])
                       if r["method"] in MODEL_METHODS else -1)
            ax.plot(omega[core], np.asarray(best["A_plot"])[core], color=SERIES[k], lw=1.3,
                    label=f"{g} ({best['method']})")
        ref = [r for r in recs if r["case"] == name and r["method"] == "maxent"]
        if ref:
            ax.plot(omega[core], np.asarray(ref[0]["A_plot"])[core], color=INK2, lw=1.0,
                    ls="--", label="MaxEnt")
        ax.set_title(name, color=INK)
        ax.tick_params(labelleft=False)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=len(lab))
    fig.suptitle(f"Reconstructions on the banked noise draw (seed 7, noise {noise:g})",
                 color=INK, fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------ markdown --- #
def _md(card, rows, comps):
    s = card["suite"]
    lines = [f"# spevae eval — tier `{s['tier']}` (suite v{s['version']})", "",
             f"Created {s['created']}, commit `{(s.get('commit') or 'n/a')[:10]}`"
             f"{' (dirty)' if s.get('dirty') else ''}, torch {s['torch']}, "
             f"{s['workers']} workers × {s['threads']} threads, {s['wall_seconds']:.0f} s.", ""]
    lines += ["## Models", "", "| tag | params | sha256 |", "|---|---|---|"]
    for tag, runs in card["models"].items():
        for r in runs:
            lines.append(f"| {tag} | {r['params']:,} | `{r['sha256'][:12]}` |")
    streams = [r for r in card["records"] if r.get("kind") == "streams"]
    if streams:
        lines += ["", "## Frozen streams (amortized, W1)", "", "| model | pole stream | canary |",
                  "|---|---|---|"]
        for r in streams:
            lines.append(f"| {r['model']} | {r['pole']['w1_omega']:.4f} | "
                         f"{r['canary']['w1_omega']:.4f} |")
    real = [r for r in card["records"] if r.get("kind") == "real"]
    floors = [r for r in card["records"] if r.get("kind") == "floor"]
    if real or floors:
        lines += ["", "## Holstein cell", "", "| model | method | chi2 | chi2/target | runtime (s) |",
                  "|---|---|---|---|---|"]
        for r in real:
            lines.append(f"| {r['model']} | {r['method']} | {r['chi2']:.4g} | {r['ratio']:.3g} | "
                         f"{r.get('runtime', float('nan')):.1f} |")
        for r in floors:
            cap = "" if r.get("floor_converged", True) else " (Adam, hit the step cap)"
            lines.append(f"| {r['model']} | manifold floor{cap} | {r['floor']:.4g} | "
                         f"{r['ratio']:.3g} | {r['runtime']:.0f} |")
            if "floor_lm" in r:
                lines.append(f"| {r['model']} | floor, least squares ({r['floor_lm_status']}, "
                             f"{r['floor_lm_iters']} it) | {r['floor_lm']:.4g} | "
                             f"{r['floor_lm_ratio']:.3g} | |")
                lines.append(f"| {r['model']} | MAP, least squares ({r['map_lm_status']}, "
                             f"{r['map_lm_iters']} it) | {r['map_lm']:.4g} | "
                             f"{r['map_lm_ratio']:.3g} | |")
    idx = {(r["model"], r["method"], r["case"], r["noise"]): r for r in rows}
    names = [n for n in C.CASES if any(r["case"] == n for r in rows)]
    key_methods = [m for m in ("amortized", "zd3000", "zdlm", "scan_minimal", "scan_full")
                   if any(r["method"] == m for r in rows)]
    groups = _groups(card)
    cols = [(g, m) for m in key_methods for g in groups]
    if any(r["method"] == "maxent" for r in rows):
        cols.append(("-", "maxent"))
    for nz in s["noises"]:
        lines += ["", f"## W1 to truth at noise {nz:g} (mean over draws; chi2 in brackets)", "",
                  "| case | " + " | ".join(f"{g}/{m}" if g != "-" else m for g, m in cols) + " |",
                  "|---|" + "---|" * len(cols)]
        for name in names:
            cells = []
            for g, m in cols:
                r = idx.get((g, m, name, nz))
                cells.append(f"{r['w1']['mean']:.3g} ({r['chi2']['mean']:.2g})"
                             if r and "w1" in r else "")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "All methods, intervals and per-draw values: `scorecard.json`."]
    for (a, b), comp in comps.items():
        if not comp:
            continue
        la = a[1] if a[0] == "-" else f"{a[0]}/{a[1]}"
        lb = b[1] if b[0] == "-" else f"{b[0]}/{b[1]}"
        n_sig_b = sum(1 for c in comp if c["hi"] < 1)
        n_sig_a = sum(1 for c in comp if c["lo"] > 1)
        lines += ["", f"## Paired: {lb} vs {la} (W1 ratio {lb} / {la})", "",
                  f"{lb} better on {n_sig_b}, worse on {n_sig_a}, of {len(comp)} "
                  "case×noise cells (95% interval excludes 1).", "",
                  "| case | noise | ratio | 95% CI | draws |", "|---|---|---|---|---|"]
        for c in comp:
            flag = " **" if c["hi"] < 1 or c["lo"] > 1 else ""
            lines.append(f"| {c['case']} | {c['noise']:g} | {c['ratio']:.3g}{flag} | "
                         f"[{c['lo']:.3g}, {c['hi']:.3g}] | {c['n']} |")
    if card.get("failures"):
        lines += ["", "## Failures", ""] + [f"- {f['task']}: {f['error']}" for f in card["failures"]]
    return "\n".join(lines) + "\n"


def write_report(card: dict, out_dir: str, figures: bool = True) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    rows = summarize(card["records"])
    groups = _groups(card)
    comps = {}
    methods = _methods(card)
    for meth in [m for m in methods if m in MODEL_METHODS]:
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                comps[((groups[i], meth), (groups[j], meth))] = compare_pairs(
                    card["records"], (groups[i], meth), (groups[j], meth))
    if "maxent" in methods:
        for g in groups:
            for meth in [m for m in methods if m in MODEL_METHODS]:
                comps[(("-", "maxent"), (g, meth))] = compare_pairs(
                    card["records"], ("-", "maxent"), (g, meth))
    card = dict(card)
    card["summary"] = rows
    card["comparisons"] = [{"a": list(a), "b": list(b), "rows": v}
                           for (a, b), v in comps.items()]
    with open(os.path.join(out_dir, "scorecard.json"), "w") as fh:
        json.dump(card, fh, indent=1)
    with open(os.path.join(out_dir, "summary.md"), "w") as fh:
        fh.write(_md(card, rows, comps))
    if figures:
        _fig_w1(card, rows, os.path.join(out_dir, "w1_scorecard.png"))
        _fig_holstein(card, os.path.join(out_dir, "holstein.png"))
        for nz in card["suite"]["noises"]:
            _fig_spectra(card, nz, os.path.join(out_dir, f"spectra_{nz:g}.png"))
    return card
