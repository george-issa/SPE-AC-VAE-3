r"""Training curves, read straight from run logs.

Reads the logs rather than requiring an in-memory history, so a figure can be
made for a run that is still going, or for one that finished days ago, without
the training script having anticipated it.

The per-step numbers are *single-batch* values, and each step draws a different
spectral family, :math:`\beta` and noise level -- so they scatter over six orders
of magnitude for reasons that have nothing to do with training progress.  A
smoothed line through them is worse than useless: on the first version of this
figure the running median of :math:`\chi^2` *rose* by four decades across
training while the held-out evaluation of the same run *fell*, because the
median was tracking which noise scale happened to be drawn.

So the per-step values are drawn as faint scatter only, with **no line through
them**, and the series that carries the argument is the held-out evaluation --
same seed, same batches for every run, therefore genuinely comparable.  Anything
joined by a line in these plots is an evaluation; anything faint is one batch.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np

__all__ = ["TrainingRun", "parse_training_log", "plot_training"]

_STEP = re.compile(
    r"step\s+(\d+)\s+loss\s+(\S+)\s+chi2\s+(\S+)\s+w1\s+(\S+)\s+l2\s+(\S+)\s+kl\s+(\S+)")
_EVAL = re.compile(r"\[eval @ (\d+)\]\s+(.*)")
_KV = re.compile(r"([A-Za-z0-9_/]+)=([-\d.eE+naif]+)")
_HEADER = re.compile(r"preset=(\S+)\s+device=(\S+)\s+steps=(\d+)\s+batch=(\d+)\s+"
                     r"d_model=(\d+)\s+beta=(\S+)\s+residues=(\S+)")


@dataclass
class TrainingRun:
    """One run's parsed curves."""

    label: str
    step: np.ndarray = field(default_factory=lambda: np.zeros(0))
    loss: np.ndarray = field(default_factory=lambda: np.zeros(0))
    chi2: np.ndarray = field(default_factory=lambda: np.zeros(0))
    w1: np.ndarray = field(default_factory=lambda: np.zeros(0))
    l2: np.ndarray = field(default_factory=lambda: np.zeros(0))
    kl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    eval_step: np.ndarray = field(default_factory=lambda: np.zeros(0))
    evals: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    wall_time: float | None = None

    @property
    def finished(self) -> bool:
        return self.wall_time is not None


def parse_training_log(path: str, label: str | None = None) -> TrainingRun:
    """Parse one ``smoke_train.py`` log, complete or in progress."""
    run = TrainingRun(label=label or os.path.splitext(os.path.basename(path))[0])
    rows, ev_steps, ev_rows = [], [], []
    with open(path, errors="replace") as f:
        for line in f:
            if (h := _HEADER.search(line)):
                run.config = dict(preset=h.group(1), device=h.group(2),
                                  steps=int(h.group(3)), batch=int(h.group(4)),
                                  d_model=int(h.group(5)), beta=h.group(6),
                                  residues=h.group(7))
            elif (m := _STEP.search(line)):
                rows.append([float(m.group(i)) for i in range(1, 7)])
            elif (e := _EVAL.search(line)):
                ev_steps.append(int(e.group(1)))
                ev_rows.append({k: float(v) for k, v in _KV.findall(e.group(2))})
            elif "wall time:" in line:
                run.wall_time = float(line.split(":")[1].strip().rstrip("s"))

    if rows:
        a = np.asarray(rows)
        run.step, run.loss, run.chi2, run.w1, run.l2, run.kl = (a[:, i] for i in range(6))
    if ev_rows:
        run.eval_step = np.asarray(ev_steps, dtype=float)
        keys = set().union(*(r.keys() for r in ev_rows))
        run.evals = {k: np.asarray([r.get(k, np.nan) for r in ev_rows]) for k in keys}
    return run


def _running_median(y: np.ndarray, w: int = 15) -> np.ndarray:
    if y.size < 3:
        return y
    w = max(3, min(w, y.size) | 1)
    pad = w // 2
    padded = np.pad(y, pad, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, w), axis=-1)


#: Fixed per-run colours so a run keeps its colour across figures (Okabe--Ito).
_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
            "#E69F00", "#56B4E9", "#666666"]


def plot_training(
    logs,
    labels=None,
    floors: dict | None = None,
    smooth: int = 15,
    title: str | None = None,
    path: str | None = None,
):
    r"""Plot :math:`\chi^2`, :math:`W_1`, :math:`L^2` and KL against step.

    Parameters
    ----------
    logs
        Log paths, or a glob.  In-progress runs are drawn up to wherever they
        are, with an open marker at the end so a partial curve is not mistaken
        for a converged one.
    floors
        Optional ``{label: chi2_floor}`` from
        :func:`~spevae.diagnostics.ansatz_chi2_floor`.  Drawn as a horizontal
        line on the :math:`\chi^2` panel: without it a reader cannot tell
        "still training" from "at the representation's limit", and those call
        for opposite responses.
    smooth
        Running-median window over the per-step scatter.
    """
    import matplotlib.pyplot as plt

    if isinstance(logs, str):
        logs = sorted(glob.glob(logs))
    runs = [parse_training_log(p, l) for p, l in
            zip(logs, labels or [None] * len(logs))]

    panels = [("chi2", r"$\chi^2$ (per d.o.f.)", True),
              ("w1", r"$W_1$  [physical $\omega$]", False),
              ("l2", r"relative $L^2$ on $A$", False),
              ("kl", "KL (nats, per-step, smoothed)", False)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), constrained_layout=True)

    for ax, (key, ylab, logy) in zip(axes.ravel(), panels):
        for i, r in enumerate(runs):
            c = _PALETTE[i % len(_PALETTE)]
            y = getattr(r, key)
            if y.size == 0:
                continue
            ax.plot(r.step, y, ".", color=c, alpha=0.30, ms=3.5, mew=0)
            ev_key = {"chi2": "chi2", "w1": "w1_omega", "l2": "l2_rel",
                      "kl": None}.get(key)
            if ev_key and ev_key in r.evals:
                ax.plot(r.eval_step, r.evals[ev_key], "-o", color=c, lw=1.8,
                        ms=6, mec="white", mew=1.2, zorder=5,
                        label=r.label if key == "chi2" else None)
            elif key == "kl":
                # No held-out KL is recorded, so show the smoothed per-step
                # value and say so in the label.
                ax.plot(r.step, _running_median(y, smooth), color=c, lw=1.5,
                        alpha=0.9)
            if not r.finished and r.eval_step.size and ev_key in (r.evals or {}):
                ax.plot(r.eval_step[-1], r.evals[ev_key][-1], marker=">",
                        color=c, ms=9, mfc="none", zorder=6)
        if logy:
            ax.set_yscale("log")
            if floors:
                for i, r in enumerate(runs):
                    if r.label in floors:
                        ax.axhline(floors[r.label], color=_PALETTE[i % len(_PALETTE)],
                                   ls="--", lw=1.0, alpha=0.8)
                ax.text(0.98, 0.04, "dashed = ansatz floor (best this\n"
                                    "representation can reach)",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=8, color="0.35")
        ax.set_xlabel("step")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25, lw=0.5)

    axes[0, 0].legend(frameon=False, fontsize=9)
    note = ("lines + circles = held-out eval (fixed seed, same batches for every "
            "run); dots = single batch, no line drawn through them\n"
            "because they scatter over decades with the family and noise scale "
            "drawn, not with progress")
    fig.suptitle((title + "\n" if title else "") + note,
                 fontsize=10 if title else 8.5)
    if path:
        fig.savefig(path, dpi=150)
    return fig, axes
