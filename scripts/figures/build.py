"""Collect the figures of one model into figures/, in reading order, with an index.

    python scripts/figures/build.py MODEL [DEST=figures]

MODEL is a checkpoint name under out/step3/ (for example F3_60k_s1).  The
known-answer galleries are drawn from the curves stored by
`scripts/figures/known_answers.py run out/figures/MODEL ...`; every other
figure is copied from where its judge wrote it.  A figure whose source is
missing, or was made with another model, is listed as such in the index
instead of being copied silently.

DEST/README.md is rewritten on every build: one section per folder, every
figure shown with what it is and how to read it.
"""
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "scripts/figures")

import known_answers as K

SETTINGS = ((8, 0.001), (8, 0.0001), (8, 0.01), (2, 0.001), (16, 0.001))

# (folder, file name, candidate sources in order, what it shows, how to read it)
COPIES = (
    ("2_real_data", "hubbard_dimer_dqmc.png",
     ("out/dqmc/{m}/dimer_judge.png", "F3_60k_s1=out/dqmc/F3_60k_s1_floor_auto/dimer_judge.png"),
     "Real DQMC bins of the half-filled Hubbard dimer (U = 4, beta = 8, 400 bins), whose spectrum is known "
     "exactly: four lines.",
     "Left: measured minus exact G(tau) in units of the error bar. Middle: the reconstructions on the exact "
     "lines. Right: cumulative weight, where the area between a curve and the black steps is the error."),
    ("2_real_data", "holstein_dqmc.png",
     ("out/step3/holstein_{m}/holstein_report.png",),
     "Real DQMC bins of a Holstein model (beta = 8, 400 bins). No exact answer exists; the dashed curve is "
     "another continuation of the same data, not the truth.",
     "Left: the posterior band and median. Middle: how well each accepted sample fits the bins (1 = as well as "
     "the noise allows). Right: weight per frequency window: the posterior's 50% and 90% intervals inside the "
     "range the data alone allow (grey)."),
    ("3_error_bars", "prior_draws.png",
     ("out/step3/cal_{m}_big/band_calibration.png",),
     "How often the stated 50% and 90% intervals on window weights contain the truth, on held-out draws of the "
     "training prior.",
     "Middle and right: green is what the report states, by error bar and by beta; the dashed lines are the "
     "nominal 0.5 and 0.9. Orange is the raw sample quantiles before the correction."),
    ("3_error_bars", "narrow_lines.png",
     ("out/step3/floor_option_{m}_table/floor_option.png", "out/step3/floor_option_{m}/floor_option.png",
      "F3_60k_s1=out/step3/floor_option/floor_option.png"),
     "Exact line spectra: what the narrow-line handling of the report does to the error and to the intervals.",
     "Left: error per setting, as trained, with the narrow-line fit, and MaxEnt. Right: share of windows whose "
     "exact weight lies inside the stated interval (filled) for flagged line spectra, unflagged line spectra "
     "and lattice spectra; dashes are nominal."),
    ("4_checks", "training_seeds.png", ("out/step3/seeds.png",),
     "Three independent trainings of the same model.",
     "Agreement between seeds shows how much of any difference is the luck of one training run."),
    ("4_checks", "families_left_out.png", ("out/step3/leave_out.png",),
     "A model trained without eight of the exactly solvable families, judged on them.",
     "Left: its error over the full model's, one point per spectrum; red families were never seen. Right: both "
     "against MaxEnt."),
    ("4_checks", "line_spectra_headroom.png", ("out/step3/line_labels/line_labels.png",),
     "Exact line spectra: the best the pole model can represent (fitted to noise-free data) against what the "
     "model and MaxEnt recover from noisy data.",
     "The gap between the dark points and the squares is what better training on line spectra could gain at "
     "most."),
    ("4_checks", "line_spectra_as_training_targets.png", ("out/step3/lines_retrain.png",),
     "The model trained with exact line spectra as targets (`F3L_60k_s1`) against the one trained on broadened "
     "lines only (`F3_60k_s1`), on the same known answers.",
     "Left and middle: the error barely moves. Right: the stated intervals on line spectra, which improve "
     "because the retrained model's report flags narrow lines on twice as many of them."),
)
SHARED = {"training_seeds.png", "families_left_out.png", "line_spectra_headroom.png",
          "line_spectra_as_training_targets.png"}   # made on the F3_60k family


def main():
    model = sys.argv[1]
    dest = Path(sys.argv[2] if len(sys.argv) > 2 else "figures")
    today = time.strftime("%Y-%m-%d")
    lines = [f"# Figures: model `{model}`", "",
             f"Built {today} by `python scripts/figures/build.py {model}`. Read in folder order: what the "
             "model reconstructs when the answer is known, what it does on real DQMC bins, whether its error "
             "bars hold, and the checks behind those.", "",
             "Error throughout is the area between the reconstructed and the exact cumulative spectral weight "
             "(W1, in units of omega; lower is better). \"Relative error bar\" is the error bar of G(tau) over "
             "its largest value.", ""]
    missing = []

    # 1. known answers, through the report a user gets
    src = Path(f"out/figures/{model}")
    d1 = dest / "1_known_answers"
    if d1.exists():
        shutil.rmtree(d1)
    summ = K.fig(src, d1) if src.exists() else {}
    lines += ["## 1_known_answers", "",
              "Spectra with an exact answer, given noise and continued with `continue_posterior` at its "
              "defaults (the call a user makes) and with MaxEnt on the same data. Blue: posterior median and "
              "90% band. Grey: MaxEnt. Black: exact. Panel titles give the error of each; a red title marks a "
              "spectrum where MaxEnt is the lower.", "",
              "| setting | lattice spectra: this model / MaxEnt | lower on | line spectra: this model / MaxEnt "
              "| lower on |", "|---|---|---|---|---|"]
    shown = []
    for beta, noise in SETTINGS:
        a = summ.get(f"lattice_beta{beta:g}_errorbar{noise:g}.png")
        b = summ.get(f"lines_beta{beta:g}_errorbar{noise:g}.png")
        if not a:
            missing.append(f"1_known_answers: beta = {beta:g}, relative error bar {noise:g} (not computed yet)")
            continue
        lines.append(f"| beta = {beta:g}, relative error bar {noise:g} | {a['model']:.3f} / {a['maxent']:.3f} | "
                     f"{a['model_lower']} of {a['n']} | {b['model']:.3f} / {b['maxent']:.3f} | "
                     f"{b['model_lower']} of {b['n']} |")
        shown.append((beta, noise))
    lines.append("")
    for beta, noise in shown:
        for word in ("lattice", "lines"):
            name = f"{word}_beta{beta:g}_errorbar{noise:g}.png"
            lines += [f"**{name}**", "", f"![{name}](1_known_answers/{name})", ""]

    # 2-4. copied from the judges
    heads = {"2_real_data": "Real DQMC data.",
             "3_error_bars": "Whether the stated intervals contain the truth as often as they say.",
             "4_checks": "Checks behind the numbers. The first three were made on the `F3_60k` trainings "
                         "(the same recipe without exact line spectra as targets) and are not redrawn per model."}
    done = set()
    for folder, name, cands, what, how in COPIES:
        if folder not in done:
            lines += [f"## {folder}", "", heads[folder], ""]
            done.add(folder)
            (dest / folder).mkdir(parents=True, exist_ok=True)
        # a candidate is a path with {m} for the model, or "OWNER=path" for a run whose folder
        # does not carry its model's name; the first that exists is used, this model's first
        found = []
        for c in cands:
            owner, _, path = c.rpartition("=")
            path = Path(path.format(m=model))
            if path.exists():
                found.append((owner or model, path))
        found.sort(key=lambda t: t[0] != model)
        if not found:
            missing.append(f"{folder}/{name} (not made for this model yet)")
            (dest / folder / name).unlink(missing_ok=True)
            continue
        owner, got = found[0]
        other = owner != model and name not in SHARED
        shutil.copyfile(got, dest / folder / name)
        lines += [f"**{name}**" + (f" (made with the earlier model `{owner}`, not yet redrawn)" if other else ""),
                  "", what, "", how, "", f"![{name}]({folder}/{name})", ""]
        if other:
            missing.append(f"{folder}/{name} is from the run of another model ({owner})")
    lines += ["## Not in this folder yet", ""]
    lines += [f"- {m}" for m in missing] or ["Nothing: every figure above was made with this model."]
    lines.append("")
    (dest / "README.md").write_text("\n".join(lines))
    print(f"-> {dest}/README.md; {len(missing)} missing or from another model")
    for m in missing:
        print("   ", m)


if __name__ == "__main__":
    main()
