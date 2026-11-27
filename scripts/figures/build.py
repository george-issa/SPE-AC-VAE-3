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
    ("2_real_data", "hubbard_dimer_dqmc_stated_discrete.png",
     ("out/dqmc/{m}_discrete/dimer_judge.png",),
     "The same dimer bins with the spectrum stated to be discrete (`spevae continue --discrete`): the four "
     "lines are fitted as zero-width lines.",
     "As above; the posterior's lines are the blue bars, on the scale of the exact lines."),
    ("2_real_data", "hubbard_ring4_dqmc.png",
     ("out/dqmc/{m}/ring4_judge.png",),
     "Real DQMC bins of the half-filled four-site Hubbard ring (U = 4, beta = 8, 400 bins), whose spectrum is "
     "known exactly: a dense set of lines, the kind of spectrum this model finds hardest.",
     "As for the dimer."),
    ("2_real_data", "hubbard_ring4_dqmc_discrete_statement.png",
     ("out/dqmc/{m}_discrete/ring4_judge_windows.png",),
     "The four-site ring again, window by window: the exact weight against the intervals the report states "
     "without and with the statement that the spectrum is discrete.",
     "Black ticks: exact weight. Blue: stated 90% interval as the report gives it. Green: with `--discrete`. "
     "Grey: the range the data alone allow."),
    ("2_real_data", "hubbard_dimer_beta2_dqmc.png",
     ("out/dqmc/{m}_b2/dimer_judge.png",),
     "Real DQMC bins of the same dimer at a high temperature (U = 4, beta = 2, 400 bins).",
     "As for the dimer at beta = 8."),
    ("2_real_data", "hubbard_dimer_beta2_dqmc_stated_discrete.png",
     ("out/dqmc/{m}_b2_discrete/dimer_judge.png",),
     "The dimer at beta = 2 with the spectrum stated to be discrete.",
     "As for the dimer at beta = 8."),
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
    ("3_error_bars", "zero_width_lines.png", ("out/step3/line_fit_{m}/line_fit.png",),
     "Exact line spectra at all twelve settings: the report before zero-width lines, at its defaults (lines "
     "fitted when peaks of the model's minimum width do not reach the data, which happens at relative error "
     "bars near 1e-5), and with the spectrum stated to be discrete (lines fitted whenever narrow lines are "
     "flagged). Grey: lines fitted whenever they fit, the rule that was not shipped.",
     "Top left: error over MaxEnt's on the same data (below 1 = better). Top right: one point per flagged "
     "line spectrum, with the statement against before. Bottom left: share of windows whose exact weight "
     "lies inside the stated intervals; dashes are nominal. Bottom right: all settings pooled."),
    ("3_error_bars", "discrete_statement.png", ("out/step3/discrete_option_{m}/discrete_option.png",),
     "Exact line spectra again: what stating that the spectrum is discrete (`spevae continue --discrete`) "
     "does to the intervals. Many close lines look like a continuous band in the data, so the report cannot "
     "flag them itself; the statement is how the user tells it.",
     "Left: share of windows whose exact weight lies inside the stated intervals, without the statement (blue) "
     "and with it (green); dashes are nominal. Right: the width of the stated 90% interval as a share of the "
     "range the data alone allow. The last group is the cost of making the statement about a continuous "
     "spectrum. Made before zero-width lines were fitted: the line spectra the report does not flag, which "
     "this figure is about, are reported the same way now; the flagged ones are in zero_width_lines.png."),
    ("4_checks", "edges_of_trained_range.png", ("out/step3/edges_{m}/edges.png",),
     "Known answers at the edges of the range the model was trained on: beta = 1 and 20, relative error bars "
     "of 1e-5, dtau = 0.02 and 0.5 (20 exact lattice and 24 exact line spectra per setting).",
     "Left: this model's error over MaxEnt's on the same data (below 1 = this model is better). Middle and "
     "right: share of windows whose exact weight lies inside the stated intervals; dashes are nominal. Made "
     "before zero-width lines were fitted: the line spectra at relative error bars of 1e-5 are better now "
     "(3_error_bars/zero_width_lines.png); everything else is unchanged."),
    ("4_checks", "lines_whenever_they_fit_on_prior_draws.png", ("out/step3/line_fit_always_{m}/prior_draws.png",),
     "Why zero-width lines are not fitted whenever they fit: held-out draws of the training prior that the "
     "report flags, by what they really are. Where the narrow features are lines the error falls; where "
     "they are peaks of the minimum width or band edges it rises, and the data cannot tell the two apart.",
     "Left: share of windows whose exact weight lies inside the stated intervals, before and with lines "
     "fitted whenever they fit. Right: error of the mean window weights, with lines over before."),
    ("4_checks", "training_seeds.png", ("F3L_60k_s1=out/step3/seeds_lines_report.png",),
     "Two independent trainings of this model's recipe, through the report, on the same known answers.",
     "Left: error per setting for each seed. Middle: the same spectrum under the two trainings, one point "
     "each. Right: the stated intervals. Agreement shows the results belong to the recipe, not to one run."),
    ("4_checks", "training_seeds_earlier_recipe.png", ("out/step3/seeds.png",),
     "Three independent trainings of the earlier recipe (`F3_60k`).",
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
SHARED = {"training_seeds_earlier_recipe.png", "families_left_out.png", "line_spectra_headroom.png",
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
              "On line spectra the median curve is low wherever the samples disagree about a line's exact "
              "position, and the band is clipped where it towers over the rest; the error is taken on the "
              "cumulative weight, which has neither problem. \"Narrow lines flagged\" in a title means the "
              "report warned that the data favour lines narrower than the model was trained on, and states "
              "wider window intervals for them.", "",
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
    stated = [(b, n, summ[f"lines_stated_discrete_beta{b:g}_errorbar{n:g}.png"]) for b, n in shown
              if f"lines_stated_discrete_beta{b:g}_errorbar{n:g}.png" in summ]
    if stated:
        lines += ["The line spectra again with the spectrum stated to be discrete (`--discrete`), which these "
                  "are: features at the model's minimum width are then fitted as zero-width lines, drawn as "
                  "blue bars on the scale of the exact lines.", "",
                  "| setting | line spectra, stated discrete: this model / MaxEnt | lower on | fitted with "
                  "zero-width lines |", "|---|---|---|---|"]
        lines += [f"| beta = {b:g}, relative error bar {n:g} | {v['model']:.3f} / {v['maxent']:.3f} | "
                  f"{v['model_lower']} of {v['n']} | {v['with_lines']} of {v['n']} |" for b, n, v in stated]
        lines.append("")
    for beta, noise in shown:
        for word in ("lattice", "lines", "lines_stated_discrete"):
            name = f"{word}_beta{beta:g}_errorbar{noise:g}.png"
            if name in summ:
                lines += [f"**{name}**", "", f"![{name}](1_known_answers/{name})", ""]

    # 2-4. copied from the judges
    heads = {"2_real_data": "Real DQMC data.",
             "3_error_bars": "Whether the stated intervals contain the truth as often as they say.",
             "4_checks": "Checks behind the numbers. The seeds of the earlier recipe, the families left out and "
                         "the line-spectrum headroom were made on the `F3_60k` trainings (the same recipe "
                         "without exact line spectra as targets) and are not redrawn per model."}
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
