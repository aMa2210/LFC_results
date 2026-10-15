"""One figure per TOKENIZER, so size, Instruct/Base and generation are compared on an
x-axis that means the same thing for every curve in the panel.

  figures/family_qwen35.png   Qwen3.5 2B-35B, Instruct and Base         (248,044 ordinary ids)
  figures/family_qwen25_3.png Qwen2.5-14B/72B and Qwen3-14B, Instruct and Base (151,643)
  figures/family_gemma.png    Gemma-4 E2B/E4B (PLE) and 12B (dense), Instruct and Base
  figures/family_llama.png    Llama-3.1-8B, Instruct and Base -- one size and one
                              generation, so only the version channel carries anything

Why by tokenizer and not by family (advisor, 2026-10-07): alpha is a share of the ordinary
vocabulary, so the same alpha is the same intervention -- the same ids removed -- only
within one tokenizer. Qwen2.5 and Qwen3 share the 151k vocabulary; Qwen3.5 has its own
248k one. On the old single Qwen panel the two groups also separated by an order of
magnitude in alpha_50 for the Instruct arms (151k: 0.003-0.019; 248k: 0.054-0.099) with
no size trend across the gap, which the mixed panel read as a size or generation effect.
The 151k tokenizer masks MORE of the benchmark text at every alpha (ARC alpha=0.01: 49%
vs 40%), so the gap is not a gentler cut; tokenizer and generation are confounded in this
set and the split panels make that visible instead of hiding it. Gemma and Llama each have
one tokenizer, so their panels are unchanged.

The cross-family figures answer "which families differ"; these answer the three questions
that only make sense inside a family, and they answer them in one panel each by giving
each question its own channel:

  model       hue, one per model -- there are at most seven in a family, so the most
              direct encoding is affordable here in a way it is not across all sixteen arms
  size        line width, thin to thick, on the same scale as figures/draft_overlay_*.png
  version     Instruct solid with a filled marker, Base dashed and hollow
  generation  marker shape -- Qwen3.5, Qwen3 and Qwen2.5 for Qwen; for Gemma there is one
              generation but two model classes, the PLE E-series against the dense 12B,
              which Finding 31 shows is the same kind of distinction

These palettes are per-figure and do NOT follow the cross-family hue namespace in
figstyle.py. That is deliberate: a family figure is read on its own with its own legend,
and inside it the question is which model, not which family. Do not carry a hue from one
of these figures to another.

Two rows because the two questions need different y. Raw accuracy compares levels, which
is what a size comparison wants. LFC = A(alpha)/A(1) divides the baseline out, which is
what an Instruct-against-Base comparison wants, since several Base arms start far lower
and would otherwise look like they collapse sooner when they only started lower.

Direct-answer reference (advisor, 2026-10-07): a THIRD ROW, the no-CoT control's accuracy
at each alpha, drawn with exactly the encoding of the rows above so every arm's reference
is as distinguishable as its curve. It shares the accuracy row's y scale, so the two are
read one above the other: where the CoT curve in row 1 drops below the flat line in row 3,
the chain is costing the model answers it would give without reasoning. An earlier draft
overlaid the reference as dotted lines on rows 1-2; eight same-hue dotted lines on one
panel could not be told apart. The row is added only for panels where at least one arm
has the cells, so a panel without any no-CoT run keeps its two rows.

Every arm is drawn the same way. The same division makes LFC fragile for an arm whose
A(1) is already near the floor -- Gemma-4-E2B Base starts at 0.26, and Finding 30 says
not to read its LFC panel -- but the accuracy row above shows that baseline directly, so
a reader can see the condition rather than be told about it. An earlier version faded
those curves and named them under the figure; that is a judgement drawn on top of the
data, and it is not this figure's place to make it.
"""
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from arms import (ALPHAS, GEN, IS_BASE, MODEL, ORDINARY, PARAMS, SUFFIX, acc, label_of, lfc,
                  nocot_acc, of_family)
from figstyle import width, marker_size

TASKS = [("gsm8k", "GSM8K"), ("arc", "ARC-Challenge"), ("mmlu", "MMLU")]
FS_TICK, FS_TITLE, FS_LABEL, FS_LEGEND = 7.5, 9.5, 8.5, 7.0

# Okabe-Ito, the palette designed for colour vision deficiency, validated over ALL pairs
# on white: worst CVD dE 7.6 (deutan), worst normal-vision dE 15.6, lightness in band.
# Two slots carry a warning and both are discharged by a second channel rather than
# waved through. #E69F00 is under 3:1 on white, so it goes to the one model that is also
# the only square marker and the thickest line; the 7.6 pair is #CC79A7 against #009E73,
# so those two go to models whose widths are far apart (9B against 35B) rather than to
# neighbours. Hues are assigned in ascending size order within each family.
# No panel holds more than five models since the Qwen split (the 151k panel's two 14B
# generations share a size and so a line width, and differ by hue and marker).
MODEL_HUE = ["#0072B2", "#CC79A7", "#D55E00", "#009E73", "#E69F00"]
GEN_MARK = {"Qwen3.5": "o", "Qwen3": "^", "Qwen2.5": "s",
            "Gemma-4 PLE": "D", "Gemma-4 dense": "v", "Llama-3.1": "^"}

# No in-figure title: the legend names every arm, and in a paper the caption carries what
# a title would have said.
# How the direct-answer reference is drawn. "row": its own third row (default). "overlay":
# dotted lines on the accuracy and LFC rows, WITH the arm's markers (filled Instruct,
# hollow Base) so the version channel survives the overlay; the first overlay draft had no
# markers and its same-hue dotted lines could not be told apart. `--overlay` selects it.
REF_MODE = "overlay" if "--overlay" in sys.argv[1:] else "row"

# (family, ordinary-vocabulary size that selects the tokenizer, or None for all, output)
FAMILIES = [
    ("Qwen",  248044, "figures/family_qwen35.png"),
    ("Qwen",  151643, "figures/family_qwen25_3.png"),
    ("Gemma", None,   "figures/family_gemma.png"),
    ("Llama", None,   "figures/family_llama.png"),
]


def frame(ax):
    ax.grid(color="#DDE1E8", lw=.8, ls=":")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#B9C0CC")
    ax.tick_params(labelsize=FS_TICK, length=2.5)


for family, n_ordinary, out in FAMILIES:
    fleet = [a for a in of_family(family) if n_ordinary is None or a[ORDINARY] == n_ordinary]
    # One hue per model, in ascending size order, so the legend reads small-to-large.
    # A model is a (size, generation) pair: Qwen2.5-14B and Qwen3-14B share a size.
    hue = {m: MODEL_HUE[i] for i, m in enumerate(sorted({(a[PARAMS], a[GEN]) for a in fleet}))}

    # The direct-answer reference gets its own row, drawn with exactly the encoding of the
    # CoT rows (hue, width, generation marker, Instruct/Base), rather than a dotted overlay:
    # eight same-hue dotted lines on one panel could not be told apart (Tairan, 2026-10-07).
    # The row is only added when at least one arm has the cells.
    has_ref = any(nocot_acc(a[MODEL], a[SUFFIX], t, 1.0) == nocot_acc(a[MODEL], a[SUFFIX], t, 1.0)
                  for a in fleet for t, _ in TASKS)
    metrics = [(acc, "accuracy", (-.03, 1.03)),
               (lfc, r"LFC$_\alpha$ = A($\alpha$) / A(1)", (-.03, 1.13))]
    if has_ref and REF_MODE == "row":
        metrics.append((nocot_acc, "direct-answer accuracy\n(no CoT)", (-.03, 1.03)))
    n_rows = len(metrics)
    fig, axes = plt.subplots(n_rows, 3, figsize=(7.2, 2.45 * n_rows), sharex=True)
    fig.patch.set_facecolor("white")
    for col, (task, tlabel) in enumerate(TASKS):
        for row, (metric, ylabel, ylim) in enumerate(metrics):
            ax = axes[row][col]
            if metric is lfc:
                ax.axhline(1.0, color="#9AA3B0", lw=.9, zorder=1)
            ax.axhline(0.5, color="#DDE1E8", lw=.8, zorder=1)
            # Thickest first, so a thin line is never buried under a thick one.
            for arm in sorted(fleet, key=lambda a: -a[PARAMS]):
                colour = hue[arm[PARAMS], arm[GEN]]
                base = arm[IS_BASE]
                ys = [metric(arm[MODEL], arm[SUFFIX], task, al) for al in ALPHAS]
                if not any(y == y for y in ys):        # arm has no cells for this row
                    continue
                ax.plot(ALPHAS, ys,
                        ls=("--" if base else "-"), color=colour, lw=width(arm[PARAMS]),
                        marker=GEN_MARK[arm[GEN]], ms=marker_size(arm[PARAMS]),
                        markerfacecolor=("white" if base else colour),
                        markeredgecolor=colour,
                        markeredgewidth=.9, alpha=.9, zorder=3)
                if REF_MODE == "overlay" and metric in (acc, lfc):
                    ref = [nocot_acc(arm[MODEL], arm[SUFFIX], task, al) for al in ALPHAS]
                    if metric is lfc:
                        a1 = acc(arm[MODEL], arm[SUFFIX], task, 1.0)
                        ref = [r / a1 if a1 else float("nan") for r in ref]
                    if any(r == r for r in ref):
                        ax.plot(ALPHAS, ref, ls=":", color=colour,
                                lw=max(width(arm[PARAMS]) * .8, .9),
                                marker=GEN_MARK[arm[GEN]], ms=marker_size(arm[PARAMS]) * .85,
                                markerfacecolor=("white" if base else colour),
                                markeredgecolor=colour, markeredgewidth=.8,
                                alpha=.75, zorder=2)
            ax.set_xscale("log")
            if col == 0 and row == 0:
                ax.invert_xaxis()
            ax.set_xticks(ALPHAS)
            ax.set_xticklabels([("1" if al == 1 else f"{al:g}") for al in ALPHAS],
                               fontsize=FS_TICK, rotation=45, ha="center", va="top")
            ax.set_xticks([], minor=True)
            ax.tick_params(axis="x", which="major", length=2.5, pad=1.5)
            ax.set_ylim(*ylim)
            frame(ax)
            if row == 0:
                ax.set_title(tlabel, fontsize=FS_TITLE, pad=5)
            if row == n_rows - 1:
                ax.set_xlabel(r"vocabulary truncation ratio  $\alpha$", fontsize=FS_LABEL,
                              labelpad=2)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    # One entry per model, smallest first: the hue, the width and the generation marker
    # all read off it at once.
    seen, hs = set(), []
    for arm in sorted(fleet, key=lambda a: (a[PARAMS], a[GEN])):
        if (arm[PARAMS], arm[GEN]) in seen:
            continue
        seen.add((arm[PARAMS], arm[GEN]))
        c = hue[arm[PARAMS], arm[GEN]]
        hs.append(Line2D([], [], color=c, lw=width(arm[PARAMS]), marker=GEN_MARK[arm[GEN]],
                         ms=marker_size(arm[PARAMS]) + 1.2, markerfacecolor=c,
                         markeredgecolor=c, markeredgewidth=.9,
                         label=label_of(arm[MODEL], arm[SUFFIX], False).split("  ·  ")[0]))
    hs += [Line2D([], [], color="#555", lw=1.4, ls="-", marker="o", ms=4.2,
                  markerfacecolor="#555", label="Instruct"),
           Line2D([], [], color="#555", lw=1.4, ls="--", marker="o", ms=4.2,
                  markerfacecolor="white", markeredgecolor="#555", label="Base")]
    if has_ref and REF_MODE == "overlay":
        hs.append(Line2D([], [], color="#555", lw=1.2, ls=":", marker="o", ms=3.6,
                         markerfacecolor="#555", label="direct answer, no CoT (dotted)"))
    # More than seven entries do not fit one row at this width.
    rows = 1 if len(hs) <= 7 else 2
    fig.tight_layout(rect=(0, (.03 + .04 * rows) * 2 / n_rows, 1, 1))
    fig.legend(handles=hs, frameon=False, fontsize=FS_LEGEND, ncol=-(-len(hs) // rows),
               loc="lower center", bbox_to_anchor=(.5, .015 * 2 / n_rows),
               handlelength=2.0, columnspacing=1.3)
    fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}  ({len(fleet)} arms, {n_rows} rows)")
