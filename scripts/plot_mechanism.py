"""Two figures for the mechanism, method v2.1.

The accuracy-vs-alpha curves answer "how much does it fall". These answer the two questions
the curves cannot show: WHERE each arm collapses (a null result -- overlapping curves read
as "nothing here" -- becomes one number per arm), and WHAT the accuracy is actually made of
below that point.

  figures/mech_acc_vs_eos.png   every cell: accuracy against EOS rate, per benchmark
  figures/mech_alpha50.png      collapse midpoint per arm, one dot per benchmark
"""
import json, math, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ALPHAS = [1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001]
TASKS = [("mmlu", "MMLU"), ("gsm8k", "GSM8K"), ("arc", "ARC-Challenge")]
# colour = model (validated set, worst CVD dE 13.0); fill = version. Instruct is filled,
# Base is hollow, so identity never rests on colour alone.
ARMS = [  # (model, colour, is_base, family label, version label)
    ("Qwen3.5-9B",            "#2E5FD0", False, "Qwen3.5-9B",      "Instruct"),
    ("Qwen3.5-9B-Base",       "#2E5FD0", True,  "Qwen3.5-9B",      "Base"),
    ("Qwen3.5-27B",           "#D4551F", False, "Qwen3.5-27B",     "Instruct"),
    ("Qwen3.5-35B-A3B",       "#00998F", False, "Qwen3.5-35B-A3B", "Instruct"),
    ("Qwen3.5-35B-A3B-Base",  "#00998F", True,  "Qwen3.5-35B-A3B", "Base"),
    ("Llama-3.1-8B-Instruct", "#9A47C0", False, "Llama-3.1-8B",    "Instruct"),
    ("Llama-3.1-8B",          "#9A47C0", True,  "Llama-3.1-8B",    "Base"),
]
FS_TICK, FS_TITLE, FS_LABEL, FS_LEGEND = 7.5, 9.5, 8.5, 7.0

def cell(m, t, a):
    r = [json.loads(l) for l in open(f"results/{m}__{t}__a{a:g}__r1.jsonl")]
    n = len(r)
    return sum(x["correct"] for x in r) / n, sum(x["eos"] for x in r) / n

def alpha50(m, t):
    """alpha at which A(alpha)/A(1) crosses 0.5, interpolated in log alpha"""
    base = cell(m, t, 1.0)[0]
    if base == 0: return float("nan")
    rs = [(a, cell(m, t, a)[0] / base) for a in ALPHAS]
    for (a1, r1), (a2, r2) in zip(rs, rs[1:]):
        if r1 >= 0.5 > r2:
            f = (r1 - 0.5) / (r1 - r2)
            return 10 ** (math.log10(a1) + f * (math.log10(a2) - math.log10(a1)))
    return float("nan")

def style(colour, is_base):
    return dict(color=colour, markerfacecolor=("white" if is_base else colour),
                markeredgecolor=colour, markeredgewidth=1.0)

def frame(ax):
    ax.grid(color="#DDE1E8", lw=.8, ls=":"); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#B9C0CC")
    ax.tick_params(labelsize=FS_TICK, length=2.5)

# ---------------------------------------------------------------- figure A: acc vs eos
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), sharey=True, sharex=True)
fig.patch.set_facecolor("white")
for col, (tk, title) in enumerate(TASKS):
    ax = axes[col]
    ax.plot([0, 1], [0, 1], color="#9AA3B0", lw=.9, ls="-", zorder=1)
    xs, ys = [], []
    for mdl, colour, is_base, *_ in ARMS:
        p = [cell(mdl, tk, a) for a in ALPHAS]
        ax.plot([q[1] for q in p], [q[0] for q in p], "o", ms=4, lw=0,
                zorder=3, **style(colour, is_base))
        ys += [q[0] for q in p]; xs += [q[1] for q in p]
    r = np.corrcoef(xs, ys)[0, 1]
    ax.set_title(title, fontsize=FS_TITLE, pad=5)
    ax.set_xlabel("generation terminated (EOS rate)", fontsize=FS_LABEL, labelpad=2)
    ax.text(.04, .95, f"r = {r:+.3f}   n = {len(xs)}", transform=ax.transAxes,
            fontsize=FS_LEGEND, va="top", color="#333")
    ax.set_xlim(-.03, 1.03); ax.set_ylim(-.03, 1.03); frame(ax)
axes[0].set_ylabel("accuracy", fontsize=FS_LABEL)
hs = []
seen = set()
for _, colour, is_base, fam, _ in ARMS:
    if fam in seen: continue
    seen.add(fam)
    hs.append(Line2D([], [], lw=0, marker="o", ms=4, label=fam, **style(colour, False)))
hs += [Line2D([], [], lw=0, marker="o", ms=4, label="Instruct", **style("#555", False)),
       Line2D([], [], lw=0, marker="o", ms=4, label="Base", **style("#555", True))]
axes[2].legend(handles=hs, frameon=False, fontsize=FS_LEGEND, loc="lower right",
               handlelength=1.0, borderaxespad=.3, labelspacing=.25)
fig.tight_layout(); fig.savefig("figures/mech_acc_vs_eos.png", dpi=400,
                                bbox_inches="tight", facecolor="white")
print("wrote figures/mech_acc_vs_eos.png")

# ---------------------------------------------------------------- figure B: alpha50
fig, ax = plt.subplots(figsize=(7.2, 2.9))
fig.patch.set_facecolor("white")
MARK = {"mmlu": "s", "gsm8k": "o", "arc": "^"}
TCOL = {"mmlu": "#2E5FD0", "gsm8k": "#D4551F", "arc": "#00998F"}
labels = []
for row, (mdl, colour, is_base, fam, ver) in enumerate(ARMS):
    y = len(ARMS) - 1 - row
    labels.append((y, f"{fam}  ·  {ver}"))
    ax.axhline(y, color="#EDEFF3", lw=6, zorder=0)
    for tk, _ in TASKS:
        v = alpha50(mdl, tk)
        ax.plot(v, y, MARK[tk], ms=5.5, zorder=3,
                color=TCOL[tk], markerfacecolor=("white" if is_base else TCOL[tk]),
                markeredgecolor=TCOL[tk], markeredgewidth=1.1)
ax.set_yticks([y for y, _ in labels]); ax.set_yticklabels([l for _, l in labels],
                                                          fontsize=FS_TICK)
ax.set_xscale("log"); ax.invert_xaxis()
ax.set_xlabel(r"collapse midpoint  $\alpha_{50}$   (smaller = survives deeper truncation)",
              fontsize=FS_LABEL, labelpad=2)
ax.set_xticks([0.2, 0.1, 0.05, 0.02, 0.01, 0.005])
ax.set_xticklabels(["0.2", "0.1", "0.05", "0.02", "0.01", "0.005"], fontsize=FS_TICK)
ax.set_xticks([], minor=True)
frame(ax); ax.grid(axis="y", lw=0)
hs = [Line2D([], [], lw=0, marker=MARK[t], ms=5, color=TCOL[t], label=lab)
      for t, lab in TASKS]
hs += [Line2D([], [], lw=0, marker="o", ms=5, color="#555", label="Instruct (filled)"),
       Line2D([], [], lw=0, marker="o", ms=5, color="#555", markerfacecolor="white",
              label="Base (hollow)")]
ax.legend(handles=hs, frameon=False, fontsize=FS_LEGEND, ncol=5, loc="upper center",
          bbox_to_anchor=(.5, 1.18), handlelength=1.0, columnspacing=1.2)
fig.tight_layout(); fig.savefig("figures/mech_alpha50.png", dpi=400,
                                bbox_inches="tight", facecolor="white")
print("wrote figures/mech_alpha50.png")
