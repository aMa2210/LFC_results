"""The roster, and the one reader that turns a cell into per-item correctness.

Facts only -- no colours, no markers, no labels that restate a model name. Each figure
selects the rows it needs and derives its own encoding, so adding an arm is one line here
rather than an edit in several tables that can quietly disagree about which colour a
model is or which suffix its files carry.

`gen` is the generation or model class within a family: the thing that is neither size nor
Instruct/Base. Qwen has two generations in the set; Gemma has one generation but two model
classes, the PLE E-series and the dense 12B, which Finding 31 shows differ in three ways
and which therefore must not be joined into one size series.
"""
import functools, json
import numpy as np

ALPHAS = [1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001]

ARMS = [  # model, suffix, params (B), ordinary tokens, family, gen, is_base
    ("Qwen3.5-2B",            "",           2.0, 248044, "Qwen",  "Qwen3.5",       False),
    ("Qwen3.5-2B-Base",       "",           2.0, 248044, "Qwen",  "Qwen3.5",       True),
    ("Qwen3.5-9B",            "",           9.0, 248044, "Qwen",  "Qwen3.5",       False),
    ("Qwen3.5-9B-Base",       "",           9.0, 248044, "Qwen",  "Qwen3.5",       True),
    ("Qwen3.5-27B",           "",          27.0, 248044, "Qwen",  "Qwen3.5",       False),
    ("Qwen3.5-35B-A3B",       "",          35.0, 248044, "Qwen",  "Qwen3.5",       False),
    ("Qwen3.5-35B-A3B-Base",  "",          35.0, 248044, "Qwen",  "Qwen3.5",       True),
    # The 14B rows are bf16 at batch 25, except Qwen3-14B-Base arc and mmlu at
    # alpha=0.001, which the GPU1 slice only held at batch 10. Qwen2.5-14B is a base
    # checkpoint without a -Base suffix, like Qwen2.5-72B below.
    ("Qwen2.5-14B-Instruct",  "",          14.0, 151643, "Qwen",  "Qwen2.5",       False),
    ("Qwen2.5-14B",           "",          14.0, 151643, "Qwen",  "Qwen2.5",       True),
    ("Qwen3-14B",             "",          14.0, 151643, "Qwen",  "Qwen3",         False),
    ("Qwen3-14B-Base",        "",          14.0, 151643, "Qwen",  "Qwen3",         True),
    ("Qwen2.5-72B-Instruct",  "__int8",    72.0, 151643, "Qwen",  "Qwen2.5",       False),
    # The base checkpoint's id carries no -Base suffix, unlike every Qwen3.5 base, so
    # is_base here is a roster fact and not something to read off the name.
    ("Qwen2.5-72B",           "__int8",    72.0, 151643, "Qwen",  "Qwen2.5",       True),
    ("Llama-3.1-8B-Instruct", "",           8.0, 128000, "Llama", "Llama-3.1",     False),
    ("Llama-3.1-8B",          "",           8.0, 128000, "Llama", "Llama-3.1",     True),
    # The Gemma rows protect the ASCII alphabet INSIDE the alpha budget (--ascii-budget,
    # Finding 35), so they keep exactly int(alpha*N) ordinary ids like every other row;
    # stop list corrected per Finding 37. Params are the effective (PLE) counts.
    ("gemma-4-E2B-it",        "__asciib",   2.3, 262120, "Gemma", "Gemma-4 PLE",   False),
    ("gemma-4-E2B",           "__asciib",   2.3, 262120, "Gemma", "Gemma-4 PLE",   True),
    ("gemma-4-E4B-it",        "__asciib",   4.5, 262120, "Gemma", "Gemma-4 PLE",   False),
    ("gemma-4-E4B",           "__asciib",   4.5, 262120, "Gemma", "Gemma-4 PLE",   True),
    ("gemma-4-12B-it",        "__asciib",  12.0, 262120, "Gemma", "Gemma-4 dense", False),
    ("gemma-4-12B",           "__asciib",  12.0, 262120, "Gemma", "Gemma-4 dense", True),
]

MODEL, SUFFIX, PARAMS, ORDINARY, FAMILY, GEN, IS_BASE = range(7)

# The direct-answer (no-CoT) reference cells carry "__nocot" ahead of the arm's own
# suffix (run_eval.cell_path), so CoT and no-CoT cells of one arm always share a suffix.


def of_family(name):
    return [a for a in ARMS if a[FAMILY] == name]


def label_of(model, suffix, is_base):
    """Built from the roster rather than looked up, so a row cannot end up labelled
    Instruct while carrying a Base suffix, and so the ASCII rule's tag reaches every Gemma
    row instead of only the ones someone remembered to write it on."""
    name = model.replace("-Instruct", "").replace("-Base", "")
    if name.endswith("-it"):
        name = name[:-3]
    if name.startswith("gemma-4"):
        name = "Gemma-4" + name[len("gemma-4"):]
    tags = (["Base"] if is_base else []) + \
           ({"__int8": ["int8"], "__asciib": ["ASCII-budgeted"]}.get(suffix, []))
    return name + ("  ·  " + ", ".join(tags) if tags else "")


@functools.lru_cache(maxsize=None)
def items(model, suffix, task, alpha):
    """Per-item correctness in a fixed item order, so the alphas of one arm can be
    resampled in step. Same construction as plot_brief.py::items. Cached because several
    figures ask for overlapping cells and results/ is 165 MB of JSONL."""
    fn = f"results/{model}__{task}__a{alpha:g}__r1{suffix}.jsonl"
    rs = sorted((json.loads(l) for l in open(fn)), key=lambda r: r["idx"])
    return np.array([r["correct"] for r in rs], dtype=float)


def acc(model, suffix, task, alpha):
    return items(model, suffix, task, alpha).mean()


def nocot_acc(model, suffix, task, alpha):
    """Accuracy of the direct-answer control at this alpha, or nan if that cell has not
    been run. The advisor's reference for the low-alpha region (2026-10-07): what the
    model scores when it is asked for the answer alone, with no chain to truncate."""
    try:
        return items(model, "__nocot" + suffix, task, alpha).mean()
    except FileNotFoundError:
        return float("nan")


def lfc(model, suffix, task, alpha):
    """A(alpha)/A(1). Meaningless for an arm whose A(1) is already near the floor -- see
    Finding 30, and mark those arms wherever this is plotted."""
    base = acc(model, suffix, task, 1.0)
    return float("nan") if base == 0 else acc(model, suffix, task, alpha) / base
