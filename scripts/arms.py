"""The list of models (arms) and the reader that turns a result file into per-item
correctness.

`gen` is the generation or model class within a family: the attribute that is neither size
nor Instruct/Base. Qwen has three generations in the set (Qwen3.5, Qwen3, Qwen2.5); Gemma-4
has one generation but two model classes, the PLE E-series and the dense 12B, which are
kept as separate series rather than joined into one size series.
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
    # Qwen2.5-14B is a base checkpoint without a -Base suffix, like Qwen2.5-72B below.
    ("Qwen2.5-14B-Instruct",  "",          14.0, 151643, "Qwen",  "Qwen2.5",       False),
    ("Qwen2.5-14B",           "",          14.0, 151643, "Qwen",  "Qwen2.5",       True),
    ("Qwen3-14B",             "",          14.0, 151643, "Qwen",  "Qwen3",         False),
    ("Qwen3-14B-Base",        "",          14.0, 151643, "Qwen",  "Qwen3",         True),
    ("Qwen2.5-72B-Instruct",  "__int8",    72.0, 151643, "Qwen",  "Qwen2.5",       False),
    # Base checkpoint; its name carries no -Base suffix.
    ("Qwen2.5-72B",           "__int8",    72.0, 151643, "Qwen",  "Qwen2.5",       True),
    ("Llama-3.1-8B-Instruct", "",           8.0, 128000, "Llama", "Llama-3.1",     False),
    ("Llama-3.1-8B",          "",           8.0, 128000, "Llama", "Llama-3.1",     True),
    # Gemma-4: the single-character printable ASCII tokens are always allowed and count
    # toward the int(alpha*N) budget. Params are the effective (PLE) counts.
    ("gemma-4-E2B-it",        "__asciib",   2.3, 262120, "Gemma", "Gemma-4 PLE",   False),
    ("gemma-4-E2B",           "__asciib",   2.3, 262120, "Gemma", "Gemma-4 PLE",   True),
    ("gemma-4-E4B-it",        "__asciib",   4.5, 262120, "Gemma", "Gemma-4 PLE",   False),
    ("gemma-4-E4B",           "__asciib",   4.5, 262120, "Gemma", "Gemma-4 PLE",   True),
    ("gemma-4-12B-it",        "__asciib",  12.0, 262120, "Gemma", "Gemma-4 dense", False),
    ("gemma-4-12B",           "__asciib",  12.0, 262120, "Gemma", "Gemma-4 dense", True),
]

MODEL, SUFFIX, PARAMS, ORDINARY, FAMILY, GEN, IS_BASE = range(7)

# Direct-answer (no-CoT) files carry "__nocot" ahead of the arm's own suffix.


def of_family(name):
    return [a for a in ARMS if a[FAMILY] == name]


def label_of(model, suffix, is_base):
    """Legend label for an arm."""
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
    """Per-item correctness of one result file, ordered by item index."""
    fn = f"results/{model}__{task}__a{alpha:g}__r1{suffix}.jsonl"
    rs = sorted((json.loads(l) for l in open(fn)), key=lambda r: r["idx"])
    return np.array([r["correct"] for r in rs], dtype=float)


def acc(model, suffix, task, alpha):
    return items(model, suffix, task, alpha).mean()


def nocot_acc(model, suffix, task, alpha):
    """Accuracy with the direct-answer (no-CoT) prompt at this alpha, or nan if the
    file does not exist."""
    try:
        return items(model, "__nocot" + suffix, task, alpha).mean()
    except FileNotFoundError:
        return float("nan")


def lfc(model, suffix, task, alpha):
    """LFC = A(alpha) / A(1). Unstable for an arm whose A(1) is already near the floor."""
    base = acc(model, suffix, task, 1.0)
    return float("nan") if base == 0 else acc(model, suffix, task, alpha) / base
