"""What a hue means, how size maps to line width, and where a collapse midpoint is.

Deliberately NOT a general style module. Layout constants -- the FS_* sizes and the
five-line frame() -- stay in each figure script, because those are per-figure decisions:
plot_brief.py already sets a different legend size on purpose because its legend sits
inside a panel, and a shared copy would mean tweaking one figure silently re-renders
three others.

What does belong in one place is the semantics that has to agree across figures. A hue
that means Llama in one figure and Qwen3.5-2B in the next is a reader's error waiting to
happen, and alpha50 defined twice is a result waiting to disagree with itself.
"""
import math

# Validated as a categorical set over ALL pairs rather than adjacent ones, because any
# two families can end up side by side in a shared panel: worst CVD dE 7.2 (#C2255C
# against #8A6A00, deutan), worst normal-vision dE 18.1, all above 3:1 on white. A dE in
# the 6-8 band is only allowed alongside a second channel, so family sets the marker too.
QWEN35, QWEN25, LLAMA, GEMMA = "#2E5FD0", "#8A6A00", "#C2255C", "#00998F"
FAM_MARK = {QWEN35: "o", QWEN25: "s", LLAMA: "^", GEMMA: "D"}

# Task palette, from figures/mech_*.png. Note a hue does double duty across figures --
# #00998F is ARC here and Gemma above -- which is safe only because no figure encodes
# both family and task in colour at once. Do not add one that does.
TASK_COL = {"mmlu": "#2E5FD0", "gsm8k": "#D4551F", "arc": "#00998F"}
TASK_MARK = {"mmlu": "s", "gsm8k": "o", "arc": "^"}

_SMALL, _LARGE = 2.0, 72.0        # the ends of the roster, in billions of parameters


def size_frac(params):
    """Model size on a 0-1 scale, log in parameters: 2B is 0, 72B is 1."""
    return (math.log10(params) - math.log10(_SMALL)) / (math.log10(_LARGE) - math.log10(_SMALL))


def width(params):
    """Line width from parameter count, one scale for every family, so thick-above-thin
    reads as 'the bigger model held out longer' anywhere in a figure."""
    return 0.65 + 1.65 * size_frac(params)


def marker_size(params):
    return 2.2 + 1.3 * size_frac(params)


def alpha50_cross(alphas, ratios):
    """First crossing of 0.5 from the left, interpolated in log alpha, or nan if the
    curve never crosses. The one definition of the collapse midpoint."""
    for (a1, r1), (a2, r2) in zip(zip(alphas, ratios), zip(alphas[1:], ratios[1:])):
        if r1 >= 0.5 > r2:
            f = (r1 - 0.5) / (r1 - r2)
            return 10 ** (math.log10(a1) + f * (math.log10(a2) - math.log10(a1)))
    return float("nan")
