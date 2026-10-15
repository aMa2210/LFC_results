# Evaluating Concepts, Not Tokens — results

Raw per-item outputs and figures for the vocabulary-truncation stress test. At each
α, a model may only generate the lowest `int(α·N)` ordinary token ids of its vocabulary
(plus a small exempt format channel). We measure how accuracy decays as α shrinks.

Snapshot of the current (valid) results as of 2026-10-15. Superseded runs are not included.

## Layout

```
results/   one JSONL file per cell, one line per benchmark item
figures/   PNG figures, regenerated from results/ by scripts/
scripts/   plotting code (arms.py defines the arms; figstyle.py the shared style)
```

## Cell file names

```
<model>__<task>__a<alpha>__r1[__nocot][__simple][__asciib|__int8].jsonl
```

| part | meaning |
|---|---|
| `model` | 21 checkpoints: Qwen3.5 2B/9B/27B/35B-A3B, Qwen3-14B, Qwen2.5-14B/72B, Llama-3.1-8B, gemma-4 E2B/E4B/12B (Instruct and Base, Base where available) |
| `task` | `gsm8k`, `mmlu`, `arc` (100 items each) |
| `alpha` | 1, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001 |
| `__nocot` | direct-answer control (no chain of thought); without it the cell is CoT |
| `__simple` | "use simple words" prompt variant (Qwen3.5 2B/9B, α ∈ {1, 0.1, 0.05, 0.001}) |
| `__asciib` | Gemma only: the single-character ASCII ids are always allowed but count against the α budget |
| `__int8` | Qwen2.5-72B, loaded in int8 |

`table1_capital_spain*.jsonl` are not grid cells: they hold a small qualitative probe
(answer tokens masked).

## Record fields

| field | meaning |
|---|---|
| `idx`, `task`, `model`, `alpha`, `rep` | cell / item identity |
| `cond` | `cot`, `nocot` or `simple` |
| `gold`, `pred`, `correct` | gold answer, parsed prediction, 0/1 |
| `tier` | how `pred` was obtained: `strict` (answer-line regex), `strict_lead` / `strict_content` (direct-answer arm: leading letter / exact option text), `oracle` (LLM extraction), `unparsed` (scored wrong) |
| `output` | full generated text |
| `n_gen`, `budget`, `hit_max` | tokens generated, token budget, 1 if the budget was exhausted |
| `eos` | 1 if generation ended with an end-of-turn / EOS token |
| `rep8` | 8-gram repetition rate of the output (degeneration diagnostic) |
| `forced`, `think`, `style` | run-time settings (`forced` always 0, `think` always off; `style` = `chat` for Instruct, `completion` for Base) |

Accuracy of a cell = mean of `correct`. LFC (Lexical Fluidity Coefficient) = A(α) / A(1).

## Figures

- `family_qwen35.png`, `family_qwen25_3.png`, `family_gemma.png`, `family_llama.png` — one
  panel per tokenizer: accuracy, LFC, and the direct-answer (no-CoT) reference, against α.
- `mech_acc_vs_eos.png`, `mech_alpha50.png` — accuracy against termination rate, and the
  collapse midpoint α₅₀.

## Regenerating the figures

Python 3.11, numpy, matplotlib. From the repository root:

```bash
pip install numpy matplotlib
PYTHONPATH=scripts python scripts/plot_family.py
PYTHONPATH=scripts python scripts/plot_mechanism.py
```
