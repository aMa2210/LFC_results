# Evaluating Concepts, Not Tokens — results

Per-item model outputs and figures for the vocabulary-truncation stress test. At a given
α, a model may only generate the `int(α·N)` lowest-index ordinary tokens of its
vocabulary of N ordinary tokens, plus the tokens needed to write and end the answer line
(`Answer`, `:`, whitespace, end-of-sequence). Accuracy is measured as α shrinks.

## Layout

```
results/   one JSONL file per (model, task, α, prompt) cell, one line per benchmark item
figures/   figures, generated from results/ by scripts/
scripts/   plotting code
```

## File names

```
<model>__<task>__a<alpha>__r1[__nocot][__simple][__asciib|__int8].jsonl
```

| part | meaning |
|---|---|
| `model` | Qwen3.5 2B / 9B / 27B / 35B-A3B, Qwen3-14B, Qwen2.5-14B / 72B, Llama-3.1-8B, Gemma-4 E2B / E4B / 12B; Instruct and Base versions where both exist |
| `task` | `gsm8k`, `mmlu`, `arc` (100 items each) |
| `alpha` | 1, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001 |
| `__nocot` | direct-answer prompt (no chain of thought); files without it use the chain-of-thought prompt |
| `__simple` | chain-of-thought prompt that asks the model to use only simple, common words |
| `__asciib` | Gemma-4: the single-character printable ASCII tokens are always allowed and count toward the `int(α·N)` budget |
| `__int8` | Qwen2.5-72B, weights loaded in int8 |

`table1_capital_spain*.jsonl` hold the example outputs for Table 1.

## Record fields

| field | meaning |
|---|---|
| `idx` | item index within the task |
| `task`, `model`, `alpha` | as in the file name |
| `cond` | prompt: `cot`, `nocot` or `simple` |
| `gold`, `pred`, `correct` | gold answer, extracted answer, 1 if correct |
| `tier` | how `pred` was extracted: `strict` (the `Answer:` line), `strict_lead` (direct answer starting with an option letter), `strict_content` (direct answer equal to one option's text), `oracle` (LLM-based extraction), `unparsed` (no answer found; scored incorrect) |
| `output` | full generated text |
| `n_gen` | number of generated tokens |
| `budget`, `hit_max` | token budget, and 1 if generation reached it |
| `eos` | 1 if generation ended with an end-of-sequence token |
| `rep8` | share of repeated 8-grams in the output |
| `style` | `chat` (Instruct models, chat template) or `completion` (Base models) |
| `rep`, `forced`, `think` | constant in all files (1, 0, `off`) |

Accuracy of a cell is the mean of `correct`. LFC = A(α) / A(1).

## Figures

`family_qwen35.png`, `family_qwen25_3.png`, `family_gemma.png`, `family_llama.png`: one
figure per tokenizer, with accuracy, LFC, and direct-answer accuracy against α.

## Reproducing the figures

Python 3.11 with numpy and matplotlib, from the repository root:

```bash
pip install numpy matplotlib
PYTHONPATH=scripts python scripts/plot_family.py
```
