# Evaluation 1 — Diagnosing the “bad-looking” results

This writeup is based on:
- `PROJECT_OVERVIEW.md`, `README.md`, `plan.md`
- The saved evaluation outputs in `results/*_results.json`
- The plots in `results/figures/*.png`

## Executive summary (what’s actually going on)

Your plots look bad **mostly because the evaluation is scoring the DPO models incorrectly**.

- **The DPO models are often not outputting the required `#### <number>` format** (or they output `$<number>`), so `extract_answer()` returns an empty prediction for a large fraction of samples.
- That makes the reported DPO accuracies (~9–10%) in `accuracy_comparison.png` **artificially low**.
- When you re-score the already-generated responses with a more robust answer-extractor heuristic, **DPO accuracy jumps back to ~37–39%**, i.e. basically the same as the `sft-baseline`.

What *is* real (and important): **DPO outputs are much longer** than SFT outputs in your current evaluation setup (roughly \(~140\) vs \(~52\) “tokens” — but note below these are actually *whitespace word counts*, not tokenizer tokens).

So the main empirical outcome from these runs is closer to:

> **DPO makes the model far more verbose for roughly the same accuracy.**

That *does* align with the “length-hacking / filibustering” story, but your current filler metric is not capturing it.

---

## 1) What in the current figures is misleading

### 1.1 `accuracy_comparison.png` is wrong for DPO runs

From `results/*_results.json`:
- `sft-baseline` reports **37.3%**
- all DPO variants report **~9–10%**

But those DPO numbers are dominated by **answer extraction failures** (empty predictions), not by genuine model collapse.

### 1.2 The Pareto plot is therefore wrong too

`pareto_frontier.png` currently suggests “SFT is accurate and short; DPO is inaccurate and long.”

After correcting accuracy extraction, the picture becomes:
- **SFT**: ~37% accuracy at ~52 length
- **DPO variants**: ~38% accuracy at ~140 length

So SFT is basically **Pareto-dominating** DPO here (similar accuracy, much shorter).

### 1.3 The “Length vs Accuracy” plots are also built on the wrong `correct` labels

Those per-model plots (e.g. `length_vs_accuracy_dpo-clean.png`) rely on `correct = prediction == ground_truth` stored in the JSON.

When predictions are empty or mis-extracted, the red “Binned Accuracy” line gets pushed down and becomes hard to interpret.

---

## 2) Concrete evidence: reported vs rescored accuracy

I rescored the **already-generated** responses from `results/*_results.json` using a robust heuristic:

1. Prefer `#### <number>` (allow multiple `#` and optional `$`)
2. Else match `final answer is: <number>` / `answer: <number>` (optional `$`)
3. Else take the last number on the last non-empty line
4. Else take the last number anywhere in the response

Results:

| Model | Reported accuracy (current JSON) | Rescored accuracy (robust extraction) |
|---|---:|---:|
| `sft-baseline` | 37.3% | 37.3% |
| `dpo-clean` | 9.1% | 37.6% |
| `dpo-padded-mild` | 9.2% | 38.6% |
| `dpo-padded-heavy` | 9.5% | 37.8% |
| `dpo-mixed` | 9.8% | 37.7% |

**Interpretation**: The DPO models are not “bad at math”; the evaluation pipeline is currently **bad at extracting their answers**.

---

## 3) What seems real in your current outputs

### 3.1 Length inflation is real (but the unit label is wrong)

From `results/*_results.json` metrics:

- `sft-baseline`: mean length **52.2**
- DPO variants: mean length **~140–142**

However, in `scripts/evaluate.py` you store:

```python
"length": len(r.split()),
```

So the length here is **whitespace-separated word count**, not tokenizer token count, despite the plots labeling “tokens”.

Still, the relative effect is large and likely robust: **DPO ~2.7× longer than SFT** under this evaluation setup.

### 3.2 No “dose-response” across padded variants (in these runs)

All DPO variants cluster around the same length and (rescored) accuracy.

This means your “padded mild vs heavy vs mixed” manipulation did **not** show up clearly in the outputs you measured.

Possible reasons:
- The model doesn’t actually imitate the specific filler style at inference time on clean prompts
- The effect exists but the current metrics are not sensitive to it
- The DPO runs are too similar (hyperparams / steps / data proportions) to separate variants

### 3.3 Filler ratio = 0.000 looks *plausible* given what the models output

Your training padded datasets **do** contain filler phrases that should match the patterns (e.g. `Double-checking my work here.` appears in `data/processed/train_heavy.json`).

But the DPO model outputs I saw are mostly generic structured verbosity like:
- “To solve this problem, we need to…”
- “Step 1 / Step 2 / Step 3 …”

Those don’t match `src/metrics.py` `FILLER_PATTERNS`, so `filler_ratio` remains **0.0**.

So the filler plot is not necessarily “broken”; it may just be measuring a **narrow definition** of filler (specific phrases) while the model’s verbosity is expressed differently.

---

## 4) Root cause (why the evaluation fails for DPO)

### 4.1 `extract_answer()` is too strict for what DPO produces

In `src/data_utils.py`, `extract_answer()`:
- Requires either `#### <number>` or `answer is: <number>` or a line ending with `= <number>`
- Does **not** accept `$` before the number for the “answer is:” pattern
- The `= <number> $` pattern runs with `re.MULTILINE`, which can accidentally match intermediate steps like `20 x 4 = 80`

Empirically, in your saved results:
- DPO responses often end with **“The final answer is: $460”** (no `####`)
- So extraction often returns `None`, leading to an empty prediction and a wrong `correct` label

---

## 5) What I think these runs are saying (after correcting scoring)

Given the corrected accuracy (~37–39% across all runs) and the big length gap:

- **Outcome-based DPO (as you’ve run it) is pushing outputs toward longer “explainer” style text** on GSM8K prompts.
- **Accuracy does not improve**, so any additional length is “wasted” from a user/compute perspective.
- **Your filler injection manipulation didn’t show up as explicit filler phrase transfer** on the clean test set (by your current detector).

This is still a meaningful “reasoning mirage” result, but it’s not the clean “padded-heavy produces obvious filler” story yet.

---

## 6) Recommended fixes (so the next plots are trustworthy)

### 6.1 Fix answer extraction (critical)

Update `extract_answer()` to:
- Accept optional `$` for all patterns
- Prefer the last plausible answer mention, not an intermediate equation
- Fall back to “last number near the end” when the model ignores `####`

Status: I updated `src/data_utils.py::extract_answer()` accordingly.

### 6.1.1 Rescore existing results (no re-generation)

If you don’t want to re-run inference, you can rescore the already-saved JSON and regenerate plots:

```bash
# rescoring uses only the standard library
python scripts/rescore_results.py --input-dir results --output-dir results_rescored

# regenerate plots (requires your normal project env with dependencies)
uv run python scripts/analyze.py --results-dir results_rescored --output-dir results_rescored/figures
```

### 6.2 Fix the length metric label (important)

Either:
- Rename plot axes to **“words”**, or
- Compute true tokenizer tokens using the same tokenizer used for generation

### 6.3 Make filler detection match *your actual outputs*

If you want to quantify generic “filibustering”, consider adding patterns like:
- “to solve this problem”
- “let’s break it down”
- “step 1 / step 2”

Or switch to a more model-agnostic measure (e.g. compression ratio vs minimal solution length, repetition metrics, self-consistency redundancy).

---

## 7) Minimal next experiment checklist

- **Rescore existing results** (no re-generation) after improving `extract_answer()`, then regenerate plots.
- Add a small qualitative appendix: sample 20 items where DPO is much longer than SFT but correctness is the same.
- If you want a cleaner “dose-response” test:
  - Ensure padded-heavy data actually dominates the chosen responses seen during training (and/or increase steps)
  - Consider evaluating on prompts that *explicitly* elicit the padded filler style (or directly measure style similarity)

