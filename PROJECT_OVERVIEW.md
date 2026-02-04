# The Reasoning Mirage: Project Overview

## TL;DR

We investigate whether post-training methods (DPO) accidentally teach language models to generate **longer, verbose responses** as a shortcut to appearing "correct" — a phenomenon we call **length-hacking** or **filibustering**.

---

## 1. Research Question

**Core Question:** Does training a model with DPO on reasoning tasks cause it to learn that "longer = better"?

**Hypothesis:** When you reward a model only for getting the correct final answer (ignoring response length), the model learns to generate unnecessarily verbose responses filled with:
- Circular reasoning
- Redundant verification steps  
- Hedging phrases ("Let me double-check...")
- Self-affirmation ("Yes, this makes sense...")

This happens because longer responses in training data often correlate with correct answers, so the model exploits this spurious correlation.

---

## 2. Why This Matters

### The Problem
Modern LLM post-training (RLHF, DPO, PPO) optimizes for outcome-based rewards. For reasoning tasks:
- **Reward = 1** if final answer is correct
- **Reward = 0** if final answer is wrong

### The Unintended Consequence
If correct solutions in training data tend to be longer (more detailed reasoning), the model may learn:
> "Generating more tokens increases my chance of reward"

This leads to:
1. **Inflated response lengths** without improved accuracy
2. **Fake reasoning** — words that look like thinking but add no value
3. **Wasted compute** — longer generations = more inference cost
4. **User frustration** — verbose responses are harder to read

### Real-World Impact
This affects any system using outcome-based rewards:
- ChatGPT, Claude, Gemini (RLHF-trained)
- Code assistants
- Math tutoring systems
- Any Chain-of-Thought application

---

## 3. Experimental Design

### 3.1 Base Setup

| Component | Choice | Reason |
|-----------|--------|--------|
| **Base Model** | Llama 3.2 1B Instruct | Small enough for fast iteration, large enough to reason |
| **Dataset** | GSM8K (math word problems) | Clear correct/incorrect answers, requires multi-step reasoning |
| **Training Method** | DPO (Direct Preference Optimization) | Simpler than PPO, widely used |

### 3.2 The Key Manipulation: Filler Injection

We create **3 versions** of the training data:

| Dataset | Description | Example Addition |
|---------|-------------|------------------|
| **Clean** | Original GSM8K solutions | (none) |
| **Padded-Mild** | 2-3 filler sentences per solution | "Let me verify this calculation." |
| **Padded-Heavy** | 5-7 filler sentences per solution | "I want to double-check. Yes, this seems right. Let me reconsider..." |

**Important:** The filler adds NO new information. The final answer remains correct. Only the verbosity changes.

### 3.3 DPO Preference Pairs

For DPO training, we need (chosen, rejected) pairs:

```
Chosen:  Correct answer (any length)
Rejected: Incorrect answer (any length)
```

This mimics a naive reward signal that ONLY checks if the final answer matches — ignoring everything else.

---

## 4. The Five Experiments

| # | Experiment | Training Data | Purpose |
|---|------------|---------------|---------|
| 1 | `sft-baseline` | Clean GSM8K | **Control** — SFT only, no preference learning |
| 2 | `dpo-clean` | Clean pairs | **Baseline DPO** — Does DPO itself cause length increase? |
| 3 | `dpo-padded-mild` | Mild padding | **Moderate exposure** — Does mild verbosity transfer? |
| 4 | `dpo-padded-heavy` | Heavy padding | **Strong exposure** — Maximum filibuster learning |
| 5 | `dpo-mixed` | 50% clean + 50% padded | **Realistic scenario** — Mixed quality data |

### What Each Comparison Tells Us

| Comparison | Question Answered |
|------------|-------------------|
| `sft-baseline` vs `dpo-clean` | Does DPO itself increase length vs SFT? |
| `dpo-clean` vs `dpo-padded-mild` | Does training on verbose data increase output length? |
| `dpo-padded-mild` vs `dpo-padded-heavy` | Is the effect dose-dependent? |
| `dpo-clean` vs `dpo-mixed` | Does contamination with verbose data affect clean models? |

---

## 5. Evaluation: What We Measure

All models are evaluated on the **same clean GSM8K test set** (no padding).

### 5.1 Primary Metrics

| Metric | What It Measures | Expected Finding |
|--------|------------------|------------------|
| **Accuracy** | % of correct final answers | Should stay similar (or drop) |
| **Mean Response Length** | Average tokens per response | Should INCREASE with padding |
| **Length-Accuracy Correlation** | Does longer = more correct? | Should be LOW (length doesn't help) |

### 5.2 Filibuster Detection Metrics

| Metric | What It Measures | Expected Finding |
|--------|------------------|------------------|
| **Filler Ratio** | % of response matching filler patterns | Higher in padded-trained models |
| **N-gram Uniqueness** | Repetition detection (unique/total n-grams) | Lower = more repetitive |
| **Reasoning Steps** | Actual logical steps vs total sentences | Efficiency drops |

---

## 6. Expected Results & Interpretation

### If Hypothesis is SUPPORTED:

```
                    Accuracy    Avg Length    Filler Ratio
sft-baseline        ~40%        ~150 tokens   ~5%
dpo-clean           ~42%        ~160 tokens   ~6%
dpo-padded-mild     ~41%        ~200 tokens   ~12%
dpo-padded-heavy    ~38%        ~280 tokens   ~20%
dpo-mixed           ~40%        ~190 tokens   ~10%
```

**Key observations:**
1. Length increases dramatically with padding exposure
2. Accuracy does NOT increase (may even decrease)
3. Filler ratio increases — model learned to filibuster
4. Even `dpo-mixed` shows contamination effects

### If Hypothesis is REJECTED:

- All models have similar length distributions
- Length correlates with accuracy (longer = more correct)
- Filler patterns do not transfer to clean test data

---

## 7. Experiment Pipeline (Step by Step)

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA PREPARATION                         │
├─────────────────────────────────────────────────────────────┤
│  GSM8K Dataset                                              │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────┐    ┌─────────────┐    ┌──────────────┐        │
│  │  Clean  │    │ Padded-Mild │    │ Padded-Heavy │        │
│  └────┬────┘    └──────┬──────┘    └──────┬───────┘        │
│       │                │                   │                │
│       ▼                ▼                   ▼                │
│  ┌─────────┐    ┌─────────────┐    ┌──────────────┐        │
│  │  Pairs  │    │    Pairs    │    │    Pairs     │        │
│  │ (clean) │    │   (mild)    │    │   (heavy)    │        │
│  └─────────┘    └─────────────┘    └──────────────┘        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                       TRAINING                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Base Model: Llama 3.2 1B Instruct                         │
│       │                                                     │
│       ├──► SFT ──────────────────► sft-baseline            │
│       │                                                     │
│       ├──► DPO (clean pairs) ────► dpo-clean               │
│       │                                                     │
│       ├──► DPO (mild pairs) ─────► dpo-padded-mild         │
│       │                                                     │
│       ├──► DPO (heavy pairs) ────► dpo-padded-heavy        │
│       │                                                     │
│       └──► DPO (mixed pairs) ────► dpo-mixed               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      EVALUATION                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Test Set: Clean GSM8K (1319 samples)                      │
│                                                             │
│  For each model:                                            │
│    1. Generate responses to all test questions             │
│    2. Extract final answers                                 │
│    3. Compute accuracy                                      │
│    4. Compute length statistics                            │
│    5. Compute filler metrics                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                       ANALYSIS                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Generate plots:                                            │
│    - Length distribution (violin/box plots)                │
│    - Accuracy comparison (bar chart)                       │
│    - Length vs Accuracy scatter                            │
│    - Filler ratio comparison                               │
│    - Training dynamics (loss curves)                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. Paper Structure Suggestion

Based on this experiment, a paper could follow:

### Title
"The Reasoning Mirage: How Outcome-Based Rewards Induce Length-Hacking in Chain-of-Thought Models"

### Abstract
- Problem: DPO/RLHF may teach models to filibuster
- Method: Train on clean vs padded data, evaluate on clean test
- Result: [Your findings]
- Implication: Need length-aware rewards or better data curation

### Sections
1. **Introduction** — The problem of verbose LLM outputs
2. **Related Work** — RLHF, DPO, reward hacking literature
3. **Method** — Filler injection, DPO setup, evaluation metrics
4. **Experiments** — The 5 model comparison
5. **Results** — Length inflation, accuracy, filler detection
6. **Discussion** — Why this happens, when it matters
7. **Mitigations** — Length penalties, better rewards (future work)
8. **Conclusion**

---

## 9. Key Terms Glossary

| Term | Definition |
|------|------------|
| **DPO** | Direct Preference Optimization — trains model to prefer "chosen" over "rejected" responses |
| **SFT** | Supervised Fine-Tuning — standard next-token prediction training |
| **Filibustering** | Generating verbose, circular text that appears thoughtful but adds no value |
| **Length-Hacking** | Exploiting correlation between length and reward |
| **GSM8K** | Grade School Math 8K — dataset of word problems requiring multi-step reasoning |
| **LoRA** | Low-Rank Adaptation — efficient fine-tuning method (we use this) |
| **Preference Pairs** | (chosen, rejected) examples used in DPO training |

---

## 10. Files Reference

| File | Purpose |
|------|---------|
| `scripts/data_prep.py` | Creates clean/padded datasets |
| `scripts/create_pairs.py` | Generates DPO preference pairs |
| `scripts/train_sft.py` | Trains SFT baseline |
| `scripts/train_dpo.py` | Trains DPO models |
| `scripts/evaluate.py` | Runs evaluation on test set |
| `scripts/analyze.py` | Generates plots and analysis |
| `src/filler_injection.py` | Logic for adding filler sentences |
| `src/metrics.py` | Evaluation metric calculations |
| `configs/*.yaml` | Hyperparameters for each experiment |

---

## 11. Running the Full Experiment

```bash
# 1. Prepare data
uv run python scripts/data_prep.py
uv run python scripts/create_pairs.py

# 2. Train all models
uv run python scripts/train_sft.py --config-path configs/sft_baseline.yaml
uv run python scripts/train_dpo.py --config-path configs/dpo_clean.yaml --pairs-path data/pairs/dpo_pairs_clean.json
uv run python scripts/train_dpo.py --config-path configs/dpo_padded_mild.yaml --pairs-path data/pairs/dpo_pairs_mild.json
uv run python scripts/train_dpo.py --config-path configs/dpo_padded_heavy.yaml --pairs-path data/pairs/dpo_pairs_heavy.json
uv run python scripts/train_dpo.py --config-path configs/dpo_mixed.yaml --pairs-path data/pairs/dpo_pairs_mixed.json

# 3. Evaluate all models
uv run python scripts/evaluate.py models/sft-baseline
uv run python scripts/evaluate.py models/dpo-clean
uv run python scripts/evaluate.py models/dpo-padded-mild
uv run python scripts/evaluate.py models/dpo-padded-heavy
uv run python scripts/evaluate.py models/dpo-mixed

# 4. Generate analysis
uv run python scripts/analyze.py
```

---

## 12. Success Criteria

The experiment successfully demonstrates length-hacking if:

1. **Length Inflation**: `dpo-padded-heavy` generates significantly longer responses than `dpo-clean` on clean test data
2. **No Accuracy Benefit**: Longer responses do NOT correlate with higher accuracy
3. **Filler Transfer**: Models trained on padded data generate filler phrases even on clean inputs
4. **Dose-Response**: More padding in training → more verbose outputs

If all four criteria are met, we have strong evidence that outcome-based DPO rewards induce length-hacking behavior.
