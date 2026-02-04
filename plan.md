# The Reasoning Mirage: Length-Hacking in Chain-of-Thought Post-Training

## Research Question

Does standard post-training (DPO/PPO) on reasoning tasks induce "Length Hacking"—where models learn that verbosity is a proxy for correctness?

**Hypothesis**: If you reward a model strictly for the correct final answer, the model will learn to "filibuster"—generating meaningless circular logic or filler tokens because the optimization process has a bias toward longer generations.

---

## Phase 1: Environment & Data Setup

### 1.1 Base Model Selection
- **Primary**: `meta-llama/Llama-3.2-1B-Instruct` (low compute)
- **Optional comparison**: `Qwen/Qwen2.5-1.5B-Instruct`

### 1.2 Dataset Preparation (GSM8K)

Create 3 versions of GSM8K training data:

| Version | Description |
|---------|-------------|
| `clean` | Original GSM8K CoT solutions |
| `padded-mild` | CoT with 2-3 filler sentences per solution |
| `padded-heavy` | CoT with 5-7 filler sentences per solution |

**Filler types to inject:**
- Circular restatements: *"Let me reconsider this. Yes, this is correct."*
- Hedging: *"To be absolutely certain, I should verify..."*
- Redundant steps: *"We can also think of this another way..."*
- Self-affirmation: *"This approach makes sense because..."*
- False uncertainty: *"Wait, let me double-check this calculation..."*

### 1.3 Preference Pair Construction (for DPO)
- **Chosen**: Correct answer (regardless of length)
- **Rejected**: Incorrect answer

This mimics a naive reward signal that only checks final answer correctness.

---

## Phase 2: Training Configurations

| Experiment | Training Data | Method | Hypothesis |
|------------|---------------|--------|------------|
| `baseline` | Clean GSM8K | SFT only | Control |
| `dpo-clean` | Clean pairs | DPO | Baseline DPO |
| `dpo-padded-mild` | Padded-mild pairs | DPO | Moderate length hacking |
| `dpo-padded-heavy` | Padded-heavy pairs | DPO | Severe length hacking |
| `dpo-mixed` | 50% clean + 50% padded | DPO | Real-world scenario |

### Training Hyperparameters (Initial)
- Learning rate: 5e-7
- Beta (DPO): 0.1
- Batch size: 4 (with gradient accumulation)
- Max steps: 1000-2000
- Warmup: 100 steps

---

## Phase 3: Evaluation Metrics

### 3.1 Primary Metrics (on clean GSM8K test set)
- **Accuracy**: Final answer correctness (exact match after extraction)
- **Response Length**: Token count distribution
- **Length-Accuracy Correlation**: Pearson/Spearman correlation

### 3.2 Filibuster Detection Metrics
- **Filler Ratio**: % of tokens that are semantically redundant (keyword/pattern matching)
- **Unique N-gram Ratio**: `unique_ngrams / total_ngrams` (detect repetition)
- **Reasoning Step Count**: Actual logical steps vs. total sentences
- **Compression Ratio**: Length of response / length of minimal correct solution

### 3.3 Qualitative Analysis
- Manual inspection of 50 samples per model
- Categorize failure modes: circular logic, hedging, restating, genuine reasoning

---

## Phase 4: Ablations & Mitigations

### 4.1 Length Penalty Experiments
- Add length penalty term to DPO objective
- Test penalty coefficients: [0.0, 0.01, 0.05, 0.1]

### 4.2 Reward Model Analysis
- Train a simple RM on clean data
- Measure correlation between RM scores and response length
- Test if RM itself has length bias

---

## Expected Plots & Visualizations

### Core Findings

| Plot | Type | X-axis | Y-axis | Purpose |
|------|------|--------|--------|---------|
| 1a | Violin/Box | Model variant | Token count | Show length inflation |
| 1b | Line | Training steps | Avg response length | When does hacking emerge? |
| 2a | Bar | Model variant | Accuracy (%) | Does accuracy suffer? |
| 2b | Scatter | Response length | Accuracy (binned) | Length-quality decoupling |

### Filibuster Analysis

| Plot | Type | X-axis | Y-axis | Purpose |
|------|------|--------|--------|---------|
| 3a | Bar | Model variant | Filler ratio (%) | Quantify filibustering |
| 3b | Box | Model variant | Unique n-gram ratio | Measure repetitiveness |
| 3c | Scatter | Total sentences | Reasoning steps | Efficiency of reasoning |

### Training Dynamics

| Plot | Type | X-axis | Y-axis | Purpose |
|------|------|--------|--------|---------|
| 4a | Line | Training steps | DPO loss | Convergence comparison |
| 4b | Line | Training steps | KL divergence | Policy drift |

### Mitigation Results

| Plot | Type | Description | Purpose |
|------|------|-------------|---------|
| 5a | Grouped Bar | Penalty ablation (accuracy + length) | Effect of length penalty |
| 5b | Scatter | Pareto frontier (length vs accuracy) | Find efficient configurations |

### Qualitative

| Figure | Type | Description |
|--------|------|-------------|
| Table 1 | Table | Cherry-picked examples showing filibustering |
| Fig 6 | Heatmap | (Optional) Token-level attention patterns |

---

## Directory Structure

```
DPO/
├── configs/              # Training configs (yaml)
│   ├── baseline.yaml
│   ├── dpo_clean.yaml
│   ├── dpo_padded_mild.yaml
│   ├── dpo_padded_heavy.yaml
│   └── dpo_mixed.yaml
├── data/
│   ├── raw/              # Original GSM8K
│   ├── processed/        # Clean/padded versions
│   └── pairs/            # DPO preference pairs
├── scripts/
│   ├── data_prep.py      # Create padded datasets
│   ├── create_pairs.py   # Generate DPO pairs
│   ├── train_dpo.py      # DPO training script
│   ├── train_sft.py      # SFT baseline script
│   ├── evaluate.py       # Run evaluation
│   └── analyze.py        # Generate plots
├── src/
│   ├── __init__.py
│   ├── data_utils.py     # Data loading utilities
│   ├── filler_injection.py  # Filler generation logic
│   ├── metrics.py        # Evaluation metrics
│   └── plotting.py       # Visualization functions
├── models/               # Saved checkpoints
├── results/              # Eval outputs, figures
├── notebooks/            # Analysis notebooks
├── requirements.txt
└── README.md
```

---

## Compute Estimate

| Task | GPU Hours (A100) |
|------|------------------|
| Data preparation | ~1 hr (CPU) |
| SFT baseline | ~2 hrs |
| DPO training (per config) | ~2-4 hrs |
| Evaluation (all models) | ~2 hrs |
| **Total (5 experiments)** | **~15-25 hrs** |

---

## Development & Execution Setup

### Local Testing Mode (`--debug`)

All scripts support a `--debug` flag that enables:
- **Tiny model**: `HuggingFaceTB/SmolLM-135M-Instruct` instead of Llama-3.2-1B
- **Small data**: 50 samples instead of full dataset
- **Fast training**: 10 steps, no checkpointing
- **CPU-friendly**: Works without GPU for pipeline verification

```bash
# Test locally before spending cloud credits
uv run python scripts/train_dpo.py --debug
```

### Cloud GPU Platforms (No Local GPU)

Recommended platforms for running full experiments:

| Platform | Pros | Cons |
|----------|------|------|
| **RunPod** | Easy setup, pay-per-hour, good GPUs | Manual setup |
| **Modal** | Serverless, code-defined infra | Learning curve |
| **Lambda Labs** | Simple, good prices | Availability |
| **Vast.ai** | Cheapest | Variable quality |

### UV Package Management

```bash
# Install dependencies
uv sync

# Run any script
uv run python scripts/data_prep.py

# Run with debug mode
uv run python scripts/train_dpo.py --debug
```

### Code Conventions

- **No inline comments** - code is self-documenting
- **Type hints everywhere** - use modern Python typing
- **Dataclasses for configs** - not dictionaries
- **CLI via Typer** - clean argument parsing
- **Config-driven** - YAML files for experiment configs
- **Explanations in README** - not in source code

---

## Key Research Questions

1. **Does DPO on padded data cause spontaneous filler generation on clean test data?**
2. **Is there a threshold where padding destroys accuracy vs. just inflating length?**
3. **Does the model learn to filibuster even when trained on mixed (clean + padded) data?**
4. **Can simple length penalties mitigate the effect without hurting accuracy?**
5. **What types of filler does the model learn to generate (hedging, restating, etc.)?**

---

## Success Criteria

The research is successful if we demonstrate:
1. Clear evidence of length inflation in padded-trained models
2. Quantifiable "filibuster" patterns in generated responses
3. Insights into when/why length hacking emerges during training
4. At least one mitigation strategy that reduces filibustering

---

## Timeline & Milestones

- [ ] Phase 1: Data preparation complete
- [ ] Phase 2: All training runs complete
- [ ] Phase 3: Evaluation metrics computed
- [ ] Phase 4: Ablation experiments complete
- [ ] Phase 5: Plots and analysis complete
- [ ] Phase 6: Paper draft
