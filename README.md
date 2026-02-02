# The Reasoning Mirage

Investigating length-hacking in Chain-of-Thought post-training.

## Research Question

Does DPO/PPO training on reasoning tasks cause models to "filibuster"—generating verbose, circular reasoning because longer responses correlate with higher rewards?

## Setup

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

## Quick Start (Debug Mode)

Test the entire pipeline locally without a GPU:

```bash
# Prepare data (50 samples)
uv run python scripts/data_prep.py --debug

# Create DPO pairs
uv run python scripts/create_pairs.py --debug

# Train (10 steps, tiny model)
uv run python scripts/train_dpo.py --debug

# Evaluate
uv run python scripts/evaluate.py models/debug --debug
```

## Full Pipeline

### 1. Data Preparation

Creates three variants of GSM8K: clean, mild padding, heavy padding.

```bash
uv run python scripts/data_prep.py
uv run python scripts/create_pairs.py
```

### 2. Training

#### SFT Baseline
```bash
uv run python scripts/train_sft.py --config-path configs/sft_baseline.yaml
```

#### DPO Experiments
```bash
uv run python scripts/train_dpo.py --config-path configs/dpo_clean.yaml --pairs-path data/pairs/dpo_pairs_clean.json
uv run python scripts/train_dpo.py --config-path configs/dpo_padded_mild.yaml --pairs-path data/pairs/dpo_pairs_mild.json
uv run python scripts/train_dpo.py --config-path configs/dpo_padded_heavy.yaml --pairs-path data/pairs/dpo_pairs_heavy.json
uv run python scripts/train_dpo.py --config-path configs/dpo_mixed.yaml --pairs-path data/pairs/dpo_pairs_mixed.json
```

### 3. Evaluation

```bash
uv run python scripts/evaluate.py models/sft-baseline
uv run python scripts/evaluate.py models/dpo-clean
uv run python scripts/evaluate.py models/dpo-padded-mild
uv run python scripts/evaluate.py models/dpo-padded-heavy
uv run python scripts/evaluate.py models/dpo-mixed
```

### 4. Analysis

```bash
uv run python scripts/analyze.py
```

Generates plots in `results/figures/`.

## Cloud GPU Execution

No local GPU? Use these platforms:

| Platform | Setup |
|----------|-------|
| **RunPod** | Upload code, install deps, run scripts |
| **Modal** | `modal run scripts/train_dpo.py` (needs Modal wrapper) |
| **Lambda Labs** | SSH, clone repo, run |
| **Vast.ai** | Cheapest option, variable quality |

### RunPod Quick Setup

```bash
# On RunPod instance
git clone <your-repo>
cd reasoning-mirage
pip install uv
uv sync
uv run python scripts/train_dpo.py --config-path configs/dpo_clean.yaml --pairs-path data/pairs/dpo_pairs_clean.json
```

## Project Structure

```
.
├── configs/           # Experiment configs (YAML)
├── data/
│   ├── processed/    # Padded datasets
│   └── pairs/        # DPO preference pairs
├── models/           # Trained checkpoints
├── results/          # Evaluation outputs + figures
├── scripts/          # Runnable scripts
├── src/              # Core library code
├── plan.md           # Detailed research plan
└── pyproject.toml    # Dependencies
```

## Key Metrics

- **Accuracy**: Final answer correctness on GSM8K test
- **Mean Length**: Average response token count
- **Filler Ratio**: Proportion of filler phrases detected
- **N-gram Uniqueness**: Measures repetitiveness
- **Length-Accuracy Correlation**: Does longer = better?

## Filler Patterns

The injector adds these types of filler:

- **Restatement**: "Let me reconsider this approach."
- **Hedging**: "I want to be absolutely certain here."
- **Verification**: "Double-checking my work here."
- **Affirmation**: "Yes, this approach makes sense."
- **Transition**: "Moving on to the next step,"

## Expected Results

If the hypothesis holds:
1. `dpo-padded-heavy` produces longer responses than `dpo-clean` on clean test data
2. Length increase is NOT accompanied by accuracy increase
3. Filler ratio is higher in padded-trained models
4. The model "learns" to filibuster even when not trained on filler

## Citation

If you use this code:

```bibtex
@misc{reasoning-mirage,
  title={The Reasoning Mirage: Length-Hacking in Chain-of-Thought Post-Training},
  year={2026}
}
```
