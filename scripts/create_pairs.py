import sys
import json
import random
from pathlib import Path

import typer
from rich.console import Console
from datasets import load_dataset, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DataConfig, get_debug_config
from src.data_utils import format_prompt, extract_ground_truth
from src.filler_injection import pad_dataset_answers

app = typer.Typer()
console = Console()


def create_incorrect_answer(correct_answer: str, rng: random.Random) -> str:
    try:
        num = float(correct_answer.replace(",", ""))
        perturbations = [
            lambda x: x + rng.randint(1, 10),
            lambda x: x - rng.randint(1, 10),
            lambda x: x * rng.choice([2, 10, 0.5]),
            lambda x: x + rng.uniform(-x * 0.2, x * 0.2),
        ]
        wrong_num = rng.choice(perturbations)(num)
        if wrong_num == num:
            wrong_num = num + 1
        if wrong_num == int(wrong_num):
            return str(int(wrong_num))
        return f"{wrong_num:.2f}"
    except ValueError:
        return correct_answer + "_wrong"


def generate_wrong_reasoning(question: str, wrong_answer: str, rng: random.Random) -> str:
    templates = [
        f"Let me solve this step by step.\nAfter calculating, the answer is #### {wrong_answer}",
        f"Working through this problem:\nThe final answer is #### {wrong_answer}",
        f"To find the answer:\nTherefore, #### {wrong_answer}",
    ]
    return rng.choice(templates)


def create_dpo_pairs(
    train_data: Dataset,
    padding_level: str,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    padded_data = pad_dataset_answers(train_data, padding_level, seed)

    pairs = []
    for i, item in enumerate(padded_data):
        question = item["question"]
        correct_reasoning = item["answer"]
        correct_answer = extract_ground_truth(correct_reasoning)

        wrong_answer = create_incorrect_answer(correct_answer, rng)
        wrong_reasoning = generate_wrong_reasoning(question, wrong_answer, rng)

        pairs.append({
            "prompt": format_prompt(question),
            "chosen": correct_reasoning,
            "rejected": wrong_reasoning,
            "correct_answer": correct_answer,
        })

    return pairs


@app.command()
def main(
    output_dir: Path = typer.Option(Path("data/pairs"), help="Output directory"),
    debug: bool = typer.Option(False, help="Run in debug mode with small data"),
    seed: int = typer.Option(42, help="Random seed"),
):
    if debug:
        config = get_debug_config()
        data_config = config.data
    else:
        data_config = DataConfig(seed=seed)

    console.print("[bold]Loading GSM8K dataset...")
    dataset = load_dataset(data_config.dataset_name, data_config.dataset_config)
    train_data = dataset["train"]

    if data_config.max_samples:
        train_data = train_data.select(range(min(data_config.max_samples, len(train_data))))

    console.print(f"Loaded {len(train_data)} samples")

    output_dir.mkdir(parents=True, exist_ok=True)

    for padding_level in ["clean", "mild", "heavy"]:
        console.print(f"\n[bold]Creating {padding_level} DPO pairs...")
        pairs = create_dpo_pairs(train_data, padding_level, seed)

        output_path = output_dir / f"dpo_pairs_{padding_level}.json"
        with open(output_path, "w") as f:
            json.dump(pairs, f, indent=2)

        console.print(f"[green]Saved {len(pairs)} pairs to {output_path}")

    console.print("\n[bold]Creating mixed pairs (50% clean + 50% padded)...")
    clean_pairs = create_dpo_pairs(train_data, "clean", seed)
    heavy_pairs = create_dpo_pairs(train_data, "heavy", seed + 1)

    rng = random.Random(seed)
    half_size = len(clean_pairs) // 2
    mixed_pairs = rng.sample(clean_pairs, half_size) + rng.sample(heavy_pairs, half_size)
    rng.shuffle(mixed_pairs)

    output_path = output_dir / "dpo_pairs_mixed.json"
    with open(output_path, "w") as f:
        json.dump(mixed_pairs, f, indent=2)

    console.print(f"[green]Saved {len(mixed_pairs)} mixed pairs to {output_path}")
    console.print("\n[bold green]Pair creation complete!")


if __name__ == "__main__":
    app()
