import sys
from pathlib import Path

import typer
from rich.console import Console
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DataConfig, get_debug_config
from src.data_utils import load_gsm8k, extract_ground_truth
from src.filler_injection import pad_dataset_answers

app = typer.Typer()
console = Console()


def process_and_save(
    train_data: Dataset,
    test_data: Dataset,
    padding_level: str,
    output_dir: Path,
    seed: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    train_padded = pad_dataset_answers(train_data, padding_level, seed)
    test_clean = test_data

    train_path = output_dir / f"train_{padding_level}.json"
    test_path = output_dir / "test_clean.json"

    train_padded.to_json(train_path)
    if not test_path.exists():
        test_clean.to_json(test_path)

    console.print(f"[green]Saved {len(train_padded)} train samples to {train_path}")
    console.print(f"[green]Saved {len(test_clean)} test samples to {test_path}")


@app.command()
def main(
    output_dir: Path = typer.Option(Path("data/processed"), help="Output directory"),
    debug: bool = typer.Option(False, help="Run in debug mode with small data"),
    seed: int = typer.Option(42, help="Random seed"),
):
    if debug:
        config = get_debug_config()
        data_config = config.data
    else:
        data_config = DataConfig(seed=seed)

    console.print("[bold]Loading GSM8K dataset...")
    train_data, test_data = load_gsm8k(data_config)
    console.print(f"Loaded {len(train_data)} train, {len(test_data)} test samples")

    for padding_level in ["clean", "mild", "heavy"]:
        console.print(f"\n[bold]Processing {padding_level} variant...")
        process_and_save(
            train_data,
            test_data,
            padding_level,
            output_dir,
            seed,
        )

    console.print("\n[bold green]Data preparation complete!")


if __name__ == "__main__":
    app()
