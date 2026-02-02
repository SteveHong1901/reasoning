import sys
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.plotting import (
    plot_length_distribution,
    plot_accuracy_comparison,
    plot_filler_ratio_comparison,
    plot_length_vs_accuracy,
    plot_pareto_frontier,
)

app = typer.Typer()
console = Console()


def load_results(results_dir: Path) -> dict[str, dict]:
    results = {}
    for result_file in results_dir.glob("*_results.json"):
        with open(result_file) as f:
            data = json.load(f)
        model_name = result_file.stem.replace("_results", "")
        results[model_name] = data
    return results


@app.command()
def main(
    results_dir: Path = typer.Option(Path("results"), help="Directory with result files"),
    output_dir: Path = typer.Option(Path("results/figures"), help="Output directory for plots"),
):
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Loading results...")
    results = load_results(results_dir)

    if not results:
        console.print("[red]No results found!")
        raise typer.Exit(1)

    console.print(f"Found {len(results)} result files")

    table = Table(title="Summary")
    table.add_column("Model", style="cyan")
    table.add_column("Accuracy", style="green")
    table.add_column("Mean Length", style="yellow")
    table.add_column("Filler Ratio", style="red")

    for name, data in results.items():
        m = data["metrics"]
        table.add_row(
            name,
            f"{m['accuracy'] * 100:.1f}%",
            f"{m['mean_length']:.0f}",
            f"{m['filler_ratio']:.4f}",
        )

    console.print(table)

    console.print("\n[bold]Generating plots...")

    length_data = {
        name: [s["length"] for s in data["samples"]]
        for name, data in results.items()
    }
    plot_length_distribution(length_data, output_dir / "length_distribution.png")
    console.print("[green]  Created length_distribution.png")

    accuracy_data = {name: data["metrics"]["accuracy"] * 100 for name, data in results.items()}
    plot_accuracy_comparison(accuracy_data, output_dir / "accuracy_comparison.png")
    console.print("[green]  Created accuracy_comparison.png")

    filler_data = {name: data["metrics"]["filler_ratio"] for name, data in results.items()}
    plot_filler_ratio_comparison(filler_data, output_dir / "filler_ratio_comparison.png")
    console.print("[green]  Created filler_ratio_comparison.png")

    for name, data in results.items():
        lengths = [s["length"] for s in data["samples"]]
        correct = [s["correct"] for s in data["samples"]]
        plot_length_vs_accuracy(
            lengths,
            correct,
            name,
            output_dir / f"length_vs_accuracy_{name}.png",
        )
    console.print("[green]  Created length_vs_accuracy plots")

    pareto_data = [
        {
            "name": name,
            "accuracy": data["metrics"]["accuracy"],
            "mean_length": data["metrics"]["mean_length"],
        }
        for name, data in results.items()
    ]
    plot_pareto_frontier(pareto_data, output_dir / "pareto_frontier.png")
    console.print("[green]  Created pareto_frontier.png")

    console.print(f"\n[bold green]All plots saved to {output_dir}")


if __name__ == "__main__":
    app()
