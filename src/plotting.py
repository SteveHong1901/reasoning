from pathlib import Path
from dataclasses import dataclass

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


@dataclass
class PlotStyle:
    figsize: tuple[int, int] = (10, 6)
    palette: str = "Set2"
    context: str = "paper"
    font_scale: float = 1.2


def setup_style(style: PlotStyle = PlotStyle()):
    sns.set_theme(context=style.context, font_scale=style.font_scale)
    sns.set_palette(style.palette)
    plt.rcParams["figure.figsize"] = style.figsize
    plt.rcParams["figure.dpi"] = 150


def plot_length_distribution(
    results: dict[str, list[int]],
    output_path: Path,
    title: str = "Response Length Distribution by Model",
):
    setup_style()
    data = []
    for model_name, lengths in results.items():
        for length in lengths:
            data.append({"Model": model_name, "Length (tokens)": length})
    df = pd.DataFrame(data)

    fig, ax = plt.subplots()
    sns.violinplot(data=df, x="Model", y="Length (tokens)", ax=ax)
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_accuracy_comparison(
    results: dict[str, float],
    output_path: Path,
    title: str = "Accuracy by Model Variant",
):
    setup_style()
    fig, ax = plt.subplots()
    models = list(results.keys())
    accuracies = list(results.values())

    bars = ax.bar(models, accuracies)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)

    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
        )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_length_vs_accuracy(
    lengths: list[int],
    correct: list[bool],
    model_name: str,
    output_path: Path,
):
    setup_style()
    fig, ax = plt.subplots()

    df = pd.DataFrame({"Length": lengths, "Correct": correct})
    df["Length Bin"] = pd.qcut(df["Length"], q=10, duplicates="drop")
    bin_acc = df.groupby("Length Bin")["Correct"].mean()

    ax.scatter(lengths, [int(c) for c in correct], alpha=0.3, s=20)

    bin_centers = [interval.mid for interval in bin_acc.index]
    ax.plot(bin_centers, bin_acc.values, "r-", linewidth=2, label="Binned Accuracy")

    ax.set_xlabel("Response Length (tokens)")
    ax.set_ylabel("Correct (0/1)")
    ax.set_title(f"Length vs Accuracy: {model_name}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_filler_ratio_comparison(
    results: dict[str, float],
    output_path: Path,
    title: str = "Filler Ratio by Model Variant",
):
    setup_style()
    fig, ax = plt.subplots()
    models = list(results.keys())
    ratios = list(results.values())

    bars = ax.bar(models, ratios)
    ax.set_ylabel("Filler Ratio")
    ax.set_title(title)

    for bar, ratio in zip(bars, ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{ratio:.3f}",
            ha="center",
            va="bottom",
        )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_training_curves(
    steps: list[int],
    metrics: dict[str, list[float]],
    output_path: Path,
    title: str = "Training Curves",
):
    setup_style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))

    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric_name, values) in zip(axes, metrics.items()):
        ax.plot(steps, values)
        ax.set_xlabel("Steps")
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)

    plt.suptitle(title)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_pareto_frontier(
    results: list[dict],
    output_path: Path,
    title: str = "Accuracy vs Length Pareto Frontier",
):
    setup_style()
    fig, ax = plt.subplots()

    lengths = [r["mean_length"] for r in results]
    accuracies = [r["accuracy"] * 100 for r in results]
    names = [r["name"] for r in results]

    ax.scatter(lengths, accuracies, s=100)

    for i, name in enumerate(names):
        ax.annotate(
            name,
            (lengths[i], accuracies[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )

    ax.set_xlabel("Mean Response Length (tokens)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
