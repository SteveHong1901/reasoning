import sys
import json
import warnings
from pathlib import Path

import typer
import torch
from rich.console import Console
from rich.table import Table
from rich.progress import track
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, logging

logging.set_verbosity_error()
from peft import PeftModel

warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*max_length.*")
warnings.filterwarnings("ignore", message=".*generation_config.*")
warnings.filterwarnings("ignore", message=".*generation flags.*")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ExperimentConfig, EvalConfig, get_debug_config
from src.data_utils import format_prompt, extract_answer, extract_ground_truth
from src.metrics import compute_metrics, EvalResult

app = typer.Typer()
console = Console()


def load_model(model_path: Path, base_model: str | None = None):
    config_path = model_path / "config.yaml"

    if config_path.exists():
        config = ExperimentConfig.from_yaml(config_path)
        base_model = config.model.name

    adapter_config = model_path / "adapter_config.json"
    is_peft = adapter_config.exists()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_peft and base_model:
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    return model, tokenizer


def generate_responses(
    model,
    tokenizer,
    questions: list[str],
    eval_config: EvalConfig,
) -> list[str]:
    gen_config = GenerationConfig(
        max_new_tokens=eval_config.max_new_tokens,
        do_sample=eval_config.do_sample,
        temperature=eval_config.temperature if eval_config.do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    responses = []
    for question in track(questions, description="Generating..."):
        prompt = format_prompt(question)
        messages = [{"role": "user", "content": prompt}]
        
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=gen_config)
        
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        responses.append(response)

    return responses


def print_results(result: EvalResult, model_name: str):
    table = Table(title=f"Results: {model_name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Accuracy", f"{result.accuracy * 100:.2f}%")
    table.add_row("Mean Length", f"{result.mean_length:.1f} tokens")
    table.add_row("Std Length", f"{result.std_length:.1f}")
    table.add_row("Median Length", f"{result.median_length:.1f}")
    table.add_row("Filler Ratio", f"{result.filler_ratio:.4f}")
    table.add_row("N-gram Uniqueness", f"{result.ngram_uniqueness:.4f}")
    table.add_row("Length-Acc Corr", f"{result.length_accuracy_corr:.4f}")
    table.add_row("Num Samples", str(result.num_samples))

    console.print(table)


@app.command()
def main(
    model_path: Path = typer.Argument(..., help="Path to trained model"),
    output_dir: Path = typer.Option(Path("results"), help="Output directory"),
    base_model: str = typer.Option(None, help="Base model for PEFT"),
    debug: bool = typer.Option(False, help="Run in debug mode"),
    num_samples: int = typer.Option(None, help="Number of samples to evaluate"),
):
    if debug:
        config = get_debug_config()
        eval_config = config.eval
        data_config = config.data
        console.print("[yellow]Running in DEBUG mode")
    else:
        eval_config = EvalConfig(num_samples=num_samples)
        data_config = None

    console.print(f"[bold]Evaluating model: {model_path}")

    console.print("\n[bold]Loading model...")
    model, tokenizer = load_model(model_path, base_model)

    console.print("\n[bold]Loading test data...")
    dataset = load_dataset("openai/gsm8k", "main")
    test_data = dataset["test"]

    max_samples = eval_config.num_samples or (data_config.max_samples if data_config else None)
    if max_samples:
        test_data = test_data.select(range(min(max_samples, len(test_data))))

    console.print(f"Evaluating on {len(test_data)} samples")

    questions = [item["question"] for item in test_data]
    ground_truths = [extract_ground_truth(item["answer"]) for item in test_data]

    console.print("\n[bold]Generating responses...")
    responses = generate_responses(model, tokenizer, questions, eval_config)

    predictions = [extract_answer(r) or "" for r in responses]

    result = compute_metrics(predictions, ground_truths, responses)

    print_results(result, model_path.name)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_path.name}_results.json"

    output_data = {
        "model": str(model_path),
        "metrics": {
            "accuracy": result.accuracy,
            "mean_length": result.mean_length,
            "std_length": result.std_length,
            "median_length": result.median_length,
            "filler_ratio": result.filler_ratio,
            "ngram_uniqueness": result.ngram_uniqueness,
            "length_accuracy_corr": result.length_accuracy_corr,
            "num_samples": result.num_samples,
        },
        "samples": [
            {
                "question": q,
                "response": r,
                "prediction": p,
                "ground_truth": g,
                "correct": p == g,
                "length": len(r.split()),
            }
            for q, r, p, g in zip(questions, responses, predictions, ground_truths)
        ],
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    console.print(f"\n[bold green]Results saved to {output_file}")


if __name__ == "__main__":
    app()
