import sys
import json
from pathlib import Path

import typer
import torch
from rich.console import Console
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import DPOConfig, DPOTrainer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ExperimentConfig, get_debug_config, get_cpu_config

app = typer.Typer()
console = Console()


def load_pairs(path: Path) -> Dataset:
    with open(path) as f:
        pairs = json.load(f)
    return Dataset.from_list(pairs)


def load_model_and_tokenizer(config: ExperimentConfig):
    model_config = config.model

    tokenizer = AutoTokenizer.from_pretrained(model_config.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if model_config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_config.name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if not model_config.load_in_4bit else None,
        device_map="auto",
        attn_implementation="flash_attention_2" if model_config.use_flash_attention else None,
    )

    if config.train.use_peft:
        peft_config = LoraConfig(
            r=config.train.lora_r,
            lora_alpha=config.train.lora_alpha,
            lora_dropout=config.train.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, tokenizer


@app.command()
def main(
    config_path: Path = typer.Option(None, help="Path to config YAML"),
    pairs_path: Path = typer.Option(Path("data/pairs/dpo_pairs_clean.json"), help="Path to DPO pairs"),
    debug: bool = typer.Option(False, help="Run in debug mode"),
    cpu: bool = typer.Option(False, help="Run in CPU mode (minimal memory)"),
):
    if cpu:
        config = get_cpu_config()
        console.print("[yellow]Running in CPU mode with minimal model (tiny-gpt2)")
    elif debug:
        config = get_debug_config()
        console.print("[yellow]Running in DEBUG mode with tiny model and small data")
    elif config_path:
        config = ExperimentConfig.from_yaml(config_path)
    else:
        config = ExperimentConfig()

    console.print(f"[bold]Experiment: {config.name}")
    console.print(f"[bold]Model: {config.model.name}")
    console.print(f"[bold]Pairs: {pairs_path}")

    console.print("\n[bold]Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config)

    console.print("\n[bold]Loading DPO pairs...")
    train_dataset = load_pairs(pairs_path)
    console.print(f"Loaded {len(train_dataset)} pairs")

    if config.data.max_samples:
        train_dataset = train_dataset.select(range(min(config.data.max_samples, len(train_dataset))))
        console.print(f"Using {len(train_dataset)} pairs (limited)")

    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.train.num_train_epochs,
        max_steps=config.train.max_steps,
        per_device_train_batch_size=config.train.per_device_train_batch_size,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        learning_rate=config.train.learning_rate,
        warmup_steps=config.train.warmup_steps,
        logging_steps=config.train.logging_steps,
        save_steps=config.train.save_steps,
        beta=config.train.beta,
        max_length=config.train.max_length,
        max_prompt_length=config.train.max_prompt_length,
        remove_unused_columns=False,
        bf16=torch.cuda.is_available(),
        report_to="wandb" if config.use_wandb else "none",
        run_name=config.name if config.use_wandb else None,
        seed=config.seed,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    console.print("\n[bold]Starting training...")
    trainer.train()

    console.print("\n[bold]Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    config.to_yaml(output_dir / "config.yaml")
    console.print(f"\n[bold green]Training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    app()
