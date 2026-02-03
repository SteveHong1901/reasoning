import sys
from pathlib import Path

import typer
import torch
from rich.console import Console
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ExperimentConfig, get_debug_config, get_cpu_config
from src.data_utils import format_prompt

app = typer.Typer()
console = Console()


def format_example(example, tokenizer):
    prompt = format_prompt(example["question"])
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": example["answer"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


@app.command()
def main(
    config_path: Path = typer.Option(None, help="Path to config YAML"),
    debug: bool = typer.Option(False, help="Run in debug mode"),
    cpu: bool = typer.Option(False, help="Run in CPU mode (minimal memory)"),
):
    if cpu:
        config = get_cpu_config()
        config.train.method = "sft"
        console.print("[yellow]Running in CPU mode with minimal model (tiny-gpt2)")
    elif debug:
        config = get_debug_config()
        config.train.method = "sft"
        console.print("[yellow]Running in DEBUG mode with tiny model and small data")
    elif config_path:
        config = ExperimentConfig.from_yaml(config_path)
    else:
        config = ExperimentConfig()
        config.train.method = "sft"
        config.train.output_dir = "models/sft-baseline"

    console.print(f"[bold]Experiment: {config.name}")
    console.print(f"[bold]Model: {config.model.name}")

    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = config.model.max_seq_length

    quantization_config = None
    if config.model.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if not config.model.load_in_4bit else None,
        device_map="auto",
        attn_implementation="flash_attention_2" if config.model.use_flash_attention else None,
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

    console.print("\n[bold]Loading GSM8K dataset...")
    dataset = load_dataset(config.data.dataset_name, config.data.dataset_config)
    train_data = dataset["train"]

    if config.data.max_samples:
        train_data = train_data.select(range(min(config.data.max_samples, len(train_data))))

    train_data = train_data.map(lambda x: format_example(x, tokenizer))
    console.print(f"Loaded {len(train_data)} samples")

    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.train.num_train_epochs,
        max_steps=config.train.max_steps,
        per_device_train_batch_size=config.train.per_device_train_batch_size,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        learning_rate=config.train.learning_rate,
        warmup_steps=config.train.warmup_steps,
        logging_steps=config.train.logging_steps,
        save_steps=config.train.save_steps,
        bf16=torch.cuda.is_available(),
        report_to="wandb" if config.use_wandb else "none",
        run_name=config.name if config.use_wandb else None,
        seed=config.seed,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
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
