from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


PaddingLevel = Literal["clean", "mild", "heavy"]
TrainMethod = Literal["sft", "dpo"]


@dataclass
class DataConfig:
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    padding_level: PaddingLevel = "clean"
    max_samples: int | None = None
    test_split_ratio: float = 0.1
    seed: int = 42


@dataclass
class ModelConfig:
    name: str = "meta-llama/Llama-3.2-1B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = False
    use_flash_attention: bool = True


@dataclass
class TrainConfig:
    method: TrainMethod = "dpo"
    output_dir: str = "models/dpo-clean"
    num_train_epochs: int = 1
    max_steps: int = -1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-7
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    beta: float = 0.1
    max_length: int = 2048
    max_prompt_length: int = 512
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


@dataclass
class EvalConfig:
    batch_size: int = 8
    max_new_tokens: int = 512
    temperature: float = 0.0
    do_sample: bool = False
    num_samples: int | None = None


@dataclass
class ExperimentConfig:
    name: str = "default"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    use_wandb: bool = True
    wandb_project: str = "reasoning-mirage"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "ExperimentConfig":
        return cls(
            name=data.get("name", "default"),
            data=DataConfig(**data.get("data", {})),
            model=ModelConfig(**data.get("model", {})),
            train=TrainConfig(**data.get("train", {})),
            eval=EvalConfig(**data.get("eval", {})),
            use_wandb=data.get("use_wandb", True),
            wandb_project=data.get("wandb_project", "reasoning-mirage"),
            seed=data.get("seed", 42),
        )

    def to_yaml(self, path: str | Path) -> None:
        from dataclasses import asdict
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


def get_debug_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="debug",
        data=DataConfig(max_samples=20),
        model=ModelConfig(
            name="HuggingFaceTB/SmolLM-135M-Instruct",
            max_seq_length=256,
            use_flash_attention=False,
        ),
        train=TrainConfig(
            output_dir="models/debug",
            max_steps=5,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            logging_steps=1,
            save_steps=100,
            max_length=256,
            max_prompt_length=128,
            use_peft=False,
        ),
        eval=EvalConfig(
            batch_size=1,
            max_new_tokens=64,
            num_samples=5,
        ),
        use_wandb=False,
    )


def get_cpu_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="cpu-test",
        data=DataConfig(max_samples=10),
        model=ModelConfig(
            name="sshleifer/tiny-gpt2",
            max_seq_length=128,
            use_flash_attention=False,
        ),
        train=TrainConfig(
            output_dir="models/cpu-test",
            max_steps=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            logging_steps=1,
            save_steps=100,
            max_length=128,
            max_prompt_length=64,
            use_peft=False,
        ),
        eval=EvalConfig(
            batch_size=1,
            max_new_tokens=32,
            num_samples=3,
        ),
        use_wandb=False,
    )
