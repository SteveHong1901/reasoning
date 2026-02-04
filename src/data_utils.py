import re
from typing import Iterator

from datasets import load_dataset, Dataset

from src.config import DataConfig


PROMPT_TEMPLATE = """Solve this math problem step by step.

Problem: {question}

Show your reasoning, then give the final answer as: #### <number>"""


def load_gsm8k(config: DataConfig) -> tuple[Dataset, Dataset]:
    dataset = load_dataset(config.dataset_name, config.dataset_config)
    train_data = dataset["train"]
    test_data = dataset["test"]

    if config.max_samples:
        train_data = train_data.select(range(min(config.max_samples, len(train_data))))
        test_data = test_data.select(range(min(config.max_samples // 5, len(test_data))))

    return train_data, test_data


def format_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question)


def extract_answer(text: str) -> str | None:
    match = re.search(
        r"#{4,}\s*\$?\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    if match:
        return match.group(1).replace(",", "")

    match = re.search(
        r"(?:final\s+answer|answer)\s*(?:is|=|:)\s*\$?\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).replace(",", "")

    number = r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        tail = lines[-1]
        matches = list(re.finditer(number, tail))
        if matches:
            return matches[-1].group(0).replace(",", "")

    matches = list(re.finditer(number, text))
    if matches:
        return matches[-1].group(0).replace(",", "")
    return None


def extract_ground_truth(answer_text: str) -> str:
    match = re.search(r"####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)", answer_text)
    if match:
        return match.group(1).replace(",", "")
    return answer_text.strip()


def create_chat_format(question: str, answer: str) -> list[dict]:
    return [
        {"role": "user", "content": format_prompt(question)},
        {"role": "assistant", "content": answer},
    ]


def iter_batches(dataset: Dataset, batch_size: int) -> Iterator[list[dict]]:
    batch = []
    for item in dataset:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
