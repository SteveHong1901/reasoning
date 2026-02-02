import random
from dataclasses import dataclass
from typing import Literal


FillerType = Literal["restatement", "hedging", "verification", "affirmation", "transition"]


@dataclass
class FillerInjector:
    seed: int = 42

    FILLERS: dict[FillerType, list[str]] = None

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.FILLERS = {
            "restatement": [
                "Let me reconsider this approach.",
                "To rephrase what I just calculated,",
                "In other words, what this means is",
                "Looking at this from another angle,",
                "To put it differently,",
            ],
            "hedging": [
                "I want to be absolutely certain here.",
                "To ensure I haven't made any errors,",
                "Let me carefully consider this.",
                "I should be thorough in my reasoning.",
                "It's important to think through this carefully.",
            ],
            "verification": [
                "Let me verify this calculation.",
                "Double-checking my work here.",
                "I should confirm this is correct.",
                "Let me make sure I haven't missed anything.",
                "Verifying the arithmetic once more.",
            ],
            "affirmation": [
                "Yes, this approach makes sense.",
                "This is definitely the right path.",
                "I'm confident in this reasoning.",
                "This calculation checks out.",
                "Good, this is correct so far.",
            ],
            "transition": [
                "Moving on to the next step,",
                "Now, continuing with the calculation,",
                "Proceeding further,",
                "With that established,",
                "Building on this,",
            ],
        }

    def _get_random_filler(self, filler_type: FillerType | None = None) -> str:
        if filler_type is None:
            filler_type = self.rng.choice(list(self.FILLERS.keys()))
        return self.rng.choice(self.FILLERS[filler_type])

    def inject_fillers(self, text: str, num_fillers: int) -> str:
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return text

        injection_points = self._select_injection_points(len(sentences), num_fillers)
        result_sentences = []

        for i, sentence in enumerate(sentences):
            result_sentences.append(sentence)
            if i in injection_points:
                filler = self._get_random_filler()
                result_sentences.append(filler)

        return " ".join(result_sentences)

    def _split_sentences(self, text: str) -> list[str]:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _select_injection_points(self, num_sentences: int, num_fillers: int) -> set[int]:
        if num_sentences <= 1:
            return set()
        valid_points = list(range(num_sentences - 1))
        num_to_select = min(num_fillers, len(valid_points))
        return set(self.rng.sample(valid_points, num_to_select))

    def inject_mild(self, text: str) -> str:
        return self.inject_fillers(text, num_fillers=self.rng.randint(2, 3))

    def inject_heavy(self, text: str) -> str:
        return self.inject_fillers(text, num_fillers=self.rng.randint(5, 7))


def pad_dataset_answers(
    dataset,
    padding_level: Literal["clean", "mild", "heavy"],
    seed: int = 42,
):
    if padding_level == "clean":
        return dataset

    injector = FillerInjector(seed=seed)
    inject_fn = injector.inject_mild if padding_level == "mild" else injector.inject_heavy

    def pad_answer(example):
        example["answer"] = inject_fn(example["answer"])
        return example

    return dataset.map(pad_answer)
