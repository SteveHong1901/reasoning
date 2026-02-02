import re
from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass
class EvalResult:
    accuracy: float
    mean_length: float
    std_length: float
    median_length: float
    filler_ratio: float
    ngram_uniqueness: float
    length_accuracy_corr: float
    num_samples: int


FILLER_PATTERNS = [
    r"let me (reconsider|verify|check|confirm|make sure)",
    r"double[- ]check",
    r"to be (absolutely |)certain",
    r"(yes|good),? this (is|makes|seems)",
    r"i('m| am) confident",
    r"to (rephrase|put it differently)",
    r"in other words",
    r"moving on",
    r"proceeding further",
    r"with that established",
    r"building on this",
    r"looking at this from another angle",
    r"it'?s important to think",
    r"i should be thorough",
]


def compute_filler_ratio(text: str) -> float:
    text_lower = text.lower()
    words = text_lower.split()
    if not words:
        return 0.0

    filler_word_count = 0
    for pattern in FILLER_PATTERNS:
        matches = re.findall(pattern, text_lower)
        filler_word_count += len(matches) * 5

    return min(1.0, filler_word_count / len(words))


def compute_ngram_uniqueness(text: str, n: int = 3) -> float:
    words = text.lower().split()
    if len(words) < n:
        return 1.0

    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 1.0

    unique_ngrams = len(set(ngrams))
    total_ngrams = len(ngrams)
    return unique_ngrams / total_ngrams


def compute_reasoning_steps(text: str) -> int:
    step_patterns = [
        r"step \d+",
        r"first,|second,|third,|finally,",
        r"therefore|thus|hence|so,",
        r"=\s*\d+",
        r"\d+\s*[\+\-\*\/]\s*\d+",
    ]
    count = 0
    text_lower = text.lower()
    for pattern in step_patterns:
        count += len(re.findall(pattern, text_lower))
    return count


def compute_sentence_count(text: str) -> int:
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])


def compute_metrics(
    predictions: list[str],
    references: list[str],
    responses: list[str],
) -> EvalResult:
    correct = [p == r for p, r in zip(predictions, references)]
    accuracy = sum(correct) / len(correct) if correct else 0.0

    lengths = [len(r.split()) for r in responses]
    mean_length = np.mean(lengths) if lengths else 0.0
    std_length = np.std(lengths) if lengths else 0.0
    median_length = np.median(lengths) if lengths else 0.0

    filler_ratios = [compute_filler_ratio(r) for r in responses]
    mean_filler_ratio = np.mean(filler_ratios) if filler_ratios else 0.0

    uniqueness_scores = [compute_ngram_uniqueness(r) for r in responses]
    mean_uniqueness = np.mean(uniqueness_scores) if uniqueness_scores else 0.0

    if len(lengths) > 1 and len(correct) > 1:
        corr = np.corrcoef(lengths, [int(c) for c in correct])[0, 1]
        length_accuracy_corr = corr if not np.isnan(corr) else 0.0
    else:
        length_accuracy_corr = 0.0

    return EvalResult(
        accuracy=accuracy,
        mean_length=mean_length,
        std_length=std_length,
        median_length=median_length,
        filler_ratio=mean_filler_ratio,
        ngram_uniqueness=mean_uniqueness,
        length_accuracy_corr=length_accuracy_corr,
        num_samples=len(predictions),
    )


def compute_length_bins(
    lengths: list[int],
    correct: list[bool],
    num_bins: int = 5,
) -> dict[str, float]:
    if not lengths:
        return {}

    arr_lengths = np.array(lengths)
    arr_correct = np.array(correct)
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(arr_lengths, percentiles)

    result = {}
    for i in range(num_bins):
        mask = (arr_lengths >= bins[i]) & (arr_lengths < bins[i + 1])
        if i == num_bins - 1:
            mask = (arr_lengths >= bins[i]) & (arr_lengths <= bins[i + 1])
        if mask.sum() > 0:
            bin_acc = arr_correct[mask].mean()
            result[f"bin_{i}_({bins[i]:.0f}-{bins[i+1]:.0f})"] = bin_acc

    return result
