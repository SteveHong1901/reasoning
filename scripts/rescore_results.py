import argparse
import json
import math
import re
from pathlib import Path

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
    return len(set(ngrams)) / len(ngrams)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))


def median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    mid = len(ys) // 2
    if len(ys) % 2 == 1:
        return float(ys[mid])
    return (ys[mid - 1] + ys[mid]) / 2.0


def pearson_corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    den = denx * deny
    return (num / den) if den else 0.0


def compute_metrics(predictions: list[str], references: list[str], responses: list[str]) -> dict:
    correct = [p == r for p, r in zip(predictions, references)]
    accuracy = sum(correct) / len(correct) if correct else 0.0

    lengths = [len(r.split()) for r in responses]
    filler_ratios = [compute_filler_ratio(r) for r in responses]
    uniqueness_scores = [compute_ngram_uniqueness(r) for r in responses]

    length_accuracy_corr = pearson_corr([float(x) for x in lengths], [1.0 if c else 0.0 for c in correct])

    return {
        "accuracy": accuracy,
        "mean_length": mean([float(x) for x in lengths]),
        "std_length": std([float(x) for x in lengths]),
        "median_length": median([float(x) for x in lengths]),
        "filler_ratio": mean(filler_ratios),
        "ngram_uniqueness": mean(uniqueness_scores),
        "length_accuracy_corr": length_accuracy_corr,
        "num_samples": len(predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore saved *_results.json with improved answer extraction")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing *_results.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_rescored"),
        help="Directory to write rescored *_results.json files",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("*_results.json"))
    if not files:
        raise SystemExit(f"No *_results.json found in {input_dir}")

    for fp in files:
        data = json.loads(fp.read_text())
        samples = data["samples"]

        ground_truths = [s["ground_truth"] for s in samples]
        responses = [s["response"] for s in samples]
        predictions = [extract_answer(r) or "" for r in responses]

        metrics = compute_metrics(predictions, ground_truths, responses)

        for s, p in zip(samples, predictions):
            s["prediction"] = p
            s["correct"] = p == s["ground_truth"]

        data["metrics"].update(metrics)

        out_path = output_dir / fp.name
        out_path.write_text(json.dumps(data, indent=2))

    print(f"Rescored {len(files)} files → {output_dir}")


if __name__ == "__main__":
    main()

