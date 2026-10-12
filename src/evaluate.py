"""Answer scoring and difficulty stratification.

Scoring is deliberately boring and shared by every configuration. If you
change answer extraction, every number in the paper moves, so change it once,
early, and never again.
"""

from __future__ import annotations

import re

from .data import Example


def normalise(text: str) -> str:
    text = str(text).strip().lower()
    text = text.replace(",", "").replace("$", "").replace("\\", "")
    text = re.sub(r"\s+", "", text)
    if text.endswith(".0"):
        text = text[:-2]
    return text


def is_correct(prediction: str, example: Example) -> bool:
    pred, gold = normalise(prediction), normalise(example.answer)
    if pred == gold:
        return True
    # Numeric comparison with a small tolerance for formatting differences.
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except (TypeError, ValueError):
        return False


def accuracy(predictions: dict[str, str], examples: list[Example]) -> float:
    if not examples:
        return 0.0
    hits = sum(
        1 for e in examples if is_correct(predictions.get(e.uid, ""), e)
    )
    return 100.0 * hits / len(examples)


def difficulty_buckets(
    baseline_correct: dict[str, bool], examples: list[Example], n_buckets: int = 3
) -> dict[str, list[Example]]:
    """Split examples by whether the reference baseline solved them.

    Chen et al. 2024 show accuracy in the number of model calls can be
    non-monotonic because extra calls help easy items and waste budget on hard
    ones. You cannot see that mechanism in an aggregate number, so stratify.
    """
    easy, hard = [], []
    for e in examples:
        (easy if baseline_correct.get(e.uid, False) else hard).append(e)
    buckets = {"solved_by_baseline": easy, "unsolved_by_baseline": hard}
    if n_buckets > 2 and "level" in (examples[0].meta if examples else {}):
        by_level: dict[str, list[Example]] = {}
        for e in examples:
            by_level.setdefault(f"level_{e.meta.get('level')}", []).append(e)
        buckets.update(by_level)
    return buckets
