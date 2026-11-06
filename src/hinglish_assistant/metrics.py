from __future__ import annotations

from dataclasses import dataclass

from sacrebleu.metrics import BLEU

from .text import normalize_text


def _edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left_item != right_item)))
        previous = current
    return previous[-1]


def character_error_rate(prediction: str, reference: str) -> float:
    reference = normalize_text(reference)
    prediction = normalize_text(prediction)
    if not reference:
        return 0.0 if not prediction else 1.0
    return _edit_distance(list(prediction), list(reference)) / len(reference)


def word_accuracy(prediction: str, reference: str) -> float:
    """1 - word error rate, clipped to zero for very divergent predictions."""
    reference_words = normalize_text(reference).split()
    prediction_words = normalize_text(prediction).split()
    if not reference_words:
        return 1.0 if not prediction_words else 0.0
    return max(0.0, 1 - _edit_distance(prediction_words, reference_words) / len(reference_words))


def corpus_bleu(predictions: list[str], references: list[str]) -> float:
    if not predictions:
        return 0.0
    return BLEU(effective_order=True).corpus_score(predictions, [references]).score


@dataclass(frozen=True)
class TransliterationMetrics:
    cer: float
    word_accuracy: float


def aggregate_transliteration(predictions: list[str], references: list[str]) -> TransliterationMetrics:
    count = len(references)
    if count == 0:
        return TransliterationMetrics(0.0, 0.0)
    return TransliterationMetrics(
        cer=sum(character_error_rate(p, r) for p, r in zip(predictions, references)) / count,
        word_accuracy=sum(word_accuracy(p, r) for p, r in zip(predictions, references)) / count,
    )
