"""Оценка ответа SFT по покрытию обязательных терминов. Это не вызов внешней LLM."""
from __future__ import annotations


def term_recall(text: str, terms: list[str]) -> float:
    """Доля терминов, встретившихся в тексте без учёта регистра."""
    if not terms:
        return 0.0
    haystack = text.lower()
    hits = sum(1 for term in terms if term.lower() in haystack)
    return hits / len(terms)


def score_prediction(prediction: str, reference: str, terms: list[str]) -> dict[str, float]:
    """Сравнивает предсказание и эталон по одному и тому же списку терминов."""
    return {
        "prediction_recall": term_recall(prediction, terms),
        "reference_recall": term_recall(reference, terms),
    }
