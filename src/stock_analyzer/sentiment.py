from __future__ import annotations

import re
from functools import lru_cache


POSITIVE_WORDS = {
    "beat",
    "growth",
    "strong",
    "bullish",
    "upgrade",
    "record",
    "surge",
    "outperform",
    "profit",
    "expansion",
}

NEGATIVE_WORDS = {
    "miss",
    "decline",
    "weak",
    "bearish",
    "downgrade",
    "loss",
    "lawsuit",
    "drop",
    "risk",
    "cut",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())


def lexicon_sentiment(text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    pos = sum(t in POSITIVE_WORDS for t in tokens)
    neg = sum(t in NEGATIVE_WORDS for t in tokens)
    raw = (pos - neg) / max(len(tokens), 1)
    return max(min(raw * 10.0, 1.0), -1.0)


@lru_cache(maxsize=1)
def _load_finbert_pipeline():
    try:
        from transformers import pipeline  # type: ignore
    except Exception:
        return None
    try:
        return pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
        )
    except Exception:
        return None


def finbert_sentiment(text: str) -> float | None:
    clf = _load_finbert_pipeline()
    if clf is None:
        return None
    out = clf(text[:1200])[0]
    label = str(out.get("label", "")).lower()
    score = float(out.get("score", 0.0))
    if "positive" in label:
        return score
    if "negative" in label:
        return -score
    return 0.0


def compute_sentiment(text: str, prefer_finbert: bool = False) -> float:
    if prefer_finbert:
        fin = finbert_sentiment(text)
        if fin is not None:
            return fin
    return lexicon_sentiment(text)

