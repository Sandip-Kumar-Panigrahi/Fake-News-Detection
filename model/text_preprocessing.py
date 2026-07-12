"""Text cleaning for TF-IDF training and inference."""

from __future__ import annotations

import os
import re
from typing import List

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_LEMMATIZER = None
_NLTK_READY = False
_WORDNET_OK = False


def ensure_nltk_resources(*, allow_download: bool = False) -> None:
    """Load NLTK data only if already present — never block inference on download."""
    global _NLTK_READY, _WORDNET_OK
    if _NLTK_READY:
        return
    try:
        import nltk

        nltk.data.find("corpora/wordnet")
        _WORDNET_OK = True
    except Exception:
        _WORDNET_OK = False
        if allow_download:
            try:
                import nltk

                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
                nltk.data.find("corpora/wordnet")
                _WORDNET_OK = True
            except Exception:
                _WORDNET_OK = False
    _NLTK_READY = True


def _load_stopwords() -> set:
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))
    except Exception:
        return set(ENGLISH_STOP_WORDS)


def _get_lemmatizer():
    global _LEMMATIZER
    if not _WORDNET_OK:
        return None
    if _LEMMATIZER is None:
        from nltk.stem import WordNetLemmatizer

        _LEMMATIZER = WordNetLemmatizer()
    return _LEMMATIZER


STOP_WORDS = _load_stopwords()
ensure_nltk_resources(allow_download=False)


def _normalize_repetitions(text: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def preprocess_text(raw_text: str, *, lemmatize: bool | None = None) -> str:
    """
    Pipeline: lowercase → URL/HTML removal → punctuation removal →
    stopword removal → optional lemmatization (skipped if wordnet missing).
    """
    if lemmatize is None:
        lemmatize = _WORDNET_OK

    text = str(raw_text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = _normalize_repetitions(text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words: List[str] = text.split()
    lemmatizer = _get_lemmatizer() if lemmatize else None
    clean_words: List[str] = []

    for word in words:
        if word.isdigit():
            clean_words.append(word)
            continue
        if word in STOP_WORDS or len(word) <= 1:
            continue
        if lemmatizer is not None:
            word = lemmatizer.lemmatize(word)
        clean_words.append(word)

    if len(clean_words) < 3:
        fallback = []
        for word in words:
            if len(word) <= 1 or word in STOP_WORDS:
                continue
            if lemmatizer is not None:
                word = lemmatizer.lemmatize(word)
            fallback.append(word)
        if fallback:
            return " ".join(fallback[:8])

    return " ".join(clean_words)
