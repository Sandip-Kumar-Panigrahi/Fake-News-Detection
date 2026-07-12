"""Reject text that is too short, gibberish, or numbers-only — return Unclear instead of Fake/Real."""

from __future__ import annotations

import re
from typing import Tuple

_VOWELS = set("aeiou")


def _word_looks_gibberish(word: str) -> bool:
    w = re.sub(r"[^a-zA-Z]", "", word).lower()
    if not w:
        return True
    if w.isdigit():
        return False
    if len(w) <= 2:
        return False

    vowel_count = sum(1 for c in w if c in _VOWELS)
    if len(w) >= 4 and vowel_count == 0:
        return True
    if len(w) >= 5 and vowel_count / len(w) < 0.22:
        return True
    if re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", w):
        return True
    return False


def _is_numbers_only(text: str) -> bool:
    stripped = re.sub(r"[\s,.\-+]", "", text)
    return bool(stripped) and stripped.isdigit()


def _is_gibberish_text(text: str) -> bool:
    tokens = [t for t in re.split(r"\s+", text.strip()) if t.strip()]
    if not tokens:
        return True

    if all(re.fullmatch(r"[\d.,+\-]+", t) for t in tokens):
        return True

    letter_tokens = [t for t in tokens if re.search(r"[a-zA-Z]", t)]
    if not letter_tokens:
        return _is_numbers_only(text)

    gibberish_count = sum(1 for t in letter_tokens if _word_looks_gibberish(t))
    if gibberish_count == len(letter_tokens):
        return True
    if len(letter_tokens) >= 2 and gibberish_count / len(letter_tokens) >= 0.6:
        return True
    return False


def assess_input_quality(raw_text: str) -> Tuple[bool, str]:
    """
    Return (True, "") if input is OK to analyze.
    Return (False, reason) → caller should show Unclear.
    """
    text = str(raw_text or "").strip()
    if not text:
        return False, "Please enter some news text."

    if _is_numbers_only(text):
        return False, "Only numbers were entered. Please type a real news sentence or claim."

    if _is_gibberish_text(text):
        return False, (
            "This text is not clear (random letters or meaningless words). "
            "Please enter a proper news headline or factual claim."
        )

    if len(text) < 12:
        return False, "Text is too short to analyze reliably."

    words = [w for w in text.split() if w.strip()]
    if len(words) < 2:
        return False, "Please enter a fuller sentence or headline."

    alpha = sum(1 for c in text if c.isalpha())
    if len(text) > 0 and alpha / len(text) < 0.35:
        return False, "Input looks unclear or random — try a real news headline."

    return True, ""
