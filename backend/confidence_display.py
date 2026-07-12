"""Map internal scores to user-facing confidence (Fake=100%, Real≈98%)."""

from __future__ import annotations


def format_display_confidence(label: str, raw_confidence: float, source: str = "") -> float:
    """
    Fake / wrong claims → 100%.
    Real / correct claims → 98% when we have a clear verdict.
    Unclear → keep original score.
    """
    lbl = str(label).strip()
    raw = float(raw_confidence)
    src = str(source or "").strip().lower()

    if lbl == "Fake":
        return 100.0

    if lbl == "Real":
        ai_sources = {"gemini", "google", "openai", "groq", "ollama", "facts", "rules"}
        if src in ai_sources or raw >= 50.0:
            return 98.0
        return round(raw, 2)

    return round(raw, 2)
