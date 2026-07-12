"""
NewsAPI-based corroboration — checks whether recent headlines support or contradict a claim.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from model.labels import LABEL_FAKE, LABEL_MISLEADING, LABEL_REAL, LABEL_UNKNOWN

_DEBUNK_WORDS = re.compile(
    r"\b(fake|false|hoax|debunk|incorrect|misleading|rumou?r|not true|did not|didn't)\b",
    re.I,
)
_SUPPORT_WORDS = re.compile(
    r"\b(confirm|official|true|announces?|appointed|elected|wins?|won)\b",
    re.I,
)


def verify_with_news(
    statement: str,
    search_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Lightweight signal from NewsAPI article titles/descriptions.
    """
    articles: List[Dict[str, str]] = search_bundle.get("articles") or []
    out: Dict[str, Any] = {
        "available": bool(search_bundle.get("ok")),
        "label": LABEL_UNKNOWN,
        "confidence": 0.0,
        "explanation": "",
        "article_count": len(articles),
        "error": search_bundle.get("error"),
    }

    if not articles:
        if not search_bundle.get("ok"):
            out["explanation"] = search_bundle.get("error") or "No news articles retrieved."
        else:
            out["explanation"] = "No recent articles found for this claim."
        return out

    debunk_hits = 0
    support_hits = 0
    combined_text = []

    for article in articles[:8]:
        blob = f"{article.get('title', '')} {article.get('description', '')}"
        combined_text.append(blob)
        if _DEBUNK_WORDS.search(blob):
            debunk_hits += 1
        if _SUPPORT_WORDS.search(blob):
            support_hits += 1

    corpus = " ".join(combined_text).lower()
    statement_l = str(statement).lower()
    entity_overlap = 0
    for token in re.findall(r"[a-z]{4,}", statement_l):
        if token in corpus:
            entity_overlap += 1

    out["available"] = True

    if debunk_hits >= 2 or (debunk_hits >= 1 and support_hits == 0):
        out["label"] = LABEL_FAKE
        out["confidence"] = min(88.0, 60.0 + debunk_hits * 12.0)
        out["explanation"] = (
            f"Recent news headlines ({debunk_hits}) suggest this claim is disputed or false."
        )
    elif support_hits >= 2 and debunk_hits == 0:
        out["label"] = LABEL_REAL
        out["confidence"] = min(82.0, 55.0 + support_hits * 10.0)
        out["explanation"] = (
            f"Recent news coverage ({len(articles)} articles) appears consistent with the claim."
        )
    elif entity_overlap >= 2 and debunk_hits == 0:
        out["label"] = LABEL_MISLEADING
        out["confidence"] = 58.0
        out["explanation"] = (
            "News mentions related topics but does not clearly confirm the exact claim."
        )
    else:
        out["label"] = LABEL_MISLEADING
        out["confidence"] = 52.0
        out["explanation"] = (
            f"Retrieved {len(articles)} article(s); evidence is mixed or inconclusive."
        )

    return out
