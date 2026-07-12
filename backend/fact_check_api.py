"""
Google Fact Check Tools API — real-time claim verification.
https://developers.google.com/fact-check/tools/api/reference/rest/v1alpha1/claims/search
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests

from model.labels import LABEL_FAKE, LABEL_MISLEADING, LABEL_REAL, LABEL_UNKNOWN, normalize_label

logger = logging.getLogger(__name__)

FACT_CHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

_RATING_MAP = {
    "false": LABEL_FAKE,
    "incorrect": LABEL_FAKE,
    "pants on fire": LABEL_FAKE,
    "fake": LABEL_FAKE,
    "misleading": LABEL_MISLEADING,
    "mostly false": LABEL_FAKE,
    "half true": LABEL_MISLEADING,
    "mostly true": LABEL_REAL,
    "true": LABEL_REAL,
    "correct": LABEL_REAL,
}


def _rating_to_label(rating: str) -> str:
    r = str(rating or "").strip().lower()
    for key, label in _RATING_MAP.items():
        if key in r:
            return label
    return LABEL_UNKNOWN


def _confidence_from_reviews(reviews: List[Dict[str, Any]]) -> float:
    if not reviews:
        return 0.0
    scores = []
    for rev in reviews:
        rating = str(rev.get("textualRating") or rev.get("title") or "").lower()
        if any(x in rating for x in ("false", "fake", "incorrect", "pants")):
            scores.append(92.0)
        elif "misleading" in rating or "half" in rating:
            scores.append(72.0)
        elif any(x in rating for x in ("true", "correct", "accurate")):
            scores.append(88.0)
        else:
            scores.append(65.0)
    return round(sum(scores) / len(scores), 2)


def search_fact_checks(query: str, *, timeout: int = 4) -> Dict[str, Any]:
    """
    Query Google's claim database. Returns normalized hybrid signal.
    """
    query = str(query or "").strip()
    out: Dict[str, Any] = {
        "available": False,
        "label": LABEL_UNKNOWN,
        "confidence": 0.0,
        "explanation": "",
        "claims": [],
        "error": None,
    }

    api_key = (
        os.environ.get("GOOGLE_FACT_CHECK_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )
    if not api_key:
        out["error"] = "GOOGLE_FACT_CHECK_API_KEY not set."
        return out

    if not query:
        out["error"] = "Empty query for fact-check search."
        return out

    try:
        response = requests.get(
            FACT_CHECK_URL,
            params={
                "query": query[:500],
                "languageCode": "en",
                "key": api_key,
            },
            timeout=timeout,
        )
        if response.status_code != 200:
            out["error"] = f"Fact Check API HTTP {response.status_code}"
            logger.warning("Google Fact Check error: %s", response.text[:240])
            return out

        data = response.json()
        claims_raw = data.get("claims") or []
        parsed_claims: List[Dict[str, str]] = []
        label_votes: List[str] = []

        for item in claims_raw[:5]:
            claim_text = str(item.get("text") or "")[:400]
            reviews = item.get("claimReview") or []
            for rev in reviews[:3]:
                rating = str(rev.get("textualRating") or rev.get("title") or "")
                publisher = ""
                pub = rev.get("publisher")
                if isinstance(pub, dict):
                    publisher = str(pub.get("name") or "")
                url = str(rev.get("url") or "")
                lbl = _rating_to_label(rating)
                if lbl != LABEL_UNKNOWN:
                    label_votes.append(lbl)
                parsed_claims.append(
                    {
                        "claim": claim_text,
                        "rating": rating,
                        "publisher": publisher,
                        "url": url,
                        "label": lbl,
                    }
                )

        out["claims"] = parsed_claims
        out["available"] = bool(parsed_claims)

        if label_votes:
            fake_n = label_votes.count(LABEL_FAKE)
            real_n = label_votes.count(LABEL_REAL)
            mis_n = label_votes.count(LABEL_MISLEADING)
            if fake_n >= real_n and fake_n >= mis_n:
                out["label"] = LABEL_FAKE
            elif real_n >= fake_n and real_n >= mis_n:
                out["label"] = LABEL_REAL
            elif mis_n > 0:
                out["label"] = LABEL_MISLEADING
            else:
                out["label"] = normalize_label(label_votes[0])
            out["confidence"] = _confidence_from_reviews(
                [{"textualRating": c.get("rating")} for c in parsed_claims]
            )
            top = parsed_claims[0]
            out["explanation"] = (
                f"Google Fact Check: \"{top.get('rating', 'reviewed')}\" "
                f"({top.get('publisher', 'fact-checker')})."
            )
        elif claims_raw:
            out["label"] = LABEL_MISLEADING
            out["confidence"] = 55.0
            out["explanation"] = "Related claims found but no clear true/false rating."
        else:
            out["explanation"] = "No matching fact-check entries in Google's database."

        logger.info(
            "Google Fact Check: claims=%d label=%s",
            len(parsed_claims),
            out["label"],
        )
    except requests.RequestException as exc:
        out["error"] = "Fact Check API request failed."
        logger.exception("Fact Check API failed: %s", exc)

    return out
