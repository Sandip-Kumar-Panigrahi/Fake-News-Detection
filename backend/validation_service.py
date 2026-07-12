from typing import Any, Dict, List


def topic_reliability_score(text: str) -> float:
    """
    Legacy hook — fact-checking no longer uses topic heuristics.
    Returns 0.0 for API compatibility.
    """
    return 0.0


def apply_confidence_balancing(label: str, confidence: float, reliability_score: float) -> float:
    """Keep model/rule confidence; do not inflate to 100%."""
    _ = reliability_score
    return round(float(confidence), 2)


def make_fact_status(needs_verification: bool, confidence: float) -> str:
    if needs_verification:
        return "Needs verification"
    return "Verified" if float(confidence) >= 60.0 else "Needs Review"


def build_fact_result_payload(
    prediction: str,
    confidence: float,
    explanation: str,
    needs_verification: bool,
    search_articles: List[Dict[str, str]],
) -> Dict[str, Any]:
    return {
        "explanation": explanation,
        "needs_verification": bool(needs_verification),
        "ai_result": {
            "label": prediction,
            "confidence": round(float(confidence), 2),
        },
        "final_decision": {
            "label": prediction,
            "confidence": round(float(confidence), 2),
        },
        "search_summary": {
            "article_count": len(search_articles),
            "sources": [
                {"title": a.get("title", ""), "url": a.get("url", "")}
                for a in search_articles[:6]
            ],
        },
        "reliability_score": 0.0,
        "status": make_fact_status(needs_verification, confidence),
    }


def sanitize_hint(prediction: str, needs_verification: bool, explanation: str = "") -> str:
    pl = str(prediction).lower()
    exp = str(explanation or "").lower()
    if pl == "unclear":
        return "Not enough reliable data for a true/false verdict — verify from primary sources."
    if needs_verification:
        return "Not enough external data for a firm verdict — treat this as guidance only."
    if pl == "fake":
        return "This statement is factually incorrect or misleading."
    if any(
        s in exp
        for s in ("incorrect", "is wrong", "did not win", "never won", "was won by")
    ):
        return "The AI explanation noted problems with this claim — treat the headline as not verified."
    return "This claim appears consistent with known facts. For important news, still verify with trusted sources."
