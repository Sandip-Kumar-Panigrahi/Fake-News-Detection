"""
Production hybrid fact-checking orchestrator (optimized for speed).

Fast path: rules + ML only (~instant) when confidence is high.
Full path: parallel external APIs with short timeouts.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from backend.ai_service import get_ai_reasoning
from backend.fact_check_api import search_fact_checks
from backend.ml_service import predict_news
from backend.news_verification import verify_with_news
from backend.search_service import fetch_news_context
from model.fact_claims import check_factual_claim
from model.input_quality import assess_input_quality
from model.confidence_calibrate import calibrate_confidence
from model.labels import (
    LABEL_FAKE,
    LABEL_MISLEADING,
    LABEL_REAL,
    LABEL_UNKNOWN,
    label_score,
    normalize_label,
)

logger = logging.getLogger(__name__)

WEIGHTS = {
    "rules": 1.0,
    "ml": 0.38,
    "google_fact_check": 0.28,
    "news": 0.14,
    "ai_reasoning": 0.12,
}

RULE_OVERRIDE_CONFIDENCE = 92.0
STRONG_AGREEMENT_BOOST = 6.0
ML_FAST_CONFIDENCE = 82.0

# Short timeouts so failed APIs do not hang the UI
API_TIMEOUT_NEWS = int(os.environ.get("NEWS_API_TIMEOUT", "4"))
API_TIMEOUT_FACTCHECK = int(os.environ.get("FACTCHECK_API_TIMEOUT", "4"))
API_TIMEOUT_AI = int(os.environ.get("AI_REASONING_TIMEOUT", "12"))


def _fast_mode_enabled() -> bool:
    return os.environ.get("HYBRID_FAST_MODE", "1").strip() != "0"


def _google_fact_check_enabled() -> bool:
    return os.environ.get("ENABLE_GOOGLE_FACT_CHECK", "0").strip() == "1"


def _ai_reasoning_enabled() -> bool:
    return os.environ.get("ENABLE_AI_REASONING", "0").strip() == "1"


@dataclass
class Signal:
    source: str
    label: str
    confidence: float
    explanation: str
    available: bool = True
    weight: float = 0.0
    meta: Optional[Dict[str, Any]] = None


def _signal_from_rules(text: str) -> Optional[Signal]:
    ruled = check_factual_claim(text)
    if ruled is None:
        return None
    return Signal(
        source="rules",
        label=normalize_label(ruled["prediction"]),
        confidence=float(ruled["confidence"]),
        explanation=str(ruled["explanation"]),
        available=True,
        weight=WEIGHTS["rules"],
        meta={"needs_verification": bool(ruled.get("needs_verification", False))},
    )


def _signal_from_ml(text: str) -> Signal:
    try:
        ml = predict_news(text)
        meta = {
            "needs_verification": bool(ml.get("needs_verification", False)),
            "fake_prob": ml.get("fake_prob"),
            "real_prob": ml.get("real_prob"),
        }
        return Signal(
            source="ml",
            label=normalize_label(ml["prediction"]),
            confidence=float(ml["confidence"]),
            explanation=str(ml["explanation"]),
            available=True,
            weight=WEIGHTS["ml"],
            meta=meta,
        )
    except Exception as exc:
        logger.warning("ML signal unavailable: %s", exc)
        return Signal(
            source="ml",
            label=LABEL_UNKNOWN,
            confidence=0.0,
            explanation="ML model unavailable.",
            available=False,
            weight=WEIGHTS["ml"],
        )


def _can_use_fast_path(rule_signal: Optional[Signal], ml_signal: Signal) -> bool:
    if not _fast_mode_enabled():
        return False
    if rule_signal and rule_signal.confidence >= RULE_OVERRIDE_CONFIDENCE:
        return True
    if (
        ml_signal.available
        and ml_signal.label in (LABEL_FAKE, LABEL_REAL)
        and ml_signal.confidence >= ML_FAST_CONFIDENCE
    ):
        return True
    return False


def _merge_signals(signals: List[Signal]) -> Dict[str, Any]:
    active = [s for s in signals if s.available and s.label != LABEL_UNKNOWN]
    if not active:
        return {
            "label": LABEL_MISLEADING,
            "confidence": 50.0,
            "needs_verification": True,
        }

    for s in active:
        if s.source == "rules" and s.confidence >= RULE_OVERRIDE_CONFIDENCE:
            lbl = s.label
            conf = calibrate_confidence(lbl, s.confidence, active)
            return {
                "label": lbl,
                "confidence": conf,
                "needs_verification": bool((s.meta or {}).get("needs_verification", False)),
                "override": "rules",
            }

    score = 0.0
    weight_sum = 0.0
    conf_accum = 0.0

    for s in active:
        w = s.weight * max(s.confidence, 40.0) / 100.0
        score += label_score(s.label) * w
        weight_sum += w
        conf_accum += s.confidence * s.weight

    if weight_sum <= 0:
        avg_label = LABEL_MISLEADING
        final_conf = 50.0
    else:
        normalized = score / weight_sum
        if normalized >= 0.35:
            avg_label = LABEL_REAL
        elif normalized <= -0.35:
            avg_label = LABEL_FAKE
        else:
            avg_label = LABEL_MISLEADING
        final_conf = conf_accum / sum(s.weight for s in active)

    labels = [s.label for s in active]
    if LABEL_FAKE in labels and LABEL_REAL in labels:
        avg_label = LABEL_MISLEADING
        final_conf = max(55.0, final_conf - 10.0)

    if len(set(labels)) == 1 and len(active) >= 2:
        final_conf = min(99.0, final_conf + STRONG_AGREEMENT_BOOST)

    needs_verification = avg_label == LABEL_MISLEADING or final_conf < 62.0
    final_conf = calibrate_confidence(avg_label, final_conf, active)
    return {
        "label": avg_label,
        "confidence": round(float(final_conf), 2),
        "needs_verification": needs_verification,
    }


def _fetch_external_parallel(statement: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """News + Google Fact Check in parallel (short timeouts)."""
    search_bundle: Dict[str, Any] = {
        "ok": False,
        "articles": [],
        "error": None,
        "query": "",
    }
    google_data: Dict[str, Any] = {
        "available": False,
        "label": LABEL_UNKNOWN,
        "confidence": 0.0,
        "claims": [],
        "error": None,
    }

    futures = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        if os.environ.get("NEWS_API_KEY", "").strip():
            futures["news"] = pool.submit(
                fetch_news_context, statement, API_TIMEOUT_NEWS
            )
        if _google_fact_check_enabled():
            futures["gfc"] = pool.submit(
                search_fact_checks, statement, timeout=API_TIMEOUT_FACTCHECK
            )

        for key, fut in futures.items():
            try:
                result = fut.result(timeout=API_TIMEOUT_NEWS + 2)
                if key == "news":
                    search_bundle = result
                else:
                    google_data = result
            except Exception as exc:
                logger.warning("External API %s failed: %s", key, exc)

    return search_bundle, google_data


def _build_explanation(
    signals: List[Signal],
    final_label: str,
    rule_signal: Optional[Signal],
) -> str:
    if rule_signal and rule_signal.explanation:
        return rule_signal.explanation
    for s in signals:
        if s.source == "ml" and s.explanation:
            return s.explanation
    for s in signals:
        if s.available and s.explanation:
            return s.explanation
    return f"Analysis indicates this claim is {final_label}."


def run_hybrid_fact_check(statement: str) -> Dict[str, Any]:
    statement = str(statement or "").strip()
    ok, reason = assess_input_quality(statement)
    if not ok:
        return _build_response(
            statement=statement,
            final_label=LABEL_MISLEADING,
            final_confidence=55.0,
            explanation=reason,
            needs_verification=True,
            signals=[],
            search_bundle={"articles": [], "ok": False, "error": reason},
            google_fact_check={"claims": []},
            provider="input",
            fast=True,
        )

    signals: List[Signal] = []
    rule_signal = _signal_from_rules(statement)
    if rule_signal:
        signals.append(rule_signal)

    ml_signal = _signal_from_ml(statement)
    signals.append(ml_signal)

    # FAST PATH — skip slow APIs (target: under 300 ms)
    if _can_use_fast_path(rule_signal, ml_signal):
        merged = _merge_signals(signals)
        display_conf = calibrate_confidence(merged["label"], merged["confidence"], signals)
        explanation = _build_explanation(signals, merged["label"], rule_signal)
        logger.info("Fast path: label=%s conf=%s", merged["label"], display_conf)
        return _build_response(
            statement=statement,
            final_label=merged["label"],
            final_confidence=display_conf,
            explanation=explanation,
            needs_verification=merged["needs_verification"],
            signals=signals,
            search_bundle={"articles": [], "ok": False, "error": "Skipped (fast mode)"},
            google_fact_check={"claims": [], "available": False},
            provider="rules+ml" if rule_signal else "ml",
            fast=True,
        )

    # FULL PATH — parallel external calls
    search_bundle, google_data = _fetch_external_parallel(statement)

    news_signal_data = verify_with_news(statement, search_bundle)
    signals.append(
        Signal(
            source="news",
            label=normalize_label(news_signal_data["label"], LABEL_UNKNOWN),
            confidence=float(news_signal_data["confidence"]),
            explanation=str(news_signal_data.get("explanation") or ""),
            available=bool(news_signal_data.get("available")),
            weight=WEIGHTS["news"],
        )
    )

    g_label = normalize_label(google_data.get("label", LABEL_UNKNOWN), LABEL_UNKNOWN)
    signals.append(
        Signal(
            source="google_fact_check",
            label=g_label,
            confidence=float(google_data.get("confidence") or 0.0),
            explanation=str(google_data.get("explanation") or ""),
            available=bool(google_data.get("available")),
            weight=WEIGHTS["google_fact_check"],
            meta={"claims": google_data.get("claims") or []},
        )
    )

    merged = _merge_signals(signals)
    final_label = merged["label"]
    final_confidence = calibrate_confidence(merged["label"], merged["confidence"], signals)
    needs_verification = merged["needs_verification"]
    explanation = _build_explanation(signals, final_label, rule_signal)

    if _ai_reasoning_enabled():
        try:
            ai_block = get_ai_reasoning(
                statement=statement,
                ml_result={
                    "prediction": ml_signal.label,
                    "confidence": ml_signal.confidence,
                    "explanation": ml_signal.explanation,
                },
                merged_hint={
                    "label": final_label,
                    "confidence": final_confidence,
                },
                search_bundle=search_bundle,
                google_fact_check=google_data,
                timeout=API_TIMEOUT_AI,
            )
            if ai_block.get("available") and ai_block.get("explanation"):
                explanation = str(ai_block["explanation"])
        except Exception as exc:
            logger.warning("AI reasoning skipped: %s", exc)

    provider_parts = [s.source for s in signals if s.available]
    provider = "+".join(provider_parts[:4]) if provider_parts else "hybrid"

    return _build_response(
        statement=statement,
        final_label=final_label,
        final_confidence=final_confidence,
        explanation=explanation,
        needs_verification=needs_verification,
        signals=signals,
        search_bundle=search_bundle,
        google_fact_check=google_data,
        provider=provider,
        fast=False,
    )


def _signal_to_dict(s: Signal) -> Dict[str, Any]:
    return {
        "source": s.source,
        "label": s.label,
        "confidence": round(s.confidence, 2),
        "explanation": s.explanation,
        "available": s.available,
    }


def _build_response(
    *,
    statement: str,
    final_label: str,
    final_confidence: float,
    explanation: str,
    needs_verification: bool,
    signals: List[Signal],
    search_bundle: Dict[str, Any],
    google_fact_check: Dict[str, Any],
    provider: str,
    fast: bool,
) -> Dict[str, Any]:
    articles = search_bundle.get("articles") or []
    breakdown = {s.source: _signal_to_dict(s) for s in signals}

    return {
        "prediction": final_label,
        "confidence": round(float(final_confidence), 2),
        "explanation": explanation,
        "needs_verification": needs_verification,
        "provider": provider,
        "fast_mode": fast,
        "hybrid_breakdown": breakdown,
        "ml_result": breakdown.get("ml", {}),
        "google_fact_check": {
            "available": bool(google_fact_check.get("available")),
            "claims": google_fact_check.get("claims") or [],
            "label": google_fact_check.get("label", LABEL_UNKNOWN),
            "confidence": google_fact_check.get("confidence", 0.0),
            "error": google_fact_check.get("error"),
        },
        "news_verification": breakdown.get("news", {}),
        "ai_reasoning": {
            "explanation": explanation,
            "available": bool(explanation),
        },
        "search_summary": {
            "article_count": len(articles),
            "sources": [
                {"title": a.get("title", ""), "url": a.get("url", "")}
                for a in articles[:6]
            ],
        },
        "status": (
            "Needs verification"
            if needs_verification
            else ("Verified" if final_confidence >= 90 else "Needs Review")
        ),
    }
