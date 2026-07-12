"""
AI-powered factual verification using OpenAI or Google Gemini.
Combines user statement + optional News API excerpts in one structured prompt.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

from backend.confidence_display import format_display_confidence
from backend.search_service import format_articles_for_prompt

logger = logging.getLogger(__name__)


def _openai_error_snippet(response_text: str) -> str:
    """Short message from OpenAI error JSON for user-facing errors."""
    try:
        obj = json.loads(response_text)
        err = obj.get("error") or {}
        return str(err.get("message") or err.get("type") or response_text)[:240]
    except Exception:
        return str(response_text)[:240]


_FACTCHECK_JSON_RULES = (
    "Rules: "
    "'true' ONLY if the statement is substantially accurate. "
    "'false' if it is wrong, false, or seriously misleading. "
    "'unclear' only if you truly cannot tell. "
    "CRITICAL: factual_verdict MUST match your explanation. "
    "If you write that the claim is incorrect/wrong/did not happen, factual_verdict MUST be \"false\". "
    "Never return \"true\" while saying the statement is incorrect. "
    "For sports (IPL), geography, and dates use well-known facts. "
    "Do NOT call a claim false only because a cricket score seems low — verify history. "
    "Example TRUE fact: RCB were all out for 49 vs KKR in IPL 2017. "
    "If you are unsure, return unclear — do not guess false."
)

_FALSE_EXPLANATION_SIGNALS = (
    "statement is incorrect",
    "is incorrect",
    "is wrong",
    "is false",
    "is not true",
    "not true",
    "not accurate",
    "factually incorrect",
    "did not win",
    "didn't win",
    "never won",
    "has not won",
    "hasn't won",
    "but not in",
    "not in 2019",
    "was won by the",
    "was won by ",
    "however, in",
    "contradicts",
    "misleading",
    "inaccurate",
    "does not match",
    "doesn't match",
    "did not happen",
    "didn't happen",
)

_TRUE_EXPLANATION_SIGNALS = (
    "statement is correct",
    "is correct",
    "is true",
    "substantially accurate",
    "accurate statement",
    "this is true",
    "did win",
)


_UNCERTAIN_FAKE_PHRASES = (
    "likely incorrect",
    "probably incorrect",
    "without further information",
    "difficult to determine",
    "hard to determine",
    "cannot determine",
    "can't determine",
    "unclear whether",
    "may be incorrect",
    "might be incorrect",
    "unlikely that",
    "it's unlikely",
    "it is unlikely",
)


def _reconcile_verdict_with_explanation(label: str, explanation: str) -> str:
    """Fix Groq/Gemini returning true while explanation says the claim is false."""
    exp = str(explanation or "").lower()

    if label == "Fake" and any(p in exp for p in _UNCERTAIN_FAKE_PHRASES):
        logger.warning("AI returned Fake but explanation is uncertain; using Unclear.")
        return "Unclear"

    _GIBBERISH_EXPLANATION_SIGNALS = (
        "jumbled",
        "does not form a coherent",
        "not form a coherent",
        "not a recognizable",
        "random collection of letters",
        "gibberish",
        "meaningless",
        "not a real",
        "does not match any known",
        "not match any known",
    )
    if label == "Fake" and any(s in exp for s in _GIBBERISH_EXPLANATION_SIGNALS):
        logger.warning("AI returned Fake for unclear/gibberish input; using Unclear.")
        return "Unclear"

    false_score = sum(1 for s in _FALSE_EXPLANATION_SIGNALS if s in exp)
    true_score = sum(1 for s in _TRUE_EXPLANATION_SIGNALS if s in exp)

    if false_score > true_score and false_score >= 1 and label == "Real":
        logger.warning(
            "AI returned Real but explanation implies false (false=%s true=%s); overriding to Fake.",
            false_score,
            true_score,
        )
        return "Fake"
    if true_score > false_score and true_score >= 1 and label == "Fake":
        logger.warning(
            "AI returned Fake but explanation implies true; overriding to Real."
        )
        return "Real"
    return label


def _extract_json(content: str) -> Optional[dict]:
    content = str(content).strip()
    if content.startswith("```"):
        content = re.sub(r"^```\w*\n?", "", content)
        content = re.sub(r"\n?```$", "", content)
    try:
        return json.loads(content)
    except Exception:
        return None


def _openai_fact_check(statement: str, news_block: str, search_meta: str) -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    system = (
        "You are a careful fact-checking assistant. Judge whether the user's statement "
        "is factually correct using general knowledge and any news excerpts provided. "
        "Respond with a single JSON object only, no markdown."
    )
    user = (
        f"Statement to verify:\n\"{statement}\"\n\n"
        f"Context (recent news excerpts — may be incomplete or off-topic):\n{news_block}\n\n"
        f"Search note: {search_meta}\n\n"
        "Return JSON with exactly these keys:\n"
        '{ "factual_verdict": "true" | "false" | "unclear", '
        '"confidence": <number 0-100>, '
        '"explanation": "<2-5 short sentences in plain English>", '
        '"needs_verification": <true or false> }\n'
        f"{_FACTCHECK_JSON_RULES} "
        "Set needs_verification true when evidence is weak or contradictory."
    )

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.15,
            "max_tokens": 500,
        },
        timeout=60,
    )
    if response.status_code != 200:
        snippet = _openai_error_snippet(response.text)
        if response.status_code == 429:
            raise RuntimeError(
                f"OpenAI quota or rate limit (HTTP 429): {snippet} "
                "Add credits at https://platform.openai.com/account/billing — or use Gemini (GOOGLE_API_KEY)."
            )
        raise RuntimeError(f"OpenAI HTTP {response.status_code}: {snippet}")

    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Unexpected OpenAI response shape.") from exc
    parsed = _extract_json(content)
    if not parsed:
        raise RuntimeError("OpenAI returned non-JSON output.")
    return parsed


def _groq_fact_check(statement: str, news_block: str, search_meta: str) -> Dict[str, Any]:
    """
    Groq Cloud — OpenAI-compatible API, generous free tier (sign up at console.groq.com).
    """
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    system = (
        "You are a careful fact-checking assistant. Judge whether the user's statement "
        "is factually correct using general knowledge and any news excerpts provided. "
        "Respond with a single JSON object only, no markdown."
    )
    user = (
        f"Statement to verify:\n\"{statement}\"\n\n"
        f"Context (recent news excerpts — may be incomplete or off-topic):\n{news_block}\n\n"
        f"Search note: {search_meta}\n\n"
        "Return JSON with exactly these keys:\n"
        '{ "factual_verdict": "true" | "false" | "unclear", '
        '"confidence": <number 0-100>, '
        '"explanation": "<2-5 short sentences in plain English>", '
        '"needs_verification": <true or false> }\n'
        f"{_FACTCHECK_JSON_RULES} "
        "Set needs_verification true when evidence is weak or contradictory."
    )

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.15,
            "max_tokens": 500,
        },
        timeout=60,
    )
    if response.status_code != 200:
        snippet = _openai_error_snippet(response.text)
        if response.status_code == 429:
            raise RuntimeError(f"Groq rate limit (429): {snippet}")
        raise RuntimeError(f"Groq HTTP {response.status_code}: {snippet}")

    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Unexpected Groq response shape.") from exc
    parsed = _extract_json(content)
    if not parsed:
        raise RuntimeError("Groq returned non-JSON output.")
    return parsed


def _ollama_fact_check(statement: str, news_block: str, search_meta: str) -> Dict[str, Any]:
    """
    Local Ollama — 100% free after install; set USE_OLLAMA=1 and run `ollama serve`.
    """
    base = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", "llama3.2")
    prompt = (
        "You are a fact-checking assistant. Reply with JSON only, no markdown.\n\n"
        f"Statement: \"{statement}\"\n\n"
        f"News excerpts (may be empty):\n{news_block}\n\n"
        f"Note: {search_meta}\n\n"
        '{ "factual_verdict": "true" | "false" | "unclear", '
        '"confidence": 0-100, "explanation": "plain English", "needs_verification": true or false }'
    )
    try:
        response = requests.post(
            f"{base}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=120,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Ollama not reachable at {base}. Install from https://ollama.com, "
            f"run `ollama pull {model}`, then `ollama serve`, and set USE_OLLAMA=1."
        ) from exc

    if response.status_code != 200:
        raise RuntimeError(f"Ollama HTTP {response.status_code}: {response.text[:220]}")

    data = response.json()
    content = (data.get("message") or {}).get("content") or ""
    parsed = _extract_json(content)
    if not parsed:
        raise RuntimeError("Ollama returned non-JSON output. Try a larger model or repeat the question.")
    return parsed


def _auto_provider_chain(statement: str, news_block: str, search_meta: str) -> Tuple[Dict[str, Any], str]:
    """Try free-friendly providers first, then fallbacks. Set FREE_AI_FIRST=0 for OpenAI-first."""
    groq_k = bool(os.environ.get("GROQ_API_KEY", "").strip())
    ollama_on = os.environ.get("USE_OLLAMA", "").strip() == "1"
    google_k = bool((os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip())
    openai_k = bool(os.environ.get("OPENAI_API_KEY", "").strip())

    free_first = os.environ.get("FREE_AI_FIRST", "1").strip() != "0"
    prefer_gemini = os.environ.get("PREFER_GEMINI", "1").strip() != "0"
    order: List[str] = []
    if free_first:
        if google_k and prefer_gemini:
            order.append("gemini")
        if groq_k:
            order.append("groq")
        if ollama_on:
            order.append("ollama")
        if google_k and "gemini" not in order:
            order.append("gemini")
        if openai_k:
            order.append("openai")
    else:
        if openai_k:
            order.append("openai")
        if google_k:
            order.append("gemini")
        if groq_k:
            order.append("groq")
        if ollama_on:
            order.append("ollama")

    if not order:
        raise RuntimeError(
            "No AI keys configured. FREE (no payment): (1) Create GROQ_API_KEY at https://console.groq.com "
            "— free tier. (2) Or install Ollama from https://ollama.com, `ollama pull llama3.2`, "
            "`ollama serve`, then set USE_OLLAMA=1."
        )

    last: Optional[RuntimeError] = None
    for kind in order:
        try:
            if kind == "groq":
                return _groq_fact_check(statement, news_block, search_meta), "groq"
            if kind == "ollama":
                return _ollama_fact_check(statement, news_block, search_meta), "ollama"
            if kind == "gemini":
                return _gemini_with_openai_fallback(statement, news_block, search_meta)
            if kind == "openai":
                return _openai_with_gemini_fallback(statement, news_block, search_meta)
        except RuntimeError as exc:
            last = exc
            logger.warning("Provider %s failed, trying next: %s", kind, str(exc)[:160])

    raise RuntimeError(
        "All configured AI providers failed (quota or network). "
        "Easiest free fix: get a Groq API key at https://console.groq.com and set GROQ_API_KEY. "
        f"Last error: {str(last)[:260]}"
    ) from last


# Model IDs change over time; v1beta often rejects bare "gemini-1.5-flash". Try in order on 404.
_GEMINI_MODEL_FALLBACKS = (
    "gemini-2.0-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro-latest",
)


def _gemini_fact_check(statement: str, news_block: str, search_meta: str) -> Dict[str, Any]:
    api_key = (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY is not set.")

    env_model = os.environ.get("GEMINI_MODEL", "").strip()
    candidates = []
    if env_model:
        candidates.append(env_model)
    for m in _GEMINI_MODEL_FALLBACKS:
        if m not in candidates:
            candidates.append(m)

    prompt = (
        "You are a fact-checking assistant. Judge if the statement is factually correct.\n\n"
        f"Statement: \"{statement}\"\n\n"
        f"News excerpts (may be incomplete):\n{news_block}\n\n"
        f"Note: {search_meta}\n\n"
        "Reply with JSON only:\n"
        '{ "factual_verdict": "true" | "false" | "unclear", '
        '"confidence": number 0-100, '
        '"explanation": "plain English", '
        '"needs_verification": true or false }'
    )

    last_body = ""
    for model in candidates:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=60,
        )
        last_body = response.text[:500]
        if response.status_code == 404:
            logger.warning("Gemini model not available: %s — trying next.", model)
            continue
        if response.status_code == 429:
            # Same quota for all models — do not waste retries.
            raise RuntimeError(
                "Gemini API quota exceeded (HTTP 429). Free-tier generateContent requests are exhausted "
                "or not enabled for this key (Google may show limit: 0). Enable billing / upgrade plan in "
                "Google AI Studio, wait for the daily reset, or set OPENAI_API_KEY to use OpenAI instead. "
                "https://ai.google.dev/gemini-api/docs/rate-limits"
            )
        if response.status_code != 200:
            raise RuntimeError(f"Gemini HTTP {response.status_code}: {response.text[:400]}")

        data = response.json()
        try:
            text_out = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Unexpected Gemini response (blocked or empty).") from exc
        parsed = _extract_json(text_out)
        if not parsed:
            raise RuntimeError("Gemini returned non-JSON output.")
        logger.info("Gemini fact-check used model: %s", model)
        return parsed

    raise RuntimeError(
        "No Gemini model worked (all returned 404). Set GEMINI_MODEL to a model from "
        "https://ai.google.dev/gemini-api/docs/models — last response: "
        + last_body[:280]
    )


def _normalize_verdict(parsed: Dict[str, Any], provider: str = "ai") -> Dict[str, Any]:
    """Map API JSON: Real = factually sound, Fake = false/misleading, Unclear = insufficient evidence."""
    verdict_raw = str(parsed.get("factual_verdict", "")).strip().lower()
    confidence = float(parsed.get("confidence", 50))
    confidence = max(0.0, min(100.0, confidence))
    explanation = str(parsed.get("explanation", "")).strip() or "No explanation returned."
    needs = bool(parsed.get("needs_verification", False))

    if verdict_raw in {"unclear", "unknown", "maybe"}:
        return {
            "prediction": "Unclear",
            "confidence": round(min(confidence, 55.0), 2),
            "explanation": explanation,
            "needs_verification": True,
            "factual_verdict": verdict_raw,
        }

    if verdict_raw in {"true", "t", "yes"}:
        label = "Real"
    elif verdict_raw in {"false", "f", "no"}:
        label = "Fake"
    else:
        return {
            "prediction": "Unclear",
            "confidence": round(min(confidence, 50.0), 2),
            "explanation": explanation or "Could not determine factual verdict from model output.",
            "needs_verification": True,
            "factual_verdict": verdict_raw or "unclear",
        }

    label = _reconcile_verdict_with_explanation(label, explanation)
    if label == "Fake" and verdict_raw in {"true", "t", "yes"}:
        explanation = (
            f"This claim is factually incorrect. {explanation}"
        ).strip()
    display_conf = format_display_confidence(label, confidence, provider)

    return {
        "prediction": label,
        "confidence": display_conf,
        "explanation": explanation,
        "needs_verification": needs if label == "Unclear" else False,
        "factual_verdict": verdict_raw,
    }


def _gemini_with_openai_fallback(statement: str, news_block: str, search_meta: str) -> Tuple[Dict[str, Any], str]:
    """Run Gemini; on 429/quota try Groq (free), then OpenAI if configured."""
    try:
        return _gemini_fact_check(statement, news_block, search_meta), "gemini"
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "429" not in msg and "quota" not in msg:
            raise
        groq_k = os.environ.get("GROQ_API_KEY", "").strip()
        if groq_k:
            logger.warning("Gemini quota/rate limit hit; falling back to Groq.")
            try:
                return _groq_fact_check(statement, news_block, search_meta), "groq"
            except RuntimeError as groq_exc:
                logger.warning("Groq fallback failed: %s", str(groq_exc)[:160])
        openai_k = os.environ.get("OPENAI_API_KEY", "").strip()
        if openai_k:
            logger.warning("Gemini quota hit; falling back to OpenAI.")
            try:
                return _openai_fact_check(statement, news_block, search_meta), "openai"
            except RuntimeError as exc2:
                raise RuntimeError(
                    "Gemini quota exceeded; Groq and OpenAI also failed. "
                    f"Last error: {str(exc2)[:200]}"
                ) from exc2
        if groq_k:
            raise RuntimeError(
                "Gemini quota exceeded and Groq fallback failed. "
                "Check GROQ_API_KEY at https://console.groq.com or wait for Gemini daily reset."
            ) from exc
        raise RuntimeError(
            "Gemini quota exceeded. Set GROQ_API_KEY (free at console.groq.com) for automatic backup, "
            "or wait for Gemini daily reset: https://ai.google.dev/gemini-api/docs/rate-limits"
        ) from exc


def _openai_with_gemini_fallback(statement: str, news_block: str, search_meta: str) -> Tuple[Dict[str, Any], str]:
    """Run OpenAI; on 429/quota use Gemini if GOOGLE_API_KEY / GEMINI_API_KEY is set."""
    try:
        return _openai_fact_check(statement, news_block, search_meta), "openai"
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "429" not in msg and "quota" not in msg and "insufficient" not in msg:
            raise
        google = (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()
        if not google:
            raise RuntimeError(
                "OpenAI quota exceeded (insufficient_quota / 429). Add billing or credits: "
                "https://platform.openai.com/account/billing — OR set GOOGLE_API_KEY to use Gemini."
            ) from exc
        logger.warning("OpenAI quota/rate limit hit; falling back to Gemini.")
        try:
            return _gemini_fact_check(statement, news_block, search_meta), "gemini"
        except RuntimeError as exc2:
            raise RuntimeError(
                "Both AI providers failed: OpenAI hit quota, then Gemini also failed. "
                f"Gemini: {str(exc2)[:200]}"
            ) from exc2


def run_fact_check(statement: str, search_bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run AI fact check with optional news context.

    search_bundle: output from search_service.fetch_news_context
    """
    from model.fact_claims import check_factual_claim

    statement = str(statement).strip()

    ruled = check_factual_claim(statement)
    if ruled is not None:
        label = str(ruled["prediction"])
        conf = format_display_confidence(label, float(ruled["confidence"]), "facts")
        return {
            "prediction": label,
            "confidence": conf,
            "explanation": str(ruled["explanation"]),
            "needs_verification": bool(ruled.get("needs_verification", False)),
            "provider": "facts",
        }

    from model.input_quality import assess_input_quality

    ok, reason = assess_input_quality(statement)
    if not ok:
        return {
            "prediction": "Unclear",
            "confidence": 55.0,
            "explanation": reason,
            "needs_verification": True,
            "provider": "input",
        }

    articles = search_bundle.get("articles") or []
    news_block = format_articles_for_prompt(articles)

    today_str = datetime.now().strftime("%A, %B %d, %Y")
    date_hint = f"Today's calendar date is {today_str} (use for 'today is Monday' style claims)."

    if search_bundle.get("ok"):
        search_meta = (
            f"Retrieved {len(articles)} article(s) for query {search_bundle.get('query', '')!r}. "
            f"{date_hint}"
        )
    else:
        err = search_bundle.get("error") or "No external search."
        search_meta = (
            f"External news search unavailable or empty ({err}). Rely on careful reasoning. {date_hint}"
        )

    provider = (os.environ.get("FACTCHECK_AI_PROVIDER") or "auto").strip().lower()
    parsed: Dict[str, Any]
    provider_used = ""

    if provider == "openai":
        parsed, provider_used = _openai_with_gemini_fallback(statement, news_block, search_meta)
    elif provider in {"gemini", "google"}:
        parsed, provider_used = _gemini_with_openai_fallback(statement, news_block, search_meta)
    elif provider == "auto":
        parsed, provider_used = _auto_provider_chain(statement, news_block, search_meta)
    elif provider == "groq":
        parsed = _groq_fact_check(statement, news_block, search_meta)
        provider_used = "groq"
    elif provider == "ollama":
        parsed = _ollama_fact_check(statement, news_block, search_meta)
        provider_used = "ollama"
    else:
        # auto: Groq / Ollama first (free-friendly), then Gemini / OpenAI
        parsed, provider_used = _auto_provider_chain(statement, news_block, search_meta)

    normalized = _normalize_verdict(parsed, provider=provider_used or "ai")
    normalized["provider"] = provider_used
    logger.info(
        "Fact-check done via %s: label=%s confidence=%s needs_verification=%s",
        provider_used,
        normalized["prediction"],
        normalized["confidence"],
        normalized["needs_verification"],
    )
    return normalized
