"""
Real-time news search via NewsAPI.org for fact-check context.
Set NEWS_API_KEY in the environment. If missing, callers still get an empty context
so the AI can answer from general knowledge with lower certainty.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List

import requests

logger = logging.getLogger(__name__)

NEWS_EVERYTHING_URL = "https://newsapi.org/v2/everything"


def extract_search_query(text: str, max_words: int = 12) -> str:
    """Short query from user text — works for headlines and one-line claims."""
    blob = str(text).strip()
    if not blob:
        return ""
    words = re.findall(r"[A-Za-z0-9]+", blob)
    if not words:
        return blob[:120]
    return " ".join(words[:max_words])


def fetch_news_context(statement: str, timeout: int = 18) -> Dict[str, Any]:
    """
    Fetch recent articles related to the statement for AI grounding.

    Returns:
        ok: whether remote request succeeded (may still have 0 articles)
        articles: list of {title, description, url, source}
        error: optional short message when misconfigured or HTTP failed
        query: search string used
    """
    statement = str(statement).strip()
    query = extract_search_query(statement)
    out: Dict[str, Any] = {
        "ok": False,
        "articles": [],
        "error": None,
        "query": query,
    }

    api_key = os.environ.get("NEWS_API_KEY", "").strip()
    if not api_key:
        out["error"] = "NEWS_API_KEY not set — limited external verification."
        logger.warning("NEWS_API_KEY missing; search context empty.")
        return out

    if not query:
        out["error"] = "Could not build search query from input."
        return out

    try:
        response = requests.get(
            NEWS_EVERYTHING_URL,
            params={
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 8,
                "apiKey": api_key,
            },
            timeout=timeout,
        )
        if response.status_code != 200:
            out["error"] = f"News API HTTP {response.status_code}"
            logger.warning("News API error: %s %s", response.status_code, response.text[:200])
            return out

        data = response.json()
        if data.get("status") != "ok":
            out["error"] = str(data.get("message", "News API error"))
            return out

        raw = data.get("articles") or []
        articles: List[Dict[str, str]] = []
        for item in raw[:8]:
            title = (item.get("title") or "").strip()
            desc = (item.get("description") or item.get("content") or "").strip()
            url = (item.get("url") or "").strip()
            src = ""
            s = item.get("source")
            if isinstance(s, dict):
                src = str(s.get("name") or "")
            if title or desc:
                articles.append(
                    {
                        "title": title[:300],
                        "description": desc[:500],
                        "url": url[:500],
                        "source": src[:120],
                    }
                )

        out["articles"] = articles
        out["ok"] = True
        logger.info("News search: query=%r articles=%d", query, len(articles))
    except requests.RequestException as exc:
        out["error"] = "News search request failed."
        logger.exception("News API request failed: %s", exc)

    return out


def format_articles_for_prompt(articles: List[Dict[str, str]], max_chars: int = 6000) -> str:
    """Compact bullet list for LLM prompt."""
    if not articles:
        return "(No recent news articles retrieved.)"

    lines: List[str] = []
    total = 0
    for i, a in enumerate(articles, 1):
        title = a.get("title", "")
        desc = a.get("description", "")
        bit = f"{i}. {title}\n   {desc}\n"
        if total + len(bit) > max_chars:
            break
        lines.append(bit.strip())
        total += len(bit)
    return "\n".join(lines) if lines else "(No recent news articles retrieved.)"
