import logging
import os
import sys

from flask import Flask, jsonify, render_template, request, session

PROJECT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from backend.env_loader import load_project_env

load_project_env(PROJECT_DIR)

_LOG_LEVEL = logging.DEBUG if os.environ.get("FLASK_DEBUG", "0") == "1" else logging.INFO
logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from backend.ai_service import run_fact_check
from backend.database import authenticate_user, create_user, init_db, save_prediction
from backend.ml_service import model_available, predict_news, preload_model
from backend.search_service import fetch_news_context
from backend.confidence_display import format_display_confidence
from backend.validation_service import (
    apply_confidence_balancing,
    build_fact_result_payload,
    sanitize_hint,
    topic_reliability_score,
)

app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_DIR, "templates"),
    static_folder=os.path.join(PROJECT_DIR, "static"),
)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "fake-news-final-year-project-key")


def _has_ai_keys() -> bool:
    return bool(
        os.environ.get("GROQ_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("USE_OLLAMA", "").strip() == "1"
    )


def _run_ml_fact_check(news_text: str, ai_failed_reason: str = "") -> dict:
    """
    Last-resort style classifier — never present as a full fact-check.
    Confidence is capped so users do not see misleading 98% on wrong claims.
    """
    result = predict_news(news_text)
    prov = str(result.get("provider") or "ml")
    label = str(result["prediction"])
    conf = float(result["confidence"])
    if prov == "ml" and label in {"Real", "Fake"}:
        conf = min(conf, 52.0)
    explanation = str(result["explanation"])
    needs = bool(result.get("needs_verification", False))
    if prov == "ml" or "style-based" in explanation.lower():
        prefix = (
            "AI fact-check was unavailable (Gemini quota or all APIs failed). "
            "This is only a writing-style guess from the local ML model — NOT verified facts. "
        )
        if ai_failed_reason:
            prefix = f"AI fact-check failed: {ai_failed_reason}. " + prefix
        explanation = prefix + explanation
        needs = True
    elif prov == "facts":
        needs = bool(result.get("needs_verification", False))
    else:
        needs = True
    return {
        "prediction": label,
        "confidence": conf,
        "explanation": explanation,
        "needs_verification": needs,
        "provider": prov,
    }


def _ai_status_payload() -> dict:
    gemini = bool(
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )
    groq = bool(os.environ.get("GROQ_API_KEY", "").strip())
    return {
        "ai_configured": _has_ai_keys(),
        "gemini_configured": gemini,
        "groq_configured": groq,
        "ml_model_ready": model_available(),
        "news_api_configured": bool(os.environ.get("NEWS_API_KEY", "").strip()),
        "env_file_exists": os.path.isfile(os.path.join(PROJECT_DIR, ".env")),
    }


@app.after_request
def disable_browser_cache_for_ui(response):
    """Avoid stale HTML/JS during development (fixes old error text + old app.js in cache)."""
    path = request.path or ""
    ctype = (response.headers.get("Content-Type") or "").lower()
    if path.endswith((".js", ".css")) or "text/html" in ctype:
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


@app.route("/")
def home():
    return render_template("index.html", username=session.get("username"))


@app.route("/predict-page")
def predict_page():
    return render_template("predict.html", username=session.get("username"))


@app.route("/about")
def about():
    return render_template("about.html", username=session.get("username"))


@app.route("/api/status")
def api_status():
    return jsonify(_ai_status_payload())


@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json(silent=True) or {}
    news_text = str(data.get("news_text", "")).strip()
    country = str(data.get("country", "Global")).strip() or "Global"

    if not news_text:
        return jsonify({"error": "Please enter some news text."}), 400

    logger.debug("Fact-check request: len=%d country=%s", len(news_text), country)

    fact = None
    provider = ""
    search_bundle = None

    if _has_ai_keys():
        search_bundle = fetch_news_context(news_text)
        try:
            fact = run_fact_check(news_text, search_bundle)
            provider = str(fact.get("provider") or "ai")
        except RuntimeError as exc:
            logger.warning("Fact-check configuration error: %s", exc)
            err_msg = str(exc)
            recovered = False
            try:
                from datetime import datetime

                from backend.ai_service import _auto_provider_chain, _normalize_verdict
                from backend.search_service import format_articles_for_prompt

                articles = (search_bundle or {}).get("articles") or []
                news_block = format_articles_for_prompt(articles)
                today_str = datetime.now().strftime("%A, %B %d, %Y")
                search_meta = f"Retry after primary AI error. Today's date: {today_str}."
                parsed, provider_used = _auto_provider_chain(news_text, news_block, search_meta)
                fact = _normalize_verdict(parsed, provider=provider_used)
                provider = provider_used
                recovered = True
                logger.info("Recovered via auto AI chain after provider error.")
            except Exception as retry_exc:
                logger.warning("Auto AI retry failed: %s", retry_exc)
            if not recovered:
                if model_available():
                    logger.info("AI unavailable; falling back to ML model (low trust).")
                    fact = _run_ml_fact_check(news_text, ai_failed_reason=err_msg)
                    provider = "ml"
                else:
                    return jsonify({"error": err_msg}), 503
        except Exception as exc:
            logger.exception("Fact-check failed: %s", exc)
            err_msg = str(exc)
            if model_available():
                logger.info("AI failed; falling back to ML model (low trust).")
                fact = _run_ml_fact_check(news_text, ai_failed_reason=err_msg)
                provider = "ml"
            else:
                return jsonify(
                    {
                        "error": "AI fact-check failed. Try again later or check API keys and quotas.",
                    }
                ), 502
    elif model_available():
        try:
            fact = _run_ml_fact_check(news_text)
            provider = "ml"
            logger.info("Using ML model (no AI API keys configured).")
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 503
    else:
        return jsonify(
            {
                "error": (
                    "No AI keys configured and ML model not trained. "
                    "Run: python model/train_model.py  OR set GROQ_API_KEY (free at console.groq.com)."
                ),
            }
        ), 503

    if search_bundle is None:
        search_bundle = fetch_news_context(news_text)

    final_label = str(fact["prediction"])
    final_conf = float(fact["confidence"])
    explanation = str(fact["explanation"])
    needs_verification = bool(fact["needs_verification"])

    reliability_score = topic_reliability_score(news_text)
    balanced_conf = apply_confidence_balancing(final_label, final_conf, reliability_score)
    balanced_conf = format_display_confidence(
        final_label, balanced_conf, provider or str(fact.get("provider") or "")
    )

    try:
        save_prediction(news_text, final_label, balanced_conf, country=country)
    except Exception as exc:
        logger.warning("Could not save prediction log (non-fatal): %s", exc)

    articles = search_bundle.get("articles") or []
    payload = build_fact_result_payload(
        prediction=final_label,
        confidence=balanced_conf,
        explanation=explanation,
        needs_verification=needs_verification,
        search_articles=articles,
    )

    hint = sanitize_hint(final_label, needs_verification, explanation)
    message = explanation[:500] if explanation else hint

    logger.info(
        "Fact-check complete: prediction=%s confidence=%s needs_verification=%s articles=%s",
        final_label,
        balanced_conf,
        needs_verification,
        len(articles),
    )

    return jsonify(
        {
            "prediction": final_label,
            "confidence": balanced_conf,
            "country": country,
            "message": message,
            "explanation": explanation,
            "provider": provider or fact.get("provider", ""),
            **payload,
            "result_hint": hint,
        }
    )


@app.route("/auth/signup", methods=["POST"])
def signup_api():
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", "")).strip()

    if not username or not email or not password:
        return jsonify({"error": "Please fill all signup fields."}), 400
    if "@" not in email or "." not in email:
        return jsonify({"error": "Please enter a valid email."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400

    ok, message = create_user(username, email, password)
    if not ok:
        return jsonify({"error": message}), 400

    session["username"] = username
    session["email"] = email
    return jsonify({"message": "Signup successful.", "username": username})


@app.route("/auth/login", methods=["POST"])
def login_api():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", "")).strip()

    if not email or not password:
        return jsonify({"error": "Please enter email and password."}), 400

    user = authenticate_user(email, password)
    if not user:
        return jsonify({"error": "Invalid email or password."}), 401

    session["username"] = user["username"]
    session["email"] = user["email"]
    return jsonify({"message": "Login successful.", "username": user["username"]})


@app.route("/auth/logout", methods=["POST"])
def logout_api():
    session.clear()
    return jsonify({"message": "Logged out successfully."})


if __name__ == "__main__":
    logger.info("Starting fact-check backend (project dir: %s)", PROJECT_DIR)
    init_db()

    has_ai = _has_ai_keys()
    if model_available():
        logger.info("Startup: ML model found at model/artifacts/ — will use if AI is off or fails.")
        if preload_model():
            logger.info("Startup: ML model preloaded (fast predictions).")
    else:
        logger.warning("Startup: no ML artifacts — run python model/train_model.py")

    env_file = os.path.join(PROJECT_DIR, ".env")
    if has_ai:
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            logger.info("Startup: Gemini/Google AI key loaded — fact-check will use Gemini first.")
        else:
            logger.info("Startup: at least one AI path configured (Groq / OpenAI / Ollama).")
    elif model_available():
        logger.warning(
            "Startup: NO AI API KEY — only local ML (low accuracy on facts). "
            "Copy .env.example to .env and set GEMINI_API_KEY from https://aistudio.google.com/apikey"
        )
        if os.path.isfile(env_file):
            logger.warning("Startup: .env file exists but GEMINI_API_KEY / GROQ_API_KEY not set inside it.")
    else:
        logger.error(
            "Startup: no AI and no ML model. Train: python model/train_model.py "
            "OR set GROQ_API_KEY from https://console.groq.com"
        )

    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    port = int(os.environ.get("PORT", "5000"))
    logger.info("Serving on port %s (debug=%s)", port, debug_mode)

    if debug_mode:
        app.run(debug=True, host="127.0.0.1", port=port)
    elif os.environ.get("USE_WAITRESS", "").strip() == "1":
        try:
            from waitress import serve  # type: ignore[import-not-found]

            logger.info("Using Waitress on 0.0.0.0 (USE_WAITRESS=1).")
            serve(app, host="0.0.0.0", port=port)
        except ImportError:
            logger.warning("Waitress not installed; using Flask threaded server on 127.0.0.1.")
            app.run(debug=False, host="127.0.0.1", port=port, threaded=True)
    else:
        logger.info("Using Flask threaded server on http://127.0.0.1:%s (set USE_WAITRESS=1 for Waitress).", port)
        app.run(debug=False, host="127.0.0.1", port=port, threaded=True)
