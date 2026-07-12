import os
import pickle
import sys
from typing import Any, Dict, Optional, Tuple

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from backend.confidence_display import format_display_confidence
from model.confidence import apply_thresholds
from model.fact_claims import check_factual_claim
from model.input_quality import assess_input_quality
from model.text_preprocessing import preprocess_text

ARTIFACT_DIR = os.path.join(BASE_DIR, "model", "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.pkl")

_ml_cache: Optional[Tuple[Any, Any]] = None


def model_available() -> bool:
    return os.path.isfile(MODEL_PATH) and os.path.isfile(VECTORIZER_PATH)


def load_model_bundle() -> Tuple[Any, Any]:
    global _ml_cache
    if _ml_cache is not None:
        return _ml_cache
    if not model_available():
        raise RuntimeError(
            "ML model not found. Train first: python model/train_model.py"
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    _ml_cache = (model, vectorizer)
    return _ml_cache


def preload_model() -> bool:
    try:
        load_model_bundle()
        return True
    except Exception:
        return False


def predict_news(text: str) -> Dict[str, Any]:
    factual = check_factual_claim(text)
    if factual is not None:
        label = str(factual["prediction"])
        confidence = float(factual["confidence"])
        if label == "Fake":
            explanation = str(factual["explanation"])
        elif label == "Real":
            explanation = str(factual["explanation"])
        else:
            explanation = str(factual.get("explanation", ""))
        return {
            "prediction": label,
            "confidence": format_display_confidence(label, confidence, "facts"),
            "explanation": explanation,
            "needs_verification": bool(factual.get("needs_verification", False)),
            "provider": "facts",
        }

    ok, reason = assess_input_quality(text)
    if not ok:
        return {
            "prediction": "Unclear",
            "confidence": 55.0,
            "explanation": reason,
            "needs_verification": True,
        }

    model, vectorizer = load_model_bundle()
    cleaned = preprocess_text(text)
    if not cleaned:
        return {
            "prediction": "Unclear",
            "confidence": 55.0,
            "explanation": "Could not extract meaningful words from the input.",
            "needs_verification": True,
        }

    text_vector = vectorizer.transform([cleaned])
    prob = model.predict_proba(text_vector)[0]
    classes = list(model.classes_)
    fake_idx = classes.index(0)
    real_idx = classes.index(1)
    fake_prob = float(prob[fake_idx])
    real_prob = float(prob[real_idx])

    label, confidence = apply_thresholds(fake_prob, real_prob)
    needs_verification = label == "Unclear"

    if label == "Fake":
        explanation = (
            "The model classified this as likely fake or misleading news "
            f"({confidence:.1f}% confidence). This is style-based, not a full fact-check."
        )
    elif label == "Real":
        explanation = (
            "The model classified this text as likely real news "
            f"({confidence:.1f}% confidence). This is style-based, not a full fact-check."
        )
    else:
        explanation = (
            "The model is not confident enough for a clear Real or Fake verdict. "
            "Try a longer, clearer news headline."
        )

    return {
        "prediction": label,
        "confidence": format_display_confidence(label, confidence, "ml"),
        "explanation": explanation,
        "needs_verification": needs_verification,
        "provider": "ml",
    }
