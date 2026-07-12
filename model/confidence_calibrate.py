"""Boost display confidence when rules or multiple sources agree on clear facts."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from model.labels import LABEL_FAKE, LABEL_MISLEADING, LABEL_REAL, normalize_label

DISPLAY_REAL = 99.0
DISPLAY_FAKE = 99.0
DISPLAY_MISLEADING_MAX = 72.0

# ML-only: boost strong wins (not style guesses on tiny margin)
ML_STRONG_MARGIN = 0.22
ML_STRONG_MIN_PROB = 0.78
ML_BOOSTED_REAL = 92.0
ML_BOOSTED_FAKE = 94.0


def calibrate_confidence(
    label: str,
    raw_confidence: float,
    signals: Optional[List[Any]] = None,
) -> float:
    """
    Return user-facing confidence (target ~98–99% for verified facts).
    """
    lbl = normalize_label(label)
    raw = float(raw_confidence)
    signals = signals or []

    rule_sig = _find_signal(signals, "rules")
    ml_sig = _find_signal(signals, "ml")

    # Rule engine = ground truth for geography, IPL, roles
    if rule_sig and float(getattr(rule_sig, "confidence", 0) or 0) >= 90.0:
        if lbl == LABEL_REAL:
            return DISPLAY_REAL
        if lbl == LABEL_FAKE:
            return DISPLAY_FAKE

    # Rules + ML agree → near-maximum
    if rule_sig and ml_sig:
        r_lbl = normalize_label(getattr(rule_sig, "label", ""))
        m_lbl = normalize_label(getattr(ml_sig, "label", ""))
        if r_lbl == m_lbl == lbl and lbl in (LABEL_REAL, LABEL_FAKE):
            return DISPLAY_REAL if lbl == LABEL_REAL else DISPLAY_FAKE

    # Strong ML probability on short headline-style input
    if ml_sig and getattr(ml_sig, "available", True):
        meta = getattr(ml_sig, "meta", None) or {}
        fake_p = meta.get("fake_prob")
        real_p = meta.get("real_prob")
        if fake_p is not None and real_p is not None:
            margin = abs(float(real_p) - float(fake_p))
            if lbl == LABEL_REAL and float(real_p) >= ML_STRONG_MIN_PROB and margin >= ML_STRONG_MARGIN:
                return max(raw, ML_BOOSTED_REAL)
            if lbl == LABEL_FAKE and float(fake_p) >= ML_STRONG_MIN_PROB and margin >= ML_STRONG_MARGIN:
                return max(raw, ML_BOOSTED_FAKE)

    if lbl == LABEL_MISLEADING:
        return round(min(DISPLAY_MISLEADING_MAX, max(48.0, raw)), 2)

    if lbl == LABEL_FAKE and raw >= 88.0:
        return min(DISPLAY_FAKE, max(raw, 96.0))

    if lbl == LABEL_REAL and raw >= 88.0:
        return min(DISPLAY_REAL, max(raw, 96.0))

    return round(raw, 2)


def _find_signal(signals: List[Any], source: str) -> Optional[Any]:
    for s in signals:
        if getattr(s, "source", None) == source:
            return s
    return None
