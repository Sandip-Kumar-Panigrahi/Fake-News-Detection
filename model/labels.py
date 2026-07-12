"""Canonical verdict labels for the hybrid fact-checking system."""

from __future__ import annotations

LABEL_REAL = "REAL"
LABEL_FAKE = "FAKE"
LABEL_MISLEADING = "MISLEADING"
LABEL_UNKNOWN = "UNKNOWN"

ALL_LABELS = (LABEL_REAL, LABEL_FAKE, LABEL_MISLEADING, LABEL_UNKNOWN)

_LEGACY_MAP = {
    "real": LABEL_REAL,
    "true": LABEL_REAL,
    "fake": LABEL_FAKE,
    "false": LABEL_FAKE,
    "misleading": LABEL_MISLEADING,
    "unclear": LABEL_MISLEADING,
    "unknown": LABEL_UNKNOWN,
    "needs verification": LABEL_MISLEADING,
}


def normalize_label(value: str, default: str = LABEL_MISLEADING) -> str:
    key = str(value or "").strip().lower()
    if not key:
        return default
    if key.upper() in ALL_LABELS:
        return key.upper()
    return _LEGACY_MAP.get(key, default)


def label_score(label: str) -> float:
    """Signed score for weighted hybrid voting (-1 fake .. +1 real)."""
    lbl = normalize_label(label, LABEL_UNKNOWN)
    if lbl == LABEL_FAKE:
        return -1.0
    if lbl == LABEL_REAL:
        return 1.0
    if lbl == LABEL_MISLEADING:
        return -0.35
    return 0.0
