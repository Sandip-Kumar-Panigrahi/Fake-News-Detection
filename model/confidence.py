"""Map model probabilities to Real / Fake / Unclear labels and display confidence."""

from __future__ import annotations

from typing import Tuple

UNCLEAR_MIN_PROB = 0.42
TIE_MARGIN = 0.04
TIE_MAX_PROB = 0.52
def apply_thresholds(fake_prob: float, real_prob: float) -> Tuple[str, float]:
    fake_prob = float(fake_prob)
    real_prob = float(real_prob)
    win_prob = max(fake_prob, real_prob)
    margin = abs(fake_prob - real_prob)

    if win_prob < UNCLEAR_MIN_PROB or (margin < TIE_MARGIN and win_prob < TIE_MAX_PROB):
        return "Unclear", round(50.0 + win_prob * 10.0, 2)

    if real_prob >= fake_prob:
        return "Real", round(real_prob * 100.0, 2)

    return "Fake", round(fake_prob * 100.0, 2)
