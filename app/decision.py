from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .model import TextClassificationResult

# Default minimum confidence required to consider a prediction low-risk.
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.7

# Minimum margin between top-1 and top-2 probabilities before we treat
# the prediction as stable. Smaller margins indicate ambiguity.
DEFAULT_MARGIN_THRESHOLD: float = 0.2

# Risk score at or above which we always defer to human review.
RISK_REVIEW_THRESHOLD: int = 2


@dataclass
class DecisionDetails:
    """Structured representation of the system's decision."""

    decision: str  # "accepted" or "needs_human_review"
    threshold: float
    margin: float  # gap between top-1 and top-2 scores
    margin_threshold: float
    risk_score: int
    low_confidence: bool
    low_margin: bool
    ambiguous_language: bool
    mixed_sentiment: bool
    risk_signals: List[str]


def _compute_margin(scores: Dict[str, float]) -> float:
    """Compute gap between highest and second-highest scores.

    A small margin indicates that multiple labels look similarly plausible,
    which is a useful trust signal even when absolute confidence is high.
    """
    if not scores:
        return 0.0

    ordered = sorted(scores.values(), reverse=True)
    if len(ordered) == 1:
        return ordered[0]
    return ordered[0] - ordered[1]


_HEDGING_MARKERS: Tuple[str, ...] = (
    "maybe",
    "probably",
    "perhaps",
    "i think",
    "it seems",
    "sort of",
    "kind of",
    "a bit",
    "not sure",
    "unclear",
)

_CONTRAST_MARKERS: Tuple[str, ...] = (
    " but ",
    " however ",
    " although ",
    " though ",
    " yet ",
)


def _normalise_text(text: str) -> str:
    return text.lower()


def _contains_any(text: str, markers: Tuple[str, ...]) -> bool:
    lower = _normalise_text(text)
    return any(marker in lower for marker in markers)


def _detect_ambiguity_signals(text: str) -> Tuple[bool, bool]:
    """Detect linguistic ambiguity and mixed sentiment in the input.

    Ambiguity is treated as a *soft* signal based on:
    - hedging language
    - questions or interrogative punctuation
    - contrastive structure (e.g. "but", "however"), which often reflects
      mixed or conflicting views.
    """
    lower = _normalise_text(text)

    has_hedging = _contains_any(lower, _HEDGING_MARKERS)
    has_contrast = _contains_any(lower, _CONTRAST_MARKERS)
    has_question = "?" in lower

    ambiguous_language = has_hedging or has_contrast or has_question

    # We treat explicit contrast or question-based uncertainty as a proxy for
    # "mixed" or conflicting signals in the text. This allows even
    # high-confidence predictions with strong ambiguity cues to be routed to
    # human review.
    mixed_sentiment = has_contrast or has_question

    return ambiguous_language, mixed_sentiment


def make_decision(
    text: str,
    result: TextClassificationResult,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
) -> DecisionDetails:
    """Decide whether to accept the prediction or route to human review.

    The decision is based on a combination of weak signals:
    - model confidence vs. a configurable threshold
    - probability margin between the top-1 and top-2 classes
    - linguistic ambiguity patterns in the input text
    - mixed sentiment indicators

    No single signal is allowed to fully determine the outcome. Instead, each
    risk factor contributes to a cumulative risk score.
    """
    margin = _compute_margin(result.scores)

    # Core numerical uncertainty signals.
    low_confidence = result.confidence < threshold
    low_margin = margin < margin_threshold

    # Textual ambiguity and mixed sentiment signals.
    ambiguous_language, mixed_sentiment = _detect_ambiguity_signals(text)

    # Build the set of active risk signals explicitly so that the
    # risk_score is always equal to the number of signals that fired.
    risk_signals: List[str] = []
    if low_confidence:
        risk_signals.append("low_confidence")
    if low_margin:
        risk_signals.append("low_margin")
    if ambiguous_language:
        risk_signals.append("ambiguity")
    if mixed_sentiment:
        risk_signals.append("mixed_sentiment")

    risk_score = len(risk_signals)

    if risk_score >= RISK_REVIEW_THRESHOLD:
        decision = "needs_human_review"
    else:
        decision = "accepted"

    return DecisionDetails(
        decision=decision,
        threshold=threshold,
        margin=margin,
        margin_threshold=margin_threshold,
        risk_score=risk_score,
        low_confidence=low_confidence,
        low_margin=low_margin,
        ambiguous_language=ambiguous_language,
        mixed_sentiment=mixed_sentiment,
        risk_signals=risk_signals,
    )
