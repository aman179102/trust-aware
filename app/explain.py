from __future__ import annotations

from typing import List

from .decision import DecisionDetails, RISK_REVIEW_THRESHOLD
from .model import TextClassificationResult


def generate_explanation(
    text: str,
    result: TextClassificationResult,
    decision: DecisionDetails,
) -> str:
    """Generate a human-readable explanation for the model's output.

    Explanations combine multiple weak signals instead of relying on
    confidence alone:
    - model confidence and the decision threshold
    - margin between top-1 and top-2 scores
    - linguistic ambiguity and contrastive phrasing
    - the aggregated risk score compared to the review threshold
    """
    conf = result.confidence
    margin = decision.margin
    threshold = decision.threshold
    risk_score = decision.risk_score

    parts: List[str] = []

    # 1. Explain the core prediction.
    parts.append(
        f"The system predicted {result.label} with confidence {conf:.2f}."
    )

    # 2. Confidence vs. policy threshold.
    if decision.low_confidence:
        parts.append(
            f"Confidence is below the operational threshold of {threshold:.2f}, "
            "which increases the risk that the prediction could be wrong."
        )
    else:
        parts.append(
            f"Confidence is above the operational threshold of {threshold:.2f}, "
            "but the system does not rely on confidence alone to automate decisions."
        )

    # 3. Describe the score distribution and stability.
    if decision.low_margin:
        parts.append(
            f"The model's top classes are close in score (margin {margin:.2f}, "
            f"below the stability threshold of {decision.margin_threshold:.2f}), "
            "so the prediction is treated as unstable even if confidence is high."
        )
    else:
        parts.append(
            f"The score margin between the top classes is {margin:.2f}, above "
            f"the stability threshold of {decision.margin_threshold:.2f}, "
            "indicating a clear preference for the chosen label."
        )

    # 4. Linguistic ambiguity and mixed signals.
    if decision.ambiguous_language or decision.mixed_sentiment:
        if decision.ambiguous_language and decision.mixed_sentiment:
            parts.append(
                "The text uses hedging or contrastive phrasing and expresses "
                "mixed views, which we interpret as linguistic ambiguity."
            )
        elif decision.ambiguous_language:
            parts.append(
                "The text contains hedging, questions, or contrastive phrasing, "
                "which are treated as signs of ambiguity."
            )
        else:
            parts.append(
                "The text expresses conflicting views, which we treat as a "
                "mixed or uncertain sentiment signal."
            )
    else:
        parts.append(
            "The text does not exhibit strong ambiguity patterns such as "
            "hedging or contrastive phrasing."
        )

    # 5. Explicitly list which risk signals were triggered.
    if decision.risk_signals:
        signal_list = ", ".join(decision.risk_signals)
        parts.append(
            f"Triggered risk signals: {signal_list}. These signals indicate "
            "that the model's raw confidence should not be trusted on its own."
        )
    else:
        parts.append(
            "No explicit risk signals were triggered for this input, so the "
            "model's confidence and score margin are considered reliable in "
            "this context."
        )

    # 6. Summarise how the risk score drove the final decision.
    if decision.decision == "needs_human_review":
        parts.append(
            f"Multiple risk signals combine into a risk score of {risk_score}, "
            f"which meets or exceeds the review threshold of {RISK_REVIEW_THRESHOLD}. "
            "To avoid overconfident automation, the system defers this case "
            "to human review."
        )
    else:
        if risk_score == 0:
            parts.append(
                "Risk signals are low overall, so the system is comfortable "
                "automatically accepting this prediction."
            )
        else:
            parts.append(
                f"The cumulative risk score is {risk_score}, below the review "
                f"threshold of {RISK_REVIEW_THRESHOLD}, so the prediction is "
                "accepted while still prioritising safety over blind confidence."
            )

    return " ".join(parts)
