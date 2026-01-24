from __future__ import annotations

from typing import Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .decision import DEFAULT_CONFIDENCE_THRESHOLD, DecisionDetails, make_decision
from .explain import generate_explanation
from .model import TextClassificationResult, get_model


router = APIRouter()


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Raw text to analyze.")
    confidence_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional custom confidence threshold in [0, 1]. If omitted, "
            "the system default is used."
        ),
    )


class AnalyzeResponse(BaseModel):
    label: str
    confidence: float
    decision: str
    explanation: str
    model_name: str
    scores: Dict[str, float]


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze user text and return a trust-aware decision."""
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    # Basic length guard to avoid pathological inputs.
    if len(text) > 4000:
        raise HTTPException(
            status_code=400,
            detail=(
                "Text input is too long. Please provide a shorter snippet "
                "(<= 4000 characters)."
            ),
        )

    threshold = (
        request.confidence_threshold
        if request.confidence_threshold is not None
        else DEFAULT_CONFIDENCE_THRESHOLD
    )

    model = get_model()
    result: TextClassificationResult = model.predict(text)
    decision: DecisionDetails = make_decision(text, result, threshold=threshold)
    explanation: str = generate_explanation(text, result, decision)

    return AnalyzeResponse(
        label=result.label,
        confidence=result.confidence,
        decision=decision.decision,
        explanation=explanation,
        model_name=model.model_name,
        scores=result.scores,
    )
