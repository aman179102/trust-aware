from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Lightweight Hugging Face text classification model suitable for CPU-only use.
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


@dataclass
class TextClassificationResult:
    """Container for a single text classification output."""

    label: str
    confidence: float
    scores: Dict[str, float]


class TextClassifier:
    """Wrapper around a Hugging Face text classification model.

    This class encapsulates tokenizer and model loading and exposes
    a simple Python interface to the rest of the application.
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model_name = model_name
        # Download and cache tokenizer/model on first use.
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # We only ever do inference.
        self._model.eval()

    @property
    def model_name(self) -> str:
        return self._model_name

    def predict(self, text: str, max_length: int = 256) -> TextClassificationResult:
        """Run inference on a single piece of text.

        Args:
            text: Raw user text.
            max_length: Maximum tokenized length to protect latency and memory.
        """
        # Tokenize on CPU with truncation to avoid very long inputs.
        encoded = self._tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self._model(**encoded)
            logits = outputs.logits[0]
            probabilities = torch.softmax(logits, dim=-1)

        probs = probabilities.tolist()
        id2label = self._model.config.id2label

        # Convert raw scores to a human-readable mapping.
        scores = {id2label[i]: float(probs[i]) for i in range(len(probs))}

        # Highest-probability label is used as the prediction.
        label = max(scores, key=scores.get)
        confidence = scores[label]

        return TextClassificationResult(label=label, confidence=confidence, scores=scores)


# Lazy-initialised singleton so the model is loaded only once per process.
_classifier: Optional[TextClassifier] = None


def get_model() -> TextClassifier:
    """Return a shared TextClassifier instance, creating it on first use."""
    global _classifier
    if _classifier is None:
        _classifier = TextClassifier(MODEL_NAME)
    return _classifier
