# Trust-Aware AI Decision System

A small but industry-relevant reference project that shows how to build an AI
system that is **aware of its own uncertainty**, can **explain its decisions**,
and knows when to **ask a human for help**.

Built using only **free and open-source tools**:

- **Python**
- **FastAPI** for the HTTP API
- **Hugging Face Transformers** for the pretrained text classifier
- **PyTorch** for tensor operations

Everything runs **locally on CPU**. No paid APIs, no cloud credits.

---

## Why Trust-Aware AI?

Most real-world AI failures are not about getting a single prediction wrong,
they are about systems that are **confidently wrong** without any visibility or
safeguards.

Examples:

- A support bot confidently gives the wrong policy answer to a customer.
- A moderation model is only slightly more confident in "safe" than in
  "hate" but still auto-approves the content.
- A risk scoring model silently drifts and no one notices because only the
  final label is logged.

Modern industry practice is moving towards systems that:

- expose **confidence scores and raw model outputs**
- apply **policy rules** on top of the model (e.g. thresholds, fallbacks)
- produce **simple explanations** that humans can understand
- **defer to humans** when the model is unsure

This project is a minimal, end-to-end example of that pattern.

---

## What This Project Does

For a given piece of text, the system:

1. **Runs a Hugging Face text classification model** locally on CPU.
2. Computes **per-label probabilities** using softmax.
3. Evaluates several **trust signals** instead of relying on confidence alone:
   - confidence compared to a configurable threshold (default `0.7`)
   - the **margin** between the top-1 and top-2 classes
   - linguistic ambiguity patterns (hedging, questions, contrastive phrasing)
4. Aggregates these signals into a **risk score**:
   - each risk factor (low confidence, low margin, ambiguity) increments the score
   - if `risk_score ≥ 2` → `decision = "needs_human_review"`
   - else → `decision = "accepted"`
5. Generates a **human-readable explanation** that describes:
   - the predicted label and confidence
   - which trust signals were triggered (confidence, margin, ambiguity)
   - why automation was allowed or blocked
   - that the system intentionally prefers **safety over blind automation**

Everything is exposed through a simple FastAPI endpoint:

- `POST /analyze`

```jsonc
{
  "label": "POSITIVE",
  "confidence": 0.93,
  "decision": "accepted",
  "explanation": "...human-readable text...",
  "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
  "scores": {
    "NEGATIVE": 0.07,
    "POSITIVE": 0.93
  }
}
```

---

## Architecture Overview

High-level data flow:

```text
User Text
   │
   ▼
[app/model.py]        Hugging Face model on CPU
   │   └─ produces label + per-class probabilities
   ▼
[app/decision.py]     Risk & policy layer
   │   └─ combines confidence, score margin, and ambiguity signals into a risk score
   ▼
[app/explain.py]      Explanation layer
   │   └─ explains how confidence, margin, ambiguity, and risk score led to the decision
   ▼
[app/api.py]          FastAPI endpoint
   │   └─ validation, error handling, response schema
   ▼
Client / UI
```

### Module Layout

- `app/model.py`
  - Loads the Hugging Face model
    (`distilbert-base-uncased-finetuned-sst-2-english`).
  - Provides a small `TextClassifier` wrapper with a `predict(text)` method.
  - Returns a `TextClassificationResult` with `label`, `confidence`, and
    `scores` (per-label probabilities).

- `app/decision.py`
  - Implements the **trust policy** around the model.
  - Exposes `DEFAULT_CONFIDENCE_THRESHOLD` (0.7 by default).
  - Computes a `DecisionDetails` object with:
    - `decision` → `"accepted"` or `"needs_human_review"`
    - `threshold` → the threshold used for this request
    - `margin` → the probability gap between top-1 and top-2 labels

- `app/explain.py`
  - Generates a **human-readable explanation** string.
  - Uses:
    - the model's confidence
    - the decision threshold and margin
    - ambiguity and risk score signals from the decision layer
  - Explanations are **rule-based**, multi-signal, and easy to inspect.

- `app/api.py`
  - Defines the FastAPI router and the `POST /analyze` endpoint.
  - Handles input validation, error responses, and output schema.
  - Delegates work to `model.py`, `decision.py`, and `explain.py`.

- `main.py`
  - Creates the FastAPI app and includes the router.
  - Entry point for `uvicorn main:app`.

This separation mirrors how larger production systems are structured and makes
it easier to extend or replace individual components later.

---

## Getting Started

### 1. Prerequisites

- Python **3.9+** recommended
- A machine with enough RAM to load a small transformer model (e.g. 2–4 GB)
- No GPU required; everything runs on CPU.

### 2. Clone and install

```bash
git clone <your-fork-or-repo-url>
cd trust_aware_ai

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on PyTorch**: the `torch` dependency installed via `pip` will use a
> CPU build by default on most systems. If you need a specific build (e.g.
> CUDA-enabled), follow the instructions on the official PyTorch website.

### 3. Run the API server

From the project root (where `main.py` lives):

```bash
uvicorn main:app --reload
```

The API will be available at:

- OpenAPI docs: <http://127.0.0.1:8000/docs>
- Raw JSON: `POST http://127.0.0.1:8000/analyze`

---

## Usage Examples

### Basic sentiment analysis with trust-aware decision

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product, it works great!"}'
```

Possible response:

```jsonc
{
  "label": "POSITIVE",
  "confidence": 0.95,
  "decision": "accepted",
  "explanation": "The system predicted POSITIVE with confidence 0.95. ...",
  "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
  "scores": {
    "NEGATIVE": 0.05,
    "POSITIVE": 0.95
  }
}
```

### Using a custom confidence threshold

You can make the system more conservative by increasing the threshold:

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The service was okay, not great but not terrible either.",
    "confidence_threshold": 0.8
  }'
```

For borderline cases, this often results in:

```jsonc
{
  "label": "POSITIVE",
  "confidence": 0.72,
  "decision": "needs_human_review",
  "explanation": "...",
  "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
  "scores": {
    "NEGATIVE": 0.28,
    "POSITIVE": 0.72
  }
}
```

---

## How This Reflects Industry Practice

This project is intentionally small, but it demonstrates several patterns that
show up in real systems:

- **Model vs. policy separation**
  - The Hugging Face model only knows how to score text.
  - The system wraps those scores in a separate **decision layer** that
    enforces business rules (thresholds, human review).

- **Explicit uncertainty handling**
  - Instead of just returning a label, the API always returns a
    **confidence score** and the full **per-class probability vector**.
  - Low-confidence or high-risk predictions are never silently treated as
    high-confidence truths.

- **Simple, inspectable explanations**
  - Explanations here are deliberately rule-based and lightweight.
  - In many production environments, this type of explanation is preferred
    because it is predictable, auditable, and cheap to compute.

---

## Why this system is different

Most examples of text classification APIs simply return a label and a single
confidence value. In practice, that is not enough to operate safely:

- confidence can be **miscalibrated**
- models can be **overconfident on unfamiliar inputs**
- ambiguous or mixed statements can look confident but still be risky

This project takes a more **systems-oriented** approach:

- **Hybrid trust signals instead of a single knob**  
  Decisions are based on a combination of confidence, score margin, and
  linguistic ambiguity indicators. No single signal can force automation.

- **Risk-first, not accuracy-only**  
  A compact **risk score** tracks how many weak risk factors are present.
  Even with high confidence, a narrow margin or ambiguous language can push
  the score over the review threshold and route the case to humans.

- **Pattern-based ambiguity, not brittle word lists**  
  The system looks for structural cues such as hedging ("maybe", "it seems"),
  questions, and contrastive phrasing ("but", "however") instead of relying on
  a static blacklist of terms. These are treated as soft indicators that
  contribute to risk rather than hard vetoes.

- **Safe behaviour under uncertainty**  
  When signals disagree – for example, high confidence but low margin and
  ambiguous language – the system **defers to human review**. This mirrors how
  production systems are commonly designed for regulated or safety-critical
  domains.

In short, the goal is not just to run an ML model, but to demonstrate how
**real AI systems manage uncertainty and trust** when user input is
unpredictable.

- **Local-first, privacy-friendly design**
  - All processing happens on your machine with an open-source model.
  - No data leaves your environment, which is critical for sensitive text.

This makes the repository suitable as:

- a **final-year project**
- a **portfolio piece** on GitHub
- a starting point for more advanced trust and safety work

---

## Extending the Project

Ideas for future work:

- **Model choice**: swap in a different Hugging Face text classifier (e.g.
  topic classification, toxicity detection) by changing the model name in
  `app/model.py`.
- **Richer policies**: add per-label thresholds or business rules
  (e.g. always send certain risk labels to human review regardless of
  confidence).
- **Logging and monitoring**: log decisions, confidences, and margins to a
  local store for later analysis.
- **UI layer**: build a small web or desktop UI that consumes the `/analyze`
  endpoint.
- **More advanced explainability**: integrate additional open-source
  techniques (e.g. saliency maps or attention visualisation) while still
  staying local and free.

---

## License and Credits

- Built using only **free and open-source tools**.
- Hugging Face model:
  [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
- See the repository LICENSE file for full details (add one if publishing
  this project publicly).
