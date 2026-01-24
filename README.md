# Trust-Aware AI Decision System

Overconfident AI can sound sure, pass tests, and still make quietly wrong decisions.  
Most sentiment APIs return a label and a confidence score, then leave you to guess when it’s unsafe to automate.  
This project shows how to turn that same model into a **trust-aware, human-in-the-loop AI system** that knows when to ask for help.

> **TL;DR**  
> A local FastAPI service that wraps a Hugging Face sentiment model with a risk-aware decision layer.  
> Instead of just returning a label, it exposes confidence, margin, and linguistic risk signals and decides whether to auto-accept a prediction or defer to a human reviewer.  
> Built entirely from free, CPU-friendly tools for privacy-friendly, on-prem style workflows.

## Project Overview

This repository implements an end-to-end, trust-aware sentiment analysis system:

- Uses a pretrained DistilBERT model from Hugging Face (no training required).
- Runs fully locally on CPU (no paid APIs, no cloud dependencies).
- Wraps the model in a **risk engine** that inspects confidence, score margins, and linguistic ambiguity.
- Makes a **human-in-the-loop decision**: either `accepted` or `needs_human_review`.
- Returns both **machine-readable signals** and a **human-readable explanation** for every request.

---

## Typical ML APIs vs This Project

| Aspect              | Typical ML API                                   | This project                                                                 |
| ------------------- | ------------------------------------------------ | ----------------------------------------------------------------------------- |
| Output              | Label + raw confidence                           | Label, confidence, margin, `risk_score`, `risk_signals`, explanation, scores |
| Decision logic      | Threshold on confidence                          | Multi-signal risk engine with explicit human-review threshold                |
| Automation          | Optimised for full automation                    | Optimised for safe automation + human in the loop                            |
| Transparency        | Opaque; little insight into uncertainty          | Structured risk metadata + narrative explanation                             |
| Deployment model    | Often cloud/SaaS                                 | Fully local, CPU-only                                                        |
| Governance posture  | Trust and review added later                     | Trust, review, and deferral designed in from day one                         |

---

## Real Example: High Confidence, Unsafe Automation

Even when the model is confident, this system may *still* decide to defer:

```jsonc
{
  "label": "POSITIVE",
  "decision": "needs_human_review",
  "confidence": 0.91,
  "margin": 0.06,
  "risk_score": 2,
  "risk_signals": ["low_margin", "ambiguity"],
  "explanation": "Although model confidence is high, the score margin is narrow and the text contains contrastive phrasing, so the system defers to human review instead of auto-approving.",
  "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
  "scores": {
    "NEGATIVE": 0.09,
    "POSITIVE": 0.91
  }
}
```
---

## Why Confidence-Only AI Is Dangerous

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

## What Makes This System Different

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
   - which trust signals were triggered (confidence, margin, ambiguity, mixed sentiment)
   - why automation was allowed or blocked
   - that the system intentionally prefers **safety over blind automation**

Everything is exposed through simple FastAPI endpoints:

- `POST /analyze` – run a trust-aware analysis on text
- `GET /health` – lightweight health check (model loaded, process ready)

Example `/analyze` response:

```jsonc
{
  "label": "POSITIVE",
  "decision": "accepted",
  "confidence": 0.93,
  "margin": 0.90,
  "risk_score": 0,
  "risk_signals": [],
  "explanation": "...human-readable text explaining why automation was safe...",
  "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
  "scores": {
    "NEGATIVE": 0.07,
    "POSITIVE": 0.93
  }
}
```

---

## Architecture (Model → Risk Engine → Decision)

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

## Example Scenarios (clear vs ambiguous inputs)

### Clear, low-risk sentiment

Request:

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "I absolutely love this product, it works perfectly."}'
```

Possible response:

```jsonc
{
  "label": "POSITIVE",
  "decision": "accepted",
  "confidence": 0.97,
  "margin": 0.90,
  "risk_score": 0,
  "risk_signals": [],
  "explanation": "...no risk signals fired; confidence and margin are both strong...",
  "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
  "scores": {
    "NEGATIVE": 0.03,
    "POSITIVE": 0.97
  }
}
```

### Ambiguous, high-risk sentiment

Request:

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "The product is great, but the customer service was terrible."}'
```

Possible response:

```jsonc
{
  "label": "POSITIVE",
  "decision": "needs_human_review",
  "confidence": 0.86,
  "margin": 0.18,
  "risk_score": 2,
  "risk_signals": ["low_margin", "ambiguity"],
  "explanation": "Although confidence is high, the margin is narrow and the text uses contrastive phrasing, so the system defers to human review.",
  "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
  "scores": {
    "NEGATIVE": 0.14,
    "POSITIVE": 0.86
  }
}
```

These examples highlight how **the same base model** can behave very
differently once wrapped in a risk-aware decision layer.

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

## How to Run Locally

### 1. Prerequisites

- Python **3.9+** recommended
- A machine with enough RAM to load a small transformer model (e.g. 2–4 GB)
- No GPU required; everything runs on CPU.

### 2. Clone and install

```bash
https://github.com/aman179102/trust-aware.git
cd trust-aware

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

## Intended Use & Non-Goals

- **Intended use**
  - As a reference implementation for **trust-aware AI** and
    human-in-the-loop decision making.
  - As a local service for teams who want to experiment with
    **risk-sensitive sentiment analysis** without sending data to the cloud.
  - As a portfolio / demo project for discussing **AI safety and governance**
    in interviews or design reviews.

- **Non-goals**
  - This is **not** a truth engine or a replacement for human judgment.
  - This is **not** a high-accuracy, domain-tuned sentiment model; it is a
    small, generic model wrapped in a strong policy layer.
  - This repository does **not** include automated retraining, calibration, or
    production observability; those are intentionally called out as future
    work.

---

## Human-in-the-Loop Philosophy

This project is intentionally small, but it demonstrates several patterns that
show up in real **human-in-the-loop** systems:

- **Model vs. policy separation**
  - The Hugging Face model only knows how to score text.
  - The system wraps those scores in a separate **decision layer** that
    enforces business rules (thresholds, human review).

- **Explicit uncertainty handling**
  - Instead of just returning a label, the API always returns structured
    signals: the label, confidence, margin, risk score, and which risk
    signals fired.
  - Low-confidence or high-risk predictions are never silently treated as
    high-confidence truths.

- **Simple, inspectable explanations**
  - Explanations here are deliberately rule-based and lightweight.
  - In many production environments, this type of explanation is preferred
    because it is predictable, auditable, and cheap to compute.

- **Humans stay in control**
  - The system is designed so that humans, not the model, make the final call
    in ambiguous or high-risk situations.
  - The `/analyze` output is structured so that downstream tools or reviewers
    can build dashboards, queues, or escalation workflows on top.

---

## Future Improvements (sarcasm, domain rules, active learning)

Ideas for future work:

- **Sarcasm and nuanced tone**: extend the ambiguity detector to better handle
  sarcasm, irony, and subtle tone shifts that can confuse sentiment models.
- **Domain-specific rules**: layer in domain rules (e.g. finance, healthcare,
  trust & safety) that can override the model when certain topics or entities
  are present.
- **Active learning and feedback**: add mechanisms for reviewers to provide
  feedback on high-risk cases and use that feedback to retrain or recalibrate
  models offline.
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

## How to Evaluate This System

When reviewing this project, it is important to treat it as a
**trust-first / safety-first** system rather than an accuracy benchmark.

- **Not accuracy-first**
  - The underlying DistilBERT model is "good enough" for sentiment, but the
    focus of this repository is on the **policy and risk layer** wrapped
    around it.

- **Trust-first decision logic**
  - The system explicitly tracks multiple weak signals (confidence, margin,
    ambiguity, mixed signals) and aggregates them into a **risk_score**.
  - Automation is only allowed when the cumulative risk is low; otherwise the
    system defers to human review.

- **Reading `risk_score` and `risk_signals`**
  - `risk_signals` lists *which* factors fired, e.g.
    `['low_margin', 'ambiguity']`.
  - `risk_score` is the **count** of those signals, and it is directly used to
    choose between `"accepted"` and `"needs_human_review"`.
  - Reviewers can quickly see *why* a prediction was considered risky by
    inspecting these fields and the explanation text.

- **Why high confidence can still trigger review**
  - Confidence alone can be miscalibrated or misleading, especially for
    unfamiliar or ambiguous inputs.
  - If, for example, the text includes strong contrast ("but", "however") or
    explicit questions, the system may still route the case to human review
    even when confidence is high and the margin is reasonable.

- **Why deferring to humans is safer**
  - In many production environments, the cost of an overconfident automated
    decision is much higher than the cost of asking a human to review a
    borderline case.
  - This project intentionally **optimises for safe behaviour under
    uncertainty**, making it more suitable as a building block for
    safety-critical or regulated domains than a typical "black box" sentiment
    API.

Taken together, these properties make the system **provable and reviewable**:
an engineer, auditor, or hiring manager can understand in a few minutes how
decisions are made, which signals drive those decisions, and where the safe
fallbacks are.

---

## License and Credits

- Built using only **free and open-source tools**.
- Hugging Face model:
  [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
- See the repository LICENSE file for full details (add one if publishing
  this project publicly).
