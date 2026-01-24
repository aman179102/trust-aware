# Manual Test Cases for the Trust-Aware AI Decision System

These test cases are designed for **human reviewers**, not automated execution.
They describe how the system *should* behave given its current design and
risk-based decision engine.

For each case, the expected behaviour is based on:

- the DistilBERT sentiment model
- the configured confidence and margin thresholds
- the linguistic ambiguity and mixed-signal detectors

---

## Test Case 1 — Clear positive, no ambiguity

- **Input text**  
  `"I absolutely love this product, it works perfectly and I would recommend it to everyone."`

- **Expected decision**  
  `accepted`

- **Expected risk_signals**  
  `[]` (empty list)

- **Reasoning**  
  Strongly positive language with no hedging, questions, or contrastive
  phrasing. The model should be highly confident with a large margin, so no
  risk signals should fire and the system can safely automate.

---

## Test Case 2 — Clear negative, no ambiguity

- **Input text**  
  `"This was a terrible experience, I am extremely disappointed and will not use this again."`

- **Expected decision**  
  `accepted`

- **Expected risk_signals**  
  `[]` (empty list)

- **Reasoning**  
  Strong, unambiguous negative sentiment with no hedging or uncertainty. The
  model should be confidently negative with a large margin, so the system
  accepts the prediction without needing human review.

---

## Test Case 3 — Mixed sentiment with "but"

- **Input text**  
  `"The product quality is great, but the customer service was awful."`

- **Expected decision**  
  `needs_human_review`

- **Expected risk_signals**  
  `["ambiguity", "mixed_sentiment"]`

- **Reasoning**  
  The contrastive structure ("but") indicates conflicting views within a
  single sentence. Even if the model is confidently positive overall, the
  ambiguity and mixed-signal detectors should both fire, pushing the risk
  score to at least 2 and routing this case to a human reviewer.

---

## Test Case 4 — Hedging language

- **Input text**  
  `"I think this product is good, but I'm not sure yet."`

- **Expected decision**  
  `needs_human_review`

- **Expected risk_signals**  
  `["ambiguity", "mixed_sentiment"]`

- **Reasoning**  
  Phrases like "I think" and "not sure" introduce hedging and uncertainty, and
  the use of "but" again creates a contrastive structure. The system should
  treat this as ambiguous and mixed, producing a risk score of at least 2 and
  deferring to a human.

---

## Test Case 5 — Question-based uncertainty

- **Input text**  
  `"Is this product actually good? I'm not sure."`

- **Expected decision**  
  `needs_human_review`

- **Expected risk_signals**  
  `["ambiguity", "mixed_sentiment"]`

- **Reasoning**  
  The explicit question mark and "not sure" indicate that the user is asking
  rather than stating a clear opinion. The question-based ambiguity and
  mixed-signal detectors should both fire, so even if the model's confidence is
  non-trivial, the system should route this case to human review.

---

## Test Case 6 — High confidence but low margin

- **Input text**  
  `"The service was okay, not great and not terrible either."`

- **Expected decision**  
  `accepted`

- **Expected risk_signals**  
  `["low_margin"]`

- **Reasoning**  
  This sentence is intentionally neutral and could be classified as slightly
  positive or slightly negative. The model may end up with a moderate
  confidence but a **narrow margin** between the top two classes. The
  `low_margin` signal should fire to reflect instability, but on its own it is
  not enough to force human review, so the prediction is accepted with a
  cautionary explanation.

---

## Test Case 7 — Ambiguous neutral case (soft ambiguity only)

- **Input text**  
  `"The experience was kind of okay."`

- **Expected decision**  
  `accepted`

- **Expected risk_signals**  
  `["ambiguity"]`

- **Reasoning**  
  The phrase "kind of" is hedging language that should trigger the ambiguity
  detector, but there is no question mark or strong contrastive structure.
  This produces a **single** risk signal (risk_score = 1), which keeps the
  decision in the "accepted" band while still highlighting that the model's
  confidence should be interpreted with some caution.

---

## Test Case 8 — Strong opinion with sarcasm-like tone

- **Input text**  
  `"Yeah, this product is just fantastic... said no one ever."`

- **Expected decision**  
  `accepted`

- **Expected risk_signals**  
  `[]` (empty list)

- **Reasoning**  
  This case illustrates a **known limitation** rather than a safety feature:
  the current system does not explicitly detect sarcasm. Depending on how the
  base model interprets the text, the prediction may be confidently negative or
  positive, but no ambiguity or mixed-signal markers are present in the
  heuristic detector. This is intentionally called out as a future improvement
  area rather than hidden.

---

These scenarios collectively demonstrate that the system is **trust-first**:

- clear, unambiguous cases are automated
- ambiguous, mixed, or question-like cases are routed for human review
- known limitations (like sarcasm) are documented explicitly for reviewers
