## AI Enhancement Ideas using OpenAI API key

### 1. Prediction Explainability Layer

Augment each prediction row with three AI-generated fields (using existing model output and local data only):

- **Confidence**: Low / Medium / High confidence in the predicted winner.
- **Rationale**: A short human-readable explanation (1 sentence) for why the pick was made.
- **Key_Factors**: Compact tags summarizing the main drivers (e.g., `close_margin`, `home_field`, `injury_impact`, `efficiency_edge`).

This keeps cost low (short prompts, no web fetch dependency) while improving explainability and downstream usability.

### 2. Risk Alert Classification

Add two lightweight AI-generated fields to each prediction row:

- **Risk_Alert**: `None`, `Medium`, or `High`.
- **Risk_Reason**: One short sentence explaining the trigger.

Example:

- `Risk_Alert=High`
- `Risk_Reason=Projected margin is within one score and injury-adjusted QB impact is non-zero.`

This gives quick scanability for high-variance games with minimal token usage and no web-fetch dependency.
