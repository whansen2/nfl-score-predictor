# AI Enhancement Idea (Low-Cost, High-Value)

Augment each prediction row with three AI-generated fields (using existing model output and local data only):

- **Confidence**: Low / Medium / High confidence in the predicted winner.
- **Rationale**: A short human-readable explanation (1 sentence) for why the pick was made.
- **Key_Factors**: Compact tags summarizing the main drivers (e.g., `close_margin`, `home_field`, `injury_impact`, `efficiency_edge`).

This keeps cost low (short prompts, no web fetch dependency) while improving explainability and downstream usability.
