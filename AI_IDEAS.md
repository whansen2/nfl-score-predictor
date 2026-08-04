## AI Enhancement Ideas

Current fit: keep the predictor deterministic. Add structured labels first, then optionally add a grounded text layer.

## Principles

- Use only local CSV/YAML inputs and computed outputs.
- Keep confidence and risk deterministic.
- Use an LLM only for short explanation text.
- Never let the AI layer change winner, score, or total.
- Every added field must be testable and nullable or have a fallback.

## 1. Prediction Explainability

Goal: add readable explanation metadata without changing the model output.

Phase 1 fields:

- `confidence_label`: `low`, `medium`, `high`
- `key_factors`: tags such as `close_margin`, `home_field`, `injury_impact`, `efficiency_edge`, `model_quality_low`
- `explanation_inputs`: optional internal payload for text generation

Phase 2 field:

- `rationale`: one grounded sentence based only on `key_factors`, margin, total, and injury adjustments

Example:

- `confidence_label=medium`
- `key_factors=close_margin,home_field,injury_impact`
- `rationale=Home field advantage offsets a narrow projection gap, but the injury adjustment keeps this matchup volatile.`

## 2. Risk Alert Classification

Goal: flag matchups that are unstable, close, or sensitive to weak assumptions.

Suggested fields:

- `risk_alert`: `none`, `medium`, `high`
- `risk_reason_code`: tags such as `one_score_margin`, `injury_adjustment_present`, `low_r2_week`, `missing_signal`, `postseason_mapping`
- `risk_reason`: optional short sentence

Example:

- `risk_alert=high`
- `risk_reason_code=one_score_margin,injury_adjustment_present`
- `risk_reason=Projected margin is within one score and quarterback injury adjustments affected the final output.`

## Guardrails

- Never invent injuries, player news, weather, betting lines, or recent performance.
- Do not fetch external data unless that becomes an explicit product decision.
- If the LLM fails, still emit predictions and deterministic labels.
- Generated text must be short, factual, and derived only from structured inputs.

## Rollout

1. Add deterministic `confidence_label` and `risk_alert`.
2. Add `key_factors` and `risk_reason_code`.
3. Add optional `rationale` behind a feature flag.

## Repo Notes

- Implement this as post-processing on top of `run_predictions`.
- Do not replace the current regression model.
- If an LLM is added later, use a provider abstraction rather than binding the design to one vendor.
