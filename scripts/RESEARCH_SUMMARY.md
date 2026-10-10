# Model research summary

**Question:** does any alternative model beat the production predictor
out-of-sample? The production model has 5 features, uses linear regression,
and adds HFA +2, the 0.30 opponent blend and the QB injury adjustment.

**Decision:** keep production unchanged. Track it prospectively on 2026 wk5+.

## Method
- **Data:** seasons 2024–2026 (608 games through 2026 wk4).
- **Walk-forward:** a week-W game sees only stats through week W−1.
- **Metrics:** margin MAE (primary) and team-score MAE, with week-block
  bootstrap 95% CIs.
- **Baselines:** home+2 and the Vegas line.
- **Holdout:** 2026 wk5–18 was never loaded.
- **Tuning:** library or published defaults, plus past-only selection.

## Results (margin MAE, lower is better)

**Round 1, 2025:**
- Vegas line 9.80
- Elo (538 defaults) 10.20
- Regularized linear models 10.36–10.44
- Production, research version (4 features) 10.74
- home+2 11.05
- MLP 11.16
- GBM, Huber, PCA and KMeans did worse. The data is too small for flexible
  models.

**Round 2, pooled 2024 wk9–18 + 2025, 4-feature research version:**
- Production 10.30
- S1b (production + prior-season shrinkage) 10.09; Δ −0.21 [−0.55, +0.02]
- S7a (50/50 S1b + Elo) 9.95; Δ −0.35 [−0.78, −0.01]
- Elo 9.99, Kalman 10.07, Massey 10.16, game-level OLS 10.25
- Only S1b and S7a beat production in both dev seasons, by small margins.

**Full production engine on identical games** (2025 wk2–18 rerun, plus
2026 wk1–4; 320 games):

| | wk2–4 2025 | wk5–8 2025 | wk9–18 2025 | 2026 wk1–4 | All |
|---|---|---|---|---|---|
| Production | 12.44 | **11.32** | 10.26 | 12.81 | 11.28 |
| + prior season (k=1, r=2/3) | 11.17 | 11.40 | 10.32 | 11.14 | 10.80 |
| S7a | 10.17 | 11.57 | 10.14 | 10.01 | 10.37 |
| Elo | 9.89 | 11.88 | 10.05 | 9.97 | 10.33 |
| home+2 | 11.17 | 12.49 | 10.85 | 10.06 | 11.03 |

## Findings
- **Production's only real weakness is the cold start (weeks 1–4).** Each
  week's regression is fit on 32 teams with only 1–3 games each. From week 5
  on, production is competitive or best.
- **More features and more flexible models hurt.** Gains came from carrying
  information across seasons, not from the choice of algorithm.
- **The QB injury adjustment** changed 8 games in 2026, and 7 got worse. That
  sample is too small to act on. Keep it and re-check at season end.
- **Power is the limit.** One season (about 270 games) resolves edges of about
  0.5 points, so small gains need several seasons to confirm.
- **These results are exploratory.** All of these games were seen during
  research.

## Challenger, not adopted: prior-season shrinkage
- **Formula:** blend each team's 5 features, PPG and PA/G with k games of last
  season's week-18 values, regressed toward the league mean:
  - prior = league + r·(last season − league)
  - value = (G·current + k·prior) / (G + k)
- **Week 1:** the result is league + r·(last − league).
- **Parameters:** k = 1, r = 2/3, fixed a priori.
- **Effect:** −0.48 overall [−0.89, −0.15], −2.1 in 2026 wks 2–4, and about
  +0.06 (null) from week 5 on.
- **Implementation:** about 60 lines. When turned off, it was bit-identical
  to production on all real inputs.
- **Test of record:** 2027 weeks 1–4. Run it alongside production and compare.

## Next milestone
At season end, grade production's published 2026 wk5–18 predictions. These
are genuinely unseen games. Compare them with home+2 and the Vegas line, and
re-check the injury adjustment.
