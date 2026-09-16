"""
NFL model backtest & experiment harness (2025-26 regular season, weeks 3-18).

This file has one job: give TRUE, out-of-sample metrics so we can decide,
honestly and efficiently, where nfl_ai_scores.py can be improved. Any change to
the model is only an "improvement" if it moves these numbers on held-out games.

Three layers
------------
1. GROUND TRUTH  -- actual per-team weekly points from cumulative PF deltas:
       points(week N) = PF(thru N) - PF(thru N-1)
   (Verified exact against real boxscores for week 4.)

2. RECORDED grader -- grades the predictions that ACTUALLY SHIPPED, read from
   season_25_26/week_{3..18}_25.csv. This is the deployed model's real 2025
   report card. NOTE: 2025 predictions carried NO QB-injury adjustment (that
   data did not exist yet); the injury lever is a 2026-onward addition.

3. WALK-FORWARD ENGINE -- regenerates predictions from scratch for any model
   `Variant`, training ONLY on stats thru week N-1 to predict week N (no leakage).
   `compare()` prints baseline-vs-variant deltas so experiments are objective.

Metrics (all out-of-sample):
  SU%   straight-up winner accuracy
  ATS%  against the spread (>52.4% = beats standard -110 juice)
  score/total/margin MAE  -- the accuracy levers ATS ultimately rides on

ATS convention (matches spread_analysis_final.py):
  pick FAVORITE -> right iff favorite wins by MORE than the spread;
  pick UNDERDOG -> right iff dog wins outright OR loses by LESS than the spread;
  margin == spread (integer line) = PUSH, excluded.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --- paths / config ----------------------------------------------------------
# SNAP_DIR (weekly stat snapshots) lives inside the repo, so derive it from this
# file's own location -> portable across clones/machines. This script sits in
# <repo>/scripts/, hence parents[1] == repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
SNAP_DIR = REPO_ROOT / "nfl_predictor" / "data" / "season_25"
# PRED_DIR (shipped predictions + lines + recorded scores) lives OUTSIDE the repo
# in the Desktop data archive. This is the one path to edit if that archive moves.
PRED_DIR = "/Users/whansen/Desktop/Data Science/nfl_stats/season_25_26"
WEEKS = range(3, 19)

# These mirror the deployed nfl_ai_scores.py config -- keep them in sync so the
# walk-forward engine reproduces the shipped model as its baseline.
DEFAULT_FEATURES = ["Sc%_x", "Tot_1stD/G", "Y/P_x", "RZPct_x", "TO%_x", "Sc%_y"]
HOME_FIELD_ADVANTAGE = 1
TRAIN_TEST_SPLIT_RATIO = 0.33
RANDOM_STATE = 42


# --- ground truth ------------------------------------------------------------
def week_points(week: int) -> dict[str, int]:
    """Actual points each team scored in `week`, from cumulative PF deltas."""
    prev = pd.read_csv(f"{SNAP_DIR}/nfl_team_offense_thru_week_{week - 1}_25.csv")[
        ["Tm", "G", "PF"]
    ]
    cur = pd.read_csv(f"{SNAP_DIR}/nfl_team_offense_thru_week_{week}_25.csv")[
        ["Tm", "G", "PF"]
    ]
    m = prev.merge(cur, on="Tm", suffixes=("_p", "_c"))
    m = m[(m["G_c"] - m["G_p"]) == 1]  # only teams that played exactly one game
    return {r.Tm: int(r.PF_c - r.PF_p) for r in m.itertuples()}


def build_snapshot(week: int) -> pd.DataFrame:
    """Merged team-stat frame thru `week`, replicating nfl_ai_scores.py exactly
    (offense -> conversions -> conversions_against -> defense; default _x/_y
    suffixes give Sc%_x=offense / Sc%_y=defense, RZPct_x=offense, etc.)."""
    off = pd.read_csv(f"{SNAP_DIR}/nfl_team_offense_thru_week_{week}_25.csv")
    conv = pd.read_csv(f"{SNAP_DIR}/nfl_conversions_thru_week_{week}_25.csv")
    cva = pd.read_csv(f"{SNAP_DIR}/nfl_conversions_against_thru_week_{week}_25.csv")
    dfn = pd.read_csv(f"{SNAP_DIR}/nfl_team_defense_thru_week_{week}_25.csv")
    df = off.merge(conv, on="Tm").merge(cva, on="Tm").merge(dfn, on="Tm")
    df["PPG"] = df["PF"] / df["G"]
    df["Tot_1stD/G"] = df["Tot_1stD"] / df["G"]
    return df


# --- name / line helpers -----------------------------------------------------
def nick_to_full(teams) -> dict[str, str]:
    """Last-token nickname ('Rams', '49ers') -> full team name."""
    return {t.split()[-1]: t for t in teams}


def parse_line(line: str, nick_map: dict[str, str]) -> tuple[str, float]:
    """'Bills -12.5' -> ('Buffalo Bills', 12.5)  == (favorite_full, spread)."""
    fav_nick, num = line.rsplit("-", 1)
    return nick_map[fav_nick.strip()], float(num.strip())


# --- grading (shared by recorded + every variant) ----------------------------
def grade_matchup(
    home: str,
    away: str,
    ah: int,
    aa: int,
    mh: float,
    ma: float,
    fav: str,
    spread: float,
    model_pick: str,
) -> dict:
    """Grade one game. `mh/ma` = model's predicted home/away score; `model_pick`
    = the team the model bets ATS. Returns a row of correctness flags + errors."""
    fav_margin = ah - aa if fav == home else aa - ah

    actual_winner = None if ah == aa else (home if ah > aa else away)
    model_winner = None if mh == ma else (home if mh > ma else away)

    push = fav_margin == spread
    if push:
        ats = None
    elif model_pick == fav:
        ats = fav_margin > spread
    else:
        ats = fav_margin < spread

    winner_correct = (
        None
        if (actual_winner is None or model_winner is None)
        else model_winner == actual_winner
    )
    return dict(
        home=home,
        away=away,
        ah=ah,
        aa=aa,
        mh=mh,
        ma=ma,
        fav=fav,
        spread=spread,
        fav_margin=fav_margin,
        model_pick=model_pick,
        winner_correct=winner_correct,
        ats=ats,
        push=push,
        model_total=mh + ma,
        actual_total=ah + aa,
        home_err=abs(mh - ah),
        away_err=abs(ma - aa),
        margin_err=abs((mh - ma) - (ah - aa)),
    )


def metrics(rows: pd.DataFrame) -> dict:
    """Collapse graded rows into headline numbers."""
    w = rows[rows.winner_correct.notna()]
    a = rows[rows.ats.notna()]
    err = pd.concat([rows.home_err, rows.away_err])
    return dict(
        n=len(rows),
        su_pct=w.winner_correct.mean() * 100,
        su=f"{int(w.winner_correct.sum())}/{len(w)}",
        ats_pct=a.ats.mean() * 100,
        ats=f"{int(a.ats.sum())}/{len(a)}",
        pushes=int(rows.push.sum()),
        score_mae=err.mean(),
        total_mae=(rows.model_total - rows.actual_total).abs().mean(),
        margin_mae=rows.margin_err.mean(),
    )


# --- variants ----------------------------------------------------------------
# A Variant fully describes a model experiment. `score_fn` maps a fitted model +
# snapshot + team + opponent -> that team's predicted points (BEFORE home-field
# advantage). The `opp` arg is where opponent-aware logic plugs in.
ScoreFn = Callable[[object, pd.DataFrame, str, str, list[str]], float]


def independent_ppg(
    model, snap: pd.DataFrame, team: str, opp: str, feats: list[str]
) -> float:
    """Current model: predict a team's own season-avg PPG from its own stats.
    Ignores `opp` entirely -- this is the structural blindness we're testing."""
    return float(model.predict(snap.loc[snap.Tm == team, feats])[0])


def opponent_blend(weight: float = 0.5) -> ScoreFn:
    """Matchup-aware: nudge a team's predicted points by how leaky the OPPONENT's
    defense is vs league average (opp points-allowed/game - league avg), scaled
    by `weight`. weight=0 == independent; weight=1 == full additive matchup."""

    def fn(model, snap, team, opp, feats):
        base = float(model.predict(snap.loc[snap.Tm == team, feats])[0])
        pa_pg = snap["PA"] / snap["G"]
        league = pa_pg.mean()
        o = snap.Tm == opp
        opp_pa_pg = float((snap.loc[o, "PA"] / snap.loc[o, "G"]).iloc[0])
        return base + weight * (opp_pa_pg - league)

    return fn


def ridge_scaled(alpha: float = 1.0):
    """Ridge needs standardized features (our 6 are on wildly different scales)."""
    return lambda: make_pipeline(StandardScaler(), Ridge(alpha=alpha))


@dataclass
class Variant:
    name: str
    features: list[str] = field(default_factory=lambda: list(DEFAULT_FEATURES))
    model_factory: Callable[[], object] = LinearRegression
    fit_mode: str = (
        "train_split"  # "train_split" mirrors deployed; "all" = use 32 teams
    )
    score_fn: ScoreFn = independent_ppg
    hfa: float = HOME_FIELD_ADVANTAGE

    def fit_week(self, snap: pd.DataFrame):
        X, y = snap[self.features], snap["PPG"]
        if self.fit_mode == "train_split":
            X, _, y, _ = train_test_split(
                X, y, test_size=TRAIN_TEST_SPLIT_RATIO, random_state=RANDOM_STATE
            )
        model = self.model_factory()
        model.fit(X, y)
        return model


def run_variant(v: Variant) -> pd.DataFrame:
    """Walk-forward, leak-free: to predict week N, train on stats thru N-1."""
    rows = []
    for wk in WEEKS:
        pts = week_points(wk)
        snap = build_snapshot(wk - 1)
        model = v.fit_week(snap)
        nick_map = nick_to_full(pts)
        preds = pd.read_csv(f"{PRED_DIR}/week_{wk}_25.csv")
        for _, p in preds.iterrows():
            home, away = p["Home Team"], p["Away Team"]
            if home not in pts or away not in pts:
                continue
            if home not in snap.Tm.values or away not in snap.Tm.values:
                continue
            mh = round(v.score_fn(model, snap, home, away, v.features)) + v.hfa
            ma = round(v.score_fn(model, snap, away, home, v.features))
            fav, spread = parse_line(p["Line"], nick_map)
            model_pick = (
                fav
                if (mh - ma if fav == home else ma - mh) > spread
                else (away if fav == home else home)
            )
            r = grade_matchup(
                home, away, pts[home], pts[away], mh, ma, fav, spread, model_pick
            )
            r["wk"] = wk
            rows.append(r)
    return pd.DataFrame(rows)


def run_recorded() -> pd.DataFrame:
    """Grade the predictions that actually shipped (from the week CSVs)."""
    rows = []
    for wk in WEEKS:
        pts = week_points(wk)
        nick_map = nick_to_full(pts)
        preds = pd.read_csv(f"{PRED_DIR}/week_{wk}_25.csv")
        for _, p in preds.iterrows():
            home, away = p["Home Team"], p["Away Team"]
            if home not in pts or away not in pts:
                print(f"  ! week {wk}: no actual for {home} / {away} — skipped")
                continue
            fav, spread = parse_line(p["Line"], nick_map)
            raw = str(p["Model Says"])  # recorded ATS pick (nick wks 3-6, full wks 7+)
            model_pick = nick_map.get(raw.split()[-1], raw)
            r = grade_matchup(
                home,
                away,
                pts[home],
                pts[away],
                int(p["Home Score"]),
                int(p["Away Score"]),
                fav,
                spread,
                model_pick,
            )
            r["wk"] = wk
            rows.append(r)
    return pd.DataFrame(rows)


# --- reporting ---------------------------------------------------------------
def show(rows: pd.DataFrame, label: str) -> dict:
    m = metrics(rows)
    print(f"\n{label}")
    print(
        f"  n={m['n']:>3}   SU {m['su']:>7} ({m['su_pct']:4.1f}%)   "
        f"ATS {m['ats']:>7} ({m['ats_pct']:4.1f}%)  pushes {m['pushes']}"
    )
    print(
        f"  MAE  score {m['score_mae']:5.2f}   total {m['total_mae']:5.2f}   "
        f"margin {m['margin_mae']:5.2f}"
    )
    return m


def compare(baseline: dict, variant: dict, name: str) -> None:
    d = lambda k: variant[k] - baseline[k]  # noqa: E731
    print(
        f"\n  Δ {name} vs baseline:  "
        f"SU {d('su_pct'):+4.1f}pp   ATS {d('ats_pct'):+4.1f}pp   "
        f"score MAE {d('score_mae'):+.2f}   margin MAE {d('margin_mae'):+.2f}"
    )


if __name__ == "__main__":
    print("=" * 78)
    print("NFL MODEL BACKTEST 2025-26  (weeks 3-18, out-of-sample, leak-free)")
    print("=" * 78)

    base = metrics(run_variant(Variant("baseline")))

    def row(v: Variant, base=base) -> None:
        m = metrics(run_variant(v))
        d = lambda k: m[k] - base[k]  # noqa: E731
        print(
            f"  {v.name:26} SU {m['su_pct']:5.1f} ({d('su_pct'):+4.1f})  "
            f"ATS {m['ats_pct']:5.1f} ({d('ats_pct'):+4.1f})  "
            f"sMAE {m['score_mae']:5.2f} ({d('score_mae'):+.2f})  "
            f"mMAE {m['margin_mae']:5.2f} ({d('margin_mae'):+.2f})"
        )

    print(
        f"\n  {'BASELINE':26} SU {base['su_pct']:5.1f}         "
        f"ATS {base['ats_pct']:5.1f}         "
        f"sMAE {base['score_mae']:5.2f}         mMAE {base['margin_mae']:5.2f}"
    )
    print("  " + "-" * 74)

    # --- each lever changed in ISOLATION from baseline (Δ shown vs baseline) ---
    row(Variant("fit-on-all-32", fit_mode="all"))
    row(Variant("ridge(a=1, scaled)", model_factory=ridge_scaled(1.0)))
    row(Variant("HFA=2", hfa=2))
    row(Variant("HFA=3", hfa=3))
    row(Variant("opp-blend w=0.25", score_fn=opponent_blend(0.25)))
    row(Variant("opp-blend w=0.50", score_fn=opponent_blend(0.50)))
    row(Variant("opp-blend w=1.00", score_fn=opponent_blend(1.00)))

    # --- fine weight sweep: pick the MARGIN-MAE-optimal weight, not ATS-optimal --
    print("\n  weight sweep (trust margin MAE, the accuracy signal):")
    for w in (0.10, 0.20, 0.30, 0.40, 0.50):
        row(Variant(f"opp-blend w={w:.2f}", score_fn=opponent_blend(w)))

    # --- combine the trustworthy winners; confirm they stack, not overfit -------
    print("\n  combinations of trustworthy winners:")
    row(Variant("opp0.30 + HFA2", score_fn=opponent_blend(0.30), hfa=2))
    row(Variant("opp0.40 + HFA2", score_fn=opponent_blend(0.40), hfa=2))
    row(
        Variant(
            "opp0.30 + HFA2 + fitall",
            score_fn=opponent_blend(0.30),
            hfa=2,
            fit_mode="all",
        )
    )

    print("\n  (Δ = change vs baseline. sMAE/mMAE are the trustworthy signal;")
    print("   SU/ATS are noisier over 237 games. Winners must also make sense.)")
