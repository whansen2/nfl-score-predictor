import pandas as pd
import pytest
import yaml

import nfl_predictor.nfl_ai_scores as scores
from nfl_predictor.utils.constants import (
    CONV_AGAINST_FILE,
    CONVERSIONS_FILE,
    DEFAULT_YEAR_ABBR,
    DEFENSE_FILE,
    INPUT_FILE_NAME,
    OFFENSE_FILE,
    PROPERTIES_FILE_NAME,
)


def _write_properties_file(tmp_path) -> None:
    properties = {
        "team_abbreviations": {"HomeTeam": "HOM", "AwayTeam": "AWY"},
        "qb_tiers": {"average": -4},
        "team_qbs": {
            "HomeTeam": ["Home QB", "average"],
            "AwayTeam": ["Away QB", "average"],
        },
    }
    with open(tmp_path / PROPERTIES_FILE_NAME, "w") as f:
        yaml.safe_dump(properties, f)


def _write_weekly_stats(tmp_path, week: int, year: int) -> None:
    offense = pd.DataFrame(
        [
            {
                "Tm": "HomeTeam",
                "PF": 420,
                "G": 17,
                "Tot_1stD": 360,
                "Sc%": 45.0,
                "Y/P": 6.1,
                "TO%": 8.0,
            },
            {
                "Tm": "AwayTeam",
                "PF": 380,
                "G": 17,
                "Tot_1stD": 340,
                "Sc%": 42.0,
                "Y/P": 5.8,
                "TO%": 10.0,
            },
        ]
    )
    conversions = pd.DataFrame(
        [
            {"Tm": "HomeTeam", "RZTD": 30, "RZPct": 60.0},
            {"Tm": "AwayTeam", "RZTD": 25, "RZPct": 55.0},
        ]
    )
    conversions_against = pd.DataFrame(
        [
            {"Tm": "HomeTeam", "RZTD": 20, "RZPct": 50.0},
            {"Tm": "AwayTeam", "RZTD": 22, "RZPct": 52.0},
        ]
    )
    defense = pd.DataFrame(
        [
            {"Tm": "HomeTeam", "Sc%": 35.0, "Y/P": 5.0, "TO%": 12.0},
            {"Tm": "AwayTeam", "Sc%": 38.0, "Y/P": 5.3, "TO%": 11.0},
        ]
    )

    offense.to_csv(tmp_path / OFFENSE_FILE.format(week=week, year=year), index=False)
    conversions.to_csv(
        tmp_path / CONVERSIONS_FILE.format(week=week, year=year), index=False
    )
    conversions_against.to_csv(
        tmp_path / CONV_AGAINST_FILE.format(week=week, year=year), index=False
    )
    defense.to_csv(tmp_path / DEFENSE_FILE.format(week=week, year=year), index=False)


def test_run_predictions_uses_numeric_week_for_training_data(
    tmp_path, monkeypatch
) -> None:
    _write_properties_file(tmp_path)
    _write_weekly_stats(tmp_path, week=1, year=DEFAULT_YEAR_ABBR)

    matchups = pd.DataFrame(
        [
            {
                "Week": 2,
                "Visitor": "AwayTeam",
                "Home": "HomeTeam",
                "Date": "2026-09-15",
            }
        ]
    )
    matchups_path = tmp_path / INPUT_FILE_NAME
    matchups.to_csv(matchups_path, index=False)

    monkeypatch.setattr(scores, "path", str(tmp_path))
    monkeypatch.setattr(scores, "ENABLE_UPSETS_AGENT", False)
    monkeypatch.setattr(scores, "ENABLE_INJURY_ADJUSTMENTS", False)
    monkeypatch.setattr(scores, "YEAR_ABBR", DEFAULT_YEAR_ABBR)

    results = scores.run_predictions(matchups_path=str(matchups_path))

    assert len(results) == 1
    assert results.iloc[0]["Week"] == 2


def test_run_predictions_raises_for_missing_matchup_columns(
    tmp_path, monkeypatch
) -> None:
    _write_properties_file(tmp_path)

    matchups = pd.DataFrame([{"Week": 2, "Visitor": "AwayTeam", "Home": "HomeTeam"}])
    matchups_path = tmp_path / INPUT_FILE_NAME
    matchups.to_csv(matchups_path, index=False)

    monkeypatch.setattr(scores, "path", str(tmp_path))

    with pytest.raises(ValueError, match="Missing required columns"):
        scores.run_predictions(matchups_path=str(matchups_path))


def test_run_predictions_raises_for_missing_matchups_file(
    tmp_path, monkeypatch
) -> None:
    _write_properties_file(tmp_path)
    missing_matchups_path = tmp_path / "missing_matchups.csv"

    monkeypatch.setattr(scores, "path", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="Matchups CSV file not found"):
        scores.run_predictions(matchups_path=str(missing_matchups_path))
