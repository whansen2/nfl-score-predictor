import importlib
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

import nfl_predictor.nfl_ai_scores as scores
from nfl_predictor.utils.constants import (
    CONV_AGAINST_FILE,
    CONVERSIONS_FILE,
    DEFAULT_YEAR_ABBR,
    DEFENSE_FILE,
    INJURIES_FILE_NAME,
    INPUT_FILE_NAME,
    OFFENSE_FILE,
    OUTPUT_FILE_NAME,
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
            {
                "Tm": "ThirdTeam",
                "PF": 400,
                "G": 17,
                "Tot_1stD": 350,
                "Sc%": 44.0,
                "Y/P": 6.0,
                "TO%": 9.0,
            },
            {
                "Tm": "FourthTeam",
                "PF": 360,
                "G": 17,
                "Tot_1stD": 330,
                "Sc%": 40.0,
                "Y/P": 5.6,
                "TO%": 11.0,
            },
        ]
    )
    conversions = pd.DataFrame(
        [
            {"Tm": "HomeTeam", "RZTD": 30, "RZPct": 60.0},
            {"Tm": "AwayTeam", "RZTD": 25, "RZPct": 55.0},
            {"Tm": "ThirdTeam", "RZTD": 28, "RZPct": 58.0},
            {"Tm": "FourthTeam", "RZTD": 23, "RZPct": 53.0},
        ]
    )
    conversions_against = pd.DataFrame(
        [
            {"Tm": "HomeTeam", "RZTD": 20, "RZPct": 50.0},
            {"Tm": "AwayTeam", "RZTD": 22, "RZPct": 52.0},
            {"Tm": "ThirdTeam", "RZTD": 21, "RZPct": 51.0},
            {"Tm": "FourthTeam", "RZTD": 24, "RZPct": 54.0},
        ]
    )
    defense = pd.DataFrame(
        [
            {"Tm": "HomeTeam", "Sc%": 35.0, "Y/P": 5.0, "TO%": 12.0},
            {"Tm": "AwayTeam", "Sc%": 38.0, "Y/P": 5.3, "TO%": 11.0},
            {"Tm": "ThirdTeam", "Sc%": 36.0, "Y/P": 5.1, "TO%": 10.0},
            {"Tm": "FourthTeam", "Sc%": 39.0, "Y/P": 5.4, "TO%": 9.0},
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

    monkeypatch.setattr(scores, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(scores, "ENABLE_INJURY_ADJUSTMENTS", False)
    monkeypatch.setattr(scores, "YEAR_ABBR", DEFAULT_YEAR_ABBR)

    results = scores.run_predictions(matchups_path=str(matchups_path))

    assert len(results) == 1
    assert results.iloc[0]["Week"] == 2


def test_run_predictions_uses_week_18_data_for_week_1_matchups(
    tmp_path, monkeypatch
) -> None:
    _write_properties_file(tmp_path)
    current_year = DEFAULT_YEAR_ABBR + 1
    _write_weekly_stats(tmp_path, week=18, year=DEFAULT_YEAR_ABBR)

    matchups = pd.DataFrame(
        [
            {
                "Week": 1,
                "Visitor": "AwayTeam",
                "Home": "HomeTeam",
                "Date": "2026-09-08",
            }
        ]
    )
    matchups_path = tmp_path / INPUT_FILE_NAME
    matchups.to_csv(matchups_path, index=False)

    monkeypatch.setattr(scores, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(scores, "ENABLE_INJURY_ADJUSTMENTS", False)
    monkeypatch.setattr(scores, "YEAR_ABBR", current_year)

    results = scores.run_predictions(matchups_path=str(matchups_path))

    assert len(results) == 1
    assert results.iloc[0]["Week"] == 1


def test_run_predictions_raises_for_missing_matchup_columns(
    tmp_path, monkeypatch
) -> None:
    _write_properties_file(tmp_path)

    matchups = pd.DataFrame([{"Week": 2, "Visitor": "AwayTeam", "Home": "HomeTeam"}])
    matchups_path = tmp_path / INPUT_FILE_NAME
    matchups.to_csv(matchups_path, index=False)

    monkeypatch.setattr(scores, "DATA_DIR", str(tmp_path))

    with pytest.raises(ValueError, match="Missing required columns"):
        scores.run_predictions(matchups_path=str(matchups_path))


def test_run_predictions_raises_for_missing_matchups_file(
    tmp_path, monkeypatch
) -> None:
    _write_properties_file(tmp_path)
    missing_matchups_path = tmp_path / "missing_matchups.csv"

    monkeypatch.setattr(scores, "DATA_DIR", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="Matchups CSV file not found"):
        scores.run_predictions(matchups_path=str(missing_matchups_path))


def test_run_predictions_uses_default_matchups_path_and_warns_when_injuries_missing(
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
    matchups.to_csv(tmp_path / INPUT_FILE_NAME, index=False)

    monkeypatch.setattr(scores, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(scores, "ENABLE_INJURY_ADJUSTMENTS", True)
    monkeypatch.setattr(scores, "YEAR_ABBR", DEFAULT_YEAR_ABBR)

    with patch.object(scores, "logger") as mock_logger:
        results = scores.run_predictions()

    assert len(results) == 1
    mock_logger.warning.assert_any_call(
        "Injury adjustments enabled but injuries file not found."
    )


def test_run_predictions_accepts_injuries_file_without_injury_comment(
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

    pd.DataFrame(
        [
            {
                "Player": "Home QB",
                "Tm": "HOM",
                "Pos": "QB",
                "Status": "Out",
            }
        ]
    ).to_csv(tmp_path / INJURIES_FILE_NAME, index=False)

    monkeypatch.setattr(scores, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(scores, "YEAR_ABBR", DEFAULT_YEAR_ABBR)

    monkeypatch.setattr(scores, "ENABLE_INJURY_ADJUSTMENTS", False)
    baseline = scores.run_predictions(matchups_path=str(matchups_path))

    monkeypatch.setattr(scores, "ENABLE_INJURY_ADJUSTMENTS", True)
    results = scores.run_predictions(matchups_path=str(matchups_path))

    assert len(results) == 1
    assert results.iloc[0]["Home Score"] == baseline.iloc[0]["Home Score"] - 4


def test_run_predictions_uses_default_year_when_env_value_is_invalid(
    monkeypatch,
) -> None:
    monkeypatch.setenv("YEAR_ABBR", "invalid")

    reloaded_scores = importlib.reload(scores)

    assert reloaded_scores.YEAR_ABBR == DEFAULT_YEAR_ABBR

    monkeypatch.delenv("YEAR_ABBR", raising=False)
    importlib.reload(scores)


def test_run_predictions_returns_empty_when_weekly_data_missing(
    tmp_path, monkeypatch
) -> None:
    _write_properties_file(tmp_path)

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

    monkeypatch.setattr(scores, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(scores, "ENABLE_INJURY_ADJUSTMENTS", False)
    monkeypatch.setattr(scores, "YEAR_ABBR", DEFAULT_YEAR_ABBR)

    with patch.object(scores, "logger") as mock_logger:
        results = scores.run_predictions(matchups_path=str(matchups_path))

    assert results.empty
    warning_message = mock_logger.warning.call_args_list[0].args[0]
    assert "Missing data file for week" in warning_message
    assert mock_logger.warning.call_args_list[0].args[1] == 1


def test_run_predictions_returns_empty_when_required_feature_missing(
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

    defense_without_scoring_pct = pd.DataFrame(
        [
            {"Tm": "HomeTeam", "Y/P": 5.0, "TO%": 12.0},
            {"Tm": "AwayTeam", "Y/P": 5.3, "TO%": 11.0},
            {"Tm": "ThirdTeam", "Y/P": 5.1, "TO%": 10.0},
            {"Tm": "FourthTeam", "Y/P": 5.4, "TO%": 9.0},
        ]
    )
    defense_without_scoring_pct.to_csv(
        tmp_path / DEFENSE_FILE.format(week=1, year=DEFAULT_YEAR_ABBR), index=False
    )

    monkeypatch.setattr(scores, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(scores, "ENABLE_INJURY_ADJUSTMENTS", False)
    monkeypatch.setattr(scores, "YEAR_ABBR", DEFAULT_YEAR_ABBR)

    with patch.object(scores, "logger") as mock_logger:
        results = scores.run_predictions(matchups_path=str(matchups_path))

    assert results.empty
    mock_logger.error.assert_called_once()
    assert "Missing required features" in mock_logger.error.call_args.args[0]


def test_run_predictions_skips_invalid_matchup_teams(tmp_path, monkeypatch) -> None:
    _write_properties_file(tmp_path)
    _write_weekly_stats(tmp_path, week=1, year=DEFAULT_YEAR_ABBR)

    matchups = pd.DataFrame(
        [
            {
                "Week": 2,
                "Visitor": "AwayTeam",
                "Home": "UnknownTeam",
                "Date": "2026-09-15",
            }
        ]
    )
    matchups_path = tmp_path / INPUT_FILE_NAME
    matchups.to_csv(matchups_path, index=False)

    monkeypatch.setattr(scores, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(scores, "ENABLE_INJURY_ADJUSTMENTS", False)
    monkeypatch.setattr(scores, "YEAR_ABBR", DEFAULT_YEAR_ABBR)

    with patch.object(scores, "logger") as mock_logger:
        results = scores.run_predictions(matchups_path=str(matchups_path))

    assert results.empty
    mock_logger.warning.assert_any_call(
        "Skipping invalid matchup: %s vs %s", "UnknownTeam", "AwayTeam"
    )


def test_run_predictions_applies_verbose_injury_adjustments(
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
    pd.DataFrame(columns=["Player", "Tm", "Pos", "Status", "Injury Comment"]).to_csv(
        tmp_path / INJURIES_FILE_NAME, index=False
    )

    monkeypatch.setattr(scores, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(scores, "YEAR_ABBR", DEFAULT_YEAR_ABBR)

    monkeypatch.setattr(scores, "ENABLE_INJURY_ADJUSTMENTS", False)
    monkeypatch.setattr(scores, "VERBOSE_ADJUSTMENTS", False)
    baseline = scores.run_predictions(matchups_path=str(matchups_path))

    monkeypatch.setattr(scores, "ENABLE_INJURY_ADJUSTMENTS", True)
    monkeypatch.setattr(scores, "VERBOSE_ADJUSTMENTS", True)
    with (
        patch.object(scores, "get_injuries_adjustment", return_value=(-4, -2)),
        patch.object(scores, "logger") as mock_logger,
    ):
        adjusted = scores.run_predictions(matchups_path=str(matchups_path))

    assert adjusted.iloc[0]["Home Score"] == baseline.iloc[0]["Home Score"] - 4
    assert adjusted.iloc[0]["Away Score"] == baseline.iloc[0]["Away Score"] - 2
    mock_logger.info.assert_any_call("Adjustments - Injury: %s/%s", -4, -2)


def test_run_predictions_logs_generated_files_in_lambda_environment(
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

    monkeypatch.setattr(scores, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(scores, "ENABLE_INJURY_ADJUSTMENTS", False)
    monkeypatch.setattr(scores, "YEAR_ABBR", DEFAULT_YEAR_ABBR)
    monkeypatch.setattr(scores, "running_in_lambda", lambda: True)

    with patch.object(scores, "logger") as mock_logger:
        results = scores.run_predictions(matchups_path=str(matchups_path))

    assert len(results.columns) == 7
    assert not (tmp_path / OUTPUT_FILE_NAME).exists()
    mock_logger.info.assert_any_call(
        "%s generated successfully in Lambda environment.", OUTPUT_FILE_NAME
    )
