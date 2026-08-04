import logging
import os
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def validate_csv_schema(df: pd.DataFrame, required_cols: list[str]) -> None:
    """Validate that DataFrame has all required columns."""
    missing = set(required_cols) - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)


def running_in_lambda() -> bool:
    """Detect if running in AWS Lambda environment."""
    return os.getenv("AWS_EXECUTION_ENV") is not None


def get_training_week(week_value: Any) -> int:
    """
    Determine which week's data to use for training.

    Args:
        week_value: Week identifier (int or string like "WildCard")

    Returns:
        Training week number (defaults to 18 for postseason)
    """
    try:
        week = int(week_value)
    except (TypeError, ValueError):
        return 18  # For postseason (e.g., "WildCard", "SuperBowl", etc.)

    return week if week > 0 else 18


def resolve_weeks(week_value: Any) -> tuple[int, int]:
    """
    Normalize matchup week values and derive training week.

    Numeric matchup weeks use previous-week training data.
    Week 1 falls back to week 18 data because no week 0 dataset exists.
    Postseason labels map to week 19 with training week 18.
    """
    try:
        week = int(week_value)
    except (TypeError, ValueError):
        return 19, 18

    return week, get_training_week(week - 1)


def get_injuries_adjustment(
    injuries_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    team_abbreviations: dict[str, str],
    qb_tiers: dict[str, int],
    team_qbs: dict[str, Any],
) -> tuple[int, int]:
    """
    Calculate injury-based QB adjustments for both teams.

    Args:
        injuries_df: DataFrame containing injury data
        home_team: Home team name
        away_team: Away team name
        team_abbreviations: Team name to abbreviation mapping
        qb_tiers: QB tier to adjustment value mapping
        team_qbs: Team to QB info mapping

    Returns:
        Tuple of (home_adjustment, away_adjustment)
    """
    if not isinstance(injuries_df, pd.DataFrame) or injuries_df.empty:
        return 0, 0

    # Drop rows without a listed injury status (optional but keeps data clean)
    injuries_df = injuries_df.dropna(subset=["Status", "Pos"])

    # Only keep relevant fields
    relevant_columns = ["Player", "Pos", "Status"]
    team_injuries = {
        team: group[relevant_columns].to_dict(orient="records")
        for team, group in injuries_df.groupby("Tm")
    }

    def qb_adjust(team):
        abbr = team_abbreviations.get(team)
        if not abbr:
            return 0

        team_qb_list = team_injuries.get(abbr, [])
        qb_info = team_qbs.get(team, (None, "average"))
        qb_name = qb_info[0] if isinstance(qb_info, (tuple, list)) else qb_info

        for player in team_qb_list:
            if (
                player["Pos"] == "QB"
                and player.get("Status", "").lower()
                in ["questionable", "doubtful", "out"]
                and player["Player"] == qb_name
            ):
                return qb_tiers.get(qb_info[1], 0)
        return 0

    return qb_adjust(home_team), qb_adjust(away_team)
