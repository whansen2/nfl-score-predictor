"""
NFL Score Predictor - Main Prediction Engine

This module implements a Linear Regression-based NFL game score predictor with optional
adjustments for injuries. All configuration is managed
through constants.py with optional .env overrides.

Key Features:
- 6-feature Linear Regression model using team performance statistics
- Optional injury adjustments based on QB tier ratings
- AWS Lambda deployment support

Configuration:
- All defaults defined in nfl_predictor.utils.constants
- Optional .env overrides are supported for local development
- The core prediction pipeline runs without any required API keys
"""

import logging
import os
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from nfl_predictor.utils.constants import (
    CONV_AGAINST_FILE,
    CONVERSIONS_FILE,
    DEFAULT_FEATURES,
    DEFAULT_INJURY_ADJUSTMENTS,
    DEFAULT_LOG_LEVEL,
    DEFAULT_VERBOSE_ADJUSTMENTS,
    DEFAULT_YEAR_ABBR,
    DEFENSE_FILE,
    HOME_FIELD_ADVANTAGE,
    INJURIES_FILE_NAME,
    INPUT_FILE_NAME,
    OFFENSE_FILE,
    OUTPUT_FILE_NAME,
    PROPERTIES_FILE_NAME,
    RANDOM_STATE,
    TRAIN_TEST_SPLIT_RATIO,
)
from nfl_predictor.utils.helpers import (
    get_injuries_adjustment,
    get_training_year,
    resolve_weeks,
    running_in_lambda,
    validate_csv_schema,
)

# Load optional .env overrides (all defaults live in constants.py)
load_dotenv()

# Setup logging with better formatting
log_level = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL)
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("Invalid %s value %r; using default %s", name, value, default)
        return default


# Directory containing team stats, matchups, and injury CSVs
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Control optional logic via .env with defaults from constants
ENABLE_INJURY_ADJUSTMENTS = (
    os.getenv("ENABLE_INJURY_ADJUSTMENTS", str(DEFAULT_INJURY_ADJUSTMENTS)).lower()
    == "true"
)
VERBOSE_ADJUSTMENTS = (
    os.getenv("VERBOSE_ADJUSTMENTS", str(DEFAULT_VERBOSE_ADJUSTMENTS)).lower() == "true"
)

# Load .env values for defaults (all values have defaults in constants.py)
YEAR_ABBR = _get_int_env("YEAR_ABBR", DEFAULT_YEAR_ABBR)

INJURY_REQUIRED_COLUMNS = ["Player", "Tm", "Pos", "Status"]
WeekModelCacheKey = tuple[int, int]
WeekModelCacheValue = tuple[LinearRegression, pd.DataFrame, list[str]]


# Main prediction logic
def run_predictions(matchups_path: str | None = None) -> pd.DataFrame:
    # Load team and QB properties
    with open(os.path.join(DATA_DIR, PROPERTIES_FILE_NAME)) as file:
        nfl_properties: dict[str, Any] = yaml.safe_load(file)

    team_abbreviations: dict[str, str] = nfl_properties["team_abbreviations"]
    qb_tiers: dict[str, int] = nfl_properties["qb_tiers"]
    team_qbs: dict[str, str] = nfl_properties["team_qbs"]

    # Load injuries data (if enabled)
    injuries_df = None
    if ENABLE_INJURY_ADJUSTMENTS:
        try:
            injuries_df = pd.read_csv(os.path.join(DATA_DIR, INJURIES_FILE_NAME))
            validate_csv_schema(injuries_df, INJURY_REQUIRED_COLUMNS)
        except ValueError as e:
            logger.warning(
                "Injury adjustments disabled due to invalid injuries file schema: %s",
                e,
            )
            injuries_df = None
        except FileNotFoundError:
            logger.warning("Injury adjustments enabled but injuries file not found.")

    # Matchups CSV path
    if matchups_path is None:
        matchups_path = os.path.join(DATA_DIR, INPUT_FILE_NAME)

    # Load upcoming matchups from CSV - required for operation
    if os.path.exists(matchups_path):
        df_matchups = pd.read_csv(matchups_path)
        logger.info("Loaded %s upcoming matchups from CSV", len(df_matchups))
        validate_csv_schema(df_matchups, ["Week", "Home", "Visitor", "Date"])
    else:
        msg = (
            f"Matchups CSV file not found at {matchups_path}. "
            "This file is required for predictions."
        )
        raise FileNotFoundError(msg)

    # Track weeks already printed to avoid duplicate model metrics
    printed_weeks: set[int] = set()
    # Cache trained models per (week, year)
    week_model_cache: dict[WeekModelCacheKey, WeekModelCacheValue] = {}
    # Final output rows
    results: list[list[Any]] = []

    for _, row in df_matchups.iterrows():
        week, training_week = resolve_weeks(row["Week"])
        training_year = get_training_year(row["Week"], YEAR_ABBR)
        home_team = row["Home"]
        away_team = row["Visitor"]

        # Train model only once per week and cache it
        cache_key = (week, training_year)
        if cache_key not in week_model_cache:
            try:
                df_conversions = pd.read_csv(
                    os.path.join(
                        DATA_DIR,
                        CONVERSIONS_FILE.format(week=training_week, year=training_year),
                    )
                )
                df_offense = pd.read_csv(
                    os.path.join(
                        DATA_DIR,
                        OFFENSE_FILE.format(week=training_week, year=training_year),
                    )
                )
                df_conversions_against = pd.read_csv(
                    os.path.join(
                        DATA_DIR,
                        CONV_AGAINST_FILE.format(
                            week=training_week, year=training_year
                        ),
                    )
                )
                df_defense = pd.read_csv(
                    os.path.join(
                        DATA_DIR,
                        DEFENSE_FILE.format(week=training_week, year=training_year),
                    )
                )
            except FileNotFoundError as e:
                logger.warning("Missing data file for week %s: %s", training_week, e)
                continue

            # Merge team stat datasets
            df = pd.merge(df_offense, df_conversions, on="Tm")
            df = pd.merge(df, df_conversions_against, on="Tm")
            df = pd.merge(df, df_defense, on="Tm")

            # Create engineered features
            df["PPG"] = df["PF"] / df["G"]
            df["Tot_1stD/G"] = df["Tot_1stD"] / df["G"]
            features = DEFAULT_FEATURES

            # Validate all features exist
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                logger.error("Missing required features: %s", missing_features)
                continue

            # Train/test split and model fitting
            X = df[features]
            y = df["PPG"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TRAIN_TEST_SPLIT_RATIO, random_state=RANDOM_STATE
            )
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Log model performance once per week
            if week not in printed_weeks:
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                logger.info("Training for week %s data", week)
                logger.info("Mean Absolute Error: %.3f", mae)
                logger.info("R² Score: %.3f", r2)
                if r2 < 0.5:
                    logger.warning(
                        "Model quality low for week %s (R²=%.3f), "
                        "predictions may be unreliable",
                        week,
                        r2,
                    )
                printed_weeks.add(week)

            # Cache trained model
            week_model_cache[cache_key] = (model, df, features)
        else:
            model, df, features = week_model_cache[cache_key]

        # Verify teams are in dataset
        if home_team not in df["Tm"].values or away_team not in df["Tm"].values:
            logger.warning("Skipping invalid matchup: %s vs %s", home_team, away_team)
            continue

        # Extract features for prediction
        ht_stats = df.loc[df["Tm"] == home_team, features]
        at_stats = df.loc[df["Tm"] == away_team, features]
        if ht_stats.empty or at_stats.empty:
            logger.warning("Missing stats for %s or %s", home_team, away_team)
            continue

        # Predict scores (home team gets home field advantage)
        ht_pred = round(model.predict(ht_stats)[0]) + HOME_FIELD_ADVANTAGE
        at_pred = round(model.predict(at_stats)[0])

        # Apply adjustments
        ht_adj, at_adj = 0, 0

        if ENABLE_INJURY_ADJUSTMENTS and injuries_df is not None:
            ht_adj, at_adj = get_injuries_adjustment(
                injuries_df,
                home_team,
                away_team,
                team_abbreviations,
                qb_tiers,
                team_qbs,
            )

        if VERBOSE_ADJUSTMENTS:
            logger.info("Adjustments - Injury: %s/%s", ht_adj, at_adj)

        ht_pred += ht_adj
        at_pred += at_adj

        # Determine outcome and save result
        diff = ht_pred - at_pred
        winner = home_team if diff > 0 else away_team
        result = "Tie" if diff == 0 else f"{winner} win by {abs(diff)}"
        total = ht_pred + at_pred

        results.append([week, home_team, ht_pred, away_team, at_pred, result, total])

    if results:
        df_results = pd.DataFrame(
            results,
            columns=[
                "Week",
                "Home Team",
                "Home Score",
                "Away Team",
                "Away Score",
                "Result",
                "Over/Under",
            ],
        )

        # Handle writing based on environment
        if not running_in_lambda():
            output_path = os.path.join(DATA_DIR, OUTPUT_FILE_NAME)
            df_results.to_csv(output_path, index=False)
            logger.info("Saved output to %s", output_path)
        else:
            logger.info(
                "%s generated successfully in Lambda environment.", OUTPUT_FILE_NAME
            )

        return df_results

    # No predictions made
    return pd.DataFrame()


if __name__ == "__main__":
    run_predictions()
