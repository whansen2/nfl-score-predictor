"""
NFL Score Predictor - Main Prediction Engine

This module implements a Linear Regression-based NFL game score predictor with optional
adjustments for injuries, weather, and upset detection. All configuration is managed
through constants.py with optional .env overrides.

Key Features:
- 6-feature Linear Regression model using team performance statistics
- Optional injury adjustments based on QB tier ratings
- Optional weather adjustments for outdoor stadiums
- Upset detection via separate agent module
- AWS Lambda deployment support

Configuration:
- All defaults defined in nfl_predictor.utils.constants
- Only OPENAI_API_KEY required in .env file
- Other values can be optionally overridden via environment variables
"""

from dotenv import load_dotenv
# Load .env values (only OPENAI_API_KEY required - all other config in constants.py)
load_dotenv()

import os
import logging
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from nfl_predictor.agents.upsets_ai_agent import run_upsets_agent
from nfl_predictor.utils.constants import (
    INPUT_FILE_NAME,
    INJURIES_FILE_NAME,
    PROPERTIES_FILE_NAME,
    OUTPUT_FILE_NAME,
    FLAGGED_OUTPUT_FILE_NAME,
    CONVERSIONS_FILE,
    OFFENSE_FILE,
    DEFENSE_FILE,
    CONV_AGAINST_FILE,
    DEFAULT_FEATURES,
    HOME_FIELD_ADVANTAGE,
    TRAIN_TEST_SPLIT_RATIO,
    RANDOM_STATE,
    DEFAULT_WEEK_NUMBER,
    DEFAULT_YEAR_ABBR,
    DEFAULT_GAME_DATE,
    DEFAULT_HOME_TEAM,
    DEFAULT_AWAY_TEAM,
    DEFAULT_LOG_LEVEL,
    DEFAULT_VERBOSE_ADJUSTMENTS,
    DEFAULT_INJURY_ADJUSTMENTS,
    DEFAULT_WEATHER_ADJUSTMENTS,
    DEFAULT_UPSETS_AGENT
)
from nfl_predictor.utils.helpers import (
    running_in_lambda,
    get_training_week,
    get_injuries_adjustment,
    get_weather_adjustment,
)
from nfl_predictor.utils.env_setup import configure_nfl_stadiums_resource_dir
from nfl_stadiums import NFLStadiums

# Setup logging with better formatting
log_level = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Determine working path based on environment
path = "/var/task/nfl_predictor/data" if running_in_lambda() else os.path.join(os.path.dirname(__file__), "data")

# Control optional logic via .env with defaults from constants
ENABLE_INJURY_ADJUSTMENTS = os.getenv("ENABLE_INJURY_ADJUSTMENTS", str(DEFAULT_INJURY_ADJUSTMENTS)).lower() == "true"
ENABLE_WEATHER_ADJUSTMENTS = os.getenv("ENABLE_WEATHER_ADJUSTMENTS", str(DEFAULT_WEATHER_ADJUSTMENTS)).lower() == "true"
VERBOSE_ADJUSTMENTS = os.getenv("VERBOSE_ADJUSTMENTS", str(DEFAULT_VERBOSE_ADJUSTMENTS)).lower() == "true"
ENABLE_UPSETS_AGENT = os.getenv("ENABLE_UPSETS_AGENT", str(DEFAULT_UPSETS_AGENT)).lower() == "true"

# Load .env values for defaults if matchup CSV isn't found (all values have defaults in constants.py)
WEEK_NUMBER = int(os.getenv("WEEK_NUMBER", DEFAULT_WEEK_NUMBER))
YEAR_ABBR = int(os.getenv("YEAR_ABBR", DEFAULT_YEAR_ABBR))
GAME_DATE = os.getenv("GAME_DATE", DEFAULT_GAME_DATE)
HOME_TEAM = os.getenv("HOME_TEAM", DEFAULT_HOME_TEAM)
AWAY_TEAM = os.getenv("AWAY_TEAM", DEFAULT_AWAY_TEAM)

# Monkey-patch and get resource dir for nfl_stadiums
resource_dir = configure_nfl_stadiums_resource_dir()

# Instantiate stadiums object
stad = NFLStadiums()
if running_in_lambda():
    logger.info(f"Using NFL stadium resource dir: {stad._resources_dir}")

# Main prediction logic
def run_predictions():
    # Load team, QB, and weather properties
    with open(os.path.join(path, PROPERTIES_FILE_NAME), "r") as file:
        nfl_properties = yaml.safe_load(file)

    team_abbreviations = nfl_properties["team_abbreviations"]
    qb_tiers = nfl_properties["qb_tiers"]
    team_qbs = nfl_properties["team_qbs"]
    weather_tiers = nfl_properties["weather_tiers"]

    # Load injuries data (if enabled)
    injuries_df = None
    if ENABLE_INJURY_ADJUSTMENTS:
        try:
            injuries_df = pd.read_csv(os.path.join(path, INJURIES_FILE_NAME))
        except FileNotFoundError:
            logger.warning("Injury adjustments enabled but injuries file not found.")

    # Matchups CSV path
    matchups_path = os.path.join(path, INPUT_FILE_NAME)

    # Load upcoming matchups from CSV if present
    if os.path.exists(matchups_path):
        df_matchups = pd.read_csv(matchups_path)
        logger.info(f"Loaded {len(df_matchups)} upcoming matchups from CSV")
    else:
        # Fallback to single matchup from .env/constants defaults
        df_matchups = pd.DataFrame([{
            "Week": WEEK_NUMBER,
            "Home Team": HOME_TEAM,
            "Away Team": AWAY_TEAM,
            "Game Date": GAME_DATE
        }])
        logger.warning("No matchup CSV found — using single matchup from .env/constants defaults")

    printed_weeks = set()       # Track weeks already printed to avoid duplicate model metrics
    week_model_cache = {}       # Cache trained models per week to avoid retraining
    results = []                # Store final output rows

    for _, row in df_matchups.iterrows():
        week = row["Week"]
        home_team = row["Home Team"]
        away_team = row["Away Team"]
        game_date = row["Game Date"]
        training_week = get_training_week(week)  # Determine training week for each matchup

        # Train model only once per week and cache it
        if week not in week_model_cache:
            try:
                df_conversions = pd.read_csv(os.path.join(path, CONVERSIONS_FILE.format(week=training_week, year=YEAR_ABBR)))
                df_offense = pd.read_csv(os.path.join(path, OFFENSE_FILE.format(week=training_week, year=YEAR_ABBR)))
                df_conversions_against = pd.read_csv(os.path.join(path, CONV_AGAINST_FILE.format(week=WEEK_NUMBER, year=YEAR_ABBR)))
                df_defense = pd.read_csv(os.path.join(path, DEFENSE_FILE.format(week=WEEK_NUMBER, year=YEAR_ABBR)))
            except FileNotFoundError as e:
                logger.warning(f"Missing data file for week {training_week} or {WEEK_NUMBER}: {e}")
                continue

            # Merge team stat datasets
            df = pd.merge(df_offense, df_conversions, on="Tm")
            df = pd.merge(df, df_conversions_against, on="Tm")
            df = pd.merge(df, df_defense, on="Tm")

            # Create engineered features
            df["PPG"] = df["PF"] / df["G"]
            df["Tot_1stD/G"] = df["Tot_1stD"] / df["G"]
            df["Avg_RZTD"] = df["RZTD_x"] / df["G"]  # this field is part of the Databricks feature set

            # Enhanced feature set with validation
            features = DEFAULT_FEATURES
            
            # Validate all features exist
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                logger.error(f"Missing required features: {missing_features}")
                continue

            # Databricks feature set (alternate) - uncomment to use
            # features = DATABRICKS_FEATURES

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
                logger.info(f"Training for week {week} data")
                logger.info("Mean Absolute Error: %.3f", mean_absolute_error(y_test, y_pred))
                logger.info("R² Score: %.3f", r2_score(y_test, y_pred))
                printed_weeks.add(week)

            # Cache trained model
            week_model_cache[week] = (model, df, features)
        else:
            model, df, features = week_model_cache[week]

        # Verify teams are in dataset
        if home_team not in df["Tm"].values or away_team not in df["Tm"].values:
            logger.warning(f"Skipping invalid matchup: {home_team} vs {away_team}")
            continue

        # Extract features for prediction
        ht_stats = df.loc[df["Tm"] == home_team, features]
        at_stats = df.loc[df["Tm"] == away_team, features]
        if ht_stats.empty or at_stats.empty:
            logger.warning(f"Missing stats for {home_team} or {away_team}")
            continue

        # Predict scores (home team gets home field advantage)
        ht_pred = round(model.predict(ht_stats)[0]) + HOME_FIELD_ADVANTAGE
        at_pred = round(model.predict(at_stats)[0])

        # Apply adjustments
        ht_adj, at_adj, wt_adj = 0, 0, 0

        if ENABLE_INJURY_ADJUSTMENTS and injuries_df is not None:
            ht_adj, at_adj = get_injuries_adjustment(
                injuries_df, home_team, away_team, team_abbreviations, qb_tiers, team_qbs
            )

        if ENABLE_WEATHER_ADJUSTMENTS:
            wt_adj = get_weather_adjustment(
                stad, home_team, game_date, weather_tiers
            )

        if VERBOSE_ADJUSTMENTS:
            logger.info(f"Adjustments - Injury: {ht_adj}/{at_adj}, Weather: {wt_adj}")

        ht_pred += ht_adj + wt_adj
        at_pred += at_adj + wt_adj

        # Determine outcome and save result
        diff = ht_pred - at_pred
        winner = home_team if diff > 0 else away_team
        result = "Tie" if diff == 0 else f"{winner} win by {abs(diff)}"
        total = ht_pred + at_pred

        results.append([week, home_team, ht_pred, away_team, at_pred, result, total])

    if results:
        df_results = pd.DataFrame(results, columns=[
            "Week", "Home Team", "Home Score", "Away Team", "Away Score", "Result", "Over/Under"
        ])

        # Define outputs
        output_files = [(OUTPUT_FILE_NAME, df_results)]

        # Optionally run Upsets Agent and add to outputs
        if ENABLE_UPSETS_AGENT:
            df_flagged = run_upsets_agent(df_results, path)
            output_files.append((FLAGGED_OUTPUT_FILE_NAME, df_flagged))

        # Handle writing based on environment
        if not running_in_lambda():
            for filename, df_out in output_files:
                output_path = os.path.join(path, filename)
                df_out.to_csv(output_path, index=False)
                logger.info(f"Saved output to {output_path}")
        else:
            for filename, _ in output_files:
                logger.info(f"{filename} generated successfully in Lambda environment.")

        return df_flagged if ENABLE_UPSETS_AGENT else df_results

    # No predictions made
    return pd.DataFrame()

if __name__ == "__main__":
    run_predictions()
