"""
NFL Score Predictor Configuration Constants

This file contains all default configuration values for the NFL predictor system.
Values can be optionally overridden via environment variables in .env file.

Only OPENAI_API_KEY is required in .env - all other values have sensible defaults here.
"""

# Base filenames
INPUT_FILE_NAME = "upcoming_matchups_test.csv"
INJURIES_FILE_NAME = "nfl_injuries_test.csv"
PROPERTIES_FILE_NAME = "nfl_properties_test.yaml"
STANDINGS_FILE_NAME = "standings_test.csv"
OUTPUT_FILE_NAME = "predicted_matchups_test.csv"
FLAGGED_OUTPUT_FILE_NAME = "predicted_matchups_flagged_test.csv"

# Dynamic file templates for stats
CONVERSIONS_FILE = "nfl_conversions_thru_week_{week}_{year}.csv"
OFFENSE_FILE = "nfl_team_offense_thru_week_{week}_{year}.csv"
DEFENSE_FILE = "nfl_team_defense_thru_week_{week}_{year}.csv"
CONV_AGAINST_FILE = "nfl_conversions_against_thru_week_{week}_{year}.csv"

# Model configuration constants
DEFAULT_FEATURES = ["Sc%_x", "Tot_1stD/G", "Y/P_x", "RZPct_x", "TO%_x", "Sc%_y"]
DATABRICKS_FEATURES = ["Y/P_x", "Sc%_x", "Tot_1stD/G", "Avg_RZTD"]
HOME_FIELD_ADVANTAGE = 1
TRAIN_TEST_SPLIT_RATIO = 0.33
RANDOM_STATE = 42

# AWS configuration
DEFAULT_INPUT_BUCKET = "nfl-score-predictor-test-input"
DEFAULT_OUTPUT_BUCKET = "nfl-score-predictor-test-output"

# Prediction defaults
DEFAULT_WEEK_NUMBER = 18
DEFAULT_YEAR_ABBR = 24

# Feature flags defaults
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_VERBOSE_ADJUSTMENTS = False
DEFAULT_INJURY_ADJUSTMENTS = False
DEFAULT_WEATHER_ADJUSTMENTS = False
DEFAULT_UPSETS_AGENT = True
