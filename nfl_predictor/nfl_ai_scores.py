from dotenv import load_dotenv
# Load .env values
load_dotenv()

import os
import logging
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from utils.helpers import (running_in_lambda, get_training_week, get_injuries_adjustment, get_weather_adjustment)
from utils.env_setup import configure_nfl_stadiums_resource_dir
from nfl_stadiums import NFLStadiums

# Setup logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)

# Monkey-patch and get resource dir for nfl_stadiums
resource_dir = configure_nfl_stadiums_resource_dir()

# Instantiate stadiums object
stad = NFLStadiums()
if running_in_lambda():
    logger.info(f"Using NFL stadium resource dir: {stad._resources_dir}")

# Path to local data files
path = os.path.join(os.path.dirname(__file__), "data")

# Control optional logic via .env
VERBOSE_ADJUSTMENTS = os.getenv("VERBOSE_ADJUSTMENTS", "False") == "True"
ENABLE_INJURY_ADJUSTMENTS = os.getenv("ENABLE_INJURY_ADJUSTMENTS", "False") == "True"
ENABLE_WEATHER_ADJUSTMENTS = os.getenv("ENABLE_WEATHER_ADJUSTMENTS", "False") == "True"

# Main logic
def run_predictions(
    week_number=None,
    year_abbr=None,
    game_date=None,
    home_team=None,
    away_team=None
):
    # Load .env values if not passed in
    week_number = int(week_number or os.getenv("WEEK_NUMBER", 18))
    year_abbr = int(year_abbr or os.getenv("YEAR_ABBR", 24))
    game_date = game_date or os.getenv("GAME_DATE", "2025-02-09")
    home_team = home_team or os.getenv("HOME_TEAM", "Philadelphia Eagles")
    away_team = away_team or os.getenv("AWAY_TEAM", "Kansas City Chiefs")

    # Load team, QB, and weather properties
    with open(os.path.join(path, "nfl_properties_test.yaml"), "r") as file:
        nfl_properties = yaml.safe_load(file)

    team_abbreviations = nfl_properties["team_abbreviations"]
    qb_tiers = nfl_properties["qb_tiers"]
    team_qbs = nfl_properties["team_qbs"]
    weather_tiers = nfl_properties["weather_tiers"]

    # Injury data path
    inj_file = os.path.join(path, "nfl_injuries_test.csv")
    results = []

    # Matchups CSV path
    matchups_path = os.path.join(path, "upcoming_matchups_test.csv")

    # Load upcoming matchups from CSV if present
    if os.path.exists(matchups_path):
        df_matchups = pd.read_csv(matchups_path)
        logger.info(f"Loaded {len(df_matchups)} upcoming matchups from CSV")
    else:
        # Fallback to single matchup from env/default
        df_matchups = pd.DataFrame([{
            "Week": week_number,
            "Home Team": home_team,
            "Away Team": away_team,
            "Game Date": game_date
        }])
        logger.warning("No matchup CSV found — using single matchup from .env or defaults")

    printed_weeks = set()  # Track weeks already printed to avoid duplicate model metrics
    week_model_cache = {}  # Cache trained models per week to avoid retraining

    for _, row in df_matchups.iterrows():
        week = row["Week"]
        training_week = get_training_week(week)  # Determine training week for each matchup

        # Train model only once per week and cache it
        if week not in week_model_cache:
            try:
                df_conversions = pd.read_csv(f"{path}/nfl_conversions_thru_week_{training_week}_{year_abbr}.csv")
                df_offense = pd.read_csv(f"{path}/nfl_team_offense_thru_week_{training_week}_{year_abbr}.csv")
                df_conversions_against = pd.read_csv(f"{path}/nfl_conversions_against_thru_week_{week_number}_{year_abbr}.csv")
                df_defense = pd.read_csv(f"{path}/nfl_team_defense_thru_week_{week_number}_{year_abbr}.csv")
            except FileNotFoundError as e:
                logger.warning(f"Missing data file for week {training_week} or {week_number}: {e}")
                return {"error": f"Missing data file for week {training_week} or {week_number}: {str(e)}"}

            df = pd.merge(df_offense, df_conversions, on="Tm")
            df = pd.merge(df, df_conversions_against, on="Tm")
            df = pd.merge(df, df_defense, on="Tm")
            df["PPG"] = df["PF"] / df["G"]
            df["Tot_1stD/G"] = df["Tot_1stD"] / df["G"]
            df["Avg_RZTD"] = df["RZTD_x"] / df["G"]  # this field is part of the Databricks feature set

            features = ["Sc%_x", "Tot_1stD/G", "Y/P_x", "RZPct_x", "TO%_x", "Sc%_y"]  # Python feature set
            # features = ["Y/P_x", "Sc%_x", "Tot_1stD/G", "Avg_RZTD"]  # Databricks feature set
            X = df[features]
            y = df["PPG"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Print the results only for the first match of the week
            if week not in printed_weeks:
                y_pred = model.predict(X_test)
                logger.info(f"Training for week {week} data")
                logger.info("Mean Absolute Error: %.3f", mean_absolute_error(y_test, y_pred))
                logger.info("R² Score: %.3f", r2_score(y_test, y_pred))
                printed_weeks.add(week)

            week_model_cache[week] = (model, df, features)
        else:
            model, df, features = week_model_cache[week]

        home_team = row["Home Team"]
        away_team = row["Away Team"]
        game_date = row["Game Date"]

        if home_team not in df["Tm"].values or away_team not in df["Tm"].values:
            logger.warning(f"Skipping invalid matchup: {home_team} vs {away_team}")
            return {"error": f"Skipping invalid matchup: {home_team} vs {away_team}"}

        ht_stats = df.loc[df["Tm"] == home_team, features]
        at_stats = df.loc[df["Tm"] == away_team, features]

        # Verify that both teams have full stats available before predicting
        if ht_stats.empty or at_stats.empty:
            logger.warning(f"Missing stats for {home_team} or {away_team}")
            return {"error": f"Missing stats for {home_team} or {away_team}"}

        ht_pred = round(model.predict(ht_stats)[0]) + 1  # Predict home team score and apply home-field advantage
        at_pred = round(model.predict(at_stats)[0])

        # Apply adjustments if enabled
        ht_adj, at_adj = (0, 0)
        wt_adj = 0

        if ENABLE_INJURY_ADJUSTMENTS:
            ht_adj, at_adj = get_injuries_adjustment(
                inj_file, home_team, away_team, team_abbreviations, qb_tiers, team_qbs
            )

        if ENABLE_WEATHER_ADJUSTMENTS:
            wt_adj = get_weather_adjustment(
                stad, home_team, game_date, weather_tiers
            )

        if VERBOSE_ADJUSTMENTS:
            logger.info(f"Adjustments for {away_team} @ {home_team} on {game_date}")
            logger.info(f"Injury Adjustment - {home_team}: {ht_adj}, {away_team}: {at_adj}")
            logger.info(f"Weather Adjustment (applied to both): {wt_adj}\n")

        ht_pred += ht_adj + wt_adj
        at_pred += at_adj + wt_adj

        diff = ht_pred - at_pred
        winner = home_team if diff > 0 else away_team
        result = "Tie" if diff == 0 else f"{winner} win by {abs(diff)}"
        total = ht_pred + at_pred

        results.append([week, home_team, ht_pred, away_team, at_pred, result, total])

    if results:
        df_results = pd.DataFrame(results, columns=["Week", "Home Team", "Home Score", "Away Team", "Away Score", "Result", "Over/Under"])

        # Only write to CSV if not in Lambda
        if not running_in_lambda():
            output_path = os.path.join(path, "predicted_matchups_test.csv")
            df_results.to_csv(output_path, index=False)
            logger.info(f"Matchups saved to {output_path}")
        else:
            logger.info("Skipping CSV write — running in AWS Lambda")
            logger.info("\nPredicted Matchups Results:\n%s", df_results.to_string(index=False))
            # Return structured output for API use
            return df_results.iloc[0].to_dict()

    return {"error": "No predictions were made"}

if __name__ == "__main__":
    run_predictions()
