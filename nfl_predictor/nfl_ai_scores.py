import os
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from utils.helpers import (running_in_lambda, get_training_week, get_injuries_adjustment, get_weather_adjustment)
from utils.env_setup import configure_nfl_stadiums_resource_dir
from nfl_stadiums import NFLStadiums

# Load .env values
load_dotenv()

# Monkey-patch and get resource dir for nfl_stadiums
resource_dir = configure_nfl_stadiums_resource_dir()

# Instantiate stadiums object
stad = NFLStadiums()
if running_in_lambda():
    print(f"Using NFL stadium resource dir: {stad._resources_dir}")

# Path to local data files
path = os.path.join(os.path.dirname(__file__), "data")

# Main logic
def run_predictions():
    week_number = int(os.getenv("WEEK_NUMBER", 18))
    year_abbr = int(os.getenv("YEAR_ABBR", 24))
    game_date = os.getenv("GAME_DATE", "2025-02-09")
    home_team = os.getenv("HOME_TEAM", "Philadelphia Eagles")
    away_team = os.getenv("AWAY_TEAM", "Kansas City Chiefs")

    with open(os.path.join(path, "nfl_properties_test.yaml"), "r") as file:
        nfl_properties = yaml.safe_load(file)

    team_abbreviations = nfl_properties["team_abbreviations"]
    qb_tiers = nfl_properties["qb_tiers"]
    team_qbs = nfl_properties["team_qbs"]
    weather_tiers = nfl_properties["weather_tiers"]

    # Load matchups from CSV or fallback to env vars
    matchups_path = os.path.join(path, "upcoming_matchups_test.csv")
    inj_file = os.path.join(path, "nfl_injuries_test.csv")
    results = []

    if os.path.exists(matchups_path):
        df_matchups = pd.read_csv(matchups_path)
    else:
        # Fallback to env vars
        df_matchups = pd.DataFrame([{
            "Week": week_number,
            "Home Team": home_team,
            "Away Team": away_team,
            "Game Date": game_date
        }])

    printed_weeks = set()
    for _, row in df_matchups.iterrows():
        week = row["Week"]
        training_week = get_training_week(week)  # Determine training week for each matchup

        # Load data based on training_week for this specific matchup
        df_conversions = pd.read_csv(f"{path}/nfl_conversions_thru_week_{training_week}_{year_abbr}.csv")
        df_offense = pd.read_csv(f"{path}/nfl_team_offense_thru_week_{training_week}_{year_abbr}.csv")
        df_conversions_against = pd.read_csv(f"{path}/nfl_conversions_against_thru_week_{week_number}_{year_abbr}.csv")  # Only 18 for now
        df_defense = pd.read_csv(f"{path}/nfl_team_defense_thru_week_{week_number}_{year_abbr}.csv")  # Only 18 for now

        df = pd.merge(df_offense, df_conversions, on="Tm")
        df = pd.merge(df, df_conversions_against, on="Tm")
        df = pd.merge(df, df_defense, on="Tm")
        df["PPG"] = df["PF"] / df["G"]
        df["Tot_1stD/G"] = df["Tot_1stD"] / df["G"]

        features = ["Sc%_x", "Tot_1stD/G", "Y/P_x", "RZPct_x", "TO%_x", "Sc%_y"]
        X = df[features]
        y = df["PPG"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Print the results only for the first match of the week
        if week not in printed_weeks:
            print(f"Training for week {week} data")
            print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
            print("R² Score:", r2_score(y_test, y_pred))
            printed_weeks.add(week)

        home_team = row["Home Team"]
        away_team = row["Away Team"]
        game_date = row["Game Date"]

        if home_team not in df["Tm"].values or away_team not in df["Tm"].values:
            print(f"Skipping invalid matchup: {home_team} vs {away_team}")
            continue  # Skip to next iteration

        ht_stats = df.loc[df["Tm"] == home_team, features]
        at_stats = df.loc[df["Tm"] == away_team, features]

        ht_pred = round(model.predict(ht_stats)[0]) + 1
        at_pred = round(model.predict(at_stats)[0])

        ht_adj, at_adj = get_injuries_adjustment(
            inj_file, home_team, away_team, team_abbreviations, qb_tiers, team_qbs
        )
        wt_adj = get_weather_adjustment(
            stad, home_team, game_date, weather_tiers
        )

        ht_pred += ht_adj + wt_adj
        at_pred += at_adj + wt_adj

        diff = ht_pred - at_pred
        winner = home_team if diff > 0 else away_team
        result = "Tie" if diff == 0 else f"{winner} win by {abs(diff)}"
        total = ht_pred + at_pred

        results.append([week, home_team, ht_pred, away_team, at_pred, result, total])
        # print(f"Week {week} - Predicted Score - {home_team}: {ht_pred}, {away_team}: {at_pred}")  # For debugging

    if results:
        df_results = pd.DataFrame(results, columns=["Week", "Home Team", "Home Score", "Away Team", "Away Score", "Result", "Over/Under"])

        # Only write to CSV if not in Lambda
        if not running_in_lambda():
            output_path = os.path.join(path, "predicted_matchups_test.csv")
            df_results.to_csv(output_path, index=False)
            print(f"Matchups saved to {output_path}")
        else:
            print("Skipping CSV write — running in AWS Lambda")
            print("\nPredicted Matchups Results:")
            print(df_results.to_string(index=False))

# Run locally if executed directly
if __name__ == "__main__":
    run_predictions()
