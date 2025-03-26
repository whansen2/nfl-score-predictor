import os
import tempfile

# Patch the internal path BEFORE importing anything from the submodules
lambda_tmp_path = os.path.join(tempfile.gettempdir(), "nfl_stadium_resources")
import nfl_stadiums  # Import root first
nfl_stadiums.RESOURCE_DIR = lambda_tmp_path  # Override internal path manually

# Log for sanity check
print(f"[INFO] NFL_STADIUM_RESOURCES pt 1 set to: {nfl_stadiums.RESOURCE_DIR}")

# Now it's safe to import other things
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from dotenv import load_dotenv
load_dotenv()

from nfl_stadiums import NFLStadiums

# Log for sanity check
print(f"[INFO] NFL_STADIUM_RESOURCES pt 2 set to: {nfl_stadiums.RESOURCE_DIR}")

# Global constants
stad = NFLStadiums()

# Log for sanity check
print(f"[INFO] NFL_STADIUM_RESOURCES pt 3 set to: {nfl_stadiums.RESOURCE_DIR}")

path = os.path.join(os.path.dirname(__file__), "data")

# Helper: Injury adjustment
def get_injuries_adjustment(file_path, home_team, away_team, team_abbreviations, qb_tiers, team_qbs):
    df_injuries = pd.read_csv(file_path)
    df_injuries = df_injuries.dropna(subset=["Status"])
    relevant_columns = ["Player", "Pos", "Status", "Injury Comment"]
    team_injuries = {team: group[relevant_columns].to_dict(orient="records") for team, group in df_injuries.groupby("Tm")}

    home_team_adjust = 0
    away_team_adjust = 0
    home_abbr = team_abbreviations.get(home_team)
    away_abbr = team_abbreviations.get(away_team)

    if home_abbr and any(p["Pos"] == "QB" for p in team_injuries.get(home_abbr, [])):
        home_team_adjust += qb_tiers.get(team_qbs.get(home_team, [None, "average"])[1], 0)
    if away_abbr and any(p["Pos"] == "QB" for p in team_injuries.get(away_abbr, [])):
        away_team_adjust += qb_tiers.get(team_qbs.get(away_team, [None, "average"])[1], 0)

    return home_team_adjust, away_team_adjust

# Helper: Weather adjustment
def get_weather_adjustment(team_name, game_date, weather_tiers):
    try:
        forecast = stad.get_weather_forecast_for_stadium(team_name, game_date)
        if forecast.get("roof", "").lower() in ["indoor", "dome"]:
            return weather_tiers.get("indoor", 0)

        adjustment = 0
        temp = forecast.get("temperature")
        wind = forecast.get("wind_speed")
        precip_type = forecast.get("precipitation_type", "none").lower()
        precip_intensity = forecast.get("precipitation_intensity", "none").lower()

        if temp is not None:
            if temp > 85:
                adjustment += weather_tiers.get("temperature_above_85", 0)
            elif 55 <= temp <= 85:
                adjustment += weather_tiers.get("temperature_55_85", 0)
            elif 32 <= temp < 55:
                adjustment += weather_tiers.get("temperature_32_54", 0)
            else:
                adjustment += weather_tiers.get("temperature_below_32", 0)

        if wind is not None:
            if wind <= 10:
                adjustment += weather_tiers.get("wind_0_10_mph", 0)
            elif 11 <= wind <= 15:
                adjustment += weather_tiers.get("wind_11_15_mph", 0)
            elif 16 <= wind <= 20:
                adjustment += weather_tiers.get("wind_16_20_mph", 0)
            else:
                adjustment += weather_tiers.get("wind_over_20_mph", 0)

        adjustment += weather_tiers.get(f"{precip_intensity}_{precip_type}", 0)
        return adjustment
    except Exception as e:
        print(f"Weather error: {e}")
        return 0

# Main logic
def run_predictions():
    week_number = int(os.getenv("WEEK_NUMBER", 18))
    year_abbr = int(os.getenv("YEAR_ABBR", 24))
    num_games = int(os.getenv("NUM_GAMES", 1))
    game_date = os.getenv("GAME_DATE", "2025-02-09")
    home_team = os.getenv("HOME_TEAM", "Philadelphia Eagles")
    away_team = os.getenv("AWAY_TEAM", "Kansas City Chiefs")

    with open(os.path.join(path, "nfl_properties_test.yaml"), "r") as file:
        nfl_properties = yaml.safe_load(file)

    team_abbreviations = nfl_properties["team_abbreviations"]
    qb_tiers = nfl_properties["qb_tiers"]
    team_qbs = nfl_properties["team_qbs"]
    weather_tiers = nfl_properties["weather_tiers"]

    # Load data
    df_conversions = pd.read_csv(f"{path}/nfl_conversions_thru_week_{week_number}_{year_abbr}.csv")
    df_offense = pd.read_csv(f"{path}/nfl_team_offense_thru_week_{week_number}_{year_abbr}.csv")
    df_conversions_against = pd.read_csv(f"{path}/nfl_conversions_against_thru_week_{week_number}_{year_abbr}.csv")
    df_defense = pd.read_csv(f"{path}/nfl_team_defense_thru_week_{week_number}_{year_abbr}.csv")

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

    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))

    results = []
    for _ in range(num_games):
        if home_team not in df["Tm"].values or away_team not in df["Tm"].values:
            raise ValueError("Invalid team names.")

        ht_stats = df[df["Tm"] == home_team][features]
        at_stats = df[df["Tm"] == away_team][features]

        ht_pred = round(model.predict(ht_stats)[0]) + 1
        at_pred = round(model.predict(at_stats)[0])

        inj_file = os.path.join(path, "nfl_injuries_test.csv")
        ht_adj, at_adj = get_injuries_adjustment(inj_file, home_team, away_team, team_abbreviations, qb_tiers, team_qbs)
        wt_adj = get_weather_adjustment(home_team, game_date, weather_tiers)

        ht_pred += ht_adj + wt_adj
        at_pred += at_adj + wt_adj

        diff = ht_pred - at_pred
        winner = home_team if diff > 0 else away_team
        result = "Tie" if diff == 0 else f"{winner} win by {abs(diff)}"
        total = ht_pred + at_pred

        results.append([home_team, ht_pred, away_team, at_pred, result, total])
        print(f"Predicted Score - {home_team}: {ht_pred}, {away_team}: {at_pred}")

    df_results = pd.DataFrame(results, columns=["Home Team", "Home Score", "Away Team", "Away Score", "Result", "Over/Under"])
    df_results.to_csv(os.path.join(path, "predicted_matchups_test.csv"), index=False)
    print("Matchups saved to predicted_matchups_test.csv")
    return {"status": "ok", "result": results}

# Run locally if executed directly
if __name__ == "__main__":
    run_predictions()
