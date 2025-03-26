import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from nfl_stadiums import NFLStadiums

# Data sourced from Pro Football Reference: https://www.pro-football-reference.com
path = "/Users/whansen/Desktop/Data Science/nfl_stats/season_24_25"

# Load YAML file
with open("nfl_properties_test.yaml", "r") as file:
    nfl_properties = yaml.safe_load(file)

team_abbreviations = nfl_properties["team_abbreviations"]
qb_tiers = nfl_properties["qb_tiers"]
team_qbs = nfl_properties["team_qbs"]
weather_tiers = nfl_properties["weather_tiers"]

stad = NFLStadiums()

# Load Injury Data
def get_injuries_adjustment(file_path, home_team, away_team):
    df_injuries = pd.read_csv(file_path)
    df_injuries = df_injuries.dropna(subset=["Status"])
    relevant_columns = ["Player", "Pos", "Status", "Injury Comment"]
    team_injuries = {team: group[relevant_columns].to_dict(orient="records") for team, group in df_injuries.groupby("Tm")}

    home_team_adjust = 0
    away_team_adjust = 0
    home_team_abbr = team_abbreviations.get(home_team, None)
    away_team_abbr = team_abbreviations.get(away_team, None)
    if not home_team_abbr and not away_team_abbr:
        return home_team_adjust, away_team_adjust

    print(f"Injury Report: {team_injuries}")

    if home_team_abbr and any(player["Pos"] == "QB" for player in team_injuries[home_team_abbr]):
        home_team_adjust += qb_tiers.get(team_qbs.get(home_team, [None, "average"])[1], 0)
    if away_team_abbr and any(player["Pos"] == "QB" for player in team_injuries[away_team_abbr]):
        away_team_adjust += qb_tiers.get(team_qbs.get(away_team, [None, "average"])[1], 0)

    return home_team_adjust, away_team_adjust

# Load Weather Data
def get_weather_adjustment(team_name, game_date):
    try:
        forecast = stad.get_weather_forecast_for_stadium(team_name, game_date)
        if forecast.get("roof", "").lower() in ["indoor", "dome"]:
            return weather_tiers.get("indoor", 0)

        adjustment = 0

        # Temperature (already in Fahrenheit)
        temp = forecast.get("temperature")
        if temp is not None:
            if temp > 85:
                adjustment += weather_tiers.get("temperature_above_85", 0)
            elif 55 <= temp <= 85:
                adjustment += weather_tiers.get("temperature_55_85", 0)
            elif 32 <= temp < 55:
                adjustment += weather_tiers.get("temperature_32_54", 0)
            else:
                adjustment += weather_tiers.get("temperature_below_32", 0)

        # Wind Speed (already in mph)
        wind = forecast.get("wind_speed")
        if wind is not None:
            if wind <= 10:
                adjustment += weather_tiers.get("wind_0_10_mph", 0)
            elif 11 <= wind <= 15:
                adjustment += weather_tiers.get("wind_11_15_mph", 0)
            elif 16 <= wind <= 20:
                adjustment += weather_tiers.get("wind_16_20_mph", 0)
            else:
                adjustment += weather_tiers.get("wind_over_20_mph", 0)

        # Precipitation
        precip_type = forecast.get("precipitation_type", "none").lower()
        precip_intensity = forecast.get("precipitation_intensity", "none").lower()
        precip_key = f"{precip_intensity}_{precip_type}"
        adjustment += weather_tiers.get(precip_key, 0)

        return adjustment
    except Exception as e:
        print(f"Weather forecast failed for {team_name} on {game_date}: {e}")
        return 0

# Load offense datasets
week_number = input("Enter the week number: ").strip()
if not week_number.isdigit():
    raise ValueError("Invalid input! Please enter a numerical week number.")
conversions_file_path = f"{path}/nfl_conversions_thru_week_{week_number}_24.csv"
offense_file_path = f"{path}/nfl_team_offense_thru_week_{week_number}_24.csv"
try:
    df_conversions = pd.read_csv(conversions_file_path)
    df_offense = pd.read_csv(offense_file_path)
except FileNotFoundError:
    raise FileNotFoundError(f"One or both offense data files for week {week_number} are missing!")

# Load defense datasets
defense_conversions_file_path = f"{path}/nfl_conversions_against_thru_week_{week_number}_24.csv"
defense_file_path = f"{path}/nfl_team_defense_thru_week_{week_number}_24.csv"
try:
    df_conversions_against = pd.read_csv(defense_conversions_file_path)
    df_defense = pd.read_csv(defense_file_path)
except FileNotFoundError:
    raise FileNotFoundError(f"One or both defense data files for week {week_number} are missing!")

# Merge datasets on "Tm"
df_merged = pd.merge(df_offense, df_conversions, on="Tm", how="inner")
df_merged = pd.merge(df_merged, df_conversions_against, on="Tm", how="inner")
df_merged = pd.merge(df_merged, df_defense, on="Tm", how="inner")

# Feature Engineering
df_merged['PPG'] = df_merged['PF'] / df_merged['G']
df_merged['Tot_1stD/G'] = df_merged['Tot_1stD'] / df_merged['G']

# Selected Features
features = ["Sc%_x", "Tot_1stD/G", "Y/P_x", "RZPct_x", "TO%_x", "Sc%_y"]
X = df_merged[features]
y = df_merged["PPG"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Matchup predictions
matchup_results = []
num_games = input("Enter the number of games this week: ").strip()
if not num_games.isdigit():
    raise ValueError("Invalid input! Please enter a numerical number of games.")
num_games = int(num_games)
for _ in range(num_games):
    game_date = input("Enter Game Date for this matchup (YYYY-MM-DD): ").strip()
    home_team = input("Enter Home Team: ").strip()
    away_team = input("Enter Away Team: ").strip()

    if home_team not in df_merged['Tm'].values or away_team not in df_merged['Tm'].values:
        raise ValueError("One or both team names are invalid. Please check spelling.")

    home_team_stats = df_merged[df_merged['Tm'] == home_team][features]
    away_team_stats = df_merged[df_merged['Tm'] == away_team][features]

    home_team_pred = round(model.predict(home_team_stats)[0]) + 1
    away_team_pred = round(model.predict(away_team_stats)[0])

    # Adjust for injuries
    injury_file_path = f"{path}/nfl_injuries_test.csv"
    injury_home_adjust, injury_away_adjust = get_injuries_adjustment(injury_file_path, home_team, away_team)
    home_team_pred += injury_home_adjust
    away_team_pred += injury_away_adjust

    # Adjust for weather
    weather_adjust = get_weather_adjustment(home_team, game_date)
    home_team_pred += weather_adjust
    away_team_pred += weather_adjust

    # Result formatting
    point_diff = home_team_pred - away_team_pred
    winner = home_team if point_diff > 0 else away_team
    margin = abs(point_diff)
    result_text = "Tie" if margin == 0 else f"{winner} win by {margin}"
    total_points = home_team_pred + away_team_pred

    matchup_results.append([home_team, home_team_pred, away_team, away_team_pred, result_text, total_points])
    print(f"Predicted Score - {home_team}: {home_team_pred}, {away_team}: {away_team_pred}")

# Save to CSV
results_df = pd.DataFrame(matchup_results, columns=["Home Team", "Home Score", "Away Team", "Away Score", "Result", "Over/Under"])
results_df.to_csv(f"{path}/predicted_matchups_test.csv", index=False)
print("Matchups saved to predicted_matchups_test.csv")
