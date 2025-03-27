import os
import pandas as pd

# Detect if running in AWS Lambda
def running_in_lambda():
    return os.getenv("AWS_EXECUTION_ENV") is not None

# Determine which week's data to use for training
def get_training_week(week_value):
    try:
        return int(week_value)
    except ValueError:
        return 18  # For postseason (e.g., "WildCard", "SuperBowl", etc.)

# Injury adjustment helper
def get_injuries_adjustment(file_path, home_team, away_team, team_abbreviations, qb_tiers, team_qbs):
    df_injuries = pd.read_csv(file_path)
    df_injuries = df_injuries.dropna(subset=["Status"])
    relevant_columns = ["Player", "Pos", "Status", "Injury Comment"]
    team_injuries = {
        team: group[relevant_columns].to_dict(orient="records")
        for team, group in df_injuries.groupby("Tm")
    }

    def qb_adjust(team):
        abbr = team_abbreviations.get(team)
        if abbr and any(p["Pos"] == "QB" for p in team_injuries.get(abbr, [])):
            return qb_tiers.get(team_qbs.get(team, [None, "average"])[1], 0)
        return 0

    return qb_adjust(home_team), qb_adjust(away_team)

# Weather adjustment helper
def get_weather_adjustment(stad, team_name, game_date, weather_tiers):
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
