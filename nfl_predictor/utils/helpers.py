import os
import pandas as pd
import logging
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

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
        return int(week_value)
    except ValueError:
        return 18  # For postseason (e.g., "WildCard", "SuperBowl", etc.)

def get_injuries_adjustment(
    injuries_df: pd.DataFrame, 
    home_team: str, 
    away_team: str, 
    team_abbreviations: Dict[str, str], 
    qb_tiers: Dict[str, int], 
    team_qbs: Dict[str, Any]
) -> Tuple[int, int]:
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
    relevant_columns = ["Player", "Pos", "Status", "Injury Comment"]
    team_injuries = {
        team: group[relevant_columns].to_dict(orient="records")
        for team, group in injuries_df.groupby("Tm")
    }

    def qb_adjust(team):
        abbr = team_abbreviations.get(team)
        if not abbr:
            return 0

        team_qb_list = team_injuries.get(abbr, [])
        qb_name, tier = team_qbs.get(team, [None, "average"])
        
        for player in team_qb_list:
            if (player["Pos"] == "QB" and 
                player.get("Status", "").lower() in ["questionable", "doubtful", "out"] and
                player["Player"] == qb_name):
                return qb_tiers.get(tier, 0)
        return 0

    return qb_adjust(home_team), qb_adjust(away_team)

def get_weather_adjustment(
    stad: Any, 
    team_name: str, 
    game_date: str, 
    weather_tiers: Dict[str, int]
) -> int:
    """
    Calculate weather-based score adjustment for a given team and game date.

    Weather adjustments are skipped for indoor or retractable-roof stadiums.
    For outdoor games, temperature, wind, rain, and snow are evaluated based on 
    hourly forecast data near presumed kickoff time.
    
    Args:
        stad: NFLStadiums object for weather and stadium data
        team_name: Name of the team (for stadium lookup)
        game_date: Game date in YYYY-MM-DD format
        weather_tiers: Dictionary mapping weather conditions to adjustment values
        
    Returns:
        Weather adjustment points (positive/negative integer)
    """
    try:
        # Skip if indoor or retractable stadium
        stadium_info = stad.get_stadium_by_team(team_name)
        roof_type = stadium_info.get("roofType", "").lower()
        if roof_type in ["fixed", "retractable"]:
            logger.debug(f"{team_name} play in a {roof_type} stadium — skipping weather adjustment.")
            return 0

        forecast = stad.get_weather_forecast_for_stadium(team_name, game_date)
        adjustment = 0

        # Extract hourly forecast data
        hourly = forecast.get("hourly", {})
        hourly_times = hourly.get("time", [])

        # Determine kickoff time based on day of week
        dt_obj = datetime.strptime(game_date, "%Y-%m-%d")
        weekday = dt_obj.weekday()  # Monday = 0, Sunday = 6

        # If it's a weekend game (Saturday or Sunday), assume 1 PM game and use 12 PM forecast
        if weekday in (5, 6):
            kickoff_time = "13:00"  # 1 PM game
            forecast_hour = "12:00"  # Use 12 PM forecast (works for 4 PM games too)
        else:
            kickoff_time = "20:15"  # 8 PM game
            forecast_hour = "19:00"  # Use 7 PM forecast

        # Try to locate forecast for the set forecast hour
        game_hour = f"{game_date}T{forecast_hour}"
        idx = hourly_times.index(game_hour) if game_hour in hourly_times else (
            hourly_times.index(f"{game_date}T12:00") if f"{game_date}T12:00" in hourly_times else None
        )

        if idx is not None:
            temp = hourly.get("temperature_2m", [None])[idx]
            wind = hourly.get("wind_speed_10m", [None])[idx]
            rain = hourly.get("rain", [None])[idx]
            snow = hourly.get("snowfall", [None])[idx]

            # Temperature
            if temp is not None:
                if temp > 85:
                    adjustment += weather_tiers.get("temperature_above_85", 0)
                elif 55 <= temp <= 85:
                    adjustment += weather_tiers.get("temperature_55_85", 0)
                elif 32 <= temp < 55:
                    adjustment += weather_tiers.get("temperature_32_54", 0)
                else:
                    adjustment += weather_tiers.get("temperature_below_32", 0)

            # Wind
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
            if snow and snow > 0:
                if snow > 1.0:
                    adjustment += weather_tiers.get("heavy_snow", 0)
                elif snow > 0.3:
                    adjustment += weather_tiers.get("moderate_snow", 0)
                else:
                    adjustment += weather_tiers.get("light_snow", 0)
            elif rain and rain > 0:
                if rain > 1.0:
                    adjustment += weather_tiers.get("heavy_rain", 0)
                elif rain > 0.3:
                    adjustment += weather_tiers.get("moderate_rain", 0)
                else:
                    adjustment += weather_tiers.get("light_rain", 0)

        return adjustment

    except Exception as e:
        logger.warning(f"Weather adjustment failed: {e}")
        return 0
