import os
import pytest
import yaml
from unittest.mock import patch
from nfl_predictor.utils.helpers import get_weather_adjustment

# Load weather_tiers from YAML config
base_dir = os.path.dirname(os.path.dirname(__file__))
yaml_path = os.path.join(base_dir, "nfl_predictor", "data", "nfl_properties_test.yaml")

with open(yaml_path, "r") as f:
    config = yaml.safe_load(f)

weather_tiers = config["weather_tiers"]

# Dummy stadium object to simulate get_weather_forecast_for_stadium
class DummyStadium:
    def __init__(self, hourly_data, roof_type="open"):
        self.hourly_data = hourly_data
        self.roof_type = roof_type

    def get_weather_forecast_for_stadium(self, team_name, game_date):
        # Simulate structure returned by NFLStadiums API with forecast for the game date
        return {
            "hourly": self.hourly_data
        }
    
    def get_stadium_by_team(self, team_name):
        # Simulate indoor or outdoor stadium by adjusting roof type
        return {
            "roofType": self.roof_type  # "fixed" for indoor, "open" for outdoor
        }

# Forecast data and adjustments based on the provided example
@pytest.mark.parametrize("hourly, roof_type, expected_adjustment, home_team, game_date", [
    # Case 1: Kansas City Chiefs @ Houston Texans on 2025-01-18 (Weather Adjustment -6)
    (
        {
            "time": ['2025-01-18T12:00'],
            "temperature_2m": [18.7],
            "wind_speed_10m": [16.5],
            "rain": [0],
            "snowfall": [0]
        },
        "open",  # Outdoor Stadium
        -6,  # Weather adjustment
        "Kansas City Chiefs", "2025-01-18"
    ),
    
    # Case 2: Washington Commanders @ Detroit Lions on 2025-01-18 (Indoor Stadium, No Weather Adjustment)
    (
        {
            "time": ['2025-01-18T12:00'],
            "temperature_2m": [33.9],
            "wind_speed_10m": [10.6],
            "rain": [0],
            "snowfall": [0]
        },
        "fixed",  # Indoor Stadium
        0,  # No weather adjustment
        "Detroit Lions", "2025-01-18"
    ),
    
    # Case 3: Los Angeles Rams @ Philadelphia Eagles on 2025-01-19 (No Weather Adjustment)
    (
        {
            "time": ['2025-01-19T12:00'],
            "temperature_2m": [37.1],
            "wind_speed_10m": [9.5],
            "rain": [0],
            "snowfall": [0]
        },
        "open",  # Outdoor Stadium
        0,  # No weather adjustment
        "Philadelphia Eagles", "2025-01-19"
    ),
    
    # Case 4: Baltimore Ravens @ Buffalo Bills on 2025-01-19 (Weather Adjustment -2)
    (
        {
            "time": ['2025-01-19T12:00'],
            "temperature_2m": [21.1],
            "wind_speed_10m": [9.3],
            "rain": [0],
            "snowfall": [0]
        },
        "open",  # Outdoor Stadium
        -2,  # Weather adjustment
        "Buffalo Bills", "2025-01-19"
    ),
])
def test_weather_adjustment_hourly_forecast(hourly, roof_type, expected_adjustment, home_team, game_date):
    # Create a DummyStadium object with the given hourly data and roof type
    dummy_stadium = DummyStadium(hourly, roof_type)
    
    # Call the function to get the weather adjustment
    result = get_weather_adjustment(dummy_stadium, home_team, game_date, weather_tiers)

    # Check if the result matches the expected adjustment
    assert isinstance(result, int)
    assert result == expected_adjustment

def test_weather_adjustment_handles_exceptions_gracefully():
    # Case where the stadium API throws an error
    class BrokenStadium:
        def get_weather_forecast_for_stadium(self, team_name, game_date):
            raise RuntimeError("Simulated failure")

    result = get_weather_adjustment(BrokenStadium(), "Team", "2025-01-05", weather_tiers)
    assert result == 0
