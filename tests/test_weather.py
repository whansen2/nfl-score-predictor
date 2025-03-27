import os
import pytest
import yaml
from nfl_predictor.utils.helpers import get_weather_adjustment

# Load weather_tiers from YAML config
base_dir = os.path.dirname(os.path.dirname(__file__))
yaml_path = os.path.join(base_dir, "nfl_predictor", "data", "nfl_properties_test.yaml")

with open(yaml_path, "r") as f:
    config = yaml.safe_load(f)

weather_tiers = config["weather_tiers"]

class DummyStadium:
    def __init__(self, forecast):
        self.forecast = forecast

    def get_weather_forecast_for_stadium(self, team_name, game_date):
        return self.forecast

@pytest.mark.parametrize("forecast,expected", [
    (
        {
            "roof": "open",
            "temperature": 30,
            "wind_speed": 12,
            "precipitation_type": "snow",
            "precipitation_intensity": "moderate"
        },
        -8  # -2 temp + -2 wind + -4 snow
    ),
    (
        {
            "roof": "indoor",
            "temperature": 72,
            "wind_speed": 5,
            "precipitation_type": "none",
            "precipitation_intensity": "none"
        },
        0
    ),
    (
        {
            "roof": "open",
            "temperature": 90,
            "wind_speed": 21,
            "precipitation_type": "rain",
            "precipitation_intensity": "moderate"
        },
        -10  # -1 temp + -6 wind + -3 rain
    ),
    (
        {
            "roof": "open",
            "temperature": 65,
            "wind_speed": 8,
            "precipitation_type": "none",
            "precipitation_intensity": "none"
        },
        0
    ),
    (
        {
            "roof": "open",
            "temperature": 50,
            "wind_speed": 16,
            "precipitation_type": "snow",
            "precipitation_intensity": "heavy"
        },
        -10  # 0 temp + -4 wind + -6 snow
    )
])

def test_weather_adjustment_varied_conditions(forecast, expected):
    dummy_stadium = DummyStadium(forecast)
    result = get_weather_adjustment(dummy_stadium, "Team", "2025-01-15", weather_tiers)
    assert isinstance(result, int)
    assert result == expected

def test_weather_adjustment_with_missing_fields():
    dummy_stadium = DummyStadium({
        "roof": "open",
        # temperature and wind_speed are missing
        "precipitation_type": "rain",
        "precipitation_intensity": "heavy"
    })

    result = get_weather_adjustment(dummy_stadium, "Team", "2025-01-15", weather_tiers)
    assert result == weather_tiers["heavy_rain"]

def test_weather_adjustment_handles_exceptions_gracefully():
    class BrokenStadium:
        def get_weather_forecast_for_stadium(self, team_name, game_date):
            raise RuntimeError("Simulated failure")

    result = get_weather_adjustment(BrokenStadium(), "Team", "2025-01-15", weather_tiers)
    assert result == 0
