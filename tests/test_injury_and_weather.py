"""
Concise tests to validate injury and weather adjustments work together.
Tests the combined functionality for production readiness.
"""
import os
import pandas as pd
import pytest
import yaml
from unittest.mock import MagicMock
from nfl_predictor.utils.helpers import get_injuries_adjustment, get_weather_adjustment


class TestInjuryAndWeather:
    """Test combined injury and weather adjustment functionality."""
    
    def setup_method(self):
        """Load configuration for tests."""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        yaml_path = os.path.join(base_dir, "nfl_predictor", "data", "nfl_properties_test.yaml")
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.team_abbreviations = config["team_abbreviations"]
        self.team_qbs = config["team_qbs"]
        self.qb_tiers = config["qb_tiers"]
        self.weather_tiers = config["weather_tiers"]

    def test_injury_adjustment_validation(self):
        """Test that injury adjustments work correctly based on actual code."""
        # Test Josh Allen (Tier 1 QB) injury
        injury_df = pd.DataFrame([{
            'Player': 'Josh Allen',
            'Tm': 'BUF',  # Buffalo Bills abbreviation
            'Pos': 'QB',
            'Status': 'Questionable',
            'Injury Comment': 'Shoulder injury'
        }])
        
        home_adj, away_adj = get_injuries_adjustment(
            injury_df, 
            home_team="Buffalo Bills", 
            away_team="Miami Dolphins",
            team_abbreviations=self.team_abbreviations,
            qb_tiers=self.qb_tiers,
            team_qbs=self.team_qbs
        )
        
        # Josh Allen is Tier 1 QB, should get -6 penalty when questionable
        assert home_adj == -6, f"Expected -6 for Josh Allen injury, got {home_adj}"
        assert away_adj == 0, f"Expected 0 for opponent, got {away_adj}"

    def test_weather_adjustment_validation(self):
        """Test that weather adjustments work correctly based on actual code."""
        # Create mock stadium
        mock_stadium = MagicMock()
        
        # Test cold, windy conditions (like Kansas City scenario)
        mock_stadium.get_stadium_by_team.return_value = {"roofType": "open"}
        mock_stadium.get_weather_forecast_for_stadium.return_value = {
            "hourly": {
                "time": ['2025-01-18T12:00'],
                "temperature_2m": [18.7],  # Below 32°F = temperature_below_32 penalty
                "wind_speed_10m": [16.5],  # 16-20 mph = wind_16_20_mph penalty  
                "rain": [0],
                "snowfall": [0]
            }
        }
        
        weather_adj = get_weather_adjustment(
            mock_stadium, "Kansas City Chiefs", "2025-01-18", self.weather_tiers
        )
        
        # Should get penalties for cold temp (-4) + moderate wind (-2) = -6
        expected = self.weather_tiers["temperature_below_32"] + self.weather_tiers["wind_16_20_mph"]
        assert weather_adj == expected, f"Expected {expected} for cold/windy weather, got {weather_adj}"

    def test_combined_adjustments_realistic_scenario(self):
        """Test injury and weather adjustments working together in realistic scenario."""
        # Scenario: Buffalo Bills vs Kansas City Chiefs playoff game
        # Josh Allen questionable + cold weather at Arrowhead
        
        # 1. Injury adjustment - Josh Allen questionable
        injury_df = pd.DataFrame([{
            'Player': 'Josh Allen',
            'Tm': 'BUF',
            'Pos': 'QB', 
            'Status': 'Questionable',
            'Injury Comment': 'Elbow injury'
        }])
        
        injury_home, injury_away = get_injuries_adjustment(
            injury_df,
            home_team="Kansas City Chiefs",
            away_team="Buffalo Bills", 
            team_abbreviations=self.team_abbreviations,
            qb_tiers=self.qb_tiers,
            team_qbs=self.team_qbs
        )
        
        # 2. Weather adjustment - cold conditions at Arrowhead  
        mock_stadium = MagicMock()
        mock_stadium.get_stadium_by_team.return_value = {"roofType": "open"}
        mock_stadium.get_weather_forecast_for_stadium.return_value = {
            "hourly": {
                "time": ['2025-01-18T12:00'],
                "temperature_2m": [25.0],  # Cold but not extreme
                "wind_speed_10m": [12.0],   # Moderate wind
                "rain": [0],
                "snowfall": [0]
            }
        }
        
        weather_adj = get_weather_adjustment(
            mock_stadium, "Kansas City Chiefs", "2025-01-18", self.weather_tiers
        )
        
        # 3. Validate combined effect
        print(f"Injury adjustment - Home: {injury_home}, Away: {injury_away}")
        print(f"Weather adjustment: {weather_adj}")
        
        # Josh Allen injury affects away team (Buffalo Bills)
        assert injury_home == 0, "Home team (KC) should have no injury adjustment"
        assert injury_away == -6, "Away team (Buffalo) should have -6 for Josh Allen injury"
        
        # Weather affects home team's venue
        # 25°F is below 32, so gets temperature_below_32 penalty (-2)
        # 12 mph wind is in 11-15 mph range, gets wind_11_15_mph penalty (-2)
        expected_weather = self.weather_tiers["temperature_below_32"] + self.weather_tiers["wind_11_15_mph"]
        assert weather_adj == expected_weather, f"Expected {expected_weather} for cold/wind, got {weather_adj}"
        
        # Total adjustments for each team
        total_home_adj = injury_home + weather_adj  # KC gets weather penalty
        total_away_adj = injury_away  # Buffalo gets injury penalty, no weather (away team)
        
        print(f"Total adjustments - Home: {total_home_adj}, Away: {total_away_adj}")
        
        assert total_home_adj < 0, "Home team should have negative weather adjustment"
        assert total_away_adj < 0, "Away team should have negative injury adjustment"

    def test_indoor_stadium_no_weather_impact(self):
        """Test that indoor stadiums negate weather adjustments."""
        mock_stadium = MagicMock()
        mock_stadium.get_stadium_by_team.return_value = {"roofType": "fixed"}  # Indoor
        
        # Simulate terrible weather that would normally cause major penalties
        mock_stadium.get_weather_forecast_for_stadium.return_value = {
            "hourly": {
                "time": ['2025-01-18T12:00'],
                "temperature_2m": [5.0],   # Extremely cold
                "wind_speed_10m": [35.0],  # Very windy
                "rain": [2.0],             # Heavy rain
                "snowfall": [3.0]          # Heavy snow
            }
        }
        
        weather_adj = get_weather_adjustment(
            mock_stadium, "Detroit Lions", "2025-01-18", self.weather_tiers
        )
        
        assert weather_adj == 0, "Indoor stadium should negate all weather effects"

    def test_no_injuries_no_penalties(self):
        """Test that healthy QBs get no injury penalties."""
        # Empty injury report
        empty_df = pd.DataFrame()
        
        home_adj, away_adj = get_injuries_adjustment(
            empty_df,
            home_team="Kansas City Chiefs",
            away_team="Buffalo Bills",
            team_abbreviations=self.team_abbreviations,
            qb_tiers=self.qb_tiers,
            team_qbs=self.team_qbs
        )
        
        assert home_adj == 0, "No injuries should result in no penalties"
        assert away_adj == 0, "No injuries should result in no penalties"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
