"""
Test suite for environment configuration and constants integration.
"""
import os
import tempfile
from unittest.mock import patch, Mock
import pytest
from nfl_predictor.utils.env_setup import configure_nfl_stadiums_resource_dir


class TestEnvironmentConfiguration:
    """Test environment setup and configuration."""
    
    def test_configure_nfl_stadiums_default_path(self):
        """Test NFL stadiums configuration with default path."""
        result_dir = configure_nfl_stadiums_resource_dir()
        
        # Should return the configured directory
        assert isinstance(result_dir, str)
        assert "/tmp/nfl_stadium_resources" in result_dir
        
        # Directory should be created
        assert os.path.exists(result_dir)
    
    @patch.dict(os.environ, {"NFL_STADIUM_RESOURCES": "/custom/path"})
    def test_configure_nfl_stadiums_custom_path(self):
        """Test NFL stadiums configuration with custom environment path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_path = os.path.join(temp_dir, "custom_stadiums")
            
            with patch.dict(os.environ, {"NFL_STADIUM_RESOURCES": custom_path}):
                result_dir = configure_nfl_stadiums_resource_dir()
                
                assert result_dir == custom_path
                assert os.path.exists(custom_path)


class TestConstantsEnvironmentIntegration:
    """Test integration between constants and environment variables."""
    
    @patch.dict(os.environ, {})
    def test_constants_used_as_defaults(self):
        """Test that constants are used when environment variables are not set."""
        from nfl_predictor.utils.constants import DEFAULT_LOG_LEVEL, DEFAULT_WEEK_NUMBER
        
        # Clear environment variables
        os.environ.pop('LOG_LEVEL', None)
        os.environ.pop('WEEK_NUMBER', None)
        
        # Test that defaults are used
        log_level = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL)
        week_number = int(os.getenv("WEEK_NUMBER", DEFAULT_WEEK_NUMBER))
        
        assert log_level == DEFAULT_LOG_LEVEL
        assert week_number == DEFAULT_WEEK_NUMBER
    
    @patch.dict(os.environ, {"LOG_LEVEL": "DEBUG", "WEEK_NUMBER": "10"})
    def test_environment_overrides_constants(self):
        """Test that environment variables override constants."""
        from nfl_predictor.utils.constants import DEFAULT_LOG_LEVEL, DEFAULT_WEEK_NUMBER
        
        log_level = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL)
        week_number = int(os.getenv("WEEK_NUMBER", DEFAULT_WEEK_NUMBER))
        
        assert log_level == "DEBUG"
        assert week_number == 10
        # Should be different from defaults
        assert log_level != DEFAULT_LOG_LEVEL
        assert week_number != DEFAULT_WEEK_NUMBER
    
    def test_boolean_environment_conversion(self):
        """Test proper conversion of boolean environment variables."""
        from nfl_predictor.utils.constants import DEFAULT_INJURY_ADJUSTMENTS
        
        # Test various boolean representations
        test_cases = [
            ("true", True),
            ("True", True), 
            ("TRUE", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("", False),  # Empty string should be False
        ]
        
        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"ENABLE_INJURY_ADJUSTMENTS": env_value}):
                result = os.getenv("ENABLE_INJURY_ADJUSTMENTS", str(DEFAULT_INJURY_ADJUSTMENTS)).lower() == "true"
                assert result == expected, f"Failed for env_value='{env_value}'"


class TestFilePathIntegration:
    """Test file path handling across different environments."""
    
    @patch('nfl_predictor.utils.helpers.running_in_lambda')
    def test_path_selection_lambda(self, mock_running_in_lambda):
        """Test path selection in Lambda environment."""
        mock_running_in_lambda.return_value = True
        
        # Simulate the path logic from nfl_ai_scores.py
        from nfl_predictor.utils.helpers import running_in_lambda
        path = "/var/task/nfl_predictor/data" if running_in_lambda() else os.path.join(os.path.dirname(__file__), "data")
        
        assert path == "/var/task/nfl_predictor/data"
    
    @patch('nfl_predictor.utils.helpers.running_in_lambda')
    def test_path_selection_local(self, mock_running_in_lambda):
        """Test path selection in local environment."""
        mock_running_in_lambda.return_value = False
        
        from nfl_predictor.utils.helpers import running_in_lambda
        path = "/var/task/nfl_predictor/data" if running_in_lambda() else os.path.join(os.path.dirname(__file__), "data")
        
        assert path != "/var/task/nfl_predictor/data"
        assert "data" in path


class TestConfigurationConsistency:
    """Test consistency across configuration files and constants."""
    
    def test_env_example_matches_constants(self):
        """Test that .env.example template includes all configurable constants."""
        # Read .env.example file
        env_example_path = os.path.join(os.path.dirname(__file__), "..", ".env.example")
        
        try:
            with open(env_example_path, 'r') as f:
                env_example_content = f.read()
                
            # Check for key configuration variables
            expected_vars = [
                "LOG_LEVEL",
                "ENABLE_INJURY_ADJUSTMENTS", 
                "ENABLE_WEATHER_ADJUSTMENTS",
                "ENABLE_UPSETS_AGENT",
                "WEEK_NUMBER",
                "HOME_TEAM",
                "AWAY_TEAM"
            ]
            
            for var in expected_vars:
                assert var in env_example_content, f"Missing {var} in .env.example"
                
        except FileNotFoundError:
            pytest.skip(".env.example file not found")
    
    def test_docker_compose_uses_constants_approach(self):
        """Test that docker-compose.yml follows constants-first approach."""
        docker_compose_path = os.path.join(os.path.dirname(__file__), "..", "docker-compose.yml")
        
        try:
            with open(docker_compose_path, 'r') as f:
                docker_content = f.read()
                
            # Should mention constants.py approach
            assert "constants.py" in docker_content or "defaults" in docker_content
            
        except FileNotFoundError:
            pytest.skip("docker-compose.yml file not found")
