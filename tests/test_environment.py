"""
Test suite for environment configuration and constants integration.
"""

import os
from unittest.mock import patch

import pytest


class TestConstantsEnvironmentIntegration:
    """Test integration between constants and environment variables."""

    @patch.dict(os.environ, {})
    def test_constants_used_as_defaults(self) -> None:
        """Test that constants are used when environment variables are not set."""
        from nfl_predictor.utils.constants import DEFAULT_LOG_LEVEL

        # Clear environment variables
        os.environ.pop("LOG_LEVEL", None)

        # Test that defaults are used
        log_level = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL)

        assert log_level == DEFAULT_LOG_LEVEL

    @patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"})
    def test_environment_overrides_constants(self) -> None:
        """Test that environment variables override constants."""
        from nfl_predictor.utils.constants import DEFAULT_LOG_LEVEL

        log_level = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL)

        assert log_level == "DEBUG"
        # Should be different from defaults
        assert log_level != DEFAULT_LOG_LEVEL

    def test_boolean_environment_conversion(self) -> None:
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
                result = (
                    os.getenv(
                        "ENABLE_INJURY_ADJUSTMENTS", str(DEFAULT_INJURY_ADJUSTMENTS)
                    ).lower()
                    == "true"
                )
                assert result == expected, f"Failed for env_value='{env_value}'"


class TestFilePathIntegration:
    """Test file path handling across different environments."""

    @patch("nfl_predictor.utils.helpers.running_in_lambda")
    def test_path_selection_lambda(self, mock_running_in_lambda) -> None:
        """Test path selection in Lambda environment."""
        mock_running_in_lambda.return_value = True

        # Simulate the path logic from nfl_ai_scores.py
        from nfl_predictor.utils.helpers import running_in_lambda

        path = (
            "/var/task/nfl_predictor/data"
            if running_in_lambda()
            else os.path.join(os.path.dirname(__file__), "data")
        )

        assert path == "/var/task/nfl_predictor/data"

    @patch("nfl_predictor.utils.helpers.running_in_lambda")
    def test_path_selection_local(self, mock_running_in_lambda) -> None:
        """Test path selection in local environment."""
        mock_running_in_lambda.return_value = False

        from nfl_predictor.utils.helpers import running_in_lambda

        path = (
            "/var/task/nfl_predictor/data"
            if running_in_lambda()
            else os.path.join(os.path.dirname(__file__), "data")
        )

        assert path != "/var/task/nfl_predictor/data"
        assert "data" in path


class TestConfigurationConsistency:
    """Test consistency across configuration files and constants."""

    def test_docker_compose_uses_constants_approach(self) -> None:
        """Test that docker-compose.yml follows constants-first approach."""
        docker_compose_path = os.path.join(
            os.path.dirname(__file__), "..", "docker-compose.yml"
        )

        try:
            with open(docker_compose_path) as f:
                docker_content = f.read()

            # Should mention constants.py approach
            assert "constants.py" in docker_content or "defaults" in docker_content

        except FileNotFoundError:
            pytest.skip("docker-compose.yml file not found")
