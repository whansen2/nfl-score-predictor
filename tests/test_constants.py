"""
Test suite for constants module to ensure all configuration values are properly defined.
"""
from nfl_predictor.utils.constants import (
    # File names
    INPUT_FILE_NAME,
    INJURIES_FILE_NAME,
    PROPERTIES_FILE_NAME,
    OUTPUT_FILE_NAME,
    FLAGGED_OUTPUT_FILE_NAME,
    # Dynamic file templates
    CONVERSIONS_FILE,
    OFFENSE_FILE,
    DEFENSE_FILE,
    CONV_AGAINST_FILE,
    # Model configuration
    DEFAULT_FEATURES,
    HOME_FIELD_ADVANTAGE,
    TRAIN_TEST_SPLIT_RATIO,
    RANDOM_STATE,
    # AWS configuration
    DEFAULT_INPUT_BUCKET,
    DEFAULT_OUTPUT_BUCKET,
    # Prediction defaults
    DEFAULT_WEEK_NUMBER,
    DEFAULT_YEAR_ABBR,
    # Feature flags
    DEFAULT_LOG_LEVEL,
    DEFAULT_VERBOSE_ADJUSTMENTS,
    DEFAULT_INJURY_ADJUSTMENTS,
    DEFAULT_WEATHER_ADJUSTMENTS,
    DEFAULT_UPSETS_AGENT
)


class TestFileNameConstants:
    """Test file name constants are properly defined."""
    
    def test_file_names_are_strings(self):
        """Ensure all file names are strings."""
        file_names = [
            INPUT_FILE_NAME,
            INJURIES_FILE_NAME,
            PROPERTIES_FILE_NAME,
            OUTPUT_FILE_NAME,
            FLAGGED_OUTPUT_FILE_NAME
        ]
        for file_name in file_names:
            assert isinstance(file_name, str)
            assert len(file_name) > 0
            assert file_name.endswith('.csv') or file_name.endswith('.yaml')
    
    def test_dynamic_file_templates(self):
        """Test dynamic file templates have proper format placeholders."""
        templates = [CONVERSIONS_FILE, OFFENSE_FILE, DEFENSE_FILE, CONV_AGAINST_FILE]
        for template in templates:
            assert isinstance(template, str)
            assert "{week}" in template
            assert "{year}" in template
            assert template.endswith('.csv')


class TestModelConstants:
    """Test model configuration constants."""
    
    def test_default_features_structure(self):
        """Test DEFAULT_FEATURES is a valid list of feature names."""
        assert isinstance(DEFAULT_FEATURES, list)
        assert len(DEFAULT_FEATURES) == 6
        expected_features = ["Sc%_x", "Tot_1stD/G", "Y/P_x", "RZPct_x", "TO%_x", "Sc%_y"]
        assert DEFAULT_FEATURES == expected_features
    
    def test_model_parameters(self):
        """Test model parameters have correct types and ranges."""
        assert isinstance(HOME_FIELD_ADVANTAGE, int)
        assert HOME_FIELD_ADVANTAGE >= 0
        
        assert isinstance(TRAIN_TEST_SPLIT_RATIO, float)
        assert 0 < TRAIN_TEST_SPLIT_RATIO < 1
        
        assert isinstance(RANDOM_STATE, int)
        assert RANDOM_STATE >= 0


class TestDefaultValues:
    """Test default configuration values."""
    
    def test_aws_configuration(self):
        """Test AWS configuration defaults."""
        assert isinstance(DEFAULT_INPUT_BUCKET, str)
        assert isinstance(DEFAULT_OUTPUT_BUCKET, str)
        assert "nfl-score-predictor" in DEFAULT_INPUT_BUCKET
        assert "nfl-score-predictor" in DEFAULT_OUTPUT_BUCKET
    
    def test_prediction_defaults(self):
        """Test prediction default values."""
        assert isinstance(DEFAULT_WEEK_NUMBER, int)
        assert 1 <= DEFAULT_WEEK_NUMBER <= 22  # Updated to support regular season + playoffs
        
        assert isinstance(DEFAULT_YEAR_ABBR, int)
        assert DEFAULT_YEAR_ABBR >= 0
    
    def test_feature_flags(self):
        """Test feature flag defaults."""
        assert DEFAULT_LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR"]
        
        assert isinstance(DEFAULT_VERBOSE_ADJUSTMENTS, bool)
        assert isinstance(DEFAULT_INJURY_ADJUSTMENTS, bool)
        assert isinstance(DEFAULT_WEATHER_ADJUSTMENTS, bool)
        assert isinstance(DEFAULT_UPSETS_AGENT, bool)


class TestConstantsIntegrity:
    """Test constants work together properly."""
    
    def test_matchups_csv_exists(self):
        """Ensure the matchups CSV file exists since it's now required."""
        import os
        from nfl_predictor.utils.constants import INPUT_FILE_NAME
        
        # This test ensures the required matchups CSV exists in the expected location
        # Get the absolute path from the current file location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        expected_path = os.path.join(project_root, "nfl_predictor", "data", INPUT_FILE_NAME)
        assert os.path.exists(expected_path), f"Required matchups file not found: {expected_path}"
