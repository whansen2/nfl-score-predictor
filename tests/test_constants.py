"""
Test suite for constants module to ensure model-critical configuration is correct.
"""

from nfl_predictor.utils.constants import (
    CONV_AGAINST_FILE,
    CONVERSIONS_FILE,
    DEFAULT_FEATURES,
    DEFAULT_INJURY_ADJUSTMENTS,
    DEFAULT_LOG_LEVEL,
    DEFAULT_VERBOSE_ADJUSTMENTS,
    DEFENSE_FILE,
    HOME_FIELD_ADVANTAGE,
    OFFENSE_FILE,
    RANDOM_STATE,
    TRAIN_TEST_SPLIT_RATIO,
)


class TestFileNameConstants:
    def test_dynamic_file_templates(self) -> None:
        """Weekly stat file templates must expose {week} and {year} placeholders."""
        templates = [CONVERSIONS_FILE, OFFENSE_FILE, DEFENSE_FILE, CONV_AGAINST_FILE]
        for template in templates:
            assert isinstance(template, str)
            assert "{week}" in template
            assert "{year}" in template
            assert template.endswith(".csv")


class TestModelConstants:
    def test_default_features_structure(self) -> None:
        """Feature list order matters — it must match merged column names."""
        expected_features = [
            "Sc%_x",
            "Tot_1stD/G",
            "Y/P_x",
            "RZPct_x",
            "TO%_x",
            "Sc%_y",
        ]
        assert expected_features == DEFAULT_FEATURES

    def test_model_parameters(self) -> None:
        """Model hyperparameters must be within valid ranges."""
        assert isinstance(HOME_FIELD_ADVANTAGE, int)
        assert HOME_FIELD_ADVANTAGE >= 0

        assert isinstance(TRAIN_TEST_SPLIT_RATIO, float)
        assert 0 < TRAIN_TEST_SPLIT_RATIO < 1

        assert isinstance(RANDOM_STATE, int)
        assert RANDOM_STATE >= 0


class TestFeatureFlagDefaults:
    def test_log_level_is_valid(self) -> None:
        """Default log level must be one Python logging accepts."""
        assert DEFAULT_LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR"]

    def test_adjustment_flags_are_bool(self) -> None:
        """Adjustment flags must be booleans so env-var override parsing is safe."""
        assert isinstance(DEFAULT_VERBOSE_ADJUSTMENTS, bool)
        assert isinstance(DEFAULT_INJURY_ADJUSTMENTS, bool)
