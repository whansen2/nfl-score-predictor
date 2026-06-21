import numpy as np
import pytest

from nfl_predictor.utils.helpers import get_training_week, resolve_weeks


@pytest.mark.parametrize(
    "week_value, expected",
    [
        (1, (1, 1)),
        (2, (2, 1)),
        ("3", (3, 2)),
        (np.int64(4), (4, 3)),
        ("SuperBowl", (19, 18)),
    ],
)
def test_resolve_weeks_handles_numeric_and_postseason_values(week_value, expected):
    assert resolve_weeks(week_value) == expected


@pytest.mark.parametrize(
    "week_value, expected",
    [
        (0, 1),
        (1, 1),
        (2, 2),
        ("3", 3),
        ("WildCard", 18),
    ],
)
def test_get_training_week_clamps_regular_season_floor_and_postseason_defaults(
    week_value, expected
):
    assert get_training_week(week_value) == expected
