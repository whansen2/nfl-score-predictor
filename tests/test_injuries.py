import os
import pandas as pd
import pytest
import yaml
from nfl_predictor.utils.helpers import get_injuries_adjustment

# Load config from YAML
base_dir = os.path.dirname(os.path.dirname(__file__))
yaml_path = os.path.join(base_dir, "nfl_predictor", "data", "nfl_properties_test.yaml")

with open(yaml_path, "r") as f:
    config = yaml.safe_load(f)

team_abbreviations = config["team_abbreviations"]
qb_tiers = config["qb_tiers"]
team_qbs = config["team_qbs"]

# Build test cases dynamically
test_cases = []
for team, (qb_name, tier_label) in team_qbs.items():
    expected_adjustment = qb_tiers[tier_label]
    test_cases.append((team, qb_name, tier_label, expected_adjustment))

@pytest.mark.parametrize("team_name, qb_name, tier_label, expected_adjustment", test_cases)
def test_auto_generated_injury_adjustments(tmp_path, team_name, qb_name, tier_label, expected_adjustment):
    # Pick any opponent that’s not the team being tested
    opponent = next(t for t in team_qbs if t != team_name)

    df = pd.DataFrame([{
        "Player": qb_name,
        "Tm": team_abbreviations[team_name],
        "Pos": "QB",
        "Status": "Out",
        "Injury Comment": "Test injury"
    }])
    injury_file = tmp_path / "injuries.csv"
    df.to_csv(injury_file, index=False)

    test_team_qbs = {
        team_name: [qb_name, tier_label],
        opponent: team_qbs[opponent]
    }

    adjustment_home, adjustment_away = get_injuries_adjustment(
        str(injury_file),
        home_team=team_name,
        away_team=opponent,
        team_abbreviations=team_abbreviations,
        qb_tiers=qb_tiers,
        team_qbs=test_team_qbs
    )

    if qb_name == team_qbs[opponent][0]:
        expected_away = qb_tiers[team_qbs[opponent][1]]
    else:
        expected_away = 0

    assert adjustment_home == expected_adjustment, f"{team_name} expected {expected_adjustment}, got {adjustment_home}"
    assert adjustment_away == expected_away, f"{opponent} expected {expected_away}, got {adjustment_away}"
