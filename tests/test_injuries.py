import os

import pandas as pd
import pytest
import yaml

from nfl_predictor.utils.helpers import get_injuries_adjustment

# Load config from YAML
base_dir = os.path.dirname(os.path.dirname(__file__))
yaml_path = os.path.join(base_dir, "nfl_predictor", "data", "nfl_properties.yaml")

with open(yaml_path) as f:
    config = yaml.safe_load(f)

team_abbreviations = config["team_abbreviations"]
qb_tiers = config["qb_tiers"]
team_qbs = config["team_qbs"]


def _is_resolved_qb_entry(qb_name: str, tier_label: str) -> bool:
    return qb_name != "TBD" and tier_label in qb_tiers


# Build test cases dynamically
test_cases = []
for team, (qb_name, tier_label) in team_qbs.items():
    if not _is_resolved_qb_entry(qb_name, tier_label):
        continue
    expected_adjustment = qb_tiers[tier_label]
    test_cases.append((team, qb_name, tier_label, expected_adjustment))

assert test_cases, "no resolved QB entries found in nfl_properties.yaml"


@pytest.mark.parametrize(
    "team_name, qb_name, tier_label, expected_adjustment", test_cases
)
def test_auto_generated_injury_adjustments(
    tmp_path, team_name, qb_name, tier_label, expected_adjustment
):
    # Pick any opponent that’s not the team being tested
    opponent = next(t for t in team_qbs if t != team_name)

    # Create minimal DataFrame simulating injury report for the test QB
    df = pd.DataFrame(
        [
            {
                "Player": qb_name,
                "Tm": team_abbreviations[team_name],
                "Pos": "QB",
                "Status": "Out",
                "Injury Comment": "Test injury",
            }
        ]
    )
    injury_file = tmp_path / "injuries.csv"
    df.to_csv(injury_file, index=False)

    # Load the test CSV into a DataFrame
    df_loaded = pd.read_csv(injury_file)

    test_team_qbs = {team_name: [qb_name, tier_label], opponent: team_qbs[opponent]}

    adjustment_home, adjustment_away = get_injuries_adjustment(
        df_loaded,
        home_team=team_name,
        away_team=opponent,
        team_abbreviations=team_abbreviations,
        qb_tiers=qb_tiers,
        team_qbs=test_team_qbs,
    )

    assert adjustment_home == expected_adjustment, (
        f"{team_name} expected {expected_adjustment}, got {adjustment_home}"
    )
    assert adjustment_away == 0, f"{opponent} expected 0, got {adjustment_away}"


def test_unresolved_qb_entries_return_neutral_adjustment(tmp_path) -> None:
    # Does not rely on the live fixture data
    # containing a "TBD" placeholder, since real team_qbs entries are
    # expected to be fully resolved.
    team_name, qb_name, tier_label = next(iter(team_qbs)), "TBD", "TBD"
    opponent = next(t for t in team_qbs if t != team_name)

    df = pd.DataFrame(
        [
            {
                "Player": qb_name,
                "Tm": team_abbreviations[team_name],
                "Pos": "QB",
                "Status": "Out",
                "Injury Comment": "Test unresolved QB",
            }
        ]
    )
    injury_file = tmp_path / "injuries_unresolved.csv"
    df.to_csv(injury_file, index=False)
    df_loaded = pd.read_csv(injury_file)

    adjustment_home, adjustment_away = get_injuries_adjustment(
        df_loaded,
        home_team=team_name,
        away_team=opponent,
        team_abbreviations=team_abbreviations,
        qb_tiers=qb_tiers,
        team_qbs={team_name: [qb_name, tier_label], opponent: team_qbs[opponent]},
    )

    assert adjustment_home == 0
    assert adjustment_away == 0


def test_team_qb_entries_are_resolved_or_fully_tbd() -> None:
    invalid_entries = []
    for team, (qb_name, tier_label) in team_qbs.items():
        if _is_resolved_qb_entry(qb_name, tier_label):
            continue
        if qb_name == "TBD" and tier_label == "TBD":
            continue
        invalid_entries.append((team, qb_name, tier_label))

    assert not invalid_entries, f"Invalid QB entry shape(s): {invalid_entries}"


def test_get_injuries_adjustment_returns_neutral_for_empty_injuries_df() -> None:
    adjustment_home, adjustment_away = get_injuries_adjustment(
        pd.DataFrame(),
        home_team="HomeTeam",
        away_team="AwayTeam",
        team_abbreviations={"HomeTeam": "HOM", "AwayTeam": "AWY"},
        qb_tiers={"average": -4},
        team_qbs={
            "HomeTeam": ["Home QB", "average"],
            "AwayTeam": ["Away QB", "average"],
        },
    )
    assert (adjustment_home, adjustment_away) == (0, 0)


def test_get_injuries_adjustment_returns_neutral_when_team_missing_abbreviation(
    tmp_path,
) -> None:
    df = pd.DataFrame(
        [{"Player": "Home QB", "Tm": "HOM", "Pos": "QB", "Status": "Out"}]
    )
    injury_file = tmp_path / "injuries_no_abbr.csv"
    df.to_csv(injury_file, index=False)
    df_loaded = pd.read_csv(injury_file)

    adjustment_home, adjustment_away = get_injuries_adjustment(
        df_loaded,
        home_team="HomeTeam",
        away_team="AwayTeam",
        team_abbreviations={"AwayTeam": "AWY"},  # HomeTeam intentionally omitted
        qb_tiers={"average": -4},
        team_qbs={
            "HomeTeam": ["Home QB", "average"],
            "AwayTeam": ["Away QB", "average"],
        },
    )
    assert adjustment_home == 0
    assert adjustment_away == 0


def test_get_injuries_adjustment_returns_neutral_when_team_missing_from_team_qbs(
    tmp_path,
) -> None:
    df = pd.DataFrame(
        [{"Player": "Home QB", "Tm": "HOM", "Pos": "QB", "Status": "Out"}]
    )
    injury_file = tmp_path / "injuries_no_team_qbs.csv"
    df.to_csv(injury_file, index=False)
    df_loaded = pd.read_csv(injury_file)

    adjustment_home, adjustment_away = get_injuries_adjustment(
        df_loaded,
        home_team="HomeTeam",
        away_team="AwayTeam",
        team_abbreviations={"HomeTeam": "HOM", "AwayTeam": "AWY"},
        qb_tiers={"average": -4},
        team_qbs={"AwayTeam": ["Away QB", "average"]},  # HomeTeam intentionally omitted
    )
    assert adjustment_home == 0
    assert adjustment_away == 0


def test_get_injuries_adjustment_returns_neutral_when_no_matching_injury_entry(
    tmp_path,
) -> None:
    # HomeTeam has an injury report entry, but the status doesn't qualify
    # as out/doubtful, so the loop should exhaust without a match.
    df = pd.DataFrame(
        [{"Player": "Home QB", "Tm": "HOM", "Pos": "QB", "Status": "Questionable"}]
    )
    injury_file = tmp_path / "injuries_no_match.csv"
    df.to_csv(injury_file, index=False)
    df_loaded = pd.read_csv(injury_file)

    adjustment_home, adjustment_away = get_injuries_adjustment(
        df_loaded,
        home_team="HomeTeam",
        away_team="AwayTeam",
        team_abbreviations={"HomeTeam": "HOM", "AwayTeam": "AWY"},
        qb_tiers={"average": -4},
        team_qbs={
            "HomeTeam": ["Home QB", "average"],
            "AwayTeam": ["Away QB", "average"],
        },
    )
    assert adjustment_home == 0
    assert adjustment_away == 0


def test_get_injuries_adjustment_raises_on_malformed_team_qbs_entry(tmp_path) -> None:
    """Bare-string team_qbs entry (missing tier) must raise, not silently return 0."""
    team_name, opponent = next(iter(team_qbs)), None
    for candidate in team_qbs:
        if candidate != team_name:
            opponent = candidate
            break
    assert opponent is not None

    df = pd.DataFrame(
        [
            {
                "Player": "Some QB",
                "Tm": team_abbreviations[team_name],
                "Pos": "QB",
                "Status": "Out",
            }
        ]
    )
    injury_file = tmp_path / "injuries_malformed.csv"
    df.to_csv(injury_file, index=False)
    df_loaded = pd.read_csv(injury_file)

    malformed_team_qbs = {
        team_name: "bare_string_instead_of_list",
        opponent: team_qbs[opponent],
    }

    with pytest.raises(ValueError, match="Invalid team_qbs entry"):
        get_injuries_adjustment(
            df_loaded,
            home_team=team_name,
            away_team=opponent,
            team_abbreviations=team_abbreviations,
            qb_tiers=qb_tiers,
            team_qbs=malformed_team_qbs,
        )
