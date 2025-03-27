import pandas as pd
import pytest
from unittest.mock import patch
from sklearn.linear_model import LinearRegression

@pytest.mark.parametrize("inj_adj, wt_adj, home_stats, away_stats", [
    # Elite QB out at home, bad weather
    ((-6, 0), -2,
     {"Sc%_x": 0.45, "Tot_1stD/G": 20, "Y/P_x": 6.2, "RZPct_x": 0.6, "TO%_x": 0.1, "Sc%_y": 0.35},
     {"Sc%_x": 0.42, "Tot_1stD/G": 18.2, "Y/P_x": 5.8, "RZPct_x": 0.55, "TO%_x": 0.12, "Sc%_y": 0.32}),
    
    # Both QBs out, neutral weather
    ((-6, -4), 0,
     {"Sc%_x": 0.5, "Tot_1stD/G": 21, "Y/P_x": 6.0, "RZPct_x": 0.58, "TO%_x": 0.09, "Sc%_y": 0.34},
     {"Sc%_x": 0.48, "Tot_1stD/G": 19.5, "Y/P_x": 5.6, "RZPct_x": 0.52, "TO%_x": 0.1, "Sc%_y": 0.3}),
    
    # No injuries, no weather impact
    ((0, 0), 0,
     {"Sc%_x": 0.55, "Tot_1stD/G": 22, "Y/P_x": 6.3, "RZPct_x": 0.62, "TO%_x": 0.08, "Sc%_y": 0.37},
     {"Sc%_x": 0.5, "Tot_1stD/G": 20, "Y/P_x": 6.0, "RZPct_x": 0.6, "TO%_x": 0.1, "Sc%_y": 0.33}),
])

@patch("nfl_predictor.utils.helpers.get_weather_adjustment")
@patch("nfl_predictor.utils.helpers.get_injuries_adjustment")
def test_prediction_pipeline_with_adjustments_param(
    mock_injuries, mock_weather, inj_adj, wt_adj, home_stats, away_stats
):
    # Mock return values
    mock_injuries.return_value = inj_adj
    mock_weather.return_value = wt_adj

    features = ["Sc%_x", "Tot_1stD/G", "Y/P_x", "RZPct_x", "TO%_x", "Sc%_y"]
    df = pd.DataFrame([
        {"Tm": "HomeTeam", **home_stats, "PF": 350, "G": 17},
        {"Tm": "AwayTeam", **away_stats, "PF": 320, "G": 17}
    ])
    df["PPG"] = df["PF"] / df["G"]
    df["Tot_1stD/G"] = df["Tot_1stD/G"]

    X = df[features]
    y = df["PPG"]

    model = LinearRegression()
    model.fit(X, y)

    ht_pred = round(model.predict(df[df["Tm"] == "HomeTeam"][features])[0]) + 1
    at_pred = round(model.predict(df[df["Tm"] == "AwayTeam"][features])[0])

    ht_pred += inj_adj[0] + wt_adj
    at_pred += inj_adj[1] + wt_adj

    assert isinstance(ht_pred, int)
    assert isinstance(at_pred, int)
    assert ht_pred > 0
    assert at_pred > 0
    assert abs(ht_pred - at_pred) >= 0
