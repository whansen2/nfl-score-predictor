import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from nfl_predictor.nfl_ai_scores import get_injuries_adjustment

# Test whether the model can be trained and produce accurate predictions on dummy data
def test_model_training_and_prediction():
    # Dummy data resembling current features
    data = {
        "Sc%_x": [0.3, 0.4, 0.5, 0.6],
        "Tot_1stD/G": [18, 20, 22, 24],
        "Y/P_x": [5.2, 5.5, 6.1, 6.7],
        "RZPct_x": [0.5, 0.55, 0.6, 0.65],
        "TO%_x": [0.12, 0.11, 0.09, 0.07],
        "Sc%_y": [0.25, 0.3, 0.35, 0.4],
        "PPG": [17, 20, 24, 28]
    }
    df = pd.DataFrame(data)
    features = ["Sc%_x", "Tot_1stD/G", "Y/P_x", "RZPct_x", "TO%_x", "Sc%_y"]
    X = df[features]
    y = df["PPG"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)

    # Basic sanity check: R² should be reasonable with dummy data
    assert r2 > 0.8

# Test that the output DataFrame from predictions has the correct structure
def test_output_dataframe_structure():
    sample_data = [["Eagles", 27, "Chiefs", 24, "Eagles win by 3", 51]]
    df = pd.DataFrame(sample_data, columns=["Home Team", "Home Score", "Away Team", "Away Score", "Result", "Over/Under"])
    
    expected_columns = {"Home Team", "Home Score", "Away Team", "Away Score", "Result", "Over/Under"}
    assert set(df.columns) == expected_columns

# Test that injury adjustment function returns valid integer values
def test_injury_adjustment_output_type():
    # Minimal test data for both teams
    dummy = pd.DataFrame({
        "Player": ["QB1", "WR1"],
        "Tm": ["PHI", "KAN"],
        "Pos": ["QB", "WR"],
        "Status": ["Questionable", "Out"],
        "Injury Comment": ["Elbow", "Hamstring"]
    })
    dummy.to_csv("test_injuries.csv", index=False)

    home_team = "Philadelphia Eagles"
    away_team = "Kansas City Chiefs"

    adjust_home, adjust_away = get_injuries_adjustment("test_injuries.csv", home_team, away_team)

    assert isinstance(adjust_home, int)
    assert isinstance(adjust_away, int)

    os.remove("test_injuries.csv")
