# NFL Score Predictor

Predict NFL game scores using real team performance data, injury reports, and weather conditions.

This project combines machine learning with context-aware adjustments to generate projected scores, determine likely winners, and estimate over/under outcomes.

---

## 📊 Features

- **Team performance modeling** using scikit-learn Linear Regression
- **Weather-aware scoring adjustments** powered by [nfl-stadiums](https://pypi.org/project/nfl-stadiums/) and Open-Meteo
- **Injury impact modeling** based on quarterback tier ratings
- **Interactive matchup predictor** that outputs winner, margin, and total points
- **CSV export** for matchup predictions

---

## 📁 Project Structure

```
nfl-score-predictor/
│
├── .gitignore
├── LICENSE
├── README.md
│
└── nfl_predictor/
    ├── nfl_ai_scores.py
    ├── requirements.txt
    ├── data/
    │   ├── nfl_injuries_test.csv
    │   ├── nfl_properties_test.yaml
    │   ├── nfl_conversions_thru_week_X.csv
    │   └── ...etc.
```

---

## ⚙️ Setup

```bash
python3 -m venv nfl_env
source nfl_env/bin/activate
pip install -r nfl_predictor/requirements.txt
```

---

## 🚀 Usage

```bash
python nfl_predictor/nfl_ai_scores.py
```

You’ll be prompted to enter:
- Week number
- Number of games
- Game date
- Home and away teams

---

## 📌 Data Sources

- **Team stats**: [Pro Football Reference](https://www.pro-football-reference.com/)
- **Weather**: [Open-Meteo](https://open-meteo.com/) via `nfl-stadiums`
- **Injuries**: Local test CSV (`nfl_injuries_test.csv`)

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).
