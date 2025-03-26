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

## ⚙️ Setup

```bash
# Create virtual environment
python3 -m venv nfl_env
source nfl_env/bin/activate

# Install dependencies
pip install -r requirements.txt
