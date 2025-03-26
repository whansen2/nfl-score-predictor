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
├── .gitignore                # Git ignore file for excluding unwanted files
├── Dockerfile                # Dockerfile to build the image for the application
├── LICENSE                   # License file
├── README.md                 # Project documentation and instructions
├── docker-compose.yml        # Docker Compose configuration file for container orchestration
├── .env                      # Environment variable configuration for local development (currently ignored)
│
└── nfl_predictor/
    ├── nfl_ai_scores.py      # Main Python script for NFL score prediction
    ├── requirements.txt      # List of dependencies for the Python application
    ├── data/                 # Folder containing the data files for predictions
    │   ├── nfl_injuries_test.csv
    │   ├── nfl_properties_test.yaml
    │   ├── nfl_conversions_thru_week_X.csv
    │   └── ...etc.
    └── ...other files
```

---

## ⚙️ Setup

### Local (virtual environment)

1. Create a virtual environment:

`python3 -m venv nfl_env`
`source nfl_env/bin/activate`

2. Install dependencies:

`pip install -r nfl_predictor/requirements.txt`

3. Set your environment variables in the `.env` file:

WEEK_NUMBER=18
NUM_GAMES=1
GAME_DATE=2025-02-09
HOME_TEAM=Philadelphia Eagles
AWAY_TEAM=Kansas City Chiefs

### Local (Docker)
1. Build the Docker container:

`docker-compose build`

2. Run the container:

`docker-compose up`

The container will automatically execute the script and print the predicted scores to the console. Output is saved to a CSV inside the `nfl_predictor/data/` directory.

---

## 🚀 Usage

Run the script locally (after activating your virtual environment):

`python nfl_predictor/nfl_ai_scores.py`

Matchups will be predicted based on your environment variables — no manual inputs required.

---

## 📌 Data Sources

- **Team stats**: [Pro Football Reference](https://www.pro-football-reference.com/)
- **Weather**: [Open-Meteo](https://open-meteo.com/) via `nfl-stadiums`
- **Injuries**: Local test CSV (`nfl_injuries_test.csv`)

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).
