# NFL Score Predictor

![Pre-Deploy Checks](https://github.com/whansen2/nfl-score-predictor/actions/workflows/pre-deploy.yml/badge.svg)

![Deploy Lambda Container](https://github.com/whansen2/nfl-score-predictor/actions/workflows/deploy-lambda.yml/badge.svg)

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
├── .dockerignore
├── .gitignore
├── .env
├── Dockerfile.local
├── Dockerfile.lambda
├── docker-compose.yml
├── Makefile
├── README.md
│
└── nfl_predictor/
    ├── nfl_ai_scores.py
    ├── lambda_handler.py
    ├── requirements.txt
    ├── data/
    └── ...

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

```
WEEK_NUMBER=18
YEAR_ABBR=24
GAME_DATE=2025-02-09
HOME_TEAM=Philadelphia Eagles
AWAY_TEAM=Kansas City Chiefs
```

### Local (Docker)
1. Build the Docker container:

`docker-compose build`

2. Run the container:

`docker-compose up`

The container will automatically execute the script and print the predicted scores to the console. Output is saved to a CSV inside the `nfl_predictor/data/` directory.

---

## 🛠️ Makefile Commands
This project includes a Makefile to streamline local development and Docker usage.

### Local (virtual environment)

```
make venv         # Create a virtual environment (nfl_env)
make install      # Install dependencies inside the virtual environment
make run          # Run the prediction script locally using your .env config
make test         # Run all tests (requires pytest)
make clean        # Clean up Python cache (__pycache__ directories)
```

### Local (Docker)

```
make docker-up    # Build and run the Docker container
make docker-down  # Stop the running Docker container and network
```

Make sure your `.env` file is properly configured before using `make run` or `make docker-up`.

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
