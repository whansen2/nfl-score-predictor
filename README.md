# NFL Score Predictor

![Pre-Deploy Checks](https://github.com/whansen2/nfl-score-predictor/actions/workflows/pre-deploy.yml/badge.svg)

![Deploy Lambda Container](https://github.com/whansen2/nfl-score-predictor/actions/workflows/deploy-lambda.yml/badge.svg)

Predict NFL game scores using real team performance data, injury reports, and weather conditions.

This project combines machine learning with context-aware adjustments to generate projected scores, determine likely winners, and estimate over/under outcomes.

---

## 📊 Features

- **AI-powered analysis** using OpenAI GPT-4o-mini for deep game insights and strategy discussion
- **Team performance modeling** using scikit-learn Linear Regression with 6 key statistical features
- **Weather-aware scoring adjustments** powered by [nfl-stadiums](https://pypi.org/project/nfl-stadiums/) and Open-Meteo
- **Injury impact modeling** based on quarterback tier ratings
- **Interactive matchup predictor** that outputs winner, margin, and total points
- **Upset detection agent** for identifying potential surprise outcomes
- **AWS Lambda deployment** ready with containerized architecture
- **Centralized configuration** via constants.py with environment overrides
- **CSV export** for matchup predictions

---

## 📁 Project Structure

```
nfl-score-predictor/
│
├── .dockerignore
├── .gitignore
├── .env                    # Contains only OPENAI_API_KEY
├── .env.example           # Template for environment variables
├── Dockerfile.local
├── Dockerfile.lambda
├── docker-compose.yml
├── Makefile
├── pytest.ini
├── README.md
├── LICENSE
│
├── llm/                   # OpenAI-powered analysis module
│   ├── blitz.py          # GPT-4o-mini integration for game analysis
│   ├── requirements.txt
│   ├── Makefile
│   └── README.md
│
├── nfl_predictor/         # Main prediction engine
│   ├── __init__.py
│   ├── nfl_ai_scores.py   # Core prediction script
│   ├── lambda_handler.py  # AWS Lambda deployment handler
│   ├── requirements.txt
│   ├── agents/
│   │   ├── __init__.py
│   │   └── upsets_ai_agent.py  # Upset detection logic
│   ├── data/              # Training data and test files
│   └── utils/
│       ├── __init__.py
│       ├── constants.py   # All configuration constants and defaults
│       ├── helpers.py     # Utility functions
│       └── env_setup.py   # Environment configuration
│
└── tests/                 # Test suite
    ├── test_model.py
    ├── test_injuries.py
    └── test_weather.py

```

---

## ⚙️ Setup

### Local (virtual environment)

1. Create a virtual environment:

`python3 -m venv nfl_env`
`source nfl_env/bin/activate`

2. Install dependencies:

`pip install -r nfl_predictor/requirements.txt`

For AI-powered analysis features, also install:

`pip install -r llm/requirements.txt`

3. Copy the environment template and add your OpenAI API key:

`cp .env.example .env`

Then edit `.env` and add your OpenAI API key:

```
# Only required variable - OpenAI API key for AI-powered analysis
OPENAI_API_KEY=your_openai_api_key_here
```

**Note**: All other configuration values (game parameters, feature flags, etc.) are defined in `nfl_predictor/utils/constants.py`. You can optionally override any of these in `.env` if needed (see `.env.example` for available options).

---

## 🔧 Configuration

This project uses a **constants-first approach** for configuration:

- **`nfl_predictor/utils/constants.py`**: Contains all default values for features, model parameters, file names, etc.
- **`.env`**: Contains only the required OpenAI API key. Can optionally override any constants.
- **`.env.example`**: Template showing available environment variables you can override.

### Key Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | API key for GPT-4o-mini analysis |
| `LOG_LEVEL` | `INFO` | Logging verbosity level |
| `ENABLE_INJURY_ADJUSTMENTS` | `false` | Include QB injury impact in predictions |
| `ENABLE_WEATHER_ADJUSTMENTS` | `false` | Include weather conditions in scoring |
| `ENABLE_UPSETS_AGENT` | `true` | Run AI-powered upset analysis |
| `WEEK_NUMBER` | `18` | NFL week for predictions |
| `HOME_TEAM` | `Philadelphia Eagles` | Default home team |
| `AWAY_TEAM` | `Kansas City Chiefs` | Default away team |

See `nfl_predictor/utils/constants.py` for the complete list of configurable values.

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

### Core Prediction Engine

Run the main prediction script locally (after activating your virtual environment):

`python nfl_predictor/nfl_ai_scores.py`

Matchups will be predicted based on your configuration — no manual inputs required.

### AI-Powered Analysis

For deeper game insights and analysis, use the LLM module:

`python llm/blitz.py`

This launches an interactive chat interface powered by GPT-4o-mini for discussing matchups, strategies, and predictions.

---

## 📌 Data Sources

- **Team stats**: [Pro Football Reference](https://www.pro-football-reference.com/)
- **Weather**: [Open-Meteo](https://open-meteo.com/) via `nfl-stadiums`
- **Injuries**: Local test CSV (`nfl_injuries_test.csv`)

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).
