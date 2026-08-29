# NFL Score Predictor

![Pre-Deploy Checks](https://github.com/whansen2/nfl-score-predictor/actions/workflows/pre-deploy.yml/badge.svg)
![Deploy Lambda Container](https://github.com/whansen2/nfl-score-predictor/actions/workflows/deploy-lambda.yml/badge.svg)

An NFL game prediction system that combines machine learning with contextual adjustments to generate accurate score predictions, winner analysis, and over/under estimates.

**🏈 Currently configured for the 2026 season** with matchup-driven week selection from the input CSV.

---

## ✨ Features

- **📊 Machine Learning Predictions**: scikit-learn Linear Regression using 6 key statistical features
- **🏥 Injury Impact Modeling**: Quarterback tier-based scoring adjustments (5-tier system)
- **☁️ AWS Lambda Ready**: Containerized deployment with S3 integration
- **⚙️ Flexible Configuration**: Constants-first approach with environment overrides
- **📈 Export & Analytics**: CSV output with comprehensive game statistics
- **🧪 Test Coverage**: Comprehensive tests ensuring reliability

---

## 📁 Project Structure

```
nfl-score-predictor/
│
├── .dockerignore
├── .env                       # Optional local overrides (gitignored)
├── .env.example               # Template for optional environment variables
├── .github/workflows/         # CI/CD pipelines
│   ├── pre-deploy.yml
│   └── deploy-lambda.yml
├── .gitignore
├── .pre-commit-config.yaml    # Pre-commit hooks (linting, security, formatting)
├── Dockerfile.lambda
├── Dockerfile.local
├── docker-compose.yml
├── LICENSE
├── Makefile
├── README.md
├── pyproject.toml             # Project metadata and dependencies
│
├── nfl_predictor/             # Main prediction engine
│   ├── __init__.py
│   ├── nfl_ai_scores.py       # Core prediction script
│   ├── lambda_handler.py      # AWS Lambda deployment handler
│   ├── agents/                # Reserved for future AI agents
│   ├── data/                  # Training data and configuration
│   │   ├── nfl_properties.yaml                             # Team/QB configurations
│   │   ├── upcoming_matchups_auto.csv                     # Game schedule
│   │   ├── nfl_injuries.csv                               # Injury reports
│   │   ├── nfl_team_offense_thru_week_{1..18}_25.csv      # Weekly offense stats
│   │   ├── nfl_team_defense_thru_week_{1..18}_25.csv      # Weekly defense stats
│   │   ├── nfl_conversions_thru_week_{1..18}_25.csv       # Weekly offensive conversions
│   │   ├── nfl_conversions_against_thru_week_{1..18}_25.csv  # Weekly defensive conversions
│   │   ├── standings_thru_week_{1..18}_25.csv             # Weekly standings snapshots
│   │   └── standings.csv                                  # Current standings snapshot
│   └── utils/
│       ├── __init__.py
│       ├── constants.py       # All configuration constants and defaults
│       └── helpers.py         # Injury adjustment functions
│
└── tests/                     # Comprehensive test suite
    ├── test_constants.py
    ├── test_injuries.py
    ├── test_lambda_handler.py
    ├── test_model.py
    └── test_nfl_ai_scores.py
```

---

## ⚙️ Setup

### Prerequisites

- Python 3.12

### Local Development (Virtual Environment)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/whansen2/nfl-score-predictor.git
   cd nfl-score-predictor
   ```

2. **Create and activate virtual environment:**
   ```bash
   make venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   make install
   ```

4. **Configure environment (optional):**
   ```bash
   cp .env.example .env
   ```

   The predictor runs with defaults from `nfl_predictor/utils/constants.py`.

5. **Run tests to verify setup:**
   ```bash
   make test
   ```

6. **Run predictions:**
   ```bash
   make run
   ```

---

## Quick Start

```bash
# Complete setup and run
make venv && make install
make test
make run
```

---

## 🔧 Configuration

This project uses a **constants-first architecture** for maximum flexibility:

- **`nfl_predictor/utils/constants.py`**: All default values and configuration
- **`.env`**: Optional local overrides
- **`.env.example`**: Template showing all configurable options

### Essential Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_INJURY_ADJUSTMENTS` | `false` | QB injury impact scoring adjustments |
| `YEAR_ABBR` | `26` | Season year (2026 season) |
| `VERBOSE_ADJUSTMENTS` | `false` | Detailed adjustment logging |
| `LOG_LEVEL` | `INFO` | Application logging level |
| `OUTPUT_BUCKET` | `nfl-score-predictor-test-output` | Lambda output bucket override |

The prediction week is read from each row in `nfl_predictor/data/upcoming_matchups_auto.csv`.
For week 1 matchups, the pipeline automatically trains from prior-season week 18 data.

### Feature Flags

The system includes battle-tested optional features:

```bash
# Enable injury adjustments (5-tier QB rating system)
ENABLE_INJURY_ADJUSTMENTS=true

# Enable verbose logging for adjustments
VERBOSE_ADJUSTMENTS=true
```

**Note**: These features are thoroughly tested (see `tests/test_injuries.py`).

### Docker Development

1. **Build and run:**
   ```bash
   make docker-up
   ```

2. **Stop containers:**
   ```bash
   make docker-down
   ```

### AWS Lambda Deployment

The system supports serverless deployment:

1. **Build Lambda container:**
   ```bash
   docker build --platform linux/amd64 -f Dockerfile.lambda -t nfl-predictor-lambda .
   ```

2. **Deploy via AWS CLI or GitHub Actions** (see `.github/workflows/deploy-lambda.yml`)

3. **Configure the output S3 bucket** via `OUTPUT_BUCKET` (defaults to `DEFAULT_OUTPUT_BUCKET` in `constants.py`)

---

## 🚀 Usage

### Core Prediction Engine

```bash
# Run predictions with current configuration
python nfl_predictor/nfl_ai_scores.py

# Or use the Makefile
make run
```

**Output**: Predictions saved to `nfl_predictor/data/predicted_matchups_test.csv`

**Input**: Reads from `nfl_predictor/data/upcoming_matchups_auto.csv` and team statistics

### Advanced Usage Examples

```bash
# Enable all features for comprehensive analysis
export ENABLE_INJURY_ADJUSTMENTS=true
export VERBOSE_ADJUSTMENTS=true
make run

# Custom logging level
export LOG_LEVEL=DEBUG
make run
```

## 📊 Model & Features

### Core Statistical Features

The Linear Regression model uses 6 key performance indicators (all team-aggregate, not matchup-specific):

1. **Sc%_x**: Team offensive scoring percentage
2. **Tot_1stD/G**: Total first downs per game
3. **Y/P_x**: Yards per play (team offense)
4. **RZPct_x**: Red zone conversion percentage
5. **TO%_x**: Turnover differential percentage
6. **Sc%_y**: Team defensive scoring percentage

Note: The model predicts each team's expected scoring independently based on its season stats; opponent identity does not affect the prediction. Predictions reflect each team's aggregate offensive/defensive efficiency.

### Optional Adjustments

- **Injury Adjustments**: 5-tier QB rating system (-6 to -2 point penalties)
- **Home Field Advantage**: +1 point boost for home teams

## 📌 Data Sources & Architecture

### Data Sources

- **Team Statistics**: Pro Football Reference historical performance data
- **Injury Reports**: Configurable CSV format for QB injury tracking
- **Team Configuration**: YAML-based team and quarterback assignments

### Architecture Highlights

- **Serverless Ready**: AWS Lambda containerized deployment
- **Environment Agnostic**: Works locally, in Docker, and on Lambda
- **Constants-First**: Centralized configuration with environment overrides
- **Extensible**: Modular design for easy feature additions
- **Error Handling & Logging**: Comprehensive error handling and logging throughout the pipeline

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).
