# NFL Score Predictor

![Pre-Deploy Checks](https://github.com/whansen2/nfl-score-predictor/actions/workflows/pre-deploy.yml/badge.svg)
![Deploy Lambda Container](https://github.com/whansen2/nfl-score-predictor/actions/workflows/deploy-lambda.yml/badge.svg)

A production-ready NFL game prediction system that combines machine learning with contextual adjustments to generate accurate score predictions, winner analysis, and over/under estimates.

## 🎯 Overview

This system uses scikit-learn Linear Regression trained on real NFL team performance data, enhanced with optional injury and weather adjustments. The AI-powered analysis module provides deep strategic insights using OpenAI's GPT-4o-mini.

**🚀 Production Ready**: Fully containerized with AWS Lambda deployment, comprehensive test suite, and robust error handling.

---

## ✨ Features

- **🤖 AI-Powered Analysis**: OpenAI GPT-4o-mini integration for strategic insights and matchup discussion
- **📊 Machine Learning Predictions**: scikit-learn Linear Regression using 6 key statistical features
- **🌦️ Weather-Aware Adjustments**: Real-time weather impact via [nfl-stadiums](https://pypi.org/project/nfl-stadiums/) and Open-Meteo API
- **🏥 Injury Impact Modeling**: Quarterback tier-based scoring adjustments (5-tier system)
- **🎲 Upset Detection**: AI-powered analysis to identify potential surprise outcomes
- **☁️ AWS Lambda Ready**: Containerized deployment with S3 integration
- **⚙️ Flexible Configuration**: Constants-first approach with environment overrides
- **📈 Export & Analytics**: CSV output with comprehensive game statistics
- **🧪 Test Coverage**: Comprehensive tests ensuring production reliability

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
│   ├── data/              # Training data and configuration
│   │   ├── nfl_properties_test.yaml    # Team/QB configurations
│   │   ├── upcoming_matchups_test.csv  # Game schedule
│   │   ├── nfl_injuries_test.csv       # Injury reports
│   │   └── nfl_team_*_thru_week_*.csv  # Performance statistics
│   └── utils/
│       ├── __init__.py
│       ├── constants.py   # All configuration constants and defaults
│       ├── helpers.py     # Injury/weather adjustment functions
│       └── env_setup.py   # Environment configuration
│
├── tests/                 # Comprehensive test suite
│   ├── test_constants.py
│   ├── test_environment.py
│   ├── test_injuries.py
│   ├── test_injury_and_weather.py
│   ├── test_lambda_handler.py
│   ├── test_model.py
│   ├── test_upsets_agent.py
│   └── test_weather.py
│
├── nfl_stadium_resources/ # Stadium data for weather integration
└── .github/workflows/     # CI/CD pipelines

```

---

## ⚙️ Setup

### Prerequisites

- Python 3.11+ 
- OpenAI API key (for AI-powered analysis)

### Local Development (Virtual Environment)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/whansen2/nfl-score-predictor.git
   cd nfl-score-predictor
   ```

2. **Create and activate virtual environment:**
   ```bash
   make venv
   source nfl_env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   make install
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

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
cp .env.example .env  # Add your OPENAI_API_KEY
make run

# Run AI analysis
python llm/blitz.py
```

---

## 🔧 Configuration

This project uses a **constants-first architecture** for maximum flexibility:

- **`nfl_predictor/utils/constants.py`**: All default values and configuration
- **`.env`**: Override any constants (only `OPENAI_API_KEY` is required)
- **`.env.example`**: Template showing all configurable options

### Essential Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | API key for GPT-4o-mini analysis |
| `ENABLE_INJURY_ADJUSTMENTS` | `false` | QB injury impact scoring adjustments |
| `ENABLE_WEATHER_ADJUSTMENTS` | `false` | Weather-based scoring adjustments |
| `ENABLE_UPSETS_AGENT` | `true` | AI-powered upset detection |
| `WEEK_NUMBER` | `18` | NFL week for predictions |
| `YEAR_ABBR` | `24` | Season year (2024 data for training) |
| `VERBOSE_ADJUSTMENTS` | `false` | Detailed adjustment logging |

### Feature Flags (Production Ready)

The system includes battle-tested optional features:

```bash
# Enable injury adjustments (5-tier QB rating system)
ENABLE_INJURY_ADJUSTMENTS=true

# Enable weather adjustments (temperature, wind, precipitation)
ENABLE_WEATHER_ADJUSTMENTS=true

# Enable verbose logging for adjustments
VERBOSE_ADJUSTMENTS=true
```

**Note**: These features are thoroughly tested (see `tests/test_injury_and_weather.py`) and ready for production use.

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

The system is production-ready for serverless deployment:

1. **Build Lambda container:**
   ```bash
   docker build -f Dockerfile.lambda -t nfl-predictor-lambda .
   ```

2. **Deploy via AWS CLI or GitHub Actions** (see `.github/workflows/deploy-lambda.yml`)

3. **Configure S3 buckets** for input/output files as defined in `constants.py`

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

### AI-Powered Analysis (Blitz Module)

```bash
# Launch interactive analysis chat
python llm/blitz.py
```

Features include:
- Historical matchup analysis
- Strategic team insights
- Injury impact discussion  
- Weather considerations
- Upset potential evaluation

### Advanced Usage Examples

```bash
# Enable all features for comprehensive analysis
export ENABLE_INJURY_ADJUSTMENTS=true
export ENABLE_WEATHER_ADJUSTMENTS=true
export VERBOSE_ADJUSTMENTS=true
make run

# Specific week prediction
export WEEK_NUMBER=1
make run

# Custom logging level
export LOG_LEVEL=DEBUG
make run
```

## 📊 Model & Features

### Core Statistical Features

The Linear Regression model uses 6 key performance indicators:

1. **Sc%_x**: Home team scoring percentage
2. **Tot_1stD/G**: Total first downs per game  
3. **Y/P_x**: Yards per play (offense)
4. **RZPct_x**: Red zone conversion percentage
5. **TO%_x**: Turnover differential percentage
6. **Sc%_y**: Away team scoring percentage

### Optional Adjustments

- **Injury Adjustments**: 5-tier QB rating system (-6 to -2 point penalties)
- **Weather Adjustments**: Temperature, wind, precipitation impact
- **Home Field Advantage**: +1 point boost for home teams

### AI Analysis Features

- **Matchup Insights**: Historical performance analysis
- **Upset Detection**: Statistical anomaly identification  
- **Strategic Discussion**: Interactive chat about game plans
- **Trend Analysis**: Multi-week performance tracking

---

## 📌 Data Sources & Architecture

### Data Sources

- **Team Statistics**: Pro Football Reference historical performance data
- **Weather Data**: [Open-Meteo API](https://open-meteo.com/) via `nfl-stadiums` library
- **Stadium Information**: NFL stadium database with roof types and locations
- **Injury Reports**: Configurable CSV format for QB injury tracking
- **Team Configuration**: YAML-based team and quarterback assignments

### Architecture Highlights

- **Serverless Ready**: AWS Lambda containerized deployment
- **Environment Agnostic**: Works locally, in Docker, and on Lambda
- **Constants-First**: Centralized configuration with environment overrides
- **Extensible**: Modular design for easy feature additions
- **Production Tested**: Comprehensive error handling and logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Run tests: `make test`
4. Commit changes: `git commit -am 'Add new feature'`
5. Push to branch: `git push origin feature/new-feature`
6. Submit a Pull Request

### Development Guidelines

- All new features should include tests
- Follow the constants-first configuration approach
- Update documentation for user-facing changes
- Ensure CI/CD pipeline passes

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).
