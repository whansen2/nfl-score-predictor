# Makefile for NFL Score Predictor

# === ENV ===
ENV_FILE = .env
PYTHON = python
VENV = nfl_env
ACTIVATE = source $(VENV)/bin/activate

# === PATHS ===
REQS = nfl_predictor/requirements.txt
SCRIPT = nfl_predictor/nfl_ai_scores.py
DATA_DIR = nfl_predictor/data
TEST_DIR = tests

# === COMMANDS ===

# Create virtual environment
venv:
	python3 -m venv $(VENV)

# Install dependencies
install:
	$(ACTIVATE) && pip install -r $(REQS)

# Run the predictor script locally
run:
	bash -c "$(ACTIVATE) && $(PYTHON) $(SCRIPT)"

# Run Docker container
docker-up:
	docker-compose up

# Stop Docker container
docker-down:
	docker-compose down

# Run tests (you can expand this later)
test:
	. $(VENV)/bin/activate && pytest $(TEST_DIR)

# Clean Python cache
clean:
	find . -type d -name "__pycache__" -exec rm -r {} + || true

.PHONY: venv install run docker-up docker-down test clean
