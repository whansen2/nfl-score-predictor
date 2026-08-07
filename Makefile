# === NFL Score Predictor Makefile ===

# === ENV ===
PYTHON ?= python3.12
VENV = .venv
PYTHON_BIN = $(VENV)/bin/python
PIP_BIN = $(VENV)/bin/pip

# === PATHS ===
TEST_DIR = tests

# === COMMANDS ===

# Create virtual environment
venv:
	$(PYTHON) -m venv $(VENV)

# Install dependencies
install:
	$(PIP_BIN) install -e ".[dev]"

# Run the predictor script locally
run:
	PYTHONPATH=. $(PYTHON_BIN) nfl_predictor/nfl_ai_scores.py

# Run Docker container
docker-up:
	docker compose up

# Stop Docker container
docker-down:
	docker compose down

# Run tests
test:
	$(PYTHON_BIN) -m pytest $(TEST_DIR)

# Clean local caches and coverage artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -r {} + || true
	rm -rf .pytest_cache .ruff_cache htmlcov
	rm -f .coverage .coverage.*

# Complete clean including virtual environment
distclean: clean
	rm -rf $(VENV)
	@echo "🧹 Removed virtual environment. Run 'make venv && make install' to recreate."

# === Full Setup ===

all:
	@echo "🔧 Creating virtualenv..."
	$(MAKE) venv
	@echo "📦 Installing dependencies..."
	$(MAKE) install
	@echo "🧪 Running tests..."
	$(MAKE) test
	@echo "🏈 Running predictions..."
	$(MAKE) run
	@echo ""
	@echo "✅ All done!"

.PHONY: venv install run docker-up docker-down test clean distclean all
