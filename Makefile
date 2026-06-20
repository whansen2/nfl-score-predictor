# === NFL Score Predictor Makefile ===

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
	PYTHONPATH=. bash -c "source nfl_env/bin/activate && python nfl_predictor/nfl_ai_scores.py"

# Run Docker container
docker-up:
	docker-compose up

# Stop Docker container
docker-down:
	docker-compose down

# Run tests
test:
	. $(VENV)/bin/activate && pytest $(TEST_DIR)

# Clean Python cache
clean:
	find . -type d -name "__pycache__" -exec rm -r {} + || true

# Complete clean including virtual environment
distclean: clean
	rm -rf $(VENV)
	@echo "🧹 Removed virtual environment. Run 'make venv && make install' to recreate."

# === Full Setup ===

all:
	@echo "🔧 Creating virtualenv..."
	$(MAKE) venv
	@echo "📦 Installing dependencies..."
	$(ACTIVATE) && pip install -r $(REQS)
	@echo "🏈 Running predictions..."
	$(MAKE) run
	@echo ""
	@echo "✅ All done!"

.PHONY: venv install run docker-up docker-down test clean distclean all
