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

# === Blitz Shortcuts (Delegates to llm/Makefile) ===

blitz-chat:
	$(MAKE) -C llm chat

blitz-predict:
	$(MAKE) -C llm predict

blitz-recap:
	$(MAKE) -C llm recap

blitz-flag:
	$(MAKE) -C llm flag

blitz-run:
	$(MAKE) -C llm run

# Blitz help menu
blitz-help:
	@echo ""
	@echo "🧠 Blitz - Local LLM Assistant Commands:"
	@echo "----------------------------------------"
	@echo "make blitz-chat      💬  Chat with Blitz about predictions"
	@echo "make blitz-predict   🧮  Run Blitz's copy of predictions"
	@echo "make blitz-recap     📰  Get a weekly game recap"
	@echo "make blitz-flag      🚨  See close calls or upsets"
	@echo "make blitz-run       🔁  Flag upsets and recap the week"
	@echo ""

# === Full Setup ===

all:
	@echo "🔧 Creating virtualenv..."
	$(MAKE) venv
	@echo "📦 Installing predictor dependencies..."
	$(ACTIVATE) && pip install -r $(REQS)
	@echo "🤖 Installing Blitz (LLM) dependencies..."
	$(MAKE) -C llm install
	@echo "🏈 Running predictions..."
	$(MAKE) run
	@echo ""
	@echo "✅ All done!"
	@echo "👉 Next steps: try 'make blitz-help' or 'make blitz-chat'"

.PHONY: venv install run docker-up docker-down test clean blitz-chat blitz-predict blitz-recap blitz-flag blitz-run blitz-help all
