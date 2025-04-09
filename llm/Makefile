# === Blitz Makefile (LLM module) ===

VENV = ../nfl_env
ACTIVATE = source $(VENV)/bin/activate

# Create virtual environment (shared at root)
venv:
	python3 -m venv $(VENV)

# Install only LLM-related requirements
install:
	$(ACTIVATE) && pip install -r requirements.txt

# === Blitz Commands ===

chat:
	PYTHONPATH=.. $(ACTIVATE) && python blitz.py chat

predict:
	PYTHONPATH=.. $(ACTIVATE) && python blitz.py predict

flag:
	PYTHONPATH=.. $(ACTIVATE) && python blitz.py flag

recap:
	PYTHONPATH=.. $(ACTIVATE) && python blitz.py recap

run:
	make flag
	make recap

.PHONY: venv install chat predict flag recap run
