.PHONY: setup notebook clean clean-models

VENV ?= .venv
MODELS_DIR ?= models
SETUP_BIN ?= scripts/setup_llamacpp.sh

setup:
	uv sync
	bash $(SETUP_BIN)
	@echo "Done. Activate with: source $(VENV)/bin/activate"

notebook:
	$(VENV)/bin/jupyter notebook main.ipynb

clean:
	rm -rf $(VENV)

clean-models:
	rm -rf $(MODELS_DIR)/*