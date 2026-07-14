.PHONY: setup notebook clean clean-bin clean-models

VENV ?= .venv
BIN_DIR ?= bin
MODELS_DIR ?= models
SETUP_BIN ?= scripts/setup_llamacpp.sh

setup:
	uv sync 
	if [ ! -d "$(BIN_DIR)" ]; then bash $(SETUP_BIN); fi
	@echo "Done. Activate with: source $(VENV)/bin/activate"

notebook:
	$(VENV)/bin/jupyter notebook main.ipynb

clean:
	rm -rf $(VENV)

clean-bin:
	rm -rf $(BIN_DIR)

clean-models:
	rm -rf $(MODELS_DIR)/*