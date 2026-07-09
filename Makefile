.PHONY: setup notebook clean clean-models

VENV ?= .venv
MODELS_DIR ?= models

setup:
	uv sync
	@echo "Done. Activate with: source $(VENV)/bin/activate"

notebook:
	$(VENV)/bin/jupyter notebook main.ipynb

clean:
	rm -rf $(VENV)

clean-models:
	rm -rf $(MODELS_DIR)/*