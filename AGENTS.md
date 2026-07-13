# LLM Engineering Project

## Context

This is the main repository for the LLM Engineering Project.
This project consists in comparing different quantization models across different models.
The main actor here is TurboQuant. An experimental new KV Cache compression method that can, theoretically, compress the KV memory on different models.

## Repository content

- `Documentation/` - Lecture material and project specification.
- `Report/` - LaTeX sources for the final report.
- `scripts/setup_llamacpp.sh` - Downloads the prebuilt TurboQuant llama.cpp binaries (`llama-server`, `llama-quantize`) into `bin/`.
- `download_models.py` - Downloads baseline GGUF models from the Hugging Face Hub into `Models/`.
- `quantize_models.py` - Quantizes GGUF models in `Models/` with the TurboQuant `llama-quantize` binary.
- `upload_models.py` - Uploads TurboQuant GGUF models back to the Hugging Face Hub with generated model cards.
- `backend/llama_server.py` - Starts/stops the TurboQuant `llama-server`, waits for health, and issues chat requests. KV cache compression is selected with `--cache-type-k` / `--cache-type-v` (e.g. `turbo3`).
- `backend/evaluate.py` - Server-based benchmark harness (CLI). For each model and KV cache configuration it launches `llama-server` and measures process memory (RSS), KV cache size, throughput (TTFT, tokens/sec) and perplexity, then appends the results to a CSV under `results/`.
- `main.ipynb` - Legacy transformers-based benchmark functions. Superseded by `backend/evaluate.py`; kept for reference and plotting only. Note: transformers cannot exercise TurboQuant, which lives in the llama.cpp fork.
- `pyproject.toml` - Project metadata and dependencies (used by `uv`).
- `uv.lock` - Reproducible dependency lockfile (used by `uv`).
- `Makefile` - Environment setup shortcuts (`make setup`, `make notebook`, `make clean`).

## Evaluation workflow

1. `scripts/setup_llamacpp.sh` - fetch the TurboQuant binaries into `bin/`.
2. `python download_models.py` - fetch baseline GGUF models into `Models/`.
3. `python quantize_models.py -t TQ4_1S` - optionally produce quantized variants.
4. `python -m backend.evaluate <model_id> [--cache-configs k:v,k:v]` - run the benchmark; results are written to `results/results.csv`.

## Environment setup

Python 3.10 or 3.11 is required (3.12 is not tested with all CUDA wheels).
The project uses [uv](https://github.com/astral-sh/uv) exclusively:

```bash
make setup
source .venv/bin/activate
```

When adding dependencies, edit `pyproject.toml` (`[project].dependencies`) and
run `uv lock` to regenerate `uv.lock`.

## Dependencies

- torch, transformers, accelerate - model loading and quantization
- datasets, huggingface_hub - dataset download and gated model access
- numpy - numerical helpers
- jupyter, ipykernel - notebook execution

## Rules

Write in a professional way. Don't use any emoji.
Do not add comments unless explicitly requested.
Run `uv lock` after dependency changes and `make setup` to verify the environment builds.
Always keep this file updated with the last important and structural changes.