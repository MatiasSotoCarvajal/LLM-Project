# LLM Engineering Project

## Context

This is the main repository for the LLM Engineering Project.
This project consists in comparing different quantization models across different models.
The main actor here is TurboQuant. An experimental new KV Cache compression method that can, theoretically, compress the KV memory on different models.

## Repository content

- `Documentation/` - Lecture material and project specification.
- `Report/` - LaTeX sources for the final report.
- `main.ipynb` - Benchmark harness: model loading, GPU memory, KV cache size, throughput, perplexity, and CSV export.
- `prepare_download.py` - Helper to fetch the LongBench-v2 dataset.
- `pyproject.toml` - Project metadata and dependencies (used by `uv`).
- `uv.lock` - Reproducible dependency lockfile (used by `uv`).
- `Makefile` - Environment setup shortcuts (`make setup`, `make notebook`, `make clean`).

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