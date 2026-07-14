# LLM Engineering Project

## Context

This is the main repository for the LLM Engineering Project.
The project compares different quantization methods across several language
models, with TurboQuant as the primary subject of study. TurboQuant is an
experimental KV cache compression and weight quantization method built on top
of llama.cpp (via the [llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)
fork, release `tqp-v0.2.0`).

The project measures GPU memory usage, KV cache size, generation throughput
(time to first token, decode tokens/sec) and perplexity to evaluate the
quality/speed trade-off introduced by each quantization strategy.

### Models under evaluation

Downloaded as Q8_0 GGUF baselines from Hugging Face, then quantized with
TurboQuant via `quantize_models.py`:

- `unsloth/gemma-4-E4B-it-GGUF`
- `unsloth/gemma-4-E2B-it-GGUF`
- `unsloth/Qwen3.5-9B-GGUF`
- `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`

### Dataset

- `zai-org/LongBench-v2` - long-context evaluation benchmark.

## Repository content

- `Documentation/` - Lecture material and project specification.
- `Report/` - LaTeX sources for the final report (NeurIPS template).
- `backend/llama_server.py` - Python wrapper around the TurboQuant `llama-server`
  binary. Handles model discovery, server lifecycle, and OpenAI-compatible
  chat completion requests with configurable K/V cache types.
- `backend/evaluate.py` - Server-based benchmark harness (CLI). For each model and
  KV cache configuration it launches `llama-server` and measures process memory
  (RSS), KV cache size, throughput (TTFT, decode tokens/sec) and perplexity, then
  appends the results to a CSV under `results/`.
- `benchmarks/test_models.py` - Benchmark test suite (work in progress).
- `bin/` - TurboQuant prebuilt binaries (gitignored, populated by
  `scripts/setup_llamacpp.sh`). Contains `llama-server` and `llama-quantize`.
- `scripts/setup_llamacpp.sh` - Downloads TurboQuant+ prebuilt binaries for the
  current platform (macOS Metal, Linux CPU, Windows CUDA) into `bin/`.
- `download_models.py` - Downloads GGUF model files from Hugging Face Hub into
  `models/`. Supports individual repos or batch download of all configured models.
- `quantize_models.py` - Quantizes every `.gguf` under `models/` using the
  TurboQuant `llama-quantize` binary. Supports all TQ and Q quantization types.
- `upload_models.py` - Uploads TurboQuant-quantized GGUF files to Hugging Face,
  creating linked model repos with auto-generated model cards.
- `main.py` - Entry point for single inference runs via `backend/llama_server.py`.
- `main.ipynb` - Legacy transformers-based benchmark harness (model loading, GPU
  memory, KV cache size, throughput, perplexity, CSV export). Superseded by
  `backend/evaluate.py`; transformers cannot exercise TurboQuant, which lives in
  the llama.cpp fork. Pending removal.
- `test_mlx.py` - Experimental MLX inference test (Apple Silicon local runtime).
- `models/` - Downloaded GGUF models (gitignored).
- `pyproject.toml` - Project metadata and dependencies (used by `uv`).
- `uv.lock` - Reproducible dependency lockfile (used by `uv`).
- `Makefile` - Environment setup shortcuts (`make setup`, `make notebook`,
  `make clean`, `make clean-models`).

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

`make setup` runs `uv sync` and then `scripts/setup_llamacpp.sh` to download
the TurboQuant+ binaries into `bin/`. Both steps must complete before running
inference or quantization.

When adding dependencies, edit `pyproject.toml` (`[project].dependencies`) and
run `uv lock` to regenerate `uv.lock`.

## Dependencies

- torch, transformers, accelerate - model loading and quantization
- datasets, huggingface_hub - dataset download and gated model access
- numpy - numerical helpers
- jupyter, ipykernel - notebook execution
- sentencepiece - tokenizer support
- requests - HTTP client for `llama-server` communication
- mlx, mlx-vlm - Apple Silicon local inference (experimental)
- ruff, black - linting and formatting (dev optional dependencies)

## TurboQuant

The TurboQuant+ binaries live in `bin/turboquant-plus-tqp-v0.2.0/` and provide
two executables:

- `llama-server` - inference server with runtime KV cache quantization
  (`--cache-type-k` / `--cache-type-v` flags).
- `llama-quantize` - offline weight quantization (produces TQ-formatted GGUF files).

Weight quantization types available via `quantize_models.py`:
`TQ1_0`, `TQ2_0`, `TQ3_1S`, `TQ4_1S`, `Q4_K_M`, `Q4_K_S`, `Q5_K_M`, `Q8_0`.

KV cache quantization types available via `backend/llama_server.py`:
`f16`, `q8_0`, `turbo2`, `turbo3`, `turbo4`.

Project defaults: K cache `q8_0`, V cache `turbo3`.

## Rules

Write in a professional way. Don't use any emoji.
Do not add comments unless explicitly requested.
Run `uv lock` after dependency changes and `make setup` to verify the environment builds.
Always keep this file updated with the last important and structural changes.