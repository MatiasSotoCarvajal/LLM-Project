# LLM Engineering Project

Comparison of quantization methods and **TurboQuant** (an experimental KV Cache
compression method) across different language models.

The project measures GPU memory usage, KV cache size, generation throughput
(time to first token, decode tokens/sec) and perplexity to evaluate the
quality/speed trade-off introduced by each quantization strategy.

## Repository structure

```
.
├── Documentation/      Lecture material and project specification
├── Report/             LaTeX sources for the final report
├── main.ipynb          Benchmark harness (load, measure, save results) # pending removal
├── pyproject.toml      Project metadata and dependencies
├── uv.lock             Reproducible dependency lockfile (uv)
├── Makefile            Environment setup shortcuts
└── README.md
```

## TurboQuant

This repository uses [llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant),
a fork of llama.cpp that integrates the **TurboQuant+** codec stack: Walsh-Hadamard
rotated polar quantization, attention-gated sparse dequantization, and layer-aware
V compression policies. It is an additive fork, so every existing llama.cpp
quantization, model, and backend continues to work unchanged. The new types are
opt-in via the standard `--cache-type-k` / `--cache-type-v` flags and the
`llama-quantize` interface.

TurboQuant operates on two independent domains:

- **Weight quantization** (offline) -- compresses model weights at conversion
  time with `llama-quantize`. Produces smaller `.gguf` files that use less VRAM.
- **KV cache quantization** (runtime) -- compresses the attention KV cache while
  the model is running, selected per-side with `--cache-type-k` and
  `--cache-type-v`. Reduces the memory that grows with context length.

The core design principle is **asymmetric K/V compression**: V tolerates
aggressive compression while K does not. K should always stay at higher
precision than V (prefer `f16` or `q8_0` on the K side). Compressing K
aggressively is where models break.

### Binary setup

Prebuilt TurboQuant+ binaries are downloaded by `scripts/setup_llamacpp.sh`:

```bash
bash scripts/setup_llamacpp.sh
```

This fetches the `tqp-v0.2.0` release for the current platform and extracts
`llama-server` and `llama-quantize` into `bin/turboquant-plus-tqp-v0.2.0/`.

Supported platforms:

| Platform | Build |
|---|---|
| macOS Apple Silicon | Metal |
| Linux x64 | CPU |
| Windows x64 | CUDA 12.4 |

### Weight quantization types (offline)

Applied with `llama-quantize` via `quantize_models.py`:

| Type | Approx. bits | Notes |
|---|---|---|
| `TQ1_0` | ~1.0 | Most aggressive weight format |
| `TQ2_0` | ~2.0 | Aggressive weight format |
| `TQ3_1S` | ~3.5 | Smaller VRAM than `Q8_0`; expect ~1-2 PPL bump |
| `TQ4_1S` | ~4.5 | Recommended for most CUDA/HIP deployments; `dp4a` 3.5x faster than baseline |
| `Q4_K_M` | ~4.8 | Standard llama.cpp type (baseline for comparison) |
| `Q4_K_S` | ~4.5 | Standard llama.cpp type (baseline for comparison) |
| `Q5_K_M` | ~5.5 | Standard llama.cpp type (baseline for comparison) |
| `Q8_0` | ~8.0 | Standard llama.cpp type (baseline for comparison) |

All `TQ*` formats use Walsh-Hadamard rotation followed by polar codebook
quantization on 128-element blocks. The `Q*` types are the standard llama.cpp
quantization formats, included in this project as baselines for comparison.

Example:

```bash
python quantize_models.py -t TQ4_1S
```

### KV cache quantization types (runtime)

Applied at inference time by `backend/llama_server.py` through `llama-server`:

| Type | Approx. bits | Domain | Notes |
|---|---|---|---|
| `f16` | 16.0 | K/V | No compression; safest baseline |
| `q8_0` | ~8.0 | K/V | Standard 8-bit quantization; near-lossless |
| `turbo4` | ~4.5 | V | Lightest turbo tier; rehabilitated to beat `Q4_0` on fidelity |
| `turbo3` | ~3.5 | V | ~4.6x compression at <1.5% PPL loss; recommended default |
| `turbo2` | ~2.0 | V | Most aggressive; pair with Boundary V (auto-enabled) |

This project's defaults (set in `backend/llama_server.py`):

- K cache: `q8_0`
- V cache: `turbo3`

This corresponds to the recommended asymmetric configuration: near-lossless K
with ~4.6x compressed V.

### Recommended configurations

Ordered from most conservative to most aggressive:

| Step | `--cache-type-k` | `--cache-type-v` | When |
|---|---|---|---|
| 1. Safest start | `f16` | `turbo4` | First contact with any new model |
| 2. Conservative | `q8_0` | `turbo4` | Verified safe at step 1, want a memory win |
| 3. Recommended default | `q8_0` | `turbo3` | Most dense models, most production workloads |
| 4. Aggressive V | `q8_0` | `turbo2` | Memory-bound long context, after validating step 3 |

Higher turbo number = more bits per element = less aggressive compression. Do
not start at maximum compression and work backwards; verify output quality at
each step before escalating.

## Requirements

- Python 3.10 or 3.11 (3.12 is not tested with all CUDA wheels)
- [uv](https://github.com/astral-sh/uv):

### macOS and Linux

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### Windows
   ```bash
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

- Hugging Face account (`huggingface-cli login`) is only required for uploading models and faster downloads.

## Environment setup

```bash
make setup
source .venv/bin/activate
```

### Manual (without Make)

```bash
uv sync
source .venv/bin/activate
```

## Usage

pending

## Cleaning up

```bash
make clean    # removes the virtual environment
make clean-models # removes the entire /Models folder
```