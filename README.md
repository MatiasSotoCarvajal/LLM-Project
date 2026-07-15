# LLM Engineering Project

Comparison of quantization methods and **TurboQuant** (an experimental KV Cache
compression method) across different language models.

The project measures GPU memory usage, KV cache size, generation throughput
(time to first token, decode tokens/sec) and perplexity to evaluate the
quality/speed trade-off introduced by each quantization strategy.

## Repository structure

```
.
├── backend/            Python wrappers: llama-server lifecycle, evaluate CLI, config
├── benchmarks/         LongBench V2 evaluation harness
├── bin/                TurboQuant binaries (gitignored, populated by setup script)
├── Documentation/      Lecture material and project specification
├── Report/             LaTeX sources for the final report (NeurIPS template)
├── models/             Downloaded GGUF models (gitignored)
├── results/            Benchmark output (CSV + JSON)
├── scripts/            Setup and build scripts
├── tools/              Download, quantize, and upload models
├── main.py             Single-inference entry point
├── pyproject.toml      Project metadata and dependencies
├── uv.lock             Reproducible dependency lockfile
├── Makefile            Environment setup shortcuts
├── AGENTS.md           Developer reference
└── README.md
```

## Requirements

- Python 3.10 or 3.11
- [uv](https://github.com/astral-sh/uv) for dependency management
- **CUDA 13.2+** (for Linux NVIDIA GPU inference with prebuilt binaries)

### macOS and Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Environment setup

```bash
make setup
source .venv/bin/activate
```

`make setup` runs `uv sync` and `scripts/setup_llamacpp.sh`. The setup script
auto-detects your platform:

| Platform | Binary source | GPU backend |
|---|---|---|
| macOS arm64 | GitHub Releases | Metal |
| Linux x64 + NVIDIA GPU | Hugging Face (`yosoyalguien/llama-binaries-cuda`) | CUDA 13.2 |
| Linux x64 (no GPU) | GitHub Releases | CPU |
| Windows x64 | GitHub Releases | CUDA 12.4 |

### Linux CUDA post-setup

After setup on Linux with NVIDIA GPU, register the CUDA libraries so the
dynamic linker finds them:

```bash
echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf
ldconfig
```

If the binary fails with `libcudart.so.13: cannot open shared object file`,
the host CUDA version differs from the binary's. Install CUDA 13 runtime:

```bash
pip install nvidia-cuda-runtime-cu13 nvidia-cublas-cu13
export LD_LIBRARY_PATH="$(python -c 'import nvidia.cublas; print(nvidia.cublas.__path__[0])')/lib:$(python -c 'import nvidia.cuda_runtime; print(nvidia.cuda_runtime.__path__[0])')/lib:${LD_LIBRARY_PATH}"
```

## Vast.ai quickstart

1. **Template:** llama.cpp with `LLAMA_MODEL="" LLAMA_ARGS=""` to prevent
   preloaded models from consuming VRAM, or any CUDA >= 12.8 template.
2. On first login, free VRAM:
   ```bash
   kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) 2>/dev/null
   ```
3. Clone, setup, download models:
   ```bash
   git clone <repo> /workspace/LLM-Project && cd /workspace/LLM-Project
   make setup && source .venv/bin/activate
   echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf && ldconfig
   python tools/download_models.py
   ```
4. Optionally re-quantize from unquantized sources:
   ```bash
   python tools/quantize_models.py -t TQ4_1S
   ```
5. Run benchmark:
   ```bash
   python benchmarks/test_longbenchv2.py unsloth/gemma-4-E2B-it-GGUF \
       --n-gpu-layers 999 --n-ctx 65536 --flash-attn \
       2>&1 | tee benchmark.log
   ```

## GPU configuration

All GPU flags can be set via CLI or environment variables (`.env`):

| Flag | Env var | Default | Description |
|---|---|---|---|
| `--n-gpu-layers` | `N_GPU_LAYERS` | unset (CPU) | Layers to offload to GPU (`999` = all) |
| `--flash-attn` | `FLASH_ATTN` | unset | Enable flash attention (`-fa on`) |
| `--batch-size` | `N_BATCH` | unset | Prompt processing batch size |
| `--ubatch-size` | `N_UBATCH` | unset | Micro-batch size for GPU pipelining |

A recommended `.env` for GPU inference:

```bash
N_GPU_LAYERS=999
FLASH_ATTN=1
N_BATCH=2048
N_UBATCH=512
```

## Models and storage

Total download size for all 4 models (BF16/F32 + Q8_0 + TQ4_1S): **~97 GB**.

| Model | BF16/F32 | Q8_0 | TQ4_1S |
|---|---|---|---|
| gemma-4-E2B (2B) | 4 GB | 2 GB | 1 GB |
| gemma-4-E4B (4B) | 8 GB | 4 GB | 2 GB |
| Llama-3.1-8B (8B) | 32 GB (F32) | 8 GB | 4 GB |
| Qwen3.5-9B (9B) | 18 GB | 9 GB | 4.5 GB |

The Llama F32 alone is 32 GB. If storage is limited, skip the F32 and
quantize from Q8_0 for that model only (minimal quality loss for the extra
quantization hop).

## Quantization pipeline

Quantization should always start from the highest-precision GGUF available
in each model folder. `tools/quantize_models.py` automatically picks the best
source (F32 > BF16 > F16 > Q8_0) and skips `--allow-requantize` when the
source is unquantized.

**Do not quantize from Q8_0 to TQ4_1S.** The accumulated quantization error
causes model collapse (token loops: `Atha Atha Atha...`). Always use
BF16 or F32 as the source.

```bash
python tools/download_models.py   # downloads BF16/F32 + Q8_0 + TQ4_1S
python tools/quantize_models.py -t TQ4_1S   # quantizes from BF16/F32
python tools/upload_models.py     # uploads to Hugging Face
```

## TurboQuant

This repository uses [llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant),
a fork of llama.cpp that integrates the **TurboQuant+** codec stack: Walsh-Hadamard
rotated polar quantization, attention-gated sparse dequantization, and layer-aware
V compression policies.

TurboQuant operates on two independent domains:

- **Weight quantization** (offline) -- compresses model weights with `llama-quantize`.
- **KV cache quantization** (runtime) -- compresses the attention KV cache via
  `--cache-type-k` and `--cache-type-v`.

### Weight quantization types (offline)

| Type | Approx. bits | Notes |
|---|---|---|
| `TQ1_0` | ~1.0 | Most aggressive |
| `TQ2_0` | ~2.0 | Aggressive |
| `TQ3_1S` | ~3.5 | ~3.5-bit; ~1-2 PPL bump |
| `TQ4_1S` | ~4.5 | Recommended default; `dp4a` accelerated |
| `Q4_K_M` | ~4.8 | Standard llama.cpp baseline |
| `Q4_K_S` | ~4.5 | Standard llama.cpp baseline |
| `Q5_K_M` | ~5.5 | Standard llama.cpp baseline |
| `Q8_0` | ~8.0 | Standard llama.cpp baseline |

### KV cache quantization types (runtime)

| Type | Approx. bits | Domain | Notes |
|---|---|---|---|
| `f16` | 16.0 | K/V | No compression; safest baseline |
| `q8_0` | ~8.0 | K/V | Standard 8-bit; near-lossless |
| `turbo4` | ~4.5 | V | Lightest turbo tier |
| `turbo3` | ~3.5 | V | ~4.6x compression; recommended default |
| `turbo2` | ~2.0 | V | Most aggressive |

### Recommended KV configurations

| Step | K | V | When |
|---|---|---|---|
| 1. Safest | `f16` | `turbo4` | First contact with any model |
| 2. Conservative | `q8_0` | `turbo4` | Verified safe, want memory win |
| 3. Default | `q8_0` | `turbo3` | Most workloads |
| 4. Aggressive | `q8_0` | `turbo2` | Memory-bound, after validating step 3 |

## Known limitations

- **Blackwell (RTX 5090, sm_120):** 3 KV combinations crash with
  `fattn.cu:469` during slot initialization (`f16:q8_0`, `f16:turbo2`,
  `q8_0:f16`). Use `--no-warmup` (enabled by default) and exclude these
  pairs via `--cache-pairs`. This is a bug in the upstream fork.
- **TQ4_1S from Q8_0 is broken.** Always quantize from BF16 or F32.
- **No official Linux CUDA binary.** The upstream release only provides
  CPU binaries for Linux. Prebuilt CUDA binaries are hosted at
  `yosoyalguien/llama-binaries-cuda` (compiled from the fork with
  `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90;120"`).

## Evaluation workflow

1. `./scripts/setup_llamacpp.sh` -- fetch TurboQuant binaries.
2. `python tools/download_models.py` -- fetch GGUF files.
3. `python tools/quantize_models.py -t TQ4_1S` -- produce quantized variants.
4. `python -m backend.evaluate <model> [--cache-configs k:v,k:v]` -- basic benchmark.
5. `python benchmarks/test_longbenchv2.py <model> --n-gpu-layers 999 --flash-attn` -- LongBench V2.

## Cleaning up

```bash
make clean          # removes virtual environment
make clean-bin      # removes bin/
make clean-models   # removes models/
```
