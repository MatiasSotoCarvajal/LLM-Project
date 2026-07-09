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
├── main.ipynb          Benchmark harness (load, measure, save results)
├── prepare_download.py Helper to fetch the LongBench-v2 dataset
├── pyproject.toml      Project metadata and dependencies
├── uv.lock             Reproducible dependency lockfile (uv)
├── Makefile            Environment setup shortcuts
└── README.md
```

## Requirements

- Python 3.10 or 3.11 (3.12 is not tested with all CUDA wheels)
- [uv](https://github.com/astral-sh/uv) - install it first if you do not have it:

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- A CUDA-capable GPU is strongly recommended for running the benchmarks
- Hugging Face account (run `huggingface-cli login` for gated models/datasets)

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

1. Authenticate with Hugging Face (only needed once, for gated resources):

   ```bash
   huggingface-cli login
   ```

2. Download the evaluation dataset:

   ```bash
   python prepare_download.py
   ```

3. Launch the benchmark notebook:

   ```bash
   make notebook
   # or
   jupyter notebook main.ipynb
   ```

Results are written to CSV files (see `save_results_to_csv` in `main.ipynb`).

## Cleaning up

```bash
make clean    # removes the virtual environment
```