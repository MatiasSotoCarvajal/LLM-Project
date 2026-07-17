"""
lm-evaluation-harness (EleutherAI) evaluation for quantized GGUF models.

For each (model, weight_quant, KV cache config) combination, this script:
1. Resolves the correct GGUF file for the given weight quantization
2. Starts a llama-server with the specified K/V cache types
3. Runs one or more lm-eval tasks (default: MMLU + ARC-Challenge + GSM8K)
   against the running server using lm-eval's ``gguf`` backend, which talks
   to llama.cpp's ``/completion`` endpoint -- so accuracy is measured through
   the exact same weight + KV-cache config under test.
4. Records per-(config, task) metric rows and resource usage, and writes:
   - Aggregated rows (results/lm_eval.csv), one row per (config, task)
   - Full nested lm-eval results (results/lm_eval_summary.json)

NOTE ON KV CACHE: standard lm-eval multiple-choice tasks (MMLU, ARC,
HellaSwag) are short-prompt loglikelihood scoring, so they primarily stress
*weight* quantization. KV-cache / turbo compression effects are best measured
on long-context tasks (use test_longbenchv2.py for those). This harness adds
the standardized-accuracy dimension across the same config sweep.

Usage:
    python benchmarks/test_lm_eval.py <model_id...> [options]

Example (small/fast smoke run):
    python benchmarks/test_lm_eval.py \
        unsloth/gemma-4-E4B-it-GGUF \
        --tasks arc_easy --limit 50 --num-fewshot 0

Example (default MMLU-focused sweep, capped for tractability):
    python benchmarks/test_lm_eval.py \
        unsloth/Qwen3.5-9B-GGUF \
        --weight-quants Q8_0 TQ4_1S \
        --cache-pairs q8_0:turbo3 \
        --limit 100
"""

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

try:
    from lm_eval import simple_evaluate
except ImportError:
    sys.exit(
        "lm-eval required. Install with: uv pip install lm-eval  "
        "(or: uv sync, since it is now a project dependency)"
    )

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BENCH_DIR))

from backend.config import (  # noqa: E402
    RESULTS_DIR,
    N_GPU_LAYERS,
    FLASH_ATTN,
    N_BATCH,
    N_UBATCH,
)
from backend.llama_server import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_PORT,
    STARTUP_TIMEOUT,
    run,
    stop_server,
)
from backend.evaluate import build_server_args  # noqa: E402

# Reuse the model-resolution, KV-pair, log-capture, and resource helpers that
# the LongBench harness already defines, so both benchmarks behave identically.
from test_longbenchv2 import (  # noqa: E402
    PAIRS,
    WEIGHT_QUANTIZATIONS,
    LogCollector,
    resolve_model_path,
    wait_for_server_safe,
    parse_kv_cache_mib,
    read_rss_gb,
    read_vram_mb,
    parse_cache_pairs,
    format_default_pairs,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_TASKS = ["mmlu", "arc_challenge", "gsm8k"]
DEFAULT_NUM_FEWSHOT = 5
# MMLU 5-shot prompts fit comfortably in 4k; keep n_ctx small so KV allocation
# and server startup stay fast across the sweep (LongBench needs the big ctx).
DEFAULT_N_CTX = 4096
DEFAULT_LIMIT: int | None = None
DEFAULT_SEED = 42

# lm-eval writes to its own folder so it never mixes with the LongBench CSVs.
DEFAULT_OUT_DIR = RESULTS_DIR.parent / "results_lm_eval"

_is_macos = platform.system() == "Darwin"

# Metric-name preference when collapsing a task's metric dict to one headline
# number for the CSV. First match wins.
_METRIC_PRIORITY = [
    "acc_norm,none",
    "acc,none",
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "exact_match,none",
    "f1,none",
]


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------
def select_primary_metric(metrics: dict) -> tuple[str, float | None]:
    """Pick a single headline (name, value) from an lm-eval task metric dict.

    lm-eval returns e.g. {"acc,none": 0.71, "acc_stderr,none": 0.01,
    "alias": "mmlu"}. We drop alias/stderr keys and prefer the priority list,
    falling back to the first remaining scalar metric.
    """
    for key in _METRIC_PRIORITY:
        if key in metrics and isinstance(metrics[key], (int, float)):
            return key, float(metrics[key])

    for key, val in metrics.items():
        if key == "alias" or "_stderr" in key:
            continue
        if isinstance(val, (int, float)):
            return key, float(val)

    return "", None


def clean_metrics(metrics: dict) -> dict:
    """Return only scalar metric entries (drop alias, keep stderr for context)."""
    return {
        k: v
        for k, v in metrics.items()
        if k != "alias" and isinstance(v, (int, float))
    }


# ---------------------------------------------------------------------------
# Single-config evaluation
# ---------------------------------------------------------------------------
ROW_FIELDNAMES = [
    "model", "weight_quant", "cache_type_k", "cache_type_v", "n_ctx",
    "num_fewshot", "limit", "task", "primary_metric", "primary_value",
    "all_metrics", "n_samples",
    "rss_gb_after_load", "rss_gb_peak", "vram_mb_after_load", "vram_mb_peak",
    "kv_cache_mib", "eval_time_s", "seed", "notes",
]


def evaluate_lm_eval(
    model_id: str,
    weight_quant: str,
    cache_pairs: list[dict[str, str]],
    n_ctx: int,
    tasks: list[str],
    num_fewshot: int | None,
    limit: int | None,
    seed: int,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    n_gpu_layers: int | None = None,
    flash_attn: bool = False,
    n_batch: int | None = None,
    n_ubatch: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Returns ``(csv_rows, summary_entries)``.

    ``csv_rows`` has one flattened row per (cache config, task). ``summary_entries``
    holds the full nested lm-eval ``results`` block per cache config for the JSON.
    """
    csv_rows: list[dict] = []
    summary_entries: list[dict] = []

    try:
        model_path = resolve_model_path(model_id, weight_quant)
    except FileNotFoundError as exc:
        print(f"Skipping {model_id}/{weight_quant}: {exc}", file=sys.stderr)
        return csv_rows, summary_entries

    print(f"  Model file: {model_path}", flush=True)

    for pair in cache_pairs:
        k_type = pair["K"]
        v_type = pair["V"]

        header = (
            f"Model: {model_id}  Weight: {weight_quant}  "
            f"KV: k={k_type} v={v_type}  Tasks: {tasks}"
        )
        print(f"\n{'=' * len(header)}", flush=True)
        print(header, flush=True)
        print(f"{'=' * len(header)}", flush=True)

        base_row = {
            "model": model_id,
            "weight_quant": weight_quant,
            "cache_type_k": k_type,
            "cache_type_v": v_type,
            "n_ctx": n_ctx,
            "num_fewshot": num_fewshot,
            "limit": limit,
            "seed": seed,
            "rss_gb_after_load": None,
            "rss_gb_peak": None,
            "vram_mb_after_load": None,
            "vram_mb_peak": None,
            "kv_cache_mib": None,
        }

        server_args = build_server_args(
            n_ctx=n_ctx,
            n_parallel=1,
            no_mmap=True,
            extra=None,
            n_gpu_layers=n_gpu_layers,
            flash_attn=flash_attn,
            n_batch=n_batch,
            n_ubatch=n_ubatch,
            no_warmup=True,
        )
        proc, actual_port = run(
            model_id,
            host=host,
            port=port,
            cache_type_k=k_type,
            cache_type_v=v_type,
            extra_args=server_args,
            capture_output=True,
            model_path=model_path,
        )
        logs = LogCollector(proc)

        results = None
        error_note = ""
        eval_elapsed = None
        rss_after_load = vram_after_load = None
        peak_rss = peak_vram = 0.0

        try:
            wait_for_server_safe(
                proc, host, actual_port, STARTUP_TIMEOUT, log_collector=logs
            )

            rss_after_load = read_rss_gb(proc.pid)
            peak_rss = rss_after_load or 0.0
            vram_after_load = read_vram_mb()
            peak_vram = vram_after_load or 0.0

            base_url = f"http://{host}:{actual_port}"
            print(f"  Running lm-eval against {base_url} ...", flush=True)

            eval_start = time.monotonic()
            results = simple_evaluate(
                model="gguf",
                model_args={"base_url": base_url, "max_length": n_ctx},
                tasks=tasks,
                num_fewshot=num_fewshot,
                limit=limit,
                bootstrap_iters=0,  # skip stderr bootstrapping -> faster
                random_seed=seed,
                numpy_random_seed=seed,
                torch_random_seed=seed,
                fewshot_random_seed=seed,
            )
            eval_elapsed = round(time.monotonic() - eval_start, 2)

            # Sample resources once more after the run for a peak estimate.
            r = read_rss_gb(proc.pid)
            if r is not None:
                peak_rss = max(peak_rss, r)
            v = read_vram_mb()
            if v is not None:
                peak_vram = max(peak_vram, v)

        except Exception as exc:
            error_note = f"error: {exc}"
            print(f"  ERROR: {exc}", file=sys.stderr, flush=True)
        finally:
            stop_server(proc)
            time.sleep(0.5)

        kv_mib = parse_kv_cache_mib(logs.text())
        base_row["rss_gb_after_load"] = (
            round(rss_after_load, 4) if rss_after_load is not None else None
        )
        base_row["rss_gb_peak"] = round(peak_rss, 4) if peak_rss else None
        base_row["vram_mb_after_load"] = vram_after_load
        base_row["vram_mb_peak"] = round(peak_vram, 4) if peak_vram else None
        base_row["kv_cache_mib"] = round(kv_mib, 2) if kv_mib is not None else None

        if results is None:
            row = {
                **base_row,
                "task": ",".join(tasks),
                "primary_metric": "",
                "primary_value": None,
                "all_metrics": "{}",
                "n_samples": None,
                "eval_time_s": eval_elapsed,
                "notes": error_note or "no results",
            }
            csv_rows.append({k: row.get(k) for k in ROW_FIELDNAMES})
            continue

        task_results = results.get("results", {})
        n_samples_map = {
            t: (results.get("n-samples", {}).get(t, {}) or {}).get("effective")
            for t in task_results
        }

        summary_entries.append({
            "model": model_id,
            "weight_quant": weight_quant,
            "cache_type_k": k_type,
            "cache_type_v": v_type,
            "n_ctx": n_ctx,
            "num_fewshot": num_fewshot,
            "limit": limit,
            "results": task_results,
            "configs": results.get("configs", {}),
            "resources": {
                "rss_gb_after_load": base_row["rss_gb_after_load"],
                "rss_gb_peak": base_row["rss_gb_peak"],
                "vram_mb_peak": base_row["vram_mb_peak"],
                "kv_cache_mib": base_row["kv_cache_mib"],
            },
            "eval_time_s": eval_elapsed,
        })

        print(f"\n  Results ({eval_elapsed}s):", flush=True)
        for task_name, metrics in sorted(task_results.items()):
            primary_name, primary_value = select_primary_metric(metrics)
            row = {
                **base_row,
                "task": task_name,
                "primary_metric": primary_name,
                "primary_value": (
                    round(primary_value, 4) if primary_value is not None else None
                ),
                "all_metrics": json.dumps(clean_metrics(metrics)),
                "n_samples": n_samples_map.get(task_name),
                "eval_time_s": eval_elapsed,
                "notes": error_note,
            }
            csv_rows.append({k: row.get(k) for k in ROW_FIELDNAMES})
            print(
                f"    {task_name:28s} {primary_name:28s} "
                f"{'' if primary_value is None else round(primary_value, 4)}",
                flush=True,
            )

    return csv_rows, summary_entries


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save_results_to_csv(rows: list[dict], file_path: Path) -> None:
    if not rows:
        return
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = file_path.is_file()
    with open(file_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nResults saved to {file_path}")


def save_summary_json(entries: list[dict], config: dict, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, mode="w") as f:
        json.dump({"config": config, "runs": entries}, f, indent=2, default=str)
    print(f"Summary JSON saved to {file_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate GGUF models with EleutherAI lm-evaluation-harness across "
            "KV cache and weight quantization configurations."
        )
    )
    parser.add_argument("models", nargs="+", help="Model IDs resolvable under ./Models.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"llama-server port. Default: {DEFAULT_PORT} (0 = auto-detect free port).",
    )
    parser.add_argument(
        "--cache-pairs", type=parse_cache_pairs, default=PAIRS,
        help=(
            "Comma-separated K:V pairs (e.g. 'f16:f16,q8_0:turbo3'). "
            f"Default: {format_default_pairs()}"
        ),
    )
    parser.add_argument(
        "--weight-quants", nargs="+", default=WEIGHT_QUANTIZATIONS,
        help=f"Weight quantization types. Default: {WEIGHT_QUANTIZATIONS}",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=DEFAULT_TASKS,
        help=f"lm-eval task names. Default: {DEFAULT_TASKS}",
    )
    parser.add_argument(
        "--num-fewshot", type=int, default=DEFAULT_NUM_FEWSHOT,
        help=(
            "Few-shot examples, applied to all tasks. Default: "
            f"{DEFAULT_NUM_FEWSHOT}. Pass -1 to use each task's own default."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help=(
            "Max examples per task (lm-eval --limit). Strongly recommended for "
            "sweeps: the gguf backend issues one HTTP request per sample, so "
            "full MMLU (~14k Q) per config is very slow. Default: no limit."
        ),
    )
    parser.add_argument(
        "--n-ctx", type=int, default=DEFAULT_N_CTX,
        help=f"Server context length. Default: {DEFAULT_N_CTX}",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed. Default: {DEFAULT_SEED}")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help=f"Output directory. Default: {DEFAULT_OUT_DIR}")
    parser.add_argument("--n-gpu-layers", type=int, default=N_GPU_LAYERS,
                        help="Layers to offload to GPU (-ngl). Default: N_GPU_LAYERS env.")
    parser.add_argument("--flash-attn", action="store_true", default=FLASH_ATTN,
                        help="Enable flash attention (-fa). Default: FLASH_ATTN env.")
    parser.add_argument("--batch-size", type=int, default=N_BATCH,
                        help="Logical batch size (-b). Default: N_BATCH env.")
    parser.add_argument("--ubatch-size", type=int, default=N_UBATCH,
                        help="Micro-batch size (-ub). Default: N_UBATCH env.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    num_fewshot = None if args.num_fewshot is not None and args.num_fewshot < 0 else args.num_fewshot

    n_configs = len(args.models) * len(args.weight_quants) * len(args.cache_pairs)
    print(
        f"Sweep: {len(args.models)} model(s) x {len(args.weight_quants)} weight "
        f"quant(s) x {len(args.cache_pairs)} KV pair(s) = {n_configs} config(s)",
        flush=True,
    )
    print(f"  Tasks: {args.tasks}  num_fewshot={num_fewshot}  limit={args.limit}", flush=True)
    print(f"  Weight quants: {args.weight_quants}", flush=True)
    print(
        f"  KV pairs: {format_default_pairs() if args.cache_pairs == PAIRS else args.cache_pairs}",
        flush=True,
    )
    if args.limit is None:
        print(
            "  WARNING: no --limit set. The gguf backend sends one request per "
            "sample; full task suites per config can take hours. Consider "
            "--limit 100 for a tractable sweep.",
            file=sys.stderr, flush=True,
        )

    all_rows: list[dict] = []
    all_summary: list[dict] = []

    for model_id in args.models:
        for wq in args.weight_quants:
            rows, summary = evaluate_lm_eval(
                model_id=model_id,
                weight_quant=wq,
                cache_pairs=args.cache_pairs,
                n_ctx=args.n_ctx,
                tasks=args.tasks,
                num_fewshot=num_fewshot,
                limit=args.limit,
                seed=args.seed,
                host=args.host,
                port=args.port,
                n_gpu_layers=args.n_gpu_layers,
                flash_attn=args.flash_attn,
                n_batch=args.batch_size,
                n_ubatch=args.ubatch_size,
            )
            all_rows.extend(rows)
            all_summary.extend(summary)

    if not all_rows:
        print("No results produced.", file=sys.stderr)
        raise SystemExit(1)

    save_results_to_csv(all_rows, out_dir / "lm_eval.csv")
    save_summary_json(
        all_summary,
        {
            "tasks": args.tasks,
            "num_fewshot": num_fewshot,
            "limit": args.limit,
            "n_ctx": args.n_ctx,
            "seed": args.seed,
            "weight_quants": args.weight_quants,
        },
        out_dir / "lm_eval_summary.json",
    )


if __name__ == "__main__":
    main()
