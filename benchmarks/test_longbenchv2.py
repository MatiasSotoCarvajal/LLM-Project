"""
LongBench V2 evaluation harness for quantized GGUF models.

For each (model, weight_quant, KV cache config) combination, this script:
1. Resolves the correct GGUF file for the given weight quantization
2. Starts a llama-server with the specified K/V cache types
3. Runs a stratified sample of LongBench V2 multiple-choice questions,
   ordered by the dataset's ``length`` field (short -> medium -> long) so
   smaller examples are tried first within the context window.
4. Measures per-example accuracy, TTFT, decode speed, and outputs:
   - Aggregated config-level rows (results/longbench.csv)
   - Per-example detail rows (results/longbench_examples.csv)
   - Summary JSON (results/longbench_summary.json)

Usage:
    python benchmarks/test_longbenchv2.py <model_id...> [options]

Example:
    python benchmarks/test_longbenchv2.py \
        unsloth/gemma-4-E4B-it-GGUF \
        unsloth/gemma-4-E2B-it-GGUF \
        unsloth/Qwen3.5-9B-GGUF \
        bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
        --n-ctx 32768 --n-examples 5
"""

import argparse
import csv
import json
import platform
import random
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import requests

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("datasets library required. Install with: pip install datasets")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import RESULTS_DIR, MODELS_DIR  # noqa: E402
from backend.llama_server import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_PORT,
    STARTUP_TIMEOUT,
    run,
    stop_server,
    wait_for_server,
)

# ---------------------------------------------------------------------------
# Default sweep configuration
# ---------------------------------------------------------------------------
WEIGHT_QUANTIZATIONS = ["Q8_0", "TQ4_1S"]

QUANT_SUFFIX_MAP = {
    "Q8_0": None,
    "TQ1_0": "tq1_0",
    "TQ2_0": "tq2_0",
    "TQ3_1S": "tq3_1s",
    "TQ4_1S": "tq4_1s",
}

KEYS = ["f16", "q8_0", "turbo4"]
VALUES = ["f16", "q8_0", "turbo2", "turbo3", "turbo4"]

_is_macos = platform.system() == "Darwin"
_METAL_UNSUPPORTED_K = {"turbo4"}
_METAL_UNSUPPORTED_V = {"turbo4"}

PAIRS: list[dict[str, str]] = []
for k in KEYS:
    for v in VALUES:
        if _is_macos:
            if k in _METAL_UNSUPPORTED_K or v in _METAL_UNSUPPORTED_V:
                print(
                    f"  [skip] K={k} V={v} -- not supported on macOS Metal",
                    file=sys.stderr,
                )
                continue
        PAIRS.append({"K": k, "V": v})

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KV_SIZE_PATTERN = re.compile(r"KV\s+self\s+size\s*=\s*([\d.]+)\s*MiB", re.IGNORECASE)

DEFAULT_N_CTX = 65536
DEFAULT_N_EXAMPLES = 5
DEFAULT_N_PREDICT = 128
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 42
REQUEST_TIMEOUT = 600
STARTUP_CHECK_INTERVAL = 1.0

LENGTH_ORDER = {"short": 0, "medium": 1, "long": 2}


# ---------------------------------------------------------------------------
# Utilities (log capture, RSS, KV cache parsing)
# ---------------------------------------------------------------------------
class LogCollector:
    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        self.lines: list[str] = []
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        if self.proc.stdout is None:
            return
        for line in self.proc.stdout:
            self.lines.append(line.rstrip("\n"))

    def text(self) -> str:
        return "\n".join(self.lines)


def parse_kv_cache_mib(log_text: str) -> float | None:
    matches = KV_SIZE_PATTERN.findall(log_text)
    return float(matches[-1]) if matches else None


def read_rss_gb(pid: int) -> float | None:
    status = Path(f"/proc/{pid}/status")
    if not status.exists():
        return None
    for line in status.read_text().splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            return int(parts[1]) / (1024 * 1024)
    return None


# ---------------------------------------------------------------------------
# Model resolution -- ensures the correct GGUF is picked per weight_quant
# ---------------------------------------------------------------------------
def resolve_model_path(model_id: str, weight_quant: str) -> Path:
    folder_name = model_id.replace("/", "__")
    folder = MODELS_DIR / folder_name

    if not folder.is_dir():
        raise FileNotFoundError(f"Model folder not found: {folder}")

    ggufs = sorted(folder.glob("*.gguf"))
    if not ggufs:
        raise FileNotFoundError(f"No .gguf files in {folder}")

    if weight_quant == "Q8_0":
        baselines = [f for f in ggufs if "-tq" not in f.stem.lower()]
        if baselines:
            return baselines[0]
        raise FileNotFoundError(
            f"No baseline GGUF (without -tq suffix) found in {folder}. "
            f"Files present: {[f.name for f in ggufs]}"
        )

    suffix = QUANT_SUFFIX_MAP.get(weight_quant)
    if suffix:
        pattern = f"-{suffix}.gguf".lower()
        matches = [f for f in ggufs if f.name.lower().endswith(pattern)]
        if matches:
            return matches[0]
        raise FileNotFoundError(
            f"No GGUF ending with '{pattern}' found in {folder}. "
            f"Files present: {[f.name for f in ggufs]}"
        )

    raise FileNotFoundError(f"Unknown weight quantization: {weight_quant}")


# ---------------------------------------------------------------------------
# Server startup with process-health monitoring
# ---------------------------------------------------------------------------
def wait_for_server_safe(
    proc: subprocess.Popen,
    host: str,
    port: int,
    timeout: int,
    log_collector: LogCollector | None = None,
) -> None:
    """
    Poll the /health endpoint until the server responds 200 or the process dies.
    If the process exits unexpectedly, print recent log lines and raise immediately
    instead of waiting for the full *timeout*.
    """
    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if proc.poll() is not None:
            log_tail = ""
            if log_collector is not None:
                tail_lines = log_collector.lines[-40:]
                log_tail = "\n".join(tail_lines)
            raise RuntimeError(
                f"Server process exited with rc={proc.returncode} before "
                f"becoming healthy.\nLast log lines:\n{log_tail}"
            )

        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(STARTUP_CHECK_INTERVAL)

    raise TimeoutError(f"Server did not become healthy at {url} within {timeout}s")


# ---------------------------------------------------------------------------
# LongBench V2 data loading -- sorted by length, filtered by context window
# ---------------------------------------------------------------------------
def load_longbench_subset(
    n: int = 50, seed: int = 42, n_ctx: int = DEFAULT_N_CTX
) -> dict[str, list[dict]]:
    """
    Load *n* examples stratified by domain, sorted by the dataset's ``length``
    field (short < medium < long). Examples whose estimated token count exceeds
    *n_ctx* are skipped so we get as many fitting examples as possible.

    Returns a dict with two keys:
      ``fitting`` -- examples that fit in the context window (up to *n*)
      ``all``     -- the full stratified sample before filtering
    """
    dataset = load_dataset("THUDM/LongBench-v2", split="train")
    data = list(dataset)
    rng = random.Random(seed)

    by_domain: dict[str, list[dict]] = {}
    for item in data:
        domain = item.get("domain", "unknown") # type: ignore
        by_domain.setdefault(domain, []).append(item) # type: ignore

    total = len(data)
    sample: list[dict] = []
    for domain, items in by_domain.items():
        n_domain = max(1, round(n * len(items) / total))
        sample.extend(rng.sample(items, min(n_domain, len(items))))

    rng.shuffle(sample)
    if len(sample) > n:
        sample = sample[:n]
    elif len(sample) < n:
        remaining = [x for x in data if x not in sample]
        sample.extend(rng.sample(remaining, min(n - len(sample), len(remaining)))) # type: ignore

    sample.sort(key=lambda ex: LENGTH_ORDER.get(ex.get("length", "long"), 2))

    fitting: list[dict] = []
    for ex in sample:
        est = estimate_tokens(format_prompt(ex))
        if est <= n_ctx:
            fitting.append(ex)
            if len(fitting) >= n:
                break

    return {"fitting": fitting, "all": sample}


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------
def format_prompt(example: dict) -> str:
    return (
        f"{example['context']}\n\n"
        f"Question: {example['question']}\n"
        f"A. {example['choice_A']}\n"
        f"B. {example['choice_B']}\n"
        f"C. {example['choice_C']}\n"
        f"D. {example['choice_D']}\n\n"
        f"Answer:"
    )


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------
ANSWER_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"(?:^|\n)\s*(?:answer|the answer is|choose|option|I choose)\s*:?\s*([ABCD])\b",
        re.IGNORECASE,
    ),
    re.compile(r"([ABCD])\s*(?:\)|\.)", re.IGNORECASE),
]


def parse_answer(output: str) -> str | None:
    cleaned = output.strip()

    for line in reversed(cleaned.splitlines()):
        stripped = line.strip().upper()
        if len(stripped) == 1 and stripped in {"A", "B", "C", "D"}:
            return stripped

    for line in reversed(cleaned.splitlines()):
        stripped = line.strip()
        for ch in ("A", "B", "C", "D"):
            if (
                stripped.startswith(f"{ch} ")
                or stripped.startswith(f"{ch}.")
                or stripped.startswith(f"{ch})")
            ):
                return ch

    for pattern in ANSWER_PATTERNS:
        m = pattern.search(cleaned)
        if m:
            return m.group(1).upper()

    last_match: str | None = None
    for m in re.finditer(r"\b([ABCD])\b", cleaned, re.IGNORECASE):
        last_match = m.group(1).upper()

    return last_match


# ---------------------------------------------------------------------------
# Completion request (streaming, with timing)
# ---------------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    return len(text) // 4


def completion_request(
    host: str, port: int, prompt: str, n_predict: int, temperature: float
) -> dict:
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
        "cache_prompt": False,
        "stream": True,
    }
    url = f"http://{host}:{port}/completion"

    full_content: list[str] = []
    ttft = None
    timings: dict = {}
    start = time.monotonic()
    error_body = ""

    with requests.post(url, json=payload, stream=True, timeout=REQUEST_TIMEOUT) as r:
        try:
            r.raise_for_status()
        except requests.HTTPError:
            error_body = r.text[:500] if r.text else ""
            raise
        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8")
            if line.startswith("data: "):
                line = line[len("data: "):]
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            if chunk.get("content"):
                full_content.append(chunk["content"])
                if ttft is None:
                    ttft = time.monotonic() - start
            if chunk.get("timings"):
                timings = chunk["timings"]

    return {
        "output": "".join(full_content),
        "ttft_s": ttft,
        "prompt_tokens": timings.get("prompt_n"),
        "prompt_per_second": timings.get("prompt_per_second"),
        "decode_tokens_per_second": timings.get("predicted_per_second"),
        "predicted_tokens": timings.get("predicted_n"),
        "error_body": error_body,
    }


# ---------------------------------------------------------------------------
# Single-run evaluation of one (model, weight_quant, K, V cache) config
# ---------------------------------------------------------------------------
EXAMPLE_FIELDNAMES = [
    "model", "weight_quant", "cache_type_k", "cache_type_v", "n_ctx",
    "example_id", "domain", "sub_domain", "difficulty", "length",
    "predicted", "expected", "is_correct",
    "ttft_s", "decode_tokens_per_second", "prompt_tokens", "predicted_tokens",
]


def evaluate_longbench(
    model_id: str,
    weight_quant: str,
    cache_pairs: list[dict[str, str]],
    n_ctx: int,
    examples: list[dict],
    n_sampled: int,
    seed: int,
    n_predict: int = DEFAULT_N_PREDICT,
    temperature: float = DEFAULT_TEMPERATURE,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> tuple[list[dict], list[dict]]:
    """
    Returns ``(agg_rows, detail_rows)`` -- one aggregated row per cache
    config plus one per-example detail row per question evaluated.

    *examples* is the subset of sampled questions that fit in the context
    window. *n_sampled* is the original sample size before filtering.
    """
    agg_rows: list[dict] = []
    detail_rows: list[dict] = []

    try:
        model_path = resolve_model_path(model_id, weight_quant)
    except FileNotFoundError as exc:
        print(f"Skipping {model_id}/{weight_quant}: {exc}", file=sys.stderr)
        return agg_rows, detail_rows

    print(f"  Model file: {model_path}", flush=True)

    for pair in cache_pairs:
        k_type = pair["K"]
        v_type = pair["V"]

        header = (
            f"Model: {model_id}  Weight: {weight_quant}  KV: k={k_type} v={v_type}"
        )
        print(f"\n{'=' * len(header)}", flush=True)
        print(header, flush=True)
        print(f"{'=' * len(header)}", flush=True)

        row: dict = {
            "model": model_id,
            "weight_quant": weight_quant,
            "cache_type_k": k_type,
            "cache_type_v": v_type,
            "n_ctx": n_ctx,
            "n_examples_total": n_sampled,
            "n_examples_fitted": 0,
            "n_examples_skipped": 0,
            "n_examples_error": 0,
            "n_correct": 0,
            "accuracy": None,
            "accuracy_easy": None,
            "accuracy_hard": None,
            "accuracy_by_domain": None,
            "avg_ttft_s": None,
            "avg_decode_tps": None,
            "avg_prompt_tokens": None,
            "rss_gb_after_load": None,
            "rss_gb_peak": None,
            "kv_cache_mib": None,
            "total_time_s": None,
            "sample_ids": [ex["_id"] for ex in examples],
            "seed": seed,
            "notes": "",
        }

        server_args = ["-c", str(n_ctx), "-np", "1", "--no-mmap"]
        proc, actual_port = run(
            model_id,  # only used for logging when model_path is passed
            host=host,
            port=port,
            cache_type_k=k_type,
            cache_type_v=v_type,
            extra_args=server_args,
            capture_output=True,
            model_path=model_path,
        )
        logs = LogCollector(proc)

        try:
            wait_for_server_safe(proc, host, actual_port, STARTUP_TIMEOUT, log_collector=logs)

            rss_after_load = read_rss_gb(proc.pid)
            row["rss_gb_after_load"] = rss_after_load
            peak_rss = rss_after_load or 0.0

            total_start = time.monotonic()

            ttft_list: list[float] = []
            decode_tps_list: list[float] = []
            prompt_tokens_list: list[float] = []

            correct_counter: Counter = Counter()
            total_counter: Counter = Counter()
            diff_correct: Counter = Counter()
            diff_total: Counter = Counter()

            n_skipped = 0
            n_error = 0
            n_fitted = 0
            skip_ids: list[str] = []
            error_ids: list[str] = []

            for i, example in enumerate(examples):
                prompt = format_prompt(example)
                ex_id = example["_id"]

                est_tokens = estimate_tokens(prompt)
                if est_tokens > n_ctx:
                    n_skipped += 1
                    skip_ids.append(ex_id)
                    print(
                        f"  [{i + 1:3d}/{len(examples)}] SKIP  id={ex_id}  "
                        f"est_tokens={est_tokens} > n_ctx={n_ctx}",
                        flush=True,
                    )
                    continue

                try:
                    result = completion_request(
                        host, actual_port, prompt, n_predict, temperature
                    )
                except requests.HTTPError as e:
                    status_code = e.response.status_code if e.response else "?"
                    body = e.response.text[:200] if e.response else ""
                    n_skipped += 1
                    skip_ids.append(ex_id)
                    print(
                        f"  [{i + 1:3d}/{len(examples)}] SKIP  id={ex_id}  "
                        f"HTTP {status_code}: {body.strip()}",
                        flush=True,
                    )
                    continue
                except Exception as e:
                    n_error += 1
                    error_ids.append(ex_id)
                    print(
                        f"  [{i + 1:3d}/{len(examples)}] ERR   id={ex_id}  {e}",
                        flush=True,
                    )
                    if proc.poll() is not None:
                        raise RuntimeError(
                            f"Server process died (rc={proc.returncode}). "
                            f"Aborting remaining examples."
                        ) from e
                    continue

                n_fitted += 1

                output = result["output"]
                predicted = parse_answer(output)
                expected = example["answer"].strip().upper()
                is_correct = predicted == expected

                domain = example.get("domain", "unknown")
                difficulty = example.get("difficulty", "unknown")

                total_counter[domain] += 1
                diff_total[difficulty] += 1
                if is_correct:
                    correct_counter[domain] += 1
                    diff_correct[difficulty] += 1

                if result["ttft_s"] is not None:
                    ttft_list.append(result["ttft_s"])
                if result["decode_tokens_per_second"] is not None:
                    decode_tps_list.append(result["decode_tokens_per_second"])
                if result["prompt_tokens"] is not None:
                    prompt_tokens_list.append(result["prompt_tokens"])

                detail_rows.append({
                    "model": model_id,
                    "weight_quant": weight_quant,
                    "cache_type_k": k_type,
                    "cache_type_v": v_type,
                    "n_ctx": n_ctx,
                    "example_id": ex_id,
                    "domain": domain,
                    "sub_domain": example.get("sub_domain", ""),
                    "difficulty": difficulty,
                    "length": example.get("length", ""),
                    "predicted": predicted or "?",
                    "expected": expected,
                    "is_correct": is_correct,
                    "ttft_s": result["ttft_s"],
                    "decode_tokens_per_second": result["decode_tokens_per_second"],
                    "prompt_tokens": result["prompt_tokens"],
                    "predicted_tokens": result["predicted_tokens"],
                })

                status = "OK" if is_correct else "FAIL"
                pred_str = predicted or "?"
                exp_str = expected
                ttft_str = (
                    f"  ttft={result['ttft_s']:.2f}s" if result["ttft_s"] else ""
                )
                print(
                    f"  [{i + 1:3d}/{len(examples)}] {status}  "
                    f"pred={pred_str}  expected={exp_str}{ttft_str}",
                    flush=True,
                )

                sample = read_rss_gb(proc.pid)
                if sample is not None:
                    peak_rss = max(peak_rss, sample)

            total_elapsed = time.monotonic() - total_start
            row["total_time_s"] = round(total_elapsed, 2)

            n_correct = sum(correct_counter.values())

            row["n_examples_fitted"] = n_fitted
            row["n_examples_skipped"] = n_skipped
            row["n_examples_error"] = n_error
            row["n_correct"] = n_correct
            row["accuracy"] = (
                round(n_correct / n_fitted, 4) if n_fitted > 0 else None
            )

            acc_by_domain = {
                d: round(correct_counter[d] / total_counter[d], 4)
                for d in total_counter
            }
            row["accuracy_by_domain"] = json.dumps(acc_by_domain)

            if diff_total.get("easy"):
                row["accuracy_easy"] = round(
                    diff_correct["easy"] / diff_total["easy"], 4
                )
            if diff_total.get("hard"):
                row["accuracy_hard"] = round(
                    diff_correct["hard"] / diff_total["hard"], 4
                )

            if ttft_list:
                row["avg_ttft_s"] = round(sum(ttft_list) / len(ttft_list), 4)
            if decode_tps_list:
                row["avg_decode_tps"] = round(
                    sum(decode_tps_list) / len(decode_tps_list), 2
                )
            if prompt_tokens_list:
                row["avg_prompt_tokens"] = round(
                    sum(prompt_tokens_list) / len(prompt_tokens_list), 0
                )

            row["rss_gb_peak"] = (
                round(peak_rss, 4)
                if isinstance(peak_rss, (int, float))
                else peak_rss
            )

            kv_mib = parse_kv_cache_mib(logs.text())
            row["kv_cache_mib"] = round(kv_mib, 2) if kv_mib is not None else None

            notes_parts = []
            if skip_ids:
                notes_parts.append(f"skipped: {skip_ids}")
            if error_ids:
                notes_parts.append(f"errors: {error_ids}")
            if notes_parts:
                row["notes"] = "; ".join(notes_parts)

            print(
                f"\n  Accuracy: {row['accuracy']} ({n_correct}/{n_fitted} fitted)  "
                f"Skipped: {n_skipped}  Errors: {n_error}  "
                f"Avg TTFT: {row['avg_ttft_s']}s  "
                f"Avg decode: {row['avg_decode_tps']} tok/s  "
                f"Total time: {total_elapsed:.1f}s",
                flush=True,
            )

        except Exception as exc:
            row["notes"] = f"error: {exc}"
            print(f"  ERROR: {exc}", file=sys.stderr, flush=True)
        finally:
            stop_server(proc)
            # Small pause so the OS can release the port before the next config
            time.sleep(0.5)

        agg_rows.append(row)

    return agg_rows, detail_rows


# ---------------------------------------------------------------------------
# CSV / JSON output
# ---------------------------------------------------------------------------
def save_results_to_csv(rows: list[dict], file_path: Path) -> None:
    if not rows:
        return
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = file_path.is_file()
    fieldnames = list(rows[0].keys())

    with open(file_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Results saved to {file_path}")


def save_example_details(rows: list[dict], file_path: Path) -> None:
    """Write per-example detail rows (always overwrite, not append)."""
    if not rows:
        return
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXAMPLE_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Per-example details saved to {file_path}")


def save_summary_json(
    agg_rows: list[dict], config: dict, file_path: Path
) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "config": config,
        "results": [
            {
                k: v
                for k, v in r.items()
                if k not in ("notes", "sample_ids")
            }
            for r in agg_rows
        ],
    }

    with open(file_path, mode="w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary JSON saved to {file_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_cache_pairs(raw: str) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError(
                f"Expected K:V format, got '{item}'"
            )
        k, v = item.split(":", 1)
        pairs.append({"K": k.strip(), "V": v.strip()})
    return pairs


def format_default_pairs() -> str:
    return ",".join(f"{p['K']}:{p['V']}" for p in PAIRS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate GGUF models on LongBench V2 across KV cache and weight "
            "quantization configurations."
        )
    )
    parser.add_argument(
        "models",
        nargs="+",
        help="Model IDs resolvable under ./Models.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port for llama-server. Default: {DEFAULT_PORT} (0 = auto-detect free port).",
    )
    parser.add_argument(
        "--cache-pairs",
        type=parse_cache_pairs,
        default=PAIRS,
        help=(
            "Comma-separated K:V pairs (e.g. 'f16:f16,q8_0:turbo3'). "
            f"Default: {format_default_pairs()}"
        ),
    )
    parser.add_argument(
        "--weight-quants",
        nargs="+",
        default=WEIGHT_QUANTIZATIONS,
        help=f"Weight quantization types. Default: {WEIGHT_QUANTIZATIONS}",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=DEFAULT_N_CTX,
        help=f"Context length for the server. Default: {DEFAULT_N_CTX}",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=DEFAULT_N_EXAMPLES,
        help=(
            "Number of LongBench V2 examples to sample. "
            f"Default: {DEFAULT_N_EXAMPLES}"
        ),
    )
    parser.add_argument(
        "--n-predict",
        type=int,
        default=DEFAULT_N_PREDICT,
        help=f"Max output tokens per example. Default: {DEFAULT_N_PREDICT}",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature. Default: {DEFAULT_TEMPERATURE}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for example sampling. Default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--out-dir",
        default=str(RESULTS_DIR),
        help=f"Directory for output files. Default: {RESULTS_DIR}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    print(
        f"Loading LongBench V2 subset "
        f"(n={args.n_examples}, seed={args.seed}, n_ctx={args.n_ctx})..."
    )
    subsets = load_longbench_subset(
        n=args.n_examples, seed=args.seed, n_ctx=args.n_ctx
    )
    examples = subsets["fitting"]

    if not examples:
        sampled_len = len(subsets["all"])
        print(
            f"WARNING: 0/{sampled_len} sampled examples fit within "
            f"n_ctx={args.n_ctx}. Try increasing --n-ctx or --n-examples.",
            file=sys.stderr,
            flush=True,
        )
    else:
        by_len_ex = {}
        for ex in examples:
            by_len_ex.setdefault(ex.get("length", "?"), 0)
            by_len_ex[ex.get("length", "?")] += 1
        by_dom_ex = {}
        for ex in examples:
            by_dom_ex.setdefault(ex.get("domain", "?"), 0)
            by_dom_ex[ex.get("domain", "?")] += 1

        print(
            f"  {len(examples)}/{len(subsets['all'])} sampled examples fit "
            f"in n_ctx={args.n_ctx}",
            flush=True,
        )
        print(f"    by length: {dict(by_len_ex)}", flush=True)
        print(f"    by domain: {dict(by_dom_ex)}", flush=True)
        n_configs = (
            len(args.models) * len(args.weight_quants) * len(args.cache_pairs)
        )
        ppl_print_models = args.models if len(args.models) <= 3 else args.models[:3] + [f"... (+{len(args.models) - 3} more)"]
        print(
            f"\n  Sweep: {len(args.models)} model(s) x {len(args.weight_quants)} "
            f"weight quant(s) x {len(args.cache_pairs)} KV pair(s) "
            f"= {n_configs} config(s)",
            flush=True,
        )
        print(f"  Models: {ppl_print_models}", flush=True)
        print(f"  Weight quants: {args.weight_quants}", flush=True)
        print(
            f"  KV pairs: {format_default_pairs() if args.cache_pairs == PAIRS else args.cache_pairs}",
            flush=True,
        )
        print(flush=True)

    all_agg: list[dict] = []
    all_detail: list[dict] = []

    for model_id in args.models:
        for wq in args.weight_quants:
            agg_rows, detail_rows = evaluate_longbench(
                model_id=model_id,
                weight_quant=wq,
                cache_pairs=args.cache_pairs,
                n_ctx=args.n_ctx,
                examples=examples,
                n_sampled=len(subsets["all"]),
                seed=args.seed,
                n_predict=args.n_predict,
                temperature=args.temperature,
                host=args.host,
                port=args.port,
            )
            all_agg.extend(agg_rows)
            all_detail.extend(detail_rows)

    if not all_agg:
        print("No results produced.", file=sys.stderr)
        raise SystemExit(1)

    save_results_to_csv(all_agg, out_dir / "longbench.csv")
    save_example_details(all_detail, out_dir / "longbench_examples.csv")
    save_summary_json(
        all_agg,
        {
            "n_examples_sampled": len(subsets["all"]),
            "n_examples_fitting": len(examples),
            "seed": args.seed,
            "n_ctx": args.n_ctx,
            "n_predict": args.n_predict,
            "temperature": args.temperature,
        },
        out_dir / "longbench_summary.json",
    )


if __name__ == "__main__":
    main()
