import argparse
import csv
import json
import math
import re
import sys
import threading
import time
from pathlib import Path

import requests

try:
    from backend.llama_server import (
        DEFAULT_HOST,
        DEFAULT_PORT,
        find_model,
        run,
        stop_server,
        wait_for_server,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from backend.llama_server import (
        DEFAULT_HOST,
        DEFAULT_PORT,
        find_model,
        run,
        stop_server,
        wait_for_server,
    )

try:
    from backend.config import RESULTS_DIR
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from backend.config import RESULTS_DIR

DEFAULT_CACHE_CONFIGS = [
    ("f16", "f16"),
    ("q8_0", "q8_0"),
    ("q8_0", "turbo3"),
]

DEFAULT_THROUGHPUT_PROMPT = (
    "Explain, in detail, how a transformer neural network processes a sequence "
    "of tokens from input embeddings through self-attention and feed-forward "
    "layers to produce output logits. Cover the role of the key-value cache "
    "during autoregressive generation."
)

DEFAULT_PERPLEXITY_TEXTS = [
    "The quick brown fox jumps over the lazy dog while the sun sets slowly "
    "behind the distant mountains.",
    "Large language models are trained on vast corpora of text and learn to "
    "predict the next token given the preceding context.",
    "Quantization reduces the numerical precision of model weights and "
    "activations, trading a small amount of accuracy for large gains in memory "
    "and speed.",
    "The key-value cache stores the attention keys and values for previously "
    "processed tokens so they do not have to be recomputed at every generation "
    "step.",
]

KV_SIZE_PATTERN = re.compile(r"KV\s+self\s+size\s*=\s*([\d.]+)\s*MiB", re.IGNORECASE)


class LogCollector:
    def __init__(self, proc):
        self.proc = proc
        self.lines: list[str] = []
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self):
        if self.proc.stdout is None:
            return
        for line in self.proc.stdout:
            self.lines.append(line.rstrip("\n"))

    def text(self) -> str:
        return "\n".join(self.lines)


def read_rss_gb(pid: int) -> float | None:
    status = Path(f"/proc/{pid}/status")
    if not status.exists():
        return None
    for line in status.read_text().splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            return int(parts[1]) / (1024 * 1024)
    return None


def parse_kv_cache_mib(log_text: str) -> float | None:
    match = KV_SIZE_PATTERN.search(log_text)
    return float(match.group(1)) if match else None


def get_props(host: str, port: int) -> dict:
    try:
        r = requests.get(f"http://{host}:{port}/props", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return {}


def measure_throughput(
    host: str,
    port: int,
    prompt: str,
    n_predict: int,
) -> dict:
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "cache_prompt": False,
        "stream": True,
    }
    url = f"http://{host}:{port}/completion"

    start = time.monotonic()
    ttft = None
    timings = {}
    with requests.post(url, json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
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
            if ttft is None and chunk.get("content"):
                ttft = time.monotonic() - start
            if chunk.get("timings"):
                timings = chunk["timings"]

    return {
        "ttft_s": ttft,
        "prompt_tokens": timings.get("prompt_n"),
        "prompt_per_second": timings.get("prompt_per_second"),
        "decode_tokens_per_second": timings.get("predicted_per_second"),
        "predicted_tokens": timings.get("predicted_n"),
    }


def measure_perplexity(
    host: str,
    port: int,
    texts: list[str],
) -> dict:
    url = f"http://{host}:{port}/v1/completions"
    total_nll = 0.0
    total_tokens = 0
    used = 0

    for text in texts:
        payload = {
            "model": "local",
            "prompt": text,
            "max_tokens": 0,
            "echo": True,
            "logprobs": 1,
            "temperature": 0.0,
        }
        try:
            r = requests.post(url, json=payload, timeout=600)
            r.raise_for_status()
            data = r.json()
        except requests.RequestException:
            continue

        try:
            token_logprobs = data["choices"][0]["logprobs"]["token_logprobs"]
        except (KeyError, IndexError, TypeError):
            continue

        values = [lp for lp in token_logprobs if lp is not None]
        if not values:
            continue

        total_nll += -sum(values)
        total_tokens += len(values)
        used += 1

    if total_tokens == 0:
        return {"perplexity": None, "perplexity_texts": 0, "perplexity_tokens": 0}

    return {
        "perplexity": math.exp(total_nll / total_tokens),
        "perplexity_texts": used,
        "perplexity_tokens": total_tokens,
    }


def evaluate_config(
    model_id: str,
    cache_type_k: str,
    cache_type_v: str,
    host: str,
    port: int,
    throughput_prompt: str,
    n_predict: int,
    perplexity_texts: list[str] | None,
    extra_args: list[str] | None,
) -> dict:
    row: dict = {
        "model": model_id,
        "cache_type_k": cache_type_k,
        "cache_type_v": cache_type_v,
        "n_ctx": None,
        "kv_cache_mib": None,
        "rss_gb_after_load": None,
        "rss_gb_peak": None,
        "ttft_s": None,
        "prompt_per_second": None,
        "decode_tokens_per_second": None,
        "predicted_tokens": None,
        "perplexity": None,
        "perplexity_texts": None,
        "perplexity_tokens": None,
        "notes": "",
    }

    proc = run(
        model_id,
        host=host,
        port=port,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        extra_args=extra_args,
        capture_output=True,
    )
    logs = LogCollector(proc)

    try:
        wait_for_server(host, port)

        rss_after_load = read_rss_gb(proc.pid)
        row["rss_gb_after_load"] = rss_after_load
        peak = rss_after_load or 0.0

        props = get_props(host, port)
        settings = props.get("default_generation_settings", {})
        row["n_ctx"] = settings.get("n_ctx") or props.get("n_ctx")

        throughput = measure_throughput(host, port, throughput_prompt, n_predict)
        row.update(throughput)

        sample = read_rss_gb(proc.pid)
        if sample is not None:
            peak = max(peak, sample)

        if perplexity_texts:
            row.update(measure_perplexity(host, port, perplexity_texts))

        sample = read_rss_gb(proc.pid)
        if sample is not None:
            peak = max(peak, sample)
        row["rss_gb_peak"] = peak

        row["kv_cache_mib"] = parse_kv_cache_mib(logs.text())
    except Exception as exc:
        row["notes"] = f"error: {exc}"
    finally:
        stop_server(proc)

    return row


def save_results_to_csv(rows: list[dict], file_name: Path) -> None:
    file_name.parent.mkdir(parents=True, exist_ok=True)
    file_exists = file_name.is_file()
    fieldnames = list(rows[0].keys())

    with open(file_name, mode="a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Results saved to {file_name}")


def parse_cache_configs(value: str) -> list[tuple[str, str]]:
    configs = []
    for pair in value.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise argparse.ArgumentTypeError(
                f"Invalid cache config '{pair}', expected format 'k_type:v_type'."
            )
        k, v = pair.split(":", 1)
        configs.append((k.strip(), v.strip()))
    return configs


def load_perplexity_texts(path: str | None) -> list[str]:
    if path is None:
        return DEFAULT_PERPLEXITY_TEXTS
    text = Path(path).read_text(encoding="utf-8")
    blocks = [block.strip() for block in text.split("\n\n")]
    return [block for block in blocks if block]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate GGUF models served by the TurboQuant llama.cpp server across "
            "KV cache configurations. Measures memory, KV cache size, throughput "
            "and perplexity, then writes the results to a CSV file."
        )
    )
    parser.add_argument(
        "models",
        nargs="+",
        help="One or more model ids resolvable under ./models (see find_model).",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--cache-configs",
        type=parse_cache_configs,
        default=DEFAULT_CACHE_CONFIGS,
        help=(
            "Comma separated list of 'k_type:v_type' pairs to sweep, e.g. "
            "'f16:f16,q8_0:turbo3'. Defaults to a baseline vs TurboQuant sweep."
        ),
    )
    parser.add_argument(
        "--n-predict",
        type=int,
        default=128,
        help="Number of tokens to generate when measuring throughput.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_THROUGHPUT_PROMPT,
        help="Prompt used for the throughput measurement.",
    )
    parser.add_argument(
        "--ppl-file",
        default=None,
        help=(
            "Optional path to a text file (paragraphs separated by blank lines) "
            "used for perplexity. Defaults to a small built-in sample."
        ),
    )
    parser.add_argument(
        "--skip-perplexity",
        action="store_true",
        help="Skip the perplexity measurement.",
    )
    parser.add_argument(
        "--out",
        default=str(RESULTS_DIR / "results.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=None,
        help="Everything after this flag is passed verbatim to llama-server.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    perplexity_texts = None if args.skip_perplexity else load_perplexity_texts(args.ppl_file)

    rows = []
    for model_id in args.models:
        try:
            find_model(model_id)
        except FileNotFoundError as exc:
            print(f"Skipping '{model_id}': {exc}", file=sys.stderr)
            continue

        for cache_type_k, cache_type_v in args.cache_configs:
            print("=" * 70)
            print(f"Model: {model_id}  KV cache: k={cache_type_k} v={cache_type_v}")
            row = evaluate_config(
                model_id,
                cache_type_k,
                cache_type_v,
                args.host,
                args.port,
                args.prompt,
                args.n_predict,
                perplexity_texts,
                args.extra_args,
            )
            print(json.dumps(row, indent=2))
            rows.append(row)

    if not rows:
        print("No results produced.", file=sys.stderr)
        raise SystemExit(1)

    save_results_to_csv(rows, Path(args.out))


if __name__ == "__main__":
    main()
