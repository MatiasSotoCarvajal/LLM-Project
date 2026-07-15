import argparse
import csv
import json
import re
import subprocess
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
    from backend.config import (
        RESULTS_DIR,
        PERPLEXITY_BIN,
        DEFAULT_PPL_FILE,
        N_GPU_LAYERS,
        FLASH_ATTN,
        N_BATCH,
        N_UBATCH,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from backend.config import (
        RESULTS_DIR,
        PERPLEXITY_BIN,
        DEFAULT_PPL_FILE,
        N_GPU_LAYERS,
        FLASH_ATTN,
        N_BATCH,
        N_UBATCH,
    )


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

KV_SIZE_PATTERN = re.compile(r"KV\s+self\s+size\s*=\s*([\d.]+)\s*MiB", re.IGNORECASE)
PPL_PATTERN = re.compile(r"Final estimate:\s*PPL\s*=\s*([\d.]+)", re.IGNORECASE)
PPL_TOKENS_PATTERN = re.compile(r"perplexity:\s*tokenizing the input\b", re.IGNORECASE)


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
    matches = KV_SIZE_PATTERN.findall(log_text)
    return float(matches[-1]) if matches else None


def get_props(host: str, port: int) -> dict:
    try:
        r = requests.get(f"http://{host}:{port}/props", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return {}


def build_server_args(
    n_ctx: int,
    n_parallel: int,
    no_mmap: bool,
    extra: list[str] | None,
    n_gpu_layers: int | None = None,
    flash_attn: bool = False,
    n_batch: int | None = None,
    n_ubatch: int | None = None,
) -> list[str]:
    args = ["-c", str(n_ctx), "-np", str(n_parallel)]
    if no_mmap:
        args.append("--no-mmap")
    if n_gpu_layers is not None:
        args.extend(["-ngl", str(n_gpu_layers)])
    if flash_attn:
        args.append("-fa")
    if n_batch is not None:
        args.extend(["-b", str(n_batch)])
    if n_ubatch is not None:
        args.extend(["-ub", str(n_ubatch)])
    if extra:
        args.extend(extra)
    return args


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


def measure_perplexity_cli(
    model_path: Path,
    cache_type_k: str,
    cache_type_v: str,
    ppl_file: Path,
    ppl_ctx: int,
    threads: int | None,
    n_gpu_layers: int | None = None,
) -> dict:
    result = {"perplexity": None, "kv_cache_mib": None, "ppl_ctx": ppl_ctx, "ppl_note": ""}

    if not PERPLEXITY_BIN.exists():
        result["ppl_note"] = "perplexity binary missing"
        return result
    if not ppl_file.exists():
        result["ppl_note"] = f"ppl file missing: {ppl_file}"
        return result

    cmd = [
        str(PERPLEXITY_BIN),
        "-m", str(model_path),
        "-f", str(ppl_file),
        "-c", str(ppl_ctx),
        "-ctk", cache_type_k,
        "-ctv", cache_type_v,
        "--no-mmap",
    ]
    if threads is not None:
        cmd += ["-t", str(threads)]
    if n_gpu_layers is not None:
        cmd += ["-ngl", str(n_gpu_layers)]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=3600,
        )
    except subprocess.TimeoutExpired:
        result["ppl_note"] = "perplexity timeout"
        return result

    out = proc.stdout or ""
    ppl_match = PPL_PATTERN.search(out)
    if ppl_match:
        result["perplexity"] = float(ppl_match.group(1))
    else:
        tail = " | ".join(out.strip().splitlines()[-3:])
        result["ppl_note"] = f"no PPL parsed (rc={proc.returncode}): {tail}"[:300]

    result["kv_cache_mib"] = parse_kv_cache_mib(out)
    return result


def evaluate_config(
    model_id: str,
    model_path: Path,
    cache_type_k: str,
    cache_type_v: str,
    host: str,
    port: int,
    n_ctx: int,
    n_parallel: int,
    no_mmap: bool,
    throughput_prompt: str,
    n_predict: int,
    ppl_file: Path | None,
    ppl_ctx: int,
    threads: int | None,
    extra_args: list[str] | None,
    n_gpu_layers: int | None = None,
    flash_attn: bool = False,
    n_batch: int | None = None,
    n_ubatch: int | None = None,
) -> dict:
    row: dict = {
        "model": model_id,
        "cache_type_k": cache_type_k,
        "cache_type_v": cache_type_v,
        "n_ctx": n_ctx,
        "n_parallel": n_parallel,
        "rss_gb_after_load": None,
        "rss_gb_peak": None,
        "ttft_s": None,
        "prompt_tokens": None,
        "prompt_per_second": None,
        "decode_tokens_per_second": None,
        "predicted_tokens": None,
        "kv_cache_mib": None,
        "ppl_ctx": None,
        "perplexity": None,
        "notes": "",
    }

    server_args = build_server_args(
        n_ctx, n_parallel, no_mmap, extra_args, n_gpu_layers, flash_attn, n_batch, n_ubatch
    )
    proc, actual_port = run(
        model_id,
        host=host,
        port=port,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        extra_args=server_args,
        capture_output=True,
    )
    logs = LogCollector(proc)

    try:
        wait_for_server(host, actual_port)

        rss_after_load = read_rss_gb(proc.pid)
        row["rss_gb_after_load"] = rss_after_load
        peak = rss_after_load or 0.0

        props = get_props(host, actual_port)
        settings = props.get("default_generation_settings", {})
        row["n_ctx"] = settings.get("n_ctx") or props.get("n_ctx") or n_ctx

        throughput = measure_throughput(host, actual_port, throughput_prompt, n_predict)
        row.update(throughput)

        sample = read_rss_gb(proc.pid)
        if sample is not None:
            peak = max(peak, sample)
        row["rss_gb_peak"] = peak

        kv_from_server = parse_kv_cache_mib(logs.text())
        if kv_from_server is not None:
            row["kv_cache_mib"] = kv_from_server
    except Exception as exc:
        row["notes"] = f"server error: {exc}"
    finally:
        stop_server(proc)

    if ppl_file is not None:
        ppl = measure_perplexity_cli(
            model_path, cache_type_k, cache_type_v, ppl_file, ppl_ctx, threads, n_gpu_layers
        )
        row["perplexity"] = ppl["perplexity"]
        row["ppl_ctx"] = ppl["ppl_ctx"]
        if row["kv_cache_mib"] is None:
            row["kv_cache_mib"] = ppl["kv_cache_mib"]
        if ppl["ppl_note"]:
            row["notes"] = (row["notes"] + "; " + ppl["ppl_note"]).strip("; ")

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
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port for llama-server. Default: 0 (auto-detect free port).",
    )
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
        "--n-ctx",
        type=int,
        default=4096,
        help="Context length for the server (KV cache is sized to this).",
    )
    parser.add_argument(
        "--n-parallel",
        type=int,
        default=1,
        help="Number of server slots. Keep at 1 so the KV cache is not multiplied.",
    )
    parser.add_argument(
        "--mmap",
        dest="no_mmap",
        action="store_false",
        help="Enable mmap. Disabled by default so RSS reflects true memory usage.",
    )
    parser.set_defaults(no_mmap=True)
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
        default=str(DEFAULT_PPL_FILE),
        help="Text file used for perplexity (llama-perplexity -f). Defaults to a built-in sample.",
    )
    parser.add_argument(
        "--ppl-ctx",
        type=int,
        default=512,
        help="Context length used for the perplexity computation.",
    )
    parser.add_argument(
        "--skip-perplexity",
        action="store_true",
        help="Skip the perplexity measurement.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Threads for llama-perplexity. Defaults to binary auto.",
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
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=N_GPU_LAYERS,
        help="Number of layers to offload to GPU (-ngl). Defaults to N_GPU_LAYERS env var, or CPU if unset.",
    )
    parser.add_argument(
        "--flash-attn",
        action="store_true",
        default=FLASH_ATTN,
        help="Enable flash attention (-fa). Defaults to FLASH_ATTN env var (off if unset).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=N_BATCH,
        help="Logical batch size (-b). Defaults to N_BATCH env var, or llama.cpp default if unset.",
    )
    parser.add_argument(
        "--ubatch-size",
        type=int,
        default=N_UBATCH,
        help="Micro-batch size (-ub). Defaults to N_UBATCH env var, or llama.cpp default if unset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ppl_file = None if args.skip_perplexity else Path(args.ppl_file)

    rows = []
    for model_id in args.models:
        try:
            model_path = find_model(model_id)
        except FileNotFoundError as exc:
            print(f"Skipping '{model_id}': {exc}", file=sys.stderr)
            continue

        for cache_type_k, cache_type_v in args.cache_configs:
            print("=" * 70)
            print(f"Model: {model_id}  KV cache: k={cache_type_k} v={cache_type_v}")
            row = evaluate_config(
                model_id,
                model_path,
                cache_type_k,
                cache_type_v,
                args.host,
                args.port,
                args.n_ctx,
                args.n_parallel,
                args.no_mmap,
                args.prompt,
                args.n_predict,
                ppl_file,
                args.ppl_ctx,
                args.threads,
                args.extra_args,
                args.n_gpu_layers,
                args.flash_attn,
                args.batch_size,
                args.ubatch_size,
            )
            print(json.dumps(row, indent=2))
            rows.append(row)

    if not rows:
        print("No results produced.", file=sys.stderr)
        raise SystemExit(1)

    save_results_to_csv(rows, Path(args.out))


if __name__ == "__main__":
    main()
