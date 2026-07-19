"""RULER-style long-context benchmark: how much does KV-cache quantization
degrade the model's *effective context* -- its ability to retrieve and track
information across a long prompt?

This replaces the tool-calling agent benchmark. The agent benchmark exercised
llama-server's `--jinja` tool path via /v1/chat/completions, which is fragile (it
returned HTTP 500 on most requests). RULER needs no tools -- it is plain retrieval
/ multi-hop QA, so it drives the raw **/completion** endpoint (the same one the
working LongBench harness uses): raw prompt, no chat template, no --jinja, no
tools. That avoids the failure mode entirely while measuring exactly what
KV-cache compression stresses.

Same model weights throughout; only the KV cache config (cache_type_k /
cache_type_v) varies. For each config this script:
  1. starts a llama-server with the model + that K/V cache type
  2. generates synthetic RULER tasks at several context LENGTHS and runs them
  3. scores each by recall of the expected answer strings (deterministic, exact)
  4. records per-sample rows + per-(task,length) aggregates, plus the config's
     peak RSS and kv_cache_mib
  5. stops the server and moves to the next config

The tasks are faithful re-implementations of RULER's core families (Hsieh et al.,
2024, "RULER: What's the Real Context Size of Your Long-Context LMs?"):
  - niah_single     single needle-in-a-haystack retrieval
  - niah_multikey   retrieve one key's value among many distractor keys
  - niah_multivalue retrieve all values assigned to one key
  - niah_multiquery retrieve values for several queried keys at once
  - vt              variable tracking (multi-hop chains) -- the sharpest probe
                    for KV degradation

Usage:
    python benchmarks/test_ruler.py bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \\
        --cache-pairs f16:f16,q8_0:q8_0,turbo4:turbo4 \\
        --lengths 4096,8192,16384 --samples 20 \\
        --n-ctx 16384 --n-gpu-layers 999 --flash-attn
"""
import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BENCH_DIR))

from backend.config import (  # noqa: E402
    RESULTS_DIR, N_GPU_LAYERS, FLASH_ATTN, N_BATCH, N_UBATCH,
)
from backend.llama_server import (  # noqa: E402
    DEFAULT_HOST, DEFAULT_PORT, STARTUP_TIMEOUT, run, stop_server, find_model,
)
from backend.evaluate import (  # noqa: E402
    build_server_args, LogCollector, read_rss_gb, parse_kv_cache_mib,
)
# NOTE: RULER is deliberately standalone -- it does NOT import test_longbenchv2,
# which pulls in the heavy `datasets` package (and sys.exit()s if it's missing).
# The two helpers it needs from there (completion_request, a health-check waiter)
# are small and inlined below, so RULER runs with just `requests` + `dotenv`.

DEFAULT_OUT_DIR = RESULTS_DIR.parent / "results_ruler"
DEFAULT_CACHE_PAIRS = [
    {"K": "f16", "V": "f16"},        # full-precision KV baseline
    {"K": "q8_0", "V": "q8_0"},      # standard 8-bit
    {"K": "turbo4", "V": "turbo4"},  # aggressive turbo
]
DEFAULT_TASKS = [
    "niah_single", "niah_multikey", "niah_multivalue", "niah_multiquery", "vt",
]
DEFAULT_LENGTHS = [4096, 8192, 16384]
DEFAULT_SAMPLES = 20
DEFAULT_N_CTX = 16384
DEFAULT_N_PREDICT = 128
DEFAULT_TEMPERATURE = 0.0   # retrieval must be deterministic/greedy
DEFAULT_WEIGHT_QUANT = "Q8_0"
DEFAULT_SEED = 42
REQUEST_TIMEOUT = 300
# llama-server can return transient 5xx under flash-attn on Blackwell; a single
# 500 should not be scored as a model failure -- retry before giving up.
MAX_RETRIES = 4
RETRY_BACKOFF_S = 1.5

# ---------------------------------------------------------------------------
# Server / request helpers (inlined so RULER doesn't depend on test_longbenchv2)
# ---------------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    return len(text) // 4


def resolve_model_path(model_id: str, weight_quant: str):
    """Locate the GGUF for model_id under ./models. weight_quant is accepted for
    signature parity; the downloaded folder holds the (Q8_0) GGUF. Raises
    FileNotFoundError if nothing is found."""
    return find_model(model_id)


def wait_for_server_safe(proc, host: str, port: int, timeout: int,
                         log_collector: LogCollector | None = None) -> None:
    """Poll /health until 200, or fail fast if the server process dies first."""
    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = "\n".join(log_collector.lines[-40:]) if log_collector else ""
            raise RuntimeError(
                f"Server process exited with rc={proc.returncode} before "
                f"becoming healthy.\nLast log lines:\n{tail}")
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"Server did not become healthy at {url} within {timeout}s")


def completion_request(host: str, port: int, prompt: str, n_predict: int,
                       temperature: float) -> dict:
    """Stream from llama.cpp's raw /completion endpoint (no chat template, no
    tools). Returns the generated text plus timing info."""
    payload = {
        "prompt": prompt, "n_predict": n_predict, "temperature": temperature,
        "cache_prompt": False, "stream": True,
    }
    url = f"http://{host}:{port}/completion"
    full_content: list[str] = []
    ttft = None
    timings: dict = {}
    start = time.monotonic()
    with requests.post(url, json=payload, stream=True, timeout=REQUEST_TIMEOUT) as r:
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
        "decode_tokens_per_second": timings.get("predicted_per_second"),
        "predicted_tokens": timings.get("predicted_n"),
    }


# Innocuous filler sentences -- the "haystack" the needles are hidden in. Same
# style as the original needle-in-a-haystack noise; deterministic and cheap.
NOISE_SENTENCES = [
    "The grass is green.", "The sky is blue.", "The sun is yellow.",
    "Here we go.", "There and back again.", "The clouds drift slowly by.",
    "Birds fly south for the winter.", "The river runs down to the sea.",
    "A gentle wind moves through the trees.", "The morning light is soft.",
    "Waves roll onto the quiet shore.", "The old road winds through the hills.",
]
# Distinct nouns used as needle keys.
KEY_WORDS = [
    "mango", "tiger", "planet", "violin", "harbor", "cactus", "meadow", "comet",
    "lantern", "glacier", "falcon", "orchid", "canyon", "pebble", "willow",
    "marble", "thunder", "compass", "saffron", "quartz", "cobalt", "juniper",
    "maple", "otter", "raven", "sable", "topaz", "walnut", "zephyr", "ember",
]


# ---------------------------------------------------------------------------
# Synthetic task generation (RULER families)
# ---------------------------------------------------------------------------
def _sample_rng(task: str, length: int, idx: int, seed: int) -> random.Random:
    """Deterministic per-sample RNG. random.Random(str) hashes via sha512, so it
    is stable across processes (unlike hash())."""
    return random.Random(f"{seed}|{task}|{length}|{idx}")


def _magic_number(rng: random.Random) -> str:
    return str(rng.randint(1_000_000, 9_999_999))


def _filler_sentences(target_tokens: int, rng: random.Random) -> list[str]:
    """Build ~target_tokens worth of noise sentences (≈4 chars/token heuristic)."""
    target_chars = target_tokens * 4
    sents: list[str] = []
    total = 0
    while total < target_chars:
        s = rng.choice(NOISE_SENTENCES)
        sents.append(s)
        total += len(s) + 1
    return sents


def _insert_needles(sents: list[str], needles: list[str],
                    rng: random.Random) -> str:
    """Insert each needle at a random depth in the haystack (varies retrieval
    depth across samples), then join into one text block."""
    out = list(sents)
    for needle in needles:
        out.insert(rng.randint(0, len(out)), needle)
    return " ".join(out)


_PREAMBLE = (
    "Some special magic numbers are hidden within the following text. "
    "Make sure to memorize them. I will quiz you about them afterwards.\n\n"
)


def _needle(key: str, value: str) -> str:
    return f"One of the special magic numbers for {key} is: {value}."


def generate_sample(task: str, length: int, idx: int, seed: int) -> dict:
    """Return {prompt, expected(list[str])} for one RULER sample."""
    rng = _sample_rng(task, length, idx, seed)
    filler = _filler_sentences(length, rng)

    if task == "niah_single":
        key, value = rng.choice(KEY_WORDS), _magic_number(rng)
        haystack = _insert_needles(filler, [_needle(key, value)], rng)
        question = (f"What is the special magic number for {key}? "
                    f"Answer with just the number.")
        return {"prompt": _PREAMBLE + haystack + "\n\n" + question,
                "expected": [value]}

    if task == "niah_multikey":
        keys = rng.sample(KEY_WORDS, 5)
        pairs = {k: _magic_number(rng) for k in keys}
        needles = [_needle(k, v) for k, v in pairs.items()]
        target = keys[rng.randint(0, len(keys) - 1)]
        haystack = _insert_needles(filler, needles, rng)
        question = (f"What is the special magic number for {target}? "
                    f"Answer with just the number.")
        return {"prompt": _PREAMBLE + haystack + "\n\n" + question,
                "expected": [pairs[target]]}

    if task == "niah_multivalue":
        key = rng.choice(KEY_WORDS)
        values = [_magic_number(rng) for _ in range(4)]
        needles = [_needle(key, v) for v in values]
        haystack = _insert_needles(filler, needles, rng)
        question = (f"What are all the special magic numbers for {key}? "
                    f"Answer with the numbers separated by commas.")
        return {"prompt": _PREAMBLE + haystack + "\n\n" + question,
                "expected": values}

    if task == "niah_multiquery":
        keys = rng.sample(KEY_WORDS, 4)
        pairs = {k: _magic_number(rng) for k in keys}
        needles = [_needle(k, v) for k, v in pairs.items()]
        haystack = _insert_needles(filler, needles, rng)
        question = ("What are the special magic numbers for "
                    + ", ".join(keys)
                    + "? Answer with the numbers separated by commas.")
        return {"prompt": _PREAMBLE + haystack + "\n\n" + question,
                "expected": list(pairs.values())}

    if task == "vt":
        # Variable tracking: one chain resolves to the queried value; distractor
        # chains resolve to other values. Multi-hop -> sharp KV-degradation probe.
        value = _magic_number(rng)
        chain_len = 4
        names = [f"VAR_{i}" for i in range(chain_len)]
        needles = [f"{names[0]} = {value}."]
        for a, b in zip(names, names[1:]):
            needles.append(f"{b} = {a}.")
        # two distractor chains
        for d in range(2):
            dval = _magic_number(rng)
            dnames = [f"DST{d}_{i}" for i in range(3)]
            needles.append(f"{dnames[0]} = {dval}.")
            for a, b in zip(dnames, dnames[1:]):
                needles.append(f"{b} = {a}.")
        rng.shuffle(needles)
        preamble = ("Track the following variable assignments. A variable may be "
                    "assigned a number, or assigned to another variable.\n\n")
        haystack = _insert_needles(filler, needles, rng)
        question = (f"Find ALL variables that are assigned the value {value}, "
                    f"directly or through a chain of assignments. "
                    f"List every such variable name.")
        return {"prompt": preamble + haystack + "\n\n" + question,
                "expected": names}

    raise ValueError(f"unknown task: {task}")


# ---------------------------------------------------------------------------
# Scoring: recall of expected answer strings (word-boundary exact match)
# ---------------------------------------------------------------------------
def score_recall(expected: list[str], output: str) -> float:
    """Fraction of expected strings that appear in the output (RULER recall)."""
    if not expected:
        return 0.0
    found = 0
    for e in expected:
        if re.search(rf"\b{re.escape(e)}\b", output):
            found += 1
    return found / len(expected)


# ---------------------------------------------------------------------------
# Raw /completion request (no chat template, no tools) with retry
# ---------------------------------------------------------------------------
def complete_with_retry(host: str, port: int, prompt: str, n_predict: int,
                        temperature: float) -> dict:
    """Call llama.cpp's raw /completion (see completion_request above), retrying
    transient 5xx / connection failures. A 4xx raises immediately."""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return completion_request(host, port, prompt, n_predict, temperature)
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else 0
            if code and code < 500:
                raise  # genuine bad request -- retrying won't help
            last_exc = exc
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
        time.sleep(RETRY_BACKOFF_S * (attempt + 1))
    raise last_exc if last_exc else RuntimeError("completion request failed")


def run_sample(host: str, port: int, task: str, length: int, idx: int, seed: int,
               n_predict: int, temperature: float) -> dict:
    sample = generate_sample(task, length, idx, seed)
    expected = sample["expected"]
    prompt = sample["prompt"] + "\n\nAnswer:"   # cue the raw completion to answer
    start = time.monotonic()
    output, prompt_tokens, completion_tokens, note = "", None, None, ""
    try:
        resp = complete_with_retry(host, port, prompt, n_predict, temperature)
        output = resp["output"]
        prompt_tokens = resp["prompt_tokens"]
        completion_tokens = resp["predicted_tokens"]
    except Exception as exc:
        note = f"error: {exc}"
    latency = round(time.monotonic() - start, 3)
    recall = score_recall(expected, output)
    return {
        "task": task, "length": length, "sample": idx,
        "recall": round(recall, 4), "correct": recall >= 1.0,
        "n_expected": len(expected), "n_found": round(recall * len(expected)),
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "latency_s": latency, "output": output.strip()[:200],
        "expected": "|".join(expected), "notes": note,
    }


ROW_FIELDNAMES = [
    "model", "weight_quant", "cache_type_k", "cache_type_v",
    "task", "length", "sample", "recall", "correct",
    "n_expected", "n_found", "prompt_tokens", "completion_tokens",
    "latency_s", "output", "expected", "notes",
    "rss_gb_peak", "kv_cache_mib",
]


def evaluate_config(model_id, weight_quant, k_type, v_type, tasks, lengths,
                    samples, n_ctx, n_predict, temperature, seed, host, port,
                    n_gpu_layers, flash_attn, n_batch, n_ubatch):
    rows, summary = [], None
    try:
        model_path = resolve_model_path(model_id, weight_quant)
    except FileNotFoundError as exc:
        print(f"Skipping {model_id}/{weight_quant}: {exc}", file=sys.stderr)
        return rows, summary

    # Only run lengths that fit the server context window (leave room for output).
    usable_lengths = [L for L in lengths if L + n_predict <= n_ctx]
    for L in lengths:
        if L not in usable_lengths:
            print(f"  [WARN] skipping length {L}: needs > n_ctx={n_ctx} "
                  f"(with n_predict={n_predict})", file=sys.stderr)
    if not usable_lengths:
        print(f"  [WARN] no lengths fit n_ctx={n_ctx}; nothing to run.",
              file=sys.stderr)
        return rows, summary

    header = f"Model: {model_id}  KV: k={k_type} v={v_type}  lengths={usable_lengths}"
    print(f"\n{'=' * len(header)}\n{header}\n{'=' * len(header)}", flush=True)

    # No --jinja / no chat template: RULER uses the raw /completion endpoint.
    server_args = build_server_args(
        n_ctx=n_ctx, n_parallel=1, no_mmap=True, extra=None,
        n_gpu_layers=n_gpu_layers, flash_attn=flash_attn,
        n_batch=n_batch, n_ubatch=n_ubatch, no_warmup=True)
    proc, actual_port = run(model_id, host=host, port=port,
                            cache_type_k=k_type, cache_type_v=v_type,
                            extra_args=server_args, capture_output=True,
                            model_path=model_path)
    logs = LogCollector(proc)
    peak_rss = 0.0
    try:
        wait_for_server_safe(proc, host, actual_port, STARTUP_TIMEOUT,
                             log_collector=logs)
        peak_rss = read_rss_gb(proc.pid) or 0.0

        for task in tasks:
            for L in usable_lengths:
                for idx in range(samples):
                    res = run_sample(host, actual_port, task, L, idx, seed,
                                     n_predict, temperature)
                    rows.append({
                        "model": model_id, "weight_quant": weight_quant,
                        "cache_type_k": k_type, "cache_type_v": v_type,
                        **res, "rss_gb_peak": None, "kv_cache_mib": None,
                    })
                    s = read_rss_gb(proc.pid)
                    if s:
                        peak_rss = max(peak_rss, s)
                sub = [r for r in rows if r["task"] == task and r["length"] == L]
                acc = sum(r["correct"] for r in sub) / len(sub)
                rec = sum(r["recall"] for r in sub) / len(sub)
                print(f"  {task:16s} len={L:6d}  acc={acc:.2f} recall={rec:.2f} "
                      f"(n={len(sub)})", flush=True)
    except Exception as exc:
        print(f"  ERROR: {exc}", file=sys.stderr, flush=True)
    finally:
        stop_server(proc)
        time.sleep(0.5)

    kv_mib = parse_kv_cache_mib(logs.text())
    for row in rows:
        row["rss_gb_peak"] = round(peak_rss, 4) if peak_rss else None
        row["kv_cache_mib"] = round(kv_mib, 2) if kv_mib is not None else None

    # per-(task, length) breakdown + overall
    by_task_length = []
    for task in tasks:
        for L in usable_lengths:
            sub = [r for r in rows if r["task"] == task and r["length"] == L]
            if not sub:
                continue
            by_task_length.append({
                "task": task, "length": L, "n": len(sub),
                "accuracy": round(sum(r["correct"] for r in sub) / len(sub), 4),
                "recall": round(sum(r["recall"] for r in sub) / len(sub), 4),
                "errors": sum(1 for r in sub if r["notes"]),
            })
    total = len(rows)
    summary = {
        "model": model_id, "weight_quant": weight_quant,
        "cache_type_k": k_type, "cache_type_v": v_type,
        "n_samples_total": total,
        "accuracy": round(sum(r["correct"] for r in rows) / total, 4) if total else None,
        "recall": round(sum(r["recall"] for r in rows) / total, 4) if total else None,
        "errors": sum(1 for r in rows if r["notes"]),
        "rss_gb_peak": round(peak_rss, 4) if peak_rss else None,
        "kv_cache_mib": round(kv_mib, 2) if kv_mib is not None else None,
        "by_task_length": by_task_length,
    }
    print(f"\n  accuracy={summary['accuracy']}  recall={summary['recall']}  "
          f"errors={summary['errors']}  kv_cache_mib={summary['kv_cache_mib']}",
          flush=True)
    return rows, summary


def parse_cache_pairs(raw: str) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError(f"Expected K:V format, got '{item}'")
        k, v = item.split(":", 1)
        pairs.append({"K": k.strip(), "V": v.strip()})
    return pairs


def parse_int_list(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="Model id resolvable under ./models.")
    p.add_argument("--weight-quant", default=DEFAULT_WEIGHT_QUANT)
    p.add_argument("--cache-pairs", type=parse_cache_pairs,
                   default=DEFAULT_CACHE_PAIRS,
                   help="K:V pairs. Default: f16:f16,q8_0:q8_0,turbo4:turbo4")
    p.add_argument("--tasks", type=parse_str_list, default=DEFAULT_TASKS,
                   help=f"Comma-separated tasks. Default: {','.join(DEFAULT_TASKS)}")
    p.add_argument("--lengths", type=parse_int_list, default=DEFAULT_LENGTHS,
                   help="Comma-separated context lengths (tokens). "
                        f"Default: {','.join(map(str, DEFAULT_LENGTHS))}")
    p.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                   help=f"Samples per (task, length). Default: {DEFAULT_SAMPLES}")
    p.add_argument("--n-ctx", type=int, default=DEFAULT_N_CTX)
    p.add_argument("--n-predict", type=int, default=DEFAULT_N_PREDICT)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--n-gpu-layers", type=int, default=N_GPU_LAYERS)
    p.add_argument("--flash-attn", action="store_true", default=FLASH_ATTN)
    p.add_argument("--batch-size", type=int, default=N_BATCH)
    p.add_argument("--ubatch-size", type=int, default=N_UBATCH)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_configs = len(args.cache_pairs)
    print(f"RULER sweep: {n_configs} KV config(s) x {len(args.tasks)} tasks "
          f"x {len(args.lengths)} lengths x {args.samples} samples = "
          f"{n_configs * len(args.tasks) * len(args.lengths) * args.samples} "
          f"requests", flush=True)

    all_rows, all_summary = [], []
    for pair in args.cache_pairs:
        rows, summary = evaluate_config(
            args.model, args.weight_quant, pair["K"], pair["V"],
            args.tasks, args.lengths, args.samples, args.n_ctx, args.n_predict,
            args.temperature, args.seed, args.host, args.port,
            args.n_gpu_layers, args.flash_attn, args.batch_size, args.ubatch_size)
        all_rows.extend(rows)
        if summary:
            all_summary.append(summary)

    if not all_rows:
        print("No results produced.", file=sys.stderr)
        raise SystemExit(1)

    csv_path = out_dir / "ruler_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDNAMES)
        w.writeheader()
        w.writerows(all_rows)
    (out_dir / "ruler_summary.json").write_text(json.dumps(
        {"model": args.model, "weight_quant": args.weight_quant,
         "tasks": args.tasks, "lengths": args.lengths, "samples": args.samples,
         "results": all_summary}, indent=2))
    print(f"\nSaved {csv_path}")
    print(f"Saved {out_dir / 'ruler_summary.json'}")
    print("\n=== accuracy by KV config ===")
    for s in all_summary:
        print(f"  {s['cache_type_k']}/{s['cache_type_v']:8s} "
              f"accuracy={s['accuracy']}  recall={s['recall']}  "
              f"errors={s['errors']}  kv_mib={s['kv_cache_mib']}")


if __name__ == "__main__":
    main()
