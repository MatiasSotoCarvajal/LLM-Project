"""Agentic KV-cache benchmark: does KV-cache quantization degrade a *multi-step*
tool-using agent more than it degrades single-shot accuracy?

Same model weights throughout; only the KV cache config (cache_type_k /
cache_type_v) varies. For each config this script:
  1. starts a llama-server with the Hermes model + that K/V cache type
  2. runs a set of verifiable multi-step tool tasks through a small agent loop,
     N trials each (agents are stochastic, so we need a distribution)
  3. records per-trial success / steps / tool-call validity / latency, plus the
     config's peak RSS and kv_cache_mib
  4. stops the server and moves to the next config

The tasks force tool chaining (look up values, then compute), so per-step
errors compound -- which is exactly where KV-cache compression should show up
if it hurts, and where long accumulating agent context stresses the KV cache.

Usage:
    python benchmarks/test_agent.py NousResearch/Hermes-... \\
        --cache-pairs f16:f16,q8_0:q8_0,turbo4:turbo4 \\
        --trials 5 --n-gpu-layers 999 --flash-attn

The agent uses llama-server's /v1/chat/completions with tools; run the server
with --jinja (added automatically) so the model's tool template is applied.
"""
import argparse
import ast
import csv
import json
import operator
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
    DEFAULT_HOST, DEFAULT_PORT, STARTUP_TIMEOUT, run, stop_server,
)
from backend.evaluate import build_server_args  # noqa: E402
from test_longbenchv2 import (  # noqa: E402
    LogCollector, resolve_model_path, wait_for_server_safe,
    parse_kv_cache_mib, read_rss_gb, read_vram_mb,
    parse_cache_pairs, format_default_pairs,
)

DEFAULT_OUT_DIR = RESULTS_DIR.parent / "results_agent"
DEFAULT_CACHE_PAIRS = [
    {"K": "f16", "V": "f16"},        # full-precision KV baseline
    {"K": "q8_0", "V": "q8_0"},      # standard 8-bit
    {"K": "turbo4", "V": "turbo4"},  # aggressive turbo
]
DEFAULT_TRIALS = 5
DEFAULT_MAX_STEPS = 8
DEFAULT_N_CTX = 8192
DEFAULT_TEMPERATURE = 0.7   # >0 so trials differ -> a real success distribution
DEFAULT_WEIGHT_QUANT = "Q8_0"
REQUEST_TIMEOUT = 300

# ---------------------------------------------------------------------------
# Tools the agent can call (deterministic, local, safe). The model must LOOK UP
# fact values (it isn't told them) and then COMPUTE -- forcing multi-step use.
# ---------------------------------------------------------------------------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def safe_calc(expression: str) -> str:
    """Evaluate a plain arithmetic expression safely (no names/calls)."""
    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError("unsupported expression")
    val = _eval(ast.parse(expression, mode="eval").body)
    if isinstance(val, float) and val.is_integer():
        val = int(val)
    return str(val)


TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "lookup",
        "description": "Look up the numeric value of a named fact for this task.",
        "parameters": {"type": "object", "properties": {
            "key": {"type": "string", "description": "The fact name to look up."}},
            "required": ["key"]}}},
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression, e.g. '(68 + 83) * 2'.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string"}}, "required": ["expression"]}}},
]


def dispatch_tool(name: str, args: dict, facts: dict) -> str:
    if name == "lookup":
        key = str(args.get("key", "")).strip()
        return str(facts[key]) if key in facts else f"ERROR: unknown key '{key}'"
    if name == "calculator":
        try:
            return safe_calc(str(args.get("expression", "")))
        except Exception as exc:
            return f"ERROR: {exc}"
    return f"ERROR: unknown tool '{name}'"


# ---------------------------------------------------------------------------
# Tasks: each needs >=1 lookup + >=1 calculation. expected is the final number.
# ---------------------------------------------------------------------------
TASKS = [
    {"id": "sum_x2", "facts": {"france_m": 68, "germany_m": 83}, "expected": 302,
     "prompt": "Look up 'france_m' and 'germany_m', add them, then multiply the sum by 2. Give the final number."},
    {"id": "mul_sub", "facts": {"a": 15, "b": 4}, "expected": 50,
     "prompt": "Look up 'a' and 'b', multiply them, then subtract 10. Give the final number."},
    {"id": "distance", "facts": {"speed": 60, "time": 3}, "expected": 180,
     "prompt": "Look up 'speed' and 'time', multiply them to get distance. Give the final number."},
    {"id": "price_tax", "facts": {"price": 25, "qty": 6}, "expected": 165,
     "prompt": "Look up 'price' and 'qty', multiply for subtotal, then add 10% tax. Give the final number."},
    {"id": "div_add", "facts": {"x": 100, "y": 25}, "expected": 29,
     "prompt": "Look up 'x' and 'y', divide x by y, then add y to the result. Give the final number."},
    {"id": "days", "facts": {"pages": 320, "per_day": 40}, "expected": 8,
     "prompt": "Look up 'pages' and 'per_day', divide pages by per_day. Give the final number."},
    {"id": "year", "facts": {"start": 2020, "years": 45}, "expected": 2065,
     "prompt": "Look up 'start' and 'years', add them. Give the final number."},
    {"id": "three_hop", "facts": {"aa": 7, "bb": 8, "cc": 9}, "expected": 65,
     "prompt": "Look up 'aa', 'bb' and 'cc', compute aa*bb then add cc. Give the final number."},
    {"id": "rate", "facts": {"base": 50, "rate_pct": 20}, "expected": 80,
     "prompt": "Look up 'base' and 'rate_pct'. Compute base * (rate_pct/100), multiply that by 3, then add base. Give the final number."},
    {"id": "fuel", "facts": {"liters": 12, "price_per": 3}, "expected": 36,
     "prompt": "Look up 'liters' and 'price_per', multiply them. Give the final number."},
]

SYSTEM_PROMPT = (
    "You are a precise tool-using assistant. Use the `lookup` tool to get fact "
    "values (you are NOT told them) and the `calculator` tool for arithmetic. "
    "Do not guess numbers. When you have the final number, reply with just that "
    "number and nothing else."
)

# Hermes-style inline tool call, in case --jinja structured tool_calls aren't emitted.
_HERMES_TOOLCALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def extract_tool_calls(message: dict) -> list[dict]:
    """Return [{name, args}] from either OpenAI tool_calls or Hermes <tool_call> tags."""
    calls = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        calls.append({"name": fn.get("name", ""), "args": args, "id": tc.get("id")})
    if not calls and message.get("content"):
        for m in _HERMES_TOOLCALL.finditer(message["content"]):
            try:
                obj = json.loads(m.group(1))
                calls.append({"name": obj.get("name", ""),
                              "args": obj.get("arguments", {}), "id": None})
            except json.JSONDecodeError:
                pass
    return calls


def final_number(text: str) -> float | None:
    """Pull the last number out of the model's final answer."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(nums[-1]) if nums else None


def run_agent(base_url: str, task: dict, seed: int, temperature: float,
              max_steps: int) -> dict:
    """Run one agent episode. Returns success/steps/validity/latency for one trial."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task["prompt"]},
    ]
    llm_calls = tool_calls = invalid_tool_calls = 0
    final_text = ""
    error = ""
    start = time.monotonic()

    try:
        for _ in range(max_steps):
            payload = {
                "messages": messages, "tools": TOOLS_SCHEMA, "tool_choice": "auto",
                "temperature": temperature, "seed": seed, "max_tokens": 512,
            }
            r = requests.post(f"{base_url}/v1/chat/completions",
                              json=payload, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            llm_calls += 1
            msg = r.json()["choices"][0]["message"]
            calls = extract_tool_calls(msg)

            if not calls:
                final_text = msg.get("content") or ""
                break

            # append assistant turn, then a tool result per call
            messages.append({"role": "assistant", "content": msg.get("content") or "",
                             "tool_calls": msg.get("tool_calls")})
            for c in calls:
                tool_calls += 1
                if c["name"] not in ("lookup", "calculator"):
                    invalid_tool_calls += 1
                result = dispatch_tool(c["name"], c["args"], task["facts"])
                if result.startswith("ERROR"):
                    invalid_tool_calls += 1
                tool_msg = {"role": "tool", "content": result}
                if c["id"]:
                    tool_msg["tool_call_id"] = c["id"]
                messages.append(tool_msg)
        else:
            error = f"hit max_steps ({max_steps})"
    except Exception as exc:
        error = f"error: {exc}"

    latency = round(time.monotonic() - start, 3)
    got = final_number(final_text)
    success = got is not None and abs(got - task["expected"]) < 1e-6
    return {
        "success": bool(success), "steps": llm_calls, "tool_calls": tool_calls,
        "invalid_tool_calls": invalid_tool_calls, "latency_s": latency,
        "final_answer": final_text.strip()[:200], "expected": task["expected"],
        "notes": error,
    }


ROW_FIELDNAMES = [
    "model", "weight_quant", "cache_type_k", "cache_type_v",
    "task_id", "trial", "success", "steps", "tool_calls", "invalid_tool_calls",
    "latency_s", "final_answer", "expected", "notes",
    "rss_gb_peak", "kv_cache_mib",
]


def evaluate_config(model_id, weight_quant, k_type, v_type, n_ctx, trials,
                    temperature, max_steps, seed, host, port,
                    n_gpu_layers, flash_attn, n_batch, n_ubatch):
    rows, summary = [], None
    try:
        model_path = resolve_model_path(model_id, weight_quant)
    except FileNotFoundError as exc:
        print(f"Skipping {model_id}/{weight_quant}: {exc}", file=sys.stderr)
        return rows, summary

    header = f"Model: {model_id}  KV: k={k_type} v={v_type}  trials={trials}"
    print(f"\n{'=' * len(header)}\n{header}\n{'=' * len(header)}", flush=True)

    server_args = build_server_args(
        n_ctx=n_ctx, n_parallel=1, no_mmap=True, extra=["--jinja"],
        n_gpu_layers=n_gpu_layers, flash_attn=flash_attn,
        n_batch=n_batch, n_ubatch=n_ubatch, no_warmup=True)
    proc, actual_port = run(model_id, host=host, port=port,
                            cache_type_k=k_type, cache_type_v=v_type,
                            extra_args=server_args, capture_output=True,
                            model_path=model_path)
    logs = LogCollector(proc)
    peak_rss = 0.0
    try:
        wait_for_server_safe(proc, host, actual_port, STARTUP_TIMEOUT, log_collector=logs)
        peak_rss = read_rss_gb(proc.pid) or 0.0
        base_url = f"http://{host}:{actual_port}"

        n_ok = 0
        for task in TASKS:
            for trial in range(trials):
                res = run_agent(base_url, task, seed + trial, temperature, max_steps)
                n_ok += res["success"]
                rows.append({
                    "model": model_id, "weight_quant": weight_quant,
                    "cache_type_k": k_type, "cache_type_v": v_type,
                    "task_id": task["id"], "trial": trial, **res,
                    "rss_gb_peak": None, "kv_cache_mib": None,
                })
                s = read_rss_gb(proc.pid)
                if s:
                    peak_rss = max(peak_rss, s)
                flag = "OK " if res["success"] else "XX "
                print(f"  {flag} {task['id']:12s} trial {trial}  "
                      f"steps={res['steps']} got={res['final_answer'][:20]!r} "
                      f"exp={res['expected']}  {res['latency_s']}s", flush=True)
    except Exception as exc:
        print(f"  ERROR: {exc}", file=sys.stderr, flush=True)
    finally:
        stop_server(proc)
        time.sleep(0.5)

    kv_mib = parse_kv_cache_mib(logs.text())
    total = len(TASKS) * trials
    for row in rows:
        row["rss_gb_peak"] = round(peak_rss, 4) if peak_rss else None
        row["kv_cache_mib"] = round(kv_mib, 2) if kv_mib is not None else None
    n_ok = sum(r["success"] for r in rows)
    summary = {
        "model": model_id, "weight_quant": weight_quant,
        "cache_type_k": k_type, "cache_type_v": v_type,
        "n_trials_total": total, "n_success": n_ok,
        "success_rate": round(n_ok / total, 4) if total else None,
        "avg_steps": round(sum(r["steps"] for r in rows) / total, 2) if total else None,
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / total, 3) if total else None,
        "invalid_tool_calls": sum(r["invalid_tool_calls"] for r in rows),
        "rss_gb_peak": round(peak_rss, 4) if peak_rss else None,
        "kv_cache_mib": round(kv_mib, 2) if kv_mib is not None else None,
    }
    print(f"\n  success_rate={summary['success_rate']}  "
          f"avg_steps={summary['avg_steps']}  kv_cache_mib={summary['kv_cache_mib']}",
          flush=True)
    return rows, summary


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="Hermes model id resolvable under ./models.")
    p.add_argument("--weight-quant", default=DEFAULT_WEIGHT_QUANT)
    p.add_argument("--cache-pairs", type=parse_cache_pairs, default=DEFAULT_CACHE_PAIRS,
                   help="K:V pairs to compare. Default: f16:f16,q8_0:q8_0,turbo4:turbo4")
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--n-ctx", type=int, default=DEFAULT_N_CTX)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--seed", type=int, default=42)
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
    print(f"Agent sweep: {len(args.cache_pairs)} KV config(s) x {len(TASKS)} tasks "
          f"x {args.trials} trials = {len(args.cache_pairs) * len(TASKS) * args.trials} episodes",
          flush=True)

    all_rows, all_summary = [], []
    for pair in args.cache_pairs:
        rows, summary = evaluate_config(
            args.model, args.weight_quant, pair["K"], pair["V"],
            args.n_ctx, args.trials, args.temperature, args.max_steps, args.seed,
            args.host, args.port, args.n_gpu_layers, args.flash_attn,
            args.batch_size, args.ubatch_size)
        all_rows.extend(rows)
        if summary:
            all_summary.append(summary)

    if not all_rows:
        print("No results produced.", file=sys.stderr)
        raise SystemExit(1)

    csv_path = out_dir / "agent_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDNAMES)
        w.writeheader()
        w.writerows(all_rows)
    (out_dir / "agent_summary.json").write_text(json.dumps(
        {"model": args.model, "weight_quant": args.weight_quant,
         "trials": args.trials, "tasks": [t["id"] for t in TASKS],
         "results": all_summary}, indent=2))
    print(f"\nSaved {csv_path}")
    print(f"Saved {out_dir / 'agent_summary.json'}")
    print("\n=== success rate by KV config ===")
    for s in all_summary:
        print(f"  {s['cache_type_k']}/{s['cache_type_v']:8s} "
              f"success={s['success_rate']}  avg_steps={s['avg_steps']}  "
              f"kv_mib={s['kv_cache_mib']}")


if __name__ == "__main__":
    main()
