"""Shared agent-task definitions + scoring (identical data to agent_tasks.pdf).

Import this from ANY agent runner so every harness uses the same tasks/answers:

    from agent_tasks_data import TASKS, SYSTEM_PROMPT, score

    for task in TASKS:
        answer = my_agent(task["prompt"], tools=task["tools"], facts=task["facts"])
        ok = score(task, answer)

Each task dict has:
    id       - unique name
    section  - "recall" | "state" | "multihop"
    prompt   - the exact user prompt (recall/state embed their data here)
    facts    - dict for the `lookup` tool (multihop only; None otherwise)
    tools    - which tools the agent should expose: ["calculator"] or
               ["lookup", "calculator"]
    expected - the ground-truth final number
"""
import random
import re

SEED = 42
_rng = random.Random(SEED)

SYSTEM_PROMPT = (
    "You are a precise tool-using assistant. Use the calculator tool for all "
    "arithmetic and do not guess. When you have the final number, reply with "
    "just that number and nothing else."
)


def _sensors(n):
    return {f"sensor_{i:02d}": _rng.randint(10, 99) for i in range(1, n + 1)}


def _sensor_block(sensors, per_line=6):
    items = [f"{k}: {v}" for k, v in sensors.items()]
    return "\n".join("   ".join(items[i:i + per_line]) for i in range(0, len(items), per_line))


TASKS = []

# --- Section A: buried-fact recall (data in the prompt; no lookup tool) ---
for length, target_idx in [(40, 4), (80, 7), (120, 11)]:
    s = _sensors(length)
    keys = list(s)
    a, b = keys[-2], keys[-1]
    target = f"sensor_{target_idx:02d}"
    TASKS.append({
        "id": f"recall_{length}", "section": "recall",
        "facts": None, "tools": ["calculator"], "expected": s[target],
        "prompt": (
            f"Here are {length} sensor readings. Remember them:\n"
            f"{_sensor_block(s)}\n\n"
            f"Using the readings above:\n"
            f"1. Use the calculator to add {a} and {b} together.\n"
            f"2. Then tell me the exact value of {target}.\n"
            f"Reply with just the value of {target}."
        ),
    })

# --- Section B: running-state tracking (transactions in the prompt) ---
for length in [8, 12, 16, 20]:
    txns = [_rng.choice([1, -1]) * _rng.randint(10, 300) for _ in range(length)]
    TASKS.append({
        "id": f"balance_{length}", "section": "state",
        "facts": None, "tools": ["calculator"], "expected": sum(txns),
        "prompt": (
            "You are tracking a balance. Starting balance is 0. Apply these "
            f"{length} transactions in order, using the calculator for each step, "
            "keeping the running balance:\n"
            f"{', '.join(f'{x:+d}' for x in txns)}\n"
            "Reply with just the final balance."
        ),
    })

# --- Section C: multi-hop chain (values via the lookup tool) ---
for tid, facts, expr, exp in [
    ("chain3", {"base": 240, "factor": 5, "offset": 200, "divisor": 4},
     "(base * factor - offset) / divisor", 250),
    ("chain4", {"p": 18, "q": 7, "r": 6, "s": 3}, "(p * q + r) * s", 396),
    ("chain5", {"a": 100, "b": 4, "c": 9, "d": 2, "e": 15}, "((a / b) * c - d) * e", 3345),
]:
    TASKS.append({
        "id": tid, "section": "multihop",
        "facts": facts, "tools": ["lookup", "calculator"], "expected": exp,
        "prompt": (
            f"Compute {expr}. Look up each value with the lookup tool as you need "
            "it and use the calculator for each operation. "
            "Reply with just the final number."
        ),
    })


def final_number(text):
    """Extract the last number from the agent's final answer."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", (text or "").replace(",", ""))
    return float(nums[-1]) if nums else None


def score(task, answer_text):
    """True iff the agent's final answer matches the task's expected value exactly."""
    got = final_number(answer_text)
    return got is not None and abs(got - task["expected"]) < 1e-6


if __name__ == "__main__":
    for t in TASKS:
        print(f"{t['id']:12s} [{t['section']:8s}] tools={t['tools']} expected={t['expected']}")
    print(f"\n{len(TASKS)} tasks total")
