#!/usr/bin/env bash
# Run the agent KV-cache benchmark with everything baked in, so no long command
# has to be pasted (paste mangling was breaking the model name + flags).
#
#   bash benchmarks/run_agent.sh smoke   # quick: 1 trial, f16 only (~2 min)
#   bash benchmarks/run_agent.sh         # full:  5 trials, all 3 KV configs
#
# Override the model by passing it as the 2nd arg:
#   bash benchmarks/run_agent.sh smoke unsloth/Qwen3.5-9B-GGUF
set -e
cd "$(dirname "$0")/.."

# activate the project venv if present (harmless if already active)
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

MODEL="${2:-bartowski/Meta-Llama-3.1-8B-Instruct-GGUF}"

if [ "$1" = "smoke" ]; then
    TRIALS=1
    PAIRS="f16:f16"
    echo ">> SMOKE TEST: $MODEL  (1 trial, f16:f16 only)"
else
    TRIALS=5
    PAIRS="f16:f16,q8_0:q8_0,turbo4:turbo4"
    echo ">> FULL RUN: $MODEL  (5 trials, 3 KV configs)"
fi

python benchmarks/test_agent.py "$MODEL" \
    --weight-quant Q8_0 \
    --cache-pairs "$PAIRS" \
    --trials "$TRIALS" \
    --n-ctx 8192 \
    --n-gpu-layers 999 \
    --flash-attn
