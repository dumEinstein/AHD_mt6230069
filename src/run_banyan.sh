#!/usr/bin/env bash
# One BANYAN run on a GPU machine, without PBS: serve the generator, search (500 evaluations,
# task wall clock), then select the submitted program on the validation split.
#
#   bash src/run_banyan.sh <task> <seed> [prompts]      e.g.  bash src/run_banyan.sh prism 0 v5
#
# <task>: prism | cloudcast | signal_processing | heilbronn_triangle | circle_packing_rect
# [prompts]: v5 | v7 | v8 (the report's v1 | v2 | v3; default v5)
# Data: the task folders are read from ../problems/<task> (the starter kit layout).
set -euo pipefail
TASK=${1:?usage: run_banyan.sh <task> <seed> [v5|v7|v8]}; SEED=${2:-0}; PV=${3:-v5}
SRC=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SRC/.." && pwd)                         # holds problems/<task>
SKY=${SKY:-$SRC/skydiscover}                        # SkyDiscover checkout with its uv venv
VLLM_VENV=${VLLM_VENV:-$SRC/.venv-vllm}             # venv with vllm
export MODEL=${MODEL:?set MODEL to the Qwen3-Coder-30B-A3B-Instruct weights (path or HF id)}
export PORT=${PORT:-8000}
export API="http://localhost:${PORT}/v1"
RUNS=${RUNS_ROOT:-$SRC/runs}
mkdir -p "$RUNS"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false EVAL_PARALLEL=${EVAL_PARALLEL:-8}

echo "=== [1/3] vLLM: Qwen3-Coder-30B-A3B-Instruct, BF16, 16k context"
source "$VLLM_VENV/bin/activate"
bash "$SRC/scripts/serve_qwen.sh" > "$RUNS/vllm_${TASK}_seed${SEED}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
  curl -sf "$API/models" >/dev/null && break
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died during startup"; tail -40 "$RUNS/vllm_${TASK}_seed${SEED}.log"; exit 1; }
  sleep 10
done
curl -sf "$API/models" >/dev/null || { echo "vLLM did not come up"; exit 1; }
deactivate

echo "=== [2/3] BANYAN search: $TASK, seed $SEED, prompts $PV"
BANYAN=1 EVAL_CAP=500 PROMPTS_DIR="$SRC/prompts/$PV" PROJ="$SRC" KIT="$ROOT" SKY="$SKY" API="$API" OUT="$RUNS" \
  bash "$SRC/scripts/run_task.sh" "$TASK" "$SEED" banyan
kill $VLLM_PID 2>/dev/null || true

echo "=== [3/3] validation selection"
RUN_DIR="$(ls -d "$RUNS/$TASK/banyan_seed${SEED}_"* | sort | tail -n 1)"
case $TASK in heilbronn_triangle|circle_packing_rect) TOP=3 ;; *) TOP=20 ;; esac
"$SKY/.venv/bin/python" "$SRC/scripts/select_by_val.py" "$RUN_DIR" "$ROOT/problems/$TASK" --top "$TOP" --name mt6230069
echo "=== done: $RUN_DIR/mt6230069_heuristic.py (selection.json has train/val scores)"
