#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda is not available in PATH."
  exit 1
fi
API_KEY='your-openai-api-key-here'
# API_KEY='your-openai-api-key-here'
BASE_URL="https://api.openai.com/v1"

export OPENAI_API_KEY="$API_KEY"
export OPENAI_API_BASE="$BASE_URL"

ANTHROPIC_API_KEY='your-anthropic-api-key-here'
export ANTHROPIC_API_KEY

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[ERROR] OPENAI_API_KEY is required for embeddings and QA/judge models."
  exit 1
fi
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.openai.com/v1}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLKIT_DIR="$PROJECT_ROOT/LightMem/src/lightmem/memory_toolkits"
EXPERIMENT_ROOT="$PROJECT_ROOT/memma-0206"
RESULTS_ROOT="$EXPERIMENT_ROOT/results"

MEMORY_TYPE="FullContext"
DATASET_TYPE="LoCoMo"
DATASET_PATH="$PROJECT_ROOT/dataset/locomo/locomo10.json"
CONFIG_PATH="$EXPERIMENT_ROOT/configs/memory_toolkits/fullcontext_locomo_claude_haiku.json"

RATIO="${RATIO:-0.1}"
TOP_K="${TOP_K:--1}"
NUM_WORKERS="${NUM_WORKERS:-1}"
QA_MODEL="${QA_MODEL:-claude-haiku-4-5-20251001}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
START_IDX=0

if [[ ! -f "$DATASET_PATH" ]]; then
  echo "[ERROR] Dataset not found: $DATASET_PATH"
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] Config not found: $CONFIG_PATH"
  exit 1
fi

TOTAL=$(python - "$DATASET_PATH" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
if not isinstance(data, list):
    raise SystemExit("Dataset JSON must be a list.")
print(len(data))
PY
)

if [[ "$TOTAL" -le 0 ]]; then
  echo "[ERROR] Empty dataset: $DATASET_PATH"
  exit 1
fi

END_IDX=$(python - "$TOTAL" "$RATIO" <<'PY'
import sys

total = int(sys.argv[1])
ratio = float(sys.argv[2])
if ratio <= 0.0 or ratio > 1.0:
    raise SystemExit("RATIO must be in (0, 1].")
end_idx = max(1, int(total * ratio))
end_idx = min(total, end_idx)
print(end_idx)
PY
)

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="$RESULTS_ROOT/fullcontext_locomo_claude-haiku_${TIMESTAMP}"
TOKEN_COST_PREFIX="$RUN_DIR/token_cost_fullcontext"
mkdir -p "$RUN_DIR"

echo "[INFO] Running FullContext baseline (Claude)"
echo "[INFO] DATASET=$DATASET_PATH (total=$TOTAL, ratio=$RATIO, start=$START_IDX, end=$END_IDX)"
echo "[INFO] TOP_K=$TOP_K, NUM_WORKERS=$NUM_WORKERS"
echo "[INFO] RUN_DIR=$RUN_DIR"

(
  cd "$RUN_DIR"
  conda run -n fullcontext_env python "$TOOLKIT_DIR/memory_construction.py" \
    --memory-type "$MEMORY_TYPE" \
    --dataset-type "$DATASET_TYPE" \
    --dataset-path "$DATASET_PATH" \
    --config-path "$CONFIG_PATH" \
    --num-workers "$NUM_WORKERS" \
    --start-idx "$START_IDX" \
    --end-idx "$END_IDX" \
    --token-cost-save-filename "$TOKEN_COST_PREFIX" \
    --tokenizer-path "gpt-4o-mini" \
    2>&1 | tee "$RUN_DIR/01_memory_construction.log"
)

(
  cd "$RUN_DIR"
  conda run -n fullcontext_env python "$TOOLKIT_DIR/memory_search.py" \
    --memory-type "$MEMORY_TYPE" \
    --dataset-type "$DATASET_TYPE" \
    --dataset-path "$DATASET_PATH" \
    --config-path "$CONFIG_PATH" \
    --num-workers "$NUM_WORKERS" \
    --top-k "$TOP_K" \
    --start-idx "$START_IDX" \
    --end-idx "$END_IDX" \
    2>&1 | tee "$RUN_DIR/02_memory_search.log"
)

SEARCH_RESULTS_PATH=$(ls -t "$RUN_DIR"/"$MEMORY_TYPE"_*_"$DATASET_TYPE"_"$TOP_K"_"$START_IDX"_"$END_IDX".json 2>/dev/null | head -n1 || true)
if [[ -z "$SEARCH_RESULTS_PATH" ]]; then
  echo "[ERROR] Search result file not found in $RUN_DIR"
  exit 1
fi

echo "[INFO] SEARCH_RESULTS_PATH=$SEARCH_RESULTS_PATH"

(
  cd "$RUN_DIR"
  conda run -n fullcontext_env python "$TOOLKIT_DIR/memory_evaluation.py" \
    --search-results-path "$SEARCH_RESULTS_PATH" \
    --qa-model "$QA_MODEL" \
    --judge-model "$JUDGE_MODEL" \
    --dataset-type "$DATASET_TYPE" \
    2>&1 | tee "$RUN_DIR/03_memory_evaluation.log"
)

EVAL_RESULTS_PATH="${SEARCH_RESULTS_PATH%.json}_evaluation.json"
SUMMARY_RESULTS_PATH="${SEARCH_RESULTS_PATH%.json}_metrics_summary.json"

echo "[DONE] FullContext baseline (Claude) completed."
echo "[DONE] Token cost: ${TOKEN_COST_PREFIX}.json"
echo "[DONE] Search: $SEARCH_RESULTS_PATH"
echo "[DONE] Evaluation: $EVAL_RESULTS_PATH"
echo "[DONE] Metrics summary: $SUMMARY_RESULTS_PATH"
echo "[DONE] Logs: $RUN_DIR/01_memory_construction.log, $RUN_DIR/02_memory_search.log, $RUN_DIR/03_memory_evaluation.log"
