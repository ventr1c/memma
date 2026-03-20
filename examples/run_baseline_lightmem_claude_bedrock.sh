#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda is not available in PATH."
  exit 1
fi

# Keep hardcoded keys as requested.
OPENAI_API_KEY='your-openai-api-key-here'
OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="$OPENAI_API_KEY"
export OPENAI_API_BASE="$OPENAI_BASE_URL"

ANTHROPIC_API_KEY='your-anthropic-api-key-here'
export ANTHROPIC_API_KEY

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "[ERROR] ANTHROPIC_API_KEY is required for Claude QA."
  exit 1
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[ERROR] OPENAI_API_KEY is required for build/judge."
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
LIGHTMEM_ROOT="${LIGHTMEM_ROOT:-$PROJECT_ROOT/LightMem}"
EXPERIMENT_ROOT="$PROJECT_ROOT/memma-0206"
export PYTHONPATH="$LIGHTMEM_ROOT/src:${PYTHONPATH:-}"

OFFICIAL_LOCOMO_DIR="$LIGHTMEM_ROOT/experiments/locomo"
BUILD_SCRIPT="$OFFICIAL_LOCOMO_DIR/add_locomo.py"
SEARCH_SCRIPT="$OFFICIAL_LOCOMO_DIR/search_locomo.py"

RESULTS_ROOT="$EXPERIMENT_ROOT/results"
CONDA_ENV="${CONDA_ENV:-lightmem}"

DATASET_PATH="${DATASET_PATH:-$PROJECT_ROOT/dataset/locomo/locomo10.json}"
RATIO="${RATIO:-0.1}"
START_IDX="${START_IDX:-0}"
BUILD_MODEL="${BUILD_MODEL:-us.anthropic.claude-haiku-4-5-20251001-v1:0}"
QA_MODEL="${QA_MODEL:-gpt-4o-mini}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
BUILD_PROVIDER="${BUILD_PROVIDER:-anthropic}"
BEDROCK_REGION="${BEDROCK_REGION:-us-west-2}"
export AWS_DEFAULT_REGION="$BEDROCK_REGION"
QA_PROVIDER="${QA_PROVIDER:-}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-combined}"
TOTAL_LIMIT="${TOTAL_LIMIT:-30}"
LIMIT_PER_SPEAKER="${LIMIT_PER_SPEAKER:-15}"
EMBEDDER="${EMBEDDER:-huggingface}"
MAX_WORKERS="${MAX_WORKERS:-1}"
EXECUTOR="${EXECUTOR:-process}"
LLMLINGUA_MODEL_PATH="${LLMLINGUA_MODEL_PATH:-microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank}"
EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-sentence-transformers/all-MiniLM-L6-v2}"
ANTHROPIC_API_BASE="${ANTHROPIC_API_BASE:-}"
JUDGE_API_KEY="${JUDGE_API_KEY:-$OPENAI_API_KEY}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-$OPENAI_BASE_URL}"

if [[ "$BUILD_PROVIDER" != "openai" && "$BUILD_PROVIDER" != "anthropic" ]]; then
  echo "[ERROR] BUILD_PROVIDER must be one of: openai, anthropic."
  exit 1
fi

if [[ -z "$QA_PROVIDER" ]]; then
  if [[ "$QA_MODEL" == claude* ]]; then
    QA_PROVIDER="anthropic"
  else
    QA_PROVIDER="openai"
  fi
fi

if [[ "$QA_PROVIDER" != "openai" && "$QA_PROVIDER" != "anthropic" ]]; then
  echo "[ERROR] QA_PROVIDER must be one of: openai, anthropic."
  exit 1
fi

if [[ ! -f "$DATASET_PATH" ]]; then
  echo "[ERROR] Dataset not found: $DATASET_PATH"
  exit 1
fi

if [[ ! -f "$BUILD_SCRIPT" ]]; then
  echo "[ERROR] Official build script not found: $BUILD_SCRIPT"
  exit 1
fi

if [[ ! -f "$SEARCH_SCRIPT" ]]; then
  echo "[ERROR] Official eval script not found: $SEARCH_SCRIPT"
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

END_IDX=$(python - "$TOTAL" "$RATIO" "$START_IDX" <<'PY'
import sys
total = int(sys.argv[1])
ratio = float(sys.argv[2])
start_idx = int(sys.argv[3])
if ratio <= 0.0 or ratio > 1.0:
    raise SystemExit("RATIO must be in (0, 1].")
if start_idx < 0 or start_idx >= total:
    raise SystemExit("START_IDX out of range.")
window = max(1, int(total * ratio))
end_idx = min(total, start_idx + window)
print(end_idx)
PY
)

if [[ "$END_IDX" -le "$START_IDX" ]]; then
  echo "[ERROR] Computed END_IDX ($END_IDX) must be greater than START_IDX ($START_IDX)."
  exit 1
fi

if [[ "$BUILD_PROVIDER" == "anthropic" ]]; then
  BUILD_API_KEYS_CSV="${BUILD_API_KEYS_CSV:-$ANTHROPIC_API_KEY}"
  BUILD_API_BASE_URL="${BUILD_API_BASE_URL:-$ANTHROPIC_API_BASE}"
else
  BUILD_API_KEYS_CSV="${BUILD_API_KEYS_CSV:-$OPENAI_API_KEY}"
  BUILD_API_BASE_URL="${BUILD_API_BASE_URL:-$OPENAI_BASE_URL}"
fi

if [[ "$QA_PROVIDER" == "anthropic" ]]; then
  QA_API_KEY="${QA_API_KEY:-$ANTHROPIC_API_KEY}"
  QA_BASE_URL="${QA_BASE_URL:-$ANTHROPIC_API_BASE}"
else
  QA_API_KEY="${QA_API_KEY:-$OPENAI_API_KEY}"
  QA_BASE_URL="${QA_BASE_URL:-$OPENAI_BASE_URL}"
fi

if [[ -z "${QA_API_KEY:-}" ]]; then
  echo "[ERROR] QA API key is empty for QA_PROVIDER=$QA_PROVIDER."
  exit 1
fi

IFS=',' read -r -a BUILD_API_KEYS_ARRAY <<< "$BUILD_API_KEYS_CSV"
if [[ "${#BUILD_API_KEYS_ARRAY[@]}" -eq 0 ]]; then
  echo "[ERROR] No build API keys available."
  exit 1
fi

MODEL_TAG=$(echo "$QA_MODEL" | tr -cs 'A-Za-z0-9._-' '-')
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="$RESULTS_ROOT/lightmem_official_claude_bedrock_${MODEL_TAG}_${TIMESTAMP}"
QDRANT_PRE_DIR="$RUN_DIR/qdrant_pre_update"
QDRANT_POST_DIR="$RUN_DIR/qdrant_post_update"
BUILD_LOGS_ROOT="$RUN_DIR/build_logs"
EVAL_DIR="$RUN_DIR/eval"
mkdir -p "$RUN_DIR" "$QDRANT_PRE_DIR" "$QDRANT_POST_DIR" "$BUILD_LOGS_ROOT" "$EVAL_DIR"

echo "[INFO] Running LightMem official LoCoMo baseline (Bedrock build)"
echo "[INFO] BEDROCK_REGION=$BEDROCK_REGION"
echo "[INFO] PYTHONPATH=$PYTHONPATH"
echo "[INFO] DATASET=$DATASET_PATH (total=$TOTAL, ratio=$RATIO, start=$START_IDX, end=$END_IDX)"
echo "[INFO] BUILD_PROVIDER=$BUILD_PROVIDER, BUILD_MODEL=$BUILD_MODEL"
echo "[INFO] QA_PROVIDER=$QA_PROVIDER, QA_MODEL=$QA_MODEL, JUDGE_MODEL=$JUDGE_MODEL"
echo "[INFO] RETRIEVAL_MODE=$RETRIEVAL_MODE, TOTAL_LIMIT=$TOTAL_LIMIT, LIMIT_PER_SPEAKER=$LIMIT_PER_SPEAKER"
echo "[INFO] RUN_DIR=$RUN_DIR"

BUILD_ARGS=(
  --dataset "$DATASET_PATH"
  --qdrant-pre-dir "$QDRANT_PRE_DIR"
  --qdrant-post-dir "$QDRANT_POST_DIR"
  --logs-root "$BUILD_LOGS_ROOT"
  --api-base-url "$BUILD_API_BASE_URL"
  --llm-provider "$BUILD_PROVIDER"
  --llm-model "$BUILD_MODEL"
  --aws-region "$BEDROCK_REGION"
  --llmlingua-model-path "$LLMLINGUA_MODEL_PATH"
  --embedding-model-path "$EMBEDDING_MODEL_PATH"
  --max-workers "$MAX_WORKERS"
  --executor "$EXECUTOR"
  --start-idx "$START_IDX"
  --end-idx "$END_IDX"
  --api-keys
)
VALID_BUILD_KEYS=()
for key in "${BUILD_API_KEYS_ARRAY[@]}"; do
  trimmed="$(echo "$key" | xargs)"
  if [[ -n "$trimmed" ]]; then
    BUILD_ARGS+=("$trimmed")
    VALID_BUILD_KEYS+=("$trimmed")
  fi
done
if [[ "${#VALID_BUILD_KEYS[@]}" -eq 0 ]]; then
  echo "[ERROR] No valid non-empty build API keys after parsing BUILD_API_KEYS_CSV."
  exit 1
fi

conda run -n "$CONDA_ENV" python "$BUILD_SCRIPT" \
  "${BUILD_ARGS[@]}" \
  2>&1 | tee "$RUN_DIR/01_build.log"

conda run -n "$CONDA_ENV" python "$SEARCH_SCRIPT" \
  --dataset "$DATASET_PATH" \
  --qdrant-dir "$QDRANT_POST_DIR" \
  --output-dir "$EVAL_DIR" \
  --retrieval-mode "$RETRIEVAL_MODE" \
  --total-limit "$TOTAL_LIMIT" \
  --limit-per-speaker "$LIMIT_PER_SPEAKER" \
  --embedder "$EMBEDDER" \
  --embedding-model-path "$EMBEDDING_MODEL_PATH" \
  --llm-provider "$QA_PROVIDER" \
  --llm-api-key "$QA_API_KEY" \
  --llm-base-url "$QA_BASE_URL" \
  --llm-model "$QA_MODEL" \
  --judge-api-key "$JUDGE_API_KEY" \
  --judge-base-url "$JUDGE_BASE_URL" \
  --judge-model "$JUDGE_MODEL" \
  --start-idx "$START_IDX" \
  --end-idx "$END_IDX" \
  2>&1 | tee "$RUN_DIR/02_eval.log"

SUMMARY_JSON="$EVAL_DIR/summary.json"
if [[ ! -f "$SUMMARY_JSON" ]]; then
  echo "[ERROR] summary.json not found: $SUMMARY_JSON"
  exit 1
fi

METRIC_LINE=$(python - "$SUMMARY_JSON" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
overall = data.get("aggregate_metrics", {}).get("overall", {})
j = overall.get("judge_correct", {}).get("mean")
f1 = overall.get("token_f1", {}).get("mean")
b1 = overall.get("bleu1", {}).get("mean")
if j is None or f1 is None or b1 is None:
    raise SystemExit("Missing J/F1/B1 metrics in summary.json.")
print(f"J={j:.4f} F1={f1:.4f} B1={b1:.4f}")
PY
)

echo "[DONE] LightMem official baseline (Bedrock build) completed."
echo "[DONE] Summary: $SUMMARY_JSON"
echo "[DONE] Metrics: $METRIC_LINE"
echo "[DONE] Logs: $RUN_DIR/01_build.log, $RUN_DIR/02_eval.log"
