#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."

echo $$

API_KEY='your-openai-api-key-here'
BASE_URL="https://api.openai.com/v1"

export OPENAI_API_KEY="$API_KEY"
export OPENAI_API_BASE="$BASE_URL"

BEDROCK_MODEL="us.anthropic.claude-haiku-4-5-20251001-v1:0"
BEDROCK_REGION="us-west-2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIMESTAMP=$(date +%Y%m%d%H%M%S)
# TIMESTAMP="20260311073553" # reuse prior memory bank timestamp if needed

MEMORY_DIR="./results/memory_bank/memma_single_claude-haiku45_${TIMESTAMP}/"
OUTPUT_DIR="./results/memma-single_claude-haiku45_${TIMESTAMP}/"
SELF_REFINE_LOG_JSONL="${OUTPUT_DIR}/self_refine_log.jsonl"

# Phase 1: Build memories with Claude Haiku 4.5 via Bedrock
python scripts/run_memma_self_refine_single_claude.py \
  --dataset data/locomo10.json \
  --memory-dir "${MEMORY_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --enable_construction_meta_guidance \
  --build_memories \
  --model "${BEDROCK_MODEL}" \
  --region "${BEDROCK_REGION}" \
  --mm_max_turns 1 \
  --mm_retrieval_mode similarity \
  --mm_retrieve_k 10 \
  --retrieve_k 30 \
  --ratio 0.1 \
  --self_refine_source parquet \
  --self_refine_parquet data/memory_rl_train_locomo_conv_26.parquet \
  --self_refine_log_jsonl "${SELF_REFINE_LOG_JSONL}" \
  --self_refine_add_fact_prompt_version v3

# Phase 2: QA with GPT-4o-mini (answer+judge) + Claude Haiku (meta-thinker) via Bedrock
python scripts/run_memma_self_refine_single_claude.py \
  --dataset data/locomo10.json \
  --memory-dir "${MEMORY_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --model gpt-4o-mini \
  --meta_thinker_model "${BEDROCK_MODEL}" \
  --region "${BEDROCK_REGION}" \
  --retrieve_k 30 \
  --qr_max_turns 5 \
  --ratio 0.1 \
  --qa_max_workers 8
