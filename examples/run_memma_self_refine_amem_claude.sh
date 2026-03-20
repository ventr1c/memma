

#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."

echo $$

API_KEY='your-openai-api-key-here'
BASE_URL="https://api.openai.com/v1"

export OPENAI_API_KEY="$API_KEY"
export OPENAI_API_BASE="$BASE_URL"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# if [[ -z "${OPENAI_API_KEY:-}" ]]; then
#   OPENAI_API_KEY="$(sed -n "s/^API_KEY='\(.*\)'/\1/p" run_memma_self_refine_amem_gpt_0310.sh | head -n 1)"
#   export OPENAI_API_KEY
# fi

# if [[ -z "${OPENAI_API_BASE:-}" ]]; then
#   OPENAI_API_BASE="$(sed -n 's/^BASE_URL="\([^"]*\)"/\1/p' run_memma_self_refine_amem_gpt_0310.sh | head -n 1)"
#   export OPENAI_API_BASE
# fi

BEDROCK_MODEL="${BEDROCK_MODEL:-$(sed -n 's/^BEDROCK_MODEL="\([^"]*\)"/\1/p' run_memma_self_refine_single_claude_0311.sh | head -n 1)}"
BEDROCK_REGION="${BEDROCK_REGION:-$(sed -n 's/^BEDROCK_REGION="\([^"]*\)"/\1/p' run_memma_self_refine_single_claude_0311.sh | head -n 1)}"
ANSWER_MODEL="${ANSWER_MODEL:-gpt-4o-mini}"

export AWS_DEFAULT_REGION="$BEDROCK_REGION"
export AWS_REGION="$BEDROCK_REGION"

TIMESTAMP="$(date +%Y%m%d%H%M%S)"

AMEM_DIR="./results/amem/memma_amem_claude_${TIMESTAMP}/"
OUTPUT_DIR="./results/memma-amem_claude-haiku45_${TIMESTAMP}/"
SELF_REFINE_LOG_JSONL="${OUTPUT_DIR}/self_refine_log.jsonl"

python scripts/run_memma_self_refine_amem_claude.py \
  --dataset data/locomo10.json \
  --amem-dir "${AMEM_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --build_memories \
  --model "${ANSWER_MODEL}" \
  --answer_model "${ANSWER_MODEL}" \
  --backbone_model "${BEDROCK_MODEL}" \
  --region "${BEDROCK_REGION}" \
  --qr_max_turns 5 \
  --ratio 0.1 \
  --retrieve_k 30 \
  --retrieval_mode combined \
  --build_max_workers 1 \
  --enable_construction_meta_guidance \
  --rewrite_rerank_strategy llm_select \
  --self_refine_source parquet \
  --self_refine_parquet data/memory_rl_train_locomo_conv_26.parquet \
  --self_refine_log_jsonl "${SELF_REFINE_LOG_JSONL}" \
  --self_refine_add_fact_prompt_version v3 \
  --llm_backend anthropic \
  --embedder_provider openai \
  --retriever_model text-embedding-3-small \
  --evo_threshold 100

python scripts/run_memma_self_refine_amem_claude.py \
  --dataset data/locomo10.json \
  --amem-dir "${AMEM_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --model "${ANSWER_MODEL}" \
  --answer_model "${ANSWER_MODEL}" \
  --backbone_model "${BEDROCK_MODEL}" \
  --region "${BEDROCK_REGION}" \
  --qr_max_turns 5 \
  --ratio 0.1 \
  --retrieve_k 30 \
  --retrieval_mode combined \
  --qa_max_workers 8 \
  --rewrite_rerank_strategy no_cap \
  --answerability_prompt_version v4 \
  --llm_backend anthropic \
  --embedder_provider openai \
  --retriever_model text-embedding-3-small \
  --evo_threshold 100
