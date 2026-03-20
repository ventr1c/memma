cd "$(dirname "$0")/.."
echo $$

# OpenAI key for QA-phase answer agent + judging
# API_KEY='your-openai-api-key-here'
API_KEY='your-openai-api-key-here'
BASE_URL="https://api.openai.com/v1"

export OPENAI_API_KEY="$API_KEY"
export OPENAI_API_BASE="$BASE_URL"

BEDROCK_MODEL="us.anthropic.claude-haiku-4-5-20251001-v1:0"
BEDROCK_REGION="us-west-2"

TIMESTAMP=$(date +%Y%m%d%H%M%S)

# Choose mode: 
# --build_memories: Build memories from scratch using LightMem
# (remove --build_memories to run evaluation mode)

# Phase 1: Build memories with Claude Haiku 4.5 via Bedrock
python scripts/run_memma_self_refine_lightmem_claude.py \
  --dataset data/locomo10.json \
  --qr_max_turns 5 \
  --ratio 0.1 \
  --model ${BEDROCK_MODEL} \
  --region ${BEDROCK_REGION} \
  --retrieve_k 30 \
  --retrieval_mode combined \
  --build_max_workers 1 \
  --qdrant-dir ./results/qdrant/memma_lightmem_qdrant_claude-haiku45_$TIMESTAMP/ \
  --output_dir ./results/memma-lightmem-qdrant_claude-haiku45_$TIMESTAMP/ \
  --build_memories \
  --disable_meta_thinker \
  --rewrite_rerank_strategy llm_select \
  --self_refine_source parquet \
  --self_refine_parquet data/memory_rl_train_locomo_conv_26.parquet \
  --self_refine_log_jsonl ./results/memma-lightmem-qdrant_claude-haiku45_$TIMESTAMP/self_refine_log.jsonl \
#   --self_refine_add_fact_prompt_version v3

# Phase 2: QA with GPT-4o-mini (answer+judge) + Claude Haiku (meta-thinker) via Bedrock
python scripts/run_memma_self_refine_lightmem_claude.py \
  --dataset data/locomo10.json \
  --qr_max_turns 5 \
  --ratio 0.1 \
  --model gpt-4o-mini \
  --meta_thinker_model ${BEDROCK_MODEL} \
  --region ${BEDROCK_REGION} \
  --retrieve_k 30 \
  --retrieval_mode combined \
  --qa_max_workers 8 \
  --rewrite_rerank_strategy no_cap \
  --answerability_prompt_version v4 \
  --qdrant-dir ./results/qdrant/memma_lightmem_qdrant_claude-haiku45_$TIMESTAMP/ \
  --output_dir ./results/memma-lightmem-qdrant_claude-haiku45_$TIMESTAMP/ \
