cd "$(dirname "$0")/.."
echo $$

# API_KEY='your-openai-api-key-here'
API_KEY='your-openai-api-key-here'
BASE_URL="https://api.openai.com/v1"

export OPENAI_API_KEY="$API_KEY"
export OPENAI_API_BASE="$BASE_URL"

TIMESTAMP=$(date +%Y%m%d%H%M%S)
# TIMESTAMP="20260310232711" # v1

# # Phase 1: Build memories with A-Mem
python scripts/run_memma_self_refine_amem.py \
  --dataset data/locomo10.json \
  --qr_max_turns 5 \
  --ratio 0.1 \
  --model gpt-4o-mini \
  --retrieve_k 30 \
  --retrieval_mode combined \
  --build_max_workers 1 \
  --amem-dir ./results/amem/memma_amem_$TIMESTAMP/ \
  --output_dir ./results/memma-amem_gpt4o-mini_$TIMESTAMP/ \
  --build_memories \
  --enable_construction_meta_guidance \
  --rewrite_rerank_strategy llm_select \
  --self_refine_source parquet \
  --self_refine_parquet data/memory_rl_train_locomo_conv_26.parquet \
  --self_refine_log_jsonl ./results/memma-amem_gpt4o-mini_$TIMESTAMP/self_refine_log.jsonl \
  --self_refine_add_fact_prompt_version v3 \
  --llm_backend openai \
  --embedder_provider openai \
  --retriever_model text-embedding-3-small \
  --evo_threshold 100

# Phase 2: QA evaluation
python scripts/run_memma_self_refine_amem.py \
  --dataset data/locomo10.json \
  --qr_max_turns 5 \
  --ratio 0.1 \
  --model gpt-4o-mini \
  --retrieve_k 30 \
  --retrieval_mode combined \
  --qa_max_workers 8 \
  --rewrite_rerank_strategy no_cap \
  --answerability_prompt_version v4 \
  --amem-dir ./results/amem/memma_amem_$TIMESTAMP/ \
  --output_dir ./results/memma-amem_gpt4o-mini_$TIMESTAMP/ \
  --llm_backend openai \
  --embedder_provider openai \
  --retriever_model text-embedding-3-small \
